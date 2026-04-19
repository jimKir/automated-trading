# Automated Trading System

[![Leakage Audit](https://github.com/jimKir/automated-trading/actions/workflows/leakage-audit.yml/badge.svg)](https://github.com/jimKir/automated-trading/actions/workflows/leakage-audit.yml)

A production-grade multi-factor momentum + mean-reversion strategy with regime-aware
signal blending, credit-enhanced choppy regime detection, and multi-layer risk management.

**Status:** Paper trading  
**Last validated:** April 13, 2026  
**Walk-forward 12M OOS (Apr 2025–Apr 2026):** Sharpe 3.01, CAGR +58.1%, MaxDD –7.0%  
**Leakage audit:** 46/46 tests passed — zero future knowledge leakage confirmed

---

## Strategy Overview

The strategy selects the top quartile of a 20-stock point-in-time universe each week,
using a composite score of five regime-conditional factors. Weights were fitted on
2018–2022 in-sample data and have been frozen since — no refitting during OOS.

| Component | Detail |
|---|---|
| Universe | 20 stocks per year, point-in-time (PIT) |
| Signals | TS momentum, mean reversion, MACD, RSI, PMO (bear only) |
| Regime | Bull (VIX < 20 AND SPY > 200MA) / Bear |
| Selection | Top quartile by composite score |
| Weighting | Equal weight |
| Rebalance | Weekly (last trading day of each week) |
| Cost model | 0.126% round-trip × 52 weeks × 30% turnover = 1.97%/yr drag |

---

## Validated Performance

### Walk-Forward 12-Month OOS (Apr 14, 2025 → Apr 11, 2026)

Locked IS parameters (2018–2022). No refitting. 4 non-overlapping quarterly folds.

| Metric | Strategy | SPY B&H |
|---|---|---|
| Sharpe | 3.005 | 2.109 |
| Sortino | 4.959 | 3.020 |
| CAGR | +58.07% | +27.98% |
| Max Drawdown | –6.99% | –8.88% |
| Calmar | 8.306 | 3.149 |
| Volatility | 19.32% | 13.27% |
| Win Rate | 54.6% | 56.5% |

**Alpha analysis:** Beta 1.148, annualized alpha +18.55%, information ratio 1.843, correlation 0.787.

**Walk-forward fold consistency:**

| Fold | Strategy Sharpe | SPY Sharpe | Strategy CAGR | SPY CAGR |
|---|---|---|---|---|
| Apr–Jun 2025 | 7.50 | 5.86 | +159.2% | +96.4% |
| Jul–Sep 2025 | 3.12 | 4.12 | +48.8% | +36.0% |
| Oct–Dec 2025 | 1.83 | 0.87 | +39.5% | +10.9% |
| Jan–Apr 2026 | 1.49 | –0.02 | +28.0% | –0.3% |

4/4 folds positive CAGR. 4/4 folds beat SPY. 3/4 folds higher Sharpe than SPY.

### Historical Validation Summary

| Period | Type | Sharpe | Notes |
|---|---|---|---|
| 2018–2022 | IS (parameter fitting) | 0.78 | Regime weight optimization |
| 2023–Q1 2026 | OOS extended | 1.18 | ChoppyDetector v2 |
| Sep 2025–Apr 2026 | OOS 7-month | 0.90 | ChoppyDetector v2 + PositionAnomalyScorer |
| Apr 2025–Apr 2026 | WF 12M OOS | 3.01 | Production params, zero refitting, 46/46 leakage tests |
| Dec 2025–Apr 2026 | Paper emulation | 0.29 | MaxDD –2.67% vs SPY –8.88% |

---

## Architecture

```
Market Data (yfinance / Alpaca / CCXT / Databento)
        │
DynamicUniverseScanner  ←→  PIT Universe (20 stocks/year)
        │
SignalGenerator (5 core factors: ts_mom, mr, macd, rsi, pmo)
        │
    Regime Dispatch (bull vs bear)
        │
ChoppyRegimeDetector v2 (7 groups: vol, price_vol, macro_credit,
                          event_shock, commodity_fx, breadth, sentiment)
        │
EarlyWarningSystem (6 layers: anomaly, macro, event_shock,
                     commodity_fx, intraday_regime, choppy_regime)
        │
PositionAnomalyScorer (per-instrument asymmetric scaling)
        │
RiskManager (progressive drawdown scaling, Kelly sizing, VaR/CVaR)
        │
LiveEngine → Broker Adapters (Alpaca / Binance / IBKR / Paper)
```

---

## Locked Strategy Parameters

From `data/regime_params_validated.json` — fitted 2018–2022, frozen since.

### Bull Regime (VIX < 20 AND SPY > 200d MA)

| Factor | Weight |
|---|---|
| TS Momentum (12M risk-adjusted) | 0.50 |
| Mean Reversion (20d z-score) | 0.15 |
| MACD (12/26/9) | 0.30 |
| RSI (14-period, contrarian) | 0.05 |

### Bear Regime

| Factor | Weight |
|---|---|
| TS Momentum | 0.30 |
| Mean Reversion | 0.30 |
| MACD | 0.25 |
| RSI | 0.10 |
| PMO (35/20, contrarian) | 0.05 |

---

## Project Structure

```
automated-trading/
├── main.py                       # CLI entry point (backtest, paper, live, signals)
├── config/
│   ├── settings.yaml             # Master configuration
│   ├── credentials.py            # Env-var credential loading
│   └── config.{dev,prod}.yaml    # Environment overrides
├── strategy/
│   ├── signals.py                # Multi-factor signal engine
│   ├── universe.py               # DynamicUniverseSelector
│   ├── alpaca_microstructure.py  # VWAP, gap fill, trade intensity
│   └── databento_*.py            # NASDAQ imbalance, opening cross, options flow
├── regime/
│   ├── choppy_regime.py          # ChoppyDetector v2 (7 feature groups)
│   ├── ews.py                    # 6-layer Early Warning System
│   ├── anomaly.py                # Isolation Forest anomaly detection
│   ├── macro_score.py            # FRED macro stress indicators
│   ├── event_shock.py            # VIX velocity, breadth collapse
│   ├── commodity_fx.py           # Oil/gold/copper/DXY stress
│   └── intraday_regime.py        # Intraday SPY regime (ADX + EMA + VIX)
├── risk/
│   ├── manager.py                # Progressive DD scaling, Kelly, VaR/CVaR
│   └── position_anomaly.py       # Per-instrument asymmetric scaling
├── execution/
│   ├── live_engine.py            # Main trading loop orchestrator
│   ├── paper_broker.py           # Local paper trading simulator
│   ├── alpaca_broker.py          # Alpaca broker adapter
│   ├── binance_broker.py         # Binance adapter (spot + futures)
│   └── ibkr_broker.py            # Interactive Brokers adapter
├── core/
│   ├── portfolio.py              # Position management, equity curve
│   ├── cost_model.py             # 6-layer realistic cost model
│   ├── optimizer.py              # Risk Parity / Min Variance optimization
│   ├── vol_targeting.py          # EWMA vol targeting (15% target)
│   └── kelly_sizer.py            # Fractional Kelly sizing
├── backtest/
│   ├── engine.py                 # Event-driven backtest engine
│   ├── wf_validator.py           # Walk-forward overfitting validator
│   └── reporter.py               # HTML + JSON + matplotlib reports
├── data/
│   ├── regime_params_validated.json  # LOCKED IS parameters
│   ├── pit_universe.json             # Point-in-time stock universe (2018–2026)
│   ├── feed.py                       # Multi-source data feed
│   └── historical/daily/            # Parquet price data (Git LFS)
├── tests/
│   ├── test_leakage_audit.py         # 46 future-knowledge leakage tests
│   ├── production_readiness_test.py  # Production readiness checks
│   └── test_*.py                     # Unit tests for each module
├── results/
│   ├── wf_12m_oos_results.json       # Walk-forward 12M OOS results
│   ├── wf_12m_oos_chart.png          # Performance visualization
│   └── wf_12m_*.csv                  # Daily return series
├── run_wf_12m_oos.py             # Walk-forward 12M OOS backtest script
├── run_oos_extended.py           # Extended OOS backtest (2023–2026)
├── paper_trading_emulator.py     # OOS paper trading emulator
├── docs/
│   └── options_hedge_analysis.md # Options hedge analysis (REJECTED)
├── deploy/
│   └── aws_setup.md              # ECS Fargate deployment guide
├── Dockerfile                    # Production container
├── docker-compose.yml            # Local Docker setup
└── infra/terraform/              # AWS Terraform IaC
```

---

## Quick Start

### Prerequisites

- Python 3.11+ (3.12 recommended)
- Alpaca account (free paper trading at [alpaca.markets](https://alpaca.markets))

### Install

```bash
git clone https://github.com/jimKir/automated-trading.git
cd automated-trading
pip install -r requirements.lock
```

### Set credentials

```bash
cp .env.example .env
# Edit .env with your Alpaca API key and secret
```

### Run paper trading

```bash
python main.py paper
```

### Run the walk-forward OOS backtest

```bash
python run_wf_12m_oos.py
```

### Run leakage audit tests

```bash
pip install pytest
python -m pytest tests/test_leakage_audit.py -v
```

### Today's signals

```bash
python main.py signals
```

---

## Risk Management

Five independent, multiplicative protection layers:

| Layer | Frequency | Action |
|---|---|---|
| Circuit breakers | Per cycle | Halt on >8% realized loss or progressive DD scaling |
| EWS (Early Warning) | Daily | 6 stress detectors → GREEN through CRITICAL scale |
| Vol targeting | Daily | EWMA scaler targets 15% annualized vol (max 1.5× leverage) |
| Intraday shock | Every 5 min | VIX spike >15% or equity drop >3% → scale to 25% |
| PositionAnomalyScorer | Per rebalance | Per-instrument asymmetric cuts |

**Progressive drawdown scaling:** DD <8% = 1.0×, 8–15% = linear to 0.5×, 15–25% = linear to 0.2×, >25% = 0.2× floor.

---

## ChoppyRegimeDetector v2

| Level | Score | Position Scale | Description |
|---|---|---|---|
| GREEN | < 0.17 | 100% | Trending — full exposure |
| YELLOW | 0.17–0.27 | 80% | Choppy building — light trim |
| ORANGE | 0.27–0.40 | 50% | Clearly choppy — reduce |
| RED | > 0.40 | 25% | High choppiness — defensive |

**Feature groups (v2):** vol_spike (18%), price_vol (18%), macro_credit (16%), event_shock (16%), commodity_fx (12%), breadth (12%), sentiment (8%).

Credit stress (HYG/LQD) provides 3–10 day lead time on macro events.

---

## Leakage Audit

The 12-month OOS backtest has been systematically audited for future knowledge leakage.
46 automated tests cover 11 vectors:

| Category | Tests | Status |
|---|---|---|
| Signal lookahead (all 5 factors) | 6 | Passed |
| Regime indicator lookahead | 3 | Passed |
| Parameter snooping | 5 | Passed |
| Survivorship bias (PIT universe) | 5 | Passed |
| Walk-forward fold integrity | 5 | Passed |
| Data alignment | 3 | Passed |
| Cost model integrity | 3 | Passed |
| Reproducibility | 2 | Passed |
| Results JSON consistency | 5 | Passed |
| Script structural audit | 5 | Passed |
| Composite score peeking | 3 | Passed |

Run: `python -m pytest tests/test_leakage_audit.py -v`

---

## Key Design Decisions

| Decision | Outcome |
|---|---|
| Options hedge overlay | REJECTED — 0/5 periods improve MaxDD (see `docs/options_hedge_analysis.md`) |
| ADX gate | Disabled — hurts Sharpe in all OOS windows |
| Hourly entry timing | Wired but minimal OOS edge (+17 bps) |
| USO/XLE/LQD/HYG as holdings | Rejected — hurt OOS Sharpe |
| HYG as ChoppyDetector input | Adopted — credit stress in macro_credit group |
| Adaptive rebalance | Biweekly when GREEN, weekly when YELLOW+ |
| VIX-spike forced rebalance | +20% VIX in 1 day → immediate rebalance |
| Risk Parity optimization | Default over min-variance (more stable) |

---

## Go/No-Go Thresholds

| Milestone | Status | Criteria |
|---|---|---|
| OOS validation | PASSED | Walk-forward 12M Sharpe 3.01, 4/4 folds positive |
| Leakage audit | PASSED | 46/46 tests, zero lookahead |
| Paper trading | STARTING | — |
| Live capital | WAIT | All 6 paper trading criteria met (see below) |

### Paper Trading → Live Capital Criteria

Tracked during paper trading. All 6 must pass before deploying real capital.
Full escalation rules and daily checklists in the [paper trading runbook](docs/paper_trading_runbook.md).

| # | Metric | Threshold | Current | Status |
|---|---|---|---|---|
| 1 | Annualised Sharpe | > 0.50 | — | Pending |
| 2 | Max Drawdown | < 15% | — | Pending |
| 3 | Drawdown Recovery | >= 1 episode | — | Pending |
| 4 | Win Rate | > 50% | — | Pending |
| 5 | Correlation to Backtest | > 0.60 | — | Pending |
| 6 | System Uptime | > 95% | — | Pending |

---

## Deployment

```bash
# Docker
docker-compose up

# AWS ECS Fargate (~$11/month)
./deploy/deploy_aws.sh
```

See [DEPLOY_AWS.md](DEPLOY_AWS.md) and [deploy/aws_setup.md](deploy/aws_setup.md) for details.

---

## Configuration

Master config: `config/settings.yaml`

| Section | Key Settings |
|---|---|
| `capital` | `initial_equity: 25000`, `max_portfolio_heat: 0.75` |
| `risk` | `max_position_pct: 0.15`, `daily_loss_limit: 0.08`, `max_drawdown_halt: 0.15` |
| `strategy` | `lookback_fast: 20`, `lookback_slow: 60`, `zscore_entry: 2.0` |
| `dynamic_universe` | `top_n: 20`, `adaptive_caps: true`, `equity_cap: 60–90%` |

---

## Security

Credentials loaded exclusively from environment variables via `config/credentials.py`.
Never commit `.env` — it is gitignored. See `SECURITY_NOTICE.md` for details.
