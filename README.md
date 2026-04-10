# Automated Trading System

A production-grade multi-asset momentum + mean-reversion strategy with regime-aware
position sizing, credit-enhanced choppy regime detection, and dynamic universe expansion.

**Status:** Ready for paper trading
**Last validated:** April 2026
**OOS Sharpe (Sep 2025-Apr 2026):** +0.898 (ChoppyDetector v2 + PositionAnomalyScorer)
**Paper emulation (Dec 2025-Apr 2026):** Sharpe +0.29, MaxDD -2.67% vs SPY -8.88%

---

## Architecture

```
Market Data (yfinance / Alpaca / CCXT / Databento)
        |
DynamicUniverseScanner  <->  Core Universe (18 ETFs + 3 crypto + 4 futures)
        |                     Point-in-Time stock universe (20 per year)
        |
DynamicUniverseSelector (55 candidates -> top 20 monthly)
        |
SignalGenerator (16 factors: ts_mom, mr, macd, rsi, vol_regime,
                 xsec_mom, credit_regime, obv, vol_trend, cmf,
                 h2o_trend, pv_segments, pmo, vwap_proxy, vwap_sma, stoch)
        |
    Regime Dispatch (bull vs bear/neutral vs gated-T3 choppy)
        |
ChoppyRegimeDetector v2 (7 groups: vol, price_vol, macro_credit,
                          event_shock, commodity_fx, breadth, sentiment)
        |
EarlyWarningSystem (6 layers: anomaly, macro, event_shock,
                     commodity_fx, intraday_regime, choppy_regime)
        |
PositionAnomalyScorer (per-instrument asymmetric scaling)
        |
HourlyEntryTimer (12:00 ET rule / crypto session windows)
        |
RiskManager (progressive drawdown scaling, Kelly sizing, VaR/CVaR)
        |
LiveEngine -> Broker Adapters (Alpaca / Binance / IBKR / Paper)
```

---

## Universe

### Core Trading Universe (settings.yaml)

| Instrument | Type | Role |
|---|---|---|
| SPY | Equity ETF | US large cap anchor |
| QQQ | Equity ETF | Tech/growth exposure |
| IWM | Equity ETF | Small cap diversifier |
| GLD | Commodity ETF | Gold / inflation hedge |
| TLT | Bond ETF | Long-duration rates |
| SHY | Bond ETF | Short-duration / cash proxy |
| XLU | Sector ETF | Utilities / defensive |
| XLP | Sector ETF | Consumer staples / defensive |
| VGK | Equity ETF | Europe exposure |
| EEM | Equity ETF | Emerging markets |
| XLK | Sector ETF | Technology |
| XLE | Sector ETF | Energy |
| XLF | Sector ETF | Financials |
| VNQ | REIT ETF | Real estate |
| AGG | Bond ETF | Aggregate bonds |
| EWJ | Equity ETF | Japan equities |
| EMXC | Equity ETF | EM ex-China |
| XLV | Sector ETF | Healthcare / defensive |
| BTC-USD | Crypto | High-volatility alpha |
| ETH-USD | Crypto | High-volatility alpha |
| SOL-USD | Crypto | High-volatility alpha |
| ES=F, NQ=F, GC=F, CL=F | Futures | Index / commodity futures |

**Dynamic universe:** 55 candidates ranked monthly by vol-adjusted 6-month momentum, top 20 selected with adaptive equity cap (60% bear - 90% bull).

**Dynamic expansion:** Up to 3 additional single stocks per day via Alpaca Screener API (capped at 8% weight each), gated by choppy regime.

---

## Quick Start

### Prerequisites
- Python 3.11+ (3.12+ may break some dependencies)
- Java 17 (for H2O AutoML vol forecasting)
- Alpaca account (paper trading free at alpaca.markets)

### Install

```bash
git clone https://github.com/jimKir/automated-trading.git
cd automated-trading
pip install -r requirements.txt
```

### Set credentials

```bash
cp .env.example .env
# Edit .env with your Alpaca API key and secret
```

### Run paper trading

```bash
python main.py paper
# or with Docker:
docker-compose up
```

### Run backtest

```bash
python main.py backtest
```

### Today's signals

```bash
python main.py signals
```

### What-if analysis

```bash
python whatif.py --suite all        # all 7 scenario suites
python whatif.py --suite capital    # capital range sweep
python whatif.py --suite risk       # conservative -> aggressive
```

---

## Modules

| Module | Description |
|---|---|
| `strategy/signals.py` | 16-factor signal engine with regime-conditional blending (bull/bear/T3 choppy) |
| `strategy/universe.py` | DynamicUniverseSelector (55->20), AdaptiveCaps, DynamicCandidateBuilder |
| `strategy/alpaca_microstructure.py` | VWAP distance, gap fill rate, trade intensity, options flow from Alpaca |
| `strategy/databento_imbalance.py` | NASDAQ closing auction order imbalance signal (Databento XNAS.ITCH) |
| `strategy/databento_opening_cross.py` | NASDAQ opening cross volume anomaly signal |
| `strategy/databento_options_flow.py` | OPRA options order flow imbalance signal |
| `regime/choppy_regime.py` | ChoppyRegimeDetector v2 -- 7 feature groups, IS-calibrated thresholds |
| `regime/order_flow_anomaly.py` | Volume spike + close-position anomaly detection (Group 9 for v4) |
| `regime/ews.py` | 6-layer Early Warning System orchestrator |
| `regime/anomaly.py` | Layer A: Isolation Forest position anomaly (unsupervised) |
| `regime/macro_score.py` | Layer B: FRED macro stress (yield curve, credit spreads, VIX, DXY) |
| `regime/event_shock.py` | Layer C: VIX velocity, term structure, breadth collapse |
| `regime/commodity_fx.py` | Layer D: Oil/gold/copper/DXY/JPY/EUR stress signals |
| `regime/intraday_regime.py` | Layer E: Intraday SPY regime (ADX + EMA + VIX) |
| `regime/calibrate_choppy.py` | Calibration utility for ChoppyDetector thresholds |
| `risk/position_anomaly.py` | PositionAnomalyScorer -- per-instrument asymmetric scaling |
| `risk/manager.py` | RiskManager -- progressive drawdown scaling, Kelly, VaR/CVaR, circuit breakers |
| `execution/live_engine.py` | LiveEngine -- main orchestrator wiring all modules into trading loop |
| `execution/hourly_entry_timer.py` | Intraday entry timing (12:00 ET equity / crypto session windows) |
| `execution/alpaca_broker.py` | Alpaca broker adapter (alpaca-py SDK) |
| `execution/binance_broker.py` | Binance broker adapter (spot + futures) |
| `execution/ibkr_broker.py` | Interactive Brokers adapter (TWS/Gateway) |
| `execution/paper_broker.py` | Local paper trading simulator |
| `execution/broker_base.py` | Abstract broker interface (Order, AccountInfo dataclasses) |
| `backtest/engine.py` | Event-driven backtest engine with full module integration |
| `backtest/engine_rebalance_patch.py` | Adaptive rebalance scheduler (biweekly/weekly/VIX-spike) |
| `backtest/wf_validator.py` | 3-method overfitting validator (walk-forward, sensitivity, permutation) |
| `backtest/reporter.py` | HTML + JSON + matplotlib performance reports |
| `data/dynamic_universe_scanner.py` | Alpaca Screener API, 8 hard filters, choppy gate |
| `data/feed.py` | Multi-source data feed (yfinance, CCXT, IBKR) |
| `core/portfolio.py` | Position management, target weights, trade log, equity curve |
| `core/cost_model.py` | 6-layer realistic cost model (commission, spread, impact, financing) |
| `core/optimizer.py` | Risk Parity (default) and Minimum Variance portfolio optimization |
| `core/vol_targeting.py` | EWMA volatility targeting (target 15% ann. vol, max 1.5x leverage) |
| `core/crisis_alpha_amplifier.py` | VIX-regime position scaling (crisis=1.6x, suppressed=0.8x) |
| `core/kelly_sizer.py` | Fractional Kelly sizing based on rolling IC |
| `core/h2o_trend_classifier.py` | H2O AutoML trend classifier (5 states, monthly retrain) |
| `core/h2o_vol_forecaster.py` | H2O AutoML vol forecaster (beats EWMA 6/6 metrics OOS) |
| `core/intraday_shock.py` | VIX spike + equity drop + volume shock detector |
| `core/price_volume_segments.py` | Wyckoff-inspired price-volume momentum quality scorer |
| `config/credentials.py` | Centralized env-var credential loading |
| `config/settings.yaml` | Master configuration file |
| `utils/config_loader.py` | YAML loader with ${ENV_VAR} interpolation |
| `utils/indicators.py` | Shared technical indicators (ADX) |
| `utils/logger.py` | Centralized logging with file + console output |
| `main.py` | CLI entry point (backtest, paper, live, signals, report) |
| `healthcheck.py` | HTTP health server on :8080 (/health, /status, /signals) |
| `daily_report.py` | Daily P&L report with optional SES email + S3 upload |
| `paper_trading_emulator.py` | OOS paper trading emulator (Dec 2025 - Apr 2026) |
| `whatif.py` | 7 scenario analysis suites |

---

## ChoppyRegimeDetector v2 -- Regime Levels

| Level | Score | Position Scale | Description |
|---|---|---|---|
| GREEN | < 0.17 | 100% | Trending/normal -- no action |
| YELLOW | 0.17-0.27 | 80% | Choppy building -- light trim |
| ORANGE | 0.27-0.40 | 50% | Clearly choppy -- reduce exposure |
| RED | > 0.40 | 25% | 2025/bear choppiness -- defensive |

**7 Feature Groups (v2):** vol_spike (18%), price_vol (18%), macro_credit (16%), event_shock (16%), commodity_fx (12%), breadth (12%), sentiment (8%).

Credit stress (HYG/LQD) provides 3-10 day lead time on macro events.
Order flow anomaly detector exists as standalone module for planned v4 expansion (adds credit + order_flow groups).

### EWS (Early Warning System) Levels

| Level | Score | Position Scale | Description |
|---|---|---|---|
| GREEN | < 0.25 | 100% | Full exposure |
| YELLOW | 0.25-0.40 | 70% | Trimming |
| ORANGE | 0.40-0.55 | 40% | Reducing |
| RED | 0.55-0.70 | 20% | Defensive |
| CRITICAL | > 0.70 | 5% | Near-flat |

**6 Layers:** Anomaly (30%), Macro (25%), Choppy (15%), Event Shock (15%), Commodity/FX (10%), Intraday (5%).

---

## Protection Layers (5 independent, multiplicative)

| Layer | Frequency | What it does |
|---|---|---|
| **Circuit breakers** | Per cycle | Halt on >8% realised cash loss or progressive DD scaling |
| **EWS** (Early Warning) | Daily | 6 stress detectors -> GREEN through CRITICAL scale |
| **Vol targeting** | Daily | EWMA/H2O scaler targets 15% ann. vol (max 1.5x leverage) |
| **Intraday shock** | Every 5 min | VIX spike >15% or equity drop >3% -> scale to 25% |
| **PositionAnomalyScorer** | Per rebalance | Per-instrument asymmetric cuts (crypto floor 10%, equity floor 40%, hedges never cut) |

Progressive drawdown scaling (RiskManager): DD <8% = 1.0x, 8-15% = linear to 0.5x, 15-25% = linear to 0.2x, >25% = 0.2x floor.

---

## Signal Engine: Regime-Conditional Factor Weights

### Bull Regime (VIX < 20 AND SPY > 200d MA)

| Factor | Weight |
|---|---|
| TS Momentum | 0.60 |
| Mean Reversion | 0.10 |
| MACD | 0.25 |
| RSI | 0.05 |
| PMO | 0.00 (disabled) |
| Stochastic | 0.00 (disabled) |

### Bear/Neutral Regime

| Factor | Weight |
|---|---|
| TS Momentum | 0.45 |
| Mean Reversion | 0.15 |
| MACD | 0.18 |
| RSI | 0.07 |
| PMO | 0.12 |
| Stochastic | 0.10 |

### Gated T3 (Choppy Override)

Triggered when choppy_score >= 0.17 AND SPY within 15% of 200d MA.
Shifts to: TS Momentum 0.12, Mean Reversion 0.42 (mean-reversion dominant).
OOS validation: 7/8 folds improved (2000-2022), GFC period restored.

---

## Validated Performance

| Period | Sharpe | MaxDD | Notes |
|---|---|---|---|
| IS 2018-2022 | +0.78 | -- | Parameter fitting period |
| OOS 2023-Q1 2026 | +1.18 | -- | Clean OOS with ChoppyDetector |
| 7-month OOS Sep 2025-Apr 2026 | +0.898 | -4.7% | ChoppyDetector v2 |
| Paper emulation Dec 2025-Apr 2026 | +0.29 | -2.67% | vs SPY -8.88% MaxDD |
| Dynamic 20/55 adaptive (2018-2025) | 0.113 | -22.2% | 8.4% ann. return, Calmar 0.379 |

---

## Key Design Decisions

- **VWAP Factor 15 (vwap_sma):** IC validated (+0.042 bull, -0.150 bear) but gated by regime
- **ADX gate:** disabled (hurts Sharpe in all OOS windows)
- **Hourly entry timing:** wired but minimal OOS edge (+17 bps fill improvement)
- **USO/XLE/LQD/HYG as portfolio holdings:** rejected (hurt OOS Sharpe)
- **HYG as ChoppyDetector input:** adopted (credit stress in Group C macro_credit, weight 0.16)
- **Adaptive rebalance:** biweekly when GREEN, weekly when YELLOW+ (wins 8/8 OOS folds)
- **VIX-spike forced rebalance:** +20% VIX in one day triggers immediate rebalance (crisis alpha Sharpe 2.03)
- **H2O vol forecaster:** Beats EWMA on 6/6 metrics OOS (MAE -6.3%, Bias -90.9%)
- **Risk Parity optimization:** Default over min-variance (more stable, less sensitive to covariance estimation)

---

## Go/No-Go Thresholds

- **Paper trading:** START -- OOS Sharpe +0.898 > threshold +0.50
- **Live capital:** WAIT -- need 12 months paper trading, Sharpe > 0.50 sustained, drawdown episode survived

---

## Security

Credentials loaded exclusively from environment variables via `config/credentials.py`.
Never commit `.env` -- it is gitignored.
See `SECURITY_NOTICE.md` for credential rotation history.

---

## Deployment

See [DEPLOY_AWS.md](DEPLOY_AWS.md) for Amazon Web Services deployment guide.
See [deploy/aws_setup.md](deploy/aws_setup.md) for detailed ECS Fargate setup (~11/month).

```bash
# Docker
docker-compose up

# AWS ECS Fargate
./deploy/deploy_aws.sh
```

---

## Configuration (`config/settings.yaml`)

| Section | Key Settings |
|---|---|
| `capital` | `initial_equity: 25000`, `max_portfolio_heat: 0.75` |
| `risk` | `max_position_pct: 0.15`, `daily_loss_limit: 0.08`, `max_drawdown_halt: 0.15`, `kelly_fraction: 0.25` |
| `strategy` | `lookback_fast: 20`, `lookback_slow: 60`, `zscore_entry: 2.0` |
| `dynamic_universe` | `enabled: true`, `top_n: 20`, `adaptive_caps: true`, `equity_cap: 60-90%` |
| `dynamic_candidates` | S&P 500 + NDX 100 screening, `min_avg_volume_usd: 5M` |

---

## Tags

| Tag | Description |
|---|---|
| `v1.0.0-paper-baseline` | IS-validated regime weights |
| `choppy-v3` | HYG/LQD credit features added |
| `choppy-v4` | Order flow anomaly layer added |
| `dynamic-universe-v1` | Alpaca Screener integration |
| `prod-ready-v1` | Dress rehearsal passed, paper trading ready |
