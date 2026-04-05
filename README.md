# Automated Trading System — v9 (Complete)

Multi-factor, multi-asset algorithmic trading system with full backtest,
paper trading, and live trading capability. €10,000–€50,000 capital range.

---

## What's in this system

```
trading_system/
├── main.py                      # CLI entry point
├── config/settings.yaml         # ALL parameters — single source of truth
│
├── strategy/
│   ├── signals.py               # 7-factor signal engine
│   └── universe.py              # Dynamic universe selector (55 candidates → top 20)
│                                #   with adaptive equity cap (60–90%)
│
├── core/
│   ├── portfolio.py             # Position management, order execution, P&L
│   ├── cost_model.py            # 6-layer realistic cost model
│   ├── vol_targeting.py         # EWMA vol scaler (target 22% ann. vol)
│   └── intraday_shock.py        # VIX spike + equity drop detector (every 5 min live)
│
├── risk/
│   └── manager.py               # VaR/CVaR, circuit breakers (cash-only daily halt)
│
├── regime/                      # Early Warning System (EWS)
│   ├── ews.py                   # Orchestrator → GREEN/YELLOW/ORANGE/RED → scale
│   ├── anomaly.py               # Isolation Forest on position behaviour
│   ├── macro_score.py           # FRED: yield curve, credit spreads, VIX
│   ├── event_shock.py           # VIX velocity, term structure, breadth
│   └── commodity_fx.py          # Oil, Gold/SPY, DXY, USD/JPY, EUR/USD
│
├── backtest/
│   ├── engine.py                # Event-driven backtester (all layers wired in)
│   ├── reporter.py              # HTML + chart reports
│   └── wf_validator.py          # 3-method overfitting validation framework
│
├── data/
│   └── feed.py                  # yfinance + CCXT data feed
│
├── execution/
│   ├── paper_broker.py          # Local simulation
│   ├── alpaca_broker.py         # Alpaca (equities)
│   ├── binance_broker.py        # Binance (crypto, testnet + live)
│   ├── ibkr_broker.py           # Interactive Brokers (all asset classes)
│   └── live_engine.py           # Live/paper loop with all protection layers
│
├── daily_report.py              # Daily P&L → HTML → SES email → S3
├── healthcheck.py               # HTTP :8080 /health /status /signals
├── whatif.py                    # 7 scenario suites
├── Dockerfile + docker-compose.yml
└── deploy/aws_setup.md          # ECS Fargate guide (~€11/month)
```

---

## Strategy

**Multi-Factor Momentum + Mean-Reversion + Credit Regime** across a dynamic
universe of up to 20 instruments selected monthly from 55 candidates.

### Signal Factors (7)

| # | Factor | Weight | Type |
|---|---|---|---|
| 1 | Time-series momentum (fast/slow SMA) | 40% | Reactive |
| 2 | Mean reversion (z-score) | 30% | Reactive |
| 3 | MACD histogram | 20% | Reactive |
| 4 | RSI filter | 10% | Reactive |
| 5 | Volatility regime multiplier | × | Multiplier |
| 6 | Cross-sectional momentum (equity overlay) | 30% blend | Cross-asset |
| 7 | Credit regime (HYG/LQD + VIX + yield curve) | 30% blend | Predictive |

### Dynamic Universe (55 candidates → top 20)

Each month, all 55 instruments ranked by vol-adjusted 6-month momentum.
Top 20 selected with **adaptive equity cap**:

| Market Regime | Equity Cap | Instruments |
|---|---|---|
| Bear (low breadth, SPY below 200d MA) | 60% | 12 equity, 4 futures, 4 crypto |
| Neutral | 75% | 15 equity, 3 futures, 3 crypto |
| Bull (high breadth, SPY above 200d MA) | 90% | 18 equity, 1 futures, 1 crypto |

Cap computed from 3 signals: equity breadth, SPY vs 200d MA, equity vs bond spread.

---

## Protection Layers (5 independent, multiplicative)

| Layer | Frequency | What it does |
|---|---|---|
| **Circuit breakers** | Daily | Halt on >8% realised cash loss or >15% MDD |
| **EWS** (Early Warning) | Daily | 4 stress detectors → GREEN/YELLOW/ORANGE/RED/CRITICAL scale |
| **Vol targeting** | Daily | EWMA scaler targets 22% ann. vol (up in calm, down in turbulent) |
| **Intraday shock** | Every 5 min | VIX spike >15% or equity drop >3% → immediate scale to 25% |
| **Stop-losses** | Every 5 min | ATR-based per-position stop |

All layers are multiplicative: `scale = EWS × VT × ISD`

### EWS Regime Scale
| Regime | Score | Scale |
|---|---|---|
| GREEN | < 0.25 | 100% |
| YELLOW | 0.25–0.40 | 70% |
| ORANGE | 0.40–0.55 | 40% |
| RED | 0.55–0.70 | 20% |
| CRITICAL | > 0.70 | 5% |

### Intraday Shock Scale
| State | Trigger | Scale |
|---|---|---|
| CLEAR | Normal | 100% |
| CAUTION | VIX +10% or equity -2% | 60% |
| SHOCK | VIX +15% or equity -3% | 25% |
| RECOVERY | 5-day ramp after shock | 30%→50%→75%→90%→100% |

---

## Commands

```bash
pip install -r requirements.txt

# Full 2018-2025 backtest
python main.py backtest

# 3-way comparison: baseline vs vol-targeting vs EWS+VT
python main.py compare

# Overfitting validation (walk-forward + permutation test)
python main.py validate

# Paper trading (requires Alpaca paper keys in .env)
cp .env.example .env
python main.py paper
# or: docker-compose up

# What-if analysis
python whatif.py --suite capital    # €10k–€200k
python whatif.py --suite risk       # conservative → aggressive
python whatif.py --suite strategy   # signal parameter sweep
python whatif.py --suite all        # all 7 suites

# Today's signals + EWS regime
python main.py signals
python -c "
from utils.config_loader import load_config
from regime.ews import EarlyWarningSystem
ews = EarlyWarningSystem(load_config('config/settings.yaml'))
score, scale, colour = ews.score_today()
print(f'EWS: {colour} | score={score:.3f} | position scale={scale:.0%}')
"
```

---

## Configuration (`config/settings.yaml`)

| Section | Key settings |
|---|---|
| `capital` | `initial_equity: 25000`, `max_portfolio_heat: 0.40` |
| `risk` | `max_position_pct: 0.15`, `daily_loss_limit: 0.08`, `max_drawdown_halt: 0.15` |
| `strategy` | `lookback_fast: 20`, `lookback_slow: 60`, `rebalance_frequency: weekly` |
| `strategy.predictive` | `credit_regime_enabled: true`, `credit_regime_weight: 0.30` |
| `dynamic_universe` | `enabled: true`, `top_n: 20`, `adaptive_caps: true` |
| `intraday_shock` | `enabled: true`, `vix_spike_shock: 0.15`, `equity_drop_shock: 0.03` |
| `vol_targeting` | `enabled: true`, `target_vol: 0.22`, `max_leverage: 1.5` |
| `ews` | `enabled: true`, all 4 sub-detectors on |
| `costs` | `impact_scale: 1.0`, `capital_gains_tax_rate: 0.0` |

---

## Candidate Universe (55 instruments)

**Equities/ETFs (40):** SPY, QQQ, IWM, DIA, MDY, GLD, TLT, AGG, LQD, HYG, SHY,
VGK, EEM, EMXC, EWJ, EWZ, EWY, EWA, EWC, EWG, EWU,
XLK, XLE, XLF, XLV, XLU, XLB, XLI, XLP, XLY, SOXX, VNQ,
IBB, XBI, ARKK, PDBC, DBC, USO, SLV, COPX

**Futures (7):** ES=F, NQ=F, GC=F, CL=F, SI=F, ZB=F, NG=F

**Crypto (8):** BTC-USD, ETH-USD, SOL-USD, BNB-USD, ADA-USD, AVAX-USD, DOT-USD, LINK-USD

---

## Backtest Results Summary (2018–2025, $25k)

| Configuration | Ann. Return | Sharpe | Max DD | Calmar |
|---|---|---|---|---|
| Original 17 instruments | 5.3% | 0.044 | -20.6% | 0.258 |
| Fixed 22 (EMXC) | 4.5% | 0.016 | -17.6% | 0.256 |
| Dynamic 20/55 adaptive | **8.4%** | **0.113** | -22.2% | **0.379** |
| Dynamic + ISD | 7.4% | 0.094 | -22.2% | 0.333 |

*ISD reduces CVaR from 8.1% → 6.0% per day (better tail risk) at cost of ~1pp ann. return*

---

## AWS Deployment (~€11/month)

See `deploy/aws_setup.md`. Summary:
1. `docker build -t trading-system .`
2. Push to ECR
3. Deploy to ECS Fargate (24/7, auto-restart)
4. EventBridge daily report at 18:00 UTC

---

## Moving to Live

1. Set `TRADING_MODE=live` in `.env`
2. Switch Alpaca to live URL, Binance `testnet: false`
3. Run `python main.py live` — requires explicit confirmation

**⚠ Risk warning:** Past backtest performance does not guarantee future results.
Always paper-trade for an extended period before deploying real capital.
