# Paper Trading Runbook

**Strategy:** Multi-Factor Momentum + Mean Reversion (locked IS params 2018–2022)  
**Broker:** Alpaca (paper-api.alpaca.markets)  
**Capital:** $25,000 USD  
**Start date:** April 2026  
**Go-live criteria:** 12 months paper trading, Sharpe > 0.50 sustained, at least one drawdown episode survived and recovered

---

## 1. Broker Connection

### Alpaca Paper Account Setup

1. Sign up at [alpaca.markets](https://alpaca.markets) → enable Paper Trading
2. Copy your API key and secret from the Alpaca dashboard

### Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with your Alpaca paper credentials:

```
TRADING_MODE=paper
ALPACA_API_KEY=PK...your_key
ALPACA_API_SECRET=...your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Verify connection:

```bash
python -c "
from execution.alpaca_broker import AlpacaBroker
import yaml
with open('config/settings.yaml') as f:
    cfg = yaml.safe_load(f)
broker = AlpacaBroker(cfg)
ok = broker.connect()
if ok:
    acct = broker.get_account()
    print(f'Connected. Equity: \${acct.equity:,.2f}  Cash: \${acct.cash:,.2f}')
else:
    print('Connection FAILED — check API keys')
"
```

### Key Config Verification

Before first run, confirm `config/settings.yaml` has:

| Setting | Required Value | Location |
|---|---|---|
| `system.mode` | `paper` | Top of settings.yaml |
| `brokers.alpaca.paper` | `true` | Brokers section |
| `brokers.alpaca.base_url` | `https://paper-api.alpaca.markets` | Brokers section |
| `capital.initial_equity` | `25000` | Capital section |
| `risk.max_drawdown_halt` | `0.15` | Risk section |
| `risk.daily_loss_limit` | `0.08` | Risk section |

### Starting the System

```bash
# Local
python main.py paper

# Docker
docker-compose up

# AWS ECS (on-demand via GitHub Actions)
# Go to Actions → CI/CD → Run workflow → environment=paper, action=deploy
```

### Health Check

Once running, the system exposes HTTP endpoints on port 8080:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check (returns 200 if process is running) |
| `GET /status` | System status: equity, positions, regime, last rebalance |
| `GET /signals` | Current signal scores for all instruments |

---

## 2. Position Sizing

### How Positions Are Sized

Position sizing flows through five independent layers, applied as `min()` (not multiplicative):

```
Raw Signal Score (composite of 5 factors)
    │
    ├─ Kelly Sizer → fractional Kelly (25%) blended 50/50 with inverse-vol
    │
    ├─ Risk Manager caps
    │    ├─ Max single position: 15% of equity
    │    ├─ Max sector exposure: 30% of equity
    │    └─ Max portfolio heat: 75% of equity
    │
    ├─ Regime Scale
    │    ├─ ChoppyDetector: GREEN=100%, YELLOW=80%, ORANGE=50%, RED=25%
    │    ├─ EWS: GREEN=100%, YELLOW=70%, ORANGE=40%, RED=20%, CRITICAL=5%
    │    └─ Intraday Shock: scale to 25% on VIX spike >20% or equity drop >10%
    │
    ├─ PositionAnomalyScorer (per-instrument)
    │    ├─ Crypto: floor 10% (up to 90% cut)
    │    ├─ Equities: floor 40% (up to 60% cut)
    │    ├─ ETF equities: floor 55% (up to 45% cut)
    │    └─ Hedges (TLT, GLD): always 100% (never cut)
    │
    └─ Drawdown Scale (progressive)
         ├─ DD < 8%: 1.0× (no reduction)
         ├─ DD 8–15%: linear from 1.0× to 0.5×
         ├─ DD 15–25%: linear from 0.5× to 0.2×
         └─ DD > 25%: 0.2× floor (no hard halt)
```

### Combined Scale Formula

```python
combined_scale = max(min(ews_scale, isd_scale, anomaly_scale), 0.50)
```

The floor of 50% prevents the system from going fully flat — the five independent layers provide
enough protection without stacking them multiplicatively.

### Position Count

| Setting | Value |
|---|---|
| Active positions | Top quartile of 20-stock PIT universe (~5 stocks) |
| Weighting | Equal weight within the long basket |
| Rebalance | Adaptive: biweekly when ChoppyScore < 0.17 (GREEN), weekly when YELLOW+ |
| Dynamic expansion | Up to 3 additional single stocks via Alpaca Screener (max 8% each), gated by choppy regime |

### Capital Allocation Example

With $25,000 equity, 5 equal-weight positions, and GREEN regime:

| Position | Weight | Dollar Value |
|---|---|---|
| AAPL | 20% | $5,000 |
| NVDA | 20% | $5,000 |
| META | 20% | $5,000 |
| MSFT | 20% | $5,000 |
| JPM  | 20% | $5,000 |

If ChoppyDetector goes ORANGE (50% scale), each position is halved to $2,500, freeing $12,500 to cash.

---

## 3. Daily Checklist

### Pre-Market (before 9:30 ET / 16:30 EEST)

- [ ] System is running (`GET /health` returns 200)
- [ ] Alpaca connection is live (`GET /status` shows account equity)
- [ ] No overnight alerts or error logs
- [ ] Check VIX level — if pre-market VIX > 25, expect YELLOW/ORANGE regime
- [ ] Review overnight news for macro shocks (tariff announcements, Fed surprises, geopolitical events)

```bash
# Quick health check
curl -s http://localhost:8080/health
curl -s http://localhost:8080/status | python -m json.tool

# Check logs for errors
grep -i "ERROR\|CRITICAL\|HALT\|circuit" logs/trading.log | tail -20
```

### During Market Hours (9:30–16:00 ET)

- [ ] Monitor regime colour in logs (expect `[REGIME] Choppy=GREEN/YELLOW/ORANGE/RED`)
- [ ] Confirm rebalance executes on schedule (weekly or biweekly depending on regime)
- [ ] If rebalance fires, verify orders filled (`GET /status` shows updated positions)
- [ ] Watch for intraday shock triggers (VIX spike >12% or equity drop >6% from open)

### Post-Market (after 16:00 ET / 23:00 EEST)

- [ ] Review daily P&L: `GET /status` or Alpaca dashboard
- [ ] Record in tracking spreadsheet: date, equity, daily return, regime colour, positions held
- [ ] Check drawdown: current DD from peak equity
- [ ] If DD > 8%, verify drawdown_scale is reducing positions (check logs for `drawdown_scale`)
- [ ] Review daily report email (if `daily_report.py` is configured)

```bash
# Generate daily report manually
python daily_report.py
```

### Weekly Review (end of each Friday)

- [ ] Calculate weekly Sharpe (rolling 4-week): target > 0.50 annualised
- [ ] Review turnover: should average ~30% per rebalance
- [ ] Check regime distribution for the week (bull vs bear days)
- [ ] Compare strategy return vs SPY return for the week
- [ ] Verify no orders were rejected or partially filled
- [ ] Review position concentration — no single position > 15% of equity

### Monthly Review

- [ ] Calculate monthly return and running Sharpe ratio
- [ ] Compare cumulative return vs SPY benchmark
- [ ] Review max drawdown — has the strategy recovered from any DD episodes?
- [ ] Check if any PIT universe changes are needed for the new year
- [ ] Run leakage tests to confirm backtest integrity: `python -m pytest tests/test_leakage_audit.py -v`
- [ ] Update go-live assessment: Sharpe still above 0.50? Drawdown episode survived?

---

## 4. Escalation Rules

### ChoppyDetector Regime Levels

| Level | Score | Position Scale | System Action |
|---|---|---|---|
| GREEN | < 0.17 | 100% | Normal operation, biweekly rebalance |
| YELLOW | 0.17–0.27 | 80% | Weekly rebalance activates, light trim |
| ORANGE | 0.27–0.40 | 50% | Exposure halved, dynamic expansion gated |
| RED | > 0.40 | 25% | Defensive mode — 75% cash, manual review required |

### RED Trigger Mid-Session Escalation Protocol

When ChoppyDetector triggers RED during market hours:

#### Immediate (automated — no manual action needed)

1. **Position scale drops to 25%** — the system automatically reduces all positions to 25% of normal sizing at the next rebalance cycle
2. **Dynamic universe expansion is gated** — no new single-stock positions will be added
3. **Rebalance frequency stays weekly** — the system will rebalance at the end of the current week (or sooner if VIX-spike forced rebalance triggers)
4. **EWS and ISD layers remain active** — if EWS is also RED (score > 0.55), the combined scale drops further via `min(choppy_25%, ews_20%) = 20%` with a 50% floor applied

#### Manual Review (within 30 minutes of RED trigger)

1. **Check the trigger source** — review logs for which ChoppyDetector feature groups are elevated:
   ```bash
   grep -i "choppy\|regime\|RED" logs/trading.log | tail -30
   ```
2. **Identify the driver** — the 7 feature groups and their typical triggers:

   | Group | Weight | Common RED Triggers |
   |---|---|---|
   | vol_spike | 18% | VIX jump >30, multi-day VIX surge |
   | price_vol | 18% | Realised vol + vol-of-vol spike |
   | macro_credit | 16% | HYG/LQD spread widening (credit stress) |
   | event_shock | 16% | VIX velocity >2σ, term structure inversion |
   | commodity_fx | 12% | Oil shock, DXY spike, gold surge |
   | breadth | 12% | Advance-decline collapse, low new highs |
   | sentiment | 8% | Extreme fear readings |

3. **Cross-reference with EWS** — check the EWS colour in logs. If both ChoppyDetector and EWS are RED/CRITICAL, the event is systemic.

4. **Verify positions are actually scaled down** — check Alpaca dashboard to confirm position values have been reduced. If the system hasn't rebalanced yet (rebalance day hasn't arrived), the scaling will apply at the next rebalance.

#### Decision Tree

```
ChoppyDetector goes RED
    │
    ├─ Is this a VIX spike event (>20% intraday)?
    │     YES → Intraday Shock Detector also fires → immediate scale to 25%
    │           No further action needed, system handles it
    │
    ├─ Is EWS also RED or CRITICAL?
    │     YES → Systemic stress. Combined scale = max(min(25%, 20%), 50%) = 50% floor
    │           Monitor hourly. If DD > 10%, consider manual intervention (see below).
    │
    ├─ Is this a slow-building choppy regime (score creeping up over days)?
    │     YES → Normal choppy market. System handles it. Monitor daily.
    │           Expect YELLOW → ORANGE → RED progression over 3–10 days.
    │           HYG/LQD credit stress provides early warning.
    │
    └─ Is DD already > 15%?
          YES → drawdown_scale is at 0.5× or below. Combined with RED 25% scale,
                effective exposure is ~12.5%. System is nearly flat.
                DO NOT manually override or halt — progressive scaling handles recovery.
          NO  → System is working as designed. Monitor but do not intervene.
```

#### When to Manually Intervene

Manual intervention should be rare. The five protection layers are designed to handle
all market conditions without human override. Only intervene if:

| Condition | Action |
|---|---|
| System is not running (health check fails) | Restart: `python main.py paper` or `docker-compose up` |
| Orders are being rejected by Alpaca | Check Alpaca status page, verify API keys, check account buying power |
| Drawdown > 25% AND no signs of recovery after 5 trading days | Consider pausing the system: set `desired_count=0` on ECS or `Ctrl+C` locally. Review strategy assumptions. |
| Data feed is stale (prices not updating) | Restart system. Check yfinance / Alpaca data API status. |
| Bug in position sizing (positions larger than expected) | Manually flatten all positions via Alpaca dashboard, file a bug, fix before restarting |

#### What NOT to Do

- **Do not manually close positions** because the market is falling. The drawdown scaling and regime
  detection handle this. Panic selling locks in losses.
- **Do not increase position sizes** to "buy the dip" during RED. The system will automatically
  scale back up when ChoppyDetector returns to GREEN/YELLOW.
- **Do not change strategy parameters** based on recent performance. The weights are locked from IS
  and should not be modified during paper trading. The entire point is to validate these exact params.
- **Do not disable protection layers** (EWS, ChoppyDetector, PositionAnomaly) to increase returns.
  These layers exist to prevent catastrophic drawdowns.

---

## 5. Recovery from RED

When ChoppyDetector score drops back below 0.40 (ORANGE) or 0.27 (YELLOW):

1. Position scale automatically increases (50% for ORANGE, 80% for YELLOW)
2. The system will rebuild positions at the next rebalance
3. No manual action needed — recovery is gradual by design
4. After an intraday shock, the system uses a 3-day recovery ramp: 50% → 80% → 100%

---

## 6. Logging and Monitoring

### Log Locations

| Source | Location |
|---|---|
| Application logs | `logs/trading.log` |
| Docker logs | `docker logs trading-bot` |
| AWS CloudWatch | `/ecs/trading-bot-paper` |
| Alpaca activity | Alpaca dashboard → Activity |

### Key Log Patterns to Watch

```bash
# Regime changes
grep "\[REGIME\]" logs/trading.log | tail -10

# Rebalance events
grep "rebalance\|REBAL" logs/trading.log | tail -10

# Risk events
grep "halt\|circuit\|drawdown_scale\|ISD\|shock" logs/trading.log | tail -10

# Order execution
grep "order\|fill\|reject" logs/trading.log | tail -10

# EWS alerts
grep "EWS:" logs/trading.log | tail -10
```

### AWS CloudWatch (if deployed on ECS)

```bash
# Tail live logs
aws logs tail /ecs/trading-bot-paper --follow --region eu-north-1

# Search for RED triggers in the last 24h
aws logs filter-log-events \
  --log-group-name /ecs/trading-bot-paper \
  --start-time $(date -d '24 hours ago' +%s000) \
  --filter-pattern "RED" \
  --region eu-north-1
```

---

## 7. Go-Live Criteria

After 12 months of paper trading, evaluate against these thresholds:

| Metric | Threshold | How to Measure |
|---|---|---|
| Annualised Sharpe | > 0.50 | Rolling 252-day Sharpe from daily returns |
| Max Drawdown | < 15% | Worst peak-to-trough during paper period |
| Drawdown Recovery | At least 1 episode survived | Must have experienced DD > 5% and recovered |
| Win Rate | > 50% | Daily win rate over full period |
| Correlation to Backtest | > 0.60 | Compare paper returns to WF OOS returns for same period |
| System Uptime | > 95% | Fraction of trading days where system ran without manual intervention |

If all thresholds are met, proceed to live capital deployment per `DEPLOY_AWS.md`.
If any threshold fails, extend paper trading or review strategy assumptions.
