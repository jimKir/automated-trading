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
- [ ] Review Gmail draft with daily summary (created by Lambda at 22:00 UTC) — forward to `o.zoumpou@gmail.com` if desired
- [ ] If any go-live criterion is FAILING, an alert draft appears on Sunday evenings — see [§8](#8-email-alerts-via-lambda-gmail-draft-pipeline)

```bash
# Generate daily report manually (local — does not create Gmail draft)
python daily_report.py

# Trigger the Lambda manually to create a draft now
aws lambda invoke --function-name trading-gmail-draft \
  --payload '{"source": "manual"}' --region eu-north-1 /dev/stdout
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

---

## 8. Email Alerts via Lambda (Gmail Draft Pipeline)

GitHub Actions generates the scorecard and daily summary JSON files but does
not send email — no SMTP credentials are stored in GitHub.  Instead, an AWS
Lambda function fetches the committed JSON from the repo and creates a Gmail
draft that you review and forward from `kiritsis.di@gmail.com`.

Source code: [`infra/lambda-gmail-draft/`](../infra/lambda-gmail-draft/)

### Architecture

```
GitHub Actions (generates data only)
  └─ Commits results/daily_summary.json + results/paper_monitor.json
         │
EventBridge Scheduler
  └─ Triggers Lambda on schedule
         │
Lambda (trading-gmail-draft)
  ├─ Fetches JSON from GitHub raw content
  ├─ Formats plain-text email body
  └─ Creates Gmail draft via OAuth2 API
         │
You review draft → hit Send → o.zoumpou@gmail.com also receives
```

### Schedules

| Trigger | Cron (UTC) | EEST | What |
|---|---|---|---|
| Weekdays | `0 22 ? * MON-FRI *` | 01:00+1 | Daily performance summary draft |
| Sunday | `0 19 ? * SUN *` | 22:00 | Alert draft if any go-live criterion is FAILING |

### Prerequisites

- AWS CLI and [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) installed
- A Google Cloud project with the **Gmail API** enabled
- An OAuth 2.0 Client ID (type: Desktop app) from Google Cloud Console

### Step 1 — Obtain Gmail OAuth Refresh Token

```bash
cd infra/lambda-gmail-draft
pip install google-auth-oauthlib

# Download credentials.json from Google Cloud Console
# (APIs & Services → Credentials → OAuth 2.0 Client IDs → Download)

python get_refresh_token.py
# → Opens browser — sign in with kiritsis.di@gmail.com
# → Prints JSON with client_id, client_secret, refresh_token
```

### Step 2 — Store Secrets in SSM Parameter Store

```bash
# Gmail OAuth credentials (paste JSON from step 1)
aws ssm put-parameter \
  --name /trading/gmail-oauth \
  --type SecureString \
  --value '{
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "refresh_token": "YOUR_REFRESH_TOKEN",
    "token_uri": "https://oauth2.googleapis.com/token"
  }' \
  --region eu-north-1

# GitHub fine-grained PAT (repo Contents:read scope only)
aws ssm put-parameter \
  --name /trading/github-token \
  --type SecureString \
  --value "ghp_your_fine_grained_pat_here" \
  --region eu-north-1
```

| SSM Parameter | Type | Contents |
|---|---|---|
| `/trading/gmail-oauth` | SecureString | Gmail OAuth2 JSON (client_id, client_secret, refresh_token, token_uri) |
| `/trading/github-token` | SecureString | GitHub fine-grained PAT with `Contents:read` on `jimKir/automated-trading` |

### Step 3 — Deploy the Lambda

```bash
cd infra/lambda-gmail-draft

sam build
sam deploy \
  --stack-name trading-gmail-draft \
  --resolve-s3 \
  --capabilities CAPABILITY_NAMED_IAM \
  --region eu-north-1 \
  --parameter-overrides \
    GitHubRepo=jimKir/automated-trading \
    Recipients=kiritsis.di@gmail.com,o.zoumpou@gmail.com
```

### Step 4 — Verify

```bash
# Manual invocation
aws lambda invoke \
  --function-name trading-gmail-draft \
  --payload '{"source": "manual_test"}' \
  --region eu-north-1 \
  /dev/stdout
```

Check your Gmail drafts — a test draft should appear.  Delete it after
confirming the pipeline works.

### Cost

| Resource | Cost |
|---|---|
| Lambda | ~$0.00/month (128 MB × 30 s × ~30 invocations — well within free tier) |
| EventBridge Scheduler | Free |
| SSM Parameter Store | Free (standard parameters) |

### Updating Recipients

Redeploy with a different `Recipients` parameter override, or update the
Lambda environment variable `RECIPIENTS` in the AWS console.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `No data files found in repo` | Ensure GitHub Actions has committed `results/daily_summary.json` at least once |
| `403 Forbidden` on GitHub fetch | Verify the GitHub PAT in `/trading/github-token` has `Contents:read` scope |
| `invalid_grant` from Google OAuth | Refresh token expired — re-run `get_refresh_token.py` and update SSM |
| Draft not appearing | Check Lambda CloudWatch logs: `aws logs tail /aws/lambda/trading-gmail-draft --follow` |

---

## 9. ChoppyDetector ORANGE→RED — SSM Parameter Override Procedure

During a live session where ChoppyDetector transitions from ORANGE to RED,
you may need to tighten risk parameters immediately without redeploying the
system.  The trading engine reads override parameters from SSM Parameter
Store on every rebalance cycle.  This section gives the exact commands.

### How Overrides Work

The engine checks SSM before each rebalance:

```
Rebalance cycle starts
  └─ Read /trading/overrides/* from SSM
       │
       ├─ If parameter exists → use SSM value (overrides settings.yaml)
       └─ If parameter missing → use settings.yaml default
```

Overrides are immediate on the next rebalance — no restart needed.
Delete the parameter to revert to the default from `config/settings.yaml`.

### SSM Parameter Names

| Parameter | Type | Default | Description |
|---|---|---|---|
| `/trading/overrides/position_scale` | String | `1.0` | Global position scale multiplier (0.0–1.0) |
| `/trading/overrides/max_portfolio_heat` | String | `0.75` | Max portfolio heat |
| `/trading/overrides/max_position_pct` | String | `0.15` | Max single position % |
| `/trading/overrides/rebalance_frequency` | String | `adaptive` | Force `daily`, `weekly`, or `adaptive` |
| `/trading/overrides/choppy_override` | String | _(none)_ | Force regime: `GREEN`, `YELLOW`, `ORANGE`, `RED` |
| `/trading/overrides/trading_halt` | String | `false` | Set `true` to halt all new orders |

### ORANGE → RED Escalation Commands

Run these from your terminal when ChoppyDetector score crosses 0.40:

#### Step 1 — Force position scale to 25% immediately

```bash
aws ssm put-parameter \
  --name /trading/overrides/position_scale \
  --type String \
  --value "0.25" \
  --overwrite \
  --region eu-north-1
```

#### Step 2 — Cap portfolio heat to 25% (from default 75%)

```bash
aws ssm put-parameter \
  --name /trading/overrides/max_portfolio_heat \
  --type String \
  --value "0.25" \
  --overwrite \
  --region eu-north-1
```

#### Step 3 — Force daily rebalance (faster de-risking)

```bash
aws ssm put-parameter \
  --name /trading/overrides/rebalance_frequency \
  --type String \
  --value "daily" \
  --overwrite \
  --region eu-north-1
```

#### Step 4 — (Optional) Force regime to RED if detector is lagging

Use this only if you believe ChoppyDetector is understating stress
(e.g. score is 0.38 but credit spreads are blowing out):

```bash
aws ssm put-parameter \
  --name /trading/overrides/choppy_override \
  --type String \
  --value "RED" \
  --overwrite \
  --region eu-north-1
```

#### Emergency — Full trading halt

Stops all new orders.  Existing positions remain open but no rebalancing
occurs.  Use as a last resort:

```bash
aws ssm put-parameter \
  --name /trading/overrides/trading_halt \
  --type String \
  --value "true" \
  --overwrite \
  --region eu-north-1
```

### Verification

Confirm the override is active:

```bash
# Read current value
aws ssm get-parameter \
  --name /trading/overrides/position_scale \
  --region eu-north-1 \
  --query 'Parameter.Value' --output text

# List all active overrides
aws ssm get-parameters-by-path \
  --path /trading/overrides/ \
  --region eu-north-1 \
  --query 'Parameters[].{Name:Name,Value:Value}' \
  --output table
```

Check the engine logs to confirm the override was picked up:

```bash
grep "SSM override\|param_override" logs/trading.log | tail -10
```

### Recovery — Reverting Overrides After RED Clears

When ChoppyDetector drops back below 0.40 (ORANGE) or 0.27 (YELLOW),
remove the overrides to restore normal operation:

```bash
# Remove all overrides at once
for param in position_scale max_portfolio_heat rebalance_frequency choppy_override trading_halt; do
  aws ssm delete-parameter \
    --name "/trading/overrides/$param" \
    --region eu-north-1 2>/dev/null && echo "Deleted $param" || echo "$param not set"
done
```

Or remove selectively:

```bash
# Just remove the position scale override (revert to auto)
aws ssm delete-parameter \
  --name /trading/overrides/position_scale \
  --region eu-north-1
```

### Recommended Escalation Timeline

| Time After RED | Action | Command |
|---|---|---|
| T+0 min | Scale to 25%, cap heat to 25% | Steps 1–2 above |
| T+0 min | Force daily rebalance | Step 3 |
| T+30 min | Review logs, check EWS colour | `grep -i "choppy\|ews\|RED" logs/trading.log` |
| T+30 min | If EWS also RED and DD > 10% | Force regime override (Step 4) |
| T+60 min | If DD > 15% and accelerating | Trading halt (Emergency) |
| Recovery | ChoppyScore < 0.40 for 2 consecutive days | Delete all overrides |

### What NOT to Override

- **Strategy weights** (`bull_w_ts_mom`, etc.) — these are locked from in-sample and must not change during paper trading
- **Drawdown scale** — this is computed automatically from the equity curve; overriding it breaks the progressive scaling logic
- **EWS thresholds** — these are set by risk logic, not optimised; changing them undermines the backtest validity
