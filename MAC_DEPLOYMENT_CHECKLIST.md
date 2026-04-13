# Mac Deployment Checklist — Ready to Deploy Now

Your code is **fully synced** and all infra is ready. This checklist takes ~15 min start-to-finish.

## Step 1: Sync Your Mac (1 min)

```bash
cd ~/Documents/trading/automated-trading
git pull origin main
```

This pulls the 3 new commits:
- `.github/workflows/ci.yml` (8 jobs: lint, test, docker, terraform-plan, deploy-prod/paper, control)
- `infra/terraform/aws/main.tf` (conditional EventBridge, recovery_window=10)
- `infra/terraform/aws/variables.tf` (enable_schedules, per-env tfvars)
- `infra/terraform/aws/envs/production.tfvars` (always-on, eventbridge=true)
- `infra/terraform/aws/envs/paper.tfvars` (on-demand, eventbridge=false)

---

## Step 2: Commit & Push (1 min)

```bash
git add AWS_DEPLOYMENT_GUIDE.md DEPLOYMENT_SUMMARY.txt
git commit -m "docs: deployment guides for AWS dual environments"
git push origin main
```

This triggers **automatic production deployment** via GitHub Actions.

---

## Step 3: Watch Production Deploy (5–10 min)

Go to: **GitHub** → Your repo → **Actions** → **CI / CD**

Watch the pipeline:
```
✅ lint
✅ test (pytest)
✅ dress-rehearsal (module smoke tests)
✅ docker (container build)
⏳ deploy-production (5 min) — creates cluster, deploys image, starts service
```

You'll see in the deployment summary:
```
| Commit | abc123def... |
| Environment | production (always active) |
| Schedules | EventBridge market-hours (14:25–21:05 UTC) |
```

---

## Step 4: Verify Production (2 min)

In your Mac terminal:

```bash
# Check the cluster exists and service is running
aws ecs list-services \
  --cluster trading-bot-production-cluster \
  --region eu-north-1

# Tail logs (will show "=== Trading Cycle @" every ~1 min)
aws logs tail /ecs/trading-bot-production --follow --region eu-north-1

# Press Ctrl+C to stop tailing after ~10 messages
```

**Expected output:**
```
2026-04-13 05:45:12 | INFO     | LiveEngine         | === Trading Cycle @ 2026-04-13 05:45:12 UTC ===
2026-04-13 05:45:12 | INFO     | SignalGenerator    | Latest signals: {...}
2026-04-13 05:45:12 | INFO     | AlpacaBroker       | [Alpaca] BUY SPY qty=0.01 status=accepted
```

✅ Production is live.

---

## Step 5: Deploy Paper Trading (3 min)

Go to: **GitHub** → **Actions** → **CI / CD** → **Run workflow** button

Fill in:
```
Environment: paper
Action: deploy
```

Click **Run workflow**.

Watch the same pipeline build + deploy paper environment. When done, you'll see:
```
| Environment | paper (on-demand) |
| Schedules | none (manual control) |
```

---

## Step 6: Monitor Paper (1 min)

```bash
# Tail paper logs
aws logs tail /ecs/trading-bot-paper --follow --region eu-north-1
```

Wait for first trading cycle (~60s after deployment). You should see:
```
2026-04-13 05:46:00 | INFO     | LiveEngine         | === Trading Cycle @ 2026-04-13 05:46:00 UTC ===
2026-04-13 05:46:05 | INFO     | AlpacaBroker       | [Alpaca] BUY QQQ qty=0.25 status=accepted
2026-04-13 05:46:10 | INFO     | AlpacaBroker       | [Alpaca] SELL XLF qty=0.10 status=accepted
```

✅ Paper is trading.

---

## Step 7: Check Portfolio (optional)

```bash
cd ~/Documents/trading/automated-trading
python scripts/status.py
```

Shows current positions, P&L, and risk metrics for both paper + production.

---

## Step 8: Test Quick Controls (optional)

Stop paper trading without stopping the infrastructure:

```bash
# Via GitHub Actions
GitHub → Actions → CI / CD → Run workflow
  Environment: paper
  Action: stop

# Via AWS CLI (faster)
aws ecs update-service \
  --cluster trading-bot-paper-cluster \
  --service trading-bot-paper-service \
  --desired-count 0 \
  --region eu-north-1

# Verify stopped
aws ecs describe-services \
  --cluster trading-bot-paper-cluster \
  --services trading-bot-paper-service \
  --region eu-north-1 | grep desiredCount
# Should show: "desiredCount": 0

# Resume
aws ecs update-service \
  --cluster trading-bot-paper-cluster \
  --service trading-bot-paper-service \
  --desired-count 1 \
  --region eu-north-1
```

---

## Expected Cost

| Environment | Runtime | Cost |
|---|---|---|
| Production | Market hours only (14:25–21:05 UTC, Mon–Fri) | ~$11/month |
| Paper (this session) | 10 min | ~$0.08 |
| Paper (ongoing, 1 hr/day) | 20 hours/month | ~$3–5/month |

**Total: ~$15/month** (production + light testing)

---

## What's Running Now

### Production
- **Status**: Always on during market hours
- **Schedule**: Auto-start 14:25 UTC, auto-stop 21:05 UTC
- **Trading**: Live Alpaca orders (paper mode, not live)
- **Logs**: `/ecs/trading-bot-production`
- **Cost**: ~$11/month

### Paper
- **Status**: Just deployed
- **Schedule**: Manual control only
- **Trading**: Immediate (should see cycles in logs)
- **Logs**: `/ecs/trading-bot-paper`
- **Cost**: $0 when stopped, ~$3–5 per session

---

## Next Steps

1. ✅ Watch logs for 15 min → verify both environments trading
2. ✅ Check portfolio: `python scripts/status.py`
3. ✅ Test stop/start on paper (verify quick control works)
4. Stop paper when done testing: AWS CLI or GitHub Actions
5. Monitor production during market hours (next trading day)

---

## Troubleshooting

### "Task failed to start"
```bash
aws ecs describe-tasks \
  --cluster trading-bot-production-cluster \
  --tasks $(aws ecs list-tasks --cluster trading-bot-production-cluster \
    --query 'taskArns[0]' --output text) \
  --region eu-north-1
```

Look for `stopCode` or errors in `taskStatus`.

### "No trading cycles in logs"
The service might still be initializing (~60s). Wait 90s and retry.

### "EventBridge not working"
Production is fully manual until you're confident. Ignore EventBridge during testing.

---

## Key Files (for reference)

```
.github/workflows/ci.yml          → 8 jobs (lint, test, docker, deploy)
infra/terraform/aws/main.tf       → ECR, ECS, Secrets, IAM, EventBridge
infra/terraform/aws/envs/         → production.tfvars, paper.tfvars
AWS_DEPLOYMENT_GUIDE.md           → Comprehensive reference
DEPLOYMENT_SUMMARY.txt            → Quick reference + cost breakdown
```

---

## You're All Set! 🚀

Go push and watch it deploy. Both environments are production-ready.
