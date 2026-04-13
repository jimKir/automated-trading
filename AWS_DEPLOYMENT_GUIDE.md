# AWS Deployment Guide — Production + Paper Trading

## TL;DR

You have two Terraform-based environments ready to deploy:

1. **Production** — Always active, auto-scales on market hours
2. **Paper** — On-demand strategy testing, manual control

### Quick Start (from your Mac)

```bash
cd ~/Documents/trading/automated-trading

# 1. Commit the infra changes (git push will auto-deploy production)
git add .github/ infra/
git commit -m "feat: AWS IaC v2 — dual envs, terraform plan on PRs, cost optimizations"
git push origin main

# 2. Wait for GitHub Actions to deploy production (5-10 min)
#    → You'll see in Actions: lint → test → docker → deploy-production ✅

# 3. Start paper trading on-demand via GitHub
#    Go to Actions → CI / CD → Run workflow
#    - Environment: paper
#    - Action: deploy
#    → Creates paper environment + starts trading immediately

# 4. Monitor
aws logs tail /ecs/trading-bot-production --follow --region eu-north-1
aws logs tail /ecs/trading-bot-paper --follow --region eu-north-1
```

---

## Architecture Overview

### Two Terraform Workspaces

Both share the same code but different configurations via tfvars:

```
infra/terraform/aws/
├── main.tf                (shared, conditional EventBridge)
├── variables.tf           (environment + enable_schedules)
├── outputs.tf
└── envs/
    ├── production.tfvars  (enable_schedules=true, desired_count=1)
    └── paper.tfvars       (enable_schedules=false, desired_count=0)
```

### Production Environment

- **Triggers**: Automatic deployment on every `git push` to main
- **EventBridge Schedules**:
  - Start: Mon–Fri 14:25 UTC (09:25 ET — 5 min before open)
  - Stop: Mon–Fri 21:05 UTC (17:05 ET — 5 min after close)
- **Always Running During Market Hours** (fully automated)
- **Cost**: ~$11/month (only market hours)

### Paper Environment

- **Triggers**: On-demand via GitHub Actions `workflow_dispatch`
- **No Auto-Schedules**: Manual control only
- **Three Actions**:
  1. `deploy` — Full build + Terraform apply (new image, infra changes)
  2. `start` — Scale to 1 (resume testing)
  3. `stop` — Scale to 0 (pause testing)
- **Cost**: Pay only for runtime (0 when stopped)

---

## New CI/CD Capabilities

### Terraform Plan on Pull Requests

Every PR automatically:
1. Plans both `production` and `paper` workspaces
2. Posts the diffs as PR comments (collapsible)
3. Shows exact AWS resource changes **before** merge
4. Can be reviewed before hitting main

Example PR comment:
```
✅ Terraform Plan — `production` (eu-north-1)
Plan: 0 to add, 1 to change, 0 to destroy
```

### Automated Deploy Pipeline

Main branch push triggers:
```
lint → test → dress-rehearsal → docker → deploy-production
                                          (auto applies terraform)
```

No manual steps — commit to main = deployed in 5 min.

### On-Demand Paper Testing

Test strategy changes without affecting production:
```
GitHub Actions → Run workflow → paper → deploy
→ New image built + paper env spun up
→ Start trading immediately
→ Stop when done (cost = $0)
```

---

## Cost Optimizations Built In

| Optimization | Savings |
|---|---|
| EventBridge market-hours schedules | ~$110/month (no 24/7 running) |
| Fargate SPOT capacity provider | ~20% discount on compute |
| 0.5 vCPU / 1GB RAM | Minimal but sufficient |
| ECR lifecycle (keep 5 images) | Prevent storage bloat |
| 10-day Secrets Manager recovery window | Cost + safety |

**Estimated Monthly Cost**:
- Production: $11 (market hours only)
- Paper: $0 (when stopped) — $3–5 per test session

---

## Diagnostics & Monitoring

Your code includes:

### Built-in Health Checks
```bash
python scripts/status.py                    # Portfolio dashboard
python healthcheck.py --section 1           # Environment check
python -m pytest tests/ -v                  # Full test suite
```

### CloudWatch Integration
- Auto-logs all trading cycles to CloudWatch
- CloudWatch alarm if bot stops logging > 30 min
- Tail logs in real-time:
  ```bash
  aws logs tail /ecs/trading-bot-production --follow
  ```

### Container Health Checks
- ECS health check: `pgrep -f live_engine.py` every 30s
- Auto-restarts on failure (grace period 60s)

---

## Deployment Checklist

- [ ] **On your Mac**:
  - [ ] `git add .github/ infra/`
  - [ ] `git commit -m "feat: AWS IaC v2..."`
  - [ ] `git push origin main`

- [ ] **Wait for CI/CD** (5–10 min):
  - [ ] Check GitHub Actions → CI / CD
  - [ ] Verify all gates pass (lint, test, docker)
  - [ ] See "Deploy: Production" ✅

- [ ] **Verify Production Running**:
  - [ ] `aws ecs list-services --cluster trading-bot-production-cluster --region eu-north-1`
  - [ ] `aws logs tail /ecs/trading-bot-production --follow --region eu-north-1`
  - [ ] Check for "=== Trading Cycle @" messages

- [ ] **Deploy Paper Trading**:
  - [ ] Go to GitHub Actions → CI / CD
  - [ ] Click "Run workflow"
  - [ ] Environment: `paper`, Action: `deploy`
  - [ ] Wait 3–5 min for build + deploy

- [ ] **Start Paper Trading**:
  - [ ] `aws logs tail /ecs/trading-bot-paper --follow --region eu-north-1`
  - [ ] Watch for trading cycles and orders
  - [ ] Monitor portfolio via `python scripts/status.py`

- [ ] **Test Quick Controls**:
  - [ ] GitHub Actions → Run workflow → paper → `stop`
  - [ ] Verify service scaled to 0
  - [ ] GitHub Actions → paper → `start`
  - [ ] Verify back online within 30s

---

## Terraform Workspace Management

If you need to manually apply infra changes:

```bash
cd infra/terraform/aws

# Production
terraform workspace select production || terraform workspace new production
terraform init -input=false
terraform apply -var-file=envs/production.tfvars

# Paper
terraform workspace select paper || terraform workspace new paper
terraform init -input=false
terraform apply -var-file=envs/paper.tfvars

# View state
terraform workspace list
terraform show
```

---

## Troubleshooting

### ECS Task Won't Start

```bash
# Check task logs
aws ecs describe-tasks \
  --cluster trading-bot-production-cluster \
  --tasks $(aws ecs list-tasks --cluster trading-bot-production-cluster --query 'taskArns[0]' --output text) \
  --region eu-north-1

# Check ECS events
aws ecs describe-services \
  --cluster trading-bot-production-cluster \
  --services trading-bot-production-service \
  --region eu-north-1 \
  --query 'services[0].events[:5]'
```

### Secrets Manager Error

Verify credentials in GitHub Actions secrets:
```bash
aws secretsmanager get-secret-value \
  --secret-id trading/alpaca_api_key \
  --region eu-north-1
```

### Terraform Plan Showing Unexpected Changes

Use workspace:
```bash
terraform workspace select production
terraform plan -var-file=envs/production.tfvars -no-color
```

---

## Key Files

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | 8 jobs (lint, test, docker, terraform-plan, deploy-prod, deploy-paper, control) |
| `infra/terraform/aws/main.tf` | ECR, ECS, Secrets, IAM, EventBridge (conditional) |
| `infra/terraform/aws/variables.tf` | `environment`, `enable_schedules`, `trading_mode` |
| `infra/terraform/aws/envs/*.tfvars` | Per-env config (production always-on, paper on-demand) |
| `deploy/deploy_aws.sh` | Manual deployment script (alt to GitHub Actions) |

---

## Next Steps

1. **Push to main** — triggers production deploy
2. **Monitor production** — watch logs for ~1 hour during market hours
3. **Deploy paper** — test new features without affecting production
4. **Iterate** — create a feature branch, GitHub automatically plans the changes

All CI/CD gates must pass (lint, test, dress-rehearsal) before production deploy. Paper deploys instantly with workflow_dispatch.

Good luck! 📈
