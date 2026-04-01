# AWS ML Architecture — Separate Processing Services

## Overview

The upgraded system splits into 5 independent AWS services, each with
its own compute profile, schedule, and failure isolation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        S3 (Central Hub)                             │
│   /features/daily/{symbol}/{date}.parquet                           │
│   /signals/weekly/{date}.parquet                                    │
│   /models/h2o_meta/latest/                                          │
│   /models/h2o_vol/latest/                                           │
│   /earnings_cache/{symbol}/{quarter}.json                           │
│   /options_cache/{symbol}/{date}.json                               │
└──────────┬──────────┬──────────┬──────────┬───────────┬────────────┘
           │          │          │          │           │
    ┌──────▼──┐  ┌────▼────┐  ┌─▼──────┐  ┌▼────────┐  ┌▼──────────┐
    │Feature  │  │Options  │  │Earnings│  │Meta-    │  │Trading    │
    │Extract  │  │Flow     │  │NLP     │  │Learner  │  │Engine     │
    │(daily)  │  │(daily)  │  │(daily) │  │(monthly)│  │(always-on)│
    │ECS Task │  │ECS Task │  │ECS Task│  │Batch    │  │ECS Fargate│
    │~15min   │  │~10min   │  │~20min  │  │~45min   │  │24/7       │
    │r6i.lg   │  │t3.med   │  │t3.lg   │  │r5.xlg   │  │t3.small   │
    │€4/mo    │  │€2/mo    │  │€3/mo   │  │€2/mo    │  │€11/mo     │
    └─────────┘  └─────────┘  └────────┘  └─────────┘  └───────────┘
```

## Service 1: Feature Extractor (daily, 4:45 PM ET)

**Purpose:** Distil Alpaca 1-min bars + Databento trades into weekly
features, stored as Parquet in S3 for all downstream consumers.

**Features extracted:**
- Opening gap fill rate (last 4 weeks of 1-min data)
- Trade imbalance ratio (Databento trades schema)
- VWAP distance at close (Alpaca 1-min)
- Intraday vol ratio (overnight/intraday)
- Volume-weighted return (1-min bars)

**Schedule:** `0 21 * * 1-5` (4:45 PM ET = 21:45 UTC)
**Service:** ECS Fargate task (r6i.large, 2 vCPU, 16GB RAM)
**Cost:** ~€4/month (15 min × 20 trading days/month)

```dockerfile
# Dockerfile.feature_extractor
FROM python:3.11-slim
COPY src/market_data/ .
CMD ["python", "-m", "flows.feature_flow", "--date", "today"]
```

---

## Service 2: Options Flow Processor (daily, 5:00 PM ET)

**Purpose:** Fetch OPRA options data via Databento, compute unusual
activity signals, IV skew, and PCR momentum for all active symbols.

**Schedule:** `0 22 * * 1-5` (5:00 PM ET = 22:00 UTC)
**Service:** ECS Fargate task (t3.medium)
**Cost:** ~€2/month

```python
# Entry point: python -m strategy.options_flow_signal --mode=batch --output=s3
```

---

## Service 3: Earnings NLP Processor (daily, 5:30 PM ET)

**Purpose:** Check SEC EDGAR for new 8-K filings, run FinBERT on
earnings transcripts, cache signals to S3.

**Schedule:** `30 22 * * 1-5` (5:30 PM ET = 22:30 UTC)
**Service:** ECS Fargate task (t3.large — needs RAM for FinBERT model)
**Cost:** ~€3/month

**Note:** FinBERT model (~500MB) stored in ECR or downloaded from
HuggingFace Hub on startup. EFS mount recommended for model caching.

---

## Service 4: Meta-Learner Retrainer (monthly, first Sunday)

**Purpose:** Retrain H2O AutoML meta-model on latest 12 months of
feature data and forward returns. Save to S3 model store.

**Schedule:** `0 2 1-7 * 0` (first Sunday, 2 AM UTC)
**Service:** AWS Batch (r5.xlarge — 4 vCPU, 32GB RAM for H2O JVM)
**Cost:** ~€2/month (1 run × ~45 min × €0.045/hr)

```python
# python core/h2o_meta_learner.py --retrain --train-end=today
```

**Retraining also triggered by:**
- SPY drawdown > 15% (regime change)
- Model RMSE degrades > 20% vs last training period
- Manual trigger via `aws batch submit-job`

---

## Service 5: Trading Engine (always-on, existing)

**Purpose:** Weekly rebalance + daily risk management. NOW reads
pre-computed signals from S3 instead of computing inline.

**Change from current:** Trading engine reads from S3:
```python
# New: read pre-computed signals
signals = pd.read_parquet(f"s3://{bucket}/signals/weekly/{last_friday}.parquet")

# Instead of computing inline (current behaviour)
signals = signal_generator.generate(all_data)
```

**Fallback:** If S3 signals not available (service failure), fall back
to inline computation (current behaviour unchanged).

**Cost:** ~€11/month (unchanged)

---

## Total Additional Cost: ~€11/month

| Service | Current | New | Delta |
|---|---|---|---|
| Trading engine | €11 | €11 | +€0 |
| Feature extractor | — | €4 | +€4 |
| Options flow | — | €2 | +€2 |
| Earnings NLP | — | €3 | +€3 |
| Meta-learner retraining | — | €2 | +€2 |
| S3 storage (~50GB features) | — | €1 | +€1 |
| **Total** | **€11** | **€23** | **+€12/month** |

---

## Deployment

### Step 1: Add Terraform resources

```hcl
# infra/terraform/ml_services.tf

resource "aws_ecs_task_definition" "feature_extractor" {
  family = "feature-extractor"
  cpu    = "2048"
  memory = "16384"
  # ... (see infra/terraform/ml_services.tf in repo)
}

resource "aws_batch_job_definition" "meta_learner_retrain" {
  name = "meta-learner-retrain"
  # r5.xlarge, 32GB RAM
}
```

### Step 2: Add EventBridge rules

```bash
# Feature extractor: daily at 4:45 PM ET
aws events put-rule --name "feature-extractor-daily" \
  --schedule-expression "cron(45 21 ? * MON-FRI *)"

# Meta-learner: first Sunday monthly
aws events put-rule --name "meta-learner-monthly" \
  --schedule-expression "cron(0 2 ? * SUN#1 *)"
```

### Step 3: S3 bucket structure

```
s3://trading-system-{account_id}/
├── features/
│   └── daily/
│       └── {symbol}/
│           └── {YYYY-MM-DD}.parquet
├── signals/
│   └── weekly/
│       └── {YYYY-MM-DD}.parquet    ← pre-computed by options+NLP services
├── models/
│   ├── h2o_meta/
│   │   └── {YYYY-MM-DD}/           ← versioned model snapshots
│   └── h2o_vol/
│       └── {YYYY-MM-DD}/
└── earnings_cache/
    └── {symbol}/
        └── {quarter}.json
```

---

## Failure Modes & Fallbacks

| Service fails | Impact | Fallback |
|---|---|---|
| Feature extractor | No new features | Use last available features (T-1) |
| Options flow | No options signal | Use 0.0 (neutral) for options component |
| Earnings NLP | No NLP signal | Use 0.0 (neutral) for earnings component |
| Meta-learner | No meta-signal | Use direct composite (current system) |
| S3 read fails | No pre-computed signals | Compute inline (current system) |

**Key property: every failure mode degrades gracefully to the previous
system behaviour. The trading engine never hard-fails due to ML services.**
