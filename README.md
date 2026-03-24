# Market Data Ingestion & Storage Platform

A production-grade, cloud-deployable platform for ingesting, storing, and serving historical market data — designed as the foundational data layer for an automated trading system.

**Execution Broker:** Alpaca · **Data Vendors:** Databento (historical) + Alpaca (validation) · **Language:** Python 3.11+

---

## Architecture

The platform follows a three-layer architecture built around the principle: **ingest once, reuse many times**.

```
[Databento API] ──┐
                  ├──▶ [Ingestion Service] ──▶ [Raw Data Lake] ──┬──▶ [Backtest Dataset API]
[Alpaca API] ─────┘                                              └──▶ [Feature Store for AutoML]
```

| Layer | Purpose |
|-------|---------|
| **Ingestion** | Connects to external data vendors, handles pagination, retry logic, rate limiting, and raw data capture |
| **Storage** | Maintains a canonical, normalized data lake in Parquet format, partitioned by asset class, date, symbol, and schema |
| **Serving** | Provides high-performance read access for backtesting engines and AutoML feature stores |

---

## Project Structure

```
├── .github/workflows/ci.yml        # GitHub Actions — lint, test, Docker build
├── config/
│   ├── config.yaml                  # Main configuration (env var interpolation)
│   ├── config.dev.yaml              # Development overrides
│   └── config.prod.yaml             # Production overrides
├── docker/
│   ├── Dockerfile                   # Multi-stage Python 3.11 image
│   ├── docker-compose.yml           # Full local stack (app, Prefect, Prometheus, Grafana)
│   └── prometheus.yml               # Prometheus scrape config
├── flows/
│   ├── backfill_flow.py             # Prefect: bulk historical backfill
│   ├── incremental_flow.py          # Prefect: daily incremental updates (5pm ET Mon–Fri)
│   └── feature_flow.py              # Prefect: feature generation pipeline
├── infra/terraform/                 # Azure IaC (Blob Storage, ACR, ACI, Key Vault, Monitor)
├── scripts/
│   ├── run_backfill.py              # Standalone backfill runner
│   └── setup_infra.sh               # Infrastructure bootstrap script
├── src/market_data/
│   ├── config.py                    # Pydantic settings — YAML + env var resolution
│   ├── cli.py                       # Click CLI — 6 commands
│   ├── ingestion/
│   │   ├── base.py                  # Abstract vendor client, retry logic, error types
│   │   ├── databento_client.py      # Databento Historical API (databento-python SDK)
│   │   ├── alpaca_client.py         # Alpaca Historical API (alpaca-py SDK)
│   │   ├── backfill.py              # Bulk backfill orchestrator (30-day chunks, resumable)
│   │   ├── incremental.py           # Incremental updater from last watermark
│   │   ├── rate_limiter.py          # Token bucket rate limiter + priority queue + cost tracker
│   │   └── checkpoint.py            # Checkpoint manager for resumable downloads
│   ├── storage/
│   │   ├── cloud_storage.py         # Storage abstraction (Local / Azure Blob / S3 / GCS)
│   │   ├── raw_lake.py              # Raw data lake — immutable, append-only, SHA-256 sidecars
│   │   ├── analytics_lake.py        # Normalized Parquet lake — Snappy, 128 MB row groups
│   │   └── symbol_master.py         # Symbol master (SQLite) — FIGI mapping, options contracts
│   ├── transforms/
│   │   ├── normalize.py             # DBN/JSON → Parquet, UTC timestamp standardization
│   │   ├── corporate_actions.py     # Split & dividend adjustments (apply-on-read)
│   │   └── features.py              # 17 technical features with versioning & lineage
│   ├── serving/
│   │   ├── backtest_api.py          # get_bars() → pandas/polars, LRU cache, as-of-date
│   │   ├── feature_store.py         # Precomputed features, training dataset generation
│   │   └── quality.py               # Daily quality checks, cross-vendor validation, alerts
│   └── monitoring/
│       ├── metrics.py               # Prometheus counters, histograms, gauges
│       ├── health.py                # Liveness & readiness endpoints
│       └── alerts.py                # Slack / email webhook alerts
└── tests/                           # pytest suite — 55 tests across all layers
```

---

## Features Delivered

### Ingestion Layer

| Capability | Implementation |
|------------|---------------|
| Databento client | `databento-python` SDK — trades, quotes, OHLCV (1m/1h/1d), market depth (MBP-1/10, MBO), imbalance, statistics. Auto-detects batch mode for requests > 5 GB |
| Alpaca client | `alpaca-py` SDK — stock bars (minute/hour/day), crypto bars, options bars. Pagination with page tokens. Corporate action adjustment parameter |
| Bulk backfill | 30-day chunk partitioning, checkpointing every 1000 records or 100 MB, resume-on-failure, symbol universe filters |
| Incremental updates | From last watermark timestamp, primary key deduplication (`timestamp, symbol, exchange, sequence_number`), late-arriving record overwrite |
| Rate limiting | Token bucket per vendor (Databento 1000/min, Alpaca 10000/min), priority queue for urgent symbols |
| Cost tracking | Cumulative API usage per vendor per month, configurable budget threshold with automatic halt + alert |

### Storage Layer

| Capability | Implementation |
|------------|---------------|
| Cloud abstraction | Factory pattern — `local`, `azure`, `s3`, `gcs`. Switch via `STORAGE_PROVIDER` env var |
| Raw data lake | Immutable append-only. Path: `/raw/<vendor>/<asset_class>/<schema>/<year>/<month>/<day>/<symbol>.<ext>`. Metadata sidecar JSON with SHA-256 checksum |
| Analytics lake | Vendor-agnostic Parquet, Snappy compression, 128 MB row groups. UTC nanosecond timestamps. Internal symbol IDs. Path: `/analytics/<asset_class>/<schema>/<year>/<month>/<symbol_id>.parquet` |
| Symbol master | SQLite (PostgreSQL-ready). Fields: `symbol_id, ticker, FIGI, ISIN, CUSIP, asset_class, primary_exchange, listing_date, delisting_date`. Options contract master included. Daily sync from Databento symbology + Alpaca assets |

### Transforms

| Capability | Implementation |
|------------|---------------|
| Normalization | DBN → Parquet conversion, JSON/CSV → Parquet, timestamp standardization to UTC nanoseconds, symbol ID resolution |
| Corporate actions | Split and dividend adjustment table, apply-on-read via view layer |
| Feature engineering | **17 features** with semantic versioning and lineage tracking |

**Feature catalog:**

| Category | Features |
|----------|----------|
| Price-based | 1d / 5d / 20d log returns, SMA-20, EMA-50, RSI-14, MACD (12/26/9), Bollinger Bands (20, 2σ) |
| Volume | VWAP, OBV, volume ratio (vs 20-day average) |
| Volatility | Garman-Klass (20d), Parkinson (20d), ATR-14 |

### Serving Layer

| Capability | Implementation |
|------------|---------------|
| Backtest API | `get_bars(symbols, start, end, interval, adjustment, as_of_date)` → `pandas.DataFrame` / `polars.DataFrame` / `pyarrow.Table`. LRU cache (128 entries, 512 MB). Point-in-time as-of-date queries |
| Feature store | Precomputed feature tables. Training dataset generation with feature selection, lookback window, prediction horizon, train/val/test split. Semantic versioning |
| Data quality | Daily checks: missing timestamps, price outliers (> 10σ), zero volume, negative prices. Cross-vendor validation (Alpaca vs Databento, 10 random symbols/day). Alert if S&P 500 completeness < 99.5% |

### Orchestration & Monitoring

| Capability | Implementation |
|------------|---------------|
| Workflows | 3 Prefect flows: bulk backfill, incremental daily (cron `0 17 * * 1-5`), feature generation. Market calendar aware (skip weekends/holidays). Alert after 2 consecutive failures |
| Metrics | Prometheus: ingestion throughput, query latency, error rates, data completeness |
| Health checks | HTTP liveness + readiness probes at `:8080/health/live` and `:8080/health/ready` |
| Alerting | Slack webhook + email for data quality failures, job failures, cost overruns |

### Infrastructure

| Capability | Implementation |
|------------|---------------|
| Containerization | Docker image (`python:3.11-slim`), health check built in |
| Local stack | `docker-compose` — app + Prefect Server + Prometheus + Grafana |
| Cloud IaC | Terraform for Azure (Blob Storage, Container Registry, Container Instances, Key Vault, Monitor) |
| CI/CD | GitHub Actions — ruff lint, pytest + coverage, Docker build on every push/PR to `main` |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API keys: [Databento](https://databento.com) and/or [Alpaca](https://alpaca.markets)

### 1. Clone and configure

```bash
git clone https://github.com/jimKir/automated-trading.git
cd automated-trading
cp .env.example .env
# Edit .env with your API keys
```

### 2. Install locally

```bash
pip install -e ".[vendors,dev]"
```

### 3. Or run with Docker Compose

```bash
cd docker
docker-compose up -d
```

This starts:
- **market-data** — the platform (`:8080` health, `:9090` metrics)
- **prefect-server** — workflow orchestration UI (`:4200`)
- **prometheus** — metrics collection (`:9091`)
- **grafana** — dashboards (`:3000`, admin/admin)

---

## CLI Usage

```bash
# Bulk backfill — Databento daily bars, 2019 to present
market-data backfill --vendor databento --schema ohlcv-1d \
    --start-date 2019-01-01 --symbols AAPL MSFT GOOGL TSLA NVDA

# Bulk backfill — Alpaca daily bars
market-data backfill --vendor alpaca --schema ohlcv-1d \
    --start-date 2019-01-01 --symbols AAPL MSFT

# Incremental update (from last watermark)
market-data update --vendor all

# Generate features (version 1.0.0)
market-data generate-features --symbols AAPL MSFT GOOGL TSLA NVDA \
    --start-date 2020-01-01 --end-date 2025-12-31

# Run data quality validation
market-data validate --date 2026-03-24

# Start serving API (health + metrics endpoints)
market-data serve --port 8080

# Show platform status
market-data status
```

---

## Configuration

Configuration is loaded from `config/config.yaml` with full `${ENV_VAR:-default}` interpolation. Environment-specific overlays (`config.dev.yaml`, `config.prod.yaml`) can extend the base config.

All secrets are externalized — never committed to source control:

| Variable | Purpose |
|----------|---------|
| `DATABENTO_API_KEY` | Databento API authentication |
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_API_SECRET` | Alpaca API secret |
| `STORAGE_PROVIDER` | `local`, `azure`, `s3`, or `gcs` |
| `SLACK_WEBHOOK_URL` | Slack alerts (optional) |
| `AZURE_STORAGE_ACCOUNT` | Azure Blob (if using Azure) |
| `AWS_ACCESS_KEY_ID` | S3 (if using AWS) |
| `GCS_PROJECT` | GCS (if using GCP) |

See `.env.example` for the complete list.

---

## Data Coverage

| Asset Class | Source | Coverage | Schemas |
|-------------|--------|----------|---------|
| US Equities | Databento | 2019-01-01 → present (NYSE, NASDAQ, ARCA) | Trades, TBBO, MBP-1/10, OHLCV (1m/1h/1d), Imbalance, Statistics |
| US Equities | Alpaca | 7+ years (all tradable on Alpaca) | Stock bars (1m/1h/1d), corporate actions |
| US Options | Databento | 2021-01-01 → present (OPRA) | Trades, quotes, statistics |
| US Options | Alpaca | Available symbols | Options bars, snapshots |
| Crypto | Alpaca | All supported pairs | Crypto bars (1m/1h/1d) |
| Futures | Databento | 2020-01-01 → present (CME: ES, NQ) | Trades, depth, OHLCV (optional phase) |

---

## Cloud Deployment (Azure)

```bash
cd infra/terraform
terraform init
terraform plan -var="resource_group=rg-market-data" -var="location=westeurope"
terraform apply
```

Provisions: Azure Blob Storage, Container Registry, Container Instances, Key Vault, and Monitor workspace.

---

## Running Tests

```bash
# Full suite
pytest -v

# With coverage
pytest --cov=market_data --cov-report=term-missing -v

# Specific layer
pytest tests/test_ingestion/ -v
pytest tests/test_storage/ -v
pytest tests/test_serving/ -v
pytest tests/test_transforms/ -v
```

---

## Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Bulk backfill | 5 years of minute bars for 500 equities within 48 hours |
| Incremental daily | 3000 symbols within 30 minutes |
| Feature generation | 1 year daily features for 500 symbols within 2 hours |
| Query latency | p50 < 500 ms, p95 < 2s, p99 < 5s |
| Scale | 10,000+ symbols, 10 TB raw + 50 TB analytics |
| Availability | 99.5% uptime (serving layer) |
| Data durability | 99.999999999% (11 nines, cloud object storage) |
| Disaster recovery | RTO 4 hours, RPO 0 |
| Security | RBAC, TLS 1.3, secrets in vault, 7-year audit logs |

---

## License

MIT
