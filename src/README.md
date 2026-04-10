# src/

Market data platform -- ingestion, storage, serving, and monitoring infrastructure.

## `market_data/`

A standalone market data management system for collecting, storing, and serving historical market data. Can be used independently of the trading strategy.

### Submodules

| Module | Purpose |
|---|---|
| `config.py` | Pydantic-based settings with YAML + env var resolution |
| `cli.py` | Click CLI interface (`market-data backfill`, `market-data update`, etc.) |
| `historical_store.py` | Historical data collection/update/audit in Parquet format |
| `catalogue.py` | Data catalog management |
| `cache_guard.py` | Cache integrity and staleness checks |

### `ingestion/`

Data ingestion from external vendors:
- `databento_client.py`: Databento GLBX.MDP3 dataset (trades, quotes, OHLCV)
- `alpaca_client.py`: Alpaca historical data API
- `backfill.py`: Chunk-based historical backfill logic
- `incremental.py`: Daily incremental update logic
- `checkpoint.py`: Resumable ingestion with checkpoint persistence
- `rate_limiter.py`: Token bucket rate limiter for API calls
- `base.py`: Abstract vendor client interface

### `storage/`

Data persistence layer:
- `raw_lake.py`: Raw data lake (immutable source-of-truth)
- `analytics_lake.py`: Cleaned/normalized data for analysis
- `cloud_storage.py`: S3/Azure/GCS storage backends
- `symbol_master.py`: Symbol metadata and mapping

### `serving/`

Data access layer:
- `backtest_api.py`: API for backtest engine to query data
- `feature_store.py`: Computed feature storage and retrieval
- `quality.py`: Data quality validation (completeness, outlier detection)

### `transforms/`

Data transformation pipeline:
- `normalize.py`: OHLCV normalization and cleaning
- `features.py`: Technical indicator computation (SMA, EMA, RSI, MACD, Bollinger, VWAP, OBV, ATR, Garman-Klass, Parkinson)
- `corporate_actions.py`: Stock split and dividend adjustment

### `monitoring/`

Observability:
- `health.py`: Health check endpoints
- `metrics.py`: Prometheus metrics export (port 9090)
- `alerts.py`: Slack + email alerting

## Configuration

See `config/config.yaml` for data platform settings (vendor APIs, storage paths, ingestion schedules, feature definitions, quality thresholds).

## Running Standalone

```bash
# Install as package
pip install -e ".[all]"

# CLI
market-data backfill --symbols AAPL MSFT --start-date 2024-01-01
market-data update --vendor alpaca
```
