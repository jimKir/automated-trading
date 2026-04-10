# flows/

Prefect workflow orchestration for data ingestion and feature computation.

## `backfill_flow.py`

Historical data backfill flow. Processes Databento/Alpaca vendors in chunks (30-day windows). Uses CheckpointManager for resumability and TokenBucketRateLimiter for API throttling. Retries: 2 attempts, 60s delay.

## `feature_flow.py`

Feature computation and storage flow. Reads OHLCV from AnalyticsLake, computes technical/volume/volatility features via FeatureEngineer, persists to FeatureStore. Runs per symbol/month. Retries: 1 attempt, 30s delay.

## `incremental_flow.py`

Daily incremental data updates. Scheduled at 5 PM ET weekdays (`0 17 * * 1-5`). Handles late data arrivals with 3-day lookback window. Retries: 3 attempts, 30s delay.

## Running

Requires Prefect server (see `docker/docker-compose.yml` for local setup).

```bash
# Manual backfill
python scripts/run_backfill.py --symbols AAPL MSFT --start-date 2024-01-01

# Via Prefect UI
# Start Prefect: docker-compose --profile orchestration up
# Navigate to http://localhost:4200
```
