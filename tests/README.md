# tests/

Test suite for the trading system and market data platform.

## Running Tests

```bash
# All tests
pytest -v

# Market data platform tests only
pytest tests/test_ingestion/ tests/test_serving/ tests/test_storage/ tests/test_transforms/ -v

# Production readiness checks
pytest tests/production_readiness_test.py -v

# With coverage
pytest --cov=market_data --cov-report=xml -v
```

## Test Structure

### `production_readiness_test.py`

Dress rehearsal validation for paper trading deployment. Checks:
- Config contains required core symbols (SPY, QQQ, IWM, GLD, TLT, SHY, XLU, XLP, BTC, ETH)
- `rebalance_frequency` is set to "adaptive"
- Bull weights sum to 1.0
- Alpaca paper URL configured correctly
- ChoppyDetector scoring produces valid output
- PositionAnomalyScorer produces valid per-symbol scales

### `conftest.py`

Shared pytest fixtures: `tmp_dir`, `local_storage`, `analytics_lake`, `symbol_master` (pre-populated with AAPL/MSFT/GOOGL), `corporate_actions`, `feature_engineer`, `sample_ohlcv_df` (100-row synthetic data), `sample_ohlcv_table`.

### Market Data Platform Tests

| Directory | Tests |
|---|---|
| `test_ingestion/` | Checkpoint persistence, rate limiter token bucket |
| `test_serving/` | Feature store queries, data quality validation |
| `test_storage/` | Analytics lake, cloud storage backends, symbol master |
| `test_transforms/` | Feature engineering pipeline |

## CI

GitHub Actions runs lint (ruff) -> test (pytest + coverage) -> docker build. See `.github/workflows/ci.yml`.
