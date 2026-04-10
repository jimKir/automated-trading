# scripts/

Utility scripts for infrastructure setup and data management.

## `run_backfill.py`

CLI wrapper for running historical data backfill via Prefect flows.

```bash
python scripts/run_backfill.py --symbols AAPL MSFT GOOGL --start-date 2024-01-01
python scripts/run_backfill.py --vendors databento alpaca --schemas ohlcv-1d
```

Default: AAPL, MSFT, GOOGL, AMZN, META from 2 years ago to yesterday.

## `setup_infra.sh`

Terraform-based infrastructure provisioning (for market data platform). Requires Terraform + Azure CLI.

```bash
./scripts/setup_infra.sh          # dev environment (default)
./scripts/setup_infra.sh prod     # production environment
```
