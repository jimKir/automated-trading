"""Prefect flow for incremental data updates."""

from __future__ import annotations

from datetime import date, timedelta

import structlog
from prefect import flow, task
from prefect.tasks import task_input_hash

from market_data.config import get_settings
from market_data.ingestion.incremental import IncrementalUpdater, IncrementalSummary
from market_data.ingestion.databento_client import DatabentoClient
from market_data.ingestion.alpaca_client import AlpacaClient
from market_data.ingestion.rate_limiter import TokenBucketRateLimiter
from market_data.monitoring.alerts import AlertManager, AlertSeverity

logger = structlog.get_logger(__name__)


def _is_market_open(check_date: date) -> bool:
    """Check if the market is open (basic weekday check).

    Args:
        check_date: Date to check.

    Returns:
        True if weekday (Mon-Fri).
    """
    return check_date.weekday() < 5


@task(retries=3, retry_delay_seconds=30)
def incremental_update_vendor(
    vendor: str,
    symbols: list[str],
    schemas: list[str],
    data_dir: str,
    lookback_days: int = 3,
) -> IncrementalSummary:
    """Run incremental update for a single vendor.

    Args:
        vendor: Vendor name.
        symbols: Symbols to update.
        schemas: Schemas to fetch.
        data_dir: Base data directory.
        lookback_days: Lookback window for late arrivals.

    Returns:
        Incremental update summary.
    """
    settings = get_settings()
    rate_limiter = TokenBucketRateLimiter(rate=5.0, burst=10)

    if vendor == "databento":
        client = DatabentoClient(
            api_key=settings.vendors.databento.api_key,
            rate_limiter=rate_limiter,
        )
    elif vendor == "alpaca":
        client = AlpacaClient(
            api_key=settings.vendors.alpaca.api_key,
            secret_key=settings.vendors.alpaca.secret_key,
            rate_limiter=rate_limiter,
        )
    else:
        raise ValueError(f"Unknown vendor: {vendor}")

    updater = IncrementalUpdater(
        client=client,
        output_dir=f"{data_dir}/raw/{vendor}",
        lookback_days=lookback_days,
    )

    summary = updater.run_update(symbols=symbols, schemas=schemas)

    logger.info(
        "vendor_incremental_complete",
        vendor=vendor,
        symbols_updated=summary.symbols_updated,
        records_fetched=summary.records_fetched,
        new_records=summary.new_records,
    )
    return summary


@task
def check_market_calendar() -> bool:
    """Check if today is a market day.

    Returns:
        True if market is open today.
    """
    is_open = _is_market_open(date.today())
    logger.info("market_calendar_check", is_open=is_open, date=date.today().isoformat())
    return is_open


@flow(name="Incremental Data Update", log_prints=True)
def incremental_flow(
    symbols: list[str] | None = None,
    vendors: list[str] | None = None,
    schemas: list[str] | None = None,
    skip_market_check: bool = False,
) -> dict[str, IncrementalSummary]:
    """Run incremental data update for all vendors.

    Checks market calendar before running. Fetches data since the
    last watermark with a lookback window for late arrivals.

    Args:
        symbols: Symbols to update. Defaults to config.
        vendors: Vendors to use. Defaults to config.
        schemas: Schemas to fetch. Defaults to config.
        skip_market_check: Skip market calendar check.

    Returns:
        Dictionary mapping vendor to IncrementalSummary.
    """
    settings = get_settings()

    if not skip_market_check:
        is_open = check_market_calendar()
        if not is_open:
            logger.info("market_closed_skipping_update")
            return {}

    if symbols is None:
        symbols = settings.ingestion.incremental.symbols or ["AAPL", "MSFT", "GOOGL"]
    if vendors is None:
        vendors = ["databento", "alpaca"]
    if schemas is None:
        schemas = settings.ingestion.incremental.schemas or ["ohlcv-1d"]

    data_dir = settings.storage.local_path or "data"
    lookback_days = settings.ingestion.incremental.lookback_days or 3

    logger.info(
        "incremental_flow_started",
        symbols=len(symbols),
        vendors=vendors,
        schemas=schemas,
    )

    results: dict[str, IncrementalSummary] = {}
    for vendor in vendors:
        summary = incremental_update_vendor(
            vendor=vendor,
            symbols=symbols,
            schemas=schemas,
            data_dir=data_dir,
            lookback_days=lookback_days,
        )
        results[vendor] = summary

    return results


if __name__ == "__main__":
    incremental_flow()
