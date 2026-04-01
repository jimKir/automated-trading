"""Prefect flow for historical data backfill."""

from __future__ import annotations

from datetime import date, timedelta

import structlog
from prefect import flow, task
from prefect.tasks import task_input_hash

from market_data.config import get_settings
from market_data.ingestion.backfill import BackfillOrchestrator, BackfillSummary
from market_data.ingestion.checkpoint import CheckpointManager
from market_data.ingestion.databento_client import DatabentoClient
from market_data.ingestion.alpaca_client import AlpacaClient
from market_data.ingestion.rate_limiter import TokenBucketRateLimiter
from market_data.monitoring.alerts import Alert, AlertManager, AlertSeverity

logger = structlog.get_logger(__name__)


@task(retries=2, retry_delay_seconds=60)
def backfill_vendor(
    vendor: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    schemas: list[str],
    data_dir: str,
) -> BackfillSummary:
    """Run backfill for a single vendor.

    Args:
        vendor: Vendor name (databento/alpaca).
        symbols: List of symbols to backfill.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        schemas: List of schema names.
        data_dir: Base data directory.

    Returns:
        Backfill summary with statistics.
    """
    settings = get_settings()
    checkpoint_mgr = CheckpointManager(db_path=f"{data_dir}/checkpoints.db")
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

    orchestrator = BackfillOrchestrator(
        client=client,
        checkpoint_manager=checkpoint_mgr,
        output_dir=f"{data_dir}/raw/{vendor}",
        chunk_days=30,
    )

    summary = orchestrator.run_backfill(
        symbols=symbols,
        schemas=schemas,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(
        "vendor_backfill_complete",
        vendor=vendor,
        total_chunks=summary.total_chunks,
        completed=summary.completed_chunks,
        failed=summary.failed_chunks,
    )
    return summary


@task
def send_backfill_alert(
    vendor: str,
    summary: BackfillSummary,
    alert_manager: AlertManager | None = None,
) -> None:
    """Send alert on backfill completion or failure.

    Args:
        vendor: Vendor name.
        summary: Backfill summary.
        alert_manager: Alert manager instance.
    """
    if alert_manager is None:
        return

    if summary.failed_chunks > 0:
        alert_manager.alert_warning(
            title=f"Backfill Partial Failure: {vendor}",
            message=(
                f"Backfill completed with {summary.failed_chunks} failed chunks "
                f"out of {summary.total_chunks}. "
                f"Success rate: {summary.success_rate:.1%}"
            ),
            vendor=vendor,
            total=summary.total_chunks,
            failed=summary.failed_chunks,
        )
    else:
        alert_manager.alert_info(
            title=f"Backfill Complete: {vendor}",
            message=f"Successfully backfilled {summary.completed_chunks} chunks.",
            vendor=vendor,
            total=summary.total_chunks,
        )


@flow(name="Historical Data Backfill", log_prints=True)
def backfill_flow(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    vendors: list[str] | None = None,
    schemas: list[str] | None = None,
) -> dict[str, BackfillSummary]:
    """Run historical data backfill across vendors.

    Loads configuration, partitions work by vendor, and runs
    backfill with checkpointing for crash recovery.

    Args:
        symbols: Symbols to backfill. Defaults to config.
        start_date: Start date. Defaults to 2 years ago.
        end_date: End date. Defaults to yesterday.
        vendors: Vendors to use. Defaults to config.
        schemas: Schemas to fetch. Defaults to config.

    Returns:
        Dictionary mapping vendor name to BackfillSummary.
    """
    settings = get_settings()

    if symbols is None:
        symbols = settings.ingestion.backfill.symbols or ["AAPL", "MSFT", "GOOGL"]
    if start_date is None:
        start_date = settings.ingestion.backfill.start_date or (
            date.today() - timedelta(days=730)
        ).isoformat()
    if end_date is None:
        end_date = settings.ingestion.backfill.end_date or (
            date.today() - timedelta(days=1)
        ).isoformat()
    if vendors is None:
        vendors = ["databento", "alpaca"]
    if schemas is None:
        schemas = settings.ingestion.backfill.schemas or ["ohlcv-1d"]

    data_dir = settings.storage.local_path or "data"

    logger.info(
        "backfill_flow_started",
        symbols=len(symbols),
        start_date=start_date,
        end_date=end_date,
        vendors=vendors,
    )

    results: dict[str, BackfillSummary] = {}
    for vendor in vendors:
        summary = backfill_vendor(
            vendor=vendor,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            schemas=schemas,
            data_dir=data_dir,
        )
        results[vendor] = summary
        send_backfill_alert(vendor=vendor, summary=summary)

    return results


if __name__ == "__main__":
    backfill_flow()
