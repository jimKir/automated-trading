"""Incremental updater for daily data updates from last watermark."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import structlog

from market_data.ingestion.base import (
    AssetClass,
    BaseIngestionClient,
    IngestionRequest,
    IngestionResult,
    Schema,
)
from market_data.ingestion.checkpoint import CheckpointManager

logger = structlog.get_logger(__name__)


class IncrementalUpdater:
    """Incremental data updater from last watermark timestamp.

    Fetches data from the last known timestamp to now.
    Deduplicates by primary key. Handles late-arriving records
    by looking back beyond the watermark by a configurable window.

    Args:
        client: Data vendor client.
        checkpoint_manager: Checkpoint manager for watermark tracking.
        lookback_hours: Hours to look back beyond watermark for late arrivals.
        symbols_per_batch: Maximum symbols per batch request.
    """

    def __init__(
        self,
        client: BaseIngestionClient,
        checkpoint_manager: CheckpointManager,
        lookback_hours: int = 24,
        symbols_per_batch: int = 100,
    ) -> None:
        self.client = client
        self.checkpoint_mgr = checkpoint_manager
        self.lookback_hours = lookback_hours
        self.symbols_per_batch = symbols_per_batch
        self.logger = logger.bind(vendor=client.vendor_name)

    def update(
        self,
        symbols: list[str],
        schemas: list[Schema],
        asset_class: AssetClass = AssetClass.EQUITY,
    ) -> IncrementalSummary:
        """Run incremental update for the given symbols and schemas.

        Fetches data from last watermark to now for each symbol/schema pair.
        Applies lookback window for late-arriving records.

        Args:
            symbols: List of symbols to update.
            schemas: List of schemas to fetch.
            asset_class: Asset class.

        Returns:
            IncrementalSummary with results and statistics.
        """
        now = datetime.now(tz=timezone.utc)
        results: list[IngestionResult] = []
        skipped: list[str] = []
        failed: list[dict[str, Any]] = []
        dedup_count = 0

        self.logger.info(
            "incremental_update_started",
            symbols_count=len(symbols),
            schemas_count=len(schemas),
        )

        for schema in schemas:
            # Process in batches
            for i in range(0, len(symbols), self.symbols_per_batch):
                batch = symbols[i : i + self.symbols_per_batch]

                for symbol in batch:
                    last_ts = self.client.get_last_timestamp(symbol, schema)

                    if last_ts is None:
                        # No previous data — start from yesterday
                        start_date = (now - timedelta(days=1)).date()
                    else:
                        # Apply lookback window for late arrivals
                        lookback = last_ts - timedelta(hours=self.lookback_hours)
                        start_date = lookback.date()

                    end_date = now.date()

                    if start_date >= end_date:
                        skipped.append(symbol)
                        continue

                    try:
                        request = IngestionRequest(
                            symbols=[symbol],
                            schema=schema,
                            start_date=start_date,
                            end_date=end_date,
                            asset_class=asset_class,
                        )
                        for result in self.client.fetch_with_retry(request):
                            results.append(result)

                        self.logger.debug(
                            "symbol_updated",
                            symbol=symbol,
                            schema=schema.value,
                            records=results[-1].record_count if results else 0,
                        )
                    except Exception as e:
                        failed.append({
                            "symbol": symbol,
                            "schema": schema.value,
                            "error": str(e),
                        })
                        self.logger.error(
                            "symbol_update_failed",
                            symbol=symbol,
                            error=str(e),
                        )

        summary = IncrementalSummary(
            total_symbols=len(symbols) * len(schemas),
            updated_count=len(results),
            skipped_count=len(skipped),
            failed_count=len(failed),
            failed_details=failed,
            results=results,
            total_records=sum(r.record_count for r in results),
            deduplicated_records=dedup_count,
        )

        self.logger.info(
            "incremental_update_completed",
            updated=len(results),
            skipped=len(skipped),
            failed=len(failed),
            total_records=summary.total_records,
        )
        return summary

    @staticmethod
    def deduplicate_records(
        existing_keys: set[str],
        new_records: list[dict[str, Any]],
        key_fields: list[str],
    ) -> tuple[list[dict[str, Any]], int]:
        """Deduplicate new records against existing primary keys.

        Args:
            existing_keys: Set of existing primary keys.
            new_records: New records to deduplicate.
            key_fields: Fields that form the primary key.

        Returns:
            Tuple of (unique records, number of duplicates removed).
        """
        unique: list[dict[str, Any]] = []
        dup_count = 0
        for record in new_records:
            key = "|".join(str(record.get(f, "")) for f in key_fields)
            if key not in existing_keys:
                unique.append(record)
                existing_keys.add(key)
            else:
                dup_count += 1
        return unique, dup_count


class IncrementalSummary:
    """Summary of an incremental update operation."""

    def __init__(
        self,
        total_symbols: int,
        updated_count: int,
        skipped_count: int,
        failed_count: int,
        failed_details: list[dict[str, Any]],
        results: list[IngestionResult],
        total_records: int,
        deduplicated_records: int,
    ) -> None:
        self.total_symbols = total_symbols
        self.updated_count = updated_count
        self.skipped_count = skipped_count
        self.failed_count = failed_count
        self.failed_details = failed_details
        self.results = results
        self.total_records = total_records
        self.deduplicated_records = deduplicated_records

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_symbols": self.total_symbols,
            "updated": self.updated_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count,
            "failed_details": self.failed_details,
            "total_records": self.total_records,
            "deduplicated": self.deduplicated_records,
        }
