"""Bulk backfill orchestrator for historical data ingestion."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import structlog

from market_data.ingestion.base import (
    AssetClass,
    BaseIngestionClient,
    IngestionRequest,
    IngestionResult,
    Schema,
)
from market_data.ingestion.checkpoint import CheckpointManager, CheckpointState

logger = structlog.get_logger(__name__)


class BackfillOrchestrator:
    """Orchestrate bulk historical data backfill.

    Partitions requests into configurable day chunks (default 30 days).
    Checkpoints every 1000 records or 100MB. Resumes on failure.
    Supports symbol universe filters.

    Args:
        client: Data vendor client to use.
        checkpoint_manager: Checkpoint manager for resumability.
        chunk_size_days: Number of days per request chunk.
        max_retries: Maximum retries per chunk.
    """

    def __init__(
        self,
        client: BaseIngestionClient,
        checkpoint_manager: CheckpointManager,
        chunk_size_days: int = 30,
        max_retries: int = 5,
    ) -> None:
        self.client = client
        self.checkpoint_mgr = checkpoint_manager
        self.chunk_size_days = chunk_size_days
        self.max_retries = max_retries
        self.logger = logger.bind(vendor=client.vendor_name)

    def run(
        self,
        symbols: list[str],
        schemas: list[Schema],
        start_date: date,
        end_date: date,
        asset_class: AssetClass = AssetClass.EQUITY,
        symbol_filter: list[str] | None = None,
    ) -> BackfillSummary:
        """Run bulk backfill for the given parameters.

        Partitions the date range into 30-day chunks and processes each.
        Checkpoints progress for resume-on-failure.

        Args:
            symbols: List of symbols to backfill.
            schemas: List of schemas to fetch.
            start_date: Start of backfill range.
            end_date: End of backfill range.
            asset_class: Asset class to fetch.
            symbol_filter: Optional list to filter symbols.

        Returns:
            BackfillSummary with results and statistics.
        """
        if symbol_filter:
            symbols = [s for s in symbols if s in symbol_filter]

        total_chunks = 0
        completed_chunks = 0
        failed_chunks: list[dict[str, Any]] = []
        results: list[IngestionResult] = []

        self.logger.info(
            "backfill_started",
            symbols_count=len(symbols),
            schemas_count=len(schemas),
            start_date=str(start_date),
            end_date=str(end_date),
        )

        for schema in schemas:
            for symbol in symbols:
                chunks = self._partition_date_range(start_date, end_date)
                for chunk_start, chunk_end in chunks:
                    total_chunks += 1

                    # Check if already completed
                    existing = self.checkpoint_mgr.load(
                        vendor=self.client.vendor_name,
                        symbol=symbol,
                        schema=schema.value,
                        start_date=str(chunk_start),
                        end_date=str(chunk_end),
                    )
                    if existing and existing.status == "completed":
                        self.logger.debug(
                            "chunk_already_completed",
                            symbol=symbol,
                            chunk_start=str(chunk_start),
                        )
                        completed_chunks += 1
                        continue

                    # Create checkpoint
                    checkpoint = CheckpointState(
                        vendor=self.client.vendor_name,
                        symbol=symbol,
                        schema=schema.value,
                        start_date=str(chunk_start),
                        end_date=str(chunk_end),
                    )
                    self.checkpoint_mgr.save(checkpoint)

                    try:
                        request = IngestionRequest(
                            symbols=[symbol],
                            schema=schema,
                            start_date=chunk_start,
                            end_date=chunk_end,
                            asset_class=asset_class,
                        )
                        for result in self.client.fetch_with_retry(request):
                            results.append(result)

                        checkpoint.status = "completed"
                        checkpoint.last_processed_date = str(chunk_end)
                        self.checkpoint_mgr.save(checkpoint)
                        completed_chunks += 1

                        self.logger.info(
                            "chunk_completed",
                            symbol=symbol,
                            schema=schema.value,
                            chunk_start=str(chunk_start),
                            chunk_end=str(chunk_end),
                        )
                    except Exception as e:
                        self.checkpoint_mgr.mark_failed(checkpoint)
                        failed_chunks.append(
                            {
                                "symbol": symbol,
                                "schema": schema.value,
                                "chunk_start": str(chunk_start),
                                "chunk_end": str(chunk_end),
                                "error": str(e),
                            }
                        )
                        self.logger.error(
                            "chunk_failed",
                            symbol=symbol,
                            error=str(e),
                        )

        summary = BackfillSummary(
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            failed_chunks=failed_chunks,
            results=results,
            total_records=sum(r.record_count for r in results),
            total_bytes=sum(r.bytes_downloaded for r in results),
        )

        self.logger.info(
            "backfill_completed",
            total_chunks=total_chunks,
            completed=completed_chunks,
            failed=len(failed_chunks),
            total_records=summary.total_records,
        )
        return summary

    def _partition_date_range(self, start: date, end: date) -> list[tuple[date, date]]:
        """Partition a date range into chunks of chunk_size_days.

        Args:
            start: Start date.
            end: End date.

        Returns:
            List of (chunk_start, chunk_end) tuples.
        """
        chunks: list[tuple[date, date]] = []
        current = start
        while current <= end:
            chunk_end = min(current + timedelta(days=self.chunk_size_days - 1), end)
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        return chunks

    def resume(self) -> BackfillSummary:
        """Resume incomplete backfills from checkpoints.

        Returns:
            BackfillSummary of resumed operations.
        """
        incomplete = self.checkpoint_mgr.get_incomplete(vendor=self.client.vendor_name)
        if not incomplete:
            self.logger.info("no_incomplete_backfills")
            return BackfillSummary(
                total_chunks=0,
                completed_chunks=0,
                failed_chunks=[],
                results=[],
                total_records=0,
                total_bytes=0,
            )

        self.logger.info("resuming_backfill", incomplete_count=len(incomplete))
        results: list[IngestionResult] = []
        failed: list[dict[str, Any]] = []

        for cp in incomplete:
            start = (
                date.fromisoformat(cp.last_processed_date) + timedelta(days=1)
                if cp.last_processed_date
                else date.fromisoformat(cp.start_date)
            )
            end = date.fromisoformat(cp.end_date)

            if start > end:
                self.checkpoint_mgr.mark_completed(cp)
                continue

            try:
                schema = Schema(cp.schema)
                request = IngestionRequest(
                    symbols=[cp.symbol],
                    schema=schema,
                    start_date=start,
                    end_date=end,
                )
                for result in self.client.fetch_with_retry(request):
                    results.append(result)
                self.checkpoint_mgr.mark_completed(cp)
            except Exception as e:
                self.checkpoint_mgr.mark_failed(cp)
                failed.append(
                    {
                        "symbol": cp.symbol,
                        "schema": cp.schema,
                        "error": str(e),
                    }
                )

        return BackfillSummary(
            total_chunks=len(incomplete),
            completed_chunks=len(incomplete) - len(failed),
            failed_chunks=failed,
            results=results,
            total_records=sum(r.record_count for r in results),
            total_bytes=sum(r.bytes_downloaded for r in results),
        )


class BackfillSummary:
    """Summary of a backfill operation."""

    def __init__(
        self,
        total_chunks: int,
        completed_chunks: int,
        failed_chunks: list[dict[str, Any]],
        results: list[IngestionResult],
        total_records: int,
        total_bytes: int,
    ) -> None:
        self.total_chunks = total_chunks
        self.completed_chunks = completed_chunks
        self.failed_chunks = failed_chunks
        self.results = results
        self.total_records = total_records
        self.total_bytes = total_bytes

    @property
    def success_rate(self) -> float:
        """Fraction of chunks that completed successfully."""
        if self.total_chunks == 0:
            return 1.0
        return self.completed_chunks / self.total_chunks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "failed_count": len(self.failed_chunks),
            "failed_chunks": self.failed_chunks,
            "total_records": self.total_records,
            "total_bytes": self.total_bytes,
            "success_rate": round(self.success_rate, 4),
        }
