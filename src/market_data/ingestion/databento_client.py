"""Databento Historical API client for market data ingestion."""

from __future__ import annotations

import hashlib
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import structlog

from market_data.ingestion.base import (
    AssetClass,
    BaseIngestionClient,
    IngestionError,
    IngestionRequest,
    IngestionResult,
    RateLimitError,
    Schema,
    VendorAPIError,
)
from market_data.ingestion.checkpoint import CheckpointManager, CheckpointState
from market_data.ingestion.rate_limiter import CostTracker, TokenBucketRateLimiter

logger = structlog.get_logger(__name__)

# Mapping from our schema enum to Databento schema strings
SCHEMA_MAP: dict[Schema, str] = {
    Schema.TRADES: "trades",
    Schema.QUOTES: "tbbo",
    Schema.OHLCV_1M: "ohlcv-1m",
    Schema.OHLCV_1H: "ohlcv-1h",
    Schema.OHLCV_1D: "ohlcv-1d",
    Schema.MBP_1: "mbp-1",
    Schema.MBP_10: "mbp-10",
    Schema.MBO: "mbo",
    Schema.IMBALANCE: "imbalance",
    Schema.STATISTICS: "statistics",
}

DATASET_MAP: dict[AssetClass, str] = {
    AssetClass.EQUITY: "XNAS.ITCH",
    AssetClass.OPTION: "OPRA.PILLAR",
    AssetClass.FUTURE: "GLBX.MDP3",
}


class DatabentoClient(BaseIngestionClient):
    """Databento Historical API client.

    Supports trades, quotes, OHLCV bars, market depth schemas.
    Auto-detects batch mode for requests >5GB.
    Stores raw DBN format.

    Args:
        api_key: Databento API key.
        dataset: Default dataset (e.g., GLBX.MDP3).
        rate_limit_per_min: API rate limit per minute.
        batch_threshold_gb: Size threshold for batch mode.
        output_dir: Directory for raw data output.
        checkpoint_manager: Optional checkpoint manager for resumability.
    """

    VENDOR_NAME = "databento"

    def __init__(
        self,
        api_key: str | None = None,
        dataset: str = "GLBX.MDP3",
        rate_limit_per_min: int = 1000,
        batch_threshold_gb: int = 5,
        output_dir: str | Path = "/data/raw/databento",
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        super().__init__(self.VENDOR_NAME)
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY", "")
        self.dataset = dataset
        self.batch_threshold_gb = batch_threshold_gb
        self.output_dir = Path(output_dir)
        self.rate_limiter = TokenBucketRateLimiter(
            rate_per_minute=rate_limit_per_min,
            vendor_name=self.VENDOR_NAME,
        )
        self.cost_tracker = CostTracker(vendor_name=self.VENDOR_NAME)
        self.checkpoint_mgr = checkpoint_manager
        self._client: Any = None

    def connect(self) -> None:
        """Initialize the Databento historical client."""
        try:
            import databento as db

            self._client = db.Historical(key=self.api_key)
            self.logger.info("connected_to_databento")
        except ImportError:
            self.logger.warning("databento_sdk_not_installed")
            self._client = None
        except Exception as e:
            raise IngestionError(f"Failed to connect to Databento: {e}") from e

    def disconnect(self) -> None:
        """Close Databento client connection."""
        self._client = None
        self.logger.info("disconnected_from_databento")

    def fetch_historical(
        self,
        request: IngestionRequest,
    ) -> Iterator[IngestionResult]:
        """Fetch historical data from Databento.

        Auto-detects batch mode for large requests (>5GB).
        Stores raw DBN format with metadata sidecar.

        Args:
            request: Ingestion request parameters.

        Yields:
            IngestionResult for each symbol processed.
        """
        if self._client is None:
            raise IngestionError("Databento client not connected. Call connect() first.")

        schema_str = SCHEMA_MAP.get(request.schema)
        if schema_str is None:
            raise IngestionError(f"Unsupported schema for Databento: {request.schema}")

        dataset = DATASET_MAP.get(request.asset_class, self.dataset)

        for symbol in request.symbols:
            checkpoint = self._load_checkpoint(request, symbol)
            start = (
                date.fromisoformat(checkpoint.last_processed_date) + timedelta(days=1)
                if checkpoint and checkpoint.last_processed_date
                else request.start_date
            )

            if start > request.end_date:
                self.logger.info("symbol_already_complete", symbol=symbol)
                continue

            self.rate_limiter.acquire()
            start_time = time.monotonic()

            try:
                result = self._fetch_symbol(
                    symbol=symbol,
                    schema_str=schema_str,
                    dataset=dataset,
                    asset_class=request.asset_class,
                    start_date=start,
                    end_date=request.end_date,
                )

                if checkpoint:
                    self.checkpoint_mgr.mark_completed(checkpoint) if self.checkpoint_mgr else None

                yield result
            except Exception as e:
                self.logger.error("fetch_error", symbol=symbol, error=str(e))
                if checkpoint and self.checkpoint_mgr:
                    self.checkpoint_mgr.mark_failed(checkpoint)
                raise VendorAPIError(f"Databento fetch failed for {symbol}: {e}") from e

    def _fetch_symbol(
        self,
        symbol: str,
        schema_str: str,
        dataset: str,
        asset_class: AssetClass,
        start_date: date,
        end_date: date,
    ) -> IngestionResult:
        """Fetch data for a single symbol.

        Uses cost estimation to decide between streaming and batch mode.
        """
        start_time = time.monotonic()

        # Estimate cost and size
        cost_estimate = self._estimate_cost(
            symbol=symbol,
            schema_str=schema_str,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
        )

        use_batch = cost_estimate.get("size_gb", 0) > self.batch_threshold_gb
        self.logger.info(
            "fetch_mode_selected",
            symbol=symbol,
            mode="batch" if use_batch else "streaming",
            estimated_size_gb=cost_estimate.get("size_gb", 0),
        )

        # Prepare output path
        output_path = self._build_output_path(
            symbol=symbol,
            schema_str=schema_str,
            asset_class=asset_class,
            start_date=start_date,
            end_date=end_date,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if use_batch:
            result_data = self._fetch_batch(
                symbol=symbol,
                schema_str=schema_str,
                dataset=dataset,
                start_date=start_date,
                end_date=end_date,
                output_path=output_path,
            )
        else:
            result_data = self._fetch_streaming(
                symbol=symbol,
                schema_str=schema_str,
                dataset=dataset,
                start_date=start_date,
                end_date=end_date,
                output_path=output_path,
            )

        # Record cost
        actual_cost = cost_estimate.get("cost_usd", 0.0)
        self.cost_tracker.record_cost(actual_cost, f"{symbol}/{schema_str}")

        duration = time.monotonic() - start_time
        return IngestionResult(
            vendor=self.VENDOR_NAME,
            symbol=symbol,
            schema=schema_str,
            start_date=start_date,
            end_date=end_date,
            record_count=result_data.get("record_count", 0),
            bytes_downloaded=result_data.get("bytes", 0),
            output_path=str(output_path),
            checksum=result_data.get("checksum", ""),
            duration_seconds=round(duration, 2),
            metadata={"dataset": dataset, "batch_mode": use_batch},
        )

    def _fetch_streaming(
        self,
        symbol: str,
        schema_str: str,
        dataset: str,
        start_date: date,
        end_date: date,
        output_path: Path,
    ) -> dict[str, Any]:
        """Fetch data via streaming API and write to file."""
        data = self._client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema=schema_str,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )
        file_path = str(output_path)
        data.to_file(file_path)

        file_size = output_path.stat().st_size if output_path.exists() else 0
        checksum = self._compute_checksum(output_path) if output_path.exists() else ""
        record_count = len(data) if hasattr(data, "__len__") else 0

        return {
            "record_count": record_count,
            "bytes": file_size,
            "checksum": checksum,
        }

    def _fetch_batch(
        self,
        symbol: str,
        schema_str: str,
        dataset: str,
        start_date: date,
        end_date: date,
        output_path: Path,
    ) -> dict[str, Any]:
        """Submit and download a batch job for large requests."""
        job = self._client.batch.submit_job(
            dataset=dataset,
            symbols=[symbol],
            schema=schema_str,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            encoding="dbn",
        )

        self.logger.info("batch_job_submitted", job_id=job.id, symbol=symbol)

        # Poll for completion
        while True:
            status = self._client.batch.get_job(job.id)
            if status.state == "done":
                break
            if status.state == "failed":
                raise VendorAPIError(f"Batch job failed: {job.id}")
            time.sleep(10)

        # Download result
        files = self._client.batch.download(job.id, output_dir=str(output_path.parent))
        total_bytes = sum(Path(f).stat().st_size for f in files if Path(f).exists())
        checksum = self._compute_checksum(output_path) if output_path.exists() else ""

        return {
            "record_count": 0,  # Not easily available from batch
            "bytes": total_bytes,
            "checksum": checksum,
        }

    def _estimate_cost(
        self,
        symbol: str,
        schema_str: str,
        dataset: str,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Estimate the cost and size of a request."""
        try:
            cost = self._client.metadata.get_cost(
                dataset=dataset,
                symbols=[symbol],
                schema=schema_str,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )
            return {"cost_usd": cost, "size_gb": 0}
        except Exception:
            # Rough estimate: 1GB per year for minute bars
            days = (end_date - start_date).days
            estimated_gb = (days / 365) * 0.5
            return {"cost_usd": 0.0, "size_gb": estimated_gb}

    def _build_output_path(
        self,
        symbol: str,
        schema_str: str,
        asset_class: AssetClass,
        start_date: date,
        end_date: date,
    ) -> Path:
        """Build the raw data lake output path."""
        return (
            self.output_dir
            / asset_class.value
            / schema_str
            / str(start_date.year)
            / f"{start_date.month:02d}"
            / f"{start_date.day:02d}"
            / f"{symbol}.dbn"
        )

    def _load_checkpoint(
        self, request: IngestionRequest, symbol: str
    ) -> CheckpointState | None:
        """Load checkpoint for a symbol if checkpoint manager exists."""
        if self.checkpoint_mgr is None:
            return None
        return self.checkpoint_mgr.load(
            vendor=self.VENDOR_NAME,
            symbol=symbol,
            schema=request.schema.value,
            start_date=str(request.start_date),
            end_date=str(request.end_date),
        )

    @staticmethod
    def _compute_checksum(file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_available_symbols(self, asset_class: AssetClass) -> list[str]:
        """Get available symbols from Databento symbology."""
        if self._client is None:
            return []
        dataset = DATASET_MAP.get(asset_class, self.dataset)
        try:
            result = self._client.symbology.resolve(
                dataset=dataset,
                symbols=["ALL_SYMBOLS"],
                stype_in="raw_symbol",
                stype_out="instrument_id",
                start_date=date.today().isoformat(),
            )
            return list(result.keys()) if isinstance(result, dict) else []
        except Exception as e:
            self.logger.error("symbology_resolve_error", error=str(e))
            return []

    def get_last_timestamp(self, symbol: str, schema: Schema) -> datetime | None:
        """Get the last ingested timestamp for a symbol/schema pair."""
        if self.checkpoint_mgr is None:
            return None
        incomplete = self.checkpoint_mgr.get_incomplete(vendor=self.VENDOR_NAME)
        for cp in incomplete:
            if cp.symbol == symbol and cp.schema == schema.value:
                if cp.last_processed_date:
                    return datetime.fromisoformat(cp.last_processed_date)
        return None
