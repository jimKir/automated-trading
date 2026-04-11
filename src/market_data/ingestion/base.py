"""Abstract base class for data vendor ingestion clients."""

from __future__ import annotations

import abc
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

import structlog
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)


class AssetClass(str, Enum):
    """Supported asset classes."""

    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    CRYPTO = "crypto"


class Schema(str, Enum):
    """Supported data schemas."""

    TRADES = "trades"
    QUOTES = "quotes"
    OHLCV_1M = "ohlcv-1m"
    OHLCV_1H = "ohlcv-1h"
    OHLCV_1D = "ohlcv-1d"
    MBP_1 = "mbp-1"
    MBP_10 = "mbp-10"
    MBO = "mbo"
    IMBALANCE = "imbalance"
    STATISTICS = "statistics"


@dataclass
class IngestionResult:
    """Result of a data ingestion operation."""

    vendor: str
    symbol: str
    schema: str
    start_date: date
    end_date: date
    record_count: int
    bytes_downloaded: int
    output_path: str
    checksum: str
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionRequest:
    """Request for data ingestion."""

    symbols: list[str]
    schema: Schema
    start_date: date
    end_date: date
    asset_class: AssetClass = AssetClass.EQUITY
    adjustment: str = "raw"
    metadata: dict[str, Any] = field(default_factory=dict)


class IngestionError(Exception):
    """Base exception for ingestion errors."""


class RateLimitError(IngestionError):
    """Rate limit exceeded."""


class VendorAPIError(IngestionError):
    """Vendor API returned an error."""


class BaseIngestionClient(abc.ABC):
    """Abstract base class for market data vendor clients.

    Provides retry logic with exponential backoff (max 5 retries),
    checkpointing interface, and standard logging.
    """

    MAX_RETRIES = 5
    BASE_WAIT = 1.0
    MAX_WAIT = 60.0

    def __init__(self, vendor_name: str) -> None:
        self.vendor_name = vendor_name
        self.logger = structlog.get_logger(__name__).bind(vendor=vendor_name)

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish connection to vendor API."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close connection to vendor API."""

    @abc.abstractmethod
    def fetch_historical(
        self,
        request: IngestionRequest,
    ) -> Iterator[IngestionResult]:
        """Fetch historical data for the given request.

        Args:
            request: Ingestion request parameters.

        Yields:
            IngestionResult for each completed chunk.
        """

    @abc.abstractmethod
    def get_available_symbols(self, asset_class: AssetClass) -> list[str]:
        """Get list of available symbols for an asset class."""

    @abc.abstractmethod
    def get_last_timestamp(self, symbol: str, schema: Schema) -> datetime | None:
        """Get the last ingested timestamp for a symbol/schema pair."""

    def fetch_with_retry(
        self,
        request: IngestionRequest,
    ) -> Iterator[IngestionResult]:
        """Fetch data with retry logic using exponential backoff.

        Wraps fetch_historical with tenacity retry decorator.
        Max 5 retries with exponential backoff from 1s to 60s.
        """
        start_time = time.monotonic()
        self.logger.info(
            "starting_fetch",
            symbols=request.symbols,
            schema=request.schema.value,
            start_date=str(request.start_date),
            end_date=str(request.end_date),
        )

        @retry(
            stop=stop_after_attempt(self.MAX_RETRIES),
            wait=wait_exponential(multiplier=self.BASE_WAIT, max=self.MAX_WAIT),
            retry=retry_if_exception_type((RateLimitError, VendorAPIError)),
            reraise=True,
        )
        def _fetch() -> list[IngestionResult]:
            return list(self.fetch_historical(request))

        try:
            results = _fetch()
            duration = time.monotonic() - start_time
            total_records = sum(r.record_count for r in results)
            self.logger.info(
                "fetch_complete",
                record_count=total_records,
                duration_seconds=round(duration, 2),
            )
            yield from results
        except RetryError:
            self.logger.error(
                "fetch_failed_after_retries",
                max_retries=self.MAX_RETRIES,
                symbols=request.symbols,
            )
            raise IngestionError(f"Failed to fetch data after {self.MAX_RETRIES} retries")

    def __enter__(self) -> BaseIngestionClient:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
