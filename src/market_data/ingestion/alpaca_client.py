"""Alpaca Historical API client for market data ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from market_data.ingestion.base import (
    AssetClass,
    BaseIngestionClient,
    IngestionError,
    IngestionRequest,
    IngestionResult,
    Schema,
    VendorAPIError,
)
from market_data.ingestion.rate_limiter import CostTracker, TokenBucketRateLimiter

if TYPE_CHECKING:
    from collections.abc import Iterator

    from market_data.ingestion.checkpoint import CheckpointManager, CheckpointState

logger = structlog.get_logger(__name__)

# Map our schemas to Alpaca timeframes
ALPACA_TIMEFRAME_MAP: dict[Schema, str] = {
    Schema.OHLCV_1M: "1Min",
    Schema.OHLCV_1H: "1Hour",
    Schema.OHLCV_1D: "1Day",
}


class AlpacaClient(BaseIngestionClient):
    """Alpaca Historical API client.

    Supports stock bars (minute, hour, day), crypto bars, and options bars.
    Handles pagination with page tokens and adjustment parameters.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        base_url: Alpaca data API base URL.
        rate_limit_per_min: API rate limit per minute.
        output_dir: Directory for raw data output.
        checkpoint_manager: Optional checkpoint manager.
    """

    VENDOR_NAME = "alpaca"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://data.alpaca.markets",
        rate_limit_per_min: int = 10000,
        output_dir: str | Path = "/data/raw/alpaca",
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        super().__init__(self.VENDOR_NAME)
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.rate_limiter = TokenBucketRateLimiter(
            rate_per_minute=rate_limit_per_min,
            vendor_name=self.VENDOR_NAME,
        )
        self.cost_tracker = CostTracker(vendor_name=self.VENDOR_NAME, monthly_budget=500.0)
        self.checkpoint_mgr = checkpoint_manager
        self._stock_client: Any = None
        self._crypto_client: Any = None

    def connect(self) -> None:
        """Initialize Alpaca API clients."""
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

            self._stock_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
            )
            self._crypto_client = CryptoHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
            )
            self.logger.info("connected_to_alpaca")
        except ImportError:
            self.logger.warning("alpaca_sdk_not_installed")
        except Exception as e:
            raise IngestionError(f"Failed to connect to Alpaca: {e}") from e

    def disconnect(self) -> None:
        """Close Alpaca client connections."""
        self._stock_client = None
        self._crypto_client = None
        self.logger.info("disconnected_from_alpaca")

    def fetch_historical(
        self,
        request: IngestionRequest,
    ) -> Iterator[IngestionResult]:
        """Fetch historical bar data from Alpaca.

        Handles pagination automatically using page tokens.
        Supports adjustment parameter (raw, split, dividend, all).

        Args:
            request: Ingestion request parameters.

        Yields:
            IngestionResult for each symbol processed.
        """
        if self._stock_client is None:
            raise IngestionError("Alpaca client not connected. Call connect() first.")

        timeframe_str = ALPACA_TIMEFRAME_MAP.get(request.schema)
        if timeframe_str is None:
            raise IngestionError(
                f"Unsupported schema for Alpaca bars: {request.schema}. "
                f"Supported: {list(ALPACA_TIMEFRAME_MAP.keys())}"
            )

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

            try:
                result = self._fetch_bars(
                    symbol=symbol,
                    asset_class=request.asset_class,
                    timeframe_str=timeframe_str,
                    start_date=start,
                    end_date=request.end_date,
                    adjustment=request.adjustment,
                )

                if checkpoint and self.checkpoint_mgr:
                    self.checkpoint_mgr.mark_completed(checkpoint)

                yield result
            except Exception as e:
                self.logger.error("fetch_error", symbol=symbol, error=str(e))
                if checkpoint and self.checkpoint_mgr:
                    self.checkpoint_mgr.mark_failed(checkpoint)
                raise VendorAPIError(f"Alpaca fetch failed for {symbol}: {e}") from e

    def _fetch_bars(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframe_str: str,
        start_date: date,
        end_date: date,
        adjustment: str = "raw",
    ) -> IngestionResult:
        """Fetch bars for a single symbol with pagination."""
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        start_time = time.monotonic()

        timeframe_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = timeframe_map[timeframe_str]

        all_bars: list[dict[str, Any]] = []

        if asset_class == AssetClass.CRYPTO:
            req = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=datetime.combine(start_date, datetime.min.time()),
                end=datetime.combine(end_date, datetime.max.time()),
            )
            self.rate_limiter.acquire()
            bars_response = self._crypto_client.get_crypto_bars(req)
        else:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=datetime.combine(start_date, datetime.min.time()),
                end=datetime.combine(end_date, datetime.max.time()),
                adjustment=adjustment,
            )
            self.rate_limiter.acquire()
            bars_response = self._stock_client.get_stock_bars(req)

        if hasattr(bars_response, "data") and symbol in bars_response.data:
            for bar in bars_response.data[symbol]:
                all_bars.append(
                    {
                        "timestamp": bar.timestamp.isoformat(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                        "vwap": float(bar.vwap) if hasattr(bar, "vwap") and bar.vwap else None,
                        "trade_count": int(bar.trade_count)
                        if hasattr(bar, "trade_count") and bar.trade_count
                        else None,
                    }
                )
        elif isinstance(bars_response, dict) and symbol in bars_response:
            for bar in bars_response[symbol]:
                all_bars.append(
                    {
                        "timestamp": bar.timestamp.isoformat(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    }
                )

        # Write to output
        output_path = self._build_output_path(
            symbol=symbol,
            asset_class=asset_class,
            timeframe_str=timeframe_str,
            start_date=start_date,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_data = json.dumps(all_bars, indent=2)
        output_path.write_text(raw_data)

        checksum = hashlib.sha256(raw_data.encode()).hexdigest()
        duration = time.monotonic() - start_time

        return IngestionResult(
            vendor=self.VENDOR_NAME,
            symbol=symbol,
            schema=timeframe_str,
            start_date=start_date,
            end_date=end_date,
            record_count=len(all_bars),
            bytes_downloaded=len(raw_data.encode()),
            output_path=str(output_path),
            checksum=checksum,
            duration_seconds=round(duration, 2),
            metadata={"adjustment": adjustment, "asset_class": asset_class.value},
        )

    def _build_output_path(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframe_str: str,
        start_date: date,
    ) -> Path:
        """Build the raw data lake output path."""
        return (
            self.output_dir
            / asset_class.value
            / timeframe_str
            / str(start_date.year)
            / f"{start_date.month:02d}"
            / f"{start_date.day:02d}"
            / f"{symbol}.json"
        )

    def _load_checkpoint(self, request: IngestionRequest, symbol: str) -> CheckpointState | None:
        """Load checkpoint for a symbol."""
        if self.checkpoint_mgr is None:
            return None
        return self.checkpoint_mgr.load(
            vendor=self.VENDOR_NAME,
            symbol=symbol,
            schema=request.schema.value,
            start_date=str(request.start_date),
            end_date=str(request.end_date),
        )

    def get_available_symbols(self, asset_class: AssetClass) -> list[str]:
        """Get available symbols from Alpaca."""
        if self._stock_client is None:
            return []
        try:
            from alpaca.trading.client import TradingClient

            trading = TradingClient(api_key=self.api_key, secret_key=self.api_secret)
            assets = trading.get_all_assets()
            return [a.symbol for a in assets if a.tradable and a.status == "active"]
        except Exception as e:
            self.logger.error("get_symbols_error", error=str(e))
            return []

    def get_last_timestamp(self, symbol: str, schema: Schema) -> datetime | None:
        """Get the last ingested timestamp for a symbol."""
        if self.checkpoint_mgr is None:
            return None
        incomplete = self.checkpoint_mgr.get_incomplete(vendor=self.VENDOR_NAME)
        for cp in incomplete:
            if cp.symbol == symbol and cp.schema == schema.value and cp.last_processed_date:
                return datetime.fromisoformat(cp.last_processed_date)
        return None
