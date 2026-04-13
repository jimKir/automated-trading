"""Backtest data serving API with caching and as-of-date queries."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pandas as pd
import structlog

if TYPE_CHECKING:
    from market_data.storage.analytics_lake import AnalyticsLake
    from market_data.storage.symbol_master import SymbolMaster
    from market_data.transforms.corporate_actions import CorporateActionsManager

logger = structlog.get_logger(__name__)


class LRUCache:
    """Simple LRU cache for DataFrames.

    Args:
        max_size: Maximum number of entries.
        max_memory_mb: Maximum total memory in MB (approximate).
    """

    def __init__(self, max_size: int = 128, max_memory_mb: int = 512) -> None:
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._max_size = max_size
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_bytes = 0

    def get(self, key: str) -> pd.DataFrame | None:
        """Get a cached DataFrame, moving it to most-recently-used."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Cache a DataFrame, evicting LRU entries as needed."""
        df_bytes = df.memory_usage(deep=True).sum()

        if key in self._cache:
            old_bytes = self._cache[key].memory_usage(deep=True).sum()
            self._current_bytes -= old_bytes
            del self._cache[key]

        while self._cache and (
            len(self._cache) >= self._max_size
            or self._current_bytes + df_bytes > self._max_memory_bytes
        ):
            _, evicted = self._cache.popitem(last=False)
            self._current_bytes -= evicted.memory_usage(deep=True).sum()

        self._cache[key] = df
        self._current_bytes += df_bytes

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._current_bytes = 0

    @property
    def size(self) -> int:
        """Number of entries in cache."""
        return len(self._cache)


class BacktestAPI:
    """Backtest data serving API.

    Provides get_bars() returning pandas DataFrames with support for
    symbol filtering, date ranges, bar intervals, adjustment modes,
    and as-of-date queries. Caches frequently accessed datasets.

    Args:
        analytics_lake: Analytics lake for reading normalized data.
        symbol_master: Symbol master for ticker resolution.
        corporate_actions: Corporate actions manager for adjustments.
        cache_size: Max number of cached DataFrames.
        cache_memory_mb: Max cache memory in MB.
    """

    def __init__(
        self,
        analytics_lake: AnalyticsLake,
        symbol_master: SymbolMaster,
        corporate_actions: CorporateActionsManager | None = None,
        cache_size: int = 128,
        cache_memory_mb: int = 512,
    ) -> None:
        self.lake = analytics_lake
        self.symbol_master = symbol_master
        self.corporate_actions = corporate_actions
        self._cache = LRUCache(max_size=cache_size, max_memory_mb=cache_memory_mb)

    def get_bars(
        self,
        symbols: str | list[str],
        start_date: str | date,
        end_date: str | date,
        interval: str = "1d",
        adjustment: str = "split",
        as_of_date: str | date | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV bars for backtesting.

        Args:
            symbols: Ticker or list of tickers.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            interval: Bar interval (1m, 1h, 1d).
            adjustment: Adjustment mode (raw, split, dividend, all).
            as_of_date: Point-in-time date for as-of queries.
            columns: Column subset to return.

        Returns:
            DataFrame with OHLCV data indexed by (timestamp, symbol).
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()
        if isinstance(as_of_date, date):
            as_of_date = as_of_date.isoformat()

        schema_name = f"ohlcv-{interval}"
        cache_key = self._make_cache_key(
            symbols, start_date, end_date, schema_name, adjustment, as_of_date
        )

        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("cache_hit", key=cache_key[:16])
            if columns:
                available = [c for c in columns if c in cached.columns]
                return cached[available]
            return cached

        frames: list[pd.DataFrame] = []
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        for ticker in symbols:
            symbol_id = self.symbol_master.get_symbol_id(ticker)
            if symbol_id is None:
                logger.warning("symbol_not_found", ticker=ticker)
                continue

            table = self.lake.read_date_range(
                asset_class="equity",
                schema_name=schema_name,
                symbol_id=symbol_id,
                start_year=start_dt.year,
                start_month=start_dt.month,
                end_year=end_dt.year,
                end_month=end_dt.month,
            )

            if table is None or table.num_rows == 0:
                continue

            df = table.to_pandas()
            df["ticker"] = ticker

            # Filter by timestamp range
            start_ns = int(start_dt.timestamp() * 1e9)
            end_ns = int(end_dt.timestamp() * 1e9)
            df = df[(df["timestamp_utc"] >= start_ns) & (df["timestamp_utc"] <= end_ns)]

            # Apply as-of filter
            if as_of_date:
                as_of_dt = datetime.fromisoformat(as_of_date).replace(tzinfo=UTC)
                int(as_of_dt.timestamp() * 1e9)
                ingestion_col = "ingestion_time"
                if ingestion_col in df.columns:
                    df = df[df[ingestion_col] <= pd.Timestamp(as_of_dt)]

            # Apply corporate action adjustments
            if self.corporate_actions and adjustment != "raw":
                price_cols = ["open", "high", "low", "close"]
                for col in price_cols:
                    if col in df.columns:
                        import numpy as np

                        dates_list = [
                            datetime.utcfromtimestamp(ts / 1e9).strftime("%Y-%m-%d")
                            for ts in df["timestamp_utc"].values
                        ]
                        adjusted = self.corporate_actions.adjust_prices(
                            prices=np.array(df[col].values, dtype=np.float64),
                            dates=dates_list,
                            symbol_id=symbol_id,
                            adjustment_mode=adjustment,
                            as_of_date=as_of_date,
                        )
                        df[col] = adjusted

            frames.append(df)

        if not frames:
            logger.info("no_data_found", symbols=symbols, start=start_date, end=end_date)
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["timestamp_utc", "ticker"]).reset_index(drop=True)

        # Convert timestamp to datetime for convenience
        result["datetime"] = pd.to_datetime(result["timestamp_utc"], unit="ns", utc=True)

        self._cache.put(cache_key, result)

        logger.info(
            "bars_served",
            symbols=len(symbols),
            rows=len(result),
            interval=interval,
            adjustment=adjustment,
        )

        if columns:
            available = [c for c in columns if c in result.columns]
            return result[available]
        return result

    def get_trades(
        self,
        symbol: str,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Get tick-level trade data.

        Args:
            symbol: Ticker symbol.
            start_date: Start date.
            end_date: End date.

        Returns:
            DataFrame with trade data.
        """
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()

        symbol_id = self.symbol_master.get_symbol_id(symbol)
        if symbol_id is None:
            return pd.DataFrame()

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        table = self.lake.read_date_range(
            asset_class="equity",
            schema_name="trades",
            symbol_id=symbol_id,
            start_year=start_dt.year,
            start_month=start_dt.month,
            end_year=end_dt.year,
            end_month=end_dt.month,
        )

        if table is None:
            return pd.DataFrame()

        df = table.to_pandas()
        df["datetime"] = pd.to_datetime(df["timestamp_utc"], unit="ns", utc=True)
        return df.sort_values("timestamp_utc").reset_index(drop=True)

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        logger.info("cache_cleared")

    @staticmethod
    def _make_cache_key(
        symbols: list[str],
        start_date: str,
        end_date: str,
        schema_name: str,
        adjustment: str,
        as_of_date: str | None,
    ) -> str:
        """Generate a deterministic cache key."""
        key_parts = [
            ",".join(sorted(symbols)),
            start_date,
            end_date,
            schema_name,
            adjustment,
            as_of_date or "",
        ]
        raw = "|".join(key_parts)
        return hashlib.sha256(raw.encode()).hexdigest()
