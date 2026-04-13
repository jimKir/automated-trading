"""
Unified Data Cache — Single interface for ALL cached price data
================================================================
Bridges two cache systems:
  1. data_cache/ohlcv/{SYMBOL}/daily.parquet  — bulk fetcher output (fetch_all.py)
  2. .cache/prices/prices_{hash}_{start}_{end}_{tf}.parquet — PriceCache output

Usage:
    from data.unified_cache import UnifiedCache

    cache = UnifiedCache()

    # Get data for any symbol — checks both caches transparently
    df = cache.get("AAPL", start="2023-01-01")

    # Get multiple symbols
    data = cache.get_multi(["AAPL", "MSFT", "NVDA"], start="2023-01-01")

    # Check what's available
    cache.info("AAPL")

    # Stale check
    stale = cache.stale_symbols(["AAPL", "MSFT"], max_age_days=3)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
BULK_CACHE = ROOT / "data_cache"
PRICE_CACHE = ROOT / ".cache" / "prices"


class UnifiedCache:
    """
    Unified read interface for all cached price data.
    Reads from bulk cache (fetch_all) first, falls back to PriceCache.
    Does NOT write — use fetch_all.py or PriceCache for that.
    """

    def __init__(self, bulk_dir: Path | None = None, price_dir: Path | None = None):
        self._bulk = bulk_dir or BULK_CACHE
        self._price = price_dir or PRICE_CACHE

    def get(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        is_crypto: bool = False,
    ) -> pd.DataFrame | None:
        """
        Load daily OHLCV for a symbol from any available cache.
        Returns DataFrame with DatetimeIndex, or None if not cached.
        """
        df = self._from_bulk(symbol, is_crypto)

        if df is None:
            df = self._from_price_cache(symbol)

        if df is None:
            return None

        # Apply date filters (handle tz-aware vs tz-naive)
        if start:
            ts = pd.Timestamp(start)
            if df.index.tz is not None and ts.tz is None:
                ts = ts.tz_localize(df.index.tz)
            df = df[df.index >= ts]
        if end:
            ts = pd.Timestamp(end)
            if df.index.tz is not None and ts.tz is None:
                ts = ts.tz_localize(df.index.tz)
            df = df[df.index <= ts]

        return df if not df.empty else None

    def get_multi(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load multiple symbols. Returns {symbol: DataFrame}."""
        result = {}
        for sym in symbols:
            is_crypto = "/" in sym or any(
                sym.upper().replace("-", "/").endswith(s)
                for s in ["/USD", "/USDC", "/USDT", "/BTC"]
            )
            df = self.get(sym, start=start, end=end, is_crypto=is_crypto)
            if df is not None:
                result[sym] = df
        return result

    def info(self, symbol: str, is_crypto: bool = False) -> dict:
        """Get metadata about cached data for a symbol."""
        df = self.get(symbol, is_crypto=is_crypto)
        if df is None:
            return {"symbol": symbol, "cached": False}

        return {
            "symbol": symbol,
            "cached": True,
            "rows": len(df),
            "start": str(df.index.min())[:10],
            "end": str(df.index.max())[:10],
            "columns": list(df.columns),
            "size_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
            "source": self._identify_source(symbol, is_crypto),
        }

    def stale_symbols(
        self,
        symbols: list[str],
        max_age_days: int = 3,
    ) -> list[dict]:
        """Check which symbols have stale data."""
        cutoff = (datetime.now() - timedelta(days=max_age_days + 2)).strftime("%Y-%m-%d")
        stale = []
        for sym in symbols:
            is_crypto = "/" in sym or "-USD" in sym
            df = self.get(sym, is_crypto=is_crypto)
            if df is None:
                stale.append({"symbol": sym, "last_date": None, "reason": "not_cached"})
            else:
                last = str(df.index.max())[:10]
                if last < cutoff:
                    stale.append({"symbol": sym, "last_date": last, "reason": "stale"})
        return stale

    def available_symbols(self) -> dict:
        """List all cached symbols by type."""
        equity = []
        crypto = []

        ohlcv_dir = self._bulk / "ohlcv"
        crypto_dir = self._bulk / "crypto"

        if ohlcv_dir.exists():
            equity = sorted(d.name for d in ohlcv_dir.iterdir() if d.is_dir())
        if crypto_dir.exists():
            crypto = sorted(d.name for d in crypto_dir.iterdir() if d.is_dir())

        return {"equity": equity, "crypto": crypto, "total": len(equity) + len(crypto)}

    # ── Internal loaders ──

    def _from_bulk(self, symbol: str, is_crypto: bool) -> pd.DataFrame | None:
        """Load from data_cache/ (fetch_all output)."""
        safe = symbol.replace("/", "-").replace(":", "-")
        subdir = "crypto" if is_crypto else "ohlcv"
        sym_dir = self._bulk / subdir / safe

        if not sym_dir.exists():
            return None

        # Prefer canonical
        canonical = sym_dir / "daily.parquet"
        if canonical.exists():
            return self._read_parquet(canonical)

        # Fall back to legacy date-range files (newest first)
        files = sorted(sym_dir.glob("*.parquet"), reverse=True)
        for f in files:
            df = self._read_parquet(f)
            if df is not None:
                return df
        return None

    def _from_price_cache(self, symbol: str) -> pd.DataFrame | None:
        """Load from .cache/prices/ (PriceCache output)."""
        if not self._price.exists():
            return None

        # Search for any file containing this symbol's data
        for f in sorted(self._price.glob("*.parquet"), reverse=True):
            try:
                df = pd.read_parquet(f)
                if isinstance(df.index, pd.MultiIndex) and symbol in df.index.get_level_values(0):
                    return df.loc[symbol]
                return df
            except Exception:
                continue
        return None

    def _read_parquet(self, path: Path) -> pd.DataFrame | None:
        try:
            df = pd.read_parquet(path)
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)
            return df
        except Exception:
            return None

    def _identify_source(self, symbol: str, is_crypto: bool) -> str:
        safe = symbol.replace("/", "-").replace(":", "-")
        subdir = "crypto" if is_crypto else "ohlcv"
        sym_dir = self._bulk / subdir / safe
        if sym_dir.exists() and list(sym_dir.glob("*.parquet")):
            return "bulk_cache"
        return "price_cache"
