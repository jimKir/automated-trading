"""
price_cache.py — OHLCV Price Data Cache (Alpaca + yfinance)
=============================================================
Wraps Alpaca/yfinance OHLCV fetchers with a Parquet cache so every
experiment run after the first reuses local data instead of hitting the API.

Cache directory: .cache/prices/   (gitignored — see .gitignore)
File naming:     prices_{symbols_hash8}_{start}_{end}_{timeframe}.parquet

Features:
  • Full-coverage detection — if requested date range is already cached, no API call
  • Incremental tail-fetch — if cache covers up to 2024-12-31 and you request
    up to 2026-04-01, only the missing tail is fetched and appended
  • Recent bar re-fetch — last 5 trading days are always refreshed (bars may revise)
  • Registers every fetch in the DataCatalogue (src/market_data/catalogue.py)
  • Thread-safe atomic writes

Usage:
    from momentum_signals_exploration.price_cache import PriceCache

    cache = PriceCache(api_key="...", api_secret="...")
    df = cache.get_daily(
        symbols=["AAPL", "MSFT", "NVDA"],
        start="2023-01-01",
        end="2026-04-01",
    )
    # df is MultiIndex (symbol, date) with columns [open, high, low, close, volume]
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = _REPO_ROOT / ".cache" / "prices"

# How many trailing trading-days to always re-fetch (bars may be revised)
REFRESH_TAIL_DAYS = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symbols_hash(symbols: list[str]) -> str:
    joined = ",".join(sorted(set(s.upper() for s in symbols)))
    return hashlib.md5(joined.encode()).hexdigest()[:8]


def _date_str(d: str | date | datetime) -> str:
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _cache_filename(symbols: list[str], start: str, end: str, timeframe: str) -> str:
    shash = _symbols_hash(symbols)
    tf = timeframe.replace("/", "-")
    return f"prices_{shash}_{start}_{end}_{tf}.parquet"


def _cache_path(symbols: list[str], start: str, end: str, timeframe: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / _cache_filename(symbols, start, end, timeframe)


def _atomic_write(path: Path, df: pd.DataFrame):
    """Write parquet atomically (tmp → rename) to avoid corruption on crash."""
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=True, engine="pyarrow", compression="snappy")
    os.replace(tmp, path)
    logger.debug(f"[PriceCache] Written {path.name} ({path.stat().st_size:,} bytes)")


def _load_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        logger.warning(f"[PriceCache] Corrupted cache {path.name}: {e} — will re-fetch")
        return None


def _trading_days_ago(n: int) -> str:
    """Approximate: go back n*1.5 calendar days to skip weekends."""
    return (datetime.now() - timedelta(days=int(n * 1.5))).strftime("%Y-%m-%d")


def _register_in_catalogue(
    symbols: list[str],
    start: str,
    end: str,
    timeframe: str,
    rows: int,
    cache_path: Path,
    source: str,
):
    """Register this fetch in the DataCatalogue (best-effort, never crashes)."""
    try:
        import sys

        sys.path.insert(0, str(_REPO_ROOT))
        from src.market_data.catalogue import get_catalogue

        cat = get_catalogue()
        tf_map = {
            "1Day": "1day",
            "1Hour": "1hour",
            "1Min": "1min",
            "day": "1day",
            "hour": "1hour",
            "minute": "1min",
        }
        freq = tf_map.get(timeframe, timeframe.lower())
        cat.record(
            source=source,
            dataset=source.upper(),
            schema="ohlcv",
            symbols=symbols,
            start=start,
            end=end,
            frequency=freq,
            rows=rows,
            cache_path=str(cache_path),
            notes=f"price_cache.py — {len(symbols)} symbols",
            tags=["prices", "ohlcv"],
        )
    except Exception as e:
        logger.debug(f"[PriceCache] Catalogue registration skipped: {e}")


# ---------------------------------------------------------------------------
# Main cache class
# ---------------------------------------------------------------------------


class PriceCache:
    """
    OHLCV price data with transparent Parquet caching.

    Priority order for fetching:
      1. Local Parquet cache (free, instant)
      2. Alpaca StockHistoricalDataClient  (free IEX, already authenticated)
      3. yfinance download                 (free, public, fallback)

    Params:
        api_key    — Alpaca API key  (or set ALPACA_API_KEY env var)
        api_secret — Alpaca secret   (or set ALPACA_API_SECRET env var)
        cache_dir  — override default cache directory
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        cache_dir: Path | None = None,
    ):
        self._key = api_key or os.getenv("ALPACA_API_KEY", "") or os.getenv("APCA_API_KEY_ID", "")
        self._secret = (
            api_secret or os.getenv("ALPACA_API_SECRET", "") or os.getenv("APCA_API_SECRET_KEY", "")
        )
        self._cache = cache_dir or CACHE_DIR
        self._cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for *symbols* from *start* to *end*.
        Returns MultiIndex (symbol, date) DataFrame.

        If data is already cached and covers the full requested range,
        returns from cache immediately (no API call).

        If the cache covers up to an earlier date, only the missing
        tail is fetched and merged (incremental update).

        The last REFRESH_TAIL_DAYS of data are always refreshed from
        the API even when cached, because bars may be revised.
        """
        start_s = _date_str(start)
        end_s = _date_str(end or datetime.now())
        symbols = [s.upper() for s in symbols]

        cache_path = _cache_path(symbols, start_s, end_s, "1Day")

        # Check full-coverage cache first
        if not force and cache_path.exists():
            df = _load_parquet(cache_path)
            if df is not None:
                # Verify all symbols present
                cached_syms = set(df.index.get_level_values(0).unique())
                if set(symbols).issubset(cached_syms):
                    logger.info(f"[PriceCache] HIT  {len(symbols)} syms daily {start_s}→{end_s}")
                    # Still refresh the trailing edge
                    df = self._refresh_tail(df, symbols, end_s, "1Day")
                    return df

        # Look for any partial cache covering a subset of the range
        df_cached = self._find_partial_cache(symbols, start_s, end_s, "1Day")

        if df_cached is not None and not force:
            # Determine what tail is missing
            max_cached_date = str(df_cached.index.get_level_values(1).max())[:10]
            tail_start = (
                datetime.strptime(max_cached_date, "%Y-%m-%d") - timedelta(days=REFRESH_TAIL_DAYS)
            ).strftime("%Y-%m-%d")
            logger.info(
                f"[PriceCache] PARTIAL HIT — cached up to {max_cached_date}, "
                f"fetching tail from {tail_start}"
            )
            df_tail = self._fetch_daily(symbols, tail_start, end_s)
            if df_tail is not None and not df_tail.empty:
                # Remove overlap from cache, append tail
                tail_start_ts = pd.Timestamp(tail_start)
                df_old = df_cached[df_cached.index.get_level_values(1) < tail_start_ts]
                df = pd.concat([df_old, df_tail]).sort_index()
            else:
                df = df_cached
        else:
            logger.info(f"[PriceCache] MISS — fetching {len(symbols)} syms daily {start_s}→{end_s}")
            df = self._fetch_daily(symbols, start_s, end_s)
            if df is None or df.empty:
                logger.warning("[PriceCache] No data returned from any source")
                return pd.DataFrame()

        # Save to full-coverage cache path
        _atomic_write(cache_path, df)
        _register_in_catalogue(
            symbols, start_s, end_s, "1Day", rows=len(df), cache_path=cache_path, source="alpaca"
        )
        return df

    def get_intraday(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
        timeframe: str = "1Hour",
        force: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV bars (default: 1-hour).
        Same caching logic as get_daily.
        """
        start_s = _date_str(start)
        end_s = _date_str(end or datetime.now())
        symbols = [s.upper() for s in symbols]

        cache_path = _cache_path(symbols, start_s, end_s, timeframe)

        if not force and cache_path.exists():
            df = _load_parquet(cache_path)
            if df is not None:
                cached_syms = set(df.index.get_level_values(0).unique())
                if set(symbols).issubset(cached_syms):
                    logger.info(
                        f"[PriceCache] HIT  {len(symbols)} syms {timeframe} {start_s}→{end_s}"
                    )
                    return df

        logger.info(
            f"[PriceCache] MISS — fetching {len(symbols)} syms {timeframe} {start_s}→{end_s}"
        )
        df = self._fetch_intraday(symbols, start_s, end_s, timeframe)
        if df is None or df.empty:
            return pd.DataFrame()

        _atomic_write(cache_path, df)
        _register_in_catalogue(
            symbols, start_s, end_s, timeframe, rows=len(df), cache_path=cache_path, source="alpaca"
        )
        return df

    # ------------------------------------------------------------------
    # Private: fetch from API sources
    # ------------------------------------------------------------------

    def _fetch_daily(self, symbols: list[str], start: str, end: str) -> pd.DataFrame | None:
        """Try Alpaca, then yfinance for daily bars."""
        df = self._alpaca_daily(symbols, start, end)
        if df is not None and not df.empty:
            logger.info(
                f"[PriceCache] Alpaca daily: {df.index.get_level_values(0).nunique()}/{len(symbols)} syms"
            )
            return df

        logger.info("[PriceCache] Alpaca returned nothing — trying yfinance...")
        df = self._yfinance_daily(symbols, start, end)
        if df is not None and not df.empty:
            logger.info(
                f"[PriceCache] yfinance daily: {df.index.get_level_values(0).nunique()}/{len(symbols)} syms"
            )
            return df

        return None

    def _fetch_intraday(
        self, symbols: list[str], start: str, end: str, timeframe: str
    ) -> pd.DataFrame | None:
        df = self._alpaca_intraday(symbols, start, end, timeframe)
        if df is not None and not df.empty:
            return df
        return self._yfinance_intraday(symbols, start, end, timeframe)

    def _alpaca_daily(self, symbols: list[str], start: str, end: str) -> pd.DataFrame | None:
        if not self._key:
            return None
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            client = StockHistoricalDataClient(api_key=self._key, secret_key=self._secret)
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start,
                end=end,
                adjustment="all",
            )
            resp = client.get_stock_bars(req)
            if not resp or not resp.data:
                return None

            frames = []
            for sym, bars in resp.data.items():
                if not bars:
                    continue
                rows = [
                    {
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "date": pd.Timestamp(bar.timestamp).normalize(),
                        "_sym": sym,
                    }
                    for bar in bars
                ]
                df = pd.DataFrame(rows).set_index(["_sym", "date"])
                frames.append(df)

            if not frames:
                return None
            out = pd.concat(frames).sort_index()
            out.index.names = ["symbol", "date"]
            return out

        except Exception as e:
            logger.debug(f"[PriceCache] Alpaca daily error: {e}")
            return None

    def _alpaca_intraday(
        self, symbols: list[str], start: str, end: str, timeframe: str
    ) -> pd.DataFrame | None:
        if not self._key:
            return None
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            tf_map = {
                "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            }
            tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))

            client = StockHistoricalDataClient(api_key=self._key, secret_key=self._secret)
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end,
                adjustment="all",
            )
            resp = client.get_stock_bars(req)
            if not resp or not resp.data:
                return None

            frames = []
            for sym, bars in resp.data.items():
                if not bars:
                    continue
                rows = [
                    {
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "ts": pd.Timestamp(bar.timestamp),
                        "_sym": sym,
                    }
                    for bar in bars
                ]
                df = pd.DataFrame(rows).set_index(["_sym", "ts"])
                frames.append(df)

            if not frames:
                return None
            out = pd.concat(frames).sort_index()
            out.index.names = ["symbol", "timestamp"]
            return out

        except Exception as e:
            logger.debug(f"[PriceCache] Alpaca intraday error: {e}")
            return None

    def _yfinance_daily(self, symbols: list[str], start: str, end: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf

            raw = yf.download(
                symbols,
                start=start,
                end=end,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw is None or raw.empty:
                return None
            return self._yf_to_multiindex(raw, symbols, "date")
        except Exception as e:
            logger.debug(f"[PriceCache] yfinance daily error: {e}")
            return None

    def _yfinance_intraday(
        self, symbols: list[str], start: str, end: str, timeframe: str
    ) -> pd.DataFrame | None:
        try:
            import yfinance as yf

            tf_map = {"1Min": "1m", "5Min": "5m", "1Hour": "1h"}
            interval = tf_map.get(timeframe, "1h")
            raw = yf.download(
                symbols,
                start=start,
                end=end,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw is None or raw.empty:
                return None
            return self._yf_to_multiindex(raw, symbols, "timestamp")
        except Exception as e:
            logger.debug(f"[PriceCache] yfinance intraday error: {e}")
            return None

    @staticmethod
    def _yf_to_multiindex(
        raw: pd.DataFrame,
        symbols: list[str],
        idx_name: str,
    ) -> pd.DataFrame | None:
        frames = []
        for sym in symbols:
            try:
                df = raw[sym].copy() if len(symbols) > 1 else raw.copy()
                df = df.dropna(subset=["Close"])
                if len(df) < 2:
                    continue
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )[["open", "high", "low", "close", "volume"]]
                df.index.name = idx_name
                df["_sym"] = sym
                frames.append(df.reset_index().set_index(["_sym", idx_name]))
            except Exception:
                continue

        if not frames:
            return None
        out = pd.concat(frames).sort_index()
        out.index.names = ["symbol", idx_name]
        return out

    # ------------------------------------------------------------------
    # Private: incremental helpers
    # ------------------------------------------------------------------

    def _refresh_tail(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        end: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Re-fetch the last REFRESH_TAIL_DAYS to get revised bars."""
        tail_start = _trading_days_ago(REFRESH_TAIL_DAYS)
        logger.debug(f"[PriceCache] Refreshing tail from {tail_start}")
        try:
            df_tail = self._fetch_daily(symbols, tail_start, end)
            if df_tail is not None and not df_tail.empty:
                tail_ts = pd.Timestamp(tail_start)
                df_old = df[df.index.get_level_values(1) < tail_ts]
                return pd.concat([df_old, df_tail]).sort_index()
        except Exception as e:
            logger.debug(f"[PriceCache] Tail refresh failed: {e}")
        return df

    def _find_partial_cache(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """
        Search .cache/prices/ for any file that covers the requested symbols
        and starts at (or before) the requested start date.
        Returns the best matching DataFrame or None.
        """
        shash = _symbols_hash(symbols)
        tf = timeframe.replace("/", "-")
        pattern = f"prices_{shash}_*_{tf}.parquet"

        candidates = list(self._cache.glob(pattern))
        if not candidates:
            return None

        for cpath in sorted(candidates, reverse=True):  # newest first
            # filename: prices_{hash}_{start}_{end}_{tf}.parquet
            stem = cpath.stem  # e.g. prices_a1b2c3d4_2023-01-01_2025-12-31_1Day
            parts = stem.split("_")
            if len(parts) < 5:
                continue
            try:
                cached_start = parts[2]
                cached_end = parts[3]
                if cached_start <= start and cached_end >= start:
                    df = _load_parquet(cpath)
                    if df is not None:
                        cached_syms = set(df.index.get_level_values(0).unique())
                        if set(symbols).issubset(cached_syms):
                            return df
            except Exception:
                continue

        return None

    # ------------------------------------------------------------------
    # Cache inspection
    # ------------------------------------------------------------------

    def list_cached(self):
        """Print a summary of all cached price files."""
        files = sorted(self._cache.glob("*.parquet"))
        if not files:
            print("  [PriceCache] No cached price files found.")
            return
        print()
        print("=" * 80)
        print("  CACHED PRICE FILES")
        print("=" * 80)
        print(f"  {'File':<50}  {'Size':>10}  {'Modified'}")
        print("  " + "─" * 78)
        total = 0
        for f in files:
            sz = f.stat().st_size
            total += sz
            mod = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            sz_s = f"{sz / 1024:.0f} KB" if sz < 1_048_576 else f"{sz / 1_048_576:.1f} MB"
            print(f"  {f.name:<50}  {sz_s:>10}  {mod}")
        print("=" * 80)
        total_s = f"{total / 1024:.0f} KB" if total < 1_048_576 else f"{total / 1_048_576:.1f} MB"
        print(f"  Total: {len(files)} files, {total_s}")
        print(f"  Cache dir: {self._cache}")
        print()
