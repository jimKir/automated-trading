"""
Multi-source data feed.
Supports:
  - yfinance  (backtest + paper)  — parallel batch + threaded fallback
  - ccxt      (crypto live)
  - IBKR TWS  (equities / futures live)  -- adapter stub
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from utils.logger import get_logger

log = get_logger("DataFeed")

# Max parallel threads for yfinance (avoid rate-limiting)
_YF_MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}

    # Handle duplicate columns from yfinance multi-level flatten
    # e.g. ('Close', 'SPY'), ('Close', 'SPY') → 'Close', 'Close'
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{symbol}] Missing columns: {missing}")
    df = df.dropna(subset=list(required))
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _fetch_single_symbol(sym: str, start: str, end: str, interval: str, retries: int = 3):
    """Fetch one symbol — used as thread target."""
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                sym, start=start, end=end, interval=interval,
                progress=False, auto_adjust=True, threads=False,
            )
            if raw.empty:
                return sym, None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = _validate_ohlcv(raw, sym)
            return sym, raw
        except Exception as exc:
            if attempt < retries:
                time.sleep(1.5 ** attempt)
            else:
                return sym, None
    return sym, None


# ---------------------------------------------------------------------------
# yfinance feed  (free, good for backtest + paper)
# ---------------------------------------------------------------------------

def fetch_yfinance(
    symbols: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    retries: int = 3,
    max_workers: int = _YF_MAX_WORKERS,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data from Yahoo Finance — PARALLEL via thread pool.

    Each symbol is downloaded individually in its own thread.
    This avoids yfinance batch mode issues while still parallelizing I/O.

    Parameters
    ----------
    symbols     : list of tickers
    start       : "YYYY-MM-DD"
    end         : "YYYY-MM-DD"
    interval    : "1d", "1h", "5m", etc.
    max_workers : max concurrent download threads (default 8)
    """
    result: Dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_single_symbol, sym, start, end, interval, retries): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                _, df = future.result()
                if df is not None:
                    result[sym] = df
                    log.info(f"[{sym}] Fetched {len(df)} bars ({start} → {end})")
                else:
                    log.warning(f"[{sym}] Empty data from yfinance")
            except Exception as exc:
                log.error(f"[{sym}] Download failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# CCXT feed  (crypto live / paper)
# ---------------------------------------------------------------------------

def fetch_ccxt(
    symbols: List[str],
    exchange_id: str = "binance",
    start: str = None,
    end: str = None,
    timeframe: str = "1d",
    api_key: str = "",
    api_secret: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV from any CCXT-supported exchange.
    Falls back to public endpoints if no credentials.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("pip install ccxt")

    exchange_cls = getattr(ccxt, exchange_id)
    kwargs: dict = {"enableRateLimit": True}
    if api_key:
        kwargs["apiKey"] = api_key
        kwargs["secret"] = api_secret

    exchange = exchange_cls(kwargs)

    since_ms = (
        int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000) if start else None
    )

    result: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, since=since_ms, limit=1000)
            if not ohlcv:
                log.warning(f"[{sym}] No data from {exchange_id}")
                continue
            df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            result[sym] = df
            log.info(f"[{sym}] CCXT fetched {len(df)} bars")
        except Exception as exc:
            log.error(f"[{sym}] CCXT error: {exc}")

    return result


# ---------------------------------------------------------------------------
# Universal loader
# ---------------------------------------------------------------------------

class DataFeed:
    """
    Unified interface: chooses backend based on config mode.
    In backtest/paper mode → yfinance.
    In live mode with crypto → ccxt.
    In live mode with equities/futures → IBKR (via execution adapter).
    """

    def __init__(self, config: dict):
        self.config = config
        self.mode = config["system"]["mode"]
        self._cache: Dict[str, pd.DataFrame] = {}

    def load(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        source: str = "yfinance",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data. Returns dict of {symbol: OHLCV DataFrame}.
        Uses internal cache to avoid re-downloading.
        """
        cache_key = f"{source}:{','.join(sorted(symbols))}:{start}:{end}:{interval}"
        if cache_key in self._cache:
            return {s: self._cache[f"{source}:{s}:{start}:{end}:{interval}"] for s in symbols if f"{source}:{s}:{start}:{end}:{interval}" in self._cache}

        if source == "yfinance":
            data = fetch_yfinance(symbols, start=start or "2018-01-01",
                                  end=end or datetime.today().strftime("%Y-%m-%d"),
                                  interval=interval)
        elif source == "ccxt":
            binance_cfg = self.config.get("brokers", {}).get("binance", {})
            data = fetch_ccxt(
                symbols,
                exchange_id="binance",
                start=start,
                end=end,
                timeframe=interval,
                api_key=binance_cfg.get("api_key", ""),
                api_secret=binance_cfg.get("api_secret", ""),
            )
        else:
            raise ValueError(f"Unknown data source: {source}")

        # populate cache
        for sym, df in data.items():
            self._cache[f"{source}:{sym}:{start}:{end}:{interval}"] = df

        return data

    def load_all(self, start: str = None, end: str = None) -> Dict[str, pd.DataFrame]:
        """Load all assets defined in config."""
        assets_cfg = self.config.get("assets", {})
        all_data: Dict[str, pd.DataFrame] = {}

        if assets_cfg.get("equities", {}).get("enabled"):
            syms = assets_cfg["equities"]["universe"]
            all_data.update(self.load(syms, start=start, end=end, source="yfinance"))

        if assets_cfg.get("futures", {}).get("enabled"):
            syms = assets_cfg["futures"]["universe"]
            all_data.update(self.load(syms, start=start, end=end, source="yfinance"))

        if assets_cfg.get("crypto", {}).get("enabled"):
            syms = assets_cfg["crypto"]["universe"]
            # yfinance works for daily crypto data; use ccxt in live mode
            src = "ccxt" if self.mode == "live" else "yfinance"
            all_data.update(self.load(syms, start=start, end=end, source=src))

        return all_data
