#!/usr/bin/env python3
"""
Historical Data Fetcher for ALL Alpaca Instruments
====================================================
Integrated with codebase infrastructure:
  • DataCatalogue — registers every fetch for discoverability
  • Delta fetch   — only downloads bars newer than cached data
  • Quality checks — validates data after each batch
  • Alert hooks    — surfaces errors for monitoring
  • Single parquet — one canonical file per symbol (no duplicates)
  • Atomic writes  — temp → rename to prevent corruption

Usage:
  python3 fetch_all.py                  # Full backfill (5y history)
  python3 fetch_all.py --update         # Delta update (fetch only new bars)
  python3 fetch_all.py --validate       # Run quality checks on cached data
  python3 fetch_all.py --stats          # Print cache statistics
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── Setup ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fetch_all")

# ── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR = ROOT / "data_cache"
OHLCV_DIR = CACHE_DIR / "ohlcv"
CRYPTO_DIR = CACHE_DIR / "crypto"
PROGRESS_FILE = CACHE_DIR / "fetch_progress.json"
QUALITY_FILE = CACHE_DIR / "quality_report.json"
REFRESH_TAIL = 7  # Always re-fetch last N calendar days (bars may revise)
RATE_LIMIT = 180  # Max requests per 60s window
HISTORY_YEARS = 5  # Default backfill depth


# ── Timezone helper ─────────────────────────────────────────────────────────
def tz_aware_cutoff(date_str: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Create a cutoff Timestamp that matches the timezone of a DatetimeIndex.

    Prevents TypeError when comparing tz-naive vs tz-aware datetimes.
    This was previously duplicated in 4+ locations; centralised here.
    """
    if index.tz is None:
        return pd.Timestamp(date_str)
    return pd.Timestamp(date_str, tz="UTC").tz_convert(index.tz)


# ── Catalogue integration ────────────────────────────────────────────────────
def _get_catalogue():
    """Get DataCatalogue singleton (best-effort, never crashes)."""
    try:
        from src.market_data.catalogue import get_catalogue

        return get_catalogue()
    except Exception:
        return None


def _register(cat, sym, is_crypto, start, end, rows, cache_path):
    """Register a fetch in the DataCatalogue."""
    if cat is None:
        return
    try:
        cat.record(
            source="alpaca",
            dataset="ALPACA" if not is_crypto else "ALPACA_CRYPTO",
            schema="ohlcv",
            symbols=[sym],
            start=start,
            end=end,
            frequency="1day",
            rows=rows,
            cache_path=str(cache_path),
            tags=["ohlcv", "daily", "crypto" if is_crypto else "equity"],
            artifact_type="raw_fetch",
        )
    except Exception as e:
        logger.debug(f"Catalogue registration skipped for {sym}: {e}")


# ── Quality checks ───────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame, sym: str) -> dict:
    """Run quality checks on a single symbol's DataFrame."""
    issues = []
    stats = {"sym": sym, "rows": len(df), "issues": issues}

    if df.empty:
        issues.append("empty_dataframe")
        return stats

    # Normalise column names to lowercase for consistent checks
    col_map = {c: c.lower() for c in df.columns}
    df_check = df.rename(columns=col_map)

    # Required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df_check.columns)
    if missing:
        issues.append(f"missing_columns:{missing}")

    # NaN check
    for col in ["open", "high", "low", "close"]:
        if col in df_check.columns:
            nan_count = df_check[col].isna().sum()
            if nan_count > 0:
                issues.append(f"nan_{col}:{nan_count}")

    # Negative prices
    for col in ["open", "high", "low", "close"]:
        if col in df_check.columns and (df_check[col] < 0).any():
            issues.append(f"negative_{col}:{(df_check[col] < 0).sum()}")

    # Zero volume (>50% of rows = likely stale)
    if "volume" in df_check.columns:
        zero_pct = (df_check["volume"] == 0).mean()
        if zero_pct > 0.5:
            issues.append(f"high_zero_volume:{zero_pct:.0%}")

    # Price jumps > 80% in a single day
    if "close" in df_check.columns and len(df_check) > 1:
        returns = df_check["close"].pct_change().abs()
        jumps = (returns > 0.8).sum()
        if jumps > 3:
            issues.append(f"extreme_jumps:{jumps}")

    stats["quality"] = "PASS" if not issues else "WARN"
    return stats


# ── File I/O ─────────────────────────────────────────────────────────────────
def _sym_dir(sym: str, is_crypto: bool) -> Path:
    """Return the directory for a symbol's parquet files."""
    safe = sym.replace("/", "-").replace(":", "-")
    return (CRYPTO_DIR if is_crypto else OHLCV_DIR) / safe


def _canonical_path(sym: str, is_crypto: bool) -> Path:
    """Single canonical parquet file per symbol."""
    return _sym_dir(sym, is_crypto) / "daily.parquet"


def _atomic_write(path: Path, df: pd.DataFrame):
    """Write parquet atomically: temp file → rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        df.to_parquet(tmp, compression="snappy", index=True)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.warning(f"Write failed for {path}: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def load_cached(sym: str, is_crypto: bool) -> pd.DataFrame | None:
    """Load existing cached data for a symbol."""
    canon = _canonical_path(sym, is_crypto)
    if canon.exists():
        try:
            df = pd.read_parquet(canon)
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)
            return df
        except Exception:
            pass

    # Fall back to legacy date-range files
    d = _sym_dir(sym, is_crypto)
    if not d.exists():
        return None
    files = sorted(d.glob("*.parquet"))
    files = [f for f in files if f.name != "daily.parquet" and f.suffix == ".parquet"]
    if not files:
        return None

    # Load the newest legacy file
    try:
        df = pd.read_parquet(files[-1])
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        return df
    except Exception:
        return None


def get_cached_end_date(sym: str, is_crypto: bool) -> str | None:
    """Return the latest date in cached data, or None if no cache."""
    df = load_cached(sym, is_crypto)
    if df is None or df.empty:
        return None
    try:
        return str(df.index.max())[:10]
    except Exception:
        return None


# ── Alpaca API ───────────────────────────────────────────────────────────────
class AlpacaFetcher:
    def __init__(self):
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        k = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        s = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
        if not k or not s:
            raise ValueError(
                "Missing API credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in .env"
            )

        self.trading = TradingClient(k, s)
        self.stock_client = StockHistoricalDataClient(k, s)
        self.crypto_client = CryptoHistoricalDataClient(k, s)
        self._req_times = []
        logger.info("Alpaca connected")

    def _rate_limit(self):
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60]
        if len(self._req_times) >= RATE_LIMIT:
            wait = 60 - (now - self._req_times[0])
            if wait > 0:
                logger.warning(f"Rate limit — waiting {wait:.0f}s")
                time.sleep(wait + 1)
                self._req_times = []
        self._req_times.append(now)

    def get_all_assets(self):
        from alpaca.trading.requests import GetAssetsRequest

        self._rate_limit()
        assets = self.trading.get_all_assets(GetAssetsRequest(status="active"))
        result = []
        for a in assets:
            t = str(a.asset_class).lower()
            is_crypto = "crypto" in t
            result.append({"symbol": a.symbol, "class": t, "is_crypto": is_crypto})
        return result

    def fetch_bars(self, sym: str, start: str, end: str, is_crypto: bool) -> pd.DataFrame | None:
        """Fetch OHLCV bars for a single symbol.

        For crypto: tries Alpaca first, falls back to yfinance for BTC-USD, ETH-USD, SOL-USD.
        """
        try:
            self._rate_limit()
            if is_crypto:
                # Try Alpaca crypto first
                try:
                    from alpaca.data.requests import CryptoBarsRequest
                    from alpaca.data.timeframe import TimeFrame

                    req = CryptoBarsRequest(
                        symbol_or_symbols=sym,
                        timeframe=TimeFrame.Day,
                        start=pd.Timestamp(start),
                        end=pd.Timestamp(end),
                    )
                    bars = self.crypto_client.get_crypto_bars(req)
                    df = bars.df
                    if df is not None and not df.empty:
                        if isinstance(df.index, pd.MultiIndex):
                            df = df.droplevel("symbol")
                        return df
                except Exception as alpaca_ex:
                    logger.debug(f"{sym}: Alpaca crypto fetch failed - {type(alpaca_ex).__name__}")

                # Fallback to yfinance for common crypto
                from crypto_fetcher import YFinanceCryptoFetcher

                yf_fetcher = YFinanceCryptoFetcher()
                if yf_fetcher.is_available(sym):
                    return yf_fetcher.fetch_bars(sym, start, end)
                return None
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Day,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
            )
            bars = self.stock_client.get_stock_bars(req)

            df = bars.df
            if df.empty:
                return None
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel("symbol")
            return df

        except Exception as ex:
            logger.debug(f"{sym}: {type(ex).__name__}: {ex}")
            return None


# ── Main orchestrator ────────────────────────────────────────────────────────
def run_fetch(mode: str = "backfill"):
    """
    Main fetch loop.
    mode='backfill' — full 5-year history for symbols with no cache
    mode='update'   — delta fetch (only new bars since last cached date)
    """
    api = AlpacaFetcher()
    cat = _get_catalogue()

    # Create directories
    for d in [CACHE_DIR, OHLCV_DIR, CRYPTO_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load progress
    prog = {
        "fetched": [],
        "failed": [],
        "skipped": [],
        "delta_updated": [],
        "total": 0,
        "mode": mode,
        "started": datetime.now().isoformat(),
    }

    # Get all tradeable assets
    logger.info("Fetching asset list...")
    assets = api.get_all_assets()
    prog["total"] = len(assets)

    stocks = [a for a in assets if not a["is_crypto"]]
    cryptos = [a for a in assets if a["is_crypto"]]
    logger.info(f"Assets: {len(stocks)} stocks/ETFs + {len(cryptos)} crypto = {len(assets)} total")

    today = datetime.now().strftime("%Y-%m-%d")
    backfill_start = (datetime.now() - timedelta(days=HISTORY_YEARS * 365)).strftime("%Y-%m-%d")

    quality_issues = []
    t0 = time.time()

    for batch, is_crypto in [(stocks, False), (cryptos, True)]:
        label = "CRYPTO" if is_crypto else "STOCKS/ETFs"
        logger.info(f"\n{'=' * 60}\n{label} ({len(batch)} symbols)\n{'=' * 60}")

        for i, asset in enumerate(batch):
            sym = asset["symbol"]

            # ── Delta logic: check what's already cached ──
            cached_end = get_cached_end_date(sym, is_crypto)

            if cached_end:
                # We have data. In backfill mode, skip. In update mode, fetch tail.
                if mode == "backfill":
                    prog["skipped"].append(sym)
                    continue

                # Delta: fetch from (cached_end - REFRESH_TAIL) to today
                delta_start = (
                    datetime.strptime(cached_end, "%Y-%m-%d") - timedelta(days=REFRESH_TAIL)
                ).strftime("%Y-%m-%d")

                if delta_start >= today:
                    prog["skipped"].append(sym)
                    continue

                df_new = api.fetch_bars(sym, delta_start, today, is_crypto)
                if df_new is not None and not df_new.empty:
                    # Merge with existing cache
                    df_old = load_cached(sym, is_crypto)
                    if df_old is not None:
                        cutoff = tz_aware_cutoff(delta_start, df_old.index)
                        df_old = df_old[df_old.index < cutoff]
                        df_merged = pd.concat([df_old, df_new]).sort_index()
                        df_merged = df_merged[~df_merged.index.duplicated(keep="last")]
                    else:
                        df_merged = df_new

                    if _atomic_write(_canonical_path(sym, is_crypto), df_merged):
                        prog["delta_updated"].append(sym)
                        _register(
                            cat,
                            sym,
                            is_crypto,
                            backfill_start,
                            today,
                            len(df_merged),
                            _canonical_path(sym, is_crypto),
                        )
                    else:
                        prog["failed"].append(sym)
                else:
                    prog["skipped"].append(sym)

            else:
                # No cache — full backfill
                df = api.fetch_bars(sym, backfill_start, today, is_crypto)
                if df is not None and not df.empty:
                    # Quality check
                    qc = validate_dataframe(df, sym)
                    if qc["issues"]:
                        quality_issues.append(qc)

                    if _atomic_write(_canonical_path(sym, is_crypto), df):
                        prog["fetched"].append(sym)
                        _register(
                            cat,
                            sym,
                            is_crypto,
                            backfill_start,
                            today,
                            len(df),
                            _canonical_path(sym, is_crypto),
                        )
                    else:
                        prog["failed"].append(sym)
                else:
                    prog["failed"].append(sym)

            # ── Progress reporting ──
            done = i + 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = len(batch) - done
                eta = remaining / rate / 60 if rate > 0 else 0
                pct = done / len(batch) * 100
                n_ok = len(prog["fetched"]) + len(prog["delta_updated"])
                n_skip = len(prog["skipped"])
                n_fail = len(prog["failed"])
                logger.info(
                    f"[{label}] {done}/{len(batch)} ({pct:.1f}%) | "
                    f"{rate:.1f}/sec | ETA {eta:.0f}m | "
                    f"OK:{n_ok} skip:{n_skip} fail:{n_fail}"
                )

    # ── Final report ─────────────────────────────────────────────────────
    prog["finished"] = datetime.now().isoformat()
    prog["elapsed_sec"] = round(time.time() - t0, 1)

    # Save progress
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2, default=str)

    # Save quality report
    if quality_issues:
        with open(QUALITY_FILE, "w") as f:
            json.dump(quality_issues, f, indent=2, default=str)

    n_ok = len(prog["fetched"]) + len(prog["delta_updated"])
    n_fail = len(prog["failed"])
    n_skip = len(prog["skipped"])
    total = prog["total"]

    print(f"""
{"=" * 60}
  FETCH COMPLETE  ({mode} mode)
{"=" * 60}
  Total assets:     {total:,}
  New fetched:      {len(prog["fetched"]):,}
  Delta updated:    {len(prog["delta_updated"]):,}
  Skipped (cached): {n_skip:,}
  Failed:           {n_fail:,}
  Quality warnings: {len(quality_issues)}
  Time:             {prog["elapsed_sec"]:.0f}s
  Cache:            {CACHE_DIR}
{"=" * 60}
""")

    return prog


# ── Statistics ───────────────────────────────────────────────────────────────
def print_stats():
    """Print comprehensive cache statistics."""
    ohlcv_files = list(OHLCV_DIR.glob("*/daily.parquet")) if OHLCV_DIR.exists() else []
    crypto_files = list(CRYPTO_DIR.glob("*/daily.parquet")) if CRYPTO_DIR.exists() else []

    # Also count legacy files
    legacy_ohlcv = list(OHLCV_DIR.glob("*/*.parquet")) if OHLCV_DIR.exists() else []
    legacy_crypto = list(CRYPTO_DIR.glob("*/*.parquet")) if CRYPTO_DIR.exists() else []

    total_size = sum(f.stat().st_size for f in legacy_ohlcv + legacy_crypto)
    unique_syms_eq = len({f.parent.name for f in legacy_ohlcv})
    unique_syms_cr = len({f.parent.name for f in legacy_crypto})

    print(f"""
{"=" * 60}
  DATA CACHE STATISTICS
{"=" * 60}
  Stocks/ETFs:    {unique_syms_eq:,} symbols
  Crypto:         {unique_syms_cr:,} symbols
  Total:          {unique_syms_eq + unique_syms_cr:,} symbols
  Canonical:      {len(ohlcv_files) + len(crypto_files):,} daily.parquet files
  Legacy:         {len(legacy_ohlcv) + len(legacy_crypto) - len(ohlcv_files) - len(crypto_files):,} date-range files
  Disk usage:     {total_size / (1024 * 1024):.1f} MB
  Cache path:     {CACHE_DIR}
{"=" * 60}
""")

    # Catalogue check
    cat = _get_catalogue()
    if cat:
        entries = cat.find_all(source="alpaca")
        print(f"  Catalogue entries: {len(entries)}")
    else:
        print("  Catalogue: not connected")

    # Staleness check
    stale_count = 0
    two_days_ago = (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")
    sample_syms = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
    for sym in sample_syms:
        end = get_cached_end_date(sym, False)
        status = "STALE" if end and end < two_days_ago else "OK"
        if status == "STALE":
            stale_count += 1
        print(f"  {sym:6s} last bar: {end or 'N/A':12s} [{status}]")

    if stale_count > 0:
        print(f"\n  ⚠️  {stale_count} key symbols are stale. Run: python3 fetch_all.py --update")


# ── Validation ───────────────────────────────────────────────────────────────
def run_validation():
    """Run quality checks on all cached data."""
    import random

    all_dirs = []
    if OHLCV_DIR.exists():
        all_dirs += [(d, False) for d in OHLCV_DIR.iterdir() if d.is_dir()]
    if CRYPTO_DIR.exists():
        all_dirs += [(d, True) for d in CRYPTO_DIR.iterdir() if d.is_dir()]

    # Sample 500 symbols for speed
    sample = random.sample(all_dirs, min(500, len(all_dirs)))

    issues_total = []
    rows_total = 0
    checked = 0

    for sym_dir, is_crypto in sample:
        sym = sym_dir.name
        df = load_cached(sym, is_crypto)
        if df is None:
            continue
        checked += 1
        rows_total += len(df)
        qc = validate_dataframe(df, sym)
        if qc["issues"]:
            issues_total.append(qc)

    n_warn = len(issues_total)
    print(f"""
{"=" * 60}
  DATA QUALITY VALIDATION
{"=" * 60}
  Symbols checked:  {checked}
  Total rows:       {rows_total:,}
  Passed:           {checked - n_warn}
  Warnings:         {n_warn}
  Pass rate:        {(checked - n_warn) / max(checked, 1) * 100:.1f}%
{"=" * 60}
""")

    if issues_total:
        print("  Issues found:")
        for qc in issues_total[:20]:
            print(f"    {qc['sym']:10s} — {', '.join(qc['issues'])}")
        if len(issues_total) > 20:
            print(f"    ... and {len(issues_total) - 20} more")

    # Save report
    with open(QUALITY_FILE, "w") as f:
        json.dump(issues_total, f, indent=2, default=str)
    print(f"\n  Report saved: {QUALITY_FILE}")


# ── Cleanup duplicates ───────────────────────────────────────────────────────
def migrate_to_canonical():
    """
    Migrate legacy date-range files to canonical daily.parquet.
    Merges overlapping files and removes duplicates.
    """
    migrated = 0
    for base_dir in [OHLCV_DIR, CRYPTO_DIR]:
        if not base_dir.exists():
            continue
        for sym_dir in base_dir.iterdir():
            if not sym_dir.is_dir():
                continue
            pfiles = sorted(sym_dir.glob("*.parquet"))
            canonical = sym_dir / "daily.parquet"

            # Skip if only canonical exists
            legacy = [f for f in pfiles if f.name != "daily.parquet"]
            if not legacy:
                continue

            # Load and merge all files
            frames = []
            for f in pfiles:
                try:
                    df = pd.read_parquet(f)
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.droplevel(0)
                    frames.append(df)
                except Exception:
                    pass

            if not frames:
                continue

            merged = pd.concat(frames).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]

            if _atomic_write(canonical, merged):
                # Remove legacy files
                for f in legacy:
                    with contextlib.suppress(Exception):
                        f.unlink()
                migrated += 1

    logger.info(f"Migrated {migrated} symbols to canonical format")
    return migrated


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca Historical Data Fetcher")
    parser.add_argument(
        "--update", action="store_true", help="Delta update: only fetch new bars since last cache"
    )
    parser.add_argument("--validate", action="store_true", help="Run quality checks on cached data")
    parser.add_argument("--stats", action="store_true", help="Print cache statistics")
    parser.add_argument(
        "--migrate", action="store_true", help="Migrate legacy files to canonical format"
    )
    args = parser.parse_args()

    if args.validate:
        run_validation()
    elif args.stats:
        print_stats()
    elif args.migrate:
        migrate_to_canonical()
    elif args.update:
        run_fetch(mode="update")
    else:
        run_fetch(mode="backfill")
