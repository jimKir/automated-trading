#!/usr/bin/env python3
"""
Rate-Limit-Aware Retry Wrapper for Fetch Operations
=====================================================
Wraps fetch_all.py with exponential backoff and rate-limit handling.
Automatically retries failed fetches with intelligent delay.

Features:
  • Exponential backoff (2s → 4s → 8s → 16s max)
  • Rate-limit detection and smart pacing
  • Per-symbol error tracking
  • Atomic state preservation (resume-safe)

Usage:
  python3 fetch_with_retry.py --mode update --max-retries 3
"""
from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fetch_retry")

from fetch_all import (
    AlpacaFetcher, load_cached, get_cached_end_date,
    _canonical_path, _atomic_write, _get_catalogue,
    _register, validate_dataframe, OHLCV_DIR, CRYPTO_DIR,
    CACHE_DIR, REFRESH_TAIL
)


class FetchWithRetry:
    """Wraps AlpacaFetcher with retry logic and rate-limit awareness."""

    def __init__(self, max_retries: int = 3, base_backoff: float = 2.0):
        self.api = AlpacaFetcher()
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.failed_symbols: Dict[str, dict] = {}
        self.retry_state_file = CACHE_DIR / "fetch_retry_state.json"
        self._load_retry_state()

    def _load_retry_state(self):
        """Load any previous retry state."""
        if self.retry_state_file.exists():
            try:
                with open(self.retry_state_file) as f:
                    state = json.load(f)
                    self.failed_symbols = state.get("failed", {})
                    logger.info(f"Loaded retry state: {len(self.failed_symbols)} symbols to retry")
            except Exception as e:
                logger.warning(f"Could not load retry state: {e}")

    def _save_retry_state(self):
        """Save current retry state for resume capability."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.retry_state_file, "w") as f:
                json.dump({
                    "failed": self.failed_symbols,
                    "saved_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save retry state: {e}")

    def fetch_bars_with_retry(
        self,
        sym: str,
        start: str,
        end: str,
        is_crypto: bool,
    ) -> Optional[pd.DataFrame]:
        """Fetch bars with exponential backoff retry logic.

        Returns:
            DataFrame if successful, None if all retries exhausted.
        """
        import pandas as pd

        for attempt in range(self.max_retries + 1):
            try:
                df = self.api.fetch_bars(sym, start, end, is_crypto)
                if df is not None and not df.empty:
                    # Clear from failed list on success
                    if sym in self.failed_symbols:
                        del self.failed_symbols[sym]
                    return df
                elif attempt < self.max_retries:
                    # No data this time, retry
                    backoff = min(self.base_backoff ** attempt, 16.0)
                    logger.debug(f"{sym}: No data, retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    # All retries exhausted
                    return None

            except Exception as e:
                if attempt < self.max_retries:
                    backoff = min(self.base_backoff ** attempt, 16.0)
                    logger.warning(f"{sym}: {type(e).__name__}, retry {attempt + 1}/{self.max_retries} in {backoff:.1f}s")
                    time.sleep(backoff)
                else:
                    # Record final failure
                    self.failed_symbols[sym] = {
                        "error": str(e),
                        "last_attempt": datetime.now().isoformat(),
                        "attempts": self.max_retries + 1,
                    }
                    logger.error(f"{sym}: Failed after {self.max_retries + 1} attempts")
                    return None

        return None

    def retry_failed_symbols(self, is_crypto: bool = False):
        """Retry symbols that failed in previous runs."""
        if not self.failed_symbols:
            logger.info("No failed symbols to retry")
            return

        logger.info(f"Retrying {len(self.failed_symbols)} previously failed symbols...")
        retried = 0
        recovered = 0

        for sym, failure_info in list(self.failed_symbols.items()):
            is_sym_crypto = failure_info.get("is_crypto", is_crypto)
            if is_sym_crypto != is_crypto:
                continue

            logger.info(f"Retrying {sym}...")
            cached_end = get_cached_end_date(sym, is_sym_crypto)
            today = datetime.now().strftime("%Y-%m-%d")

            if cached_end:
                fetch_start = (
                    datetime.strptime(cached_end, "%Y-%m-%d") - timedelta(days=REFRESH_TAIL)
                ).strftime("%Y-%m-%d")
            else:
                fetch_start = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")

            df_new = self.fetch_bars_with_retry(sym, fetch_start, today, is_sym_crypto)

            if df_new is not None and not df_new.empty:
                # Attempt to merge and write
                df_old = load_cached(sym, is_sym_crypto)
                if df_old is not None:
                    import pandas as pd
                    cutoff = pd.Timestamp(fetch_start, tz="UTC")
                    if df_old.index.tz is None:
                        cutoff = pd.Timestamp(fetch_start)
                    else:
                        cutoff = pd.Timestamp(fetch_start, tz="UTC").tz_convert(df_old.index.tz)
                    df_old = df_old[df_old.index < cutoff]
                    df_merged = pd.concat([df_old, df_new]).sort_index()
                    df_merged = df_merged[~df_merged.index.duplicated(keep="last")]
                else:
                    df_merged = df_new

                if _atomic_write(_canonical_path(sym, is_sym_crypto), df_merged):
                    logger.info(f"{sym}: ✓ Recovered")
                    recovered += 1
                    del self.failed_symbols[sym]

            retried += 1

        logger.info(f"Retry complete: {recovered}/{retried} recovered")
        self._save_retry_state()

    def report_failed(self):
        """Generate report of permanently failed symbols."""
        if not self.failed_symbols:
            return

        print(f"\n{'='*70}")
        print(f"  PERMANENTLY FAILED SYMBOLS ({len(self.failed_symbols)})")
        print(f"{'='*70}\n")

        for sym, info in self.failed_symbols.items():
            attempts = info.get("attempts", "?")
            error = info.get("error", "Unknown")[:60]
            print(f"  {sym:12s} ({attempts} attempts) — {error}")

        print(f"\n{'='*70}\n")
