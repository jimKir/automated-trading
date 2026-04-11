#!/usr/bin/env python3
"""
Migrate Legacy Data Files to Canonical Format
==============================================
Consolidates date-range parquet files (e.g., 2021-04-01_2026-04-04.parquet)
into single canonical files (daily.parquet) per symbol.

Preserves all data, handles duplicates (keeps latest).
Run once during transition, then archive legacy files.

Usage:
  python3 migrate_to_canonical.py               # Migrate all symbols
  python3 migrate_to_canonical.py --dry-run     # Preview without writing
  python3 migrate_to_canonical.py --symbol SPY  # Migrate single symbol
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("migrate_canonical")

from fetch_all import (
    CRYPTO_DIR,
    OHLCV_DIR,
    _atomic_write,
    _canonical_path,
    _get_catalogue,
    _register,
    _sym_dir,
)


def get_all_symbols(is_crypto: bool = False):
    """Get all symbols that have legacy files."""
    d = CRYPTO_DIR if is_crypto else OHLCV_DIR
    if not d.exists():
        return []

    symbols = []
    for sym_dir in d.iterdir():
        if not sym_dir.is_dir():
            continue
        legacy_files = [f for f in sym_dir.glob("*.parquet") if f.name != "daily.parquet"]
        if legacy_files:
            symbols.append(sym_dir.name)
    return symbols


def migrate_symbol(sym: str, is_crypto: bool, dry_run: bool = False) -> dict:
    """Migrate a single symbol's legacy files to canonical format.

    Returns dict with migration result and stats.
    """
    result = {
        "symbol": sym,
        "status": "pending",
        "files_merged": 0,
        "total_rows": 0,
        "date_range": None,
        "error": None,
    }

    sym_dir = _sym_dir(sym, is_crypto)
    legacy_files = sorted([f for f in sym_dir.glob("*.parquet") if f.name != "daily.parquet"])

    if not legacy_files:
        result["status"] = "skipped"
        return result

    try:
        dfs = []
        for legacy_file in legacy_files:
            try:
                df = pd.read_parquet(legacy_file)
                if isinstance(df.index, pd.MultiIndex):
                    df = df.droplevel(0) if "symbol" in df.index.names else df
                dfs.append(df)
            except Exception as e:
                logger.warning(f"{sym}: Failed to read {legacy_file.name}: {e}")

        if not dfs:
            result["status"] = "failed"
            result["error"] = "No files could be read"
            return result

        # Merge all dataframes, handling duplicates
        df_merged = pd.concat(dfs, axis=0).sort_index()
        df_merged = df_merged[~df_merged.index.duplicated(keep="last")]

        result["total_rows"] = len(df_merged)
        result["files_merged"] = len(dfs)
        result["date_range"] = (
            f"{str(df_merged.index.min())[:10]} to {str(df_merged.index.max())[:10]}"
        )

        if dry_run:
            result["status"] = "dry_run_ok"
            return result

        # Write canonical file
        canon_path = _canonical_path(sym, is_crypto)
        if _atomic_write(canon_path, df_merged):
            result["status"] = "success"
            # Register in catalogue
            cat = _get_catalogue()
            if cat:
                _register(
                    cat,
                    sym,
                    is_crypto,
                    str(df_merged.index.min())[:10],
                    str(df_merged.index.max())[:10],
                    len(df_merged),
                    str(canon_path),
                )
        else:
            result["status"] = "write_failed"
            result["error"] = "Atomic write failed"

        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


def run_migration(dry_run: bool = False, single_symbol: str = None):
    """Run migration for all symbols."""
    print(f"\n{'=' * 70}")
    print(f"  MIGRATE TO CANONICAL FORMAT  {('(DRY RUN)' if dry_run else '')}")
    print(f"{'=' * 70}\n")

    results = []

    # Migrate equities
    print("Scanning STOCKS/ETFs...")
    symbols_eq = (
        [single_symbol]
        if single_symbol and not single_symbol.startswith("BTC")
        else get_all_symbols(is_crypto=False)
    )

    if symbols_eq:
        print(f"Found {len(symbols_eq)} symbols with legacy files\n")
        for i, sym in enumerate(symbols_eq, 1):
            result = migrate_symbol(sym, is_crypto=False, dry_run=dry_run)
            results.append(result)

            status_str = result["status"].upper()
            if result["status"] == "success":
                status_str = f"✓ {status_str}"
            elif result["status"] == "dry_run_ok":
                status_str = f"⊘ {status_str}"
            else:
                status_str = f"✗ {status_str}"

            rows_str = f"({result['total_rows']:,} rows)" if result["total_rows"] else ""
            date_range_str = f" [{result['date_range']}]" if result["date_range"] else ""

            print(f"  {i:4d}. {sym:10s} {status_str:20s} {rows_str:20s}{date_range_str}")

    # Migrate crypto
    print("\nScanning CRYPTO...")
    symbols_cr = (
        [single_symbol]
        if single_symbol and single_symbol.startswith(("BTC", "ETH", "SOL"))
        else get_all_symbols(is_crypto=True)
    )

    if symbols_cr:
        print(f"Found {len(symbols_cr)} crypto symbols with legacy files\n")
        for i, sym in enumerate(symbols_cr, 1):
            result = migrate_symbol(sym, is_crypto=True, dry_run=dry_run)
            results.append(result)

            status_str = result["status"].upper()
            if result["status"] == "success":
                status_str = f"✓ {status_str}"
            elif result["status"] == "dry_run_ok":
                status_str = f"⊘ {status_str}"
            else:
                status_str = f"✗ {status_str}"

            rows_str = f"({result['total_rows']:,} rows)" if result["total_rows"] else ""
            date_range_str = f" [{result['date_range']}]" if result["date_range"] else ""

            print(f"  {i:4d}. {sym:10s} {status_str:20s} {rows_str:20s}{date_range_str}")

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    dry_ok = sum(1 for r in results if r["status"] == "dry_run_ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] in ("failed", "error", "write_failed"))

    print(f"\n{'=' * 70}")
    print("  MIGRATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Successful:    {success}")
    if dry_run:
        print(f"  Ready to go:   {dry_ok}")
    print(f"  Skipped:       {skipped}")
    print(f"  Failed:        {failed}")
    print(f"  Mode:          {'DRY RUN' if dry_run else 'ACTIVE'}")
    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate legacy data files to canonical format")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--symbol", type=str, help="Migrate single symbol only")
    args = parser.parse_args()

    results = run_migration(dry_run=args.dry_run, single_symbol=args.symbol)

    # Exit with error code if any failures
    if any(r["status"] in ("error", "write_failed", "failed") for r in results):
        sys.exit(1)
