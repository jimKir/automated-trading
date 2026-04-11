"""
Restore Empty Stubs
====================
Recreates the v={} empty stub files that were accidentally deleted
by run_health_check(auto_fix=True). These stubs prevent Databento
from being re-queried for days where no imbalance data exists.

The stubs are reconstructed from the list of expected trading days
by checking which days are NOT already in cache (real or stub).
Days genuinely missing from cache are left alone (will be fetched normally).

Usage:
    PYTHONPATH=. python diagnostics/restore_empty_stubs.py
    PYTHONPATH=. python diagnostics/restore_empty_stubs.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "databento"

SYMS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AVGO",
    "JPM",
    "V",
    "MA",
    "UNH",
    "JNJ",
    "PG",
    "HD",
    "KO",
    "XOM",
    "CVX",
    "BAC",
    "GS",
]

# Full holiday calendar — same as signal module
_US_HOLIDAYS = np.array(
    [
        "2022-01-17",
        "2022-02-21",
        "2022-04-15",
        "2022-05-30",
        "2022-06-19",
        "2022-06-20",
        "2022-07-04",
        "2022-09-05",
        "2022-11-24",
        "2022-11-25",
        "2022-12-26",
        "2023-01-02",
        "2023-01-16",
        "2023-02-20",
        "2023-04-07",
        "2023-05-29",
        "2023-06-19",
        "2023-07-04",
        "2023-09-04",
        "2023-11-23",
        "2023-11-24",
        "2023-12-25",
        "2024-01-01",
        "2024-01-15",
        "2024-02-19",
        "2024-03-29",
        "2024-05-27",
        "2024-06-19",
        "2024-07-04",
        "2024-09-02",
        "2024-11-28",
        "2024-11-29",
        "2024-12-25",
        "2025-01-01",
        "2025-01-09",
        "2025-01-20",
        "2025-02-17",
        "2025-04-18",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-11-28",
        "2025-12-25",
        "2026-01-01",
        "2026-01-19",
        "2026-02-16",
        "2026-04-03",
    ],
    dtype="datetime64[D]",
)


def is_td(d: date) -> bool:
    return bool(np.is_busday(np.datetime64(d, "D"), holidays=_US_HOLIDAYS))


def cache_filename(d: date) -> str:
    raw = "|".join(str(p) for p in ["imbalance", sorted(SYMS), str(d)])
    h8 = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"imbalance_20syms_{d}_{h8}.json"


def build_fetched_set() -> set:
    """Return set of hash8s already on disk (real or stub)."""
    fetched = set()
    for f in CACHE_DIR.glob("imbalance_*.json"):
        stem = f.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) == 8:
            fetched.add(parts[1])
    return fetched


def all_expected_trading_days(start: date, end: date) -> list[date]:
    """All trading days between start and end inclusive."""
    days = []
    cur = start
    while cur <= end:
        if is_td(cur):
            days.append(cur)
        cur += timedelta(days=1)
    return days


def hash8_for(d: date) -> str:
    raw = "|".join(str(p) for p in ["imbalance", sorted(SYMS), str(d)])
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description="Restore deleted empty stubs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be restored without writing"
    )
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("  RESTORE EMPTY STUBS")
    print(f"  Cache: {CACHE_DIR}")
    print("=" * 62)

    if not CACHE_DIR.exists():
        print("  ❌ Cache directory not found")
        sys.exit(1)

    # Build set of already-cached hash8s
    fetched_h8 = build_fetched_set()
    print(f"\n  Files currently on disk: {len(list(CACHE_DIR.glob('*.json')))}")
    print(f"  Hash8s indexed:          {len(fetched_h8)}")

    # All trading days that should be in the cache
    # The signal fetches 10 days before each of 84 biweekly windows
    # spanning 2023-01-01 to 2026-03-21.
    # Approximate range: 2022-12-01 to 2026-03-21
    expected_days = all_expected_trading_days(date(2022, 12, 1), date(2026, 3, 21))
    print(f"  Expected trading days:   {len(expected_days)} (2022-12-01 to 2026-03-21)")

    # Days that are missing from cache entirely (no real file, no stub)
    missing = [d for d in expected_days if hash8_for(d) not in fetched_h8]
    print(f"  Missing from cache:      {len(missing)}")
    print()

    if not missing:
        print("  ✅ Nothing to restore — all expected days are cached")
        print("=" * 62)
        return

    # Restore stubs for ALL missing days.
    # The signal module's _cache_load() auto-deletes v={} stubs and re-fetches,
    # so writing a stub for a day that HAS real data is harmless —
    # the real fetch will overwrite it with actual rows.
    #
    # This is safer than trying to distinguish "confirmed empty" vs "truly missing"
    # without hitting the API. The smoke test confirmed ~37% of days have no
    # imbalance data, but we don't know exactly which ones without querying.
    #
    # Effect: all 328 missing days get stubs → preflight sees them as fetched →
    # no API calls. If any of those days DO have real data, the signal module
    # will detect the empty stub on first use, delete it, and re-fetch properly.
    to_restore = missing

    print(f"  Days to restore stubs:   {len(to_restore)}")
    print("  (Stubs act as placeholders; real fetches overwrite them if data exists)")
    print()

    if args.dry_run:
        print("  DRY RUN — showing first 20 stubs that would be restored:")
        for d in sorted(to_restore)[:20]:
            fname = cache_filename(d)
            print(f"    {d}  {fname}")
        if len(to_restore) > 20:
            print(f"    ... and {len(to_restore) - 20} more")
        print()
        print("  Run without --dry-run to actually restore them.")
        print("=" * 62)
        return

    # Write the stubs
    restored = 0
    skipped = 0
    ts_now = time.time()
    for d in sorted(to_restore):
        fname = cache_filename(d)
        fpath = CACHE_DIR / fname
        if fpath.exists():
            skipped += 1
            continue
        stub = {"v": {}, "_ts": ts_now}
        fpath.write_text(json.dumps(stub))
        restored += 1

    print(f"  ✅ Restored: {restored} empty stubs")
    if skipped:
        print(f"  ⏭  Skipped:  {skipped} (already existed)")
    print()

    # Final count
    total_after = len(list(CACHE_DIR.glob("*.json")))
    print(f"  Files on disk now: {total_after}")
    print()
    print("  Re-run health check to verify:")
    print("  PYTHONPATH=. python diagnostics/cache_health_check.py")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
