"""
Fix Winter DST Empty Stubs
===========================
The XNAS.ITCH imbalance schema uses a closing auction window of:
  Summer (EDT, UTC-4): 3:50–4:01 PM ET = 19:50–21:01 UTC
  Winter (EST, UTC-5): 3:50–4:01 PM ET = 20:50–21:01 UTC

All empty stubs fetched on winter dates with the summer window (19:50 UTC)
returned v={} because there is no imbalance data at 19:50 UTC in winter —
the closing auction happens an hour later (20:50 UTC).

Databento confirmed this: shift the window +1 hour in winter.

This script:
  1. Identifies all empty stubs on winter dates (after DST ends each year)
  2. Deletes them so validate_databento_signals.py re-fetches with the
     correct DST-aware window (already fixed in databento_imbalance.py)
  3. Reports estimated cost to re-fetch

Usage:
    PYTHONPATH=. python diagnostics/fix_winter_dst_stubs.py --dry-run
    PYTHONPATH=. python diagnostics/fix_winter_dst_stubs.py --delete
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "databento"

# DST ends on first Sunday of November each year (US)
DST_END = {
    2023: date(2023, 11, 5),
    2024: date(2024, 11, 3),
    2025: date(2025, 11, 2),
    2026: date(2026, 11, 1),
}

date_pat = re.compile(r"_(\d{4}-\d{2}-\d{2})_")


def is_winter_date(d: date) -> bool:
    """True if date falls in winter (EST, UTC-5) period."""
    dst_end = DST_END.get(d.year)
    # DST resumes second Sunday of March each year (approx Mar 8-14)
    dst_start_approx = date(d.year, 3, 14)  # safe upper bound for DST start
    if dst_end and dst_start_approx:
        return d >= dst_end or d < dst_start_approx
    return False


def find_winter_stubs():
    """Find all empty stubs (v={}) on winter dates."""
    if not CACHE_DIR.exists():
        print(f"Cache dir not found: {CACHE_DIR}")
        return []
    stubs = []
    for f in sorted(CACHE_DIR.glob("imbalance_*.json")):
        if f.stat().st_size >= 100:
            continue  # real data file
        m = date_pat.search(f.stem)
        if not m:
            continue
        try:
            d = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if is_winter_date(d):
            stubs.append((d, f))
    return sorted(stubs)


def main():
    parser = argparse.ArgumentParser(description="Fix winter DST empty stubs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without deleting"
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete the stubs")
    args = parser.parse_args()

    if not args.dry_run and not args.delete:
        parser.print_help()
        sys.exit(0)

    stubs = find_winter_stubs()

    print()
    print("=" * 62)
    print("  WINTER DST STUB CLEANUP")
    print("=" * 62)
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Winter stubs (wrong UTC window, v={{}}): {len(stubs)}")
    print()

    if not stubs:
        print("  ✅ No winter stubs found — nothing to do")
        return

    # Group by year for display
    by_year: dict = {}
    for d, f in stubs:
        by_year.setdefault(d.year, []).append((d, f))

    for yr, items in sorted(by_year.items()):
        dst_end = DST_END.get(yr, "?")
        print(f"  {yr} (DST ended {dst_end}): {len(items)} stubs")
        for d, f in items[:5]:
            print(f"    {d}  {f.name[:55]}")
        if len(items) > 5:
            print(f"    ... and {len(items) - 5} more")
        print()

    # Cost estimate for re-fetch
    n = len(stubs)
    est_gb = n * 20 * 0.0016  # 20 symbols × 1.6MB per symbol per day
    est_cost = est_gb * 16.0
    print(
        f"  Re-fetch cost estimate: {n} dates × 20 syms × 1.6MB = {est_gb * 1024:.0f} MB ≈ ${est_cost:.2f}"
    )
    print()

    if args.dry_run:
        print("  DRY RUN — no files deleted")
        print(f"  Run with --delete to remove {len(stubs)} stubs")
    elif args.delete:
        deleted = 0
        for d, f in stubs:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"  Could not delete {f.name}: {e}")
        print(f"  ✅ Deleted {deleted}/{len(stubs)} winter stubs")
        print()
        print("  Next step: re-run validate_databento_signals.py")
        print("  The DST-aware window is already correct in databento_imbalance.py.")
        print(f"  Estimated re-fetch: ${est_cost:.2f} and ~{n * 50 // 60}–{n * 70 // 60} min")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
