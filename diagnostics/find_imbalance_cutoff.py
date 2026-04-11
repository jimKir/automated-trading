"""
Find Imbalance Cutoff Date
===========================
Binary-searches the exact last trading day where XNAS.ITCH imbalance
returns real data. Uses limit=1 probes (free metadata-level cost).

Usage:
    PYTHONPATH=. python diagnostics/find_imbalance_cutoff.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from datetime import date, datetime, timedelta

import numpy as np

KEY = os.environ.get("DATABENTO_KEY", "")

try:
    import databento as db
except ImportError:
    print("databento not installed")
    sys.exit(1)

client = db.Historical(key=KEY)

_US_HOLIDAYS = np.array(
    [
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


def prev_td(d: date) -> date:
    d -= timedelta(days=1)
    while not is_td(d):
        d -= timedelta(days=1)
    return d


def has_data(d: date) -> bool:
    """True if XNAS.ITCH imbalance returns >=1 row for date d."""
    try:
        start = datetime(d.year, d.month, d.day, 19, 50, 0)
        end = datetime(d.year, d.month, d.day, 20, 1, 0)
        store = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            schema="imbalance",
            start=start,
            end=end,
            symbols=["AAPL"],  # single symbol — cheapest possible probe
            limit=1,
        )
        df = store.to_df()
        return not df.empty
    except Exception:
        return False


print()
print("=" * 55)
print("  IMBALANCE CUTOFF — binary search")
print("=" * 55)

# We know: 2025-10-31 has data, 2025-11-03 does not.
# Narrow down day-by-day in that window first.
print("\n[1/2] Scanning Oct-Nov 2025 boundary day by day:")
print(f"  {'Date':<14} {'Has data?'}")
print("  " + "─" * 28)

# Scan every trading day 2025-10-20 to 2025-11-14
d = date(2025, 10, 20)
last_good = None
first_bad = None
while d <= date(2025, 11, 14):
    if is_td(d):
        result = has_data(d)
        flag = "✅ yes" if result else "❌ no"
        print(f"  {d!s:<14} {flag}")
        if result:
            last_good = d
        elif first_bad is None:
            first_bad = d
    d += timedelta(days=1)

print()
print("=" * 55)
print("  RESULT")
print("=" * 55)
if last_good:
    print(f"  Last date WITH imbalance data:    {last_good}")
if first_bad:
    print(f"  First date WITHOUT imbalance data: {first_bad}")
if last_good and first_bad:
    print(f"  Gap starts: {first_bad} (day after {last_good})")
    print()
    # Check if the gap is total or intermittent
    print("[2/2] Spot-checking later dates for any recovery:")
    spot = [date(2025, 12, 1), date(2026, 1, 5), date(2026, 2, 2), date(2026, 3, 2)]
    for sd in spot:
        r = has_data(sd)
        print(f"  {sd!s:<14} {'✅ yes' if r else '❌ no'}")
    print()
    print("  If all later dates also show ❌: Databento stopped receiving")
    print("  the NASDAQ closing imbalance feed after that date.")
    print("  Options:")
    print("    A) Use the signal as-is (0.0 for post-cutoff dates)")
    print("    B) Reconstruct imbalance proxy from XNAS.ITCH trades schema")
    print("       (compute order flow imbalance from last 10 min of trading)")
    print("    C) Wait for Databento to backfill (contact their support)")
print("=" * 55)
print()
