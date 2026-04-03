"""
Databento Catalog Check
========================
Queries the Databento API catalog to find:
1. What schemas are available for XNAS.ITCH after 2025-10-31
2. Whether imbalance data exists under any schema post Oct 2025
3. Any alternative datasets that carry closing imbalance

Zero cost — metadata queries are free.

Usage:
    PYTHONPATH=. python diagnostics/check_databento_catalog.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from datetime import date, datetime
from pathlib import Path

KEY = os.environ.get("DATABENTO_KEY", "db-SpVxiQLLTdDe9iD3sLwTpiqgBjtxk")

try:
    import databento as db
except ImportError:
    print("databento not installed"); sys.exit(1)

client = db.Historical(key=KEY)

CUTOFF     = date(2025, 10, 31)
TEST_SYMS  = ["AAPL", "MSFT", "NVDA"]

print()
print("=" * 68)
print("  DATABENTO CATALOG CHECK")
print(f"  Looking for imbalance data after {CUTOFF}")
print("=" * 68)

# ── 1. List all available datasets ───────────────────────────────────────────
print("\n[1/4] Available datasets:")
try:
    datasets = client.metadata.list_datasets()
    for ds in sorted(datasets):
        print(f"  {ds}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 2. List all schemas for XNAS.ITCH ────────────────────────────────────────
print("\n[2/4] Schemas available on XNAS.ITCH:")
try:
    schemas = client.metadata.list_schemas(dataset="XNAS.ITCH")
    for s in sorted(schemas):
        print(f"  {s}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 3. Check data availability for XNAS.ITCH imbalance post-cutoff ───────────
print(f"\n[3/4] XNAS.ITCH imbalance availability after {CUTOFF}:")
test_dates = [
    date(2025, 11,  3),
    date(2025, 11, 14),
    date(2025, 12,  1),
    date(2026,  1,  5),
    date(2026,  2,  2),
    date(2026,  3,  2),
]
for d in test_dates:
    try:
        # Use get_range with limit=1 to check existence cheaply
        start = datetime(d.year, d.month, d.day, 19, 50, 0)
        end   = datetime(d.year, d.month, d.day, 20,  1, 0)
        store = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            schema="imbalance",
            start=start, end=end,
            symbols=TEST_SYMS,
            limit=1,
        )
        df = store.to_df()
        status = f"✅ {len(df)} rows" if not df.empty else "❌ 0 rows"
    except Exception as e:
        status = f"❌ ERROR: {str(e)[:60]}"
    print(f"  {d}  {status}")

# ── 4. Check alternative datasets that might carry imbalance ─────────────────
print(f"\n[4/4] Alternative datasets — imbalance schema probe:")
alt_datasets = ["DBEQ.BASIC", "DBEQ.PLUS", "FINN.NLS", "XNYS.TRADES",
                "EQUS.MINI", "EQUS.SUMMARY"]
for ds in alt_datasets:
    try:
        schemas = client.metadata.list_schemas(dataset=ds)
        has_imb = "imbalance" in schemas
        has_stat = "statistics" in schemas
        markers = []
        if has_imb:  markers.append("imbalance ✅")
        if has_stat: markers.append("statistics ✅")
        if markers:
            print(f"  {ds:<20} {', '.join(markers)}")
        else:
            print(f"  {ds:<20} no imbalance/statistics schemas")
    except Exception as e:
        print(f"  {ds:<20} unavailable: {str(e)[:50]}")

# ── 5. Check XNAS.ITCH imbalance date range from catalog ─────────────────────
print(f"\n[5/4] XNAS.ITCH imbalance date range from catalog:")
try:
    info = client.metadata.get_dataset_range(dataset="XNAS.ITCH")
    print(f"  Dataset range: {info}")
except Exception as e:
    try:
        # Alternative metadata call
        cond = client.metadata.get_dataset_condition(
            dataset="XNAS.ITCH",
            date_range={"start": "2025-10-01", "end": "2026-04-01"},
        )
        print(f"  Condition: {cond}")
    except Exception as e2:
        print(f"  ERROR: {e2}")

print()
print("=" * 68)
print()
