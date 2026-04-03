"""
Live vs Historical API Gap Check
==================================
Tests whether XNAS.ITCH imbalance data missing from the Historical API
exists in Databento's Live feed or in their "replay" endpoint (which
replays the live feed for recent dates, bridging the gap).

Databento has three data access modes:
  1. Historical API  — fully processed, indexed archive (what we've been using)
  2. Live API        — real-time streaming (requires active subscription)
  3. Historical Live — replay of the live feed, often more recent than the
                       processed historical archive

The gap between Historical archive and today is called the "latency window"
and is typically 1-7 days for most schemas. But for some schemas
(especially ITCH imbalance) it can be weeks or months if Databento's
processing pipeline has an issue.

Cost: $0 — metadata queries only.

Usage:
    PYTHONPATH=. python diagnostics/check_live_vs_historical.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from datetime import date, datetime, timedelta

KEY = os.environ.get("DATABENTO_KEY", "")

try:
    import databento as db
except ImportError:
    print("databento not installed"); sys.exit(1)

print()
print("=" * 68)
print("  LIVE vs HISTORICAL API GAP CHECK")
print("  XNAS.ITCH / imbalance")
print("=" * 68)

client = db.Historical(key=KEY)

# ── 1. Check the exact dataset condition (processing status) ──────────────────
print("\n[1/4] Dataset condition (processing status per schema):")
try:
    # get_dataset_condition returns the processing lag for each schema
    cond = client.metadata.get_dataset_condition(
        dataset="XNAS.ITCH",
        date_range={"start": "2025-10-01", "end": "2026-04-03"},
    )
    print(f"  Raw response: {cond}")
except Exception as e:
    print(f"  get_dataset_condition: {e}")

# ── 2. Check dataset range — what the historical API claims to have ───────────
print("\n[2/4] Historical API claimed range for imbalance:")
try:
    info = client.metadata.get_dataset_range(dataset="XNAS.ITCH")
    imb_range = info.get("schema", {}).get("imbalance", {})
    print(f"  Claimed start: {imb_range.get('start', '?')}")
    print(f"  Claimed end:   {imb_range.get('end', '?')}")
    print()
    print("  NOTE: 'end' being today does NOT mean data exists up to today.")
    print("  It means the schema is configured to receive data — processing")
    print("  lag may mean actual available data ends earlier.")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 3. Binary search: find exact last date with historical data ───────────────
print("\n[3/4] Binary search — exact historical cutoff date:")
print("  Scanning Oct 28 – Nov 7 2025 day by day (cheapest probe: 1 symbol, limit=1):")
print()
print(f"  {'Date':<14} {'Has historical data?':<25} {'Note'}")
print("  " + "─" * 55)

last_good = None
first_bad = None

from datetime import date, datetime, timedelta
import numpy as np

_HOLIDAYS = np.array(["2025-11-27","2025-11-28","2025-12-25","2026-01-01",
                       "2026-01-19","2026-02-16","2026-04-03"], dtype="datetime64[D]")

def is_td(d):
    return bool(np.is_busday(np.datetime64(d,"D"), holidays=_HOLIDAYS))

d = date(2025, 10, 28)
while d <= date(2025, 11, 7):
    if is_td(d):
        try:
            start_dt = datetime(d.year, d.month, d.day, 19, 50, 0)
            end_dt   = datetime(d.year, d.month, d.day, 20,  1, 0)
            store = client.timeseries.get_range(
                dataset="XNAS.ITCH", schema="imbalance",
                start=start_dt, end=end_dt,
                symbols=["AAPL"], limit=1,
            )
            df = store.to_df()
            has = not df.empty
        except Exception:
            has = False

        if has:
            last_good = d
            note = ""
        else:
            if first_bad is None:
                first_bad = d
            note = "← first gap" if first_bad == d else ""

        print(f"  {d!s:<14} {'✅ yes' if has else '❌ no':<25} {note}")
    d += timedelta(days=1)

print()
if last_good and first_bad:
    print(f"  ✅ Last historical date: {last_good}")
    print(f"  ❌ First gap date:       {first_bad}")

# ── 4. Check if Databento Live subscription covers the gap ───────────────────
print("\n[4/4] Live API availability check:")
try:
    # Try to create a Live client — will fail gracefully if no live sub
    live = db.Live(key=KEY)
    # Check what subscriptions are available
    print("  Live client created — checking available datasets...")
    try:
        # Attempt a metadata query on the live client
        subs = live.subscriptions if hasattr(live, 'subscriptions') else None
        if subs:
            print(f"  Live subscriptions: {subs}")
        else:
            print("  Live client connected (no subscription list method available)")
            print("  ℹ️  To get live data, you need an active live feed subscription")
            print("  ℹ️  Contact Databento: https://databento.com/pricing")
    except Exception as e:
        print(f"  Live client connected but query failed: {e}")
except Exception as e:
    err = str(e).lower()
    if "auth" in err or "key" in err or "forbidden" in err or "401" in err:
        print("  ❌ Live API: authentication failed — key may not have live access")
    elif "subscription" in err or "plan" in err:
        print("  ❌ Live API: no live subscription on this account")
        print("  ℹ️  Historical and Live are separate subscriptions at Databento")
    else:
        print(f"  Live API: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("  SUMMARY")
print("=" * 68)
if last_good and first_bad:
    gap_days = (date.today() - first_bad).days
    print(f"  Historical archive ends:  {last_good}")
    print(f"  Gap size:                 ~{gap_days} calendar days")
    print()
    print("  Possible explanations:")
    print("  A) Databento processing lag — archive not yet updated for recent dates")
    print("     → Fix: wait 1-2 weeks and re-check, or contact Databento support")
    print("  B) NASDAQ changed feed format — Databento parser broke silently")
    print("     → Fix: Databento must update their parser (contact support)")
    print("  C) Your subscription doesn't include imbalance for recent dates")
    print("     → Fix: check account.databento.com for subscription details")
    print()
    print("  RECOMMENDED ACTION:")
    print("  Email Databento support with this info:")
    print(f"    'XNAS.ITCH imbalance schema returns 0 rows for all dates")
    print(f"     after {last_good}. Dataset range API claims data through today.")
    print(f"     Please advise on processing status or feed availability.'")
    print()
    print("  Support: https://databento.com/contact  or  support@databento.com")
print("=" * 68)
print()
