"""
Cache Health Check
==================
Non-destructive, zero-API-calls diagnostic.
Checks the .cache/databento directory and reports exactly what's covered,
what's missing, and what the next run will cost.

Usage:
    PYTHONPATH=. python diagnostics/cache_health_check.py

No data is fetched, deleted, or modified.
"""
import hashlib
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SYMS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO",
    "JPM","V","MA","UNH","JNJ","PG","HD","KO","XOM","CVX","BAC","GS",
]
OOS_START = "2023-01-01"
OOS_END   = "2026-03-21"

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "databento"

_US_HOLIDAYS = np.array([
    "2022-01-17","2022-02-21","2022-04-15","2022-05-30","2022-06-19","2022-06-20",
    "2022-07-04","2022-09-05","2022-11-24","2022-11-25","2022-12-26",
    "2023-01-02","2023-01-16","2023-02-20","2023-04-07","2023-05-29","2023-06-19",
    "2023-07-04","2023-09-04","2023-11-23","2023-11-24","2023-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27","2024-06-19",
    "2024-07-04","2024-09-02","2024-11-28","2024-11-29","2024-12-25",
    "2025-01-01","2025-01-09","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-11-28","2025-12-25",
    "2026-01-01","2026-01-19","2026-02-16","2026-04-03",
], dtype="datetime64[D]")


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_td(d: date) -> bool:
    return bool(np.is_busday(np.datetime64(d, "D"), holidays=_US_HOLIDAYS))

def hash8(schema: str, syms: list, d: date) -> str:
    raw = "|".join(str(p) for p in [schema, sorted(syms), str(d)])
    return hashlib.md5(raw.encode()).hexdigest()[:8]

def full_md5(schema: str, syms: list, d: date) -> str:
    raw = "|".join(str(p) for p in [schema, sorted(syms), str(d)])
    return hashlib.md5(raw.encode()).hexdigest()

def guard_window(sd: date, n: int = 10) -> list:
    """10 preceding weekdays (what preflight checks)"""
    days, cur = [], sd - timedelta(days=1)
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= timedelta(days=1)
    return days

def signal_window(sd: date, n: int = 10) -> list:
    """10 preceding REAL trading days (what signal fetches)"""
    days, cur = [], sd - timedelta(days=1)
    while len(days) < n:
        if is_td(cur):
            days.append(cur)
        cur -= timedelta(days=1)
    return days


# ── Scan cache directory ────────────────────────────────────────────────────────

def scan_cache(schema: str = "imbalance") -> tuple:
    """
    Returns (valid_hash8s, valid_full_md5s, all_files_info)
    valid_hash8s: set of 8-char hashes found in cache (real data only)
    valid_full_md5s: set of 32-char md5s (old format)
    """
    if not CACHE_DIR.exists():
        print(f"  ❌ Cache directory not found: {CACHE_DIR}")
        sys.exit(1)

    valid_h8   = set()
    valid_md5  = set()
    total      = 0
    real        = 0
    empty       = 0
    corrupt     = 0
    old_format  = 0
    new_format  = 0
    total_bytes = 0

    for f in CACHE_DIR.glob("*.json"):
        if f.name.startswith(".") or f.name == "catalogue.json":
            continue
        total += 1
        total_bytes += f.stat().st_size

        # Classify format
        stem = f.stem
        is_old = len(stem) == 32 and all(c in "0123456789abcdef" for c in stem)

        if is_old:
            old_format += 1
        else:
            new_format += 1

        # Validate content
        if f.stat().st_size < 100:
            empty += 1
            continue

        try:
            data = json.loads(f.read_text())
            v = data.get("v", {})
            if not isinstance(v, dict) or len(v) == 0:
                empty += 1
                continue
            real += 1
            # Register in lookup
            if is_old:
                valid_md5.add(stem)
                valid_h8.add(stem[:8])
            else:
                parts = stem.rsplit("_", 1)
                if len(parts) == 2 and len(parts[1]) == 8:
                    valid_h8.add(parts[1])
        except Exception:
            corrupt += 1

    return valid_h8, valid_md5, {
        "total": total, "real": real, "empty": empty, "corrupt": corrupt,
        "old_format": old_format, "new_format": new_format,
        "total_mb": total_bytes / 1024 / 1024,
    }


# ── Coverage analysis ──────────────────────────────────────────────────────────

def analyse_coverage(valid_h8: set, valid_md5: set, schema: str = "imbalance") -> dict:
    """
    For each biweekly step_date, count how many of its 10-day guard window
    hash8s are present in the cache. Report cached/missing/tight.
    """
    week_idx   = pd.date_range(OOS_START, OOS_END, freq="W-SUN")
    step_dates = [d.date() for i, d in enumerate(week_idx) if i % 2 == 0]

    def is_cached(d: date) -> bool:
        h8  = hash8(schema, SYMS, d)
        md5 = full_md5(schema, SYMS, d)
        return h8 in valid_h8 or md5 in valid_md5

    cached_dates  = []
    missing_dates = []
    coverage_detail = []

    for sd in step_dates:
        gw    = guard_window(sd)
        sw    = signal_window(sd)
        phantom = [d for d in gw if not is_td(d)]

        hits          = sum(1 for d in gw if is_cached(d))
        real_possible = 10 - len(phantom)   # max possible score (some days = holidays)
        passes        = hits >= 8

        coverage_detail.append({
            "date":         sd,
            "hits":         hits,
            "max_possible": real_possible,
            "phantom_days": len(phantom),
            "passes":       passes,
        })

        if passes:
            cached_dates.append(sd)
        else:
            missing_dates.append(sd)

    return {
        "step_dates":   step_dates,
        "cached":       cached_dates,
        "missing":      missing_dates,
        "detail":       coverage_detail,
    }


# ── Format helpers ─────────────────────────────────────────────────────────────

def _bar(n, total, width=30) -> str:
    filled = int(width * n / total) if total else 0
    return "█" * filled + "░" * (width - filled)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 68)
    print("  CACHE HEALTH CHECK  —  read-only, zero API calls")
    print(f"  Cache: {CACHE_DIR}")
    print("=" * 68)

    # 1. File inventory
    print("\n[1/4] File inventory")
    print("  " + "─" * 40)
    valid_h8, valid_md5, info = scan_cache("imbalance")

    bar_real = _bar(info["real"], info["total"])
    print(f"  Total files:     {info['total']:>5}")
    print(f"  Real data:       {info['real']:>5}  [{bar_real}]  {info['real']/max(info['total'],1)*100:.0f}%")
    print(f"  Empty/v={{}}:      {info['empty']:>5}")
    print(f"  Corrupt:         {info['corrupt']:>5}")
    print(f"  Old MD5 format:  {info['old_format']:>5}  (pure 32-char hash filename)")
    print(f"  New format:      {info['new_format']:>5}  (human-readable filename)")
    print(f"  Total size:      {info['total_mb']:.1f} MB")

    if info["old_format"] > 0:
        print(f"\n  ⚠️  {info['old_format']} files still in old MD5 format.")
        print(f"     Run: PYTHONPATH=. python src/market_data/cache_guard.py --check")
        print(f"     to rename them to human-readable format.")

    # 2. Coverage analysis
    print("\n[2/4] Window coverage  (biweekly OOS dates {OOS_START} → {OOS_END})".format(**globals()))
    print("  " + "─" * 40)
    cov = analyse_coverage(valid_h8, valid_md5, "imbalance")

    n_cached  = len(cov["cached"])
    n_missing = len(cov["missing"])
    n_total   = len(cov["step_dates"])
    bar_cov   = _bar(n_cached, n_total)

    print(f"  Total windows:   {n_total:>3}")
    print(f"  ✅ Cached:        {n_cached:>3}  [{bar_cov}]")
    print(f"  🌐 Need fetch:    {n_missing:>3}")

    # 3. Missing date detail
    if cov["missing"]:
        print(f"\n[3/4] Missing windows (will be fetched by next run):")
        print("  " + "─" * 40)

        # Group by year for readability
        by_year: dict = {}
        for d in cov["missing"]:
            by_year.setdefault(d.year, []).append(d)

        for yr, dates in sorted(by_year.items()):
            ds = "  ".join(str(d) for d in sorted(dates))
            print(f"  {yr}: {ds}")

        # Cost estimate
        n_miss    = len(cov["missing"])
        est_gb    = n_miss * len(SYMS) * 0.0016
        est_cost  = est_gb * 16.0
        est_min   = n_miss * 50 // 60
        est_max   = n_miss * 70 // 60

        print(f"\n  Cost estimate:")
        print(f"    Dates to fetch:  {n_miss}")
        print(f"    Est. data:       {est_gb*1024:.0f} MB")
        print(f"    Est. cost:       ${est_cost:.2f} USD  (XNAS.ITCH imbalance @ $16/GB)")
        print(f"    Est. time:       {est_min}–{est_max} min")
    else:
        print(f"\n[3/4] ✅ All windows cached — next run will be instant ($0.00)")

    # 4. Full file audit — every file checked
    print(f"\n[4/4] Full file audit ({info['total']} files):")
    print("  " + "─" * 40)

    from datetime import datetime

    issues = []
    size_buckets   = {">100KB": 0, "10-100KB": 0, "1-10KB": 0, "<1KB": 0}
    row_counts     = []
    age_days_list  = []
    name_ok        = 0
    name_bad       = 0
    hash_ok        = 0
    hash_bad       = 0

    for f in CACHE_DIR.glob("*.json"):
        if f.name.startswith(".") or f.name == "catalogue.json":
            continue

        sz  = f.stat().st_size
        stem = f.stem

        # ── Size bucket ──────────────────────────────────────────────
        if sz > 100_000:
            size_buckets[">100KB"] += 1
        elif sz > 10_000:
            size_buckets["10-100KB"] += 1
        elif sz > 1_000:
            size_buckets["1-10KB"] += 1
        else:
            size_buckets["<1KB"] += 1

        if sz < 100:
            issues.append((f.name, "EMPTY", f"size={sz}B"))
            continue

        # ── JSON parse ───────────────────────────────────────────────
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            issues.append((f.name, "CORRUPT", f"json parse: {e}"))
            continue

        v  = data.get("v", {})
        ts = data.get("_ts", 0)

        if not isinstance(v, dict):
            issues.append((f.name, "BAD_STRUCTURE", f"v is {type(v).__name__}"))
            continue

        if len(v) == 0:
            issues.append((f.name, "EMPTY_V", "v={} — no data rows"))
            continue

        row_counts.append(len(v))

        if ts:
            age = (datetime.now().timestamp() - ts) / 86400
            age_days_list.append(age)

        # ── Filename format check ────────────────────────────────────
        is_old_format = len(stem) == 32 and all(c in "0123456789abcdef" for c in stem)
        parts = stem.rsplit("_", 1)
        is_new_format = (not is_old_format and len(parts) == 2 and len(parts[1]) == 8
                         and all(c in "0123456789abcdef" for c in parts[1]))

        if is_old_format or is_new_format:
            name_ok += 1
        else:
            name_bad += 1
            issues.append((f.name, "BAD_FILENAME", "not old-MD5 nor new-readable format"))
            continue

        # ── Hash-content consistency check ───────────────────────────
        # The hash in the filename should match the MD5 of the canonical key.
        # We can't reverse-engineer the exact key (we don't know the date),
        # but for new-format files we can verify the hash8 is plausibly hex.
        # For old-format we verify the stem is all-hex (already done above).
        # Deep check: verify _ts field is a plausible Unix timestamp
        if ts and (ts < 1_600_000_000 or ts > 2_000_000_000):
            issues.append((f.name, "BAD_TIMESTAMP", f"ts={ts} out of range"))
            hash_bad += 1
        else:
            hash_ok += 1

    # ── Report ────────────────────────────────────────────────────────
    print(f"  Filename format:   {name_ok} valid,  {name_bad} malformed")
    print(f"  Content check:     {hash_ok} ok,     {len([i for i in issues if i[1] in ('CORRUPT','EMPTY','EMPTY_V','BAD_STRUCTURE')])} with issues")
    print()
    print(f"  File size distribution:")
    for bucket, count in size_buckets.items():
        bar = "█" * min(count // max(max(size_buckets.values())//20, 1), 40)
        print(f"    {bucket:>10}  {count:>5}  {bar}")
    print()
    if row_counts:
        print(f"  Row counts per file:")
        print(f"    min={min(row_counts)}  median={sorted(row_counts)[len(row_counts)//2]}  max={max(row_counts)}  avg={sum(row_counts)//len(row_counts)}")
    if age_days_list:
        print(f"  File age (days since fetch):")
        print(f"    newest={min(age_days_list):.1f}d  oldest={max(age_days_list):.1f}d  avg={sum(age_days_list)/len(age_days_list):.1f}d")
    print()

    if issues:
        print(f"  ⚠️  Issues found: {len(issues)}")
        # Show first 20, group the rest
        shown = issues[:20]
        for fname, kind, detail in shown:
            icon = "❌" if kind in ("CORRUPT","BAD_STRUCTURE","BAD_FILENAME") else "⚠️ "
            print(f"    {icon} [{kind:<14}] {fname[:55]}  {detail}")
        if len(issues) > 20:
            from collections import Counter
            kinds = Counter(k for _, k, _ in issues)
            print(f"    ... and {len(issues)-20} more: {dict(kinds)}")
        print()
        print(f"  To clean up issues:")
        print(f"    PYTHONPATH=. python src/market_data/cache_guard.py --cleanup")
    else:
        print(f"  ✅ All {info['real']} real files passed content + format checks")

    # 5. Summary verdict
    print()
    print("=" * 68)
    if n_missing == 0:
        print("  ✅ CACHE COMPLETE — all validation windows covered")
        print("     Next run: instant replay, $0 Databento cost")
    elif n_missing <= 10:
        print(f"  ⚠️  NEARLY COMPLETE — {n_missing} windows missing (~${est_cost:.2f} to complete)")
        print(f"     Re-run validate_databento_signals.py to fill the gaps")
    else:
        print(f"  🌐 PARTIAL — {n_cached}/{n_total} windows cached, {n_missing} to fetch")
        print(f"     Est. ${est_cost:.2f} and ~{est_min}–{est_max} min to complete")
    print("=" * 68)
    print()

    # Machine-readable summary
    return {
        "total_files":   info["total"],
        "real_files":    info["real"],
        "cached_windows": n_cached,
        "missing_windows": n_missing,
        "est_cost_usd":  round(est_cost, 2) if n_missing else 0,
    }


if __name__ == "__main__":
    result = main()
