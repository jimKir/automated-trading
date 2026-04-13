"""
Cache Guard — Pre/Post Fetch Validator
=======================================
Prevents unnecessary (and costly) API calls by rigorously validating
what is already cached before touching any external data source.

Run BEFORE any fetch session:
    python src/market_data/cache_guard.py --check

Run AFTER a fetch session:
    python src/market_data/cache_guard.py --verify

Or in code:
    from src.market_data.cache_guard import CacheGuard
    guard = CacheGuard()
    guard.preflight_check(dates, symbols, schema)   # raises if suspicious
    guard.post_fetch_verify()                        # confirms what was written
    guard.estimate_cost(dates, symbols, schema)      # prints cost before committing
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# ── Databento cost table (USD per GB, approximate as of 2026) ────────────────
# Source: https://databento.com/pricing
COST_PER_GB = {
    ("XNAS.ITCH", "imbalance"): 16.00,
    ("XNAS.ITCH", "statistics"): 16.00,
    ("XNAS.ITCH", "trades"): 32.00,
    ("XNAS.ITCH", "mbp-1"): 64.00,
    ("OPRA.PILLAR", "ohlcv-1d"): 150.00,
    ("OPRA.PILLAR", "trades"): 280.00,
    ("OPRA.PILLAR", "definition"): 5.00,
    ("DBEQ.BASIC", "trades"): 11.00,
    ("DBEQ.BASIC", "ohlcv-1m"): 11.00,
    ("IEXG.TOPS", "trades"): 1.00,
}

# Approximate GB per symbol per day (empirical from our usage)
GB_PER_SYMBOL_DAY = {
    ("XNAS.ITCH", "imbalance"): 0.0016,  # 1.33GB / 20syms / 42days ≈ 0.0016
    ("XNAS.ITCH", "statistics"): 0.00002,
    ("XNAS.ITCH", "trades"): 0.008,
    ("OPRA.PILLAR", "ohlcv-1d"): 0.021,  # $245 for 20syms × ~60days
    ("OPRA.PILLAR", "trades"): 0.028,  # $185 for ~60 days
    ("DBEQ.BASIC", "trades"): 0.0005,
}

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "databento"


# ── Utilities ─────────────────────────────────────────────────────────────────


def _key_raw(schema: str, symbols: list[str], day: date) -> str:
    """MD5 hash of the canonical key — matches _cache_path() in the signal modules."""
    raw = "|".join(str(p) for p in [schema, sorted(symbols), str(day)])
    return hashlib.md5(raw.encode()).hexdigest()


def _key_readable(schema: str, symbols: list[str], day: date) -> str:
    """Human-readable filename prefix — matches the new _cache_path() format.
    Order: {schema}_{syms}_{date}_{hash8}  (matches module output)
    """
    h8 = _key_raw(schema, symbols, day)[:8]
    sym_part = "-".join(sorted(symbols)[:3]) if len(symbols) <= 3 else f"{len(symbols)}syms"
    return f"{schema}_{sym_part}_{day}_{h8}"


def _file(schema: str, symbols: list[str], day: date) -> Path:
    """Return the expected cache file path (new human-readable format)."""
    return CACHE_DIR / f"{_key_readable(schema, symbols, day)}.json"


def _file_any_format(schema: str, symbols: list[str], day: date) -> Path | None:
    """Find the cache file for this key in either old (MD5) or new (readable) format."""
    # Check new readable format first
    new_path = _file(schema, symbols, day)
    if new_path.exists():
        return new_path
    # Fall back to old MD5-only format
    old_path = CACHE_DIR / f"{_key_raw(schema, symbols, day)}.json"
    if old_path.exists():
        return old_path
    return None


def _is_valid(path: Path) -> tuple[bool, str]:
    """Returns (is_valid, reason)."""
    if not path.exists():
        return False, "not found"
    size = path.stat().st_size
    if size < 100:
        return False, f"too small ({size}B — empty/failed entry)"
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return False, "not a dict"
        if "_ts" not in data:
            return False, "missing _ts timestamp"
        v = data.get("v", {})
        if not isinstance(v, dict):
            return False, f"v is {type(v).__name__} not dict"
        if len(v) == 0:
            return False, "v is empty {}"
        return True, f"ok ({len(v)} rows, {size // 1024}KB)"
    except Exception as e:
        return False, f"parse error: {e}"


def _round_trip_test(schema: str, symbols: list[str], day: date) -> tuple[bool, str]:
    """Write a test entry, read it back, confirm they match."""
    test_path = CACHE_DIR / f"_test_{_key_raw(schema, symbols, day)}.json"
    test_data = {
        "v": {"0": {"symbol": "TEST", "value": 42.0}, "1": {"symbol": "TEST2", "value": -1.0}},
        "_ts": time.time(),
    }
    try:
        test_path.write_text(json.dumps(test_data))
        loaded = json.loads(test_path.read_text())
        test_path.unlink()
        if loaded.get("v") != test_data["v"]:
            return False, "round-trip mismatch"
        return True, "ok"
    except Exception as e:
        with contextlib.suppress(OSError):
            test_path.unlink()
        return False, str(e)


# ── Main Guard Class ───────────────────────────────────────────────────────────


class CacheGuard:
    """
    Pre/post fetch validator. Prevents costly API re-fetches.

    Usage pattern:
        guard = CacheGuard()

        # Before ANY fetch:
        plan = guard.preflight(dates, symbols, "imbalance", dataset="XNAS.ITCH")
        # → prints what's cached, what's missing, estimated cost
        # → raises if cost estimate exceeds budget

        # After fetch:
        guard.verify_written(dates, symbols, "imbalance")
        # → confirms all expected files were written with real data
    """

    def __init__(self, cache_dir: Path | None = None, cost_budget_usd: float = 10.0):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cost_budget = cost_budget_usd

    # ── Preflight check ──────────────────────────────────────────────────────

    def rename_legacy_files(self, schema: str, symbols: list[str], dates: list[date]) -> int:
        """
        Rename old MD5-only files to human-readable format.
        Handles both pure-MD5 and wrong-order readable formats.
        Returns count of files renamed.
        """
        renamed = 0
        for d in dates:
            new_path = _file(schema, symbols, d)
            if new_path.exists():
                continue  # already in correct format
            # Try old pure-MD5 format
            old_md5 = CACHE_DIR / f"{_key_raw(schema, symbols, d)}.json"
            if old_md5.exists() and old_md5.stat().st_size >= 100:
                old_md5.rename(new_path)
                renamed += 1
                continue
            # Try old wrong-order readable format (schema_date_syms_hash)
            h8 = _key_raw(schema, symbols, d)[:8]
            sym_part = f"{len(symbols)}syms" if len(symbols) > 3 else "-".join(sorted(symbols)[:3])
            alt_name = CACHE_DIR / f"{schema}_{d}_{sym_part}_{h8}.json"
            if alt_name.exists() and alt_name.stat().st_size >= 100:
                alt_name.rename(new_path)
                renamed += 1
        return renamed

    def delete_orphan_md5_files(self) -> int:
        """
        Delete any remaining pure-MD5 named files (32-char hex + .json).
        These are old cache files superseded by human-readable ones.
        Run after rename_legacy_files() to clean up.
        Returns count deleted.
        """
        deleted = 0
        for f in self.cache_dir.glob("*.json"):
            if re.match(r"^[0-9a-f]{32}[.]json$", f.name):
                f.unlink(missing_ok=True)
                deleted += 1
        return deleted

    def preflight(
        self,
        dates: list[date],
        symbols: list[str],
        schema: str,
        dataset: str = "XNAS.ITCH",
        dry_run: bool = False,
        abort_on_over_budget: bool = True,
    ) -> dict:
        """
        Run before any fetch session.

        1. Round-trip test: confirm cache dir is writable and reads correctly
        2. Scan all dates: classify as CACHED / EMPTY / MISSING
        3. Delete empty/corrupt files (they cause re-fetches)
        4. Estimate cost for missing dates
        5. Print clear summary
        6. Return plan dict

        dry_run=True: analyse only, don't delete anything
        abort_on_over_budget=True: raise if estimated cost > self.cost_budget
        """
        print()
        print("=" * 62)
        print("  CACHE PREFLIGHT CHECK")
        print(f"  Schema:  {dataset} / {schema}")
        print(f"  Dates:   {len(dates)} dates  ({min(dates)} → {max(dates)})")
        print(f"  Symbols: {len(symbols)}")
        print(f"  Cache:   {self.cache_dir}")
        print("=" * 62)

        # 0. Rename legacy MD5 files to human-readable format, then clean up orphans
        if not dry_run:
            renamed = self.rename_legacy_files(schema, symbols, dates)
            deleted = self.delete_orphan_md5_files()
            if renamed or deleted:
                msg = []
                if renamed:
                    msg.append(f"renamed {renamed} legacy files")
                if deleted:
                    msg.append(f"deleted {deleted} orphan MD5 files")
                print(f"  Cache cleanup: {', '.join(msg)}")

        # 1. Round-trip test
        ok, msg = _round_trip_test(schema, symbols[:3], dates[0])
        print(f"\n  [1/4] Cache read/write test:  {'✅ ' + msg if ok else '❌ FAIL: ' + msg}")
        if not ok:
            raise RuntimeError(f"Cache directory is not functional: {msg}")

        # 2. Scan all dates
        # The module stores cache by TRADING DAY (not week-end date).
        # So we scan the actual cache directory for valid files and
        # count what proportion of expected dates are covered.
        # A date is "cached" if ANY of its 10 surrounding trading days are cached.
        # Build two hash → path indexes:
        #   valid_by_*   — files with real data rows (used for signal quality)
        #   fetched_by_* — ALL fetched files: real + empty "no data" stubs
        #                  Empty stubs (v={}, size<100B) mean "we tried, got nothing"
        #                  and count toward coverage — prevents pointless re-fetches.
        valid_by_full_md5: dict = {}  # full_md5 → Path  (real data only)
        valid_by_hash8: dict = {}  # hash8    → Path  (real data only)
        fetched_by_hash8: dict = {}  # hash8    → Path  (real + empty stubs)

        for f in self.cache_dir.glob("*.json"):
            if f.name.startswith(".") or f.name == "catalogue.json":
                continue
            stem = f.stem
            # Extract hash8 for this file regardless of size
            if len(stem) == 32 and all(c in "0123456789abcdef" for c in stem):
                h8 = stem[:8]
                full = stem
            else:
                parts = stem.rsplit("_", 1)
                h8 = parts[1] if (len(parts) == 2 and len(parts[1]) == 8) else None
                full = None

            if f.stat().st_size < 100:
                # Empty stub — counts as fetched (we tried), not as real data
                if h8:
                    fetched_by_hash8[h8] = f
                continue

            # Real data file
            if full:
                valid_by_full_md5[full] = f
                valid_by_hash8[full[:8]] = f
                fetched_by_hash8[full[:8]] = f
            elif h8:
                valid_by_hash8[h8] = f
                fetched_by_hash8[h8] = f

        def _is_cached(schema_: str, symbols_: list, day: date) -> bool:
            """True if this day was fetched (real data OR confirmed-empty stub)."""
            full_md5 = _key_raw(schema_, symbols_, day)
            hash8 = full_md5[:8]
            return (
                full_md5 in valid_by_full_md5
                or hash8 in valid_by_hash8
                or hash8 in fetched_by_hash8
            )

        def trading_days_before(d: date, n: int = 10):
            days, cur = [], d - timedelta(days=1)
            while len(days) < n:
                if cur.weekday() < 5:
                    days.append(cur)
                cur -= timedelta(days=1)
            return days

        cached, empty_deleted, missing = [], [], []
        for d in dates:
            # Check if MOST of the 10 trading days before this date are cached
            trading_days = trading_days_before(d, 10)
            hits = sum(1 for td in trading_days if _is_cached(schema, symbols, td))
            if hits >= 8:  # 8/10 days cached = this week is covered
                cached.append(d)
            elif hits > 0:
                # Partial — some days cached, some not; clean up any corrupt files
                for td in trading_days:
                    f_path = valid_by_full_md5.get(
                        _key_raw(schema, symbols, td)
                    ) or valid_by_hash8.get(_key_raw(schema, symbols, td)[:8])
                    if f_path and f_path.exists():
                        valid_check, reason = _is_valid(f_path)
                        if not valid_check:
                            if not dry_run:
                                f_path.unlink(missing_ok=True)
                            empty_deleted.append((td, reason))
                missing.append(d)  # will re-check at fetch time
            else:
                missing.append(d)

        print("\n  [2/4] Cache scan:")
        print(f"    ✅ Cached (will skip):          {len(cached):>4} dates")
        print(
            f"    🗑  Invalid/deleted:             {len(empty_deleted):>4} dates"
            + (" (dry-run, not deleted)" if dry_run else "")
        )
        print(f"    🌐 Missing (will fetch from API): {len(missing):>4} dates")

        if empty_deleted:
            print("\n  Deleted invalid files:")
            for d, reason in empty_deleted[:5]:
                print(f"    {d}: {reason}")
            if len(empty_deleted) > 5:
                print(f"    ... and {len(empty_deleted) - 5} more")

        # 3. Cost estimate
        cost_per_gb = COST_PER_GB.get((dataset, schema), 20.0)
        gb_per_sym_day = GB_PER_SYMBOL_DAY.get((dataset, schema), 0.001)
        est_gb = len(missing) * len(symbols) * gb_per_sym_day
        est_cost = est_gb * cost_per_gb

        print(f"\n  [3/4] Cost estimate for {len(missing)} missing dates:")
        print(f"    Rate:      ${cost_per_gb:.2f}/GB  (~{gb_per_sym_day * 1000:.2f} MB/symbol/day)")
        print(f"    Est. data: {est_gb * 1024:.1f} MB")
        print(f"    Est. cost: ${est_cost:.2f} USD")
        print(f"    Budget:    ${self.cost_budget:.2f} USD")

        over_budget = est_cost > self.cost_budget
        if over_budget:
            print(
                f"\n  ⚠️  OVER BUDGET — estimated ${est_cost:.2f} > budget ${self.cost_budget:.2f}"
            )
            if abort_on_over_budget:
                print(
                    f"  Aborting. To proceed anyway: CacheGuard(cost_budget_usd={est_cost * 1.1:.0f})"
                )
                raise RuntimeError(
                    f"Estimated cost ${est_cost:.2f} exceeds budget ${self.cost_budget:.2f}. "
                    f"Set cost_budget_usd={est_cost * 1.1:.0f} to proceed."
                )
        else:
            print("    Status:    ✅ Within budget")

        # 4. Summary
        print("\n  [4/4] Plan:")
        if not missing:
            print(f"    ✅ Nothing to fetch — all {len(cached)} dates cached")
        else:
            print(f"    🌐 Will fetch {len(missing)} dates (~${est_cost:.2f})")
            print(f"    📁 Will use cache for {len(cached)} dates ($0.00)")
            print(
                f"    ⏱  Est. time: {len(missing) * 50 // 60}–{len(missing) * 70 // 60} min"
                f" at ~50–70s/date"
            )
        print("=" * 62)
        print()

        return {
            "cached": cached,
            "missing": missing,
            "deleted_invalid": [d for d, _ in empty_deleted],
            "est_cost_usd": round(est_cost, 2),
            "est_gb": round(est_gb, 3),
            "over_budget": over_budget,
        }

    # ── Post-fetch verification ───────────────────────────────────────────────

    def verify_written(
        self,
        dates: list[date],
        symbols: list[str],
        schema: str,
    ) -> dict:
        """
        Run immediately after a fetch session.
        Confirms that files were written for the requested dates.

        The signal modules cache by TRADING DAY (not week-end date) and use
        their own _cache_path() key format. Rather than trying to reconstruct
        the exact filename, we:
          1. Build the MD5 hash that _cache_path() would produce for each
             trading day in a ±10-day window around each requested date.
          2. Scan the cache directory for any .json file whose name contains
             that MD5 (both legacy MD5-only and new readable formats).
        This matches both old and new filename formats automatically.
        """
        from datetime import timedelta as _td

        print()
        print("=" * 62)
        print("  POST-FETCH VERIFICATION")
        print("=" * 62)

        # Build a lookup: MD5-hash → path for all valid cache files
        valid_by_hash: dict = {}
        for f in self.cache_dir.glob("*.json"):
            if f.name.startswith(".") or f.name == "catalogue.json":
                continue
            if f.stat().st_size < 100:
                continue
            # The last 8 chars before .json is the hash8 (new format)
            # Or the whole stem is a 32-char MD5 (old format)
            stem = f.stem
            # Old format: pure 32-char MD5
            if len(stem) == 32 and all(c in "0123456789abcdef" for c in stem):
                valid_by_hash[stem] = f
            else:
                # New readable format: ends in _{hash8}
                parts = stem.rsplit("_", 1)
                if len(parts) == 2 and len(parts[1]) == 8:
                    valid_by_hash[parts[1]] = f  # hash8 key

        def _trading_days_window(d: date, n: int = 12):
            """Return trading days in a ±n-day window around d."""
            days = []
            for delta in range(-2, n + 2):
                td = d - _td(days=delta)
                if td.weekday() < 5:  # Mon-Fri
                    days.append(td)
            return days

        good, bad = [], []
        good_files = []

        for d in dates:
            # Check if ANY trading day in the window around this date is cached
            found = False
            for td in _trading_days_window(d, n=12):
                # Compute the full MD5 that the signal module would use
                full_md5 = _key_raw(schema, symbols, td)
                hash8 = full_md5[:8]
                # Check both full MD5 (old format) and hash8 (new format)
                if full_md5 in valid_by_hash:
                    good.append(d)
                    good_files.append(valid_by_hash[full_md5])
                    found = True
                    break
                if hash8 in valid_by_hash:
                    good.append(d)
                    good_files.append(valid_by_hash[hash8])
                    found = True
                    break
            if not found:
                bad.append((d, "not found", self.cache_dir / f"{schema}_{d}_<not found>"))

        total_mb = sum(f.stat().st_size for f in good_files) / 1024 / 1024

        print(f"\n  ✅ Successfully written: {len(good)}/{len(dates)} dates ({total_mb:.1f} MB)")
        if bad:
            print(f"  ❌ Failed/missing:       {len(bad)} dates")
            for d, reason, _ in bad[:10]:
                print(f"    {d}: {reason}")

        # Spot-check a few files
        import random

        sample_files = random.sample(good_files, min(3, len(good_files)))
        print(f"\n  Spot-check ({len(sample_files)} random files):")
        for f in sample_files:
            try:
                data = json.loads(f.read_text())
                v = data.get("v", {})
                rows = list(v.values())[:1]
                print(f"    {f.name[:60]}: {len(v)} rows — sample: {rows[0] if rows else 'empty'}")
            except Exception as e:
                print(f"    {f.name[:60]}: parse error — {e}")

        success_rate = len(good) / len(dates) * 100 if dates else 0
        verdict = (
            "✅ PASS" if success_rate >= 95 else ("⚠️  PARTIAL" if success_rate >= 50 else "❌ FAIL")
        )
        print(f"\n  Result: {verdict} ({success_rate:.0f}% success rate)")
        print("=" * 62)
        print()

        return {"good": good, "bad": bad, "success_rate": success_rate}

    # ── Inventory ─────────────────────────────────────────────────────────────

    def inventory(self) -> None:
        """Print a complete inventory of what's cached."""
        files = [
            f
            for f in self.cache_dir.glob("*.json")
            if not f.name.startswith(".") and f.name != "catalogue.json"
        ]

        valid = [(f, json.loads(f.read_text())) for f in files if f.stat().st_size >= 100]
        invalid = [f for f in files if f.stat().st_size < 100]

        total_mb = sum(f.stat().st_size for f, _ in valid) / 1024 / 1024

        print()
        print("=" * 62)
        print(f"  CACHE INVENTORY: {self.cache_dir}")
        print("=" * 62)
        print(f"  Valid files:   {len(valid)} ({total_mb:.1f} MB)")
        print(f"  Invalid files: {len(invalid)} (will be auto-deleted on next read)")
        print()

        # Date coverage — infer from _ts field
        ts_list = [d.get("_ts", 0) for _, d in valid if "_ts" in d]
        if ts_list:
            oldest = datetime.fromtimestamp(min(ts_list)).strftime("%Y-%m-%d %H:%M")
            newest = datetime.fromtimestamp(max(ts_list)).strftime("%Y-%m-%d %H:%M")
            print(f"  Fetched between: {oldest} → {newest}")

        # Row count stats
        row_counts = [len(d.get("v", {})) for _, d in valid]
        if row_counts:
            print(
                f"  Rows per file:   min={min(row_counts)}  max={max(row_counts)}  "
                f"avg={sum(row_counts) // len(row_counts)}"
            )
            zero_rows = sum(1 for r in row_counts if r == 0)
            if zero_rows:
                print(f"  ⚠️  Files with 0 rows: {zero_rows} (will be cleaned on next read)")

        print("=" * 62)
        print()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> int:
        """Delete all invalid/empty cache files. Returns count deleted."""
        deleted = 0
        for f in self.cache_dir.glob("*.json"):
            if f.name.startswith(".") or f.name == "catalogue.json":
                continue
            valid, _ = _is_valid(f)
            if not valid:
                f.unlink(missing_ok=True)
                deleted += 1
        print(f"  Cleaned {deleted} invalid cache files.")
        return deleted


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Databento cache guard")
    parser.add_argument("--check", action="store_true", help="Run inventory + cleanup")
    parser.add_argument("--cleanup", action="store_true", help="Delete all invalid files")
    parser.add_argument(
        "--preflight", action="store_true", help="Run preflight for imbalance signal"
    )
    parser.add_argument("--budget", type=float, default=5.0, help="Cost budget in USD")
    args = parser.parse_args()

    guard = CacheGuard(cost_budget_usd=args.budget)

    if args.check or (not any([args.cleanup, args.preflight])):
        guard.inventory()

    if args.cleanup:
        n = guard.cleanup()
        print(f"Deleted {n} invalid files.")

    if args.preflight:
        # Default: check imbalance signal for 2023-2026
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
        from datetime import date

        dates = [date(2023, 1, 1) + timedelta(days=14 * i) for i in range(84)]
        guard.preflight(
            dates, SYMS, "imbalance", dataset="XNAS.ITCH", dry_run=True, abort_on_over_budget=False
        )
