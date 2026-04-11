# Unit Test Findings Report

**Date:** 2026-04-06 (v2 — all bugs fixed)
**Test File:** `tests/test_critical_paths.py`
**Results:** 71 passed, 0 failed, 7 skipped (yfinance not in sandbox)

---

## Bugs Found & Fixed

### BUG 1: `fetch_with_retry.py` — NameError on import (CRITICAL) — FIXED

**Severity:** CRITICAL — prevents module from loading
**File:** `fetch_with_retry.py`, line 82

**Problem:** `pd.DataFrame` in return type annotation evaluated at function definition time in Python 3.10, causing `NameError: name 'pd' is not defined`.

**Fix:** Added `from __future__ import annotations` and `import pandas as pd`.

**Status:** FIXED (v1) — regression test: `TestFetchWithRetry::test_module_import_succeeds`

---

### BUG 2: `fetch_all.py` — Timezone mismatch in `run_fetch()` delta merge (CRITICAL) — FIXED

**Severity:** CRITICAL — crashes on UTC-aware cached data
**File:** `fetch_all.py`, lines 353-354

**Problem:** Naive `pd.Timestamp(delta_start)` compared against UTC-aware index raised `TypeError`.

**Fix (v1):** Inline tz-check matching daily_data_update.py pattern.
**Fix (v2):** Replaced with centralised `tz_aware_cutoff()` helper.

**Status:** FIXED — regression test: `TestTzAwareCutoff::test_cutoff_filters_utc_index_without_error`

---

### BUG 3: `crypto_fetcher.py` — MultiIndex columns from yfinance >= 0.2.40 (MEDIUM) — FIXED

**Severity:** MEDIUM — crashes on newer yfinance versions
**File:** `crypto_fetcher.py`, line 47

**Problem:** yfinance >= 0.2.40 returns MultiIndex columns for single-ticker downloads. Calling `.lower()` on tuple columns raises `AttributeError`.

**Fix:** Added `df.columns = df.columns.get_level_values(0)` before lowercasing.

**Status:** FIXED (v1) — regression test: `TestCryptoFetcher::test_multiindex_columns_flattening`

---

### BUG 4: `daily_data_update.py` — Incorrect `is_crypto` detection in retry loop (LOW) — FIXED

**Severity:** LOW — affects retry path for edge-case symbols
**File:** `daily_data_update.py`, line 174

**Problem:** The retry loop used an inefficient list comprehension:
```python
is_crypto = (sym, True) in [(s, c) for s, c in all_symbols if c]
```
This returns `False` for any crypto symbol not already in `all_symbols` (e.g., KEY_CRYPTO symbols added dynamically), causing retries to hit the wrong API endpoint.

**Fix:** Replaced with set-based lookup that also checks KEY_CRYPTO:
```python
crypto_set = {s for s, c in all_symbols if c}
is_crypto = sym in crypto_set or sym in KEY_CRYPTO
```

**Status:** FIXED (v2) — regression tests: `TestDailyUpdateIsCryptoDetection` (5 tests)

---

### BUG 5: `fetch_all.py` — `validate_dataframe` skips NaN/negative checks on Title Case columns (MEDIUM) — FIXED

**Severity:** MEDIUM — quality checks silently pass on non-standard column names
**File:** `fetch_all.py`, `validate_dataframe()`

**Problem:** The `missing_columns` check correctly lowercased column names via `set(c.lower() for c in df.columns)`, but the subsequent NaN and negative checks used `if col in df.columns` with lowercase col names. If data arrived with Title Case columns (e.g., from yfinance: `Open`, `High`, `Close`), NaN/negative checks were silently skipped because `"close" not in ["Close", ...]`.

**Fix:** Added `df_check = df.rename(columns={c: c.lower()})` at the start of validation so all checks operate on normalised lowercase columns.

**Status:** FIXED (v2) — regression tests: `test_title_case_columns_detected`, `test_title_case_nan_detected`

---

### BUG 6: Resource leaks — `json.dump(data, open(...))` (LOW) — FIXED

**Severity:** LOW — file handles not explicitly closed
**Files:** `fetch_all.py` (3 locations), `daily_data_update.py` (1 location)

**Problem:** Pattern `json.dump(data, open(path, "w"))` opens a file without a context manager. While CPython's reference counting usually closes these promptly, this is not guaranteed (e.g., in PyPy or under GC pressure), which can cause data loss if the process crashes before the OS flushes buffers.

**Fix:** Replaced all instances with explicit `with open(path, "w") as f: json.dump(data, f)`.

**Status:** FIXED (v2) — regression test: `TestResourceHandling::test_json_dump_with_context_manager`

---

## Improvements Made (v2)

### Centralised `tz_aware_cutoff()` helper

The timezone cutoff logic was duplicated in 4 locations across 3 files. Created a single `tz_aware_cutoff(date_str, index)` function in `fetch_all.py` and replaced all inline copies:

- `fetch_all.py` `run_fetch()` — delta merge path
- `daily_data_update.py` `run_daily_update()` — main update loop
- `daily_data_update.py` `run_daily_update()` — retry loop
- `fetch_with_retry.py` `retry_failed_symbols()` — retry merge path

### Added `from __future__ import annotations` to all modules

Prevents type-hint evaluation bugs (BUG 1 pattern) across the entire pipeline:

- `fetch_all.py`
- `fetch_with_retry.py`
- `crypto_fetcher.py`
- `daily_data_update.py`
- `migrate_to_canonical.py`

Regression guard: `TestFutureAnnotations` (5 parametrized tests)

---

## Test Coverage Summary

| Module | Tests | Critical Paths Covered |
|--------|-------|----------------------|
| `validate_dataframe` | 13 | Empty, NaN, negatives, zero volume, jumps, missing cols, Title Case, mixed case, multi-issue, single row, large DF |
| `tz_aware_cutoff` helper | 7 | Naive, UTC, US/Eastern, filtering, empty index |
| Timezone regression | 3 | Naive, UTC, original bug demonstration |
| `_sym_dir` / `_canonical_path` | 7 | Equity, crypto, slash, colon, BRK/B |
| `_atomic_write` / file I/O | 7 | Create, no-tmp, roundtrip, overwrite, UTC index, canonical preference, end date |
| `FetchWithRetry` | 6 | Backoff calc, cap, first-attempt, import check, state roundtrip, empty state |
| Migration logic | 5 | Overlap merge, dedup, dry-run, empty dir, 3-file gaps |
| `crypto_fetcher` | 7 | Availability, column standard, MultiIndex fix+regression, timezone |
| `is_crypto` detection (BUG 4) | 5 | Known crypto, equity, unknown KEY_CRYPTO, old vs new logic |
| Data merge/dedup | 4 | Overlap, non-overlap, sort order, mixed tz |
| Resource handling | 2 | Context manager JSON, atomic write failure cleanup |
| Edge cases | 7 | Special chars, empty cache, all-zero vol, mixed case, sym in stats, large DF, duplicate index |
| Future annotations guard | 5 | All 5 pipeline modules checked |

**Total: 78 tests across 13 test classes (71 passed, 7 skipped)**

---

## Files Modified

| File | Bug # | Change |
|------|-------|--------|
| `fetch_all.py` | BUG 2, 5, 6 | Added `tz_aware_cutoff()` helper, fixed `validate_dataframe` column normalisation, fixed resource leaks, added `from __future__ import annotations` |
| `daily_data_update.py` | BUG 4, 6 | Fixed is_crypto detection with set-based lookup, replaced inline cutoff logic with helper, fixed resource leak, added `from __future__ import annotations` |
| `fetch_with_retry.py` | — | Replaced inline cutoff logic with `tz_aware_cutoff()` helper |
| `crypto_fetcher.py` | — | Added `from __future__ import annotations` |
| `migrate_to_canonical.py` | — | Added `from __future__ import annotations` |
| `tests/test_critical_paths.py` | — | Expanded from 51 to 78 tests; added 7 new test classes |
