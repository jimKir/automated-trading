"""
Databento Signal Validation — Walk-Forward IC + Permutation Test
================================================================
OOS period: 2023-01-01 → 2026-03-21
Universe: 20 liquid US equities on NASDAQ
Validation: same 4-window walk-forward + permutation test used throughout
"""
import os, sys, warnings, json
os.environ["DATABENTO_KEY"] = "db-SpVxiQLLTdDe9iD3sLwTpiqgBjtxk"
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/workspace/trading_system")

import numpy as np
import pandas as pd
import yfinance as yf
import databento as db
from datetime import date, timedelta, datetime
from scipy import stats

SYMS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO",
    "JPM","V","MA","UNH","JNJ","PG","HD","KO","XOM","CVX","BAC","GS"
]
OOS_START = "2023-01-01"
OOS_END   = "2026-03-21"
WINDOWS   = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-03-21"),
]

# ── Migrate any existing /tmp cache to permanent home dir cache ───────────────
import shutil
from pathlib import Path
_old_roots = [
    Path("/tmp/databento_cache"),
    Path.home() / ".databento_cache",
]
_new_root = Path(__file__).parent.parent / ".cache" / "databento"
_new_root.mkdir(parents=True, exist_ok=True)
# Migration: run ONCE only (sentinel file prevents repeated copies of bad data)
_migration_done = _new_root / ".migrated"
if not _migration_done.exists():
    for _old_root in [Path("/tmp/databento_cache"), Path.home() / ".databento_cache"]:
        if _old_root.exists():
            _moved = 0
            for _f in _old_root.glob("*.json"):
                if _f.stat().st_size < 100: continue  # skip empty files
                _dest = _new_root / _f.name
                if not _dest.exists():
                    shutil.copy2(str(_f), str(_dest))
                    _moved += 1
            if _moved:
                print(f"Migrated {_moved} real files: {_old_root} → {_new_root}")
    _migration_done.touch()  # mark done — never run again

# ── CACHE HEALTH CHECK — runs before any API call ────────────────────────────
# Imports the health-check module, runs a full file audit, auto-repairs what
# it can (renames legacy files, deletes corrupt/empty stubs), and aborts if
# the cache directory itself is broken.  Never touches the Databento API.
print("\n[PRE-FLIGHT] Running cache health check...")
try:
    from diagnostics.cache_health_check import run_health_check
    _hc = run_health_check(auto_fix=True, verbose=False)
    if _hc["status"] == "ERROR":
        print(f"  ❌ Cache health check FAILED: {_hc['message']}")
        print("  Cannot continue safely. Fix the cache directory first.")
        sys.exit(1)
    _n_issues = _hc.get("issues_fixed", 0) + _hc.get("issues_remaining", 0)
    _n_miss   = _hc.get("missing_windows", 0)
    _est_cost = _hc.get("est_cost_usd", 0)
    if _hc.get("issues_fixed"):
        print(f"  ✅ Auto-fixed {_hc['issues_fixed']} file issue(s)")
    if _hc.get("issues_remaining"):
        print(f"  ⚠️  {_hc['issues_remaining']} file issue(s) could not be auto-fixed")
    print(f"  Cache: {_hc['real_files']} real files  |  "
          f"{_hc['cached_windows']}/{_hc['total_windows']} windows cached  |  "
          f"{_n_miss} to fetch  (~${_est_cost:.2f})")
except ImportError:
    # Fallback: health check module not available — do minimal check
    _cache_dir = Path(__file__).parent.parent / ".cache" / "databento"
    if not _cache_dir.exists():
        print(f"  ❌ Cache directory missing: {_cache_dir}")
        sys.exit(1)
    _n_files = sum(1 for f in _cache_dir.glob("*.json") if f.stat().st_size > 100)
    print(f"  Cache dir OK — {_n_files} real files found")

print("=" * 68)
print("  Databento Signal Validation — OOS 2023-2026")
print(f"  Universe: {len(SYMS)} liquid US equities")
print("=" * 68)

# ── FETCH PRICE DATA (returns for IC computation) ────────────────────────────
print("\n[1/5] Fetching price data...")
prices = {}
for s in SYMS:
    try:
        df = yf.download(s, start="2022-06-01", end=OOS_END,
                         auto_adjust=True, progress=False)
        if not df.empty:
            prices[s] = df
    except: pass

closes = pd.DataFrame({s: prices[s]["Close"].squeeze() for s in prices})
returns = np.log(closes / closes.shift(1))
wc = closes.resample("W").last()
wr = returns.resample("W").sum()
print(f"  Loaded {len(prices)} symbols, {len(closes)} days")

# Regime mask
spy_rv = wc["SPY"].pct_change().rolling(8).std() if "SPY" in wc.columns else pd.Series()
thresh = spy_rv.expanding(26).mean() * 1.5
calm_mask   = (spy_rv <= thresh)
crisis_mask = ~calm_mask

# ── UTILS ────────────────────────────────────────────────────────────────────
def sharpe(r, ann=52):
    r = r.dropna()
    if len(r) < 10 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(ann))

def port_ret(sig_df, fwd_df, lag=1):
    s = sig_df.shift(lag)
    pos = np.sign(s)
    act = (pos != 0).sum(axis=1).replace(0, np.nan)
    return ((pos * fwd_df).sum(axis=1) / act).dropna()

def ic_suite(sig_df, wr_df, horizons=(5, 10, 21)):
    out = {}
    for h in horizons:
        fwd = wr_df.rolling(h).sum().shift(-h)
        sig_lag = sig_df.shift(1)
        ics = []
        for d in sig_lag.dropna(how="all").index:
            if d not in fwd.index: continue
            sv = sig_lag.loc[d].dropna()
            fv = fwd.loc[d].reindex(sv.index).dropna()
            c  = sv.index.intersection(fv.index)
            if len(c) < 5: continue
            ic, _ = stats.spearmanr(sv[c], fv[c])
            if not np.isnan(ic): ics.append(ic)
        if len(ics) > 5:
            mn = float(np.mean(ics))
            _, p = stats.ttest_1samp(ics, 0)
            out[h] = (mn, float(p), len(ics))
    return out

def permutation_test(sig_df, fwd_df, n=200):
    actual = sharpe(port_ret(sig_df, fwd_df))
    np.random.seed(42)
    nulls = []
    for _ in range(n):
        shuf = sig_df.copy()
        shuf.index = np.random.permutation(shuf.index)
        shuf = shuf.sort_index()
        nulls.append(sharpe(port_ret(shuf, fwd_df)))
    nulls = np.array(nulls)
    p = float((nulls >= actual).mean())
    z = float((actual - nulls.mean()) / nulls.std()) if nulls.std() > 0 else 0.0
    return p, z

def print_results(name, sig_df, wr_df, ics):
    r = port_ret(sig_df, wr_df).dropna()
    cm = calm_mask.reindex(r.index).fillna(True)
    cr = crisis_mask.reindex(r.index).fillna(False)
    p_val, z_sc = permutation_test(sig_df, wr_df)

    ic5   = ics.get(5,  (0,1,0))
    ic10  = ics.get(10, (0,1,0))
    ic21  = ics.get(21, (0,1,0))

    print(f"\n  ── {name} ──")
    print(f"  Full Sharpe: {sharpe(r):+.3f}  "
          f"Calm: {sharpe(r[cm]):+.3f}  Crisis: {sharpe(r[cr]):+.3f}")
    print(f"  IC@5d:  {ic5[0]:+.4f}  p={ic5[1]:.4f}  {'✅' if ic5[1]<0.05 else '⚠️' if ic5[1]<0.10 else '❌'}")
    print(f"  IC@10d: {ic10[0]:+.4f}  p={ic10[1]:.4f}  {'✅' if ic10[1]<0.05 else '⚠️' if ic10[1]<0.10 else '❌'}")
    print(f"  IC@21d: {ic21[0]:+.4f}  p={ic21[1]:.4f}  {'✅' if ic21[1]<0.05 else '⚠️' if ic21[1]<0.10 else '❌'}")
    print(f"  Permutation: p={p_val:.4f} ({z_sc:.1f}σ)  "
          f"{'✅ SIGNIFICANT' if p_val<0.05 else '⚠️ marginal' if p_val<0.10 else '❌ not sig'}")

    print(f"  Walk-forward:")
    wf_pos = 0
    for label, ws, we in WINDOWS:
        m = (r.index >= ws) & (r.index <= we)
        sh = sharpe(r[m])
        if sh > 0: wf_pos += 1
        flag = "✅" if sh > 0.10 else ("⚠️" if sh > 0 else "❌")
        print(f"    {label}: {sh:+.3f} {flag}")
    print(f"  Positive: {wf_pos}/4")
    return sharpe(r), wf_pos, p_val

# ── SIGNAL 1: CLOSING AUCTION IMBALANCE ──────────────────────────────────────
print("\n[2/5] Building closing imbalance signal from XNAS.ITCH...")

from strategy.databento_imbalance import ClosingImbalanceSignal
imb_signal = ClosingImbalanceSignal()

# Walk-forward: compute weekly signal for each week in OOS
print("  Computing weekly imbalance signals...")
print("  (cached dates replay instantly, new dates hit Databento API ~11s each)")
imb_rows = {}
# Use biweekly steps to limit API calls during validation
week_dates = [d.date() for d in wc.index
              if pd.Timestamp(OOS_START) <= d <= pd.Timestamp(OOS_END)]
step_dates = week_dates[::2]  # every 2 weeks

from pathlib import Path
from strategy.databento_imbalance import CACHE_DIR as _IMB_CACHE_DIR

# Cache status: count real (non-empty) files already on disk
_all_cache_files = list(_IMB_CACHE_DIR.glob("*.json"))
_real_cached = sum(1 for f in _all_cache_files
                   if f.stat().st_size > 100)  # >100 bytes = has real data
print(f"  Cache dir : {_IMB_CACHE_DIR}")
print(f"  Cached files: {len(_all_cache_files)} total, {_real_cached} with real data")
print(f"  (Each weekly date uses 10 trading day fetches internally)")

# ── PREFLIGHT: validate cache, estimate cost, abort if over budget ──────────
from src.market_data.cache_guard import CacheGuard
import os as _env_os
from datetime import date as date
_guard = CacheGuard(cost_budget_usd=float(_env_os.environ.get("DATABENTO_BUDGET","20")))
_preflight_dates = [d if isinstance(d, date) else date.fromisoformat(str(d)) for d in step_dates]
try:
    _plan = _guard.preflight(
        dates=_preflight_dates, symbols=SYMS,
        schema="imbalance", dataset="XNAS.ITCH",
        abort_on_over_budget=True,
    )
    # Collect already-cached dates without any API call
    for _cd in [d for d in step_dates if (d if isinstance(d, date) else date.fromisoformat(str(d))) not in _plan["missing"]]:
        try:
            sigs = imb_signal.compute_weekly(SYMS, _cd)
            if sigs: imb_rows[pd.Timestamp(_cd)] = sigs
        except: pass
    # Only fetch genuinely missing dates
    step_dates = [d for d in step_dates
                  if (d if isinstance(d, date) else date.fromisoformat(str(d))) in _plan["missing"]]
except RuntimeError as _e:
    print(f"\n  ABORTED by cache guard: {_e}")
    sys.exit(1)

import time as _time
n_obs = 0
for i, d in enumerate(step_dates):
    t0 = _time.time()
    try:
        sigs = imb_signal.compute_weekly(SYMS, d)
        elapsed = _time.time() - t0
        if sigs:
            imb_rows[pd.Timestamp(d)] = sigs
            n_obs += 1
            # Fast = cache hit (<1s), slow = API fetch (>5s)
            status = "📁 cache" if elapsed < 1.0 else f"🌐 fetch ({elapsed:.0f}s)"
        else:
            status = "⚠️  empty"
    except Exception as e:
        status = f"❌ {str(e)[:40]}"
    print(f"  [{i+1:>3}/{len(step_dates)}] {d}  {status}")

imb_df = pd.DataFrame(imb_rows).T if imb_rows else pd.DataFrame()
print(f"\n  Imbalance: {len(imb_df)} observations collected")

# ── POST-FETCH VERIFICATION ───────────────────────────────────────────────────
if step_dates:  # only verify if we actually fetched anything
    _fetched_dates = [d if isinstance(d, date) else _date.fromisoformat(str(d))
                      for d in step_dates]
    _guard.verify_written(_fetched_dates, SYMS, "imbalance")

# ── SIGNAL 2: OPRA OPTIONS FLOW (disabled by default — $245 to re-fetch) ──
import os as _os
print("\n[3/5] OPRA options flow...")
if _os.environ.get("ENABLE_OPRA"):
    print("  OPRA ENABLED (ENABLE_OPRA=1)")
    from strategy.databento_options_flow import OPRAOptionsFlowSignal
    _opra = OPRAOptionsFlowSignal()
    _opra_dates = [d for d in step_dates if pd.Timestamp(d) >= pd.Timestamp("2025-04-01")]
    import time as _ot
    opra_rows = {}
    for i, d in enumerate(_opra_dates):
        _t0 = _ot.time()
        try:
            sigs = _opra.compute_weekly(SYMS, d)
            _el = _ot.time() - _t0
            if sigs:
                opra_rows[pd.Timestamp(d)] = sigs
                _st = "\U0001f4c1 cache" if _el < 1.0 else f"\U0001f310 fetch ({_el:.0f}s)"
            else: _st = "\u26a0\ufe0f  empty"
        except Exception as _e:
            _st = f"\u274c {str(_e)[:40]}"
        print(f"  [{i+1:>3}/{len(_opra_dates)}] {d}  {_st}")
    opra_df = pd.DataFrame(opra_rows).T if opra_rows else pd.DataFrame()
    print(f"  OPRA: {len(opra_df)} observations")
else:
    opra_df = pd.DataFrame()
    print("  \u26a0\ufe0f  OPRA skipped (set ENABLE_OPRA=1 to enable)")
    print("  Reason: ohlcv-1d schema costs $245 per full re-fetch if cache misses")
    print("  Safe to enable ONLY after confirming cache works with spot-check below")

# ── SIGNAL 3: OPENING CROSS ───────────────────────────────────────────────────
print("\n[4/5] Building opening cross volume anomaly signal...")

from strategy.databento_opening_cross import OpeningCrossSignal
cross_signal = OpeningCrossSignal()

cross_rows = {}
for i, d in enumerate(step_dates):
    if i % 10 == 0:
        print(f"  {i}/{len(step_dates)} {d}", end="\r")
    try:
        sigs = cross_signal.compute_weekly(SYMS, d)
        if sigs:
            cross_rows[pd.Timestamp(d)] = sigs
    except Exception as e:
        pass

cross_df = pd.DataFrame(cross_rows).T if cross_rows else pd.DataFrame()
print(f"\n  Opening cross signal: {len(cross_df)} weekly observations")

# ── EVALUATE ALL THREE ────────────────────────────────────────────────────────
print("\n[5/5] Evaluating signals...")

results = {}
for name, sig_df in [
    ("Closing Auction Imbalance", imb_df),
    ("OPRA Options Flow",         opra_df),
    ("Opening Cross Anomaly",     cross_df),
]:
    if sig_df.empty:
        print(f"\n  ── {name}: NO DATA ──")
        continue
    sig_oos = sig_df[sig_df.index >= OOS_START].reindex(wc.index).ffill()
    ics = ic_suite(sig_oos, wr)
    sh, wf, pv = print_results(name, sig_oos, wr, ics)
    results[name] = dict(sharpe=sh, wf_pos=wf, perm_p=pv,
                         ics={str(k): v[0] for k,v in ics.items()})

# ── COMPOSITE TEST ────────────────────────────────────────────────────────────
valid_signals = [(n, s) for n, s in [
    ("imbalance", imb_df),
    ("opra",      opra_df),
    ("cross",     cross_df),
] if not s.empty]

if len(valid_signals) >= 2:
    print("\n  ── Databento Composite ──")
    all_cols = list(set.union(*[set(s.columns) for _,s in valid_signals]))
    comp = pd.DataFrame(0.0, index=wc.index, columns=all_cols)
    weights = {0: 0.35, 1: 0.40, 2: 0.25}
    for i, (name, sig_df) in enumerate(valid_signals):
        sig_oos = sig_df[sig_df.index >= OOS_START].reindex(wc.index).ffill()
        w = weights.get(i, 1/len(valid_signals))
        for col in sig_oos.columns:
            if col in comp.columns:
                comp[col] += w * sig_oos[col].fillna(0)

    ics_comp = ic_suite(comp, wr)
    print_results("Databento Composite", comp, wr, ics_comp)

print()
print("=" * 68)
print("  SUMMARY TABLE")
print("=" * 68)
print(f"  {'Signal':<30} {'Sharpe':>8} {'WF':>5} {'p-val':>8} {'Verdict':>14}")
print("  " + "─" * 68)
for name, r in results.items():
    verdict = "✅ USE" if r["perm_p"] < 0.05 and r["wf_pos"] >= 3 else \
              ("⚠️  WEAK" if r["perm_p"] < 0.10 or r["wf_pos"] >= 2 else "❌ SKIP")
    print(f"  {name:<30} {r['sharpe']:>+8.3f} {r['wf_pos']:>4}/4 {r['perm_p']:>8.4f} {verdict:>14}")

with open("/tmp/databento_validation.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\n  Saved: /tmp/databento_validation.json")

# Print data catalogue so you can see exactly what's been stored
try:
    from src.market_data.catalogue import get_catalogue
    print()
    get_catalogue().summary()
except Exception:
    pass
