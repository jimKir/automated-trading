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
_old_root = Path("/tmp/databento_cache")
_new_root = Path.home() / ".databento_cache"
if _old_root.exists() and not _new_root.exists():
    print(f"Migrating cache: {_old_root} → {_new_root}")
    shutil.copytree(str(_old_root), str(_new_root))
    print("  Migration complete. Old /tmp cache preserved (delete manually if desired).")
else:
    _new_root.mkdir(parents=True, exist_ok=True)

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
imb_cache_dir = Path.home() / ".databento_cache" / "imbalance"
imb_cache_dir.mkdir(parents=True, exist_ok=True)

n_cached = n_fetched = n_failed = 0
for i, d in enumerate(step_dates):
    # Check cache status for display
    import hashlib, json as _json
    ck = imb_cache_dir / f"{hashlib.md5(str(('imbalance', sorted(SYMS), str(d))).encode()).hexdigest()}.json"
    was_cached = ck.exists()
    try:
        sigs = imb_signal.compute_weekly(SYMS, d)
        if sigs:
            imb_rows[pd.Timestamp(d)] = sigs
            if was_cached:
                n_cached += 1
                status = "📁 cache"
            else:
                n_fetched += 1
                status = "🌐 fetch"
        else:
            n_failed += 1
            status = "⚠️  empty"
    except Exception as e:
        n_failed += 1
        status = f"❌ error"
    print(f"  [{i+1:>3}/{len(step_dates)}] {d}  {status}")

imb_df = pd.DataFrame(imb_rows).T if imb_rows else pd.DataFrame()
print(f"\n  Imbalance: {len(imb_df)} observations  "
      f"(📁 {n_cached} cached  🌐 {n_fetched} fetched  ❌ {n_failed} failed)")

# ── SIGNAL 2: OPRA OPTIONS FLOW ───────────────────────────────────────────────
print("\n[3/5] Building OPRA options flow signal...")

from strategy.databento_options_flow import OPRAOptionsFlowSignal
opra_signal = OPRAOptionsFlowSignal()

opra_rows = {}
for i, d in enumerate(step_dates):
    if i % 10 == 0:
        print(f"  {i}/{len(step_dates)} {d}", end="\r")
    try:
        sigs = opra_signal.compute_weekly(SYMS, d)
        if sigs:
            opra_rows[pd.Timestamp(d)] = sigs
    except Exception as e:
        pass

opra_df = pd.DataFrame(opra_rows).T if opra_rows else pd.DataFrame()
print(f"\n  OPRA options signal: {len(opra_df)} weekly observations")

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
