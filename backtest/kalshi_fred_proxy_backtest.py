"""
FRED-proxy Kalshi backtest: reconstructs macro stress signals from FRED data
(yield curve, PCE surprises, EFFR changes) to approximate what Kalshi markets
would have priced. Runs same 7-year walk-forward as prior validations.

Run: python backtest/kalshi_fred_proxy_backtest.py --save-results
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

# ── Walk-forward folds (same as all prior backtests) ─────────────────────────
FOLDS = [
    {"name":"GFC_2008",      "test_start":"2008-09-01","test_end":"2009-03-31"},
    {"name":"Bull_2013",     "test_start":"2013-01-01","test_end":"2015-12-31"},
    {"name":"COVID_2020",    "test_start":"2020-02-01","test_end":"2020-12-31"},
    {"name":"RateHike_2022", "test_start":"2022-01-01","test_end":"2022-12-31"},
    {"name":"Recovery_2023", "test_start":"2023-01-01","test_end":"2023-12-31"},
    {"name":"Bull_2024",     "test_start":"2024-01-01","test_end":"2024-12-31"},
    {"name":"Tariff_2025",   "test_start":"2025-01-01","test_end":"2026-04-05"},
]

# ── FRED proxy stress construction ───────────────────────────────────────────

def build_fred_proxy_stress(start="2005-01-01", end="2026-04-05") -> pd.DataFrame:
    """
    Build daily macro stress series from FRED/yfinance data.
    Approximates what Kalshi markets would have priced.

    Components:
    1. yield_curve_stress:   2s10s inversion depth (proxy: ^TNX - ^FVX)
    2. fed_policy_stress:    Uncertainty around FOMC meetings (EFFR vol)
    3. inflation_stress:     PCE/CPI running hot above Fed target
    4. recession_stress:     Composite of the above + SPY momentum

    All normalized to [0,1]. Composite = weighted average.
    """
    import yfinance as yf

    print("  Fetching yield data (^TNX, ^FVX, ^IRX)...")
    tickers = {"TNX":"^TNX", "FVX":"^FVX", "IRX":"^IRX",
               "SPY":"SPY",  "TLT":"TLT",  "GLD":"GLD"}
    prices = {}
    for name, tkr in tickers.items():
        try:
            df = yf.download(tkr, start=start, end=end,
                             progress=False, auto_adjust=True)
            if not df.empty:
                col = "Close" if "Close" in df.columns else df.columns[0]
                prices[name] = df[col].squeeze()
                print(f"    {name}: {len(prices[name])} bars")
        except Exception as e:
            print(f"    {name}: failed ({e})")

    # Also load local SPY parquet for more history
    try:
        from data.data_store import DataStore
        spy_local = DataStore().load("SPY", start_date=start, end_date=end)
        if spy_local is not None and len(spy_local) > 0:
            prices["SPY_local"] = spy_local["close"]
    except Exception:
        pass

    spy = prices.get("SPY_local", prices.get("SPY"))

    # Build daily index
    idx = pd.date_range(start, end, freq="B")

    def align(s):
        if s is None or len(s) == 0:
            return pd.Series(np.nan, index=idx)
        return s.reindex(idx).ffill().bfill()

    tnx = align(prices.get("TNX"))  # 10yr yield
    fvx = align(prices.get("FVX"))  # 5yr yield
    irx = align(prices.get("IRX"))  # 3mo yield
    spy_s = align(spy)

    # ── 1. Yield curve stress ─────────────────────────────────────────────
    # 5s10s spread (proxy for 2s10s): negative = inverted = stress
    spread_5_10 = tnx - fvx  # typically +0.3 to +1.5 in normal times
    # Normalize: spread <= 0 → stress=1, spread >= 1 → stress=0
    yc_stress = (-spread_5_10).clip(lower=-0.5).map(
        lambda x: max(min((x + 0.5) / 1.5, 1.0), 0.0)
    )

    # ── 2. Fed policy stress ──────────────────────────────────────────────
    # Approximation: rapid 10yr yield changes = policy uncertainty
    tnx_5d_chg = tnx.pct_change(5).abs()
    tnx_60d_avg = tnx_5d_chg.rolling(60).mean()
    fed_stress = (tnx_5d_chg / tnx_60d_avg.replace(0, np.nan) - 1).clip(0, 3) / 3
    fed_stress = fed_stress.fillna(0)

    # ── 3. Inflation stress ───────────────────────────────────────────────
    # Proxy: 10yr yield level above 4% = high inflation expectations
    # Normalized: 2% → 0, 5%+ → 1
    infl_stress = ((tnx - 2.0) / 3.0).clip(0, 1)

    # FOMC meeting dates — mark day-before stress spike
    fomc_dates = pd.DatetimeIndex([
        # 2022 rate hike cycle
        "2022-03-15","2022-05-03","2022-06-14","2022-07-26",
        "2022-09-20","2022-11-01","2022-12-13",
        # 2023 continued hiking + pause
        "2023-02-01","2023-03-21","2023-05-02","2023-06-13",
        "2023-07-25","2023-09-19","2023-11-01","2023-12-12",
        # 2024 cuts begin
        "2024-01-30","2024-03-19","2024-05-01","2024-06-11",
        "2024-07-30","2024-09-17","2024-11-06","2024-12-17",
        # 2025
        "2025-01-28","2025-03-18","2025-05-06","2025-06-17",
        "2025-07-29","2025-09-16","2025-11-05","2025-12-16",
        # 2026
        "2026-01-28","2026-03-17",
    ])
    # Day before FOMC: inject policy uncertainty spike
    fomc_stress = pd.Series(0.0, index=idx)
    for d in fomc_dates:
        for offset in [-3, -2, -1, 0]:
            target = d + pd.Timedelta(days=offset)
            if target in idx:
                # Spike decays: 3d before = 0.3, 1d before = 0.6, day-of = 0.4
                spike = {-3: 0.25, -2: 0.35, -1: 0.55, 0: 0.40}.get(offset, 0)
                fomc_stress[target] = max(fomc_stress[target], spike)

    # Blend fed_stress with fomc_stress
    fed_combined = (0.5 * fed_stress + 0.5 * fomc_stress).clip(0, 1)

    # ── 4. Recession / growth stress ─────────────────────────────────────
    # SPY 63d momentum: sustained negative = growth worry
    if spy_s is not None and spy_s.notna().sum() > 100:
        spy_mom = spy_s.pct_change(63)
        rec_stress = (-spy_mom / 0.20).clip(0, 1)  # -20% over 3mo → score=1
        rec_stress = rec_stress.fillna(0)
    else:
        rec_stress = pd.Series(0.0, index=idx)

    # ── Composite ─────────────────────────────────────────────────────────
    composite = (
        0.30 * yc_stress +
        0.30 * fed_combined +
        0.20 * infl_stress +
        0.20 * rec_stress
    ).clip(0, 1)

    df = pd.DataFrame({
        "yield_curve_stress": yc_stress,
        "fed_policy_stress":  fed_combined,
        "inflation_stress":   infl_stress,
        "recession_stress":   rec_stress,
        "composite_stress":   composite,
    }, index=idx)

    return df.ffill().fillna(0)

# ── Strategy returns ──────────────────────────────────────────────────────────

def load_returns(start, end) -> pd.Series:
    from data.data_store import DataStore
    store = DataStore()
    spy = store.load("SPY", start_date=start, end_date=end)
    if spy is None or len(spy) < 20:
        raise ValueError(f"No SPY data for {start}-{end}")

    # Multi-asset proxy: blend SPY(40%) + QQQ(20%) + GLD(15%) + TLT(15%) + BTC(10%)
    weights = {"SPY": 0.40, "QQQ": 0.20, "GLD": 0.15, "TLT": 0.15}
    rets = {}
    for sym, w in weights.items():
        df = store.load(sym, start_date=start, end_date=end)
        if df is not None and len(df) > 10:
            rets[sym] = df["close"].pct_change()

    if not rets:
        return spy["close"].pct_change().dropna()

    combined = sum(rets[s] * w for s, w in weights.items() if s in rets)
    total_w   = sum(w for s, w in weights.items() if s in rets)
    portfolio = (combined / total_w).dropna()
    return portfolio

def apply_kalshi_scale(returns: pd.Series,
                       stress: pd.Series,
                       kalshi_weight: float = 0.25) -> pd.Series:
    """
    Apply Kalshi-derived position scale.
    Stress thresholds (same as AnomalyLayer):
      < 0.20 → 1.00x  (NORMAL)
      0.20-0.35 → 0.85x (ELEVATED)
      0.35-0.50 → 0.65x (STRESSED)
      > 0.50 → 0.40x   (CRISIS)
    Weight: 25% Kalshi, 75% unscaled (additive on top of base strategy).
    """
    def s2scale(s):
        if   s < 0.20: return 1.00
        elif s < 0.35: return 0.85
        elif s < 0.50: return 0.65
        else:          return 0.40

    aligned = stress.reindex(returns.index).ffill().fillna(0)
    scales  = aligned.map(s2scale)
    blended = (1 - kalshi_weight) * 1.0 + kalshi_weight * scales
    return returns * blended

def metrics(returns: pd.Series, label="") -> dict:
    if len(returns) < 5:
        return {"label":label,"sharpe":0,"max_dd":0,"total_return":0,"calmar":0,"sortino":0}
    r = returns.dropna()
    ann_r = r.mean() * 252
    ann_v = r.std() * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 0 else 0
    cum = (1 + r).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ann_r / downside if downside > 0 else 0
    calmar  = ann_r / abs(max_dd) if max_dd < 0 else 0
    return {
        "label": label,
        "sharpe":       round(sharpe, 3),
        "max_dd":       round(max_dd, 4),
        "total_return": round(float(cum.iloc[-1]-1), 4),
        "calmar":       round(calmar, 3),
        "sortino":      round(sortino, 3),
        "n_days":       len(r),
    }

# ── Charts ────────────────────────────────────────────────────────────────────

def make_charts(fold_results: list, stress_df: pd.DataFrame,
                save_path: str = "results/kalshi_fred_proxy_backtest.png"):
    n = len(fold_results)
    fig = plt.figure(figsize=(20, 4 * (n // 2 + 1) + 4))
    cols = 2

    # Stress timeline (full period) at top
    ax_stress = fig.add_subplot(n // 2 + 2, 1, 1)
    comp = stress_df["composite_stress"]
    ax_stress.fill_between(comp.index, comp, alpha=0.35,
                           color="salmon", label="FRED-proxy stress")
    ax_stress.axhline(0.20, color="gold",   ls="--", lw=1, alpha=0.7, label="Elevated (0.20)")
    ax_stress.axhline(0.35, color="orange", ls="--", lw=1, alpha=0.7, label="Stressed (0.35)")
    ax_stress.axhline(0.50, color="red",    ls="--", lw=1, alpha=0.7, label="Crisis (0.50)")
    # Mark test periods
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, r in enumerate(fold_results):
        ax_stress.axvspan(pd.Timestamp(r["test_start"]),
                          pd.Timestamp(r["test_end"]),
                          alpha=0.10, color=colors[i])
    ax_stress.set_title("FRED-Proxy Macro Stress (Full History) — Test Windows Shaded",
                         fontsize=12, fontweight="bold")
    ax_stress.set_ylabel("Stress Score")
    ax_stress.legend(loc="upper left", fontsize=8)
    ax_stress.set_ylim(0, 1)

    # Per-fold cumulative return panels
    for i, r in enumerate(fold_results):
        row = (i // cols) + 1
        col = (i % cols) + 1
        ax = fig.add_subplot(n // 2 + 2, cols,
                             (row) * cols + col)

        base_r   = pd.Series(r["base_returns"])
        kalshi_r = pd.Series(r["kalshi_returns"])
        base_r.index   = pd.to_datetime(base_r.index)
        kalshi_r.index = pd.to_datetime(kalshi_r.index)

        base_cum   = (1 + base_r).cumprod()
        kalshi_cum = (1 + kalshi_r).cumprod()

        ax.plot(base_cum.index,   base_cum.values,   color="gray",
                lw=1.5, ls="--", label=f"Base  Sh={r['base']['sharpe']:.2f}")
        ax.plot(kalshi_cum.index, kalshi_cum.values, color="steelblue",
                lw=2.0, label=f"Kalshi Sh={r['kalshi']['sharpe']:.2f}")

        # Shade stress regime
        st = stress_df["composite_stress"].reindex(base_r.index).ffill()
        for j in range(len(st)-1):
            if st.iloc[j] >= 0.50:
                ax.axvspan(st.index[j], st.index[j+1],
                           alpha=0.12, color="red", lw=0)
            elif st.iloc[j] >= 0.35:
                ax.axvspan(st.index[j], st.index[j+1],
                           alpha=0.08, color="orange", lw=0)

        dsh = r["delta_sharpe"]
        ddd = r["delta_max_dd"]
        color_sh = "green" if dsh > 0 else "red"
        color_dd = "green" if ddd > 0.001 else "red"

        ax.set_title(
            f"{r['fold']}\n"
            f"ΔSharpe: {dsh:+.3f}  ΔMaxDD: {ddd:+.4f}",
            fontsize=9, fontweight="bold",
            color="darkgreen" if dsh > 0 and ddd > 0 else "darkred"
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("Cum. Return")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {save_path}")

def make_summary_table(fold_results, save_path="results/kalshi_fred_proxy_table.png"):
    rows = []
    for r in fold_results:
        b = r["base"]; k = r["kalshi"]
        rows.append([
            r["fold"],
            f"{b['sharpe']:.3f}", f"{k['sharpe']:.3f}",
            f"{r['delta_sharpe']:+.3f}",
            f"{b['max_dd']:.3f}",  f"{k['max_dd']:.3f}",
            f"{r['delta_max_dd']:+.4f}",
            f"{b['total_return']:.2%}", f"{k['total_return']:.2%}",
            "✅" if r["dd_improves"] else "❌",
        ])

    cols = ["Fold","Base Sh","Kalshi Sh","ΔSh",
            "Base DD","Kalshi DD","ΔDD",
            "Base Ret","Kalshi Ret","DD↑?"]

    fig, ax = plt.subplots(figsize=(18, len(rows)*0.55 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, r in enumerate(fold_results):
        clr = "#d5f5e3" if r["dd_improves"] and r["sharpe_improves"] else \
              "#fef9e7" if r["dd_improves"] or r["sharpe_improves"] else \
              "#fadbd8"
        for j in range(len(cols)):
            tbl[i+1, j].set_facecolor(clr)

    # Summary row
    n = len(fold_results)
    n_dd = sum(1 for r in fold_results if r["dd_improves"])
    n_sh = sum(1 for r in fold_results if r["sharpe_improves"])
    mdd  = np.mean([r["delta_max_dd"] for r in fold_results])
    msh  = np.mean([r["delta_sharpe"] for r in fold_results])
    verdict = "ADOPT" if n_dd >= 5 else ("CONDITIONAL" if n_dd >= 4 else "REJECT")
    plt.title(
        f"FRED-Proxy Kalshi Macro Stress — Walk-Forward Summary\n"
        f"MaxDD improved {n_dd}/{n} | Sharpe improved {n_sh}/{n} | "
        f"Mean ΔMaxDD: {mdd:+.4f} | Mean ΔSharpe: {msh:+.3f} | "
        f"Verdict: {verdict}",
        fontsize=11, fontweight="bold", pad=12
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Table saved → {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run(save=False):
    print("\n" + "="*70)
    print("  FRED-PROXY KALSHI BACKTEST — 7-Year Walk-Forward")
    print("  Macro stress: yield curve + Fed policy + PCE + recession proxy")
    print("  Same windows as ChoppyDetector v3/v4 and AnomalyLayer backtests")
    print("="*70)

    print("\n[1/3] Building FRED-proxy stress series (2005-2026)...")
    stress_df = build_fred_proxy_stress("2005-01-01", "2026-04-05")
    comp = stress_df["composite_stress"]
    print(f"  Range: {comp.index.min().date()} → {comp.index.max().date()}")
    print(f"  Mean:  {comp.mean():.3f} | P75: {comp.quantile(0.75):.3f} | "
          f"P90: {comp.quantile(0.90):.3f} | Max: {comp.max():.3f}")

    print(f"\n[2/3] Running {len(FOLDS)} walk-forward folds...")
    fold_results = []

    hdr = (f"\n  {'Fold':<18} {'N':>5} {'Base Sh':>9} {'Kalshi Sh':>10} "
           f"{'ΔSh':>8} {'Base DD':>9} {'Kalshi DD':>10} {'ΔDD':>9}")
    print(hdr)
    print("  " + "─"*82)

    for fold in FOLDS:
        try:
            returns = load_returns(fold["test_start"], fold["test_end"])
            if len(returns) < 20:
                print(f"  {fold['name']:<18} — insufficient data, skipping")
                continue

            base_r   = returns.copy()
            kalshi_r = apply_kalshi_scale(returns, comp, kalshi_weight=0.25)

            bm = metrics(base_r,   label="base")
            km = metrics(kalshi_r, label="kalshi")
            dsh = round(km["sharpe"] - bm["sharpe"], 3)
            ddd = round(km["max_dd"] - bm["max_dd"], 4)

            sf = "✅" if dsh > 0.05  else ("⚠" if abs(dsh)  < 0.05  else "❌")
            df = "✅" if ddd > 0.005 else ("⚠" if abs(ddd) < 0.005 else "❌")

            result = {
                "fold": fold["name"],
                "test_start": fold["test_start"],
                "test_end":   fold["test_end"],
                "n_days":     len(returns),
                "base":       bm,
                "kalshi":     km,
                "delta_sharpe": dsh,
                "delta_max_dd": ddd,
                "dd_improves":  ddd > 0.005,
                "sharpe_improves": dsh > 0.05,
                "base_returns":   base_r.to_dict(),
                "kalshi_returns": kalshi_r.to_dict(),
            }
            fold_results.append(result)

            print(f"  {fold['name']:<18} {len(returns):>5} {bm['sharpe']:>9.3f} "
                  f"{km['sharpe']:>10.3f} {sf}{dsh:>+7.3f} "
                  f"{bm['max_dd']:>9.4f} {km['max_dd']:>10.4f} {df}{ddd:>+8.4f}")

        except Exception as e:
            print(f"  {fold['name']:<18} — ERROR: {e}")

    if not fold_results:
        print("No results. Check data availability.")
        return

    # Summary
    n = len(fold_results)
    n_dd = sum(1 for r in fold_results if r["dd_improves"])
    n_sh = sum(1 for r in fold_results if r["sharpe_improves"])
    mdd  = np.mean([r["delta_max_dd"] for r in fold_results])
    msh  = np.mean([r["delta_sharpe"] for r in fold_results])
    verdict = "ADOPT" if n_dd >= 5 else ("CONDITIONAL" if n_dd >= 4 else "REJECT")

    print(f"\n{'='*70}")
    print(f"  MaxDD improved:  {n_dd}/{n} folds  | Mean ΔMaxDD: {mdd:+.4f}")
    print(f"  Sharpe improved: {n_sh}/{n} folds  | Mean ΔSharpe: {msh:+.3f}")
    print(f"  Verdict:         {verdict}")
    print(f"{'='*70}")

    print("\n[3/3] Generating charts...")
    make_charts(fold_results, stress_df)
    make_summary_table(fold_results)

    if save:
        os.makedirs("results", exist_ok=True)
        output = {
            "run_date": datetime.now().isoformat(),
            "methodology": "FRED_proxy_kalshi_7yr_walkforward",
            "stress_stats": {
                "mean":   round(float(comp.mean()), 4),
                "p75":    round(float(comp.quantile(0.75)), 4),
                "p90":    round(float(comp.quantile(0.90)), 4),
                "max":    round(float(comp.max()), 4),
            },
            "folds": [{k:v for k,v in r.items()
                       if k not in ("base_returns","kalshi_returns")}
                      for r in fold_results],
            "summary": {
                "n_folds": n, "n_dd_improved": n_dd,
                "n_sharpe_improved": n_sh,
                "mean_dd_delta": round(float(mdd), 4),
                "mean_sharpe_delta": round(float(msh), 4),
                "verdict": verdict,
            }
        }
        with open("results/kalshi_fred_proxy_results.json","w") as f:
            json.dump(output, f, indent=2, default=str)
        print("  JSON saved → results/kalshi_fred_proxy_results.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-results", action="store_true")
    args = p.parse_args()
    run(save=args.save_results)
