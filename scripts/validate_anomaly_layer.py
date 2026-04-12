#!/usr/bin/env python3
"""
Validate AnomalyRegimeLayer:
  STEP 4: IS calibration on 3 known stress periods
  STEP 5: OOS comparison (Sep 2025 – Apr 2026)
"""
import json
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from data.data_store import get_store

DATA_DIR = Path("data/historical/daily")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_prices(syms, start="2017-01-01", end="2026-12-31"):
    """Load multi-asset Close prices from DataStore (local or S3)."""
    store = get_store()
    frames = {}
    for sym in syms:
        df = store.load(sym)
        if df is None:
            continue
        df.columns = [c.capitalize() for c in df.columns]
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        frames[sym] = df["Close"]
    prices = pd.DataFrame(frames).dropna(how="all")
    prices = prices[(prices.index >= start) & (prices.index <= end)]
    return prices


def main():
    print("=" * 70)
    print("ANOMALY LAYER VALIDATION")
    print("=" * 70)

    # Load core assets
    syms = ["SPY", "QQQ", "IWM", "TLT", "GLD", "HYG", "LQD", "EEM",
            "VGK", "XLK", "XLE", "XLF", "VIX", "DXY", "JPY", "EURUSD",
            "BTC", "ETH", "AGG", "SHY"]
    prices = load_prices(syms)
    print(f"Loaded {len(prices.columns)} assets, {len(prices)} days: {prices.index[0].date()} → {prices.index[-1].date()}")

    # Instantiate layer (no config = default weights)
    from regime.anomaly_layer import AnomalyRegimeLayer
    layer = AnomalyRegimeLayer()

    # ── STEP 4: IS Calibration ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: IN-SAMPLE CALIBRATION ON KNOWN STRESS PERIODS")
    print("=" * 70)

    # Use compute_series for vectorised scoring (much faster than per-day)
    print("\nComputing full anomaly series (this uses walk-forward IsolationForest)...")
    # Only use sentiment + FX + isolation (macro requires FRED which may fail)
    # We'll still instantiate full layer but expect graceful degradation

    stress_periods = {
        "Dec 2018 (Vol selloff)":     ("2018-10-01", "2018-12-31"),
        "Mar 2020 (COVID crash)":     ("2020-02-01", "2020-04-30"),
        "Jun 2022 (Rate hike shock)": ("2022-04-01", "2022-07-31"),
        "2024 Calm (baseline)":       ("2024-01-01", "2024-06-30"),
    }

    # Compute series for full range covering all stress periods
    full_result = layer.compute_series(prices, start="2018-01-01", end="2026-04-02")
    print(f"Series computed: {len(full_result)} days, sources: {[c for c in full_result.columns if c not in ('composite','label','scale')]}")

    for label, (s, e) in stress_periods.items():
        sub = full_result.loc[s:e]
        if sub.empty:
            print(f"\n{label}: NO DATA")
            continue
        comp = sub["composite"]
        labels = sub["label"]
        print(f"\n{label}:")
        print(f"  Composite: mean={comp.mean():.3f}  p75={comp.quantile(0.75):.3f}  p95={comp.quantile(0.95):.3f}  max={comp.max():.3f}")
        # Source breakdown
        for src in ["macro", "sentiment", "fx", "isolation"]:
            if src in sub.columns:
                print(f"  {src:>12s}: mean={sub[src].mean():.3f}  max={sub[src].max():.3f}")
        # Regime distribution
        for regime in ["NORMAL", "ELEVATED", "STRESSED", "CRISIS"]:
            pct = (labels == regime).sum() / len(labels) * 100
            if pct > 0:
                print(f"  {regime}: {pct:.1f}%")
        # Expected regime assertion
        peak_label = sub.loc[comp.idxmax(), "label"]
        print(f"  Peak label: {peak_label} (composite={comp.max():.3f})")

    # ── STEP 5: OOS Comparison ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: OUT-OF-SAMPLE COMPARISON (Sep 2025 – Apr 2026)")
    print("=" * 70)

    oos_start = "2025-09-01"
    oos_end = "2026-04-02"
    oos_result = full_result.loc[oos_start:oos_end]
    if len(oos_result) < 10:
        print("Not enough OOS data — skipping")
        return

    # SPY returns for performance comparison
    spy = prices["SPY"].loc[oos_start:oos_end].dropna()
    spy_ret = spy.pct_change().dropna()
    common_idx = spy_ret.index.intersection(oos_result.index)
    spy_ret = spy_ret.reindex(common_idx)

    # Get choppy scale from ChoppyRegimeDetector
    from regime.choppy_regime import ChoppyRegimeDetector
    vix = prices["VIX"] if "VIX" in prices.columns else pd.Series(20.0, index=prices.index)
    choppy_det = ChoppyRegimeDetector()
    choppy_scores = choppy_det.score_series(prices, vix)
    choppy_scales = choppy_scores.apply(lambda s: ChoppyRegimeDetector.score_to_scale(s)[0])

    # Align everything
    choppy_scales = choppy_scales.reindex(common_idx, method="ffill").fillna(1.0)
    anomaly_scales = oos_result["scale"].reindex(common_idx, method="ffill").fillna(1.0)

    # Strategy returns (SPY × scale as simplification)
    # V_choppy_only: SPY returns scaled by choppy detector only
    ret_choppy = spy_ret * choppy_scales
    # V_combined: SPY returns scaled by choppy × anomaly
    combined_scales = choppy_scales * anomaly_scales
    ret_combined = spy_ret * combined_scales

    # Performance metrics
    def metrics(rets, label):
        cum = (1 + rets).cumprod()
        total = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0
        ann_ret = total * (252 / max(len(rets), 1))
        vol = float(rets.std() * np.sqrt(252))
        sharpe = ann_ret / vol if vol > 0 else 0
        maxdd = float((cum / cum.cummax() - 1).min())
        return {
            "strategy": label,
            "total_return": round(total * 100, 2),
            "ann_return": round(ann_ret * 100, 2),
            "ann_vol": round(vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(maxdd * 100, 2),
        }

    m_choppy = metrics(ret_choppy, "V_choppy_only")
    m_combined = metrics(ret_combined, "V_combined")
    m_spy = metrics(spy_ret, "SPY_buy_hold")

    print(f"\n{'Metric':<18s} {'SPY B&H':>12s} {'Choppy Only':>12s} {'Combined':>12s}")
    print("-" * 56)
    for k in ["total_return", "sharpe", "max_drawdown"]:
        unit = "%" if k != "sharpe" else ""
        print(f"{k:<18s} {m_spy[k]:>11.2f}{unit} {m_choppy[k]:>11.2f}{unit} {m_combined[k]:>11.2f}{unit}")

    # Save results
    results = {
        "oos_period": f"{oos_start} to {oos_end}",
        "SPY_buy_hold": m_spy,
        "V_choppy_only": m_choppy,
        "V_combined": m_combined,
        "anomaly_layer_sources": [c for c in oos_result.columns if c not in ("composite", "label", "scale")],
    }
    with open(RESULTS_DIR / "anomaly_layer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/anomaly_layer_results.json")

    # Generate chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        cum_spy = (1 + spy_ret).cumprod()
        cum_choppy = (1 + ret_choppy).cumprod()
        cum_combined = (1 + ret_combined).cumprod()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot(cum_spy.index, cum_spy, label=f"SPY B&H (Sharpe={m_spy['sharpe']:.2f})", color="gray", alpha=0.7)
        ax1.plot(cum_choppy.index, cum_choppy, label=f"Choppy Only (Sharpe={m_choppy['sharpe']:.2f})", color="blue")
        ax1.plot(cum_combined.index, cum_combined, label=f"Combined (Sharpe={m_combined['sharpe']:.2f})", color="green", linewidth=2)
        ax1.set_title("OOS Performance: Sep 2025 – Apr 2026", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        # Anomaly composite score
        comp_oos = oos_result["composite"].reindex(common_idx)
        ax2.fill_between(comp_oos.index, 0, comp_oos, alpha=0.4, color="red", label="Anomaly Composite")
        ax2.axhline(0.20, color="orange", linestyle="--", alpha=0.5, label="ELEVATED")
        ax2.axhline(0.35, color="red", linestyle="--", alpha=0.5, label="STRESSED")
        ax2.axhline(0.50, color="darkred", linestyle="--", alpha=0.5, label="CRISIS")
        ax2.set_ylabel("Anomaly Score")
        ax2.set_ylim(0, 0.8)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "anomaly_layer_oos.png", dpi=150, bbox_inches="tight")
        print(f"Chart saved to results/anomaly_layer_oos.png")
    except Exception as e:
        print(f"Chart generation failed: {e}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
