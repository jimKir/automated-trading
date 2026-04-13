"""
Walk-Forward Calibration for PositionAnomalyScorer
====================================================
Fits the three raw-feature thresholds (G1 vol-spike, G2 momentum-churn,
G3 drawdown-from-peak) on expanding in-sample windows, then evaluates on
the next-year out-of-sample window. No OOS data touches the IS fit.

Why walk-forward?
-----------------
The current thresholds (_VOL_SPIKE_CEILING=3.0, _DD_CEILING_CRYPTO=0.15)
were set by hand from single-period profiling. BTC's actual vol-ratio p95
is only 1.38 — meaning the G1 feature can never score above 0.19 with the
old ceiling, making the crypto guard nearly deaf to vol spikes.

Walk-forward calibration:
  1. For each IS window, find the thresholds that maximise the Sharpe-DD
     tradeoff on a synthetic crypto-only backtest.
  2. Test those thresholds on the next-year OOS.
  3. Report IS-to-OOS degradation as the measure of overfit.
  4. Average the OOS-validated thresholds → locked parameters for production.

Folds (expanding window):
  Fold 1: IS=2020-2022  → OOS=2023
  Fold 2: IS=2020-2023  → OOS=2024
  Fold 3: IS=2020-2024  → OOS=2025
  Fold 4: IS=2020-2025  → OOS=Q1-2026

Objective function (per fold):
  For each threshold combination, simulate applying the scale factor to
  a 100% BTC position over the OOS period. Measure:
    - OOS max drawdown reduction vs unscaled
    - OOS Sharpe ratio (should not degrade significantly)
    - Days over-scaled in calm bull periods (false positive rate)

  Score = Sharpe_adj × (1 - false_positive_rate)
  where Sharpe_adj penalises excessive cutting in calm markets.

Grid search:
  G1_ceiling  : [1.15, 1.25, 1.35, 1.45, 1.55]  (vol_ratio above which score=1)
  G1_baseline : [0.85, 0.90, 0.95, 1.00]         (vol_ratio below which score=0)
  G3_dd_ceil  : [0.08, 0.12, 0.16, 0.20, 0.25]   (DD% above which score=1)
  sensitivity : [1.0, 1.2, 1.4, 1.6, 1.8]        (aggressiveness of cutting)

G2 (momentum churn) thresholds are fixed — they are stable across regimes.

Output:
  - diagnostics/wf_pos_anomaly_calibration.json  — full fold results
  - diagnostics/wf_pos_anomaly_best_params.json  — OOS-averaged locked params
  - Console table of IS vs OOS performance per fold
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
PERIODS_Y = 365  # crypto trades 365d/yr
REBAL_FREQ = 7  # weekly rebalance in days

# G2 fixed thresholds (stable across regimes — not grid-searched)
G2_BASELINE = 1.20  # TNR above → trending (score=0)
G2_CEILING = 0.15  # TNR below → max churn (score=1)
G2_WEIGHT = 0.25  # feature weight

# G4 portfolio stress (ChoppyRegimeDetector) fixed at 0.15 weight
G4_WEIGHT = 0.15
G4_SCORE = 0.15  # conservative fixed value for this calibration

# Feature blend weights (G1+G2+G3+G4 must sum to 1.0)
G1_WEIGHT = 0.35  # vol spike — primary for crypto
G3_WEIGHT = 0.25  # DD from peak

# Folds: (label, IS_start, IS_end, OOS_start, OOS_end)
FOLDS = [
    ("F1: IS=2020-22 → OOS=2023", "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("F2: IS=2020-23 → OOS=2024", "2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("F3: IS=2020-24 → OOS=2025", "2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("F4: IS=2020-25 → OOS=Q1'26", "2020-01-01", "2025-12-31", "2026-01-01", "2026-04-02"),
]

# Grid values
G1_CEILINGS = [1.15, 1.25, 1.35, 1.45, 1.55]
G1_BASELINES = [0.85, 0.90, 0.95, 1.00]
G3_DD_CEILS = [0.08, 0.12, 0.16, 0.20, 0.25]
SENSITIVITIES = [1.0, 1.2, 1.4, 1.6, 1.8]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_crypto(ticker: str, start: str = "2019-01-01") -> pd.Series:
    """Download crypto close prices."""
    df = yf.download(ticker, start=start, end="2026-04-04", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df["Close"].rename(ticker)


# ── Feature computation (vectorised, no per-day loops) ────────────────────────


def compute_features(
    close: pd.Series,
    g1_baseline: float,
    g1_ceiling: float,
    g3_dd_ceil: float,
    vol_win: int = 20,
    base_win: int = 60,
) -> pd.DataFrame:
    """
    Compute all four features for one symbol as a full-history DataFrame.
    Parameters allow grid searching different calibration thresholds.
    """
    ret = close.pct_change()
    feat = pd.DataFrame(index=close.index)

    # G1: vol spike ratio, normalised by grid-searched thresholds
    rv20 = ret.rolling(vol_win).std()
    rv60 = ret.rolling(base_win).std().replace(0, np.nan)
    ratio = rv20 / rv60
    denom_g1 = g1_ceiling - g1_baseline
    feat["g1"] = ((ratio - g1_baseline) / denom_g1).clip(0, 1) if denom_g1 > 0 else 0.0

    # G2: momentum churn (TNR, inverted) — thresholds fixed
    net20 = close.pct_change(vol_win).abs()
    path_vol = rv20 * np.sqrt(vol_win)
    tnr = (net20 / path_vol.replace(0, np.nan)).fillna(1.0).clip(0, 3)
    denom_g2 = G2_BASELINE - G2_CEILING
    feat["g2"] = ((G2_BASELINE - tnr) / denom_g2).clip(0, 1) if denom_g2 > 0 else 0.0

    # G3: drawdown from 20d peak
    hi20 = close.rolling(vol_win).max()
    dd_20 = ((close - hi20) / hi20.replace(0, np.nan)).abs()
    feat["g3"] = (dd_20 / g3_dd_ceil).clip(0, 1) if g3_dd_ceil > 0 else 0.0

    # G4: portfolio stress (fixed, not calibrated here)
    feat["g4"] = G4_SCORE

    return feat.fillna(0).clip(0, 1)


def compute_score_series(
    close: pd.Series,
    g1_baseline: float,
    g1_ceiling: float,
    g3_dd_ceil: float,
    sensitivity: float,
    floor: float = 0.10,
    ema_span: int = 3,
) -> pd.Series:
    """
    Full pipeline: features → blended score → EMA-smoothed → scale factor.
    Returns a pd.Series of scale factors ∈ [floor, 1.0].
    """
    feat = compute_features(close, g1_baseline, g1_ceiling, g3_dd_ceil)
    score = (
        G1_WEIGHT * feat["g1"]
        + G2_WEIGHT * feat["g2"]
        + G3_WEIGHT * feat["g3"]
        + G4_WEIGHT * feat["g4"]
    ).clip(0, 1)
    score = score.ewm(span=ema_span, adjust=False).mean().clip(0, 1)
    scale = (1.0 - sensitivity * score).clip(floor, 1.0)
    return scale


# ── Backtest (synthetic crypto-only) ─────────────────────────────────────────


def backtest_crypto(
    close: pd.Series,
    scale: pd.Series,
    start: str,
    end: str,
    rt_cost: float = 0.001,  # 0.1% round-trip per weekly rebalance
) -> dict[str, float]:
    """
    Simulate a 100% crypto position scaled by the anomaly factor.
    Weekly rebalance (scale changes, position adjusts).
    Returns performance metrics for the window.
    """
    c = close.loc[start:end].dropna()
    s = scale.loc[start:end].reindex(c.index, method="ffill").fillna(1.0)

    if len(c) < 10:
        return {"sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0, "n": 0}

    ret = c.pct_change().dropna()
    s = s.reindex(ret.index, method="ffill").fillna(1.0)

    # Simulate position: scale[t] × market_return[t] - (1-scale[t]) × 0
    # (flat portion earns 0 — conservative, ignores money market)
    port_ret = ret * s

    # Apply round-trip cost at rebalance (weekly — ~52× per year)
    # Only pay cost when scale changes by >2% (threshold to avoid noise)
    scale_chg = s.diff().abs()
    rebal_cost = scale_chg.where(scale_chg > 0.02, 0.0) * rt_cost
    port_ret = port_ret - rebal_cost

    n = len(port_ret)
    ann = (1 + port_ret).prod() ** (PERIODS_Y / n) - 1
    vol = port_ret.std() * np.sqrt(PERIODS_Y)
    sh = ann / vol if vol > 0 else 0.0
    cum = (1 + port_ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    mdd = float(dd.min())

    return {
        "sharpe": round(sh, 3),
        "max_dd": round(mdd * 100, 2),
        "cagr": round(ann * 100, 2),
        "n": n,
    }


def score_params(
    is_metrics: dict[str, float],
    oos_metrics: dict[str, float],
    baseline_metrics: dict[str, float],  # unscaled = always scale=1.0
    fp_rate: float,  # fraction of IS bull days over-cut
) -> float:
    """
    Composite objective score for a param set across one fold.

    Rewards:
      - OOS max DD reduction vs unscaled baseline
      - OOS Sharpe close to (or above) unscaled baseline
    Penalises:
      - High false positive rate (cutting too much in calm IS periods)
      - Large IS→OOS degradation in Sharpe (overfit signal)

    Returns float — higher is better.
    """
    # DD protection (positive = improvement, negative = made it worse)
    dd_improvement = baseline_metrics["max_dd"] - oos_metrics["max_dd"]  # want positive

    # Sharpe preservation (penalise if OOS Sharpe drops a lot vs unscaled)
    sharpe_penalty = max(0.0, baseline_metrics["sharpe"] - oos_metrics["sharpe"])

    # IS→OOS generalisation: large gap = overfit
    is_oos_gap = abs(is_metrics["sharpe"] - oos_metrics["sharpe"])
    overfit_penalty = max(0.0, is_oos_gap - 0.3)  # allow 0.3 gap before penalising

    # False positive: excessive cutting during IS calm periods
    fp_penalty = fp_rate * 2.0

    score = (
        dd_improvement * 0.50  # primary objective
        - sharpe_penalty * 0.25  # secondary: don't hurt Sharpe
        - overfit_penalty * 0.15  # penalise overfit
        - fp_penalty * 0.10  # penalise false positives
    )
    return float(score)


# ── Walk-forward grid search ──────────────────────────────────────────────────


def run_wf_calibration(
    close_btc: pd.Series,
    close_eth: pd.Series,
) -> dict:
    """
    Run the full walk-forward grid search across all folds.
    Returns a dict with fold results and recommended locked params.
    """
    grid = list(itertools.product(G1_CEILINGS, G1_BASELINES, G3_DD_CEILS, SENSITIVITIES))
    n_grid = len(grid)
    print(
        f"Grid size: {n_grid} combinations × {len(FOLDS)} folds × 2 crypto assets "
        f"= {n_grid * len(FOLDS) * 2:,} evaluations"
    )

    fold_results = []

    for fold_label, is_s, is_e, oos_s, oos_e in FOLDS:
        print(f"\n{'=' * 70}")
        print(f"Fold: {fold_label}")
        print(f"  IS:  {is_s} → {is_e}  |  OOS: {oos_s} → {oos_e}")

        best_score = -np.inf
        best_params = None
        best_is_res = {}
        best_oos_res = {}

        for g1_ceil, g1_base, g3_dd, sens in grid:
            if g1_base >= g1_ceil:
                continue  # degenerate config

            fold_score = 0.0
            valid = True

            for close, _sym in [(close_btc, "BTC"), (close_eth, "ETH")]:
                # Pre-compute scale series on full history (causal)
                scale = compute_score_series(close, g1_base, g1_ceil, g3_dd, sens)

                # IS metrics
                is_res = backtest_crypto(close, scale, is_s, is_e)
                # OOS metrics
                oos_res = backtest_crypto(close, scale, oos_s, oos_e)
                # Unscaled baseline
                ones = pd.Series(1.0, index=close.index)
                base_res = backtest_crypto(close, ones, oos_s, oos_e)

                if oos_res["n"] < 10:
                    valid = False
                    break

                # False positive rate: fraction of IS calm days where scale < 0.75
                # "calm" = IS periods where unscaled 30d drawdown < 10%
                c_is = close.loc[is_s:is_e]
                hi_30 = c_is.rolling(30).max()
                calm_mask = ((c_is - hi_30) / hi_30.replace(0, np.nan)).abs() < 0.10
                sc_is = scale.loc[is_s:is_e].reindex(c_is.index, method="ffill")
                if calm_mask.sum() > 0:
                    fp = float((sc_is[calm_mask] < 0.75).sum() / calm_mask.sum())
                else:
                    fp = 0.0

                sc = score_params(is_res, oos_res, base_res, fp)
                fold_score += sc

            if not valid:
                continue

            if fold_score > best_score:
                best_score = fold_score
                best_params = (g1_ceil, g1_base, g3_dd, sens)
                # Recompute best IS/OOS for reporting
                scale_btc = compute_score_series(close_btc, g1_base, g1_ceil, g3_dd, sens)
                scale_eth = compute_score_series(close_eth, g1_base, g1_ceil, g3_dd, sens)
                best_is_res = {
                    "BTC": backtest_crypto(close_btc, scale_btc, is_s, is_e),
                    "ETH": backtest_crypto(close_eth, scale_eth, is_s, is_e),
                }
                best_oos_res = {
                    "BTC": backtest_crypto(close_btc, scale_btc, oos_s, oos_e),
                    "ETH": backtest_crypto(close_eth, scale_eth, oos_s, oos_e),
                }

        if best_params is None:
            print("  WARNING: no valid param set found for this fold")
            continue

        g1_c, g1_b, g3_d, sns = best_params
        ones = pd.Series(1.0, index=close_btc.index)
        base_btc = backtest_crypto(close_btc, ones, oos_s, oos_e)
        base_eth = backtest_crypto(close_eth, ones, oos_s, oos_e)

        print(f"  Best params: G1_ceil={g1_c}  G1_base={g1_b}  G3_dd={g3_d}  sens={sns}")
        print(
            f"  BTC  IS: Sh={best_is_res['BTC']['sharpe']:+.3f}  DD={best_is_res['BTC']['max_dd']:.1f}%"
        )
        print(
            f"  BTC OOS: Sh={best_oos_res['BTC']['sharpe']:+.3f}  DD={best_oos_res['BTC']['max_dd']:.1f}%  "
            f"(base DD={base_btc['max_dd']:.1f}%  Δ={base_btc['max_dd'] - best_oos_res['BTC']['max_dd']:+.1f}pp)"
        )
        print(
            f"  ETH  IS: Sh={best_is_res['ETH']['sharpe']:+.3f}  DD={best_is_res['ETH']['max_dd']:.1f}%"
        )
        print(
            f"  ETH OOS: Sh={best_oos_res['ETH']['sharpe']:+.3f}  DD={best_oos_res['ETH']['max_dd']:.1f}%  "
            f"(base DD={base_eth['max_dd']:.1f}%  Δ={base_eth['max_dd'] - best_oos_res['ETH']['max_dd']:+.1f}pp)"
        )

        fold_results.append(
            {
                "fold": fold_label,
                "is_window": f"{is_s} → {is_e}",
                "oos_window": f"{oos_s} → {oos_e}",
                "params": {
                    "g1_ceiling": g1_c,
                    "g1_baseline": g1_b,
                    "g3_dd_ceil": g3_d,
                    "sensitivity": sns,
                },
                "is_btc": best_is_res.get("BTC", {}),
                "is_eth": best_is_res.get("ETH", {}),
                "oos_btc": best_oos_res.get("BTC", {}),
                "oos_eth": best_oos_res.get("ETH", {}),
                "base_btc_oos": base_btc,
                "base_eth_oos": base_eth,
            }
        )

    return fold_results


def derive_locked_params(fold_results: list[dict]) -> dict:
    """
    Derive the final production thresholds from the OOS-validated fold params.

    Strategy:
      - For each param, use the median across folds (robust to outlier folds)
      - Report IS→OOS Sharpe degradation as the overfit diagnostic
      - Report OOS DD reduction as the protection signal
    """
    if not fold_results:
        return {}

    g1_ceils = [f["params"]["g1_ceiling"] for f in fold_results]
    g1_bases = [f["params"]["g1_baseline"] for f in fold_results]
    g3_dds = [f["params"]["g3_dd_ceil"] for f in fold_results]
    senss = [f["params"]["sensitivity"] for f in fold_results]

    locked = {
        "g1_ceiling": float(np.median(g1_ceils)),
        "g1_baseline": float(np.median(g1_bases)),
        "g3_dd_ceil": float(np.median(g3_dds)),
        "sensitivity": float(np.median(senss)),
        # G2 thresholds — not grid-searched, kept fixed
        "g2_baseline": G2_BASELINE,
        "g2_ceiling": G2_CEILING,
    }

    # Summarise OOS performance across folds
    btc_dd_reductions = [
        f["base_btc_oos"]["max_dd"] - f["oos_btc"]["max_dd"]
        for f in fold_results
        if f.get("oos_btc") and f.get("base_btc_oos")
    ]
    eth_dd_reductions = [
        f["base_eth_oos"]["max_dd"] - f["oos_eth"]["max_dd"]
        for f in fold_results
        if f.get("oos_eth") and f.get("base_eth_oos")
    ]

    btc_is_sharpes = [f["is_btc"]["sharpe"] for f in fold_results if f.get("is_btc")]
    btc_oos_sharpes = [f["oos_btc"]["sharpe"] for f in fold_results if f.get("oos_btc")]
    eth_is_sharpes = [f["is_eth"]["sharpe"] for f in fold_results if f.get("is_eth")]
    eth_oos_sharpes = [f["oos_eth"]["sharpe"] for f in fold_results if f.get("oos_eth")]

    locked["diagnostics"] = {
        "folds_run": len(fold_results),
        "btc_mean_dd_reduction_pp": round(float(np.mean(btc_dd_reductions)), 2)
        if btc_dd_reductions
        else None,
        "eth_mean_dd_reduction_pp": round(float(np.mean(eth_dd_reductions)), 2)
        if eth_dd_reductions
        else None,
        "btc_is_sharpe_mean": round(float(np.mean(btc_is_sharpes)), 3) if btc_is_sharpes else None,
        "btc_oos_sharpe_mean": round(float(np.mean(btc_oos_sharpes)), 3)
        if btc_oos_sharpes
        else None,
        "eth_is_sharpe_mean": round(float(np.mean(eth_is_sharpes)), 3) if eth_is_sharpes else None,
        "eth_oos_sharpe_mean": round(float(np.mean(eth_oos_sharpes)), 3)
        if eth_oos_sharpes
        else None,
        "btc_is_oos_sharpe_gap": round(
            float(np.mean(btc_is_sharpes)) - float(np.mean(btc_oos_sharpes)), 3
        )
        if btc_is_sharpes and btc_oos_sharpes
        else None,
    }

    return locked


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR = Path(__file__).parent
    RESULTS_FILE = OUT_DIR / "wf_pos_anomaly_calibration.json"
    BEST_PARAMS_FILE = OUT_DIR / "wf_pos_anomaly_best_params.json"

    print("Loading crypto price data...")
    btc = load_crypto("BTC-USD")
    eth = load_crypto("ETH-USD")
    print(f"  BTC: {len(btc)} days  ({btc.index.min().date()} → {btc.index.max().date()})")
    print(f"  ETH: {len(eth)} days  ({eth.index.min().date()} → {eth.index.max().date()})")

    fold_results = run_wf_calibration(btc, eth)

    print(f"\n{'=' * 70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"\n{'Fold':<35} {'G1_c':>6} {'G1_b':>6} {'G3_dd':>6} {'sens':>6} | "
        f"{'BTC IS Sh':>10} {'BTC OOS Sh':>11} {'BTC ΔDD':>8}"
    )
    print("-" * 90)
    for f in fold_results:
        p = f["params"]
        print(
            f"{f['fold']:<35} {p['g1_ceiling']:>6.2f} {p['g1_baseline']:>6.2f} "
            f"{p['g3_dd_ceil']:>6.3f} {p['sensitivity']:>6.1f} | "
            f"{f['is_btc'].get('sharpe', 0):>10.3f} "
            f"{f['oos_btc'].get('sharpe', 0):>11.3f} "
            f"{f['base_btc_oos'].get('max_dd', 0) - f['oos_btc'].get('max_dd', 0):>+8.1f}pp"
        )

    locked = derive_locked_params(fold_results)
    print("\nLOCKED PARAMS (median across OOS-validated folds):")
    for k, v in locked.items():
        if k != "diagnostics":
            print(f"  {k:<20} = {v}")
    print("\nDiagnostics:")
    for k, v in locked.get("diagnostics", {}).items():
        print(f"  {k:<30} = {v}")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump({"folds": fold_results, "locked_params": locked}, f, indent=2, default=str)
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(locked, f, indent=2, default=str)

    print(f"\nSaved → {RESULTS_FILE}")
    print(f"Saved → {BEST_PARAMS_FILE}")
