"""
Walk-Forward Overfitting Validator
====================================
Tests whether the vol targeting improvement is genuine or an artefact of
in-sample fitting. Uses three independent methods:

METHOD 1 — Expanding Walk-Forward
──────────────────────────────────
  Train on 2018–2020, test on 2021–2022
  Train on 2018–2022, test on 2023–2024
  Train on 2018–2024, test on 2025
  → Does vol targeting consistently improve Sharpe on unseen periods?

METHOD 2 — Sensitivity Analysis (Parameter Stability)
──────────────────────────────────────────────────────
  Test target_vol ∈ {0.10, 0.12, 0.15, 0.18, 0.20}
  → Is the improvement robust across a range of target vols,
    or does it only work for one specific tuned value?
  A genuine improvement is monotone / plateau-shaped across this range.
  A spurious improvement shows a sharp peak at one specific value.

METHOD 3 — Permutation Test (Statistical Significance)
────────────────────────────────────────────────────────
  Shuffle the equity return series 500 times randomly.
  Apply vol targeting to each shuffled series.
  → Does vol targeting improve Sharpe on random noise?
  If it does, the improvement is statistical artefact, not a real edge.
  p-value = fraction of shuffled Sharpes that beat real Sharpe.
  We require p < 0.05 (only 5% of random shuffles beat us).

All three methods must pass for the improvement to be considered genuine.
Failing even one is a red flag worth investigating.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("WFValidator")

PERIODS_PER_YEAR = 252
RISK_FREE_RATE   = 0.04 / PERIODS_PER_YEAR


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return np.nan
    excess = returns - RISK_FREE_RATE
    return float(excess.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR))


def _max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a positive fraction."""
    eq = (1 + returns).cumprod()
    dd = (eq / eq.cummax() - 1)
    return float(abs(dd.min()))


def _ann_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(PERIODS_PER_YEAR))


def _ann_return(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    return float((1 + returns.mean()) ** PERIODS_PER_YEAR - 1)


def _apply_vol_targeting(
    returns: pd.Series,
    target_vol: float = 0.15,
    vol_window: int = 21,
    max_leverage: float = 1.5,
    min_leverage: float = 0.1,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply vol targeting to a return series.
    Returns (scaled_returns, scale_factors).
    All computations strictly causal — scale[t] uses only returns[0..t-1].
    """
    lam = 0.94
    warmup = vol_window * 2

    # Seed EWMA variance from first warmup returns
    seed_returns = returns.iloc[:warmup].dropna()
    if len(seed_returns) < 5:
        return returns.copy(), pd.Series(1.0, index=returns.index)

    ewma_var = float(seed_returns.var())
    scales   = pd.Series(np.nan, index=returns.index)

    for i in range(len(returns)):
        if i > 0:
            prev_r = returns.iloc[i - 1]
            if not np.isnan(prev_r):
                ewma_var = lam * ewma_var + (1 - lam) * (prev_r ** 2)

        if i < warmup:
            scales.iloc[i] = 1.0
            continue

        ann_vol = np.sqrt(ewma_var * PERIODS_PER_YEAR)
        if ann_vol <= 0:
            scales.iloc[i] = 1.0
        else:
            raw = target_vol / ann_vol
            scales.iloc[i] = float(np.clip(raw, min_leverage, max_leverage))

    # Smooth scale (3-day EWM)
    scales = scales.ewm(span=3, adjust=False).mean().fillna(1.0)

    # Apply: scaled return = scale[t] * return[t]
    # But scale[t] is computed from returns[0..t-1], so this is causal
    scaled = returns * scales
    return scaled, scales


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: Expanding Walk-Forward
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_test(
    returns: pd.Series,
    target_vol: float = 0.15,
    folds: Optional[List[Tuple[str, str, str, str]]] = None,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward test.

    Each fold: train on [train_start, train_end], test on [test_start, test_end].
    The vol target is FIXED (never fitted to train data — it's a constant).
    The warmup period uses train data to seed the EWMA variance.

    Returns DataFrame with one row per fold showing baseline vs vol-targeted metrics.
    """
    if folds is None:
        folds = [
            ("2018-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),  # COVID + recovery
            ("2018-01-01", "2021-12-31", "2022-01-01", "2023-06-30"),  # Rate hike bear
            ("2018-01-01", "2023-06-30", "2023-07-01", "2024-12-31"),  # Bull + AI rally
            ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),  # 2025 holdout
        ]

    results = []
    for train_start, train_end, test_start, test_end in folds:
        # Seed EWMA from training data
        train_rets = returns[
            (returns.index >= train_start) & (returns.index <= train_end)
        ].dropna()

        test_rets = returns[
            (returns.index >= test_start) & (returns.index <= test_end)
        ].dropna()

        if len(test_rets) < 20:
            log.warning(f"WF fold {test_start}–{test_end}: insufficient test data, skipping")
            continue

        # Seed EWMA variance from training data
        if len(train_rets) > 5:
            seed_var = float(train_rets.var())
        else:
            seed_var = (target_vol / np.sqrt(PERIODS_PER_YEAR)) ** 2

        lam = 0.94
        ewma_var = seed_var
        # Advance EWMA through the training period
        for r in train_rets:
            ewma_var = lam * ewma_var + (1 - lam) * (r ** 2)

        # Apply vol targeting on test period only, using seeded EWMA
        scales   = []
        curr_var = ewma_var
        for i, r in enumerate(test_rets):
            ann_vol = np.sqrt(curr_var * PERIODS_PER_YEAR)
            scale   = float(np.clip(
                target_vol / ann_vol if ann_vol > 0 else 1.0, 0.1, 1.5
            ))
            scales.append(scale)
            curr_var = lam * curr_var + (1 - lam) * (r ** 2)

        scale_series    = pd.Series(scales, index=test_rets.index)
        scale_smooth    = scale_series.ewm(span=3, adjust=False).mean()
        scaled_test     = test_rets * scale_smooth

        base_sharpe  = _sharpe(test_rets)
        vt_sharpe    = _sharpe(scaled_test)
        base_mdd     = _max_drawdown(test_rets)
        vt_mdd       = _max_drawdown(scaled_test)
        base_vol     = _ann_vol(test_rets)
        vt_vol       = _ann_vol(scaled_test)
        base_ret     = _ann_return(test_rets)
        vt_ret       = _ann_return(scaled_test)

        results.append({
            "fold":               f"{test_start[:7]} → {test_end[:7]}",
            "test_days":          len(test_rets),
            "base_sharpe":        round(base_sharpe, 4),
            "vt_sharpe":          round(vt_sharpe,   4),
            "sharpe_delta":       round(vt_sharpe - base_sharpe, 4),
            "sharpe_improved":    vt_sharpe > base_sharpe,
            "base_mdd_pct":       round(base_mdd * 100, 2),
            "vt_mdd_pct":         round(vt_mdd   * 100, 2),
            "mdd_reduced":        vt_mdd < base_mdd,
            "base_vol_pct":       round(base_vol * 100, 2),
            "vt_vol_pct":         round(vt_vol   * 100, 2),
            "base_ann_ret_pct":   round(base_ret * 100, 2),
            "vt_ann_ret_pct":     round(vt_ret   * 100, 2),
            "avg_scale":          round(float(scale_smooth.mean()), 3),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: Sensitivity / Parameter Stability
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_test(
    returns: pd.Series,
    target_vols: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Test Sharpe across a range of target volatilities.
    A genuine improvement shows a wide plateau — many values work.
    A spurious improvement shows a sharp peak at one specific value.
    """
    if target_vols is None:
        target_vols = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

    baseline_sharpe = _sharpe(returns)
    baseline_mdd    = _max_drawdown(returns)
    baseline_vol    = _ann_vol(returns)

    results = []
    for tv in target_vols:
        scaled, scales = _apply_vol_targeting(returns, target_vol=tv)
        results.append({
            "target_vol_pct":    round(tv * 100, 0),
            "sharpe":            round(_sharpe(scaled),        4),
            "ann_vol_pct":       round(_ann_vol(scaled) * 100, 2),
            "max_drawdown_pct":  round(_max_drawdown(scaled) * 100, 2),
            "ann_return_pct":    round(_ann_return(scaled) * 100, 2),
            "avg_scale":         round(float(scales.mean()),   3),
            "vs_baseline_sharpe": round(_sharpe(scaled) - baseline_sharpe, 4),
        })

    df = pd.DataFrame(results)

    # Check for sharp peak (overfitting warning)
    sharpes = df["sharpe"].values
    peak_idx = np.argmax(sharpes)
    is_plateau = (
        sharpes.max() - sharpes.min() < 0.3 and  # range < 0.3 Sharpe units
        (sharpes > baseline_sharpe).sum() >= 4    # at least 4 of 7 values beat baseline
    )
    log.info(
        f"Sensitivity: baseline_sharpe={baseline_sharpe:.4f}  "
        f"best_tv={target_vols[peak_idx]:.0%}  "
        f"plateau={'YES' if is_plateau else 'NO (potential overfit)'}"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: Permutation Test (Statistical Significance)
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(
    returns: pd.Series,
    target_vol: float = 0.15,
    n_permutations: int = 500,
    random_seed: int = 42,
) -> dict:
    """
    Randomly shuffle the return series and apply vol targeting to each shuffle.
    Measures whether vol targeting adds genuine value beyond chance.

    Under the null hypothesis (vol targeting adds no value), shuffled returns
    should produce similar Sharpe improvements as the original series.
    A low p-value means the improvement is unlikely to be due to chance.

    n_permutations=500 is sufficient for p-value precision to ±0.02.
    """
    rng = np.random.default_rng(random_seed)

    # Real improvement
    scaled_real, _ = _apply_vol_targeting(returns, target_vol=target_vol)
    real_sharpe_base   = _sharpe(returns)
    real_sharpe_scaled = _sharpe(scaled_real)
    real_improvement   = real_sharpe_scaled - real_sharpe_base

    log.info(
        f"Permutation test: real improvement = {real_improvement:+.4f} Sharpe units  "
        f"({n_permutations} permutations, seed={random_seed})"
    )

    # Null distribution
    null_improvements = []
    ret_values = returns.dropna().values.copy()

    for i in range(n_permutations):
        shuffled_vals = rng.permutation(ret_values)
        shuffled = pd.Series(shuffled_vals, index=returns.dropna().index)

        scaled_shuf, _ = _apply_vol_targeting(shuffled, target_vol=target_vol)
        null_base   = _sharpe(shuffled)
        null_scaled = _sharpe(scaled_shuf)
        null_improvements.append(null_scaled - null_base)

        if (i + 1) % 100 == 0:
            log.info(f"  Permutation {i+1}/{n_permutations}...")

    null_improvements = np.array(null_improvements)

    # p-value: fraction of null improvements >= real improvement
    p_value = float((null_improvements >= real_improvement).mean())

    result = {
        "real_sharpe_baseline":    round(real_sharpe_base,   4),
        "real_sharpe_vol_targeted": round(real_sharpe_scaled, 4),
        "real_improvement":        round(real_improvement,    4),
        "null_mean_improvement":   round(float(null_improvements.mean()), 4),
        "null_std_improvement":    round(float(null_improvements.std()),  4),
        "null_p95":                round(float(np.percentile(null_improvements, 95)), 4),
        "p_value":                 round(p_value, 4),
        "significant_at_5pct":     p_value < 0.05,
        "significant_at_10pct":    p_value < 0.10,
        "n_permutations":          n_permutations,
    }

    log.info(
        f"Permutation result: p={p_value:.4f}  "
        f"significant={'YES' if p_value < 0.05 else 'NO'}  "
        f"null_mean={null_improvements.mean():.4f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Master validator — runs all three methods
# ─────────────────────────────────────────────────────────────────────────────

def run_full_validation(
    returns: pd.Series,
    target_vol: float = 0.15,
    n_permutations: int = 500,
) -> dict:
    """
    Run all three anti-overfitting validation methods.
    Returns a summary dict with pass/fail verdicts.
    """
    log.info("=" * 65)
    log.info("  VOLATILITY TARGETING — OVERFITTING VALIDATION")
    log.info("=" * 65)
    t0 = time.time()

    # Method 1: Walk-forward
    log.info("\nMethod 1: Expanding walk-forward...")
    wf_df = walk_forward_test(returns, target_vol=target_vol)
    wf_pass_count = int(wf_df["sharpe_improved"].sum())
    wf_total      = len(wf_df)
    wf_pass       = wf_pass_count >= (wf_total * 0.75)  # 75% of folds must improve

    # Method 2: Sensitivity
    log.info("\nMethod 2: Sensitivity analysis...")
    sens_df = sensitivity_test(returns)
    beat_baseline = (sens_df["vs_baseline_sharpe"] > 0).sum()
    sens_pass = beat_baseline >= 4  # at least 4 of 7 target vols beat baseline

    # Method 3: Permutation
    log.info("\nMethod 3: Permutation test (500 shuffles)...")
    perm_result = permutation_test(returns, target_vol=target_vol, n_permutations=n_permutations)
    perm_pass   = perm_result["significant_at_5pct"]

    # Overall verdict
    all_pass = wf_pass and sens_pass and perm_pass

    elapsed = time.time() - t0

    summary = {
        "overall_pass":        all_pass,
        "verdict": ("GENUINE EDGE ✓" if all_pass else "WARNING — REVIEW RESULTS"),
        "method1_wf_pass":     wf_pass,
        "method1_folds_improved": f"{wf_pass_count}/{wf_total}",
        "method1_wf_details":  wf_df,
        "method2_sens_pass":   sens_pass,
        "method2_beat_baseline": f"{beat_baseline}/{len(sens_df)}",
        "method2_sens_details": sens_df,
        "method3_perm_pass":   perm_pass,
        "method3_p_value":     perm_result["p_value"],
        "method3_details":     perm_result,
        "elapsed_seconds":     round(elapsed, 1),
    }

    log.info("\n" + "=" * 65)
    log.info(f"  VALIDATION SUMMARY  ({elapsed:.0f}s)")
    log.info("=" * 65)
    log.info(f"  Method 1 — Walk-Forward:       {'PASS ✓' if wf_pass else 'FAIL ✗'}  "
             f"({wf_pass_count}/{wf_total} folds improved)")
    log.info(f"  Method 2 — Sensitivity:        {'PASS ✓' if sens_pass else 'FAIL ✗'}  "
             f"({beat_baseline}/{len(sens_df)} target vols beat baseline)")
    log.info(f"  Method 3 — Permutation test:   {'PASS ✓' if perm_pass else 'FAIL ✗'}  "
             f"(p={perm_result['p_value']:.4f})")
    log.info(f"  OVERALL: {summary['verdict']}")
    log.info("=" * 65)

    return summary
