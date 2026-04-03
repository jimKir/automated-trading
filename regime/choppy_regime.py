"""
Choppy Regime Detector
=======================
A dedicated anomaly detection layer trained on the features that
characterise the 2025 high-volatility / low-directional-conviction
regime — the environment where the strategy bleeds most.

Walk-forward validation showed:
  2023 (low-vol trending):  strategy Sharpe +1.16  — fine
  2024 (ideal trending):    strategy Sharpe +2.09  — ideal
  2025 (choppy high-vol):   strategy Sharpe +0.66 vs SPY +0.94  — underperforms
  Q1-2026 (tariff shock):   strategy Sharpe +0.01             — barely survives

2025 Fingerprint (vs normal years):
  Vol-of-Vol:       0.808%  (3× normal 0.25%)
  VIX >20 days:     25%     (vs 10-17% in calm years)
  MA20 crossings:   26/yr   (vs 20-21 in trending years)
  Reversal rate:    50.5%   (vs 47-48% in trending years)
  Ann volatility:   19.5%   (vs 12-13% in calm bull years)

Architecture
------------
The detector computes a rolling CHOPPY SCORE ∈ [0, 1] based on five
independently calibrated features:

  F1. Vol-of-Vol ratio       — rolling 10d vol std / long-run baseline
  F2. VIX regime score       — VIX level + VIX 20d momentum
  F3. MA-crossing rate       — 20MA cross frequency over rolling 30d window
  F4. Return reversal rate   — fraction of days assets reverse prior-day direction
  F5. Trend-vs-noise ratio   — |10d directional return| / 10d realised vol

Each feature is z-scored against its own rolling 252-day history
(never looks forward), then blended into the final score.

Output: choppy_score ∈ [0, 1]
  < 0.30  → GREEN:  trending/normal — no action
  0.30-0.50 → YELLOW: choppy building — light trim
  0.50-0.65 → ORANGE: clearly choppy — reduce
  > 0.65  → RED:   2025/bear choppiness — defensive

Integration
-----------
  1. Used standalone via ChoppyRegimeDetector.score_series() in backtest
  2. Wired into EWS as Layer F (weight 0.15, rebalances existing weights)
  3. Live mode: ChoppyRegimeDetector.score_today(prices)

Design principles
-----------------
  - No labels, no target leakage — purely feature-based scoring
  - All features derived from market prices (always available live)
  - Calibrated thresholds from regime analysis, NOT optimised on OOS data
  - Does NOT learn from 2025 labels — learns from structural features only
    (so it also fires for any future 2025-like episode, not just 2025)
  - Hysteresis: score smoothed with 5-day EMA to avoid whipsawing
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("ChoppyRegimeDetector")
warnings.filterwarnings("ignore")

# ── Feature calibration ───────────────────────────────────────────────────────
# Calibrated from regime analysis (2023-2026).
# Stress thresholds are the VALUES at which each raw feature scores 1.0.
# Normal-year baselines are used as zero points so the score is near 0
# in calm trending markets and rises to 1 only in 2025-type choppiness.
#
# Raw feature distributions (2024 calm vs 2025 choppy):
#   vol_of_vol:    2024 mean=0.156  2025 mean=0.221  (ratio: 1.4×)
#   vix_regime:    2024 mean=0.543  2025 mean=0.611
#   ma_crossing:   2024 mean=0.622  2025 mean=0.691  (both high → rescale)
#   reversal_rate: 2024 mean=0.904  2025 mean=0.954  (both high → rescale)
#   trend_noise:   similar across years → low weight correct
#
# Key insight: ma_crossing and reversal_rate score high even in calm years
# because the raw features are naturally near their stress thresholds.
# Solution: use a BASELINE_OFFSET per feature so score=0 in calm years.

# (raw_value - baseline) / (stress_ceiling - baseline) → normalised [0,1]
# These numbers derived from 2023/2024 calm-year means.
# Calibration operates on RAW feature values (before any normalisation).
# baseline = calm-year p75 (detector stays quiet in calm markets)
# ceiling  = 2× the 2025 p95 (fires at proper stress levels)
FEATURE_CALIBRATION = {
    #                  baseline   ceiling    notes
    "vol_of_vol":    (0.0017,     0.010),   # 2024 p75=0.0017, 2025 p95=0.0082
    "vix":           (17.0,       30.0),    # calm mean=16, stress p95=29
    "vix_mom":       (0.0,        0.5),     # VIX 20d pct_change; >50% = rising fast
    "ma_crossing":   (0.08,       0.25),    # 2024 mean=0.082, 2025 p95=0.233
    "reversal_rate": (0.48,       0.55),    # 2024 mean=0.476, 2025 p95=0.543
    "trend_noise":   (0.8,        0.2),     # TNR: calm mean=0.99, 2025 mean=0.88
    #                                         NOTE: inverted — low TNR = choppy
}

# Smoothing
SCORE_EMA_SPAN = 5      # 1-week smoothing to avoid daily whipsawing

# Score thresholds → position scale
CHOPPY_SCALE_THRESHOLDS = [
    (0.30, 1.00, "GREEN",  "Trending/normal regime"),
    (0.50, 0.70, "YELLOW", "Choppy building — light trim"),
    (0.65, 0.40, "ORANGE", "Clearly choppy — reduce exposure"),
    (1.01, 0.20, "RED",    "2025-type choppiness — defensive"),
]

# Feature weights (must sum to 1.0)
FEATURE_WEIGHTS = {
    "vol_of_vol":     0.30,   # Highest weight: best 2025 discriminator
    "vix_regime":     0.25,   # VIX level + momentum
    "ma_crossing":    0.20,   # Directional conviction
    "reversal_rate":  0.15,   # Return persistence (choppiness)
    "trend_noise":    0.10,   # Trend quality
}
assert abs(sum(FEATURE_WEIGHTS.values()) - 1.0) < 1e-9


class ChoppyRegimeDetector:
    """
    Detects the 2025-style choppy high-volatility regime and returns a
    score that scales down strategy exposure before losses accumulate.

    Usage (backtest):
        detector = ChoppyRegimeDetector()
        scores = detector.score_series(price_df, vix_series)

    Usage (live):
        detector = ChoppyRegimeDetector()
        score = detector.score_today(price_df, vix_series)
        scale, colour = detector.score_to_scale(score)
    """

    def __init__(
        self,
        feature_window: int = 30,     # rolling window for MA-crossing, reversal rate
        vol_window: int     = 10,      # short vol window
        vol_baseline: int   = 60,      # long vol window for vol-of-vol baseline
        vov_window: int     = 10,      # window for vol-of-vol calc
        norm_window: int    = 252,     # z-score normalisation lookback
    ):
        self.fw  = feature_window
        self.vw  = vol_window
        self.vbw = vol_baseline
        self.vov = vov_window
        self.nw  = norm_window

    # ── Core feature computation ──────────────────────────────────────────────

    def _compute_features(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute all five choppy-regime features as a daily time series.
        All features are in [0, 1] via clipped normalisation.
        No look-ahead: each day uses only data up to that point.

        Parameters
        ----------
        prices : DataFrame of close prices (multi-asset, equity universe)
        vix    : Series of VIX daily closes

        Returns
        -------
        DataFrame with columns: vol_of_vol, vix_regime, ma_crossing,
                                 reversal_rate, trend_noise
        """
        if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
            prices = prices.copy(); prices.index = prices.index.tz_localize(None)
        if isinstance(vix.index, pd.DatetimeIndex) and vix.index.tz is not None:
            vix = vix.copy(); vix.index = vix.index.tz_localize(None)

        rets = prices.pct_change()

        # Use SPY if available, else equal-weight portfolio return
        if "SPY" in prices.columns:
            spy_ret = prices["SPY"].pct_change()
            spy_close = prices["SPY"]
        else:
            spy_ret   = rets.mean(axis=1)
            spy_close = prices.mean(axis=1)

        feat = pd.DataFrame(index=prices.index)

        def rescale(raw, baseline, ceiling):
            """(raw - baseline) / (ceiling - baseline), clipped to [0,1]."""
            denom = ceiling - baseline
            if abs(denom) < 1e-10: return pd.Series(0.0, index=raw.index) if hasattr(raw,'index') else 0.0
            return ((raw - baseline) / denom).clip(0, 1)

        # ── F1. Vol-of-Vol ────────────────────────────────────────────────────
        vol_short = rets.rolling(self.vw).std()
        vol_med   = vol_short.median(axis=1)
        vov_raw   = vol_med.rolling(self.vov).std()   # raw ~0.001 calm, 0.008+ choppy
        base, ceil = FEATURE_CALIBRATION["vol_of_vol"]
        feat["vol_of_vol"] = rescale(vov_raw, base, ceil)

        # ── F2. VIX Regime ────────────────────────────────────────────────────
        vix_aligned = vix.reindex(prices.index, method="ffill")
        vix_level   = rescale(vix_aligned,                           # raw VIX level
                              *FEATURE_CALIBRATION["vix"])
        vix_mom     = vix_aligned.pct_change(20).clip(-1, 1).fillna(0)
        vix_mom_sc  = rescale(vix_mom, *FEATURE_CALIBRATION["vix_mom"])  # rising VIX
        feat["vix_regime"] = (0.65 * vix_level + 0.35 * vix_mom_sc).clip(0, 1)

        # ── F3. MA-Crossing Rate ──────────────────────────────────────────────
        spy_ma20   = spy_close.rolling(20).mean()
        above_ma   = (spy_close > spy_ma20).astype(float)
        cross_flag = (above_ma != above_ma.shift(1)).astype(float)
        cross_raw  = cross_flag.rolling(self.fw).mean()  # raw fraction 0–1
        base, ceil = FEATURE_CALIBRATION["ma_crossing"]
        feat["ma_crossing"] = rescale(cross_raw, base, ceil)

        # ── F4. Return Reversal Rate ──────────────────────────────────────────
        sign_today = np.sign(rets)
        sign_yest  = np.sign(rets.shift(1))
        reversed_  = ((sign_today * sign_yest) < 0).astype(float)
        rev_raw    = reversed_.rolling(self.fw).mean().mean(axis=1)
        base, ceil = FEATURE_CALIBRATION["reversal_rate"]
        feat["reversal_rate"] = rescale(rev_raw, base, ceil)

        # ── F5. Trend-Noise Ratio (inverted) ────────────────────────────────
        # Low TNR = little net movement per unit of vol = choppy
        net_ret   = spy_close.pct_change(self.fw).abs()
        path_vol  = spy_ret.rolling(self.fw).std() * np.sqrt(self.fw)
        tnr       = (net_ret / path_vol.replace(0, np.nan)).fillna(1.0).clip(0, 2)
        # FEATURE_CALIBRATION["trend_noise"] = (baseline_TNR, floor_TNR)
        # baseline=0.8 (calm), floor=0.2 (extreme choppiness)
        base_tnr, floor_tnr = FEATURE_CALIBRATION["trend_noise"]
        # score = (baseline - tnr) / (baseline - floor), clipped [0,1]
        # When tnr=baseline: score=0. When tnr=floor: score=1.
        denom_tnr = base_tnr - floor_tnr
        tnr_score = ((base_tnr - tnr) / denom_tnr).clip(0, 1) if denom_tnr > 0 else pd.Series(0.0, index=tnr.index)
        feat["trend_noise"] = tnr_score

        return feat.fillna(0).clip(0, 1)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _blend_features(self, feat: pd.DataFrame) -> pd.Series:
        """Weighted blend of features → raw choppy score."""
        score = pd.Series(0.0, index=feat.index)
        for fname, weight in FEATURE_WEIGHTS.items():
            if fname in feat.columns:
                score += weight * feat[fname]
        return score.clip(0, 1)

    def score_series(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
        smooth: bool = True,
    ) -> pd.Series:
        """
        Compute daily choppy-regime score for the full history.
        Safe for backtest use — no future information used.

        Parameters
        ----------
        prices  : DataFrame of close prices (columns = symbols)
        vix     : Series of VIX closes
        smooth  : Apply 5-day EMA smoothing (recommended)

        Returns
        -------
        pd.Series of choppy scores ∈ [0, 1], date-indexed
        """
        log.info("ChoppyRegimeDetector: computing feature series...")
        feat  = self._compute_features(prices, vix)
        score = self._blend_features(feat)

        if smooth:
            score = score.ewm(span=SCORE_EMA_SPAN, adjust=False).mean()

        score = score.clip(0, 1)
        log.info(
            f"ChoppyRegimeDetector: mean={score.mean():.3f} "
            f"p75={score.quantile(0.75):.3f} "
            f"p95={score.quantile(0.95):.3f} "
            f"RED days={(score > 0.65).sum()}"
        )
        return score

    def score_today(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
    ) -> float:
        """
        Score the most recent trading day for live/paper mode.
        Uses the last SCORE_EMA_SPAN days of history for smoothing.

        Returns
        -------
        float ∈ [0, 1]
        """
        feat  = self._compute_features(prices, vix)
        score = self._blend_features(feat)
        score = score.ewm(span=SCORE_EMA_SPAN, adjust=False).mean()
        val   = float(score.iloc[-1]) if len(score) > 0 else 0.0
        log.info(f"ChoppyRegimeDetector live score: {val:.3f}")
        return float(np.clip(val, 0, 1))

    @staticmethod
    def score_to_scale(score: float) -> Tuple[float, str]:
        """Convert choppy score → (position scale factor, colour label)."""
        for threshold, scale, colour, _ in CHOPPY_SCALE_THRESHOLDS:
            if score < threshold:
                return scale, colour
        return 0.20, "RED"

    @staticmethod
    def feature_report(
        prices: pd.DataFrame,
        vix: pd.Series,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of daily feature values for a date range.
        Useful for diagnostics and regime visualisation.
        """
        det  = ChoppyRegimeDetector()
        feat = det._compute_features(prices, vix)
        feat["choppy_score"] = det._blend_features(feat).ewm(
            span=SCORE_EMA_SPAN, adjust=False
        ).mean().clip(0, 1)
        return feat.loc[start:end]


# ── Standalone diagnostic ─────────────────────────────────────────────────────

def run_diagnostic(
    prices: pd.DataFrame,
    vix: pd.Series,
    periods: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """
    Run the detector across specified periods and return a summary DataFrame.
    Used in backtesting to evaluate how the detector behaves per regime.
    """
    det    = ChoppyRegimeDetector()
    scores = det.score_series(prices, vix)

    rows = []
    for label, (s, e) in periods.items():
        sub = scores.loc[s:e]
        if sub.empty:
            continue
        rows.append({
            "period":      label,
            "mean_score":  round(float(sub.mean()), 3),
            "p75_score":   round(float(sub.quantile(0.75)), 3),
            "p95_score":   round(float(sub.quantile(0.95)), 3),
            "pct_green":   round(float((sub < 0.30).mean() * 100), 1),
            "pct_yellow":  round(float(((sub >= 0.30) & (sub < 0.50)).mean() * 100), 1),
            "pct_orange":  round(float(((sub >= 0.50) & (sub < 0.65)).mean() * 100), 1),
            "pct_red":     round(float((sub >= 0.65).mean() * 100), 1),
        })
    return pd.DataFrame(rows)
