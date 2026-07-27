"""
Regression tests for the anomaly-layer calibration fix (2026-07).

The layer sat permanently in ELEVATED, scaling deployment to 0.85x and — combined with
the 0.75 heat cap — creating ~35% cash drag. Root causes, all of which are baseline
miscalibration rather than genuine stress detection:

  1. IsolationForest scores were mapped with (-raw + 0.1) / 0.3, which returns a nonzero
     anomaly score for any raw < 0.1 — ~84% of days on data that is normal by
     construction. decision_function is centred so the contamination quantile sits at 0.
  2. The SPY/TLT rotation component scored any positive stock/bond correlation as stress.
     That held pre-2022 (mean -0.39) but is the structural norm post-2022 (mean +0.12).
  3. The vol-of-VIX component used a 0.03 "calm" baseline against a true median of 0.062.

These tests pin the calibration so it cannot silently regress, and assert that genuine
stress is still detected — the fix must not amount to switching the safety layer off.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from regime.anomaly import (
    _ANOM_SATURATION,
    PositionAnomalyDetector,
    _normalise_decision_score,
)
from regime.sentiment_anomaly import (
    CORR_BASELINE_WINDOW,
    VVIX_CALM,
    SentimentAnomalyDetector,
)


class TestIsolationScoreNormalisation:
    """The decision_function -> [0,1] mapping must start at the contamination boundary."""

    def test_inlier_boundary_scores_zero(self):
        # raw == 0 is the contamination quantile: the edge of normal, not stress.
        assert _normalise_decision_score(0.0) == 0.0

    def test_typical_inlier_scores_zero(self):
        # Any positive decision_function value is an inlier and must score exactly 0.
        for raw in (0.01, 0.05, 0.10, 0.20):
            assert _normalise_decision_score(raw) == 0.0

    def test_outlier_scores_positive_and_saturates(self):
        assert _normalise_decision_score(-_ANOM_SATURATION / 2) == 0.5
        assert _normalise_decision_score(-_ANOM_SATURATION) == 1.0
        assert _normalise_decision_score(-1.0) == 1.0

    def test_normal_data_is_not_flagged_en_masse(self):
        """
        The core regression. On data that is normal by construction, the detector must
        flag roughly `contamination`, not the overwhelming majority of days. The old
        mapping produced a nonzero score on ~84% of such days.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 9))
        model = IsolationForest(
            n_estimators=100, contamination=0.05, max_samples=256, random_state=42
        ).fit(X)
        scores = np.array([_normalise_decision_score(r) for r in model.decision_function(X)])

        assert (scores > 0).mean() < 0.15, "normal data should rarely be flagged anomalous"
        assert np.median(scores) == 0.0


class TestVolOfVixCalibration:
    def test_calm_baseline_matches_measured_median(self):
        # Measured median of 10d realised vol of VIX daily returns, 2015-2026, is ~0.062.
        # A baseline materially below that guarantees a permanent nonzero floor.
        assert VVIX_CALM >= 0.05


class TestStockBondCorrelationIsRegimeRelative:
    """
    The rotation component must score correlation relative to its own trailing baseline,
    so that a structurally positive stock/bond correlation is not read as permanent stress.
    """

    def _detector_with(self, spy: pd.Series, tlt: pd.Series):
        """
        Detector serving only SPY/TLT, so the resulting score *is* the rotation
        component. _load is overridden rather than seeded via the cache because any
        symbol left unseeded would otherwise fall through to real on-disk data.
        """
        det = SentimentAnomalyDetector()
        available = {"SPY": spy, "TLT": tlt}
        det._load = available.get  # type: ignore[method-assign]
        return det

    def test_persistently_positive_correlation_is_not_stress(self):
        """
        Stocks and bonds moving together for years is the post-2022 norm. Once that has
        become the baseline, it must not register as elevated stress.
        """
        n = CORR_BASELINE_WINDOW * 3
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.default_rng(7)
        shared = rng.normal(0, 0.008, n)
        # Correlation ~+0.6 throughout, and a steady uptrend in both legs.
        spy = pd.Series(100 * np.exp(np.cumsum(shared + rng.normal(0.0003, 0.005, n))), idx)
        tlt = pd.Series(100 * np.exp(np.cumsum(shared + rng.normal(0.0001, 0.005, n))), idx)

        det = self._detector_with(spy, tlt)
        score = det.score_series_fast(idx)
        # Only the rotation component is available here, so the score *is* that component.
        tail = score.iloc[CORR_BASELINE_WINDOW:]
        assert tail.mean() < 0.30, f"steady-state correlation scored as stress: {tail.mean():.3f}"

    def test_correlation_spike_above_baseline_is_stress(self):
        """A genuine *rise* above the prevailing regime must still be detected."""
        n = CORR_BASELINE_WINDOW * 2
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.default_rng(11)

        # Baseline period: negatively correlated (bonds hedge equities).
        base = rng.normal(0, 0.008, n)
        spy_r = base + rng.normal(0.0003, 0.004, n)
        tlt_r = -base + rng.normal(0.0001, 0.004, n)

        # Final 40 days: correlation flips strongly positive and both fall (crisis shape).
        spy_r[-40:] = rng.normal(-0.004, 0.02, 40)
        tlt_r[-40:] = spy_r[-40:] * 0.9 + rng.normal(0, 0.002, 40)

        spy = pd.Series(100 * np.exp(np.cumsum(spy_r)), idx)
        tlt = pd.Series(100 * np.exp(np.cumsum(tlt_r)), idx)

        det = self._detector_with(spy, tlt)
        score = det.score_series_fast(idx)
        assert score.iloc[-20:].max() > 0.40, "correlation breakdown was not detected"


class TestCrisisStillDetected:
    """The recalibration must not flatten the layer into a constant 1.0x scale."""

    def test_volatility_explosion_is_flagged(self):
        n = 400
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.default_rng(3)
        syms = [f"S{i}" for i in range(6)]

        rets = rng.normal(0.0004, 0.008, (n, len(syms)))
        # Crash: correlated, high-amplitude moves across every name.
        shock = rng.normal(-0.01, 0.05, (30, 1)).repeat(len(syms), axis=1)
        rets[-30:] = shock + rng.normal(0, 0.01, (30, len(syms)))

        prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=syms)

        det = PositionAnomalyDetector()
        scores = det.score_series(prices)
        calm = scores.iloc[CORR_BASELINE_WINDOW : n - 60]
        crisis = scores.iloc[-20:]

        assert crisis.max() > 0.5, "vol explosion not detected"
        assert crisis.max() > calm.mean(), "crisis must score above calm baseline"
