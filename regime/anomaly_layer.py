"""
Multi-Source Anomaly Detection Layer
=====================================
Combines 4 independent anomaly sources into a composite score [0,1]
with graceful degradation when any source is unavailable.

Sources & weights:
  macro (MacroStressScorer):       0.30
  sentiment (VIX-based):           0.30
  fx (JPY/CHF/DXY):               0.25
  isolation_forest (PositionAnomaly): 0.15

Composite → regime label → position scale:
  NORMAL   (< 0.20): 1.00×
  ELEVATED (0.20–0.35): 0.85×
  STRESSED (0.35–0.50): 0.65×
  CRISIS   (≥ 0.50): 0.40×
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger("AnomalyLayer")


@dataclass
class AnomalyScore:
    composite: float  # 0-1
    label: str  # NORMAL / ELEVATED / STRESSED / CRISIS
    source_scores: dict  # {macro: 0.12, sentiment: 0.34, fx: 0.08, isolation: 0.21}
    position_scale: float  # 1.0 / 0.85 / 0.65 / 0.40


class AnomalyRegimeLayer:
    """
    Combines 4 anomaly sources into a composite stress score.
    Degrades gracefully — if a source fails, remaining sources
    are re-weighted to sum to 1.0.
    """

    THRESHOLDS = {"elevated": 0.20, "stressed": 0.35, "crisis": 0.50}
    SCALE_MAP = {"NORMAL": 1.00, "ELEVATED": 0.85, "STRESSED": 0.65, "CRISIS": 0.40}

    DEFAULT_WEIGHTS = {
        "macro": 0.30,
        "sentiment": 0.30,
        "fx": 0.25,
        "isolation": 0.15,
    }

    def __init__(self, config: dict | None = None, data_dir: Path | None = None):
        cfg = (config or {}).get("anomaly_layer", {})
        self.weights = {
            "macro": cfg.get("macro_weight", self.DEFAULT_WEIGHTS["macro"]),
            "sentiment": cfg.get("sentiment_weight", self.DEFAULT_WEIGHTS["sentiment"]),
            "fx": cfg.get("fx_weight", self.DEFAULT_WEIGHTS["fx"]),
            "isolation": cfg.get("isolation_weight", self.DEFAULT_WEIGHTS["isolation"]),
        }
        self._data_dir = data_dir
        self._macro_scorer = None
        self._fx_detector = None
        self._sentiment_detector = None
        self._isolation_detector = None
        self._initialised = False

    def _lazy_init(self):
        """Lazy-load sources to avoid import cost when not needed."""
        if self._initialised:
            return
        try:
            from regime.macro_score import MacroStressScorer

            self._macro_scorer = MacroStressScorer()
        except Exception as e:
            log.warning(f"MacroStressScorer unavailable: {e}")

        try:
            from regime.fx_anomaly import FXAnomalyDetector

            self._fx_detector = FXAnomalyDetector(data_dir=self._data_dir)
        except Exception as e:
            log.warning(f"FXAnomalyDetector unavailable: {e}")

        try:
            from regime.sentiment_anomaly import SentimentAnomalyDetector

            self._sentiment_detector = SentimentAnomalyDetector(data_dir=self._data_dir)
        except Exception as e:
            log.warning(f"SentimentAnomalyDetector unavailable: {e}")

        try:
            from regime.anomaly import PositionAnomalyDetector

            self._isolation_detector = PositionAnomalyDetector()
        except Exception as e:
            log.warning(f"PositionAnomalyDetector unavailable: {e}")

        self._initialised = True

    @staticmethod
    def get_regime_label(composite: float) -> str:
        """Map composite score to regime label."""
        if composite >= 0.50:
            return "CRISIS"
        if composite >= 0.35:
            return "STRESSED"
        if composite >= 0.20:
            return "ELEVATED"
        return "NORMAL"

    def get_position_scale(self, composite: float) -> float:
        """Returns position scale multiplier based on composite score."""
        label = self.get_regime_label(composite)
        return self.SCALE_MAP[label]

    def compute(self, prices: pd.DataFrame, date=None) -> AnomalyScore:
        """
        Compute composite anomaly score from all 4 sources.

        Parameters
        ----------
        prices : pd.DataFrame
            Multi-asset close prices (columns=symbols, index=dates).
        date : str or pd.Timestamp, optional
            Date to score. Defaults to most recent date in prices.

        Returns
        -------
        AnomalyScore with composite, label, source_scores, position_scale.
        """
        self._lazy_init()

        if date is None:
            date = prices.index[-1]
        date = pd.Timestamp(date)

        source_scores = {}
        available_weights = {}

        # Ensure tz-naive
        if hasattr(prices.index, "tz") and prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)

        # 1. Macro stress
        if self._macro_scorer is not None:
            try:
                start = (date - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
                end_str = date.strftime("%Y-%m-%d")
                series = self._macro_scorer.compute_series(start, end_str)
                if not series.empty:
                    val = float(series.asof(date))
                    if not np.isnan(val):
                        source_scores["macro"] = float(np.clip(val, 0, 1))
                        available_weights["macro"] = self.weights["macro"]
            except Exception as e:
                log.debug(f"Macro source failed: {e}")

        # 2. Sentiment (VIX-based)
        if self._sentiment_detector is not None:
            try:
                val = self._sentiment_detector.score_at(date)
                source_scores["sentiment"] = float(np.clip(val, 0, 1))
                available_weights["sentiment"] = self.weights["sentiment"]
            except Exception as e:
                log.debug(f"Sentiment source failed: {e}")

        # 3. FX anomaly
        if self._fx_detector is not None:
            try:
                val = self._fx_detector.score_at(date)
                source_scores["fx"] = float(np.clip(val, 0, 1))
                available_weights["fx"] = self.weights["fx"]
            except Exception as e:
                log.debug(f"FX source failed: {e}")

        # 4. Isolation Forest
        if self._isolation_detector is not None:
            try:
                # Fit on available history, score latest
                prices_for_iso = prices.loc[:date]
                if len(prices_for_iso) >= 120:
                    self._isolation_detector.fit(prices_for_iso)
                    val = self._isolation_detector.score(prices_for_iso)
                    source_scores["isolation"] = float(np.clip(val, 0, 1))
                    available_weights["isolation"] = self.weights["isolation"]
            except Exception as e:
                log.debug(f"Isolation Forest source failed: {e}")

        # Composite: weighted average of available sources
        if not available_weights:
            return AnomalyScore(
                composite=0.0,
                label="NORMAL",
                source_scores={},
                position_scale=1.0,
            )

        total_w = sum(available_weights.values())
        composite = sum(
            source_scores[k] * available_weights[k] / total_w for k in available_weights
        )
        composite = float(np.clip(composite, 0, 1))
        label = self.get_regime_label(composite)
        scale = self.get_position_scale(composite)

        return AnomalyScore(
            composite=composite,
            label=label,
            source_scores=source_scores,
            position_scale=scale,
        )

    def compute_series(
        self,
        prices: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute daily anomaly scores for backtest.
        Returns DataFrame with columns: composite, macro, sentiment, fx, isolation, label, scale.

        Uses vectorised methods where possible for speed.
        """
        self._lazy_init()

        if hasattr(prices.index, "tz") and prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)

        idx = prices.index
        if start:
            idx = idx[idx >= pd.Timestamp(start)]
        if end:
            idx = idx[idx <= pd.Timestamp(end)]

        result = pd.DataFrame(index=idx)

        # Vectorised sources
        available_weights = {}

        # Sentiment (vectorised)
        if self._sentiment_detector is not None:
            try:
                sent = self._sentiment_detector.score_series_fast(idx)
                result["sentiment"] = sent.reindex(idx).fillna(0).clip(0, 1)
                available_weights["sentiment"] = self.weights["sentiment"]
            except Exception as e:
                log.debug(f"Sentiment series failed: {e}")

        # FX (vectorised)
        if self._fx_detector is not None:
            try:
                fx = self._fx_detector.score_series_fast(idx)
                result["fx"] = fx.reindex(idx).fillna(0).clip(0, 1)
                available_weights["fx"] = self.weights["fx"]
            except Exception as e:
                log.debug(f"FX series failed: {e}")

        # Isolation Forest (walk-forward)
        if self._isolation_detector is not None:
            try:
                iso_scores = self._isolation_detector.score_series(prices)
                result["isolation"] = iso_scores.reindex(idx).fillna(0).clip(0, 1)
                available_weights["isolation"] = self.weights["isolation"]
            except Exception as e:
                log.debug(f"Isolation Forest series failed: {e}")

        # Macro (vectorised via compute_series)
        if self._macro_scorer is not None:
            try:
                start_str = idx[0].strftime("%Y-%m-%d")
                end_str = idx[-1].strftime("%Y-%m-%d")
                macro_s = self._macro_scorer.compute_series(start_str, end_str)
                if hasattr(macro_s.index, "tz") and macro_s.index.tz is not None:
                    macro_s.index = macro_s.index.tz_localize(None)
                result["macro"] = macro_s.reindex(idx, method="ffill").fillna(0).clip(0, 1)
                available_weights["macro"] = self.weights["macro"]
            except Exception as e:
                log.debug(f"Macro series failed: {e}")

        # Composite
        if available_weights:
            total_w = sum(available_weights.values())
            composite = pd.Series(0.0, index=idx)
            for src, w in available_weights.items():
                if src in result.columns:
                    composite += result[src] * w / total_w
            result["composite"] = composite.clip(0, 1)
        else:
            result["composite"] = 0.0

        # Labels and scales
        result["label"] = result["composite"].apply(self.get_regime_label)
        result["scale"] = result["composite"].apply(self.get_position_scale)

        return result
