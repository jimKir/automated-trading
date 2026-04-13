"""
Early Warning System (EWS)
===========================
Orchestrates five independent stress signal layers into a single
position scale factor that gates the main strategy.

Architecture:
  Layer A — Position Anomaly Score    (Isolation Forest, weight 0.30)
  Layer B — Macro Stress Score        (Rule-based FRED, weight 0.25)
  Layer C — Event Shock Score         (VIX velocity + breadth, weight 0.15)
  Layer D — Commodity/FX Score        (Oil, Gold, DXY, JPY, weight 0.10)
  Layer E — Intraday Regime Score     (ADX + SPY EMA + VIX, weight 0.05)
  Layer F — Choppy Regime Score       (2025-fingerprint detector, weight 0.15)

Combined EWS Score → Position Scale Factor:
  Score < 0.25  → 1.00×  GREEN  (full exposure)
  Score 0.25–0.40 → 0.70×  YELLOW (trim)
  Score 0.40–0.55 → 0.40×  ORANGE (reduce)
  Score 0.55–0.70 → 0.20×  RED    (defensive)
  Score > 0.70  → 0.05×  CRITICAL (near-flat)

Design principles:
  - No single layer can cause a full reduction alone (requires confirmation)
  - Scale factor never goes to zero — always keeps a small position
    (going fully flat and then re-entering has its own timing risk)
  - All layers produce independent signals, reducing false positive rate

Config keys (under ews: section):
  use_macro:      true
  use_anomaly:    true
  use_event:      true
  use_commfx:     true
  use_intraday:   true    # Layer E — intraday regime detector
  use_choppy:     true    # Layer F — 2025-fingerprint choppy regime detector
  enabled:        true
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("EWS")

# Layer weights — must sum to 1.0
# Weights rebalanced (Apr 2026) to add Layer F: Choppy Regime Detector.
# Layer A reduced 0.35→0.30, Layer B 0.30→0.25, Layer E 0.10→0.05.
# Total still sums to 1.0.
LAYER_WEIGHTS = {
    "anomaly":         0.30,   # Layer A: Isolation Forest position anomaly
    "macro":           0.25,   # Layer B: FRED macro stress
    "event_shock":     0.15,   # Layer C: VIX velocity + breadth
    "commodity_fx":    0.10,   # Layer D: Oil, Gold, DXY, JPY
    "intraday_regime": 0.05,   # Layer E: ADX + SPY EMA (intraday)
    "choppy_regime":   0.15,   # Layer F: 2025-fingerprint choppy detector (new)
}

# EWS score → position scale factor
# Thresholds set conservatively to avoid over-trading
SCALE_THRESHOLDS = [
    (0.25, 1.00, "GREEN", "Full exposure"),
    (0.40, 0.70, "YELLOW", "Trimming positions"),
    (0.55, 0.40, "ORANGE", "Reducing exposure"),
    (0.70, 0.20, "RED", "Defensive positioning"),
    (1.01, 0.05, "CRITICAL", "Near-flat — acute stress"),
]


def ews_score_to_scale(score: float) -> tuple[float, str, str]:
    """Convert EWS score to (scale_factor, colour, description)."""
    for threshold, scale, colour, desc in SCALE_THRESHOLDS:
        if score < threshold:
            return scale, colour, desc
    return 0.05, "CRITICAL", "Near-flat — acute stress"


class EarlyWarningSystem:
    """
    Main EWS class. In backtest mode, pre-computes full daily score series.
    In live mode, computes today's score on demand.
    """

    def __init__(self, config: dict):
        self.config = config
        self._use_macro     = config.get("ews", {}).get("use_macro",     True)
        self._use_anomaly   = config.get("ews", {}).get("use_anomaly",   True)
        self._use_event     = config.get("ews", {}).get("use_event",     True)
        self._use_commfx    = config.get("ews", {}).get("use_commfx",    True)
        self._use_intraday  = config.get("ews", {}).get("use_intraday",  True)
        self._use_choppy    = config.get("ews", {}).get("use_choppy",    True)
        self._enabled       = config.get("ews", {}).get("enabled",       True)

        # Lazy-loaded sub-modules
        self._anomaly_detector  = None
        self._macro_scorer      = None
        self._event_detector    = None
        self._commfx_scorer     = None
        self._intraday_scorer   = None
        self._choppy_detector   = None

        # Cached series for backtest (keyed by start+end)
        self._score_cache: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    def _load_modules(self):
        if self._anomaly_detector is None and self._use_anomaly:
            from regime.anomaly import PositionAnomalyDetector

            self._anomaly_detector = PositionAnomalyDetector()
        if self._macro_scorer is None and self._use_macro:
            from regime.macro_score import MacroStressScorer

            self._macro_scorer = MacroStressScorer()
        if self._event_detector is None and self._use_event:
            from regime.event_shock import EventShockDetector

            self._event_detector = EventShockDetector()
        if self._commfx_scorer is None and self._use_commfx:
            from regime.commodity_fx import CommodityFXScorer

            self._commfx_scorer = CommodityFXScorer()
        if self._intraday_scorer is None and self._use_intraday:
            from regime.intraday_regime import IntradayRegimeScorer

            self._intraday_scorer = IntradayRegimeScorer()
        if self._choppy_detector is None and self._use_choppy:
            from regime.choppy_regime import ChoppyRegimeDetector
            self._choppy_detector = ChoppyRegimeDetector()

    # ------------------------------------------------------------------
    def compute_backtest_scores(
        self,
        all_prices: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.Series:
        """
        Compute full daily EWS score series for a backtest period.
        Uses walk-forward for anomaly detection, real-time vintage for macro.

        Returns: pd.Series with date index, values in [0, 1]
        """
        cache_key = f"{start}_{end}_{id(all_prices)}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        if not self._enabled:
            idx = pd.date_range(start, end, freq="B")
            return pd.Series(0.0, index=idx)  # no EWS = always full exposure (score 0 → GREEN)

        self._load_modules()
        log.info("EWS: computing all layers for backtest...")

        biz_days = pd.date_range(start, end, freq="B")
        combined = pd.DataFrame(index=biz_days)

        # Layer A — Position Anomaly (walk-forward Isolation Forest)
        if self._use_anomaly and self._anomaly_detector is not None:
            try:
                log.info("EWS Layer A: running Isolation Forest walk-forward...")
                a_scores = self._anomaly_detector.score_series(all_prices)
                combined["anomaly"] = a_scores.reindex(biz_days, method="ffill").fillna(0)
                log.info("EWS Layer A: done")
            except Exception as e:
                log.warning(f"EWS Layer A failed: {e}")
                combined["anomaly"] = 0.0
        else:
            combined["anomaly"] = 0.0

        # Layer B — Macro Stress (FRED)
        if self._use_macro and self._macro_scorer is not None:
            try:
                log.info("EWS Layer B: fetching macro data from FRED...")
                b_scores = self._macro_scorer.compute_series(start, end)
                combined["macro"] = b_scores.reindex(biz_days, method="ffill").fillna(0)
                log.info("EWS Layer B: done")
            except Exception as e:
                log.warning(f"EWS Layer B failed: {e}")
                combined["macro"] = 0.0
        else:
            combined["macro"] = 0.0

        # Layer C — Event Shock
        if self._use_event and self._event_detector is not None:
            try:
                log.info("EWS Layer C: computing event shock scores...")
                c_scores = self._event_detector.compute_series(start, end, all_prices)
                combined["event_shock"] = c_scores.reindex(biz_days, method="ffill").fillna(0)
                log.info("EWS Layer C: done")
            except Exception as e:
                log.warning(f"EWS Layer C failed: {e}")
                combined["event_shock"] = 0.0
        else:
            combined["event_shock"] = 0.0

        # Layer D — Commodity/FX
        if self._use_commfx and self._commfx_scorer is not None:
            try:
                log.info("EWS Layer D: computing commodity/FX scores...")
                d_scores = self._commfx_scorer.compute_series(start, end)
                combined["commodity_fx"] = d_scores.reindex(biz_days, method="ffill").fillna(0)
                log.info("EWS Layer D: done")
            except Exception as e:
                log.warning(f"EWS Layer D failed: {e}")
                combined["commodity_fx"] = 0.0
        else:
            combined["commodity_fx"] = 0.0

        # Layer E — Intraday Regime
        if self._use_intraday and self._intraday_scorer is not None:
            try:
                log.info("EWS Layer E: computing intraday regime scores...")
                e_scores = self._intraday_scorer.compute_series(start, end)
                combined["intraday_regime"] = e_scores.reindex(biz_days, method="ffill").fillna(0)
                log.info("EWS Layer E: done")
            except Exception as e:
                log.warning(f"EWS Layer E failed: {e}")
                combined["intraday_regime"] = 0.0
        else:
            combined["intraday_regime"] = 0.0

        # Layer F — Choppy Regime Detector
        if self._use_choppy and self._choppy_detector is not None:
            try:
                log.info("EWS Layer F: computing choppy regime scores...")
                # Build price DataFrame from all_prices dict or DataFrame
                if isinstance(all_prices, dict):
                    _close = {s: df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                              for s, df in all_prices.items() if s in all_prices}
                    price_df = pd.DataFrame(_close)
                else:
                    price_df = all_prices
                # Extract VIX series (try common names)
                vix_col = None
                for vix_name in ["VIX", "^VIX", "vix"]:
                    if hasattr(price_df, "columns") and vix_name in price_df.columns:
                        vix_col = price_df[vix_name]
                        break
                if vix_col is None:
                    # Fall back: use vol_10d of SPY as VIX proxy
                    if "SPY" in price_df.columns:
                        vix_col = (price_df["SPY"].pct_change().rolling(10).std() * np.sqrt(252) * 100)
                    else:
                        vix_col = pd.Series(dtype=float)
                f_scores = self._choppy_detector.score_series(price_df, vix_col)
                combined["choppy_regime"] = f_scores.reindex(biz_days, method="ffill").fillna(0)
                log.info("EWS Layer F: done")
            except Exception as e:
                log.warning(f"EWS Layer F failed: {e}")
                combined["choppy_regime"] = 0.0
        else:
            combined["choppy_regime"] = 0.0

        # Weighted combination
        ews_scores = (
            LAYER_WEIGHTS["anomaly"]         * combined.get("anomaly",         0) +
            LAYER_WEIGHTS["macro"]           * combined.get("macro",           0) +
            LAYER_WEIGHTS["event_shock"]     * combined.get("event_shock",     0) +
            LAYER_WEIGHTS["commodity_fx"]    * combined.get("commodity_fx",    0) +
            LAYER_WEIGHTS["intraday_regime"] * combined.get("intraday_regime", 0) +
            LAYER_WEIGHTS["choppy_regime"]   * combined.get("choppy_regime",   0)
        )

        # Smooth with 3-day EMA to avoid single-day noise spikes
        ews_scores = ews_scores.ewm(span=3, adjust=False).mean()
        ews_scores = ews_scores.clip(0, 1)

        log.info(
            f"EWS: score stats — mean={ews_scores.mean():.3f} "
            f"max={ews_scores.max():.3f} "
            f"p95={ews_scores.quantile(0.95):.3f}"
        )

        # Log how many days each regime state was active
        for thr, scale, colour, desc in SCALE_THRESHOLDS:
            prev_thr = (
                SCALE_THRESHOLDS[SCALE_THRESHOLDS.index((thr, scale, colour, desc)) - 1][0]
                if SCALE_THRESHOLDS.index((thr, scale, colour, desc)) > 0
                else 0
            )
            days_in = ((ews_scores >= prev_thr) & (ews_scores < thr)).sum()
            log.info(f"  {colour:<10} ({scale:.0%} scale): {days_in} days")

        self._score_cache[cache_key] = ews_scores
        return ews_scores

    # ------------------------------------------------------------------
    def get_scale_factor(self, date: pd.Timestamp, ews_scores: pd.Series) -> tuple[float, str]:
        """
        Get the position scale factor for a given date from pre-computed scores.
        Returns (scale_factor, regime_colour).
        """
        if not self._enabled or ews_scores is None or ews_scores.empty:
            return 1.0, "GREEN"
        try:
            score = float(ews_scores.asof(date))
        except Exception:
            return 1.0, "GREEN"

        scale, colour, _ = ews_score_to_scale(score)
        return scale, colour

    # ------------------------------------------------------------------
    def score_today(self, all_prices: pd.DataFrame = None) -> tuple[float, float, str]:
        """
        Compute live EWS score for paper/live trading.
        Returns (ews_score, scale_factor, regime_colour).
        """
        if not self._enabled:
            return 0.0, 1.0, "GREEN"

        self._load_modules()

        _now = datetime.now(UTC)
        end = _now.strftime("%Y-%m-%d")
        start = (_now - timedelta(days=400)).strftime("%Y-%m-%d")

        scores_by_layer = {}

        if self._use_anomaly and self._anomaly_detector is not None and all_prices is not None:
            try:
                # Fit on available history, score today
                self._anomaly_detector.fit(all_prices)
                scores_by_layer["anomaly"] = self._anomaly_detector.score(all_prices)
            except Exception as e:
                log.warning(f"EWS live anomaly failed: {e}")
                scores_by_layer["anomaly"] = 0.0

        if self._use_macro and self._macro_scorer is not None:
            try:
                scores_by_layer["macro"] = self._macro_scorer.score_today()
            except Exception as e:
                log.warning(f"EWS live macro failed: {e}")
                scores_by_layer["macro"] = 0.0

        if self._use_event and self._event_detector is not None:
            try:
                scores_by_layer["event_shock"] = self._event_detector.score_today(all_prices)
            except Exception as e:
                log.warning(f"EWS live event failed: {e}")
                scores_by_layer["event_shock"] = 0.0

        if self._use_commfx and self._commfx_scorer is not None:
            try:
                scores_by_layer["commodity_fx"] = self._commfx_scorer.score_today()
            except Exception as e:
                log.warning(f"EWS live commfx failed: {e}")
                scores_by_layer["commodity_fx"] = 0.0

        if self._use_intraday and self._intraday_scorer is not None:
            try:
                scores_by_layer["intraday_regime"] = self._intraday_scorer.score_today()
            except Exception as e:
                log.warning(f"EWS live intraday failed: {e}")
                scores_by_layer["intraday_regime"] = 0.0

        if self._use_choppy and self._choppy_detector is not None and all_prices is not None:
            try:
                if isinstance(all_prices, dict):
                    _close = {s: df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                              for s, df in all_prices.items()}
                    price_df = pd.DataFrame(_close)
                else:
                    price_df = all_prices
                vix_col = None
                for vn in ["VIX", "^VIX", "vix"]:
                    if hasattr(price_df, "columns") and vn in price_df.columns:
                        vix_col = price_df[vn]; break
                if vix_col is None and "SPY" in price_df.columns:
                    vix_col = price_df["SPY"].pct_change().rolling(10).std() * np.sqrt(252) * 100
                if vix_col is not None:
                    scores_by_layer["choppy_regime"] = self._choppy_detector.score_today(price_df, vix_col)
            except Exception as e:
                log.warning(f"EWS live choppy failed: {e}")
                scores_by_layer["choppy_regime"] = 0.0

        ews_score = sum(
            LAYER_WEIGHTS[k] * v for k, v in scores_by_layer.items() if k in LAYER_WEIGHTS
        )
        ews_score = float(np.clip(ews_score, 0, 1))
        scale, colour, desc = ews_score_to_scale(ews_score)

        log.info(
            f"EWS Live | score={ews_score:.3f} | {colour} | scale={scale:.0%} | {desc}\n"
            + "\n".join(f"  Layer {k}: {v:.3f}" for k, v in scores_by_layer.items())
        )
        return ews_score, scale, colour

    # ------------------------------------------------------------------
    def generate_ews_report_data(self, ews_scores: pd.Series, all_prices: pd.DataFrame) -> dict:
        """
        Generate EWS report data for inclusion in the HTML backtest report.
        """
        if ews_scores is None or ews_scores.empty:
            return {}

        scale_factors = ews_scores.apply(lambda s: ews_score_to_scale(s)[0])
        colours = ews_scores.apply(lambda s: ews_score_to_scale(s)[1])

        regime_counts = colours.value_counts().to_dict()
        avg_scale = float(scale_factors.mean())

        # Days when EWS was in each state
        days_green = int((colours == "GREEN").sum())
        days_yellow = int((colours == "YELLOW").sum())
        days_orange = int((colours == "ORANGE").sum())
        days_red = int(((colours == "RED") | (colours == "CRITICAL")).sum())

        return {
            "ews_scores": ews_scores,
            "ews_scale_factors": scale_factors,
            "ews_avg_scale": avg_scale,
            "ews_days_green": days_green,
            "ews_days_yellow": days_yellow,
            "ews_days_orange": days_orange,
            "ews_days_red": days_red,
            "ews_regime_counts": regime_counts,
        }
