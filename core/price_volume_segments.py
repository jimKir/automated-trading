"""
Price-Volume Segment Analyser
==============================
Segments recent price-volume behaviour into discrete regimes that
improve momentum and trend identification.

The key insight: momentum is not just "price went up". It matters
HOW price went up — on expanding or contracting volume, with
acceleration or deceleration, in a narrow or wide range.

This module generates a **segment score** in [-1, +1] that captures
the quality of the current momentum, not just its direction.

Segment Features (12 total)
────────────────────────────
  Price segments (6):
    - ret_pct_5d, ret_pct_10d       : raw return segments
    - accel_5d                      : 5d return - 10d return (acceleration)
    - trend_consistency             : % of last 10 days return was in same direction
    - close_position_in_range       : where in the 10d high-low range price closed
    - gap_count_5d                  : number of gap-up/down opens in last 5 days

  Volume segments (6):
    - vol_pct_chg_5d, vol_pct_chg_10d : volume % change vs 20d baseline
    - vol_price_correlation_10d     : correlation(|return|, volume) over 10 days
    - buying_pressure_5d            : sum(volume where close > open) / total volume
    - vol_expansion_streak          : consecutive days of above-avg volume
    - relative_volume_at_highs      : volume when price in top 25% of range / avg vol

Segment Score Logic
────────────────────
  STRONG momentum (+0.8 to +1.0):
    - Price rising + volume expanding + high trend consistency + buying pressure
    - This is the Wyckoff "markup" phase — ride it

  MODERATE momentum (+0.3 to +0.7):
    - Price rising + volume flat/mixed
    - Momentum exists but is not strongly confirmed

  WEAK/EXHAUSTING momentum (-0.3 to +0.3):
    - Price rising but volume contracting (distribution)
    - Or choppy with low consistency

  REVERSAL signal (-0.3 to -1.0):
    - Volume expanding + price stalling (accumulation/distribution)
    - High volume at extremes with poor follow-through

Usage in signals.py:
  The segment score is blended with the existing signal:
    final = (1 - pv_weight) * existing_signal + pv_weight * segment_score
  Default pv_weight = 0.15 (configurable via strategy.pv_segment_weight)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("PVSegments")


class PriceVolumeSegmenter:
    """
    Analyzes price-volume patterns and produces a momentum quality score.
    Purely rule-based (no fitting) but designed to be replaced by H2O
    when enough training data is accumulated.
    """

    def __init__(self, config: dict):
        pv_cfg = config.get("pv_segments", config.get("strategy", {}))
        self.enabled = pv_cfg.get("pv_segments_enabled", True)
        self.weight = pv_cfg.get("pv_segment_weight", 0.15)
        self.use_h2o = pv_cfg.get("pv_use_h2o", True)

        self._h2o_model = None
        self._h2o = None
        self._h2o_loaded = False
        self._h2o_tried = False

        if self.enabled:
            log.info(f"PVSegmenter: enabled, weight={self.weight:.0%}, h2o={self.use_h2o}")

    def _try_load_h2o(self):
        """Attempt to load H2O for regression-based scoring."""
        if self._h2o_tried:
            return
        self._h2o_tried = True
        if not self.use_h2o:
            return
        try:
            import h2o

            try:
                h2o.cluster()
            except Exception:
                h2o.init(nthreads=-1, max_mem_size="2g", verbose=False)
                h2o.no_progress()
            self._h2o = h2o
            self._h2o_loaded = True
            log.info("PVSegmenter: H2O available for enhanced scoring")
        except Exception:
            self._h2o_loaded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features(
        self,
        close: pd.Series,
        volume: pd.Series | None = None,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
        open_price: pd.Series | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> dict | None:
        """
        Extract 12 price-volume segment features for the latest date.
        Returns None if insufficient data.
        """
        if as_of_date is not None:
            close = close[close.index <= as_of_date]
            if volume is not None:
                volume = volume[volume.index <= as_of_date]
            if high is not None:
                high = high[high.index <= as_of_date]
            if low is not None:
                low = low[low.index <= as_of_date]
            if open_price is not None:
                open_price = open_price[open_price.index <= as_of_date]

        if len(close) < 25:
            return None

        feats = {}

        # ── Price segments ────────────────────────────────────────────────
        feats["ret_pct_5d"] = float(close.pct_change(5).iloc[-1])
        feats["ret_pct_10d"] = float(close.pct_change(10).iloc[-1])

        # Acceleration: are we speeding up or slowing down?
        feats["accel_5d"] = feats["ret_pct_5d"] - feats["ret_pct_10d"] / 2

        # Trend consistency: what % of last 10 days moved in same direction?
        daily_rets = close.pct_change().iloc[-10:]
        if len(daily_rets) >= 10:
            net_dir = np.sign(daily_rets.sum())  # overall direction
            same_dir = (np.sign(daily_rets) == net_dir).sum()
            feats["trend_consistency"] = float(same_dir / 10)
        else:
            feats["trend_consistency"] = 0.5

        # Close position in range
        if high is not None and low is not None and len(high) >= 10:
            h10 = high.iloc[-10:].max()
            l10 = low.iloc[-10:].min()
            rng = h10 - l10
            feats["close_pos_in_range"] = float((close.iloc[-1] - l10) / rng) if rng > 0 else 0.5
        else:
            feats["close_pos_in_range"] = 0.5

        # Gap count (opens that gap above/below prior close)
        if open_price is not None and len(open_price) >= 6:
            gaps = (open_price.iloc[-5:] - close.shift(1).iloc[-5:]).abs()
            gap_threshold = close.iloc[-20:].std() * 0.5 if len(close) >= 20 else 0
            feats["gap_count_5d"] = float((gaps > gap_threshold).sum())
        else:
            feats["gap_count_5d"] = 0.0

        # ── Volume segments ───────────────────────────────────────────────
        if volume is not None and (volume > 0).sum() > 25:
            vol_20d_avg = volume.rolling(20).mean()

            # Volume % change vs baseline
            feats["vol_pct_chg_5d"] = (
                float(volume.iloc[-5:].mean() / vol_20d_avg.iloc[-1] - 1)
                if vol_20d_avg.iloc[-1] > 0
                else 0.0
            )

            feats["vol_pct_chg_10d"] = (
                float(volume.iloc[-10:].mean() / vol_20d_avg.iloc[-1] - 1)
                if vol_20d_avg.iloc[-1] > 0
                else 0.0
            )

            # Volume-price correlation: high correlation = volume confirms moves
            rets_10 = close.pct_change().iloc[-10:].abs()
            vol_10 = volume.iloc[-10:]
            if len(rets_10) >= 10 and vol_10.std() > 0:
                feats["vol_price_corr_10d"] = float(rets_10.corr(vol_10))
                if np.isnan(feats["vol_price_corr_10d"]):
                    feats["vol_price_corr_10d"] = 0.0
            else:
                feats["vol_price_corr_10d"] = 0.0

            # Buying pressure: volume on up-close days / total
            if open_price is not None and len(open_price) >= 5:
                up_days = close.iloc[-5:] > open_price.iloc[-5:]
                vol_5 = volume.iloc[-5:]
                total_vol = vol_5.sum()
                feats["buying_pressure_5d"] = (
                    float(vol_5[up_days].sum() / total_vol) if total_vol > 0 else 0.5
                )
            else:
                feats["buying_pressure_5d"] = 0.5

            # Volume expansion streak: consecutive above-avg volume days
            vol_above = (volume > vol_20d_avg).iloc[-10:]
            streak = 0
            for v in reversed(vol_above.values):
                if v:
                    streak += 1
                else:
                    break
            feats["vol_expansion_streak"] = float(streak)

            # Relative volume at price highs
            if high is not None and len(high) >= 10:
                h10 = high.iloc[-10:]
                l10 = low.iloc[-10:] if low is not None else close.iloc[-10:] * 0.99
                rng = h10 - l10
                top_25 = close.iloc[-10:] > (l10 + 0.75 * rng)
                avg_vol = volume.iloc[-10:].mean()
                vol_at_highs = volume.iloc[-10:][top_25].mean() if top_25.sum() > 0 else avg_vol
                feats["rel_vol_at_highs"] = float(vol_at_highs / avg_vol) if avg_vol > 0 else 1.0
            else:
                feats["rel_vol_at_highs"] = 1.0
        else:
            feats["vol_pct_chg_5d"] = 0.0
            feats["vol_pct_chg_10d"] = 0.0
            feats["vol_price_corr_10d"] = 0.0
            feats["buying_pressure_5d"] = 0.5
            feats["vol_expansion_streak"] = 0.0
            feats["rel_vol_at_highs"] = 1.0

        # Clean NaN/inf
        for k, v in feats.items():
            if np.isnan(v) or np.isinf(v):
                feats[k] = 0.0

        return feats

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────────

    def score(
        self,
        close: pd.Series,
        volume: pd.Series | None = None,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
        open_price: pd.Series | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> float:
        """
        Compute segment score in [-1, +1].
        Positive = strong momentum quality, negative = weak/exhausting.
        """
        if not self.enabled:
            return 0.0

        feats = self.extract_features(close, volume, high, low, open_price, as_of_date)
        if feats is None:
            return 0.0

        return self._rule_based_score(feats)

    def _rule_based_score(self, feats: dict) -> float:
        """
        Convert features to a momentum quality score using rules.

        The score captures:
          1. Direction (are we going up or down?)
          2. Quality (is volume confirming the move?)
          3. Sustainability (is the trend consistent and accelerating?)
        """
        # Direction component: weighted recent returns
        direction = (
            0.60 * np.tanh(feats["ret_pct_5d"] * 20)  # 5d return, scaled
            + 0.40 * np.tanh(feats["ret_pct_10d"] * 15)  # 10d return
        )

        # Quality component: volume confirmation
        vol_confirm = 0.0
        # Volume expanding in direction of price → confirms trend
        if feats["ret_pct_5d"] > 0:
            # Uptrend: want high buying pressure + volume expansion
            vol_confirm = (
                0.30 * (feats["buying_pressure_5d"] - 0.5) * 2  # [-1, 1]
                + 0.30 * np.tanh(feats["vol_pct_chg_5d"])  # volume growth
                + 0.20 * feats["vol_price_corr_10d"]  # price-vol alignment
                + 0.20 * min(feats["vol_expansion_streak"] / 5, 1.0)  # streak
            )
        else:
            # Downtrend: want high volume (capitulation) as potential reversal
            vol_confirm = (
                0.30 * (0.5 - feats["buying_pressure_5d"]) * 2
                + 0.30 * np.tanh(feats["vol_pct_chg_5d"])
                + 0.20 * feats["vol_price_corr_10d"]
                + 0.20 * min(feats["vol_expansion_streak"] / 5, 1.0)
            )

        # Sustainability component
        sustainability = (
            0.40 * (feats["trend_consistency"] - 0.5) * 2  # consistency [-1, 1]
            + 0.30 * np.tanh(feats["accel_5d"] * 30)  # acceleration
            + 0.30 * (feats["close_pos_in_range"] - 0.5) * 2  # position in range
        )

        # Combine: direction matters most, quality and sustainability amplify
        raw_score = 0.50 * direction + 0.25 * vol_confirm + 0.25 * sustainability

        return float(np.clip(raw_score, -1.0, 1.0))

    def score_series(
        self,
        close: pd.Series,
        volume: pd.Series | None = None,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
        open_price: pd.Series | None = None,
    ) -> pd.Series:
        """
        Compute segment score for each date in the series.
        Used in backtest for vectorized signal generation.
        """
        if not self.enabled or len(close) < 25:
            return pd.Series(0.0, index=close.index)

        scores = pd.Series(0.0, index=close.index)
        for i in range(25, len(close)):
            date = close.index[i]
            feats = self.extract_features(
                close.iloc[: i + 1],
                volume.iloc[: i + 1] if volume is not None else None,
                high.iloc[: i + 1] if high is not None else None,
                low.iloc[: i + 1] if low is not None else None,
                open_price.iloc[: i + 1] if open_price is not None else None,
            )
            if feats is not None:
                scores.iloc[i] = self._rule_based_score(feats)

        return scores
