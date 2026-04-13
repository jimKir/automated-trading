"""
Order Flow Anomaly Detector — Group 9 for ChoppyDetector v4
=============================================================
Detects abnormal order flow patterns that precede choppy regimes:
  - Volume imbalance (buy/sell proxy via close-to-range position)
  - Large-trade clustering (vol spikes in short windows)
  - Aggressor ratio proxy (tick direction × volume)

All features are computed from daily OHLCV data using close-to-range
position as a buy/sell proxy (no tick data required for daily backtest).

In live mode, this can be upgraded to use actual trade-level data
from Alpaca or Databento for more accurate order flow measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("OrderFlowAnomaly")

# Calibration: (baseline, ceiling) — derived from 2024 calm vs 2025 choppy
CALIBRATION = {
    "imbalance_dispersion": (0.15, 0.45),   # cross-asset imbalance std (calm=0.15, stress=0.45)
    "large_trade_cluster":  (0.05, 0.25),   # fraction of days with vol > 3× 20d mean in 10d
    "aggressor_skew":       (0.10, 0.40),   # abs skew of aggressor proxy in 20d window
}


class OrderFlowAnomalyDetector:
    """
    Computes an order-flow anomaly score from daily OHLCV data.

    The score measures how unusual recent order flow patterns are
    relative to calm-market baselines. High scores indicate the kind
    of institutional repositioning that precedes choppy regimes.

    Usage:
        detector = OrderFlowAnomalyDetector()
        score = detector.compute(prices_df, idx)  # returns pd.Series in [0, 1]
    """

    @staticmethod
    def _close_position(df: pd.DataFrame) -> pd.Series:
        """
        Compute close-to-range position: (close - low) / (high - low).
        Values near 1 = buying pressure, near 0 = selling pressure.
        This is a standard daily proxy for order flow direction.
        """
        hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
        return ((df["Close"] - df["Low"]) / hl_range).fillna(0.5)

    @staticmethod
    def _rescale(raw: pd.Series, baseline: float, ceiling: float) -> pd.Series:
        denom = ceiling - baseline
        if abs(denom) < 1e-10:
            return pd.Series(0.0, index=raw.index)
        return ((raw - baseline) / denom).clip(0, 1)

    def compute(
        self,
        all_prices: dict,
        idx: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Compute order flow anomaly score across all instruments.

        Parameters
        ----------
        all_prices : dict of {symbol: DataFrame with OHLCV columns}
        idx        : DatetimeIndex to align output to

        Returns
        -------
        pd.Series of scores in [0, 1]
        """
        components = []

        # Collect close-position series for all instruments
        cp_series = {}
        for sym, df in all_prices.items():
            try:
                _df = df.copy()
                if isinstance(_df.columns, pd.MultiIndex):
                    _df.columns = [c[0] for c in _df.columns]
                _df.columns = [c.capitalize() for c in _df.columns]
                if all(c in _df.columns for c in ["High", "Low", "Close"]):
                    cp = self._close_position(_df)
                    cp_series[sym] = cp
            except Exception:
                continue

        if not cp_series:
            return pd.Series(0.0, index=idx)

        cp_df = pd.DataFrame(cp_series)

        # F1: Imbalance dispersion — cross-asset std of close-position
        # High dispersion = instruments being bought/sold very differently = repositioning
        imb_disp = cp_df.rolling(10).std().mean(axis=1)
        base, ceil = CALIBRATION["imbalance_dispersion"]
        components.append(self._rescale(imb_disp, base, ceil))

        # F2: Large-trade clustering — fraction of instruments with volume > 3× 20d mean
        vol_spikes = pd.DataFrame(index=idx)
        for sym, df in all_prices.items():
            try:
                _df = df.copy()
                if isinstance(_df.columns, pd.MultiIndex):
                    _df.columns = [c[0] for c in _df.columns]
                _df.columns = [c.capitalize() for c in _df.columns]
                if "Volume" in _df.columns:
                    vol = _df["Volume"]
                    ma20 = vol.rolling(20).mean().replace(0, np.nan)
                    spike = (vol > 3 * ma20).astype(float)
                    vol_spikes[sym] = spike
            except Exception:
                continue

        if not vol_spikes.empty:
            cluster_rate = vol_spikes.rolling(10).mean().mean(axis=1)
            base, ceil = CALIBRATION["large_trade_cluster"]
            components.append(
                self._rescale(cluster_rate, base, ceil).reindex(idx, method="ffill").fillna(0)
            )

        # F3: Aggressor skew — skewness of close-position over 20d
        # High absolute skew = persistent directional aggression = institutional flow
        cp_skew = cp_df.rolling(20).apply(
            lambda x: x.skew() if len(x) >= 10 else 0, raw=False
        ).abs().mean(axis=1)
        base, ceil = CALIBRATION["aggressor_skew"]
        components.append(
            self._rescale(cp_skew, base, ceil).reindex(idx, method="ffill").fillna(0)
        )

        if not components:
            return pd.Series(0.0, index=idx)

        result = pd.concat(components, axis=1).mean(axis=1)
        return result.reindex(idx, method="ffill").fillna(0).clip(0, 1)
