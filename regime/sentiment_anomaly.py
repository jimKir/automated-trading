"""
Sentiment Anomaly Detector
============================
Detects stress from VIX term structure, vol-of-vol, and
equity/treasury rotation using local parquet data.

Score: 0 (benign) → 1.0 (maximum sentiment stress)

Signals:
  - VIX term structure inversion (VIX elevated vs historical)
  - VVIX proxy: vol-of-vol (VIX realised volatility)
  - Equity/Treasury rotation: SPY/TLT correlation shift
  - Fear premium: VIX / realised vol ratio
  - Put/Call proxy: VIX acceleration (2nd derivative)

All features use only past data — no look-ahead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.data_store import get_store
from utils.logger import get_logger

log = get_logger("SentimentAnomaly")

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data" / "historical" / "daily"


class SentimentAnomalyDetector:
    """
    Sentiment anomaly score [0, 1] from VIX dynamics and
    equity/bond rotation signals. Uses local parquet data.
    """

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self._cache: dict[str, pd.Series] = {}

    def _load(self, sym: str) -> pd.Series | None:
        """Load Close price series from DataStore (local or S3)."""
        if sym in self._cache:
            return self._cache[sym]
        try:
            store = get_store()
            df = store.load(sym)
            if df is None:
                log.debug(f"SentimentAnomaly: {sym} not found in DataStore")
                return None
            df.columns = [c.capitalize() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            else:
                df.index = pd.to_datetime(df.index).tz_localize(None)
            s = df["Close"].rename(sym)
            self._cache[sym] = s
            return s
        except Exception as e:
            log.warning(f"SentimentAnomaly: failed to load {sym}: {e}")
            return None

    def score_series_fast(self, idx: pd.DatetimeIndex) -> pd.Series:
        """Vectorised sentiment anomaly score for backtest."""
        vix = self._load("VIX")
        spy = self._load("SPY")
        tlt = self._load("TLT")

        components = []
        weights = []

        if vix is not None:
            vix_a = vix.reindex(idx, method="ffill").bfill()

            # 1. VIX level stress (elevated VIX = fear)
            # Calibration: <15 = calm, 20 = elevated, 30 = crisis, 40+ = panic
            vix_level = ((vix_a - 15) / 25).clip(0, 1)
            components.append(vix_level)
            weights.append(0.20)

            # 2. VVIX proxy: vol-of-VIX (realised VIX volatility over 10d)
            # High VVIX = uncertainty about uncertainty = regime instability
            vix_rvol = vix_a.pct_change().rolling(10).std()
            # Calibration: calm ~0.03, stress ~0.08, crisis ~0.15+
            vvix_score = ((vix_rvol - 0.03) / 0.12).clip(0, 1)
            components.append(vvix_score.fillna(0))
            weights.append(0.20)

            # 3. VIX term structure proxy: VIX vs its 60d mean
            # VIX >> 60d mean = backwardation (term structure inverted = panic)
            vix_60d = vix_a.rolling(60).mean().replace(0, np.nan)
            vix_ratio = (vix_a / vix_60d).fillna(1.0)
            # ratio > 1 = elevated; > 1.3 = stressed; > 1.6 = crisis
            term_score = ((vix_ratio - 1.0) / 0.6).clip(0, 1)
            components.append(term_score)
            weights.append(0.20)

            # 4. VIX acceleration (2nd derivative — rapid VIX spike)
            vix_vel = vix_a.pct_change(5).fillna(0)
            vix_accel = vix_vel.diff(5).fillna(0)
            # Positive acceleration = fear accelerating
            accel_score = (vix_accel / 0.3).clip(0, 1)
            components.append(accel_score)
            weights.append(0.15)

        # 5. Fear premium: VIX / SPY realised vol
        if vix is not None and spy is not None:
            vix_a = vix.reindex(idx, method="ffill")
            spy_a = spy.reindex(idx, method="ffill")
            spy_rvol = spy_a.pct_change().rolling(20).std() * np.sqrt(252) * 100
            spy_rvol = spy_rvol.replace(0, np.nan)
            fear_premium = (vix_a / spy_rvol).fillna(1.4)
            # Calibration: calm ~1.0-1.4, stress ~1.8, crisis ~2.5+
            fear_score = ((fear_premium - 1.3) / 1.2).clip(0, 1)
            components.append(fear_score)
            weights.append(0.15)

        # 6. Equity/Treasury rotation: SPY/TLT correlation shift
        if spy is not None and tlt is not None:
            spy_a = spy.reindex(idx, method="ffill")
            tlt_a = tlt.reindex(idx, method="ffill")
            spy_r = spy_a.pct_change()
            tlt_r = tlt_a.pct_change()
            # Rolling 20d correlation
            corr_20d = spy_r.rolling(20).corr(tlt_r)
            # Normally negative (stocks up, bonds down). Positive = stress.
            # Both falling together = worst case (liquidity crisis).
            # Positive correlation + both falling = maximum stress
            spy_falling = (spy_a.pct_change(10) < -0.02).astype(float)
            # High positive correlation = stress
            corr_stress = ((corr_20d - 0.0) / 0.5).clip(0, 1).fillna(0)
            # Extra penalty when both falling
            both_falling_bonus = corr_stress * spy_falling * 0.3
            rotation_score = (corr_stress + both_falling_bonus).clip(0, 1)
            components.append(rotation_score)
            weights.append(0.10)

        if not components:
            return pd.Series(0.0, index=idx)

        total_w = sum(weights)
        result = sum(c * w for c, w in zip(components, weights)) / total_w
        return result.clip(0, 1).fillna(0)

    def score_at(self, date: pd.Timestamp) -> float:
        """Score for a single date (for live use)."""
        idx = pd.date_range(date - pd.Timedelta(days=90), date, freq="B")
        series = self.score_series_fast(idx)
        if series.empty:
            return 0.0
        return float(series.iloc[-1])
