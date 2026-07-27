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

# Vol-of-VIX calibration, from the measured distribution of 10d realised vol of VIX
# daily returns (2015-2026): median 0.062, elevated ~0.12, crisis ~0.20.
VVIX_CALM = 0.06
VVIX_CRISIS = 0.20

# SPY/TLT correlation is scored relative to its own trailing median rather than to a
# fixed sign, because the stock/bond correlation regime flipped after 2022.
CORR_BASELINE_WINDOW = 252
CORR_EXCESS_SPAN = 0.50

# Calendar days of history the single-date live path pulls. Must comfortably exceed
# CORR_BASELINE_WINDOW business days so live and backtest scores agree.
SCORE_AT_LOOKBACK_DAYS = 500


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
            # Calm baseline was 0.030, which is ~half the true median: measured median of
            # this statistic is 0.062 over 2015-2026 (0.079 in 2026). That floor made a
            # typical day score ~0.27-0.41 and kept the composite pinned at ELEVATED.
            vvix_score = ((vix_rvol - VVIX_CALM) / (VVIX_CRISIS - VVIX_CALM)).clip(0, 1)
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
            # Measured against its OWN trailing baseline, not a hardcoded sign assumption.
            # The previous version scored any positive correlation as stress ((corr-0)/0.5),
            # which was true pre-2022 (mean -0.39) but is the structural norm post-2022
            # (mean +0.12, +0.34 in 2026). It therefore reported ~0.56 on calm days and was
            # the single largest contributor to the layer being stuck at ELEVATED.
            # Stress is now a *rise above* the prevailing regime, so this self-recalibrates.
            corr_baseline = corr_20d.rolling(CORR_BASELINE_WINDOW, min_periods=60).median()
            corr_baseline = corr_baseline.fillna(corr_20d.expanding(min_periods=20).median())
            corr_excess = (corr_20d - corr_baseline) / CORR_EXCESS_SPAN
            corr_stress = corr_excess.clip(0, 1).fillna(0)
            # Extra penalty when equities are also falling (liquidity-crisis signature)
            spy_falling = (spy_a.pct_change(10) < -0.02).astype(float)
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
        # Must span the longest rolling window used above (the 252-bar correlation
        # baseline) or the live score silently differs from the backtest series, which
        # only had ~64 bars here and so ran on unwarmed rolling windows.
        idx = pd.date_range(date - pd.Timedelta(days=SCORE_AT_LOOKBACK_DAYS), date, freq="B")
        series = self.score_series_fast(idx)
        if series.empty:
            return 0.0
        return float(series.iloc[-1])
