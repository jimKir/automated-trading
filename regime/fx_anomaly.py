"""
FX Anomaly Detector
====================
Detects safe-haven flows, USD strength spikes, and EM currency stress
using local parquet data (JPY, EURUSD, DXY).

Score: 0 (benign) → 1.0 (maximum FX stress)

Signals:
  - JPY strengthening (safe-haven bid): USDJPY falling = risk-off
  - DXY spike: rapid USD strengthening = global risk-off
  - EUR/USD breakdown: risk-off capital flight to USD
  - Cross-asset: gold/DXY divergence (gold up + DXY up = crisis)

All features use only past data — no look-ahead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.data_store import get_store
from utils.logger import get_logger

log = get_logger("FXAnomaly")

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data" / "historical" / "daily"


class FXAnomalyDetector:
    """
    FX anomaly score [0, 1] from safe-haven flows and USD stress.
    Uses local parquet data — no live API calls.
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
                log.debug(f"FXAnomaly: {sym} not found in DataStore")
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
            log.warning(f"FXAnomaly: failed to load {sym}: {e}")
            return None

    def score_at(self, date: pd.Timestamp) -> float:
        """Compute FX anomaly score for a single date."""
        scores = []
        weights = []

        # 1. JPY safe-haven (USDJPY falling = JPY strengthening = risk-off)
        jpy = self._load("JPY")
        if jpy is not None:
            jpy_before = jpy[jpy.index <= date]
            if len(jpy_before) >= 20:
                # 5d and 20d JPY change (JPY price = USDJPY, so falling = stress)
                chg_5d = (jpy_before.iloc[-1] - jpy_before.iloc[-5]) / jpy_before.iloc[-5]
                chg_20d = (jpy_before.iloc[-1] - jpy_before.iloc[-20]) / jpy_before.iloc[-20]
                # JPY strengthening (USDJPY falling) = negative change = stress
                jpy_stress_5d = float(np.clip(-chg_5d / 0.03, 0, 1))  # 3% drop = max
                jpy_stress_20d = float(np.clip(-chg_20d / 0.06, 0, 1))  # 6% drop = max
                jpy_score = 0.6 * jpy_stress_5d + 0.4 * jpy_stress_20d
                scores.append(jpy_score)
                weights.append(0.30)

        # 2. DXY spike (rapid USD strengthening = global risk-off)
        dxy = self._load("DXY")
        if dxy is not None:
            dxy_before = dxy[dxy.index <= date]
            if len(dxy_before) >= 20:
                chg_5d = (dxy_before.iloc[-1] - dxy_before.iloc[-5]) / dxy_before.iloc[-5]
                chg_20d = (dxy_before.iloc[-1] - dxy_before.iloc[-20]) / dxy_before.iloc[-20]
                # DXY rising = stress
                dxy_stress_5d = float(np.clip(chg_5d / 0.02, 0, 1))  # 2% rise = max
                dxy_stress_20d = float(np.clip(chg_20d / 0.04, 0, 1))  # 4% rise = max
                # Also: DXY volatility spike
                dxy_vol = dxy_before.pct_change().tail(10).std()
                dxy_vol_baseline = dxy_before.pct_change().tail(60).std()
                if dxy_vol_baseline > 0:
                    dxy_vol_ratio = float(np.clip((dxy_vol / dxy_vol_baseline - 1) / 1.5, 0, 1))
                else:
                    dxy_vol_ratio = 0.0
                dxy_score = 0.4 * dxy_stress_5d + 0.3 * dxy_stress_20d + 0.3 * dxy_vol_ratio
                scores.append(dxy_score)
                weights.append(0.35)

        # 3. EURUSD breakdown (falling EURUSD = flight to USD)
        eur = self._load("EURUSD")
        if eur is not None:
            eur_before = eur[eur.index <= date]
            if len(eur_before) >= 20:
                chg_5d = (eur_before.iloc[-1] - eur_before.iloc[-5]) / eur_before.iloc[-5]
                chg_20d = (eur_before.iloc[-1] - eur_before.iloc[-20]) / eur_before.iloc[-20]
                # EURUSD falling = stress
                eur_stress_5d = float(np.clip(-chg_5d / 0.025, 0, 1))
                eur_stress_20d = float(np.clip(-chg_20d / 0.05, 0, 1))
                eur_score = 0.6 * eur_stress_5d + 0.4 * eur_stress_20d
                scores.append(eur_score)
                weights.append(0.20)

        # 4. Gold/DXY divergence (both up = crisis flight)
        gld = self._load("GLD")
        if gld is not None and dxy is not None:
            gld_before = gld[gld.index <= date]
            dxy_before2 = dxy[dxy.index <= date]
            if len(gld_before) >= 10 and len(dxy_before2) >= 10:
                gld_chg = (gld_before.iloc[-1] - gld_before.iloc[-10]) / gld_before.iloc[-10]
                dxy_chg = (dxy_before2.iloc[-1] - dxy_before2.iloc[-10]) / dxy_before2.iloc[-10]
                # Both rising = crisis signal
                if gld_chg > 0.01 and dxy_chg > 0.005:
                    divergence = float(np.clip((gld_chg + dxy_chg) / 0.04, 0, 1))
                else:
                    divergence = 0.0
                scores.append(divergence)
                weights.append(0.15)

        if not scores:
            return 0.0

        # Weighted average of available sources
        total_w = sum(weights)
        return float(np.clip(sum(s * w for s, w in zip(scores, weights)) / total_w, 0, 1))

    def score_series(self, start: str, end: str) -> pd.Series:
        """Compute daily FX anomaly score for a date range."""
        idx = pd.date_range(start, end, freq="B")
        result = pd.Series(0.0, index=idx)
        for date in idx:
            result[date] = self.score_at(date)
        return result

    def score_series_fast(self, idx: pd.DatetimeIndex) -> pd.Series:
        """Vectorised FX anomaly score for backtest (faster than score_at loop)."""
        jpy = self._load("JPY")
        dxy = self._load("DXY")
        eur = self._load("EURUSD")
        gld = self._load("GLD")

        components = []
        weights = []

        # JPY safe-haven
        if jpy is not None:
            jpy_a = jpy.reindex(idx, method="ffill")
            chg_5d = jpy_a.pct_change(5)
            chg_20d = jpy_a.pct_change(20)
            jpy_score = 0.6 * (-chg_5d / 0.03).clip(0, 1) + 0.4 * (-chg_20d / 0.06).clip(0, 1)
            components.append(jpy_score.fillna(0))
            weights.append(0.30)

        # DXY spike
        if dxy is not None:
            dxy_a = dxy.reindex(idx, method="ffill")
            chg_5d = dxy_a.pct_change(5)
            chg_20d = dxy_a.pct_change(20)
            dxy_vol = dxy_a.pct_change().rolling(10).std()
            dxy_vol_base = dxy_a.pct_change().rolling(60).std().replace(0, np.nan)
            vol_ratio = ((dxy_vol / dxy_vol_base - 1) / 1.5).clip(0, 1).fillna(0)
            dxy_score = (
                0.4 * (chg_5d / 0.02).clip(0, 1)
                + 0.3 * (chg_20d / 0.04).clip(0, 1)
                + 0.3 * vol_ratio
            )
            components.append(dxy_score.fillna(0))
            weights.append(0.35)

        # EURUSD breakdown
        if eur is not None:
            eur_a = eur.reindex(idx, method="ffill")
            chg_5d = eur_a.pct_change(5)
            chg_20d = eur_a.pct_change(20)
            eur_score = 0.6 * (-chg_5d / 0.025).clip(0, 1) + 0.4 * (-chg_20d / 0.05).clip(0, 1)
            components.append(eur_score.fillna(0))
            weights.append(0.20)

        # Gold/DXY divergence
        if gld is not None and dxy is not None:
            gld_a = gld.reindex(idx, method="ffill")
            dxy_a2 = dxy.reindex(idx, method="ffill")
            gld_chg = gld_a.pct_change(10)
            dxy_chg = dxy_a2.pct_change(10)
            both_up = (gld_chg > 0.01) & (dxy_chg > 0.005)
            divergence = ((gld_chg + dxy_chg) / 0.04).clip(0, 1) * both_up.astype(float)
            components.append(divergence.fillna(0))
            weights.append(0.15)

        if not components:
            return pd.Series(0.0, index=idx)

        total_w = sum(weights)
        result = sum(c * w for c, w in zip(components, weights)) / total_w
        return result.clip(0, 1).fillna(0)
