"""
Position Anomaly Detector
==========================
Uses an Isolation Forest trained on normal market behaviour of each
instrument (volatility, volume, correlation, return dispersion).

Key design choices to prevent overfitting:
  - Unsupervised: no crisis labels, no look-ahead
  - Walk-forward: retrained monthly on rolling 3-year window
  - Features derived from price/volume only — always available
  - Contamination fixed at 0.05 (5% of days flagged as anomalous)
    — set by domain knowledge, not optimisation

Output: anomaly_score ∈ [0.0, 1.0]
  0.0 = perfectly normal
  1.0 = extreme outlier behaviour
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from utils.logger import get_logger

log = get_logger("AnomalyDetector")
warnings.filterwarnings("ignore", category=UserWarning)

# Walk-forward: retrain every N trading days
RETRAIN_EVERY_DAYS = 21        # monthly
TRAIN_WINDOW_DAYS  = 756       # 3 years of history
MIN_TRAIN_DAYS     = 120       # minimum before first fit


class PositionAnomalyDetector:
    """
    Isolation Forest anomaly detector on position-level market behaviour.
    Detects when instruments are acting abnormally — rising vol, volume spikes,
    correlation breakdown, unusual return dispersion.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators   = n_estimators
        self._model:  Optional[IsolationForest] = None
        self._scaler: Optional[RobustScaler]    = None
        self._last_train_idx: int = 0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    def _build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from a multi-asset price DataFrame.
        All features are stationary and normalised internally.

        Features per day:
          - Cross-sectional realised vol (median and dispersion)
          - Cross-sectional return dispersion (std of daily returns)
          - Average absolute z-score of returns vs 20d mean
          - Correlation spike: mean pairwise correlation (rolling 10d)
          - Vol-of-vol: rolling std of 10d realised vol
          - Average vol ratio: current 10d vol / 60d vol baseline
        """
        # FIX: strip timezone info to prevent datetime64[ns, UTC] vs
        # datetime64[ns] comparison errors when reindexing against tz-naive dates
        if hasattr(prices.index, "tz") and prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)
        rets = prices.pct_change().dropna()

        if rets.shape[0] < 20:
            return pd.DataFrame()

        features = pd.DataFrame(index=rets.index)

        # 1. Cross-sectional return dispersion
        features["ret_dispersion"] = rets.std(axis=1)

        # 2. Median absolute return (market-wide move size)
        features["median_abs_ret"] = rets.abs().median(axis=1)

        # 3. Average z-score of each asset's return vs its own 20d history
        roll_mean = rets.rolling(20).mean()
        roll_std  = rets.rolling(20).std().replace(0, np.nan)
        zscores   = ((rets - roll_mean) / roll_std).abs()
        features["avg_zscore"] = zscores.mean(axis=1)

        # 4. Rolling 10-day realised vol (cross-sectional median)
        vol_10 = rets.rolling(10).std()
        features["vol_10d_median"] = vol_10.median(axis=1)

        # 5. Vol ratio: 10d vol vs 60d vol baseline (vol spike detector)
        vol_60 = rets.rolling(60).std().replace(0, np.nan)
        vol_ratio = (vol_10 / vol_60)
        features["vol_ratio_median"] = vol_ratio.median(axis=1)
        features["vol_ratio_max"]    = vol_ratio.max(axis=1)

        # 6. Vol-of-vol: instability in the vol regime itself
        features["vol_of_vol"] = features["vol_10d_median"].rolling(10).std()

        # 7. Pairwise correlation spike (10d rolling, upper triangle mean)
        try:
            roll_corr_vals = []
            for i in range(len(rets)):
                start = max(0, i - 10)
                window = rets.iloc[start:i+1]
                if window.shape[0] >= 5:
                    corr_mat = window.corr().values
                    n = corr_mat.shape[0]
                    upper = corr_mat[np.triu_indices(n, k=1)]
                    roll_corr_vals.append(np.nanmean(np.abs(upper)))
                else:
                    roll_corr_vals.append(np.nan)
            features["corr_spike"] = roll_corr_vals
        except Exception:
            features["corr_spike"] = np.nan

        # 8. Max single-asset drawdown over 5 days
        roll_max  = prices.rolling(5).max()
        dd_5d     = ((prices - roll_max) / roll_max.replace(0, np.nan)).min(axis=1)
        features["max_dd_5d"] = dd_5d.abs()

        return features.dropna()

    # ------------------------------------------------------------------
    def fit(self, prices: pd.DataFrame) -> bool:
        """Fit (or refit) the Isolation Forest on historical price data."""
        features = self._build_features(prices)
        if len(features) < MIN_TRAIN_DAYS:
            log.debug(f"AnomalyDetector: not enough data to fit ({len(features)} rows)")
            return False

        # Use last TRAIN_WINDOW_DAYS only
        features = features.iloc[-TRAIN_WINDOW_DAYS:]

        self._scaler = RobustScaler()
        X = self._scaler.fit_transform(features.values)

        self._model = IsolationForest(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            max_samples   = min(256, len(X)),
            random_state  = 42,
        )
        self._model.fit(X)
        self._is_fitted = True
        log.debug(f"AnomalyDetector: fitted on {len(X)} rows")
        return True

    # ------------------------------------------------------------------
    def score(self, prices: pd.DataFrame) -> float:
        """
        Return anomaly score for the most recent day ∈ [0, 1].
        0 = normal, 1 = extreme anomaly.
        """
        if not self._is_fitted:
            return 0.0

        features = self._build_features(prices)
        if features.empty:
            return 0.0

        last_row = features.iloc[[-1]]
        X = self._scaler.transform(last_row.values)

        # decision_function: higher = more normal, lower = more anomalous
        raw_score = self._model.decision_function(X)[0]

        # Normalise: typical range is roughly [-0.2, 0.2]
        # Convert so that anomalous → high score
        normalised = float(np.clip((-raw_score + 0.1) / 0.3, 0.0, 1.0))
        return normalised

    # ------------------------------------------------------------------
    def score_series(self, prices: pd.DataFrame) -> pd.Series:
        """
        Score every day in the price history (for backtest use).
        Uses walk-forward: fits on data up to each point, never future data.
        """
        # FIX: ensure tz-naive index before feature computation
        if hasattr(prices.index, "tz") and prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)
        features = self._build_features(prices)
        if features.empty:
            return pd.Series(dtype=float)

        scores = pd.Series(index=features.index, dtype=float)
        scores[:] = 0.0

        for i in range(MIN_TRAIN_DAYS, len(features)):
            # Retrain every RETRAIN_EVERY_DAYS
            if (i - MIN_TRAIN_DAYS) % RETRAIN_EVERY_DAYS == 0:
                train_start = max(0, i - TRAIN_WINDOW_DAYS)
                train_feats = features.iloc[train_start:i]
                scaler = RobustScaler()
                X_train = scaler.fit_transform(train_feats.values)
                model = IsolationForest(
                    n_estimators  = self.n_estimators,
                    contamination = self.contamination,
                    max_samples   = min(256, len(X_train)),
                    random_state  = 42,
                )
                model.fit(X_train)
                self._model  = model
                self._scaler = scaler
                self._is_fitted = True

            if not self._is_fitted:
                continue

            X_test = self._scaler.transform(features.iloc[[i]].values)
            raw = self._model.decision_function(X_test)[0]
            scores.iloc[i] = float(np.clip((-raw + 0.1) / 0.3, 0.0, 1.0))

        return scores
