"""
H2O AutoML Trend-Following Classifier
=======================================
Uses H2O gradient-boosted models to classify the current market regime
into one of 5 trend states, then outputs a multiplier that lets the
system **ride strong trends longer** and **exit weak trends faster**.

Trend States
────────────
  STRONG_UP    (+2)  → multiplier 1.40  (boost long signals, hold longer)
  MILD_UP      (+1)  → multiplier 1.15  (slight boost)
  NEUTRAL      ( 0)  → multiplier 1.00  (no change)
  MILD_DOWN    (-1)  → multiplier 0.85  (dampen, tighten stops)
  STRONG_DOWN  (-2)  → multiplier 0.60  (aggressive risk reduction)

Features (28 total — all strictly causal)
──────────────────────────────────────────
  Price features:
    - Returns: 1d, 5d, 10d, 21d, 63d
    - Distance from 20/50/200-day MA (%)
    - Rate-of-change (ROC) 10d, 21d
    - ADX (trend strength, 14-day)
    - Price position in 63d range (0=low, 1=high)

  Volume features:
    - Volume ratio: 5d/20d, 10d/50d
    - Volume trend slope (20d OLS)
    - On-balance volume slope (20d)

  Volatility features:
    - Realised vol 5d, 21d
    - Vol ratio 5d/21d (vol contraction/expansion)
    - Parkinson range vol 10d

  Cross-asset features:
    - VIX level, VIX 5d change
    - SPY distance from MA200

  Calendar:
    - Day of week, month (cyclical)

Target label (for training):
  Forward 10-day return → bucketed into 5 quintiles → mapped to trend states.
  This is a CLASSIFICATION task, not regression — H2O AutoML picks the best
  classifier (typically GBM or XGBoost with early stopping).

Anti-overfitting:
  - Walk-forward training: retrain monthly with expanding window
  - H2O uses 5-fold CV internally with early stopping
  - Features are all standard technical indicators (no curve-fitting)
  - 5-class classification prevents overfitting to specific return levels
  - Multipliers are symmetric and bounded [0.60, 1.40]
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

warnings.filterwarnings('ignore', category=DeprecationWarning, module='h2o')
log = get_logger('H2OTrendClassifier')

MODEL_DIR = Path(__file__).parent.parent / 'models' / 'h2o_trend_model'

# Trend state → signal multiplier
TREND_MULTIPLIERS = {
    'STRONG_UP':   1.40,
    'MILD_UP':     1.15,
    'NEUTRAL':     1.00,
    'MILD_DOWN':   0.85,
    'STRONG_DOWN': 0.60,
}

# Label encoding
LABEL_MAP = {0: 'STRONG_DOWN', 1: 'MILD_DOWN', 2: 'NEUTRAL', 3: 'MILD_UP', 4: 'STRONG_UP'}
LABEL_TO_INT = {v: k for k, v in LABEL_MAP.items()}


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength regardless of direction."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Zero out when opposite DM is larger
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0)
    minus_dm = minus_dm.where(~mask, 0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(window).mean()
    return adx.fillna(0)


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = 10) -> pd.Series:
    """Parkinson range-based volatility estimator (more efficient than close-close)."""
    log_hl = np.log(high / low.replace(0, np.nan))
    return (log_hl ** 2).rolling(window).mean().apply(
        lambda x: np.sqrt(x / (4 * np.log(2)) * 252) if x > 0 else 0
    )


class H2OTrendClassifier:
    """
    Classifies the trend regime for a symbol and returns a signal multiplier.

    If H2O is not available, falls back to a simple rule-based trend detector
    (MA crossover + ADX) that approximates the same 5 states.
    """

    def __init__(self, config: dict):
        tc_cfg = config.get("trend_classifier", {})
        self.enabled = tc_cfg.get("enabled", True)
        self.weight  = tc_cfg.get("weight", 0.20)   # blend weight in final signal
        self.retrain_frequency = tc_cfg.get("retrain_frequency", "monthly")

        # H2O model state
        self._h2o_model = None
        self._h2o = None
        self._h2o_loaded = False
        self._h2o_tried = False

        # Training state for walk-forward
        self._last_train_date = None
        self._train_buffer: list = []

        if self.enabled:
            log.info(f"TrendClassifier: enabled, weight={self.weight:.0%}")

    def _try_load_h2o(self):
        """Attempt to load H2O and any pre-trained model."""
        if self._h2o_tried:
            return
        self._h2o_tried = True
        try:
            import h2o
            try:
                h2o.cluster()
            except Exception:
                h2o.init(nthreads=-1, max_mem_size='2g', verbose=False)
                h2o.no_progress()
            self._h2o = h2o
            self._h2o_loaded = True

            # Try loading pre-trained model
            model_path_file = MODEL_DIR / 'best_model_path.txt'
            if model_path_file.exists():
                mp = model_path_file.read_text().strip()
                if Path(mp).exists():
                    self._h2o_model = h2o.load_model(mp)
                    log.info(f'TrendClassifier: loaded pre-trained model from {mp}')

            if self._h2o_model is None:
                log.info('TrendClassifier: H2O available, no pre-trained model — will train on first call')
        except Exception as e:
            log.info(f'TrendClassifier: H2O not available ({e}) — using rule-based fallback')
            self._h2o_loaded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ─────────────────────────────────────────────────────────────────────────

    def build_features(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
        spy_close: Optional[pd.Series] = None,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Build feature matrix from price history. Returns DataFrame with one
        row per date. All features strictly causal (use only past data).
        """
        if as_of_date is not None:
            close = close[close.index <= as_of_date]
            if high is not None: high = high[high.index <= as_of_date]
            if low is not None: low = low[low.index <= as_of_date]
            if volume is not None: volume = volume[volume.index <= as_of_date]

        if len(close) < 210:  # need 200d MA + buffer
            return None

        feats = pd.DataFrame(index=close.index)

        # Price returns
        for w in [1, 5, 10, 21, 63]:
            feats[f'ret_{w}d'] = close.pct_change(w)

        # Distance from MAs
        for w in [20, 50, 200]:
            ma = close.rolling(w).mean()
            feats[f'dist_ma{w}'] = (close - ma) / ma

        # Rate of change
        feats['roc_10'] = close.pct_change(10)
        feats['roc_21'] = close.pct_change(21)

        # ADX (trend strength)
        if high is not None and low is not None:
            feats['adx_14'] = _adx(high, low, close, 14)
            feats['price_pos_63'] = (close - low.rolling(63).min()) / \
                (high.rolling(63).max() - low.rolling(63).min()).replace(0, np.nan)
            feats['parkinson_vol_10'] = _parkinson_vol(high, low, 10)
        else:
            feats['adx_14'] = 0.0
            feats['price_pos_63'] = 0.5
            feats['parkinson_vol_10'] = 0.0

        # Volume features
        if volume is not None and (volume > 0).sum() > 50:
            feats['vol_ratio_5_20'] = volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)
            feats['vol_ratio_10_50'] = volume.rolling(10).mean() / volume.rolling(50).mean().replace(0, np.nan)

            # Volume trend slope (20d)
            vol_norm = volume / volume.rolling(50).mean().replace(0, np.nan)
            feats['vol_slope_20'] = vol_norm.rolling(20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=True
            )

            # OBV slope
            obv = (np.sign(close.diff()) * volume).cumsum()
            obv_norm = obv / obv.rolling(50).std().replace(0, np.nan)
            feats['obv_slope_20'] = obv_norm.rolling(20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=True
            )
        else:
            feats['vol_ratio_5_20'] = 1.0
            feats['vol_ratio_10_50'] = 1.0
            feats['vol_slope_20'] = 0.0
            feats['obv_slope_20'] = 0.0

        # Volatility
        rets = close.pct_change()
        feats['rv_5'] = rets.rolling(5).std() * np.sqrt(252)
        feats['rv_21'] = rets.rolling(21).std() * np.sqrt(252)
        feats['vol_ratio_5_21'] = feats['rv_5'] / feats['rv_21'].replace(0, np.nan)

        # VIX features
        if vix is not None and len(vix) > 0:
            vix_aligned = vix.reindex(close.index, method='ffill')
            feats['vix'] = vix_aligned / 100.0
            feats['vix_chg_5'] = vix_aligned.pct_change(5)
        else:
            feats['vix'] = 0.15
            feats['vix_chg_5'] = 0.0

        # SPY distance from MA200 (cross-asset regime)
        if spy_close is not None and len(spy_close) >= 200:
            spy_al = spy_close.reindex(close.index, method='ffill')
            spy_ma200 = spy_al.rolling(200).mean()
            feats['spy_dist_ma200'] = (spy_al - spy_ma200) / spy_ma200
        else:
            feats['spy_dist_ma200'] = 0.0

        # Calendar
        feats['dow'] = close.index.dayofweek / 4.0
        feats['month'] = close.index.month / 12.0

        return feats.replace([np.inf, -np.inf], np.nan).fillna(0)

    def _build_labels(self, close: pd.Series, horizon: int = 10) -> pd.Series:
        """Build 5-class trend labels from forward returns."""
        fwd_ret = close.pct_change(horizon).shift(-horizon)
        # Quintile bucketing → 0..4
        labels = fwd_ret.rank(pct=True).apply(
            lambda x: 0 if x < 0.20 else (1 if x < 0.40 else (2 if x < 0.60 else (3 if x < 0.80 else 4)))
        )
        return labels

    # ─────────────────────────────────────────────────────────────────────────
    # Walk-forward training
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        all_data: Dict[str, pd.DataFrame],
        vix: Optional[pd.Series] = None,
        spy_close: Optional[pd.Series] = None,
        train_end: Optional[pd.Timestamp] = None,
        max_runtime_secs: int = 120,
    ) -> bool:
        """
        Train H2O AutoML trend classifier on pooled multi-symbol data.
        Uses expanding window up to train_end.
        Returns True if training succeeded.
        """
        if not self._h2o_loaded:
            self._try_load_h2o()
        if not self._h2o_loaded:
            return False

        log.info(f'TrendClassifier: training on data up to {train_end}...')

        all_features = []
        all_labels = []

        for sym, df in all_data.items():
            if 'Close' not in df.columns or len(df) < 252:
                continue

            close = df['Close']
            if train_end is not None:
                close_tr = close[close.index <= train_end]
            else:
                close_tr = close

            if len(close_tr) < 252:
                continue

            high = df.get('High')
            low = df.get('Low')
            volume = df.get('Volume')

            feats = self.build_features(close_tr, high, low, volume, vix, spy_close, train_end)
            if feats is None or len(feats) < 100:
                continue

            labels = self._build_labels(close_tr)

            # Only use rows where both features and labels are valid
            valid = labels.notna() & feats.notna().all(axis=1)
            feats_valid = feats[valid]
            labels_valid = labels[valid]

            if len(feats_valid) < 50:
                continue

            feats_valid = feats_valid.copy()
            feats_valid['_label'] = labels_valid.astype(str)
            all_features.append(feats_valid)

        if not all_features:
            log.warning('TrendClassifier: no valid training data')
            return False

        train_df = pd.concat(all_features, ignore_index=True)
        log.info(f'TrendClassifier: training on {len(train_df)} rows, '
                 f'{len(train_df.columns)-1} features')

        try:
            h2o = self._h2o
            train_h2o = h2o.H2OFrame(train_df)
            train_h2o['_label'] = train_h2o['_label'].asfactor()

            feature_cols = [c for c in train_df.columns if c != '_label']

            from h2o.automl import H2OAutoML
            aml = H2OAutoML(
                max_runtime_secs=max_runtime_secs,
                max_models=10,
                seed=42,
                sort_metric='logloss',
                exclude_algos=['DeepLearning'],  # too slow for this
            )
            aml.train(x=feature_cols, y='_label', training_frame=train_h2o)

            self._h2o_model = aml.leader
            self._last_train_date = train_end

            # Save model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = h2o.save_model(self._h2o_model, path=str(MODEL_DIR), force=True)
            (MODEL_DIR / 'best_model_path.txt').write_text(model_path)

            perf = self._h2o_model.model_performance()
            log.info(f'TrendClassifier: trained successfully. '
                     f'Logloss={perf.logloss():.4f}')
            return True

        except Exception as e:
            log.warning(f'TrendClassifier: training failed: {e}')
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
        spy_close: Optional[pd.Series] = None,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> Tuple[str, float]:
        """
        Predict trend state and return (state_name, multiplier).
        Falls back to rule-based if H2O not available.
        """
        if not self._h2o_tried:
            self._try_load_h2o()

        # Try H2O prediction
        if self._h2o_loaded and self._h2o_model is not None:
            try:
                feats = self.build_features(close, high, low, volume, vix, spy_close, as_of_date)
                if feats is not None and len(feats) > 0:
                    last_row = feats.iloc[[-1]]
                    pred_h2o = self._h2o.H2OFrame(last_row)
                    result = self._h2o_model.predict(pred_h2o).as_data_frame()
                    predicted_class = str(result['predict'].iloc[0])
                    state = LABEL_MAP.get(int(predicted_class), 'NEUTRAL')
                    return state, TREND_MULTIPLIERS[state]
            except Exception as e:
                log.debug(f'TrendClassifier H2O predict failed: {e}')

        # Rule-based fallback
        return self._rule_based_predict(close, high, low, as_of_date)

    def _rule_based_predict(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> Tuple[str, float]:
        """
        Simple rule-based trend classifier as fallback.
        Uses MA crossover + ADX for trend detection.
        """
        if as_of_date is not None:
            close = close[close.index <= as_of_date]

        if len(close) < 200:
            return 'NEUTRAL', 1.0

        price = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]

        # ADX for trend strength
        adx_val = 25.0  # default moderate
        if high is not None and low is not None:
            if as_of_date is not None:
                high = high[high.index <= as_of_date]
                low = low[low.index <= as_of_date]
            adx_series = _adx(high, low, close, 14)
            if len(adx_series) > 0:
                adx_val = adx_series.iloc[-1]

        # Score: positive = uptrend, negative = downtrend
        score = 0
        if price > ma20: score += 1
        else: score -= 1
        if price > ma50: score += 1
        else: score -= 1
        if price > ma200: score += 1
        else: score -= 1
        if ma20 > ma50: score += 1
        else: score -= 1

        # ADX amplifies: strong trend → extreme states
        strong = adx_val > 30

        if score >= 3 and strong:
            state = 'STRONG_UP'
        elif score >= 2:
            state = 'MILD_UP'
        elif score <= -3 and strong:
            state = 'STRONG_DOWN'
        elif score <= -2:
            state = 'MILD_DOWN'
        else:
            state = 'NEUTRAL'

        return state, TREND_MULTIPLIERS[state]

    def get_multiplier(
        self,
        close: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
        spy_close: Optional[pd.Series] = None,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> float:
        """Convenience: returns just the multiplier (for signal pipeline)."""
        if not self.enabled:
            return 1.0
        _, mult = self.predict(close, high, low, volume, vix, spy_close, as_of_date)
        return mult
