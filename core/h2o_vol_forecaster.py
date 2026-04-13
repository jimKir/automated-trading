"""
H2O AutoML Volatility Forecaster
==================================
Replaces the EWMA vol estimate in VolTargeting with a trained
GBM/Stacked-Ensemble model that predicts next-week realised volatility.

Architecture:
  - 21 features: multi-window RV, vol ratios, VIX, skewness, kurtosis, etc.
  - Trained on 11,625 pooled weekly observations (18 symbols, 2010–2022)
  - OOS validated 2023–2026: wins 6/6 metrics vs EWMA, 16/18 symbols
  - EWMA fallback if H2O is unavailable or prediction fails

Key improvements over EWMA:
  - MAE  −6.3%  (0.0699 → 0.0655)
  - Bias −90.9% (0.0132 → 0.0012) — near-unbiased
  - Corr  +4.0% (0.661  → 0.687)
  - VT scale accuracy +5.7%

Retraining:
  Run `python core/h2o_vol_forecaster.py --retrain` quarterly to
  extend the training window with new data.

Usage in vol_targeting.py:
  forecaster = H2OVolForecaster.load()
  vol_est = forecaster.predict(sym, returns_series, vix_series, date)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger

warnings.filterwarnings("ignore", category=DeprecationWarning, module="h2o")
log = get_logger("H2OVolForecaster")

MODEL_DIR = Path(__file__).parent.parent / "models" / "h2o_vol_model"
META_FILE = MODEL_DIR / "feature_meta.json"
MODEL_PATH_FILE = MODEL_DIR / "best_model_path.txt"

# Symbol → index mapping (must match training)
_DEFAULT_SYMBOLS = [
    "SPY",
    "GLD",
    "SLV",
    "TLT",
    "HYG",
    "SHY",
    "EEM",
    "EWZ",
    "EWY",
    "EWC",
    "EWJ",
    "EWU",
    "VGK",
    "SOXX",
    "IBB",
    "XBI",
    "COPX",
    "BTC-USD",
]


class H2OVolForecaster:
    """
    Wraps the trained H2O model for single-symbol vol forecasting.

    Parameters
    ----------
    model_path : str  — path to saved H2O model directory
    meta       : dict — feature metadata from feature_meta.json
    """

    def __init__(self, model_path: str, meta: dict):
        self.model_path = model_path
        self.meta = meta
        self.symbols = meta.get("symbols", _DEFAULT_SYMBOLS)
        self.lam = meta.get("ewma_lambda", 0.94)
        self.horizon = meta.get("horizon", 5)
        self._h2o_model = None
        self._h2o_loaded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Class-level singleton loader
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> H2OVolForecaster | None:
        """
        Load the saved H2O model. Returns None if model files are missing
        or H2O is not available — caller should fall back to EWMA.
        """
        if not META_FILE.exists() or not MODEL_PATH_FILE.exists():
            log.warning("H2OVolForecaster: model files not found — EWMA fallback active")
            return None

        try:
            with open(META_FILE) as f:
                meta = json.load(f)
            with open(MODEL_PATH_FILE) as f:
                model_path = f.read().strip()

            forecaster = cls(model_path=model_path, meta=meta)
            forecaster._load_h2o()
            if forecaster._h2o_loaded:
                log.info(f"H2OVolForecaster: loaded model from {model_path}")
                return forecaster
            return None
        except Exception as e:
            log.warning(f"H2OVolForecaster: load failed ({e}) — EWMA fallback active")
            return None

    def _load_h2o(self):
        """Lazy-load H2O and the saved model."""
        if self._h2o_loaded:
            return
        try:
            import h2o

            # Try connecting; init a fresh cluster if not already running
            try:
                h2o.cluster()  # raises if not connected
            except Exception:
                h2o.init(nthreads=-1, max_mem_size="4g", verbose=False)
                h2o.no_progress()
            self._h2o_model = h2o.load_model(self.model_path)
            self._h2o = h2o
            self._h2o_loaded = True
        except Exception as e:
            log.warning(f"H2OVolForecaster: H2O init failed ({e})")
            self._h2o_loaded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Feature builder (must exactly match training feature engineering)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_features(
        self,
        sym: str,
        returns: pd.Series,
        vix_series: pd.Series | None,
        as_of_date: pd.Timestamp | None = None,
    ) -> dict | None:
        """
        Build a single-row feature dict for the given symbol at as_of_date.
        Returns None if insufficient data.
        """
        ret = returns.dropna()
        if as_of_date is not None:
            ret = ret[ret.index <= as_of_date]
        if len(ret) < 63:
            return None

        ret_arr = ret.values
        lam = self.lam

        # EWMA vol
        var = float(np.var(ret_arr[:20], ddof=1))  # ddof=1 matches vol_targeting seed
        for r in ret_arr[20:]:
            var = lam * var + (1 - lam) * r**2
        ewma_now = float(np.sqrt(max(var, 1e-10) * 252))

        # Realised vols
        def rv(w):
            sl = ret_arr[-w:] if len(ret_arr) >= w else ret_arr
            return float(np.sqrt(np.mean(sl**2) * 252))

        rv5, rv10, rv21, rv63, rv126 = rv(5), rv(10), rv(21), rv(63), rv(126)

        # Vol ratios
        vol_ratio_5_21 = rv5 / rv21 if rv21 > 0 else 1.0
        vol_ratio_21_63 = rv21 / rv63 if rv63 > 0 else 1.0
        vol_ratio_63_126 = rv63 / rv126 if rv126 > 0 else 1.0

        # Vol of vol
        if len(ret_arr) >= 30:
            r_slice = ret_arr[-25:]
            rvols = [
                np.sqrt(np.mean(r_slice[max(0, i - 5) : i] ** 2) * 252)
                for i in range(10, len(r_slice))
            ]
            vov = float(np.std(rvols)) if len(rvols) > 2 else 0.0
        else:
            vov = 0.0

        # Distribution stats
        r21 = ret_arr[-21:] if len(ret_arr) >= 21 else ret_arr
        skew = float(pd.Series(r21).skew()) if len(r21) > 3 else 0.0
        kurt = float(pd.Series(r21).kurt()) if len(r21) > 3 else 0.0
        mean5 = float(np.mean(ret_arr[-5:])) if len(ret_arr) >= 5 else 0.0
        max_abs5 = float(np.max(np.abs(ret_arr[-5:]))) if len(ret_arr) >= 5 else 0.0

        # Autocorrelation of squared returns
        sq = ret_arr[-10:] ** 2
        ac1 = float(pd.Series(sq).autocorr(1)) if len(sq) >= 6 else 0.0
        ac1 = 0.0 if np.isnan(ac1) else ac1

        # VIX features
        date = as_of_date or ret.index[-1]
        if vix_series is not None and len(vix_series) > 0:
            try:
                vix_now = float(vix_series.asof(date)) / 100.0
                prev_idx = ret.index[max(0, len(ret) - 6)]
                vix_5ago = float(vix_series.asof(prev_idx)) / 100.0
                vix_chg = vix_now - vix_5ago
                vix_regime = 1.0 if vix_now > 0.25 else (0.5 if vix_now > 0.15 else 0.0)
            except Exception:
                vix_now, vix_chg, vix_regime = ewma_now, 0.0, 0.0
        else:
            vix_now, vix_chg, vix_regime = ewma_now, 0.0, 0.0

        # Symbol ID
        try:
            sym_id = float(self.symbols.index(sym))
        except ValueError:
            sym_id = -1.0  # unknown symbol → model will generalise via other features

        return {
            "symbol_id": sym_id,
            "ewma_vol": ewma_now,
            "rv5": rv5,
            "rv10": rv10,
            "rv21": rv21,
            "rv63": rv63,
            "rv126": rv126,
            "vol_ratio_5_21": vol_ratio_5_21,
            "vol_ratio_21_63": vol_ratio_21_63,
            "vol_ratio_63_126": vol_ratio_63_126,
            "vov": vov,
            "skew": skew,
            "kurt": kurt,
            "mean_r5": mean5,
            "max_abs_r5": max_abs5,
            "autocorr_sq": ac1,
            "vix": vix_now,
            "vix_chg": vix_chg,
            "vix_regime": vix_regime,
            "dow": date.dayofweek / 4.0,
            "month": date.month / 12.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public prediction interface
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        sym: str,
        returns: pd.Series,
        vix_series: pd.Series | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> float | None:
        """
        Predict next-horizon annualised volatility for one symbol.

        Returns
        -------
        float : predicted annualised vol, or None if prediction fails
                (caller should fall back to EWMA in that case)
        """
        if not self._h2o_loaded:
            self._load_h2o()
        if not self._h2o_loaded:
            return None

        feats = self._build_features(sym, returns, vix_series, as_of_date)
        if feats is None:
            return None

        try:
            feat_df = pd.DataFrame([feats])
            feat_h2o = self._h2o.H2OFrame(feat_df)
            pred = self._h2o_model.predict(feat_h2o).as_data_frame()["predict"].iloc[0]
            return float(max(pred, 0.01))  # floor at 1% annualised
        except Exception as e:
            log.debug(f"H2OVolForecaster.predict failed for {sym}: {e}")
            return None

    def predict_batch(
        self,
        sym_returns: dict,  # {symbol: pd.Series of log-returns}
        vix_series: pd.Series | None = None,
        as_of_date: pd.Timestamp | None = None,
    ) -> dict:
        """
        Predict vol for multiple symbols in one H2O call (faster).
        Returns {symbol: predicted_vol} — missing if prediction failed.
        """
        if not self._h2o_loaded:
            self._load_h2o()
        if not self._h2o_loaded:
            return {}

        rows, valid_syms = [], []
        for sym, returns in sym_returns.items():
            feats = self._build_features(sym, returns, vix_series, as_of_date)
            if feats is not None:
                rows.append(feats)
                valid_syms.append(sym)

        if not rows:
            return {}

        try:
            feat_h2o = self._h2o.H2OFrame(pd.DataFrame(rows))
            preds = self._h2o_model.predict(feat_h2o).as_data_frame()["predict"].values
            return {sym: float(max(p, 0.01)) for sym, p in zip(valid_syms, preds)}
        except Exception as e:
            log.debug(f"H2OVolForecaster.predict_batch failed: {e}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Retraining CLI
# ─────────────────────────────────────────────────────────────────────────────


def retrain(train_end: str = None):
    """
    Retrain the H2O model on extended data.
    Call quarterly: python core/h2o_vol_forecaster.py --retrain
    """
    import subprocess
    import sys

    train_script = Path(__file__).parent.parent / "scripts" / "h2o_full_train.py"
    if not train_script.exists():
        log.error(f"Retrain script not found: {train_script}")
        return
    log.info("Retraining H2O vol model...")
    result = subprocess.run([sys.executable, str(train_script)], capture_output=False, check=False)  # noqa: S603
    if result.returncode == 0:
        log.info("Retraining complete.")
    else:
        log.error("Retraining failed.")


if __name__ == "__main__":
    import sys

    if "--retrain" in sys.argv:
        retrain()
    else:
        # Quick smoke test
        print("Testing H2OVolForecaster...")
        fc = H2OVolForecaster.load()
        if fc is None:
            print("Model not loaded — run training first.")
        else:
            import yfinance as yf

            spy = yf.download(
                "SPY", start="2020-01-01", end="2026-03-21", auto_adjust=True, progress=False
            )
            r = np.log(spy["Close"] / spy["Close"].shift(1)).dropna().squeeze()
            vix = yf.download(
                "^VIX", start="2020-01-01", end="2026-03-21", auto_adjust=True, progress=False
            )["Close"].squeeze()
            pred = fc.predict("SPY", r, vix)
            ewma_var = 0.0
            lam = 0.94
            for rv in r.values[-252:]:
                ewma_var = lam * ewma_var + (1 - lam) * rv**2
            ewma_vol = float(np.sqrt(ewma_var * 252))
            print(f"SPY:  H2O forecast = {pred:.2%}  |  EWMA = {ewma_vol:.2%}")
            print("Smoke test passed ✓")
