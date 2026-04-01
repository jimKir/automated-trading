"""
h2o_meta_learner.py
===================
H2O AutoML meta-learner that non-linearly combines ALL signal sources.

Diagnostic context
------------------
Individual factors are ALL negative in isolation; the composite works only
through diversification.  A non-linear combiner (GBM / XGBoost / stacked
ensemble) can learn the interaction structure between signals and market
regimes that a linear IC-weighted composite cannot capture.

Target
------
10-day forward return (continuous regression), NOT classification.
This aligns with the observed IC peak at day 10.

Walk-forward regime
-------------------
Model is retrained monthly with an expanding window (minimum 500 rows).
A ``needs_retrain`` guard prevents unnecessary daily retraining.

EWMA fallback
-------------
If H2O is unavailable (not installed / initialisation fails), the class
transparently falls back to an IC-weighted EWMA blend of input signals,
exactly replicating the current system behaviour.

Feature set
-----------
Signal inputs        [-1, +1]:
    ts_momentum, mean_reversion, macd, rsi,
    cs_momentum, credit_regime, options_flow, earnings_nlp

Regime / overlay     [various]:
    trend_classifier [0.6, 1.5], pv_segment [-1, +1]

Market context       [raw/normalised]:
    vix_level, vix_5d_change, spy_ma200_distance,
    realized_vol_21d, vol_ratio_5_21

Config (YAML key ``meta_learner``)
------------------------------------
    enabled: true
    automl_seconds: 300
    min_train_rows: 500
    retrain_days: 30
    feature_importance_log: true
    blend_weight: 0.40        # 40 % meta-signal + 60 % direct composite
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# H2O is an optional heavy dependency — guard the import
_H2O_AVAILABLE = False
try:
    import h2o  # type: ignore
    from h2o.automl import H2OAutoML  # type: ignore
    _H2O_AVAILABLE = True
except ImportError:
    warnings.warn(
        "h2o package not found. H2OMetaLearner will use EWMA fallback. "
        "Install with: pip install h2o",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

SIGNAL_FEATURES: List[str] = [
    "ts_momentum",
    "mean_reversion",
    "macd",
    "rsi",
    "cs_momentum",
    "credit_regime",
    "options_flow",
    "earnings_nlp",
]

OVERLAY_FEATURES: List[str] = [
    "trend_classifier",
    "pv_segment",
]

MARKET_FEATURES: List[str] = [
    "vix_level",
    "vix_5d_change",
    "spy_ma200_distance",
    "realized_vol_21d",
    "vol_ratio_5_21",
]

ALL_FEATURES: List[str] = SIGNAL_FEATURES + OVERLAY_FEATURES + MARKET_FEATURES
TARGET_COL = "fwd_return_10d"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetaLearnerConfig:
    """Runtime configuration for H2OMetaLearner."""
    enabled: bool = True
    automl_seconds: int = 300
    min_train_rows: int = 500
    retrain_days: int = 30
    feature_importance_log: bool = True
    blend_weight: float = 0.40          # weight of meta-signal vs direct composite
    model_dir: str = "models/h2o_meta_model"
    h2o_max_mem: str = "4g"
    h2o_nthreads: int = -1             # -1 = all available cores
    # EWMA fallback weights (one per signal feature; uniform default)
    ewma_weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict) -> "MetaLearnerConfig":
        section = cfg.get("meta_learner", cfg)
        return cls(
            enabled=bool(section.get("enabled", True)),
            automl_seconds=int(section.get("automl_seconds", 300)),
            min_train_rows=int(section.get("min_train_rows", 500)),
            retrain_days=int(section.get("retrain_days", 30)),
            feature_importance_log=bool(section.get("feature_importance_log", True)),
            blend_weight=float(section.get("blend_weight", 0.40)),
            model_dir=str(section.get("model_dir", "models/h2o_meta_model")),
            h2o_max_mem=str(section.get("h2o_max_mem", "4g")),
            h2o_nthreads=int(section.get("h2o_nthreads", -1)),
            ewma_weights=dict(section.get("ewma_weights", {})),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class H2OMetaLearner:
    """H2O AutoML meta-learner that non-linearly combines signal sources.

    Workflow
    --------
    1. ``train(features_df, target_series, as_of_date)`` — trains/retrains the
       model on all data up to ``as_of_date``.
    2. ``predict(features_dict, as_of_date)`` — returns a signal in [-1, +1].
    3. ``needs_retrain(as_of_date)`` — True if > retrain_days since last train.
    4. ``save()`` / ``load()`` — persist model state to ``model_dir``.

    Parameters
    ----------
    config : MetaLearnerConfig | dict, optional
    workspace_root : str | Path, optional
        Root path for model persistence (``model_dir`` is relative to this).
    """

    def __init__(
        self,
        config: Optional[MetaLearnerConfig | dict] = None,
        workspace_root: str | Path = "/home/user/workspace",
    ) -> None:
        if config is None:
            self.cfg = MetaLearnerConfig()
        elif isinstance(config, dict):
            self.cfg = MetaLearnerConfig.from_dict(config)
        else:
            self.cfg = config

        self._workspace = Path(workspace_root)
        self._model_dir = self._workspace / self.cfg.model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._h2o_model = None           # trained H2OAutoML object
        self._last_train_date: Optional[pd.Timestamp] = None
        self._feature_importance: Optional[pd.DataFrame] = None
        self._h2o_initialised = False

        # Load persisted state if available
        self._load_metadata()

        logger.info(
            "H2OMetaLearner initialised | enabled=%s | h2o_available=%s | "
            "model_dir=%s | blend_weight=%.2f",
            self.cfg.enabled,
            _H2O_AVAILABLE,
            self._model_dir,
            self.cfg.blend_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        as_of_date: pd.Timestamp,
    ) -> None:
        """Fit (or refit) the meta-learner on data up to ``as_of_date``.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix with columns matching ALL_FEATURES, datetime-indexed.
        target_series : pd.Series
            10-day forward return, datetime-indexed.
        as_of_date : pd.Timestamp
            Training cutoff — only rows on or before this date are used.
        """
        if not self.cfg.enabled:
            logger.info("Meta-learner disabled; skipping training.")
            return

        # --- 1. Causal slice + align ---
        X, y = self._prepare_training_data(features_df, target_series, as_of_date)

        if len(X) < self.cfg.min_train_rows:
            logger.warning(
                "Insufficient training rows: %d < %d required. Skipping.",
                len(X),
                self.cfg.min_train_rows,
            )
            return

        logger.info(
            "Training meta-learner on %d rows up to %s",
            len(X),
            as_of_date.date(),
        )

        if _H2O_AVAILABLE:
            self._train_h2o(X, y, as_of_date)
        else:
            logger.info("H2O unavailable — using EWMA fallback (no model to fit).")

        self._last_train_date = as_of_date
        self._save_metadata()

    def predict(
        self,
        features_dict: Dict[str, float],
        as_of_date: pd.Timestamp,
    ) -> float:
        """Generate a meta-signal prediction in [-1, +1].

        If the H2O model is available and trained, it blends the model output
        with a direct EWMA composite (cfg.blend_weight controls the split).
        Otherwise it returns the EWMA composite signal alone.

        Parameters
        ----------
        features_dict : dict[str, float]
            Current feature values keyed by feature name.
        as_of_date : pd.Timestamp
            Prediction date (for logging/traceability only; prediction is
            always synchronous given the features provided).

        Returns
        -------
        float
            Meta-signal in [-1, +1].
        """
        if not self.cfg.enabled:
            return self._ewma_signal(features_dict)

        ewma_sig = self._ewma_signal(features_dict)

        if not _H2O_AVAILABLE or self._h2o_model is None:
            logger.debug(
                "as_of=%s: EWMA fallback signal=%.4f", as_of_date.date(), ewma_sig
            )
            return ewma_sig

        try:
            h2o_sig = self._predict_h2o(features_dict)
            blended = (
                self.cfg.blend_weight * h2o_sig
                + (1.0 - self.cfg.blend_weight) * ewma_sig
            )
            blended = float(np.clip(blended, -1.0, 1.0))
            logger.debug(
                "as_of=%s: h2o=%.4f ewma=%.4f blended=%.4f",
                as_of_date.date(),
                h2o_sig,
                ewma_sig,
                blended,
            )
            return blended

        except Exception as exc:
            logger.warning(
                "H2O prediction failed (%s); falling back to EWMA.", exc
            )
            return ewma_sig

    def needs_retrain(self, as_of_date: pd.Timestamp) -> bool:
        """Return True if the model is stale (> retrain_days since last fit).

        Always returns True if the model has never been trained.
        """
        if self._last_train_date is None:
            return True
        delta = (as_of_date - self._last_train_date).days
        return delta >= self.cfg.retrain_days

    def save(self) -> None:
        """Persist the H2O model + metadata to ``model_dir``."""
        if not _H2O_AVAILABLE or self._h2o_model is None:
            logger.debug("Nothing to save (no H2O model).")
            self._save_metadata()
            return

        try:
            save_path = str(self._model_dir)
            h2o.save_model(
                model=self._h2o_model.leader,
                path=save_path,
                force=True,
            )
            logger.info("H2O model saved to %s", save_path)
        except Exception as exc:
            logger.error("Failed to save H2O model: %s", exc)

        self._save_metadata()

    def load(self) -> bool:
        """Load a previously saved H2O model.

        Returns True on success, False otherwise.
        """
        self._load_metadata()

        if not _H2O_AVAILABLE:
            logger.info("H2O unavailable; model cannot be loaded.")
            return False

        model_files = list(self._model_dir.glob("GBM_*")) + \
                      list(self._model_dir.glob("StackedEnsemble_*")) + \
                      list(self._model_dir.glob("XGBoost_*"))

        if not model_files:
            logger.info("No saved H2O model found in %s", self._model_dir)
            return False

        try:
            self._ensure_h2o()
            leader_path = str(sorted(model_files)[-1])
            loaded = h2o.load_model(leader_path)
            # Wrap in a simple namespace so predict works uniformly
            self._h2o_model = _ModelWrapper(loaded)
            logger.info("H2O model loaded from %s", leader_path)
            return True
        except Exception as exc:
            logger.error("Failed to load H2O model: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importance from the trained model, if available."""
        return self._feature_importance

    # ------------------------------------------------------------------
    # H2O internals
    # ------------------------------------------------------------------

    def _ensure_h2o(self) -> None:
        """Initialise H2O cluster if not already running."""
        if self._h2o_initialised:
            return
        if not _H2O_AVAILABLE:
            raise RuntimeError("h2o package is not installed.")
        try:
            h2o.init(
                max_mem_size=self.cfg.h2o_max_mem,
                nthreads=self.cfg.h2o_nthreads,
                verbose=False,
            )
            self._h2o_initialised = True
            logger.info("H2O cluster initialised.")
        except Exception as exc:
            raise RuntimeError(f"H2O init failed: {exc}") from exc

    def _train_h2o(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        as_of_date: pd.Timestamp,
    ) -> None:
        """Fit H2OAutoML on the training set."""
        self._ensure_h2o()

        train_df = X.copy()
        train_df[TARGET_COL] = y.values

        h2o_frame = h2o.H2OFrame(train_df)
        h2o_frame[TARGET_COL] = h2o_frame[TARGET_COL].asnumeric()

        aml = H2OAutoML(
            max_runtime_secs=self.cfg.automl_seconds,
            sort_metric="RMSE",
            seed=42,
            include_algos=["GBM", "XGBoost", "StackedEnsemble"],
            verbosity="warn",
        )
        aml.train(x=ALL_FEATURES, y=TARGET_COL, training_frame=h2o_frame)

        self._h2o_model = aml

        # Log feature importance
        if self.cfg.feature_importance_log and aml.leader is not None:
            try:
                varimp = aml.leader.varimp(use_pandas=True)
                self._feature_importance = varimp
                logger.info(
                    "Feature importance (top 5):\n%s",
                    varimp.head(5).to_string(index=False),
                )
                # Persist to disk
                fi_path = self._model_dir / f"feature_importance_{as_of_date.date()}.csv"
                varimp.to_csv(fi_path, index=False)
            except Exception as exc:
                logger.warning("Could not extract feature importance: %s", exc)

        logger.info(
            "H2O AutoML training complete | leader=%s | RMSE=%.6f",
            aml.leader.model_id if aml.leader else "None",
            aml.leader.rmse() if aml.leader else float("nan"),
        )

    def _predict_h2o(self, features_dict: Dict[str, float]) -> float:
        """Run H2O model prediction for a single row."""
        # Build a one-row DataFrame, filling missing features with 0
        row = {feat: features_dict.get(feat, 0.0) for feat in ALL_FEATURES}
        df = pd.DataFrame([row])
        h2o_frame = h2o.H2OFrame(df)

        leader = (
            self._h2o_model.leader
            if hasattr(self._h2o_model, "leader")
            else self._h2o_model
        )
        pred_frame = leader.predict(h2o_frame)
        raw_pred = float(pred_frame.as_data_frame()["predict"].iloc[0])

        # Normalise raw return prediction → signal in [-1, +1]
        # using a soft tanh squash (scale: 5 % daily return → ±0.5 signal)
        meta_signal = float(np.tanh(raw_pred / 0.05))
        return float(np.clip(meta_signal, -1.0, 1.0))

    # ------------------------------------------------------------------
    # EWMA fallback
    # ------------------------------------------------------------------

    def _ewma_signal(self, features_dict: Dict[str, float]) -> float:
        """IC-weighted EWMA blend of signal features (current system proxy)."""
        weights = self.cfg.ewma_weights

        if not weights:
            # Uniform weights across signal features
            signals = [
                features_dict.get(f, 0.0)
                for f in SIGNAL_FEATURES
                if f in features_dict
            ]
            if not signals:
                return 0.0
            return float(np.clip(np.mean(signals), -1.0, 1.0))

        # Weighted average
        total_w = 0.0
        total_sig = 0.0
        for feat, w in weights.items():
            val = features_dict.get(feat, 0.0)
            total_sig += w * val
            total_w += abs(w)

        if total_w < 1e-9:
            return 0.0
        return float(np.clip(total_sig / total_w, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_training_data(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        as_of_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Slice, align, and validate training data (strictly causal)."""
        # Causal slice
        X = features_df[features_df.index <= as_of_date].copy()
        y = target_series[target_series.index <= as_of_date].copy()

        # Align
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Ensure all features present (fill missing with 0)
        for feat in ALL_FEATURES:
            if feat not in X.columns:
                logger.warning("Feature '%s' missing from training data — filling 0.", feat)
                X[feat] = 0.0

        X = X[ALL_FEATURES]

        # Drop rows with NaN in target
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        # Forward-fill then zero-fill feature NaNs
        X = X.ffill().fillna(0.0)

        return X, y

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def _metadata_path(self) -> Path:
        return self._model_dir / "metadata.json"

    def _save_metadata(self) -> None:
        meta = {
            "last_train_date": (
                self._last_train_date.isoformat()
                if self._last_train_date is not None
                else None
            ),
            "blend_weight": self.cfg.blend_weight,
            "retrain_days": self.cfg.retrain_days,
        }
        try:
            with open(self._metadata_path(), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save metadata: %s", exc)

    def _load_metadata(self) -> None:
        path = self._metadata_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                meta = json.load(f)
            raw_date = meta.get("last_train_date")
            if raw_date:
                self._last_train_date = pd.Timestamp(raw_date)
            logger.debug("Metadata loaded: last_train=%s", self._last_train_date)
        except Exception as exc:
            logger.warning("Could not load metadata: %s", exc)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"H2OMetaLearner("
            f"enabled={self.cfg.enabled}, "
            f"h2o_available={_H2O_AVAILABLE}, "
            f"last_train={self._last_train_date}, "
            f"blend_weight={self.cfg.blend_weight})"
        )


# ---------------------------------------------------------------------------
# Internal wrapper so a bare loaded model looks like H2OAutoML
# ---------------------------------------------------------------------------

class _ModelWrapper:
    """Thin wrapper to make a loaded H2O model behave like aml.leader."""

    def __init__(self, model) -> None:
        self._model = model

    @property
    def leader(self):
        return self._model


# ---------------------------------------------------------------------------
# Config YAML template (printed when module run directly)
# ---------------------------------------------------------------------------

_YAML_TEMPLATE = """
meta_learner:
  enabled: true
  automl_seconds: 300
  min_train_rows: 500
  retrain_days: 30
  feature_importance_log: true
  blend_weight: 0.40        # 40% meta-learner + 60% direct composite
  model_dir: models/h2o_meta_model
  h2o_max_mem: "4g"
  h2o_nthreads: -1
  ewma_weights:             # optional manual IC weights (leave empty for uniform)
    ts_momentum: 1.0
    mean_reversion: 0.8
    macd: 0.6
    rsi: 0.5
    cs_momentum: 1.2
    credit_regime: 0.7
    options_flow: 0.9
    earnings_nlp: 0.8
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("H2OMetaLearner — default config YAML:")
    print(_YAML_TEMPLATE)

    print(f"ALL_FEATURES ({len(ALL_FEATURES)}):", ALL_FEATURES)
    print(f"H2O available: {_H2O_AVAILABLE}")

    # Quick smoke-test with fake data (no H2O needed)
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.bdate_range("2022-01-01", periods=n)
    feat_data = {f: rng.uniform(-1, 1, n) for f in SIGNAL_FEATURES}
    feat_data.update({f: rng.uniform(0.6, 1.5, n) for f in OVERLAY_FEATURES})
    feat_data.update({f: rng.normal(0, 1, n) for f in MARKET_FEATURES})
    features_df = pd.DataFrame(feat_data, index=idx)
    target_series = pd.Series(rng.normal(0, 0.01, n), index=idx)

    learner = H2OMetaLearner()
    as_of = pd.Timestamp("2024-01-01")
    learner.train(features_df, target_series, as_of_date=as_of)

    sample = {f: float(rng.uniform(-1, 1)) for f in ALL_FEATURES}
    sig = learner.predict(sample, as_of_date=as_of)
    print(f"\nSample EWMA meta-signal: {sig:.4f}")
    print(f"needs_retrain(as_of+31d): {learner.needs_retrain(as_of + pd.Timedelta(days=31))}")
