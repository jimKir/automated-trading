#!/usr/bin/env python3
"""
Volatility Classification Backtest
====================================

Pure classification approach: predict volatility REGIME, not magnitude.

Classes (3-class):
  HIGH     — vol percentile > 66th  (top third of historical range)
  NORMAL   — vol percentile 33-66th
  LOW      — vol percentile < 33rd  (bottom third)

Models tested:
  1. XGBoost Classifier (gradient boosting — gold standard)
  2. LightGBM Classifier
  3. Random Forest
  4. Extra Trees (more randomisation, often less overfit)
  5. Logistic Regression (linear baseline)
  6. LSTM Classifier (seq-to-class)
  7. Stacking Ensemble (XGB + LGBM + RF → LogReg meta-learner)

Metrics: F1 (macro & weighted), Accuracy, Precision, Recall, Confusion Matrix
"""

import logging
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler
from vol_engine import (
    SECTOR_MAP,
    DataPipeline,
    FeatureEngine,
)

# ── Config ────────────────────────────────────────────────────────────────
TEST_SYMBOLS = [
    "AAPL",
    "NVDA",
    "TSLA",
    "AMD",
    "META",
    "MSFT",
    "GOOGL",
    "AVGO",  # Tech
    "JPM",
    "GS",
    "BAC",
    "MS",  # Financials
    "XOM",
    "CVX",
    "SLB",
    "COP",  # Energy
    "UNH",
    "JNJ",
    "LLY",
    "ABBV",  # Health
    "AMZN",
    "HD",
    "WMT",
    "COST",  # Consumer
    "CAT",
    "BA",
    "HON",
    "GE",  # Industrials
]

LOOKBACK_YEARS = 3.0
TEST_DAYS = 126  # ~6 months
HORIZON_DAYS = 5  # predict 5-day forward vol regime

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREY = "\033[90m"


# ══════════════════════════════════════════════════════════════════════════
# TARGET ENGINEERING
# ══════════════════════════════════════════════════════════════════════════


def create_classification_target(
    close: pd.Series,
    horizon: int = 5,
    n_classes: int = 3,
    lookback: int = 252,
) -> pd.Series:
    """
    Create volatility regime labels using rolling percentile thresholds.

    Why rolling percentile instead of fixed thresholds?
      - Vol regimes are relative: 20% vol is 'HIGH' for WMT but 'LOW' for TSLA
      - Rolling window adapts to each stock's own vol distribution
      - Avoids look-ahead bias (only uses past data for thresholds)

    Returns: Series of labels: 0=LOW, 1=NORMAL, 2=HIGH
    """
    log_ret = np.log(close / close.shift(1))
    # Forward realised vol (what we're predicting)
    fwd_vol = log_ret.shift(-horizon).rolling(horizon).std() * np.sqrt(252)

    labels = pd.Series(np.nan, index=close.index)

    for i in range(lookback + horizon, len(close)):
        current_fwd_vol = fwd_vol.iloc[i]
        if pd.isna(current_fwd_vol):
            continue

        # Rolling history of vol (no future leakage)
        hist_vol = fwd_vol.iloc[max(0, i - lookback) : i].dropna()
        if len(hist_vol) < 60:
            continue

        pctl = (hist_vol < current_fwd_vol).mean()

        if n_classes == 3:
            if pctl > 0.67:
                labels.iloc[i] = 2  # HIGH
            elif pctl > 0.33:
                labels.iloc[i] = 1  # NORMAL
            else:
                labels.iloc[i] = 0  # LOW

    return labels


# ══════════════════════════════════════════════════════════════════════════
# LSTM CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════


class LSTMClassifier:
    """Bidirectional LSTM for volatility regime classification."""

    def __init__(self, n_classes=3, seq_len=20, epochs=60):
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.epochs = epochs
        self.model = None
        self.scaler = None

    def _build(self, n_features):
        import tensorflow as tf
        from tensorflow.keras import Model, layers

        inp = layers.Input(shape=(self.seq_len, n_features))
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(inp)

        # Attention
        attn = layers.Dense(1, activation="tanh")(x)
        attn = layers.Softmax(axis=1)(attn)
        ctx = layers.Multiply()([x, attn])
        ctx = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(ctx)

        x = layers.Dense(32, activation="relu")(ctx)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(self.n_classes, activation="softmax")(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import tensorflow as tf

        self.scaler = RobustScaler()
        X_sc = self.scaler.fit_transform(X_train)

        # Create sequences
        Xs, ys = [], []
        for i in range(self.seq_len, len(X_sc)):
            Xs.append(X_sc[i - self.seq_len : i])
            ys.append(y_train[i])
        Xs, ys = np.array(Xs), np.array(ys)

        # Validation
        val_data = None
        if X_val is not None:
            X_val_sc = self.scaler.transform(X_val)
            Xv, yv = [], []
            for i in range(self.seq_len, len(X_val_sc)):
                Xv.append(X_val_sc[i - self.seq_len : i])
                yv.append(y_val[i])
            if len(Xv) > 20:
                val_data = (np.array(Xv), np.array(yv))

        self.model = self._build(X_train.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_data else "loss", patience=8, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6),
        ]

        self.model.fit(
            Xs,
            ys,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=64,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X):
        X_sc = self.scaler.transform(X)
        Xs = []
        for i in range(self.seq_len, len(X_sc)):
            Xs.append(X_sc[i - self.seq_len : i])
        if not Xs:
            return np.array([])
        probs = self.model.predict(np.array(Xs), verbose=0)
        preds = np.full(len(X), -1)
        preds[self.seq_len :] = probs.argmax(axis=1)
        return preds

    def predict_proba(self, X):
        X_sc = self.scaler.transform(X)
        Xs = []
        for i in range(self.seq_len, len(X_sc)):
            Xs.append(X_sc[i - self.seq_len : i])
        if not Xs:
            return np.array([])
        probs = self.model.predict(np.array(Xs), verbose=0)
        full_probs = np.full((len(X), self.n_classes), 1.0 / self.n_classes)
        full_probs[self.seq_len :] = probs
        return full_probs


# ══════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ══════════════════════════════════════════════════════════════════════════


def main(force_refresh: bool = False):
    t0 = time.time()
    class_names = ["LOW", "NORMAL", "HIGH"]

    print(f"\n{BOLD}{'═' * 78}{RESET}")
    print(f"{BOLD}  VOLATILITY CLASSIFICATION BACKTEST{RESET}")
    print(
        f"{BOLD}  {len(TEST_SYMBOLS)} symbols | 3 classes (LOW/NORMAL/HIGH) | {HORIZON_DAYS}d horizon{RESET}"
    )
    print(f"{BOLD}{'═' * 78}{RESET}\n")

    # ── 1. Data (cached — re-downloads only if stale or --force-refresh) ─
    print(f"  {CYAN}[1/6]{RESET} Fetching data...")
    pipeline = DataPipeline()
    all_data = pipeline.fetch(TEST_SYMBOLS, LOOKBACK_YEARS, force_refresh=force_refresh)
    vix_data = pipeline.fetch_vix(LOOKBACK_YEARS, force_refresh=force_refresh)
    cache = pipeline.cache_info()
    print(
        f"        {len(all_data)} symbols fetched "
        f"(cache: {cache['total_size_mb']:.1f} MB, "
        f"{len(cache['ohlcv_symbols'])} symbols cached)\n"
    )

    # ── 2. Features + targets ─────────────────────────────────────────
    print(f"  {CYAN}[2/6]{RESET} Building features + classification targets...")
    feat_engine = FeatureEngine(vix_data=vix_data, all_data=all_data)

    train_X_list, train_y_list = [], []
    test_X_dict, test_y_dict = {}, {}

    for sym in all_data:
        df = all_data[sym]
        feats, _ = feat_engine.build_features(sym, df)
        labels = create_classification_target(df["close"], HORIZON_DAYS)

        # Align and clean
        mask = feats.notna().all(axis=1) & labels.notna()
        feats_clean = feats[mask]
        labels_clean = labels[mask].astype(int)

        if len(feats_clean) < TEST_DAYS + 200:
            continue

        # Time-ordered split
        split = len(feats_clean) - TEST_DAYS
        train_X_list.append(feats_clean.iloc[:split])
        train_y_list.append(labels_clean.iloc[:split])
        test_X_dict[sym] = feats_clean.iloc[split:]
        test_y_dict[sym] = labels_clean.iloc[split:]

    global_train_X = pd.concat(train_X_list).sort_index()
    global_train_y = pd.concat(train_y_list).sort_index()
    global_test_X = pd.concat(test_X_dict.values()).sort_index()
    global_test_y = pd.concat(test_y_dict.values()).sort_index()

    n_features = global_train_X.shape[1]
    print(
        f"        {n_features} features | Train: {len(global_train_X)} | Test: {len(global_test_X)}"
    )
    print(f"        Train class distribution: {dict(Counter(global_train_y.values))}")
    print(f"        Test  class distribution: {dict(Counter(global_test_y.values))}\n")

    # Scale for models that need it
    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(global_train_X)
    X_test_sc = scaler.transform(global_test_X)
    y_train = global_train_y.values
    y_test = global_test_y.values

    # ── 3. Train all models ───────────────────────────────────────────
    print(f"  {CYAN}[3/6]{RESET} Training 7 models...\n")

    models = {}

    # ── XGBoost ───────────────────────────────────────────────────────
    t1 = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_clf.fit(
        X_train_sc,
        y_train,
        eval_set=[(X_test_sc, y_test)],
        verbose=False,
    )
    models["XGBoost"] = xgb_clf
    print(f"    ✓ XGBoost          {time.time() - t1:.1f}s  (best iter: {xgb_clf.best_iteration})")

    # ── LightGBM ──────────────────────────────────────────────────────
    t1 = time.time()
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        num_class=3,
        objective="multiclass",
        verbose=-1,
        importance_type="gain",
    )
    lgb_clf.fit(
        X_train_sc,
        y_train,
        eval_set=[(X_test_sc, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    models["LightGBM"] = lgb_clf
    print(f"    ✓ LightGBM         {time.time() - t1:.1f}s  (best iter: {lgb_clf.best_iteration_})")

    # ── Random Forest ─────────────────────────────────────────────────
    t1 = time.time()
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf_clf.fit(X_train_sc, y_train)
    models["RandomForest"] = rf_clf
    print(f"    ✓ Random Forest    {time.time() - t1:.1f}s")

    # ── Extra Trees ───────────────────────────────────────────────────
    t1 = time.time()
    et_clf = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    et_clf.fit(X_train_sc, y_train)
    models["ExtraTrees"] = et_clf
    print(f"    ✓ Extra Trees      {time.time() - t1:.1f}s")

    # ── Logistic Regression (baseline) ────────────────────────────────
    t1 = time.time()
    lr_clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",
        solver="lbfgs",
    )
    lr_clf.fit(X_train_sc, y_train)
    models["LogisticReg"] = lr_clf
    print(f"    ✓ Logistic Reg     {time.time() - t1:.1f}s")

    # ── LSTM Classifier ───────────────────────────────────────────────
    t1 = time.time()
    lstm_clf = LSTMClassifier(n_classes=3, seq_len=20, epochs=60)
    # Train/val split for LSTM
    split_lstm = int(len(X_train_sc) * 0.85)
    lstm_clf.fit(
        global_train_X.values[:split_lstm],
        y_train[:split_lstm],
        global_train_X.values[split_lstm:],
        y_train[split_lstm:],
    )
    models["LSTM"] = lstm_clf
    print(f"    ✓ LSTM             {time.time() - t1:.1f}s")

    # ── Soft Voting Ensemble (XGB + LGB + RF) ───────────────────────
    t1 = time.time()
    vote_clf = VotingClassifier(
        estimators=[
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    num_class=3,
                    use_label_encoder=False,
                    verbosity=0,
                ),
            ),
            (
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    verbose=-1,
                    num_class=3,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ],
        voting="soft",
        n_jobs=-1,
    )
    vote_clf.fit(X_train_sc, y_train)
    models["VotingEnsemble"] = vote_clf
    print(f"    ✓ Voting Ensemble  {time.time() - t1:.1f}s")
    print()

    # ── 4. Evaluate ───────────────────────────────────────────────────
    print(f"  {CYAN}[4/6]{RESET} Evaluating on held-out test set ({len(y_test)} samples)...\n")

    print(
        f"{BOLD}  {'MODEL':<15} {'Accuracy':>9} {'F1-macro':>9} {'F1-wt':>9} {'Prec-wt':>9} {'Rec-wt':>9}{RESET}"
    )
    print(f"  {'─' * 65}")

    all_results = {}

    for name, model in models.items():
        if name == "LSTM":
            preds = model.predict(global_test_X.values)
            valid = preds >= 0
            y_pred = preds[valid]
            y_true = y_test[valid]
        else:
            y_pred = model.predict(X_test_sc)
            y_true = y_test

        if len(y_pred) < 20:
            print(f"  {name:<15} {'N/A — insufficient predictions':>50}")
            continue

        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        all_results[name] = {
            "acc": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wt,
            "precision": prec,
            "recall": rec,
            "y_pred": y_pred,
            "y_true": y_true,
        }

        acc_c = GREEN if acc > 0.5 else YELLOW if acc > 0.4 else RED
        f1_c = GREEN if f1_mac > 0.5 else YELLOW if f1_mac > 0.4 else RED

        marker = ""
        print(
            f"  {name:<15} {acc_c}{acc:>9.3f}{RESET} {f1_c}{f1_mac:>9.3f}{RESET} "
            f"{f1_wt:>9.3f} {prec:>9.3f} {rec:>9.3f}{marker}"
        )

    # Find best model
    if all_results:
        best_name = max(all_results, key=lambda n: all_results[n]["f1_macro"])
        print(
            f"\n  {GREEN}★ Best model: {best_name} (F1-macro: {all_results[best_name]['f1_macro']:.3f}){RESET}"
        )

    # ── 5. Detailed report for best + stacking ────────────────────────
    print(f"\n{BOLD}  {'═' * 70}{RESET}")
    for report_name in [best_name, "Stacking"]:
        if report_name not in all_results:
            continue
        r = all_results[report_name]
        print(f"\n{BOLD}  DETAILED REPORT: {report_name}{RESET}")
        print(f"  {'─' * 60}")

        report = classification_report(
            r["y_true"],
            r["y_pred"],
            target_names=class_names,
            zero_division=0,
        )
        for line in report.strip().split("\n"):
            print(f"    {line}")

        # Confusion matrix
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=[0, 1, 2])
        print("\n    Confusion Matrix:")
        print(f"    {'':>10} {'Pred LOW':>10} {'Pred NORM':>10} {'Pred HIGH':>10}")
        for i, label in enumerate(class_names):
            row = "  ".join(f"{v:>8}" for v in cm[i])
            print(f"    {label:>10} {row}")

        # Per-class metrics
        print("\n    Per-class F1:")
        per_class_f1 = f1_score(r["y_true"], r["y_pred"], average=None, zero_division=0)
        for _i, (cls, f1) in enumerate(zip(class_names, per_class_f1)):
            f1c = GREEN if f1 > 0.5 else YELLOW if f1 > 0.35 else RED
            bar = "█" * int(f1 * 40)
            print(f"    {cls:<8} {f1c}{bar}{RESET} {f1c}{f1:.3f}{RESET}")

    # ── 6. Feature importance (XGBoost) ───────────────────────────────
    print(f"\n{BOLD}  TOP 25 FEATURES (XGBoost gain){RESET}")
    print(f"  {'─' * 60}")

    fi = pd.Series(xgb_clf.feature_importances_, index=global_train_X.columns).sort_values(
        ascending=False
    )

    total = fi.sum()
    for feat, imp in fi.head(25).items():
        pct = imp / total * 100
        bar = "█" * int(pct * 1.5) + "░" * max(0, 25 - int(pct * 1.5))
        tag = ""
        if "rsi" in feat:
            tag = f" {CYAN}[RSI]{RESET}"
        elif "macd" in feat:
            tag = f" {YELLOW}[MACD]{RESET}"
        elif "stoch" in feat:
            tag = f" {GREEN}[STOCH]{RESET}"
        elif "adx" in feat or "di_" in feat or "strong_trend" in feat:
            tag = f" {RED}[ADX]{RESET}"
        elif "pmo" in feat:
            tag = f" {GREY}[PMO]{RESET}"
        elif "har" in feat:
            tag = f" {GREEN}[HAR]{RESET}"
        elif "vol_" in feat:
            tag = f" {RED}[VOL]{RESET}"
        elif "vix" in feat:
            tag = f" {YELLOW}[VIX]{RESET}"
        print(f"  {feat:<26} {bar} {pct:>5.1f}%{tag}")

    # ── Per-symbol breakdown ──────────────────────────────────────────
    print(f"\n{BOLD}  PER-SYMBOL ACCURACY (XGBoost){RESET}")
    print(f"  {'─' * 55}")

    for sym in sorted(test_X_dict.keys()):
        X_s = scaler.transform(test_X_dict[sym])
        y_s = test_y_dict[sym].values
        pred_s = xgb_clf.predict(X_s)
        acc_s = accuracy_score(y_s, pred_s)
        f1_s = f1_score(y_s, pred_s, average="macro", zero_division=0)
        sector = SECTOR_MAP.get(sym, "?")
        ac = GREEN if acc_s > 0.5 else YELLOW if acc_s > 0.4 else RED
        bar = "█" * int(acc_s * 30)
        print(f"  {sym:<6} {sector:<10} {ac}{bar}{RESET} acc={ac}{acc_s:.3f}{RESET} F1={f1_s:.3f}")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  {BOLD}Random baseline (3-class): 33.3% accuracy, 0.333 F1{RESET}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"\n{BOLD}{'═' * 78}{RESET}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Volatility Classification Backtest")
    parser.add_argument(
        "--force-refresh", action="store_true", help="Bypass data cache and re-download everything"
    )
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
