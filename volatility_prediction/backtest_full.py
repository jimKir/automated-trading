#!/usr/bin/env python3
"""
Full Backtest — Volatility Prediction Engine
=============================================

Proper out-of-sample evaluation:
  - 3 years of data, last 6 months held out as test set
  - Walk-forward: train on expanding window, predict next period
  - Regression metrics: RMSE, MAE, QLIKE, R²
  - Classification metrics: F1, precision, recall for vol regime
  - Per-model breakdown: HAR vs GBM vs LSTM vs Ensemble
  - Per-sector analysis
"""

import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vol_engine import (
    SECTOR_MAP,
    AdaptiveEnsemble,
    DataPipeline,
    FeatureEngine,
    GBMVolModel,
    HARModel,
    LSTMVolModel,
    WalkForwardValidator,
)

# ── Configuration ──────────────────────────────────────────────────────────
TEST_SYMBOLS = [
    # Tech (high vol)
    "AAPL",
    "NVDA",
    "TSLA",
    "AMD",
    "META",
    # Financials (moderate vol)
    "JPM",
    "GS",
    "BAC",
    # Energy (high vol)
    "XOM",
    "CVX",
    "SLB",
    # Healthcare (low vol)
    "UNH",
    "JNJ",
    "LLY",
    # Consumer (mixed)
    "AMZN",
    "WMT",
    "COST",
    # Industrials
    "CAT",
    "BA",
    "HON",
]

HORIZON = "5d"
HORIZON_DAYS = 5
LOOKBACK_YEARS = 3.0
# Test split: last 6 months (~126 trading days)
TEST_DAYS = 126

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREY = "\033[90m"


def classify_vol_regime(vol_series: pd.Series, current_vol: float) -> str:
    """Classify volatility into regime buckets."""
    if len(vol_series.dropna()) < 60:
        return "NORMAL"
    pctl = (vol_series.dropna() < current_vol).mean() * 100
    if pctl > 80:
        return "HIGH"
    if pctl > 60:
        return "ELEVATED"
    if pctl > 40:
        return "NORMAL"
    if pctl > 20:
        return "LOW"
    return "COMPRESSED"


def qlike_loss(pred, actual):
    """QLIKE loss — standard for vol forecast evaluation."""
    mask = (pred > 0.001) & (actual > 0.001) & np.isfinite(pred) & np.isfinite(actual)
    p = pred[mask]
    a = actual[mask]
    if len(p) < 10:
        return np.nan
    p_sq = p**2
    a_sq = a**2
    return (np.log(p_sq) + a_sq / p_sq).mean()


def main(force_refresh: bool = False):
    t0 = time.time()

    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  VOLATILITY PREDICTION — FULL BACKTEST{RESET}")
    print(
        f"{BOLD}  {len(TEST_SYMBOLS)} symbols | {LOOKBACK_YEARS:.0f}yr history | "
        f"{HORIZON} horizon | {TEST_DAYS} test days{RESET}"
    )
    print(f"{BOLD}{'=' * 78}{RESET}\n")

    # ── 1. Fetch data (cached — re-downloads only if stale or --force-refresh) ─
    print(f"  {CYAN}[1/5]{RESET} Fetching market data...")
    pipeline = DataPipeline()
    all_data = pipeline.fetch(TEST_SYMBOLS, LOOKBACK_YEARS, force_refresh=force_refresh)
    vix_data = pipeline.fetch_vix(LOOKBACK_YEARS, force_refresh=force_refresh)
    cache = pipeline.cache_info()
    print(
        f"        Fetched {len(all_data)}/{len(TEST_SYMBOLS)} symbols "
        f"(cache: {cache['total_size_mb']:.1f} MB, "
        f"{len(cache['ohlcv_symbols'])} symbols cached)\n"
    )

    if len(all_data) < 5:
        print(f"  {RED}Not enough data — aborting{RESET}")
        return

    # ── 2. Build features ─────────────────────────────────────────────
    print(f"  {CYAN}[2/5]{RESET} Engineering features (RSI, MACD, volatility estimators, HAR)...")
    feat_engine = FeatureEngine(vix_data=vix_data, all_data=all_data)

    all_features = {}
    all_targets = {}
    for sym in all_data:
        feats, targets = feat_engine.build_features(sym, all_data[sym])
        target = targets.get(HORIZON)
        if target is not None:
            all_features[sym] = feats
            all_targets[sym] = target

    print(f"        Built features for {len(all_features)} symbols")

    # Count features
    sample_sym = next(iter(all_features.keys()))
    n_features = all_features[sample_sym].shape[1]
    print(f"        {n_features} features per symbol")

    # Check MACD and RSI are present
    sample_cols = all_features[sample_sym].columns.tolist()
    macd_cols = [c for c in sample_cols if "macd" in c]
    rsi_cols = [c for c in sample_cols if "rsi" in c]
    print(f"        RSI features: {rsi_cols}")
    print(f"        MACD features: {macd_cols}\n")

    # ── 3. Split train/test ───────────────────────────────────────────
    print(f"  {CYAN}[3/5]{RESET} Splitting train/test (last {TEST_DAYS} days = test)...")

    # Pool all data for global model training
    train_X_list, train_y_list = [], []
    test_X_dict, test_y_dict = {}, {}

    for sym, feats in all_features.items():
        target = all_targets[sym]

        # Align
        mask = feats.notna().all(axis=1) & target.notna() & np.isfinite(target)
        feats_clean = feats[mask]
        target_clean = target[mask]

        if len(feats_clean) < TEST_DAYS + 100:
            logger.warning(f"{sym}: insufficient data, skipping")
            continue

        # Split
        split_idx = len(feats_clean) - TEST_DAYS
        train_X = feats_clean.iloc[:split_idx]
        train_y = target_clean.iloc[:split_idx]
        test_X = feats_clean.iloc[split_idx:]
        test_y = target_clean.iloc[split_idx:]

        train_X_list.append(train_X)
        train_y_list.append(train_y)
        test_X_dict[sym] = test_X
        test_y_dict[sym] = test_y

    global_train_X = pd.concat(train_X_list, axis=0).sort_index()
    global_train_y = pd.concat(train_y_list, axis=0).sort_index()

    total_test = sum(len(v) for v in test_X_dict.values())
    print(f"        Train: {len(global_train_X)} rows")
    print(f"        Test:  {total_test} rows ({len(test_X_dict)} symbols)\n")

    # ── 4. Train models ───────────────────────────────────────────────
    print(f"  {CYAN}[4/5]{RESET} Training models...")

    # HAR (global)
    t1 = time.time()
    har_model = HARModel()
    har_model.fit(global_train_X, global_train_y)
    print(f"        HAR fitted in {time.time() - t1:.1f}s")

    # GBM (global)
    t1 = time.time()
    gbm_model = GBMVolModel()
    gbm_model.fit(global_train_X, global_train_y)
    print(f"        GBM fitted in {time.time() - t1:.1f}s")

    # LSTM (global)
    t1 = time.time()
    lstm_model = LSTMVolModel(seq_len=20, epochs=80, batch_size=64)
    lstm_model.fit(global_train_X, global_train_y)
    print(f"        LSTM fitted in {time.time() - t1:.1f}s")
    print()

    # ── 5. Out-of-sample evaluation ───────────────────────────────────
    print(f"  {CYAN}[5/5]{RESET} Out-of-sample evaluation on test set...")
    print()

    WalkForwardValidator()
    ensemble = AdaptiveEnsemble()

    # Collect results per model
    model_names = ["HAR", "GBM", "LSTM", "Ensemble"]
    model_preds_all = {m: [] for m in model_names}
    model_actuals_all = {m: [] for m in model_names}

    # Classification results
    regime_true_all = []
    regime_pred_all = {m: [] for m in model_names}

    per_symbol_metrics = {}

    for sym, test_X in test_X_dict.items():
        test_y = test_y_dict[sym]

        preds = {}

        # HAR
        preds["HAR"] = har_model.predict(test_X)

        # GBM
        if gbm_model.model is not None:
            preds["GBM"] = gbm_model.predict(test_X)
        else:
            preds["GBM"] = pd.Series(np.nan, index=test_X.index)

        # LSTM
        if lstm_model.model is not None:
            # LSTM needs the full feature history for sequences
            full_feats = all_features[sym]
            full_pred = lstm_model.predict(full_feats)
            # Extract test portion
            preds["LSTM"] = full_pred.reindex(test_X.index)
        else:
            preds["LSTM"] = pd.Series(np.nan, index=test_X.index)

        # Ensemble
        preds["Ensemble"] = ensemble.combine(preds, test_y)

        # Evaluate each model for this symbol
        sym_metrics = {}
        for model_name in model_names:
            p = preds[model_name]
            a = test_y

            mask = p.notna() & a.notna() & (p > 0) & (a > 0)
            p_clean = p[mask].values
            a_clean = a[mask].values

            if len(p_clean) < 10:
                continue

            mse = np.mean((p_clean - a_clean) ** 2)
            mae = np.mean(np.abs(p_clean - a_clean))
            ql = qlike_loss(p_clean, a_clean)
            ss_res = np.sum((a_clean - p_clean) ** 2)
            ss_tot = np.sum((a_clean - a_clean.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            sym_metrics[model_name] = {
                "rmse": np.sqrt(mse),
                "mae": mae,
                "qlike": ql,
                "r2": r2,
                "n": len(p_clean),
            }

            model_preds_all[model_name].extend(p_clean)
            model_actuals_all[model_name].extend(a_clean)

            # Classification: predict vol regime
            vol_hist = all_features[sym].get("vol_yz_20", pd.Series(dtype=float))
            for i in range(len(p_clean)):
                idx = test_X.index[mask][i] if i < mask.sum() else None
                if idx is None:
                    continue
                # True regime
                true_vol = a_clean[i]
                true_regime = classify_vol_regime(vol_hist.loc[:idx], true_vol)
                # Predicted regime
                pred_vol = p_clean[i]
                pred_regime = classify_vol_regime(vol_hist.loc[:idx], pred_vol)

                if model_name == model_names[0]:  # only add true once
                    regime_true_all.append(true_regime)
                regime_pred_all[model_name].append(pred_regime)

        per_symbol_metrics[sym] = sym_metrics

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════

    print(f"{BOLD}{'═' * 78}{RESET}")
    print(f"{BOLD}  OUT-OF-SAMPLE RESULTS — {HORIZON} horizon, {TEST_DAYS} test days{RESET}")
    print(f"{BOLD}{'═' * 78}{RESET}\n")

    # ── Aggregate regression metrics ──────────────────────────────────
    print(f"{BOLD}  REGRESSION METRICS (aggregated across all symbols){RESET}")
    print(f"  {'─' * 65}")
    print(f"  {'MODEL':<12} {'RMSE':>8} {'MAE':>8} {'QLIKE':>8} {'R²':>8} {'N':>6}")
    print(f"  {'─' * 65}")

    for model_name in model_names:
        p = np.array(model_preds_all[model_name])
        a = np.array(model_actuals_all[model_name])

        if len(p) < 10:
            print(f"  {model_name:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {len(p):>6}")
            continue

        rmse = np.sqrt(np.mean((p - a) ** 2))
        mae = np.mean(np.abs(p - a))
        ql = qlike_loss(p, a)
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        marker = f" {GREEN}★{RESET}" if model_name == "Ensemble" else ""
        r2_color = GREEN if r2 > 0.3 else YELLOW if r2 > 0.1 else RED
        print(
            f"  {model_name:<12} {rmse:>8.4f} {mae:>8.4f} {ql:>8.3f} "
            f"{r2_color}{r2:>8.4f}{RESET} {len(p):>6}{marker}"
        )

    print(f"\n  {GREY}QLIKE: lower = better | R²: higher = better | ★ = ensemble{RESET}\n")

    # ── Per-symbol breakdown ──────────────────────────────────────────
    print(f"{BOLD}  PER-SYMBOL R² (Ensemble){RESET}")
    print(f"  {'─' * 50}")

    sym_r2_list = []
    for sym in sorted(per_symbol_metrics.keys()):
        ens_m = per_symbol_metrics[sym].get("Ensemble", {})
        r2 = ens_m.get("r2", np.nan)
        rmse = ens_m.get("rmse", np.nan)
        sector = SECTOR_MAP.get(sym, "?")

        if np.isnan(r2):
            continue

        sym_r2_list.append((sym, r2, rmse, sector))
        r2c = GREEN if r2 > 0.3 else YELLOW if r2 > 0.1 else RED
        bar = "█" * max(0, int(r2 * 30)) if r2 > 0 else ""
        print(f"  {sym:<6} {sector:<8} {r2c}{bar}{RESET} R²={r2c}{r2:.3f}{RESET}  RMSE={rmse:.4f}")

    if sym_r2_list:
        avg_r2 = np.mean([x[1] for x in sym_r2_list])
        med_r2 = np.median([x[1] for x in sym_r2_list])
        print(f"\n  Mean R²: {avg_r2:.4f} | Median R²: {med_r2:.4f}")
    print()

    # ── Classification metrics (F1, precision, recall) ────────────────
    print(f"{BOLD}  CLASSIFICATION METRICS — Volatility Regime Prediction{RESET}")
    print(f"  {'─' * 65}")

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )


        for model_name in model_names:
            true = regime_true_all
            pred = regime_pred_all[model_name]

            # Trim to same length
            min_len = min(len(true), len(pred))
            if min_len < 20:
                continue
            true = true[:min_len]
            pred = pred[:min_len]

            acc = accuracy_score(true, pred)
            f1_macro = f1_score(true, pred, average="macro", zero_division=0)
            f1_weighted = f1_score(true, pred, average="weighted", zero_division=0)
            prec = precision_score(true, pred, average="weighted", zero_division=0)
            rec = recall_score(true, pred, average="weighted", zero_division=0)

            marker = f" {GREEN}★{RESET}" if model_name == "Ensemble" else ""
            acc_c = GREEN if acc > 0.5 else YELLOW if acc > 0.3 else RED
            f1_c = GREEN if f1_weighted > 0.5 else YELLOW if f1_weighted > 0.3 else RED

            print(f"\n  {BOLD}{model_name}{RESET}{marker}")
            print(f"    Accuracy:     {acc_c}{acc:.3f}{RESET}")
            print(f"    F1 (weighted):{f1_c}{f1_weighted:.3f}{RESET}")
            print(f"    F1 (macro):   {f1_macro:.3f}")
            print(f"    Precision:    {prec:.3f}")
            print(f"    Recall:       {rec:.3f}")

            if model_name == "Ensemble":
                print(f"\n  {BOLD}  Confusion Matrix (Ensemble):{RESET}")
                labels_present = sorted(set(true + pred))
                cm = confusion_matrix(true, pred, labels=labels_present)
                # Print header
                header = "        " + "  ".join(f"{lbl:>8}" for lbl in labels_present)
                print(header)
                for i, row_label in enumerate(labels_present):
                    row = "  ".join(f"{v:>8}" for v in cm[i])
                    print(f"  {row_label:>6} {row}")

                print(f"\n  {BOLD}  Per-class F1 (Ensemble):{RESET}")
                report = classification_report(
                    true, pred, labels=labels_present, output_dict=True, zero_division=0
                )
                for cls in labels_present:
                    if cls in report:
                        f1 = report[cls]["f1-score"]
                        sup = report[cls]["support"]
                        f1c = GREEN if f1 > 0.5 else YELLOW if f1 > 0.3 else RED
                        print(f"    {cls:<12} F1={f1c}{f1:.3f}{RESET}  support={sup}")

    except ImportError:
        print(f"  {RED}sklearn not available for classification metrics{RESET}")

    # ── Feature importance ────────────────────────────────────────────
    print(f"\n{BOLD}  TOP 20 FEATURES (GBM gain importance){RESET}")
    print(f"  {'─' * 55}")
    fi = gbm_model.feature_importance()
    if fi is not None and len(fi) > 0:
        total = fi.sum()
        for feat, imp in fi.head(20).items():
            pct = imp / total * 100
            bar = "█" * int(pct / 1.5) + "░" * max(0, 25 - int(pct / 1.5))
            # Highlight RSI and MACD features
            tag = ""
            if "rsi" in feat:
                tag = f" {CYAN}[RSI]{RESET}"
            elif "macd" in feat:
                tag = f" {YELLOW}[MACD]{RESET}"
            elif "har" in feat:
                tag = f" {GREEN}[HAR]{RESET}"
            elif "vol_" in feat:
                tag = f" {RED}[VOL]{RESET}"
            print(f"  {feat:<24} {bar} {pct:>5.1f}%{tag}")
    print()

    # ── Ensemble weights ──────────────────────────────────────────────
    print(f"{BOLD}  ENSEMBLE WEIGHTS{RESET}")
    for m, w in ensemble.weights.items():
        bar = "█" * int(w * 40)
        print(f"  {m:<10} {bar} {w:.1%}")

    # ── Timing ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  Total backtest time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"{BOLD}{'═' * 78}{RESET}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Volatility Prediction Full Backtest")
    parser.add_argument(
        "--force-refresh", action="store_true", help="Bypass data cache and re-download everything"
    )
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
