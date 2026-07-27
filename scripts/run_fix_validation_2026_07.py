#!/usr/bin/env python3
"""
Four-way validation of the 2026-07 remediation.

Compares, over the same window and the same data:

  new_config  — 63d selector, top_n 9, heat 0.95, recalibrated anomaly layer,
                optimizer heat restoration
  old_config  — what was actually deployed: 126d selector, top_n 20, heat 0.75
  spy         — buy and hold
  sixty_forty — 60/40 SPY/AGG rebalanced monthly (the probation benchmark)

The old-config arm reverts only the four parameters; the code fixes (anomaly
calibration, heat restoration) are not separable from the branch, so the arm
neutralises them through config instead — anomaly_layer disabled and heat
restoration monkeypatched off — to reproduce deployed behaviour.

Writes results/fix_validation_2026_07.{json,png}.
"""

from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import BacktestEngine
from core.optimizer import PortfolioOptimizer
from core.portfolio import Portfolio
from data.data_store import get_store

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "results" / "fix_validation_2026_07.json"
OUT_PNG = ROOT / "results" / "fix_validation_2026_07.png"

START = "2026-01-02"
END = "2026-07-23"
# Selection and signal windows need warm-up; the engine is fed history from here but
# only scores and trades inside [START, END].
DATA_START = "2024-06-01"

TRADING_DAYS = 252
RISK_FREE = 0.04


def load_data() -> dict[str, pd.DataFrame]:
    store = get_store()
    syms = store.list_available()
    data = {}
    for sym in syms:
        df = store.load(sym, start_date=DATA_START, end_date=END)
        if df is None or len(df) < 200:
            continue
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[sym] = df
    return data


def as_deployed(config: dict) -> dict:
    """Revert the four parameters that were live before this branch."""
    cfg = copy.deepcopy(config)
    cfg["dynamic_universe"]["momentum_window"] = 126
    cfg["dynamic_universe"]["top_n"] = 20
    cfg["capital"]["max_portfolio_heat"] = 0.75
    cfg.setdefault("risk_limits", {})["max_portfolio_heat"] = 0.75
    return cfg


def metrics_from_curve(curve: pd.Series) -> dict:
    curve = curve.dropna()
    rets = curve.pct_change().dropna()
    total = (curve.iloc[-1] / curve.iloc[0] - 1) * 100
    cummax = curve.cummax()
    mdd = ((curve - cummax) / cummax).min() * 100
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (rets.mean() * TRADING_DAYS - RISK_FREE) / vol if vol > 0 else 0.0
    return {
        "total_return_pct": round(float(total), 2),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(mdd), 2),
        "ann_volatility_pct": round(float(vol * 100), 2),
    }


def sixty_forty(spy: pd.Series, agg: pd.Series) -> pd.Series:
    """60/40 SPY/AGG rebalanced at each month boundary."""
    px = pd.DataFrame({"SPY": spy, "AGG": agg}).dropna()
    rets = px.pct_change().fillna(0.0)
    months = px.index.to_period("M").to_numpy()
    w = {"SPY": 0.60, "AGG": 0.40}
    vals = dict(w)
    out = []
    for i in range(len(rets)):
        if i > 0:
            if months[i] != months[i - 1]:
                nav = sum(vals.values())
                vals = {c: nav * w[c] for c in w}
            vals = {c: vals[c] * (1 + rets[c].iloc[i]) for c in w}
        out.append(sum(vals.values()))
    return pd.Series(out, index=px.index)


def run_arm(config: dict, data: dict, label: str, restore_heat: bool) -> dict:
    cfg = copy.deepcopy(config)
    cfg["backtest"]["start_date"] = START
    cfg["backtest"]["end_date"] = END

    max_pos = cfg["risk"]["max_position_pct"]

    # Count how many names actually receive weight each rebalance. Gross exposure is
    # bounded by funded_names * max_position_pct, and that bound turns out to bind long
    # before the heat budget does.
    funded: list[int] = []
    original_ctw = Portfolio.compute_target_weights

    def instrumented(self, signals, **kw):
        w = original_ctw(self, signals, **kw)
        funded.append(sum(1 for v in w.values() if abs(v) > 1e-9))
        return w

    original_restore = PortfolioOptimizer._restore_target_heat
    if not restore_heat:
        # Reproduce the deployed optimizer, where nothing redistributed the shortfall
        # the cap steps created.
        PortfolioOptimizer._restore_target_heat = lambda self, w, *a, **k: w
    Portfolio.compute_target_weights = instrumented
    try:
        result = BacktestEngine(cfg).run(data, benchmark_data=data.get("SPY"), run_label=label)
    finally:
        PortfolioOptimizer._restore_target_heat = original_restore
        Portfolio.compute_target_weights = original_ctw

    curve = result["equity_curve"]
    m = metrics_from_curve(curve)
    m["avg_portfolio_heat"] = round(float(result.get("avg_heat", float("nan"))), 4)
    m["trades"] = len(result.get("trades", []))
    m["avg_funded_names"] = round(float(np.mean(funded)), 2) if funded else None
    m["implied_heat_ceiling"] = round(float(np.mean(funded)) * max_pos, 4) if funded else None
    return {"metrics": m, "curve": curve}


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    data = load_data()
    print(f"Loaded {len(data)} symbols {DATA_START} → {END}")
    if "SPY" not in data or "AGG" not in data:
        sys.exit("SPY and AGG are required for the benchmark arms")

    new_cfg = config
    old_cfg = as_deployed(config)
    # The deployed anomaly layer sat permanently in ELEVATED. That behaviour lives in
    # code, not config, so the old arm runs with the layer off — which understates the
    # drag it caused and therefore flatters the old config.
    old_cfg.setdefault("anomaly_layer", {})["enabled"] = False

    arms = {
        "new_config": run_arm(new_cfg, data, "new_config", restore_heat=True),
        "old_config": run_arm(old_cfg, data, "old_config", restore_heat=False),
    }

    window = arms["new_config"]["curve"].index
    spy_px = data["SPY"]["Close"].reindex(window).ffill()
    agg_px = data["AGG"]["Close"].reindex(window).ffill()

    arms["spy"] = {"metrics": metrics_from_curve(spy_px), "curve": spy_px}
    sf = sixty_forty(spy_px, agg_px)
    arms["sixty_forty"] = {"metrics": metrics_from_curve(sf), "curve": sf}
    for k in ("spy", "sixty_forty"):
        arms[k]["metrics"]["avg_portfolio_heat"] = 1.0
        arms[k]["metrics"]["trades"] = None
        arms[k]["metrics"]["avg_funded_names"] = None
        arms[k]["metrics"]["implied_heat_ceiling"] = None

    labels = {
        "new_config": "New config (63d, top_n 9, heat 0.95)",
        "old_config": "Old deployed config (126d, top_n 20, heat 0.75)",
        "spy": "SPY buy & hold",
        "sixty_forty": "60/40 SPY/AGG (monthly rebal)",
    }

    print(f"\n{'Arm':<46} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'AvgHeat':>8} {'Names':>7}")
    print("-" * 92)
    for k, v in arms.items():
        m = v["metrics"]
        names = m["avg_funded_names"]
        print(
            f"{labels[k]:<46} {m['total_return_pct']:>7.2f}% {m['sharpe']:>8.2f} "
            f"{m['max_drawdown_pct']:>7.2f}% {m['avg_portfolio_heat']:>8.2f} "
            f"{'—' if names is None else f'{names:>7.1f}'}"
        )

    sf_sharpe = arms["sixty_forty"]["metrics"]["sharpe"]
    new_m = arms["new_config"]["metrics"]
    payload = {
        "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": {"start": START, "end": END, "trading_days": len(window)},
        "symbols_loaded": len(data),
        "probation": {
            "deadline": "2026-10-31",
            "benchmark": "60/40",
            "criterion": "bot Sharpe must exceed 60/40 Sharpe over the probation window",
            "sixty_forty_sharpe_in_validation": sf_sharpe,
            "new_config_sharpe_in_validation": new_m["sharpe"],
            "passes_in_validation_window": bool(new_m["sharpe"] > sf_sharpe),
        },
        "arms": {k: {"label": labels[k], **v["metrics"]} for k, v in arms.items()},
        "heat_diagnostic": {
            "max_position_pct": config["risk"]["max_position_pct"],
            "note": (
                "Gross exposure is bounded by funded_names * max_position_pct. With "
                "max_position_pct at 0.15 the 0.95 heat target needs at least 7 names "
                "carrying positive signals simultaneously; the 2026 window funds far "
                "fewer, so realised heat saturates well below target even after the "
                "optimizer restoration fix. Breadth, not renormalisation, is the "
                "remaining binding constraint."
            ),
        },
        "equity_curves": {
            k: {
                str(d.date()): round(float(x / v["curve"].iloc[0] * 100), 4)
                for d, x in v["curve"].items()
            }
            for k, v in arms.items()
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    colours = {
        "new_config": "#1f77b4",
        "old_config": "#d62728",
        "spy": "#7f7f7f",
        "sixty_forty": "#2ca02c",
    }
    for k, v in arms.items():
        c = v["curve"]
        ax.plot(c.index, c / c.iloc[0] * 100, label=labels[k], color=colours[k], lw=1.8)
    ax.set_title(f"Fix validation {START} → {END} (base 100)")
    ax.set_ylabel("Normalised equity")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    for k in ("new_config", "old_config"):
        h = arms[k]["curve"]
        heat = pd.Series(index=h.index, dtype=float)
        heat[:] = arms[k]["metrics"]["avg_portfolio_heat"]
        ax2.plot(heat.index, heat, color=colours[k], lw=1.4, label=f"{k} avg heat")
    ax2.set_ylabel("Avg gross exposure")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
