#!/usr/bin/env python3
"""
Short-selling overlay validation backtest (Gap B)
=================================================
Two windows, two arms each:

  Windows
    2022_bear — 2022-01-01 → 2022-12-31 (full bear year)
    2026_ytd  — 2026-04-22 → 2026-09-17 (live window, yfinance tail top-up)

  Arms
    long_only     — current fixed config as deployed (shorting disabled)
    short_overlay — identical config + bear-regime short overlay per
                    config/settings.yaml `shorting:` block

Metrics per arm: total return, Sharpe, MaxDD, ann. vol, turnover, % days with
short exposure, short P&L contribution (engine track), hard-stop covers.

Writes results/short_selling_backtest.json + results/short_selling_backtest.png.
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
from data.data_store import get_store
from strategy.short_overlay import LIQUID_ETF_UNIVERSE

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "results" / "short_selling_backtest.json"
OUT_PNG = ROOT / "results" / "short_selling_backtest.png"

TRADING_DAYS = 252
RISK_FREE = 0.04

WINDOWS = [
    {
        "label": "2022_bear",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "data_start": "2020-06-01",
    },
    {
        "label": "2026_ytd",
        "start": "2026-04-22",
        "end": "2026-09-17",
        "data_start": "2024-06-01",
    },
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def candidate_symbols(config: dict) -> list[str]:
    cands = config.get("dynamic_universe", {}).get("candidates", {})
    syms = (
        list(cands.get("equities", []))
        + list(cands.get("futures", []))
        + list(cands.get("crypto", []))
    )
    for extra in ["SPY", "AGG", "^VIX"]:
        if extra not in syms:
            syms.append(extra)
    for s in LIQUID_ETF_UNIVERSE:
        if s not in syms:
            syms.append(s)
    return syms


def _yticker(sym: str) -> str:
    """yfinance ticker name for a config candidate (^VIX, ES=F, BTC-USD pass through)."""
    return sym


def _top_up_from_yfinance(df: pd.DataFrame, sym: str, end: str) -> pd.DataFrame:
    """Extend a parquet frame to `end` using yfinance (no-op if fresh enough)."""
    have_until = df.index.max()
    if have_until >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=4):
        return df
    try:
        import yfinance as yf

        tail = yf.download(
            _yticker(sym),
            start=(have_until - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if tail.empty:
            print(f"  [data] {sym}: yfinance tail empty (stops {have_until.date()})")
            return df
        if isinstance(tail.columns, pd.MultiIndex):
            tail.columns = tail.columns.get_level_values(0)
        tail.columns = [c.capitalize() for c in tail.columns]
        if tail.index.tz is None:
            tail.index = tail.index.tz_localize("UTC")
        # yfinance Adj Close is merged by auto_adjust; keep OHLC(+Adj Close)/Volume
        merged = pd.concat([df, tail])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        n_new = len(merged) - len(df)
        print(f"  [data] {sym}: +{n_new} rows from yfinance (to {merged.index.max().date()})")
        return merged
    except Exception as e:
        print(f"  [data] {sym}: yfinance top-up failed ({e})")
        return df


def load_data(config: dict, data_start: str, end: str, fetch: bool = True) -> dict:
    store = get_store()
    data = {}
    for sym in candidate_symbols(config):
        df = store.load(sym)
        if df is None or len(df) < 100:
            print(f"  [data] {sym}: unavailable locally")
            if fetch:
                try:
                    import yfinance as yf

                    raw = yf.download(
                        _yticker(sym),
                        start=data_start,
                        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        auto_adjust=True,
                        progress=False,
                    )
                    if raw.empty:
                        continue
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    raw.columns = [c.capitalize() for c in raw.columns]
                    if raw.index.tz is None:
                        raw.index = raw.index.tz_localize("UTC")
                    df = raw
                except Exception:
                    continue
            else:
                continue
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()
        if fetch:
            df = _top_up_from_yfinance(df, sym, end)
        df = df[df.index >= pd.Timestamp(data_start, tz="UTC")]
        if len(df) >= 100:
            data[sym] = df
    return data


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


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


def run_arm(config: dict, data: dict, label: str, shorting_enabled: bool, window: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["backtest"]["start_date"] = window["start"]
    cfg["backtest"]["end_date"] = window["end"]
    cfg.setdefault("shorting", {})["enabled"] = shorting_enabled

    result = BacktestEngine(cfg).run(data, benchmark_data=data.get("SPY"), run_label=label)
    curve = result["equity_curve"]
    m = metrics_from_curve(curve)

    # Turnover: gross traded notional / average equity (annualised)
    trades = result.get("trades")
    n_days = max(1, len(curve))
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        gross = float((trades["quantity"].abs() * trades["fill_price"]).sum())
        avg_eq = float(curve.mean())
        m["turnover_x_per_year"] = round(gross / avg_eq * (TRADING_DAYS / n_days), 2)
        m["n_fills"] = len(trades)
    else:
        m["turnover_x_per_year"] = 0.0
        m["n_fills"] = 0

    m["avg_portfolio_heat"] = round(float(result.get("avg_heat", float("nan"))), 4)
    m["pct_days_with_shorts"] = result.get("pct_days_with_shorts", 0.0)
    m["short_pnl_total_usd"] = result.get("short_pnl_total_usd", 0.0)
    m["short_hard_stop_covers"] = result.get("short_hard_stop_covers", 0)
    m["short_entries"] = result.get("short_entries", 0)
    return {"metrics": m, "curve": curve}


def main() -> None:
    fetch = "--no-fetch" not in sys.argv
    config = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())

    out_windows = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, window in zip(axes, WINDOWS):
        print(f"\n=== Window: {window['label']} ({window['start']} → {window['end']}) ===")
        data = load_data(config, window["data_start"], window["end"], fetch=fetch)
        print(f"Loaded {len(data)} symbols (data_start {window['data_start']})")
        if "SPY" not in data:
            sys.exit(f"SPY required for window {window['label']}")

        arms = {
            "long_only": run_arm(config, data, "long_only", False, window),
            "short_overlay": run_arm(config, data, "short_overlay", True, window),
        }

        spy_px = data["SPY"]["Close"]
        w0, w1 = pd.Timestamp(window["start"], tz="UTC"), pd.Timestamp(window["end"], tz="UTC")
        spy_win = spy_px[(spy_px.index >= w0) & (spy_px.index <= w1)]
        arms["spy_ref"] = {"metrics": metrics_from_curve(spy_win), "curve": spy_win}

        labels = {
            "long_only": "Long-only (current config)",
            "short_overlay": "+ Bear-regime short overlay",
            "spy_ref": "SPY buy & hold",
        }
        colours = {"long_only": "#7f7f7f", "short_overlay": "#1f77b4", "spy_ref": "#d62728"}

        print(
            f"  {'Arm':<30} {'Return':>9} {'Sharpe':>7} {'MaxDD':>8} {'Turn/yr':>8} "
            f"{'%short days':>11} {'short P&L':>11}"
        )
        print("  " + "-" * 88)
        for k, v in arms.items():
            m = v["metrics"]
            print(
                f"  {labels[k]:<30} {m['total_return_pct']:>8.2f}% {m['sharpe']:>7.2f} "
                f"{m['max_drawdown_pct']:>7.2f}% {m.get('turnover_x_per_year', float('nan')):>7.1f}x "
                f"{m.get('pct_days_with_shorts', 0.0):>10.1f}% "
                f"{m.get('short_pnl_total_usd', 0.0):>10.0f}$"
            )

        for k, v in arms.items():
            c = v["curve"]
            ax.plot(c.index, c / c.iloc[0] * 100, label=labels[k], color=colours[k], lw=1.6)
        ax.set_title(f"{window['label']}: {window['start']} → {window['end']} (base 100)")
        ax.set_ylabel("Normalised equity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        out_windows[window["label"]] = {
            "start": window["start"],
            "end": window["end"],
            "trading_days": len(arms["long_only"]["curve"]),
            "symbols_loaded": len(data),
            "arms": {k: {"label": labels[k], **v["metrics"]} for k, v in arms.items()},
        }

    payload = {
        "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": "Gap B short overlay validation: long-only vs +short overlay",
        "config_shorting": config.get("shorting", {}),
        "windows": out_windows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    fig.suptitle("Short-selling overlay validation — long-only vs bear-regime short overlay")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
