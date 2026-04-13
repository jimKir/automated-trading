#!/usr/bin/env python3
"""
Paper Trading Monitor
=====================
Evaluates the 6 go-live criteria from daily paper trading data and writes
a scorecard to results/paper_monitor.json.  Optionally updates the README
tracking table with current values.

Data sources (checked in order):
  1. results/paper_state.json  — written by daily_report.py
  2. Alpaca paper account       — live API fallback (needs credentials)

Backtest reference:
  results/wf_12m_strat_returns.csv — walk-forward OOS daily returns for
  correlation comparison (criterion #5).

Usage:
    python scripts/paper_monitor.py                # evaluate + save JSON
    python scripts/paper_monitor.py --update-readme # also patch README table
    python scripts/paper_monitor.py --verbose       # detailed console output

Run daily after market close (or weekly for the README update).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAPER_STATE = ROOT / "results" / "paper_state.json"
OOS_RETURNS = ROOT / "results" / "wf_12m_strat_returns.csv"
OUTPUT_JSON = ROOT / "results" / "paper_monitor.json"
README_PATH = ROOT / "README.md"
PERIODS_YEAR = 252

# ── Go-live thresholds (must match docs/paper_trading_runbook.md §7) ─────────
THRESHOLDS = {
    "sharpe": {"op": ">", "value": 0.50, "label": "Annualised Sharpe"},
    "max_drawdown_pct": {"op": "<", "value": 15.0, "label": "Max Drawdown"},
    "dd_recovery": {"op": ">=", "value": 1, "label": "Drawdown Recovery"},
    "win_rate_pct": {"op": ">", "value": 50.0, "label": "Win Rate"},
    "backtest_corr": {"op": ">", "value": 0.60, "label": "Correlation to Backtest"},
    "uptime_pct": {"op": ">", "value": 95.0, "label": "System Uptime"},
}


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════════


def load_equity_history() -> pd.DataFrame:
    """Load equity history from paper_state.json or Alpaca fallback.
    Returns DataFrame with columns [date, equity]."""

    # Source 1: paper_state.json (from daily_report.py)
    if PAPER_STATE.exists():
        with open(PAPER_STATE) as f:
            state = json.load(f)
        history = state.get("equity_history", [])
        if history:
            df = pd.DataFrame(history)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
            df = df.set_index("date")
            return df

    # Source 2: Alpaca API (single-day snapshot — appended to output JSON)
    try:
        from execution.alpaca_broker import AlpacaBroker
        from utils.config_loader import load_config

        config = load_config()
        broker = AlpacaBroker(config)
        if broker.connect():
            acct = broker.get_account()
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            return pd.DataFrame([{"equity": acct.equity}], index=pd.to_datetime([today]))
    except Exception:
        pass

    return pd.DataFrame(columns=["equity"])


def load_oos_returns() -> pd.Series | None:
    """Load WF 12M OOS strategy returns for backtest correlation."""
    if not OOS_RETURNS.exists():
        return None
    df = pd.read_csv(OOS_RETURNS, parse_dates=["date"], index_col="date")
    return df["strategy"].dropna()


def load_prior_scorecard() -> dict:
    """Load previous scorecard for uptime tracking and append mode."""
    if OUTPUT_JSON.exists():
        return json.load(open(OUTPUT_JSON))
    return {}


# ═════════════════════════════════════════════════════════════════════════════
#  Metric computation
# ═════════════════════════════════════════════════════════════════════════════


def compute_metrics(equity: pd.DataFrame) -> dict:
    """Compute all 6 go-live metrics from an equity history DataFrame."""

    if len(equity) < 2:
        return _empty_metrics("Need at least 2 days of data")

    equities = equity["equity"].astype(float)
    returns = equities.pct_change().dropna()

    if len(returns) < 5:
        return _empty_metrics("Need at least 5 daily returns")

    n_days = len(returns)

    # 1. Annualised Sharpe
    ann_ret = (1 + returns).prod() ** (PERIODS_YEAR / n_days) - 1
    ann_vol = returns.std() * np.sqrt(PERIODS_YEAR)
    sharpe = round(float(ann_ret / ann_vol) if ann_vol > 0 else 0.0, 3)

    # 2. Max Drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd_pct = round(abs(float(dd.min())) * 100, 2)

    # 3. Drawdown recovery episodes
    #    Count distinct DD episodes > 5% that recovered (DD returned to 0)
    dd_episodes = _count_dd_recoveries(dd, threshold=-0.05)

    # 4. Win rate
    win_rate_pct = round(float((returns > 0).sum() / len(returns)) * 100, 1)

    # 5. Correlation to WF OOS backtest
    oos_ret = load_oos_returns()
    backtest_corr = _compute_backtest_correlation(returns, oos_ret)

    # 6. System uptime (trading days with data / expected trading days)
    start_date = equities.index.min()
    end_date = equities.index.max()
    # Generate the expected US trading calendar between start and end
    expected_days = pd.bdate_range(start_date, end_date)
    uptime_pct = round(float(n_days / max(len(expected_days) - 1, 1)) * 100, 1)
    # Cap at 100% (weekends/holidays can make it slightly over)
    uptime_pct = min(uptime_pct, 100.0)

    # Ancillary stats
    total_return_pct = round(float((equities.iloc[-1] / equities.iloc[0] - 1) * 100), 2)
    cagr = round(float(ann_ret * 100), 2)

    return {
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "dd_recovery": dd_episodes,
        "win_rate_pct": win_rate_pct,
        "backtest_corr": backtest_corr,
        "uptime_pct": uptime_pct,
        "n_trading_days": n_days,
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr,
        "ann_vol_pct": round(float(ann_vol * 100), 2),
        "start_date": str(equities.index.min().date()),
        "end_date": str(equities.index.max().date()),
        "current_equity": round(float(equities.iloc[-1]), 2),
        "peak_equity": round(float(equities.max()), 2),
        "error": None,
    }


def _empty_metrics(reason: str) -> dict:
    return dict.fromkeys(THRESHOLDS) | {
        "n_trading_days": 0,
        "total_return_pct": None,
        "cagr_pct": None,
        "ann_vol_pct": None,
        "start_date": None,
        "end_date": None,
        "current_equity": None,
        "peak_equity": None,
        "error": reason,
    }


def _count_dd_recoveries(dd_series: pd.Series, threshold: float = -0.05) -> int:
    """Count distinct drawdown episodes deeper than `threshold` that recovered."""
    in_episode = False
    recovered_count = 0

    for val in dd_series:
        if not in_episode and val <= threshold:
            in_episode = True
        elif in_episode and val >= -0.001:
            # Recovered to within 0.1% of peak
            recovered_count += 1
            in_episode = False

    return recovered_count


def _compute_backtest_correlation(paper_ret: pd.Series, oos_ret: pd.Series | None) -> float | None:
    """Correlate paper returns with OOS backtest returns on overlapping dates."""
    if oos_ret is None or len(oos_ret) < 10:
        return None

    # Align on common dates
    paper_ret = paper_ret.copy()
    paper_ret.index = pd.to_datetime(paper_ret.index).normalize()
    oos_ret = oos_ret.copy()
    oos_ret.index = pd.to_datetime(oos_ret.index).normalize()

    common = paper_ret.index.intersection(oos_ret.index)
    if len(common) < 10:
        return None

    corr = float(paper_ret.loc[common].corr(oos_ret.loc[common]))
    return round(corr, 3) if np.isfinite(corr) else None


# ═════════════════════════════════════════════════════════════════════════════
#  Scorecard evaluation
# ═════════════════════════════════════════════════════════════════════════════


def evaluate(metrics: dict) -> list[dict]:
    """Evaluate each threshold and return a list of criterion results."""
    results = []
    for key, spec in THRESHOLDS.items():
        current = metrics.get(key)
        if current is None:
            status = "Pending"
            display = "—"
        else:
            op = spec["op"]
            target = spec["value"]
            if op == ">":
                passed = current > target
            elif op == "<":
                passed = current < target
            elif op == ">=":
                passed = current >= target
            else:
                passed = False

            status = "PASSED" if passed else "FAILING"

            # Format display value
            if key == "dd_recovery":
                display = f"{current} episode{'s' if current != 1 else ''}"
            elif key in ("sharpe", "backtest_corr"):
                display = f"{current:.3f}"
            else:
                display = f"{current:.1f}%"

        results.append(
            {
                "index": list(THRESHOLDS.keys()).index(key) + 1,
                "metric": spec["label"],
                "threshold": _format_threshold(spec),
                "current": display,
                "status": status,
                "key": key,
                "raw_value": current,
            }
        )
    return results


def _format_threshold(spec: dict) -> str:
    op = spec["op"]
    val = spec["value"]
    if isinstance(val, float):
        val_str = str(int(val)) if val == int(val) else f"{val:.2f}"
    else:
        val_str = str(val)

    if (
        spec.get("label", "").endswith("Drawdown")
        or spec.get("label", "").endswith("Rate")
        or spec.get("label", "").endswith("Uptime")
    ):
        return f"{op} {val_str}%"
    if spec.get("label", "").endswith("Recovery"):
        return f"{op} {val_str} episode"
    return f"{op} {val_str}"


# ═════════════════════════════════════════════════════════════════════════════
#  Output
# ═════════════════════════════════════════════════════════════════════════════


def save_scorecard(metrics: dict, criteria: list[dict]) -> None:
    """Write full scorecard to results/paper_monitor.json."""
    n_passed = sum(1 for c in criteria if c["status"] == "PASSED")
    n_total = len(criteria)

    scorecard = {
        "run_date": datetime.now(UTC).isoformat(),
        "summary": f"{n_passed}/{n_total} criteria passed",
        "go_live_ready": n_passed == n_total,
        "metrics": metrics,
        "criteria": criteria,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(scorecard, f, indent=2, default=str)

    print(f"Scorecard saved → {OUTPUT_JSON}")


def update_readme(criteria: list[dict]) -> None:
    """Patch the README tracking table with current values."""
    if not README_PATH.exists():
        print("README.md not found — skipping update")
        return

    readme = README_PATH.read_text()

    # Match each row of the tracking table:
    # | 1 | Annualised Sharpe | > 0.50 | — | Tracking |
    for c in criteria:
        idx = c["index"]
        current = c["current"]
        status = c["status"]

        # Pattern: | <idx> | <metric> | <threshold> | <anything> | <anything> |
        pattern = (
            rf"(\| {idx} \| {re.escape(c['metric'])} \| {re.escape(c['threshold'])} \|)"
            rf" [^|]+ \| [^|]+ \|"
        )
        replacement = rf"\1 {current} | {status} |"
        readme_new = re.sub(pattern, replacement, readme)

        if readme_new != readme:
            readme = readme_new
        else:
            # Fallback: match by index number and metric name only
            pattern_loose = (
                rf"(\| {idx} \| {re.escape(c['metric'])} \|)"
                rf" [^|]+ \| [^|]+ \| [^|]+ \|"
            )
            replacement_loose = rf"\1 {c['threshold']} | {current} | {status} |"
            readme = re.sub(pattern_loose, replacement_loose, readme)

    README_PATH.write_text(readme)
    print(f"README updated → {README_PATH}")


def print_scorecard(criteria: list[dict], metrics: dict, verbose: bool = False) -> None:
    """Print a formatted scorecard to the console."""
    n_passed = sum(1 for c in criteria if c["status"] == "PASSED")
    n_total = len(criteria)

    print(f"\n{'=' * 70}")
    print("  PAPER TRADING GO-LIVE SCORECARD")
    print(
        f"  {metrics.get('n_trading_days', 0)} trading days "
        f"({metrics.get('start_date', '?')} → {metrics.get('end_date', '?')})"
    )
    print(f"{'=' * 70}")
    print(f"{'#':<3} {'Metric':<28} {'Threshold':<20} {'Current':<15} {'Status':<10}")
    print("-" * 70)
    for c in criteria:
        marker = "  " if c["status"] == "Pending" else ("✓ " if c["status"] == "PASSED" else "✗ ")
        print(
            f"{marker}{c['index']:<2} {c['metric']:<28} {c['threshold']:<20} "
            f"{c['current']:<15} {c['status']:<10}"
        )
    print("-" * 70)
    print(f"  Result: {n_passed}/{n_total} criteria passed", end="")
    if n_passed == n_total:
        print(" → READY FOR LIVE CAPITAL")
    elif metrics.get("error"):
        print(f" ({metrics['error']})")
    else:
        print()

    if verbose and metrics.get("n_trading_days", 0) > 0:
        print(
            f"\n  Equity: ${metrics.get('current_equity', 0):,.2f} "
            f"(peak: ${metrics.get('peak_equity', 0):,.2f})"
        )
        print(
            f"  Total return: {metrics.get('total_return_pct', 0):+.2f}%  "
            f"CAGR: {metrics.get('cagr_pct', 0):+.2f}%  "
            f"Vol: {metrics.get('ann_vol_pct', 0):.2f}%"
        )
    print()


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Paper trading go-live monitor")
    parser.add_argument(
        "--update-readme", action="store_true", help="Update the README tracking table"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    equity = load_equity_history()
    metrics = compute_metrics(equity)
    criteria = evaluate(metrics)

    print_scorecard(criteria, metrics, verbose=args.verbose)
    save_scorecard(metrics, criteria)

    if args.update_readme:
        update_readme(criteria)


if __name__ == "__main__":
    main()
