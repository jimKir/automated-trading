#!/usr/bin/env python3
"""
What-If Scenario Analyser
==========================
Run multiple backtest scenarios in parallel, each with different
parameters, and get a side-by-side comparison report.

Usage examples:
  # Run the built-in scenario suite
  python whatif.py

  # Run a specific named suite
  python whatif.py --suite capital
  python whatif.py --suite risk
  python whatif.py --suite strategy
  python whatif.py --suite universe
  python whatif.py --suite costs
  python whatif.py --suite crisis          # historical crash periods only
  python whatif.py --suite all             # everything (slow)

  # Run a single custom override inline
  python whatif.py --param capital.initial_equity=50000 --param risk.max_position_pct=0.15

  # Compare two date windows (regime analysis)
  python whatif.py --suite periods

Available parameter paths (use dot notation):
  capital.initial_equity            Starting capital
  capital.max_portfolio_heat        Max % equity at risk simultaneously
  risk.max_position_pct             Max single position size
  risk.max_drawdown_halt            Circuit-breaker drawdown threshold
  risk.daily_loss_limit             Circuit-breaker daily loss threshold
  risk.kelly_fraction               Fractional Kelly multiplier
  strategy.lookback_fast            Fast momentum window (days)
  strategy.lookback_slow            Slow momentum window (days)
  strategy.zscore_entry             Mean-reversion entry z-score
  strategy.rebalance_frequency      daily | weekly | monthly
  costs.impact_scale                Market impact multiplier (0=none, 2=conservative)
  costs.capital_gains_tax_rate      Tax on realised gains (0.15 = 15%)
  backtest.start_date               Period start
  backtest.end_date                 Period end
  assets.crypto.enabled             true | false
  assets.futures.enabled            true | false
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("WhatIf")

# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

SUITES: dict[str, list[dict]] = {
    # ── Capital sizing ────────────────────────────────────────────────────────
    "capital": [
        {"name": "€10k Capital", "capital.initial_equity": 10000},
        {"name": "€25k Capital (base)", "capital.initial_equity": 25000},
        {"name": "€50k Capital", "capital.initial_equity": 50000},
        {"name": "€100k Capital", "capital.initial_equity": 100000},
        {"name": "€200k Capital", "capital.initial_equity": 200000},
    ],
    # ── Risk parameters ───────────────────────────────────────────────────────
    "risk": [
        {
            "name": "Conservative (5% pos, 10% DD)",
            "risk.max_position_pct": 0.05,
            "risk.max_drawdown_halt": 0.10,
            "capital.max_portfolio_heat": 0.15,
        },
        {
            "name": "Base (10% pos, 15% DD)",
            "risk.max_position_pct": 0.10,
            "risk.max_drawdown_halt": 0.15,
            "capital.max_portfolio_heat": 0.20,
        },
        {
            "name": "Moderate (15% pos, 20% DD)",
            "risk.max_position_pct": 0.15,
            "risk.max_drawdown_halt": 0.20,
            "capital.max_portfolio_heat": 0.30,
        },
        {
            "name": "Aggressive (20% pos, 25% DD)",
            "risk.max_position_pct": 0.20,
            "risk.max_drawdown_halt": 0.25,
            "capital.max_portfolio_heat": 0.40,
        },
    ],
    # ── Strategy tuning ───────────────────────────────────────────────────────
    "strategy": [
        {
            "name": "Fast (10/30 momentum)",
            "strategy.lookback_fast": 10,
            "strategy.lookback_slow": 30,
        },
        {
            "name": "Base (20/60 momentum)",
            "strategy.lookback_fast": 20,
            "strategy.lookback_slow": 60,
        },
        {
            "name": "Slow (40/120 momentum)",
            "strategy.lookback_fast": 40,
            "strategy.lookback_slow": 120,
        },
        {"name": "Tight MR (z=1.5)", "strategy.zscore_entry": 1.5, "strategy.zscore_exit": 0.3},
        {"name": "Wide MR (z=2.5)", "strategy.zscore_entry": 2.5, "strategy.zscore_exit": 0.8},
        {"name": "Daily rebalance", "strategy.rebalance_frequency": "daily"},
        {"name": "Monthly rebalance", "strategy.rebalance_frequency": "monthly"},
    ],
    # ── Universe composition ──────────────────────────────────────────────────
    "universe": [
        {"name": "Equities only", "assets.crypto.enabled": False, "assets.futures.enabled": False},
        {
            "name": "Equities + Futures",
            "assets.crypto.enabled": False,
            "assets.futures.enabled": True,
        },
        {
            "name": "Equities + Crypto",
            "assets.crypto.enabled": True,
            "assets.futures.enabled": False,
        },
        {
            "name": "Full universe (base)",
            "assets.crypto.enabled": True,
            "assets.futures.enabled": True,
        },
    ],
    # ── Cost sensitivity ──────────────────────────────────────────────────────
    "costs": [
        {
            "name": "Zero costs (theoretical)",
            "costs.impact_scale": 0.0,
            "backtest.commission_pct": 0.0,
            "backtest.slippage_pct": 0.0,
        },
        {
            "name": "Low costs (IBKR Pro tier)",
            "costs.impact_scale": 0.5,
            "backtest.commission_pct": 0.0001,
            "backtest.slippage_pct": 0.0002,
        },
        {
            "name": "Base costs (realistic)",
            "costs.impact_scale": 1.0,
            "backtest.commission_pct": 0.001,
            "backtest.slippage_pct": 0.0005,
        },
        {
            "name": "High costs (retail broker)",
            "costs.impact_scale": 2.0,
            "backtest.commission_pct": 0.002,
            "backtest.slippage_pct": 0.001,
        },
        {
            "name": "Base + Greek CGT (15%)",
            "costs.impact_scale": 1.0,
            "costs.capital_gains_tax_rate": 0.15,
        },
    ],
    # ── Historical periods / crisis windows ───────────────────────────────────
    "crisis": [
        {
            "name": "COVID crash (2020)",
            "backtest.start_date": "2020-01-01",
            "backtest.end_date": "2020-12-31",
        },
        {
            "name": "2022 Bear Market",
            "backtest.start_date": "2022-01-01",
            "backtest.end_date": "2022-12-31",
        },
        {
            "name": "Bull run 2019-2021",
            "backtest.start_date": "2019-01-01",
            "backtest.end_date": "2021-12-31",
        },
        {
            "name": "Crypto winter 2022",
            "backtest.start_date": "2022-01-01",
            "backtest.end_date": "2023-06-30",
        },
        {
            "name": "Full backtest 2018-2025",
            "backtest.start_date": "2018-01-01",
            "backtest.end_date": "2025-12-31",
        },
    ],
    # ── Market regimes / rolling periods ─────────────────────────────────────
    "periods": [
        {
            "name": "2018-2019 (late cycle)",
            "backtest.start_date": "2018-01-01",
            "backtest.end_date": "2019-12-31",
        },
        {
            "name": "2020-2021 (pandemic + bull)",
            "backtest.start_date": "2020-01-01",
            "backtest.end_date": "2021-12-31",
        },
        {
            "name": "2022-2023 (bear + recovery)",
            "backtest.start_date": "2022-01-01",
            "backtest.end_date": "2023-12-31",
        },
        {
            "name": "2024-2025 (AI bull run)",
            "backtest.start_date": "2024-01-01",
            "backtest.end_date": "2025-12-31",
        },
    ],
}

# Combine all for --suite all
SUITES["all"] = [s for suite in SUITES.values() for s in suite]


# ─────────────────────────────────────────────────────────────────────────────
# Parameter override engine
# ─────────────────────────────────────────────────────────────────────────────


def apply_overrides(config: dict, overrides: dict[str, Any]) -> dict:
    """
    Apply dot-notation overrides to a config dict.
    e.g. "capital.initial_equity" = 50000
         "assets.crypto.enabled"  = False
    """
    cfg = copy.deepcopy(config)
    for key_path, value in overrides.items():
        if key_path == "name":
            continue
        parts = key_path.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        # Type coercion
        final_key = parts[-1]
        existing = node.get(final_key)
        if isinstance(existing, bool):
            if isinstance(value, str):
                value = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int) and not isinstance(value, bool):
            value = int(value)
        elif isinstance(existing, float):
            value = float(value)
        node[final_key] = value
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Single scenario runner (runs in subprocess for parallelism)
# ─────────────────────────────────────────────────────────────────────────────


def _run_scenario(args_tuple) -> dict:
    """
    Worker function: runs one backtest scenario and returns metrics dict.
    Designed to run in a ProcessPoolExecutor worker.
    """
    scenario, base_config_path = args_tuple
    import logging

    logging.disable(logging.CRITICAL)  # silence in workers

    sys.path.insert(0, str(Path(base_config_path).parent.parent))
    from backtest.engine import BacktestEngine
    from data.feed import DataFeed

    try:
        config = load_config(base_config_path)
        overrides = {k: v for k, v in scenario.items() if k != "name"}
        config = apply_overrides(config, overrides)

        start = config["backtest"]["start_date"]
        end = config["backtest"]["end_date"]

        feed = DataFeed(config)
        all_data = feed.load_all(start=start, end=end)
        bench = feed.load(
            [config["backtest"].get("benchmark", "SPY")], start=start, end=end, source="yfinance"
        )
        benchmark = bench.get(config["backtest"].get("benchmark", "SPY"))

        engine = BacktestEngine(config)
        metrics = engine.run(all_data, benchmark_data=benchmark)

        # Strip non-serialisable objects
        result = {
            k: v
            for k, v in metrics.items()
            if not isinstance(v, (pd.Series, pd.DataFrame, dict)) or k == "stress_scenarios"
        }
        result["scenario_name"] = scenario.get("name", "unnamed")
        result["overrides"] = overrides
        return result

    except Exception as exc:
        return {
            "scenario_name": scenario.get("name", "unnamed"),
            "error": str(exc),
            "overrides": {k: v for k, v in scenario.items() if k != "name"},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report generator
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_METRICS = [
    ("Total Return (%)", "total_return_pct", True),
    ("Ann. Return (%)", "ann_return_pct", True),
    ("Ann. Volatility (%)", "ann_volatility_pct", False),
    ("Sharpe Ratio", "sharpe_ratio", True),
    ("Sortino Ratio", "sortino_ratio", True),
    ("Calmar Ratio", "calmar_ratio", True),
    ("Max Drawdown (%)", "max_drawdown_pct", False),  # less negative = better
    ("MDD Duration (days)", "max_drawdown_duration_days", False),
    ("Win Rate (%)", "win_rate_pct", True),
    ("VaR 99% (%/day)", "var_hist_99_pct", False),
    ("CVaR 99% (%/day)", "cvar_hist_99_pct", False),
    ("Omega Ratio", "omega_ratio", True),
    ("Tail Ratio", "tail_ratio", True),
    ("Skewness", "skewness", True),
    ("Excess Kurtosis", "excess_kurtosis", False),
    ("Alpha Ann. (%)", "alpha_ann_pct", True),
    ("Beta", "beta", False),
    ("Total Cost ($)", "cost_total", False),
    ("Final Equity ($)", "final_equity", True),
]


def _fmt(v, key: str) -> str:
    if v is None:
        return "N/A"
    try:
        fv = float(v)
        if "pct" in key or "Return" in key or "Drawdown" in key or "Rate" in key:
            return f"{fv:.2f}%"
        if "dollar" in key.lower() or "equity" in key.lower() or "cost" in key.lower():
            return f"${fv:,.2f}"
        return f"{fv:.4f}"
    except (TypeError, ValueError):
        return str(v)


def generate_comparison_report(
    results: list[dict],
    suite_name: str,
    output_dir: str = "results",
) -> str:
    """Generate HTML comparison table + chart for all scenarios."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    good = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]

    if not good:
        log.error("All scenarios failed — no report generated")
        return ""

    # ── Table data ────────────────────────────────────────────────────────────
    table_rows = ""
    for label, key, higher_is_better in COMPARISON_METRICS:
        values = [r.get(key) for r in good]
        row_cells = f"<td><strong>{label}</strong></td>"
        numeric = [float(v) for v in values if v is not None]
        best = (max(numeric) if higher_is_better else min(numeric)) if numeric else None

        for r in good:
            v = r.get(key)
            fmt = _fmt(v, key)
            try:
                is_best = best is not None and abs(float(v) - best) < 1e-9
            except (TypeError, ValueError):
                is_best = False
            cls = "best" if is_best else ""
            row_cells += f"<td class='{cls}'>{fmt}</td>"
        table_rows += f"<tr>{row_cells}</tr>\n"

    # ── Override rows ─────────────────────────────────────────────────────────
    override_rows = ""
    for r in good:
        ov = r.get("overrides", {})
        formatted = ", ".join(f"{k}={v}" for k, v in ov.items()) if ov else "base"
        override_rows += f"<tr><td><em>Overrides</em></td><td colspan='{len(good)}'><small style='color:#8b949e'>{formatted}</small></td></tr>\n"

    # Actually put overrides per-column
    override_row = "<tr><td><em>Overrides</em></td>"
    for r in good:
        ov = r.get("overrides", {})
        formatted = (
            "<br>".join(f"<code>{k}={v}</code>" for k, v in ov.items()) if ov else "<em>base</em>"
        )
        override_row += f"<td style='font-size:10px;color:#8b949e'>{formatted}</td>"
    override_row += "</tr>"

    # ── Bar chart: key metrics ────────────────────────────────────────────────
    scenario_names = [r["scenario_name"] for r in good]
    chart_metrics = [
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Ann. Return (%)", "ann_return_pct"),
        ("Max Drawdown (%)", "max_drawdown_pct"),
        ("Calmar Ratio", "calmar_ratio"),
    ]

    n_metrics = len(chart_metrics)
    n_scenarios = len(good)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(5, n_scenarios * 0.6 + 2)))
    fig.suptitle(f"What-If Analysis: {suite_name.upper()} Suite", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#0d1117")

    plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_scenarios))

    for ax_idx, (metric_label, metric_key) in enumerate(chart_metrics):
        ax = axes[ax_idx] if n_metrics > 1 else axes
        values = [r.get(metric_key, 0) or 0 for r in good]
        bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
        bars = ax.barh(scenario_names, values, color=bar_colors, alpha=0.85)
        ax.set_title(metric_label, color="white", fontsize=10, fontweight="bold")
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#30363d")
        ax.axvline(0, color="#8b949e", linewidth=0.8)
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(
                val + (max(values) - min(values)) * 0.02
                if val >= 0
                else val - (max(values) - min(values)) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                color="white",
                fontsize=7,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = Path(output_dir) / f"whatif_{suite_name}_chart.png"
    fig.savefig(chart_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    import base64

    chart_b64 = ""
    if chart_path.exists():
        with open(chart_path, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode()

    # ── Header row ────────────────────────────────────────────────────────────
    header = "<th>Metric</th>" + "".join(f"<th>{r['scenario_name']}</th>" for r in good)

    # ── Error section ─────────────────────────────────────────────────────────
    error_section = ""
    if bad:
        error_rows = "\n".join(
            f"<tr><td>{r['scenario_name']}</td><td style='color:#f85149'>{r['error']}</td></tr>"
            for r in bad
        )
        error_section = f"""
        <h2 style="color:#f85149">Failed Scenarios</h2>
        <table><tr><th>Scenario</th><th>Error</th></tr>{error_rows}</table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>What-If Analysis — {suite_name}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
  h2 {{ color: #79c0ff; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; overflow-x: auto; display: block; margin-bottom: 20px; white-space: nowrap; }}
  th {{ background: #161b22; color: #58a6ff; padding: 10px 14px; text-align: center; position: sticky; top: 0; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 7px 14px; border-bottom: 1px solid #21262d; text-align: center; }}
  td:first-child {{ text-align: left; font-size: 13px; color: #79c0ff; }}
  tr:hover {{ background: #161b22; }}
  .best {{ color: #3fb950; font-weight: bold; background: #0d2b1a; }}
  .chart {{ width: 100%; margin: 20px 0; border-radius: 8px; }}
  code {{ background: #161b22; padding: 1px 4px; border-radius: 3px; font-size: 10px; }}
  .note {{ color: #8b949e; font-size: 12px; margin-top: 6px; }}
</style>
</head>
<body>
<h1>What-If Analysis: {suite_name.upper()}</h1>
<p class="note">Green-highlighted cells = best value per metric across all scenarios.</p>

{"<img class='chart' src='data:image/png;base64," + chart_b64 + "' />" if chart_b64 else ""}

<h2>Scenario Comparison</h2>
<table>
  <tr>{header}</tr>
  {override_row}
  {table_rows}
</table>

{error_section}

<p style="color:#8b949e;font-size:11px;margin-top:40px;">
  Generated by Trading System What-If Analyser | Past performance ≠ future results.
</p>
</body>
</html>"""

    out = Path(output_dir) / f"whatif_{suite_name}.html"
    with open(out, "w") as f:
        f.write(html)
    log.info(f"What-If report: {out}")
    return str(out)


def save_comparison_csv(results: list[dict], suite_name: str, output_dir: str = "results") -> str:
    """Save scenario comparison as CSV for further analysis in Excel/Pandas."""
    good = [r for r in results if "error" not in r]
    if not good:
        return ""
    rows = []
    for r in good:
        row = {"scenario": r["scenario_name"]}
        for _, key, _ in COMPARISON_METRICS:
            row[key] = r.get(key)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(output_dir) / f"whatif_{suite_name}.csv"
    df.to_csv(out, index=False)
    log.info(f"CSV saved: {out}")
    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="What-If Scenario Analyser")
    parser.add_argument(
        "--suite", default="risk", choices=list(SUITES.keys()), help="Named scenario suite to run"
    )
    parser.add_argument("--config", default="config/settings.yaml", help="Base config file")
    parser.add_argument(
        "--param",
        action="append",
        dest="params",
        default=[],
        metavar="key=value",
        help="Override a single parameter (repeatable). e.g. --param capital.initial_equity=50000",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    # If custom params provided, build a single-scenario suite
    if args.params:
        overrides = {}
        for p in args.params:
            k, v = p.split("=", 1)
            overrides[k.strip()] = v.strip()
        # Try to cast numerics
        for k, v in overrides.items():
            try:
                overrides[k] = float(v) if "." in v else int(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    overrides[k] = v.lower() == "true"
        overrides["name"] = "Custom: " + ", ".join(args.params)
        scenarios = [{"name": "Base (no overrides)"}, overrides]
        suite_name = "custom"
    else:
        scenarios = SUITES[args.suite]
        suite_name = args.suite

    config_path = str(Path(args.config).resolve())

    log.info(f"Running {len(scenarios)} scenarios [{suite_name}] with {args.workers} workers...")
    print(f"\n{'=' * 60}")
    print(f"  What-If Suite: {suite_name.upper()} — {len(scenarios)} scenarios")
    print(f"{'=' * 60}")

    # Run scenarios in parallel
    results = []
    worker_args = [(s, config_path) for s in scenarios]

    # Use sequential execution to avoid multiprocessing issues in some envs
    for i, wa in enumerate(worker_args):
        scenario_name = wa[0].get("name", f"scenario_{i + 1}")
        print(f"  [{i + 1}/{len(scenarios)}] Running: {scenario_name} ...", flush=True)
        result = _run_scenario(wa)
        if "error" in result:
            print(f"    ERROR: {result['error']}")
        else:
            tr = result.get("total_return_pct", 0)
            sh = result.get("sharpe_ratio", 0)
            dd = result.get("max_drawdown_pct", 0)
            print(f"    Return={tr:.2f}%  Sharpe={sh:.3f}  MaxDD={dd:.2f}%")
        results.append(result)

    print(f"\n{'=' * 60}")
    print("  Generating comparison report...")

    html_path = generate_comparison_report(results, suite_name, args.output)
    csv_path = save_comparison_csv(results, suite_name, args.output)

    # Console summary table
    good = [r for r in results if "error" not in r]
    if good:
        print(
            f"\n  {'Scenario':<40} {'Return':>9} {'Sharpe':>8} {'MaxDD':>9} {'Calmar':>8} {'Final $':>12}"
        )
        print(f"  {'-' * 40} {'-' * 9} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 12}")
        for r in good:
            print(
                f"  {r['scenario_name']:<40} "
                f"{r.get('total_return_pct', 0):>8.2f}% "
                f"{r.get('sharpe_ratio', 0) or 0:>8.3f} "
                f"{r.get('max_drawdown_pct', 0) or 0:>8.2f}% "
                f"{r.get('calmar_ratio', 0) or 0:>8.3f} "
                f"${r.get('final_equity', 0) or 0:>11,.0f}"
            )

    print(f"\n  HTML Report: {html_path}")
    print(f"  CSV Export:  {csv_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
