#!/usr/bin/env python3
"""
Trading System — Main Entry Point
==================================

Usage:
  python main.py backtest             # Run full backtest (uses config dates)
  python main.py paper                # Start paper trading loop
  python main.py live                 # Start LIVE trading (requires broker config)
  python main.py signals              # Print today's signals and exit
  python main.py report               # Re-generate report from last backtest results

Options:
  --config PATH   Path to YAML config (default: config/settings.yaml)
  --start DATE    Override backtest start date (YYYY-MM-DD)
  --end   DATE    Override backtest end date   (YYYY-MM-DD)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("Main")


def _load_all_data(config: dict, start: str, end: str) -> dict:
    """Load all candidate data, using DynamicCandidateBuilder if enabled."""
    from data.feed import DataFeed, fetch_yfinance
    from strategy.universe import DynamicCandidateBuilder

    dc_enabled = config.get("dynamic_candidates", {}).get("enabled", False)
    du_enabled = config.get("dynamic_universe", {}).get("enabled", False)

    if dc_enabled and du_enabled:
        # Build full candidate list from S&P500 + NDX100 + config fixed assets
        builder = DynamicCandidateBuilder(config)
        candidates = builder.get_full_candidate_list(config, start, end, verbose=True)
        log.info(f"Dynamic candidates: fetching data for {len(candidates)} instruments...")
        all_data = {}
        for sym in candidates:
            try:
                d = fetch_yfinance([sym], start, end)
                if sym in d and not d[sym].empty:
                    all_data[sym] = d[sym]
            except Exception:
                pass
        log.info(f"Loaded: {len(all_data)} instruments with data")
        return all_data
    if du_enabled:
        # Use static candidate list from config
        du_cfg = config.get("dynamic_universe", {}).get("candidates", {})
        all_syms = du_cfg.get("equities", []) + du_cfg.get("futures", []) + du_cfg.get("crypto", [])
        all_data = {}
        for sym in all_syms:
            try:
                d = fetch_yfinance([sym], start, end)
                if sym in d and not d[sym].empty:
                    all_data[sym] = d[sym]
            except Exception:
                pass
        return all_data
    # Use DataFeed with config universe
    feed = DataFeed(config)
    return feed.load_all(start=start, end=end)


def run_backtest(config: dict) -> None:
    from backtest.engine import BacktestEngine
    from backtest.reporter import generate_html_report, plot_results, save_metrics_json

    log.info("Mode: BACKTEST")
    bt_cfg = config.get("backtest", {})
    start = bt_cfg.get("start_date", "2018-01-01")
    end = bt_cfg.get("end_date", "2025-12-31")

    log.info(f"Fetching market data {start} → {end} ...")
    all_data = _load_all_data(config, start, end)

    # Fetch benchmark
    from data.feed import fetch_yfinance as _fyf

    bench_sym = bt_cfg.get("benchmark", "SPY")
    _bd = _fyf([bench_sym], start, end)
    bench_data = {bench_sym: _bd.get(bench_sym)}
    benchmark = bench_data.get(bench_sym)

    engine = BacktestEngine(config)
    metrics = engine.run(all_data, benchmark_data=benchmark)

    out_dir = config.get("system", {}).get("results_dir", "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    save_metrics_json(metrics, out_dir)
    chart_path = plot_results(metrics, out_dir)
    html_path = generate_html_report(metrics, out_dir, chart_path)

    log.info(f"\n{'=' * 55}")
    log.info(" Backtest complete!")
    log.info(f"  Chart : {chart_path}")
    log.info(f"  Report: {html_path}")
    log.info(f"  JSON  : {out_dir}/metrics.json")
    log.info(f"{'=' * 55}\n")
    _print_summary(metrics)


def run_comparison(config: dict) -> None:
    """Run baseline vs vol-targeted vs EWS side-by-side comparison."""
    import copy
    from pathlib import Path

    from backtest.engine import BacktestEngine
    from backtest.reporter import generate_comparison_report
    from data.feed import DataFeed

    log.info("Mode: COMPARISON (Baseline vs Vol-Targeted vs EWS+VT)")
    feed = DataFeed(config)
    bt_cfg = config.get("backtest", {})
    start = bt_cfg.get("start_date", "2018-01-01")
    end = bt_cfg.get("end_date", "2025-12-31")

    log.info(f"Fetching market data {start} → {end} ...")
    all_data = feed.load_all(start=start, end=end)
    bench_sym = bt_cfg.get("benchmark", "SPY")
    bench_data = feed.load([bench_sym], start=start, end=end, source="yfinance")
    benchmark = bench_data.get(bench_sym)

    out_dir = config.get("system", {}).get("results_dir", "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Run 1 — Baseline (no vol targeting, no EWS)
    cfg1 = copy.deepcopy(config)
    cfg1.setdefault("vol_targeting", {})["enabled"] = False
    cfg1.setdefault("ews", {})["enabled"] = False
    m_base = BacktestEngine(cfg1).run(all_data, benchmark, run_label="Baseline")

    # Run 2 — Vol targeting only
    cfg2 = copy.deepcopy(config)
    cfg2.setdefault("vol_targeting", {})["enabled"] = True
    cfg2.setdefault("ews", {})["enabled"] = False
    m_vt = BacktestEngine(cfg2).run(all_data, benchmark, run_label="Vol Targeting")

    # Run 3 — Vol targeting + EWS (full system)
    cfg3 = copy.deepcopy(config)
    cfg3.setdefault("vol_targeting", {})["enabled"] = True
    cfg3.setdefault("ews", {})["enabled"] = True
    m_full = BacktestEngine(cfg3).run(all_data, benchmark, run_label="VT + EWS")

    # Print three-way comparison
    _print_three_way(m_base, m_vt, m_full)

    # Generate HTML report (baseline vs full)
    html = generate_comparison_report(m_base, m_full, out_dir)
    log.info(f"Comparison report: {html}")


def _print_three_way(base: dict, vt: dict, full: dict) -> None:
    """Print a three-way metric comparison to the log."""
    from utils.logger import get_logger

    l = get_logger("Compare")
    rows = [
        ("Total Return (%)", "total_return_pct", True),
        ("Ann. Return (%)", "ann_return_pct", True),
        ("Ann. Volatility (%)", "ann_volatility_pct", False),
        ("Sharpe Ratio", "sharpe_ratio", True),
        ("Sortino Ratio", "sortino_ratio", True),
        ("Calmar Ratio", "calmar_ratio", True),
        ("Max Drawdown (%)", "max_drawdown_pct", False),
        ("MDD Duration (days)", "max_drawdown_duration_days", False),
        ("Win Rate (%)", "win_rate_pct", True),
        ("VaR 99% (%)", "var_hist_99_pct", False),
        ("CVaR 99% (%)", "cvar_hist_99_pct", False),
        ("Omega Ratio", "omega_ratio", True),
    ]
    l.info("\n" + "=" * 80)
    l.info("  THREE-WAY COMPARISON")
    l.info("=" * 80)
    l.info(f"  {'Metric':<28} {'Baseline':>12} {'Vol Target':>12} {'VT+EWS':>12}")
    l.info("-" * 80)
    for label, key, hb in rows:
        bv = base.get(key)
        vv = vt.get(key)
        fv = full.get(key)
        if any(x is None for x in [bv, vv, fv]):
            continue
        try:
            l.info(f"  {label:<28} {bv:>12.4f} {vv:>12.4f} {fv:>12.4f}")
        except Exception:
            pass
    l.info("=" * 80)


def run_validation(config: dict) -> None:
    """
    Run the three-method overfitting validation suite for vol targeting:
      Method 1 — Expanding walk-forward (4 out-of-sample folds)
      Method 2 — Sensitivity analysis (7 target_vol values)
      Method 3 — Permutation test (500 shuffles, p-value)
    """
    import json
    from pathlib import Path

    from backtest.engine import BacktestEngine
    from backtest.wf_validator import run_full_validation
    from data.feed import DataFeed

    log.info("Mode: VALIDATE (vol targeting overfitting test)")
    feed = DataFeed(config)
    bt_cfg = config.get("backtest", {})
    start = bt_cfg.get("start_date", "2018-01-01")
    end = bt_cfg.get("end_date", "2025-12-31")

    log.info(f"Fetching market data {start} → {end} ...")
    all_data = feed.load_all(start=start, end=end)

    # First run a baseline backtest to get the equity return series
    import copy

    cfg_base = copy.deepcopy(config)
    cfg_base.setdefault("vol_targeting", {})["enabled"] = False
    cfg_base.setdefault("ews", {})["enabled"] = False
    metrics = BacktestEngine(cfg_base).run(all_data, run_label="Baseline for validation")
    returns = metrics.get("returns")

    if returns is None or returns.empty:
        log.error("No returns from backtest — cannot validate")
        return

    target_vol = config.get("vol_targeting", {}).get("target_vol", 0.15)
    summary = run_full_validation(returns, target_vol=target_vol, n_permutations=500)

    # Save results
    out_dir = config.get("system", {}).get("results_dir", "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Walk-forward table
    wf_path = Path(out_dir) / "validation_walkforward.csv"
    summary["method1_wf_details"].to_csv(wf_path)
    log.info(f"Walk-forward results saved: {wf_path}")

    # Sensitivity table
    sens_path = Path(out_dir) / "validation_sensitivity.csv"
    summary["method2_sens_details"].to_csv(sens_path)
    log.info(f"Sensitivity results saved: {sens_path}")

    # Permutation summary
    perm_path = Path(out_dir) / "validation_permutation.json"
    perm_clean = {k: v for k, v in summary["method3_details"].items()}
    with open(perm_path, "w") as f:
        json.dump(perm_clean, f, indent=2)
    log.info(f"Permutation test saved: {perm_path}")

    print("\n" + "=" * 65)
    print("  OVERFITTING VALIDATION RESULTS")
    print("=" * 65)
    print(
        f"  Method 1 Walk-Forward:   {'PASS ✓' if summary['method1_wf_pass'] else 'FAIL ✗'}  "
        f"({summary['method1_folds_improved']} folds improved)"
    )
    print(
        f"  Method 2 Sensitivity:    {'PASS ✓' if summary['method2_sens_pass'] else 'FAIL ✗'}  "
        f"({summary['method2_beat_baseline']} target vols beat baseline)"
    )
    print(
        f"  Method 3 Permutation:    {'PASS ✓' if summary['method3_perm_pass'] else 'FAIL ✗'}  "
        f"(p={summary['method3_p_value']:.4f})"
    )
    print(f"  OVERALL: {summary['verdict']}")
    print("=" * 65)
    print(f"\n  Walk-forward table: {wf_path}")
    print(f"  Sensitivity table:  {sens_path}")
    print(f"  Permutation JSON:   {perm_path}")


def run_paper(config: dict) -> None:
    import copy

    cfg = copy.deepcopy(config)
    cfg["system"]["mode"] = "paper"
    log.info("Mode: PAPER TRADING")
    from execution.live_engine import LiveEngine

    engine = LiveEngine(cfg)
    engine.start(loop_interval_seconds=300)  # check every 5 minutes


def run_live(config: dict) -> None:
    import copy

    cfg = copy.deepcopy(config)
    cfg["system"]["mode"] = "live"
    log.warning("⚠  Mode: LIVE TRADING WITH REAL MONEY ⚠")
    log.warning("Ensure broker credentials are set in env vars / config.")
    confirm = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
    if confirm.strip() != "CONFIRM LIVE TRADING":
        log.info("Aborted.")
        return
    from execution.live_engine import LiveEngine

    engine = LiveEngine(cfg)
    engine.start(loop_interval_seconds=60)


def run_signals(config: dict) -> None:
    from datetime import datetime, timedelta

    from data.feed import DataFeed
    from strategy.signals import SignalGenerator

    log.info("Mode: SIGNALS")
    feed = DataFeed(config)
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    all_data = feed.load_all(start=start, end=end)

    gen = SignalGenerator(config)
    signals = gen.generate_latest(all_data)

    print("\n" + "=" * 50)
    print("  CURRENT SIGNALS")
    print("=" * 50)
    for sym, sig in sorted(signals.items(), key=lambda x: -abs(x[1])):
        direction = "LONG " if sig > 0 else "SHORT" if sig < 0 else "FLAT "
        bar = "█" * int(abs(sig) * 20)
        print(f"  {sym:<12} {direction}  {sig:+.3f}  {bar}")
    print("=" * 50)


def _print_summary(metrics: dict) -> None:
    keys = [
        ("Total Return", "total_return_pct", "%"),
        ("Ann. Return", "ann_return_pct", "%"),
        ("Ann. Volatility", "ann_volatility_pct", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Sortino Ratio", "sortino_ratio", ""),
        ("Calmar Ratio", "calmar_ratio", ""),
        ("Max Drawdown", "max_drawdown_pct", "%"),
        ("MDD Duration", "max_drawdown_duration_days", " days"),
        ("Win Rate", "win_rate_pct", "%"),
        ("VaR 99% (hist)", "var_hist_99_pct", "%"),
        ("CVaR 99% (hist)", "cvar_hist_99_pct", "%"),
        ("Skewness", "skewness", ""),
        ("Excess Kurtosis", "excess_kurtosis", ""),
        ("Omega Ratio", "omega_ratio", ""),
        ("Tail Ratio", "tail_ratio", ""),
        ("Alpha (ann.)", "alpha_ann_pct", "%"),
        ("Beta", "beta", ""),
        ("Information Ratio", "information_ratio", ""),
    ]
    print("\n" + "=" * 55)
    print("  BACKTEST SUMMARY")
    print("=" * 55)
    for label, key, unit in keys:
        v = metrics.get(key)
        if v is not None:
            try:
                print(f"  {label:<25} {v:>10.4f}{unit}")
            except Exception:
                print(f"  {label:<25} {v}{unit}")
    print("=" * 55)

    stress = metrics.get("stress_scenarios", {})
    if stress:
        print("\n  STRESS SCENARIOS (scaled to strategy vol):")
        for name, impact in stress.items():
            print(f"  {name[:55]:<55}  {impact * 100:>+7.1f}%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading System")
    parser.add_argument(
        "mode",
        choices=["backtest", "paper", "live", "signals", "report", "compare", "validate"],
        help="Execution mode (validate = walk-forward overfitting test for vol targeting)",
    )
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--start", default=None, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Backtest end date YYYY-MM-DD")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.start:
        config.setdefault("backtest", {})["start_date"] = args.start
    if args.end:
        config.setdefault("backtest", {})["end_date"] = args.end

    dispatch = {
        "backtest": run_backtest,
        "paper": run_paper,
        "live": run_live,
        "signals": run_signals,
        "compare": run_comparison,
        "validate": run_validation,
        "report": lambda c: log.info("Re-run backtest to regenerate report"),
    }
    dispatch[args.mode](config)


if __name__ == "__main__":
    main()
