"""
Backtest Reporter
=================
Generates a rich HTML + JSON performance report from backtest results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from utils.logger import get_logger

log = get_logger("Reporter")


def _safe(v: Any) -> Any:
    """JSON-safe conversion."""
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, pd.Series):
        return v.to_dict()
    return v


def save_metrics_json(metrics: dict, output_dir: str = "results") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    clean = {
        k: _safe(v) for k, v in metrics.items() if not isinstance(v, (pd.Series, pd.DataFrame))
    }
    out = Path(output_dir) / "metrics.json"
    with open(out, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    log.info(f"Metrics saved to {out}")
    return str(out)


def plot_results(
    metrics: dict,
    output_dir: str = "results",
    title: str = "Trading Strategy Backtest",
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    equity = metrics.get("equity_curve")
    returns = metrics.get("returns")

    if equity is None or returns is None:
        log.warning("No equity/returns data to plot")
        return ""

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- 1. Equity curve ---
    ax1 = fig.add_subplot(gs[0, :])
    equity_norm = equity / equity.iloc[0] * 100
    ax1.plot(
        equity_norm.index, equity_norm.values, color="#1f77b4", linewidth=1.8, label="Strategy"
    )
    ax1.fill_between(equity_norm.index, 100, equity_norm.values, alpha=0.15, color="#1f77b4")
    ax1.axhline(100, color="gray", linestyle="--", linewidth=0.8, label="Breakeven")
    ax1.set_title("Equity Curve (normalised to 100)", fontweight="bold")
    ax1.set_ylabel("Normalised Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 2. Drawdown ---
    ax2 = fig.add_subplot(gs[1, :])
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.6)
    ax2.set_title("Drawdown (%)", fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # --- 3. Rolling Sharpe (126-day) ---
    ax3 = fig.add_subplot(gs[2, 0])
    roll_sharpe = returns.rolling(126).apply(
        lambda r: (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    )
    ax3.plot(roll_sharpe.index, roll_sharpe.values, color="#2ca02c", linewidth=1.2)
    ax3.axhline(1.0, color="orange", linestyle="--", linewidth=0.8, label="Sharpe=1")
    ax3.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax3.set_title("Rolling Sharpe Ratio (126d)", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- 4. Monthly returns heatmap ---
    ax4 = fig.add_subplot(gs[2, 1])
    try:
        monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1) * 100
        monthly.index = [d.strftime("%Y-%m") for d in monthly.index]
        ax4.bar(
            range(len(monthly)),
            monthly.values,
            color=["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values],
        )
        step = max(1, len(monthly) // 6)
        ax4.set_xticks(range(0, len(monthly), step))
        ax4.set_xticklabels(monthly.index[::step], rotation=45, fontsize=8, ha="right")
        ax4.set_title("Monthly Returns (%)", fontweight="bold")
        ax4.axhline(0, color="black", linewidth=0.8)
        ax4.grid(True, alpha=0.3)
    except Exception as e:
        ax4.text(
            0.5,
            0.5,
            f"Monthly chart error:\n{e}",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    # --- 5. Return distribution ---
    ax5 = fig.add_subplot(gs[3, 0])
    from scipy import stats as scipy_stats

    daily_pct = returns * 100
    ax5.hist(
        daily_pct.dropna(), bins=60, color="#1f77b4", alpha=0.7, density=True, label="Strategy"
    )
    x = np.linspace(daily_pct.min(), daily_pct.max(), 200)
    mu, sigma = daily_pct.mean(), daily_pct.std()
    ax5.plot(x, scipy_stats.norm.pdf(x, mu, sigma), color="orange", linewidth=2, label="Normal fit")
    ax5.axvline(0, color="red", linestyle="--", linewidth=1)
    var_line = -float(np.percentile(daily_pct.dropna(), 1))
    ax5.axvline(
        -var_line, color="purple", linestyle="--", linewidth=1, label=f"99% VaR: {var_line:.2f}%"
    )
    ax5.set_title("Daily Return Distribution", fontweight="bold")
    ax5.set_xlabel("Daily Return (%)")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- 6. Key metrics table ---
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis("off")
    scalar_metrics = [
        ("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%"),
        ("Ann. Return", f"{metrics.get('ann_return_pct', 0):.2f}%"),
        ("Ann. Volatility", f"{metrics.get('ann_volatility_pct', 0):.2f}%"),
        ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}"),
        ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}"),
        ("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}"),
        ("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ("MDD Duration", f"{metrics.get('max_drawdown_duration_days', 0)} days"),
        ("Win Rate", f"{metrics.get('win_rate_pct', 0):.1f}%"),
        ("VaR 99% (hist)", f"{metrics.get('var_hist_99_pct', 0):.2f}%"),
        ("CVaR 99% (hist)", f"{metrics.get('cvar_hist_99_pct', 0):.2f}%"),
        ("Skewness", f"{metrics.get('skewness', 0):.3f}"),
        ("Excess Kurtosis", f"{metrics.get('excess_kurtosis', 0):.3f}"),
        ("Omega Ratio", f"{metrics.get('omega_ratio', 0):.3f}"),
        ("Tail Ratio", f"{metrics.get('tail_ratio', 0):.3f}"),
    ]
    table_data = [[k, v] for k, v in scalar_metrics]
    tbl = ax6.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, _col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax6.set_title("Performance Metrics", fontweight="bold", pad=15)

    out_path = Path(output_dir) / "backtest_report.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Report chart saved to {out_path}")
    return str(out_path)


def generate_html_report(
    metrics: dict,
    output_dir: str = "results",
    chart_path: str = "",
) -> str:
    """Generate standalone HTML report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stress = metrics.get("stress_scenarios", {})
    stress_rows = "\n".join(
        f"<tr><td>{k}</td><td class='{'neg' if v < 0 else 'pos'}'>{v * 100:.1f}%</td></tr>"
        for k, v in stress.items()
    )

    initial_equity = metrics.get("final_equity", 25000) / max(
        1, 1 + metrics.get("total_return_pct", 0) / 100
    )
    cost_rows = "\n".join(
        f"<tr><td>{label}</td><td>${metrics.get(key, 0):,.2f}</td><td class='neg'>{metrics.get(key, 0) / max(initial_equity, 1) * 100:.3f}%</td></tr>"
        for label, key in [
            ("Commission", "cost_commission_total"),
            ("Bid-Ask Spread", "cost_spread_total"),
            ("Market Impact", "cost_market_impact_total"),
            ("Overnight Financing", "cost_overnight_financing_total"),
            ("Futures Roll", "cost_futures_roll_total"),
            ("Crypto Funding", "cost_crypto_funding_total"),
        ]
    )
    cost_total = metrics.get("cost_total", 0)
    cost_total_pct = metrics.get("cost_total_pct_of_initial", 0)

    scalar_metrics = {
        "Total Return (%)": f"{metrics.get('total_return_pct', 0):.2f}",
        "Annualised Return (%)": f"{metrics.get('ann_return_pct', 0):.2f}",
        "Annualised Volatility (%)": f"{metrics.get('ann_volatility_pct', 0):.2f}",
        "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.3f}",
        "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.3f}",
        "Calmar Ratio": f"{metrics.get('calmar_ratio', 0):.3f}",
        "Max Drawdown (%)": f"{metrics.get('max_drawdown_pct', 0):.2f}",
        "Max Drawdown Duration (days)": f"{metrics.get('max_drawdown_duration_days', 0)}",
        "Win Rate (%)": f"{metrics.get('win_rate_pct', 0):.1f}",
        "VaR 99% Historical (%)": f"{metrics.get('var_hist_99_pct', 0):.2f}",
        "CVaR 99% Historical (%)": f"{metrics.get('cvar_hist_99_pct', 0):.2f}",
        "VaR 99% Parametric (%)": f"{metrics.get('var_parametric_99_pct', 0):.2f}",
        "VaR 99% Monte Carlo (%)": f"{metrics.get('var_monte_carlo_99_pct', 0):.2f}",
        "Skewness": f"{metrics.get('skewness', 0):.4f}",
        "Excess Kurtosis": f"{metrics.get('excess_kurtosis', 0):.4f}",
        "Omega Ratio": f"{metrics.get('omega_ratio', 0):.4f}",
        "Tail Ratio": f"{metrics.get('tail_ratio', 0):.4f}",
        "Alpha (ann. %)": f"{metrics.get('alpha_ann_pct') or 'N/A'}",
        "Beta": f"{metrics.get('beta') or 'N/A'}",
        "Information Ratio": f"{metrics.get('information_ratio') or 'N/A'}",
        "Trading Days": f"{metrics.get('trading_days', 0)}",
        "Final Equity ($)": f"{metrics.get('final_equity', 0):,.2f}",
    }

    metric_rows = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in scalar_metrics.items())

    import base64

    chart_b64 = ""
    if chart_path and Path(chart_path).exists():
        with open(chart_path, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trading System — Backtest Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
  h2 {{ color: #79c0ff; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th {{ background: #161b22; color: #58a6ff; padding: 10px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #161b22; }}
  .pos {{ color: #3fb950; font-weight: bold; }}
  .neg {{ color: #f85149; font-weight: bold; }}
  .chart {{ width: 100%; margin: 20px 0; border-radius: 8px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .card {{ background: #161b22; border-radius: 8px; padding: 16px; border: 1px solid #30363d; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .badge-green {{ background: #1a4731; color: #3fb950; }}
  .badge-red {{ background: #3d1a1a; color: #f85149; }}
</style>
</head>
<body>
<h1>Trading System — Backtest Report</h1>
<p>Multi-Factor Momentum + Mean-Reversion | Equities • Crypto • Futures</p>

{"<img class='chart' src='data:image/png;base64," + chart_b64 + "' />" if chart_b64 else ""}

<h2>Performance Metrics</h2>
<div class="grid">
  <div class="card">
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      {metric_rows}
    </table>
  </div>
  <div class="card">
    <h2>Stress Test Scenarios</h2>
    <p style="font-size:12px;color:#8b949e;">Expected strategy impact under historical crash regimes</p>
    <table>
      <tr><th>Scenario</th><th>Expected Impact</th></tr>
      {stress_rows}
    </table>
  </div>
</div>

<h2>Cost Breakdown</h2>
<div class="card">
  <p style="font-size:12px;color:#8b949e;">Total frictional costs charged during backtest — all deducted from cash in real-time.</p>
  <table>
    <tr><th>Cost Component</th><th>Total ($)</th><th>% of Initial Equity</th></tr>
    {cost_rows}
    <tr style="border-top: 2px solid #58a6ff;"><td><strong>TOTAL ALL-IN COST</strong></td><td><strong>${cost_total:,.2f}</strong></td><td class='neg'><strong>{cost_total_pct:.2f}%</strong></td></tr>
  </table>
</div>

<p style="color:#8b949e;font-size:11px;margin-top:40px;">
  Generated by Trading System v1.0 | Risk warning: Past performance is not indicative of future results.
  This system is provided for research purposes. Always paper-trade before deploying real capital.
</p>
</body>
</html>"""

    out = Path(output_dir) / "backtest_report.html"
    with open(out, "w") as f:
        f.write(html)
    log.info(f"HTML report saved to {out}")
    return str(out)


def generate_comparison_report(
    base_metrics: dict,
    ews_metrics: dict,
    output_dir: str = "results",
) -> str:
    """Generate a side-by-side HTML comparison: Baseline vs With EWS."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _fmt(v, suffix=""):
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.4f}{suffix}"
        except Exception:
            return str(v)

    def _delta_class(base_v, ews_v, higher_better=True):
        try:
            delta = float(ews_v) - float(base_v)
            if higher_better:
                return "pos" if delta > 0 else ("neg" if delta < 0 else "")
            return "pos" if delta < 0 else ("neg" if delta > 0 else "")
        except Exception:
            return ""

    comparison_rows = [
        ("Total Return (%)", "total_return_pct", True, "%"),
        ("Ann. Return (%)", "ann_return_pct", True, "%"),
        ("Ann. Volatility (%)", "ann_volatility_pct", False, "%"),
        ("Sharpe Ratio", "sharpe_ratio", True, ""),
        ("Sortino Ratio", "sortino_ratio", True, ""),
        ("Calmar Ratio", "calmar_ratio", True, ""),
        ("Max Drawdown (%)", "max_drawdown_pct", False, "%"),
        ("MDD Duration (days)", "max_drawdown_duration_days", False, ""),
        ("Win Rate (%)", "win_rate_pct", True, "%"),
        ("VaR 99% (%)", "var_hist_99_pct", False, "%"),
        ("CVaR 99% (%)", "cvar_hist_99_pct", False, "%"),
        ("Skewness", "skewness", True, ""),
        ("Excess Kurtosis", "excess_kurtosis", False, ""),
        ("Omega Ratio", "omega_ratio", True, ""),
        ("Tail Ratio", "tail_ratio", True, ""),
        ("Alpha Ann. (%)", "alpha_ann_pct", True, "%"),
        ("Beta", "beta", False, ""),
        ("Information Ratio", "information_ratio", True, ""),
        ("Final Equity ($)", "final_equity", True, ""),
    ]

    table_rows = ""
    for label, key, higher_better, suffix in comparison_rows:
        bv = base_metrics.get(key)
        ev = ews_metrics.get(key)
        if bv is None and ev is None:
            continue
        try:
            delta = float(ev) - float(bv)
            delta_sign = "+" if delta > 0 else ""
            delta_cls = _delta_class(bv, ev, higher_better)
            better_icon = "✓" if (delta > 0) == higher_better else ("✗" if delta != 0 else "")
            table_rows += (
                f"<tr><td>{label}</td>"
                f"<td>{_fmt(bv, suffix)}</td>"
                f"<td>{_fmt(ev, suffix)}</td>"
                f"<td class='{delta_cls}'>{delta_sign}{delta:.4f}{suffix} {better_icon}</td></tr>\n"
            )
        except Exception:
            table_rows += (
                f"<tr><td>{label}</td><td>{_fmt(bv)}</td><td>{_fmt(ev)}</td><td>—</td></tr>\n"
            )

    # EWS regime breakdown
    ews_green = ews_metrics.get("ews_days_green", 0)
    ews_yellow = ews_metrics.get("ews_days_yellow", 0)
    ews_orange = ews_metrics.get("ews_days_orange", 0)
    ews_red = ews_metrics.get("ews_days_red", 0)
    ews_avg_sc = ews_metrics.get("ews_avg_scale", 1.0)
    total_days = ews_green + ews_yellow + ews_orange + ews_red or 1

    # Build equity comparison chart using base64-embedded Chart.js data
    eq_base = base_metrics.get("equity_curve")
    eq_ews = ews_metrics.get("equity_curve")
    chart_labels = chart_base_data = chart_ews_data = "[]"
    if eq_base is not None and eq_ews is not None:
        import json as _json

        common_idx = eq_base.index.intersection(eq_ews.index)
        labels_list = [str(d.date()) for d in common_idx[::5]]  # every 5 days
        base_norm = (eq_base.reindex(common_idx) / eq_base.iloc[0] * 100).iloc[::5]
        ews_norm = (eq_ews.reindex(common_idx) / eq_ews.iloc[0] * 100).iloc[::5]
        chart_labels = _json.dumps(labels_list)
        chart_base_data = _json.dumps([round(float(v), 2) for v in base_norm])
        chart_ews_data = _json.dumps([round(float(v), 2) for v in ews_norm])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EWS Comparison Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body   {{ font-family:'Segoe UI',sans-serif; background:#0d1117; color:#c9d1d9; margin:0; padding:20px; }}
  h1     {{ color:#58a6ff; border-bottom:1px solid #30363d; padding-bottom:10px; }}
  h2     {{ color:#79c0ff; margin-top:30px; }}
  table  {{ border-collapse:collapse; width:100%; margin-bottom:20px; }}
  th     {{ background:#161b22; color:#58a6ff; padding:10px; text-align:left; }}
  td     {{ padding:8px 12px; border-bottom:1px solid #21262d; }}
  tr:hover {{ background:#161b22; }}
  .pos   {{ color:#3fb950; font-weight:bold; }}
  .neg   {{ color:#f85149; font-weight:bold; }}
  .grid  {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:20px 0; }}
  .card  {{ background:#161b22; border-radius:8px; padding:16px; border:1px solid #30363d; }}
  .regime-bar {{ display:flex; height:28px; border-radius:6px; overflow:hidden; margin:12px 0; }}
  .seg-green  {{ background:#1a7431; }}
  .seg-yellow {{ background:#9a7e0a; }}
  .seg-orange {{ background:#9a4a0a; }}
  .seg-red    {{ background:#7a1a1a; }}
  canvas      {{ background:#161b22; border-radius:8px; }}
</style>
</head>
<body>
<h1>Backtest Comparison: Baseline vs Early Warning System (EWS)</h1>
<p style="color:#8b949e;">Strategy: Multi-Factor Momentum + Mean-Reversion | 2018–2025 | $25,000 initial capital</p>

<h2>Equity Curve Comparison</h2>
<canvas id="eqChart" height="60"></canvas>
<script>
new Chart(document.getElementById('eqChart'), {{
  type: 'line',
  data: {{
    labels: {chart_labels},
    datasets: [
      {{ label:'Baseline', data:{chart_base_data}, borderColor:'#f85149',
         backgroundColor:'rgba(248,81,73,0.05)', fill:true, tension:0.3, pointRadius:0 }},
      {{ label:'With EWS', data:{chart_ews_data},  borderColor:'#3fb950',
         backgroundColor:'rgba(63,185,80,0.05)',  fill:true, tension:0.3, pointRadius:0 }},
    ]
  }},
  options:{{
    plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',maxTicksLimit:12}}}},
      y:{{ticks:{{color:'#8b949e'}},title:{{display:true,text:'Normalised Value (100=start)',color:'#8b949e'}}}}
    }}
  }}
}});
</script>

<h2>Side-by-Side Metrics</h2>
<div class="card">
  <table>
    <tr><th>Metric</th><th>Baseline</th><th>With EWS</th><th>Delta</th></tr>
    {table_rows}
  </table>
</div>

<h2>EWS Regime Distribution</h2>
<div class="grid">
  <div class="card">
    <p style="color:#8b949e;font-size:13px;">Proportion of trading days in each EWS state</p>
    <div class="regime-bar">
      <div class="seg-green"  style="width:{ews_green / total_days * 100:.1f}%" title="GREEN {ews_green}d"></div>
      <div class="seg-yellow" style="width:{ews_yellow / total_days * 100:.1f}%" title="YELLOW {ews_yellow}d"></div>
      <div class="seg-orange" style="width:{ews_orange / total_days * 100:.1f}%" title="ORANGE {ews_orange}d"></div>
      <div class="seg-red"    style="width:{ews_red / total_days * 100:.1f}%"    title="RED {ews_red}d"></div>
    </div>
    <table>
      <tr><th>Regime</th><th>Scale</th><th>Days</th><th>% of period</th></tr>
      <tr><td style="color:#3fb950">GREEN  (full exposure)</td><td>100%</td><td>{ews_green}</td><td>{ews_green / total_days * 100:.1f}%</td></tr>
      <tr><td style="color:#d4b04a">YELLOW (trim)</td>        <td>70%</td> <td>{ews_yellow}</td><td>{ews_yellow / total_days * 100:.1f}%</td></tr>
      <tr><td style="color:#e07b30">ORANGE (reduce)</td>      <td>40%</td> <td>{ews_orange}</td><td>{ews_orange / total_days * 100:.1f}%</td></tr>
      <tr><td style="color:#f85149">RED/CRITICAL (defensive)</td><td>≤20%</td><td>{ews_red}</td><td>{ews_red / total_days * 100:.1f}%</td></tr>
      <tr style="border-top:2px solid #58a6ff"><td><strong>Average scale factor</strong></td><td colspan="3"><strong>{ews_avg_sc:.1%}</strong></td></tr>
    </table>
  </div>
  <div class="card">
    <h2 style="margin-top:0">How to Read This</h2>
    <p style="font-size:13px;color:#8b949e;">
      The EWS scales <em>max_position_pct</em> and <em>max_portfolio_heat</em> down
      when multiple stress layers fire simultaneously.
      A lower average scale factor means the system was more conservative on average
      — which reduces drawdowns at the cost of some upside in bull markets.
    </p>
    <p style="font-size:13px;color:#8b949e;">
      <strong style="color:#3fb950">GREEN delta metrics</strong> = EWS improved vs baseline.<br>
      <strong style="color:#f85149">RED delta metrics</strong> = EWS gave up some return.<br>
      The key trade-off: Max Drawdown ↓ vs Total Return ↓.
    </p>
  </div>
</div>

<p style="color:#8b949e;font-size:11px;margin-top:40px;">
  Generated by Trading System v1.0 | Past performance is not indicative of future results.
</p>
</body>
</html>"""

    out = Path(output_dir) / "ews_comparison_report.html"
    with open(out, "w") as f:
        f.write(html)
    log.info(f"Comparison report saved to {out}")
    return str(out)
