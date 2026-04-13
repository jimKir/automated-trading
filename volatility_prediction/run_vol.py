#!/usr/bin/env python3
"""
run_vol.py — Production volatility prediction runner
=====================================================

Run:
    python run_vol.py                      # default: 5d horizon, full universe
    python run_vol.py --horizon 1d         # 1-day ahead
    python run_vol.py --horizon 10d        # 10-day ahead
    python run_vol.py --symbols 30         # top 30 symbols only
    python run_vol.py --no-lstm            # skip LSTM (faster)
    python run_vol.py --sector Tech        # single sector
    python run_vol.py --epochs 150         # more LSTM training
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vol_engine import (
    HORIZONS,
    SECTOR_MAP,
    UNIVERSE,
    VolatilityPredictor,
)

# ===========================================================================
# Terminal output styling
# ===========================================================================
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"

REGIME_COLOURS = {
    "HIGH_VOL": RED,
    "ELEVATED": YELLOW,
    "NORMAL": GREEN,
    "LOW": CYAN,
    "COMPRESSED": GREY,
}

DIRECTION_COLOURS = {
    "EXPANDING": RED,
    "CONTRACTING": GREEN,
    "STABLE": CYAN,
    "UNKNOWN": GREY,
}

DIRECTION_ARROWS = {
    "EXPANDING": "▲",
    "CONTRACTING": "▼",
    "STABLE": "─",
    "UNKNOWN": "?",
}


def print_banner(result: dict):
    """Print header banner with run summary."""
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    n = result.get("n_symbols", 0)
    horizon = result.get("horizon", "5d")

    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  VOLATILITY PREDICTOR  |  {ts}  |  {n} symbols  |  horizon: {horizon}{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")

    # Ensemble weights
    weights = result.get("ensemble_weights", {})
    if weights:
        w_str = "  ".join(f"{m}: {w:.0%}" for m, w in sorted(weights.items()))
        print(f"  Ensemble weights: {w_str}")

    print()


def print_sector_summary(result: dict):
    """Print sector-level volatility summary."""
    sectors = result.get("sector_summary", {})
    if not sectors:
        return

    print(
        f"{BOLD}  {'SECTOR':<12} {'#':>3}  {'CURRENT':>9}  {'PREDICTED':>9}  {'DIRECTION':<14}{RESET}"
    )
    print(f"  {'─' * 55}")

    for sec in sorted(sectors.keys()):
        info = sectors[sec]
        n = info.get("n_stocks", 0)
        cur = info.get("avg_current_vol")
        pred = info.get("avg_predicted_vol")
        d = info.get("dominant_direction", "UNKNOWN")
        dc = DIRECTION_COLOURS.get(d, GREY)
        arrow = DIRECTION_ARROWS.get(d, "?")

        cur_str = f"{cur:.1f}%" if cur else "  N/A"
        pred_str = f"{pred:.1f}%" if pred else "  N/A"

        print(f"  {sec:<12} {n:>3}  {cur_str:>9}  {pred_str:>9}  {dc}{arrow} {d}{RESET}")

    print()


def print_stock_table(result: dict, top_n: int = 20):
    """Print per-stock volatility predictions table."""
    latest = result.get("latest", {})
    if not latest:
        return

    # Sort by predicted vol (highest first — most volatile)
    sorted_stocks = sorted(
        latest.values(),
        key=lambda x: x.get("predicted_vol") or 0,
        reverse=True,
    )

    print(f"{BOLD}  TOP {top_n} BY PREDICTED VOLATILITY{RESET}")
    print(f"  {'─' * 78}")
    print(
        f"  {'SYMBOL':<8} {'SECTOR':<9} {'CURRENT':>9} {'PREDICTED':>10} {'CHANGE':>8} "
        f"{'PCTL':>5} {'REGIME':<12} {'DIR':<12}{RESET}"
    )
    print(f"  {'─' * 78}")

    for info in sorted_stocks[:top_n]:
        sym = info["symbol"]
        sec = info["sector"][:8]
        cur = info.get("current_vol")
        pred = info.get("predicted_vol")
        chg = info.get("vol_change_pct", 0)
        pctl = info.get("vol_percentile", 50)
        regime = info.get("regime", "UNKNOWN")
        direction = info.get("direction", "UNKNOWN")

        cur_str = f"{cur:.1f}%" if cur else "N/A"
        pred_str = f"{pred:.1f}%" if pred else "N/A"
        chg_str = f"{chg:+.1f}%"
        chg_colour = RED if chg > 5 else GREEN if chg < -5 else GREY

        rc = REGIME_COLOURS.get(regime, GREY)
        dc = DIRECTION_COLOURS.get(direction, GREY)
        arrow = DIRECTION_ARROWS.get(direction, "?")

        print(
            f"  {sym:<8} {sec:<9} {cur_str:>9} {pred_str:>10} "
            f"{chg_colour}{chg_str:>8}{RESET} {pctl:>4.0f}% "
            f"{rc}{regime:<12}{RESET} {dc}{arrow} {direction}{RESET}"
        )

    # Summary stats
    all_preds = [v["predicted_vol"] for v in latest.values() if v.get("predicted_vol")]
    if all_preds:
        expanding = sum(1 for v in latest.values() if v.get("direction") == "EXPANDING")
        contracting = sum(1 for v in latest.values() if v.get("direction") == "CONTRACTING")
        stable = sum(1 for v in latest.values() if v.get("direction") == "STABLE")
        print(
            f"\n  {BOLD}Summary:{RESET} avg predicted vol {sum(all_preds) / len(all_preds):.1f}%  |  "
            f"{RED}▲ {expanding}{RESET}  {GREEN}▼ {contracting}{RESET}  {CYAN}─ {stable}{RESET}"
        )

    print()


def print_model_metrics(result: dict):
    """Print aggregate model performance metrics."""
    all_metrics = result.get("metrics", {})
    if not all_metrics:
        return

    # Average metrics across all symbols
    model_names = set()
    for sym_metrics in all_metrics.values():
        model_names.update(sym_metrics.keys())

    print(f"{BOLD}  MODEL PERFORMANCE (avg across symbols){RESET}")
    print(f"  {'─' * 60}")
    print(f"  {'MODEL':<12} {'RMSE':>8} {'MAE':>8} {'QLIKE':>8} {'R²':>8} {'MZ-β':>8}{RESET}")
    print(f"  {'─' * 60}")

    for model in sorted(model_names):
        rmses, maes, qlikes, r2s, mz_betas = [], [], [], [], []
        for sym_metrics in all_metrics.values():
            m = sym_metrics.get(model, {})
            if m.get("rmse") and not __import__("math").isnan(m["rmse"]):
                rmses.append(m["rmse"])
            if m.get("mae") and not __import__("math").isnan(m["mae"]):
                maes.append(m["mae"])
            if m.get("qlike") and not __import__("math").isnan(m["qlike"]):
                qlikes.append(m["qlike"])
            if m.get("r2") and not __import__("math").isnan(m["r2"]):
                r2s.append(m["r2"])
            if m.get("mz_beta") and not __import__("math").isnan(m["mz_beta"]):
                mz_betas.append(m["mz_beta"])

        rmse = f"{sum(rmses) / len(rmses):.4f}" if rmses else "N/A"
        mae = f"{sum(maes) / len(maes):.4f}" if maes else "N/A"
        qlike = f"{sum(qlikes) / len(qlikes):.3f}" if qlikes else "N/A"
        r2 = f"{sum(r2s) / len(r2s):.3f}" if r2s else "N/A"
        mz = f"{sum(mz_betas) / len(mz_betas):.3f}" if mz_betas else "N/A"

        # Highlight best model
        marker = " ★" if model == "Ensemble" else ""
        print(f"  {model:<12} {rmse:>8} {mae:>8} {qlike:>8} {r2:>8} {mz:>8}{CYAN}{marker}{RESET}")

    print(f"\n  {GREY}MZ-β close to 1.0 = unbiased forecasts  |  QLIKE = standard vol loss{RESET}")
    print()


def print_feature_importance(result: dict, top_n: int = 15):
    """Print top feature importances from GBM."""
    fi = result.get("feature_importance")
    if fi is None or len(fi) == 0:
        return

    print(f"{BOLD}  TOP {top_n} FEATURES (GBM gain importance){RESET}")
    print(f"  {'─' * 45}")

    total = fi.sum()
    for feat, imp in fi.head(top_n).items():
        pct = imp / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
        print(f"  {feat:<22} {bar} {pct:>5.1f}%")

    print()


def print_alerts(result: dict):
    """Print actionable volatility alerts."""
    latest = result.get("latest", {})
    alerts = []

    for sym, info in latest.items():
        # High vol expanding — danger
        if info.get("regime") == "HIGH_VOL" and info.get("direction") == "EXPANDING":
            alerts.append((RED, "⚠", sym, "HIGH vol + EXPANDING — consider reducing position"))
        # Compressed vol expanding — potential breakout
        elif info.get("regime") in ("COMPRESSED", "LOW") and info.get("direction") == "EXPANDING":
            alerts.append((YELLOW, "◆", sym, "Compressed vol expanding — potential breakout"))
        # Very high predicted vol
        elif info.get("predicted_vol") and info["predicted_vol"] > 60:
            alerts.append((RED, "⚠", sym, f"Extreme predicted vol: {info['predicted_vol']:.1f}%"))

    if not alerts:
        print(f"  {GREEN}✓ No volatility alerts — all positions within normal range{RESET}\n")
        return

    print(f"{BOLD}  VOLATILITY ALERTS{RESET}")
    print(f"  {'─' * 65}")
    for colour, icon, sym, msg in alerts[:10]:
        print(f"  {colour}{icon} {sym:<8}{RESET} {msg}")
    print()


# ===========================================================================
# HTML Dashboard
# ===========================================================================


def build_dashboard(result: dict) -> str:
    """Build interactive HTML dashboard with Chart.js visualisations."""

    latest = result.get("latest", {})
    sectors = result.get("sector_summary", {})
    metrics = result.get("metrics", {})
    fi = result.get("feature_importance")
    weights = result.get("ensemble_weights", {})
    horizon = result.get("horizon", "5d")
    timestamp = result.get("timestamp", "")

    # Prepare data for charts
    sorted_stocks = sorted(
        latest.values(),
        key=lambda x: x.get("predicted_vol") or 0,
        reverse=True,
    )[:30]

    stock_labels = [s["symbol"] for s in sorted_stocks]
    current_vols = [s.get("current_vol") or 0 for s in sorted_stocks]
    predicted_vols = [s.get("predicted_vol") or 0 for s in sorted_stocks]

    sector_labels = sorted(sectors.keys())
    sector_current = [sectors[s].get("avg_current_vol") or 0 for s in sector_labels]
    sector_predicted = [sectors[s].get("avg_predicted_vol") or 0 for s in sector_labels]

    # Feature importance data
    fi_labels = []
    fi_values = []
    if fi is not None and len(fi) > 0:
        top_fi = fi.head(15)
        fi_labels = top_fi.index.tolist()
        fi_values = [round(v, 2) for v in top_fi.values]

    # Model metrics (averaged)
    model_metrics_data = {}
    for sym_m in metrics.values():
        for model, m in sym_m.items():
            if model not in model_metrics_data:
                model_metrics_data[model] = {"r2": [], "rmse": [], "qlike": []}
            for k in ["r2", "rmse", "qlike"]:
                v = m.get(k)
                if v is not None and not __import__("math").isnan(v):
                    model_metrics_data[model][k].append(v)

    model_labels = sorted(model_metrics_data.keys())
    model_r2 = [
        round(__import__("statistics").mean(model_metrics_data[m]["r2"]), 3)
        if model_metrics_data[m]["r2"]
        else 0
        for m in model_labels
    ]
    [
        round(__import__("statistics").mean(model_metrics_data[m]["rmse"]), 4)
        if model_metrics_data[m]["rmse"]
        else 0
        for m in model_labels
    ]

    # Scatter data: current vs predicted
    scatter_data = []
    for info in latest.values():
        if info.get("current_vol") and info.get("predicted_vol"):
            scatter_data.append(
                {
                    "x": info["current_vol"],
                    "y": info["predicted_vol"],
                    "label": info["symbol"],
                    "sector": info["sector"],
                }
            )

    # Colour map for sectors
    sector_colours = {
        "Tech": "#3b82f6",
        "Energy": "#f59e0b",
        "Financials": "#10b981",
        "Health": "#ef4444",
        "ConDisc": "#8b5cf6",
        "ConStap": "#06b6d4",
        "Indust": "#f97316",
        "REIT": "#ec4899",
        "Util": "#6b7280",
        "Comm": "#14b8a6",
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Volatility Prediction Dashboard | {horizon}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }}
  .header {{ text-align: center; margin-bottom: 30px; }}
  .header h1 {{ font-size: 28px; color: #f8fafc; }}
  .header .sub {{ color: #94a3b8; font-size: 14px; margin-top: 5px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px;
           border: 1px solid #334155; }}
  .card h3 {{ color: #94a3b8; font-size: 13px; text-transform: uppercase;
              letter-spacing: 1px; margin-bottom: 15px; }}
  .full-width {{ grid-column: 1 / -1; }}
  .metric-row {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }}
  .metric {{ background: #1e293b; border-radius: 10px; padding: 15px 20px;
             border: 1px solid #334155; flex: 1; min-width: 120px; text-align: center; }}
  .metric .value {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
  .metric .label {{ font-size: 11px; color: #64748b; text-transform: uppercase;
                    letter-spacing: 0.5px; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: #64748b; font-weight: 600; padding: 8px 10px;
       border-bottom: 1px solid #334155; font-size: 11px; text-transform: uppercase; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #1e293b; }}
  tr:hover td {{ background: #334155; }}
  .vol-high {{ color: #ef4444; }}
  .vol-elevated {{ color: #f59e0b; }}
  .vol-normal {{ color: #22c55e; }}
  .vol-low {{ color: #06b6d4; }}
  .vol-compressed {{ color: #6b7280; }}
  .dir-expanding {{ color: #ef4444; }}
  .dir-contracting {{ color: #22c55e; }}
  .dir-stable {{ color: #06b6d4; }}
  .weights {{ display: flex; gap: 10px; justify-content: center; margin-bottom: 20px; }}
  .weight-badge {{ background: #334155; padding: 6px 14px; border-radius: 20px;
                   font-size: 12px; color: #e2e8f0; }}
  canvas {{ max-height: 350px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Volatility Prediction Dashboard</h1>
  <div class="sub">Horizon: {horizon} | {len(latest)} symbols | {timestamp[:19]}</div>
  <div class="weights" style="margin-top:12px">
    {"".join(f'<span class="weight-badge">{m}: {w:.0%}</span>' for m, w in weights.items())}
  </div>
</div>

<!-- Summary metrics -->
<div class="metric-row">
  <div class="metric">
    <div class="value">{len(latest)}</div>
    <div class="label">Symbols Analysed</div>
  </div>
  <div class="metric">
    <div class="value">{sum(1 for v in latest.values() if v.get("direction") == "EXPANDING")}</div>
    <div class="label">Vol Expanding</div>
  </div>
  <div class="metric">
    <div class="value">{sum(1 for v in latest.values() if v.get("direction") == "CONTRACTING")}</div>
    <div class="label">Vol Contracting</div>
  </div>
  <div class="metric">
    <div class="value">{sum(1 for v in latest.values() if v.get("regime") == "HIGH_VOL")}</div>
    <div class="label">High Vol Regime</div>
  </div>
  <div class="metric">
    <div class="value">{round(sum(v.get("predicted_vol", 0) or 0 for v in latest.values()) / max(len(latest), 1), 1)}%</div>
    <div class="label">Avg Predicted Vol</div>
  </div>
</div>

<div class="grid">
  <!-- Stock bar chart -->
  <div class="card full-width">
    <h3>Current vs Predicted Volatility (Top 30)</h3>
    <canvas id="stockBarChart"></canvas>
  </div>

  <!-- Sector comparison -->
  <div class="card">
    <h3>Sector Volatility</h3>
    <canvas id="sectorChart"></canvas>
  </div>

  <!-- Scatter: current vs predicted -->
  <div class="card">
    <h3>Current vs Predicted (scatter)</h3>
    <canvas id="scatterChart"></canvas>
  </div>

  <!-- Model R² comparison -->
  <div class="card">
    <h3>Model R² (out-of-sample)</h3>
    <canvas id="modelChart"></canvas>
  </div>

  <!-- Feature importance -->
  <div class="card">
    <h3>Feature Importance (GBM)</h3>
    <canvas id="featureChart"></canvas>
  </div>
</div>

<!-- Full stock table -->
<div class="card full-width">
  <h3>All Predictions</h3>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Sector</th><th>Current Vol</th><th>Predicted Vol</th>
        <th>Change</th><th>Percentile</th><th>Regime</th><th>Direction</th>
      </tr>
    </thead>
    <tbody>
"""

    for info in sorted(latest.values(), key=lambda x: x.get("predicted_vol") or 0, reverse=True):
        regime_cls = f"vol-{info.get('regime', '').lower().replace('_', '')}"
        if regime_cls == "vol-highvol":
            regime_cls = "vol-high"
        dir_cls = f"dir-{info.get('direction', '').lower()}"
        cur = f"{info['current_vol']:.1f}%" if info.get("current_vol") else "N/A"
        pred = f"{info['predicted_vol']:.1f}%" if info.get("predicted_vol") else "N/A"
        chg = f"{info.get('vol_change_pct', 0):+.1f}%"
        chg_cls = (
            "vol-high"
            if info.get("vol_change_pct", 0) > 5
            else "vol-normal"
            if info.get("vol_change_pct", 0) < -5
            else ""
        )

        html += f"""      <tr>
        <td><strong>{info["symbol"]}</strong></td>
        <td>{info["sector"]}</td>
        <td>{cur}</td>
        <td><strong>{pred}</strong></td>
        <td class="{chg_cls}">{chg}</td>
        <td>{info.get("vol_percentile", 50):.0f}%</td>
        <td class="{regime_cls}">{info.get("regime", "")}</td>
        <td class="{dir_cls}">{DIRECTION_ARROWS.get(info.get("direction", ""), "?")} {info.get("direction", "")}</td>
      </tr>
"""

    html += f"""    </tbody>
  </table>
  </div>
</div>

<script>
const stockLabels = {json.dumps(stock_labels)};
const currentVols = {json.dumps(current_vols)};
const predictedVols = {json.dumps(predicted_vols)};
const sectorLabels = {json.dumps(sector_labels)};
const sectorCurrent = {json.dumps(sector_current)};
const sectorPredicted = {json.dumps(sector_predicted)};
const modelLabels = {json.dumps(model_labels)};
const modelR2 = {json.dumps(model_r2)};
const fiLabels = {json.dumps(fi_labels)};
const fiValues = {json.dumps(fi_values)};
const scatterData = {json.dumps(scatter_data)};
const sectorColours = {json.dumps(sector_colours)};

// Stock bar chart
new Chart(document.getElementById('stockBarChart'), {{
  type: 'bar',
  data: {{
    labels: stockLabels,
    datasets: [
      {{ label: 'Current Vol (%)', data: currentVols, backgroundColor: 'rgba(59,130,246,0.6)' }},
      {{ label: 'Predicted Vol (%)', data: predictedVols, backgroundColor: 'rgba(239,68,68,0.6)' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b', maxRotation: 90 }} }},
      y: {{ title: {{ display: true, text: 'Annualised Vol (%)', color: '#64748b' }},
            ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Sector chart
new Chart(document.getElementById('sectorChart'), {{
  type: 'bar',
  data: {{
    labels: sectorLabels,
    datasets: [
      {{ label: 'Current', data: sectorCurrent, backgroundColor: 'rgba(59,130,246,0.7)' }},
      {{ label: 'Predicted', data: sectorPredicted, backgroundColor: 'rgba(239,68,68,0.7)' }},
    ]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b' }} }}
    }}
  }}
}});

// Scatter plot
new Chart(document.getElementById('scatterChart'), {{
  type: 'scatter',
  data: {{
    datasets: Object.keys(sectorColours).map(sec => ({{
      label: sec,
      data: scatterData.filter(d => d.sector === sec).map(d => ({{x: d.x, y: d.y}})),
      backgroundColor: sectorColours[sec] || '#6b7280',
      pointRadius: 5,
    }})).filter(ds => ds.data.length > 0)
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', boxWidth: 10 }}, position: 'right' }},
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{
            const d = scatterData.filter(d => d.sector === ctx.dataset.label)[ctx.dataIndex];
            return d ? d.label + ': ' + d.x + '% → ' + d.y + '%' : '';
          }}
        }}
      }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Current Vol (%)', color: '#64748b' }},
            ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ title: {{ display: true, text: 'Predicted Vol (%)', color: '#64748b' }},
            ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Model R² chart
new Chart(document.getElementById('modelChart'), {{
  type: 'bar',
  data: {{
    labels: modelLabels,
    datasets: [{{ label: 'R²', data: modelR2,
      backgroundColor: modelLabels.map(m => m === 'Ensemble' ? 'rgba(34,197,94,0.7)' : 'rgba(99,102,241,0.5)') }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ title: {{ display: true, text: 'R²', color: '#64748b' }},
            ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
      x: {{ ticks: {{ color: '#64748b' }} }}
    }}
  }}
}});

// Feature importance chart
new Chart(document.getElementById('featureChart'), {{
  type: 'bar',
  data: {{
    labels: fiLabels,
    datasets: [{{ label: 'Importance', data: fiValues,
      backgroundColor: 'rgba(251,146,60,0.6)' }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }} }}
    }}
  }}
}});
</script>

</body>
</html>"""

    return html


# ===========================================================================
# JSON export
# ===========================================================================


def save_json(result: dict, path: str):
    """Save predictions and metrics to JSON."""
    export = {
        "timestamp": result.get("timestamp"),
        "horizon": result.get("horizon"),
        "n_symbols": result.get("n_symbols"),
        "ensemble_weights": result.get("ensemble_weights"),
        "predictions": {},
        "sector_summary": result.get("sector_summary"),
    }

    for sym, info in result.get("latest", {}).items():
        metrics = result.get("metrics", {}).get(sym, {})
        export["predictions"][sym] = {
            **info,
            "metrics": {
                m: {k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()}
                for m, met in metrics.items()
            },
        }

    with open(path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    logger.info(f"JSON saved: {path}")


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Volatility Prediction Runner")
    parser.add_argument(
        "--horizon",
        default="5d",
        choices=list(HORIZONS.keys()),
        help="Prediction horizon (1d, 5d, 10d)",
    )
    parser.add_argument("--symbols", type=int, default=0, help="Number of symbols (0 = all)")
    parser.add_argument(
        "--sector", type=str, default="", help="Filter to single sector (e.g. Tech, Energy)"
    )
    parser.add_argument("--top", type=int, default=25, help="Show top N in terminal output")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM model (faster)")
    parser.add_argument("--no-gbm", action="store_true", help="Skip GBM model")
    parser.add_argument("--epochs", type=int, default=80, help="LSTM training epochs")
    parser.add_argument("--lookback", type=float, default=3.0, help="Years of historical data")
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output directory for dashboard and JSON"
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Bypass data cache and re-download everything"
    )
    args = parser.parse_args()

    # Build symbol list
    symbols = UNIVERSE
    if args.sector:
        symbols = [s for s, sec in SECTOR_MAP.items() if sec == args.sector]
        if not symbols:
            print(f"Unknown sector: {args.sector}")
            print(f"Available: {', '.join(sorted(set(SECTOR_MAP.values())))}")
            sys.exit(1)
    if args.symbols > 0:
        symbols = symbols[: args.symbols]

    # Run predictor
    predictor = VolatilityPredictor(
        horizon=args.horizon,
        use_lstm=not args.no_lstm,
        use_gbm=not args.no_gbm,
        lstm_epochs=args.epochs,
    )

    result = predictor.run(
        symbols=symbols,
        lookback_years=args.lookback,
        force_refresh=args.force_refresh,
    )

    if "error" in result:
        print(f"\n{RED}  ✗ {result['error']}{RESET}\n")
        sys.exit(1)

    # Terminal output
    print_banner(result)
    print_sector_summary(result)
    print_stock_table(result, args.top)
    print_model_metrics(result)
    print_feature_importance(result)
    print_alerts(result)

    # Save outputs
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Dashboard
    dashboard_path = os.path.join(out_dir, "vol_prediction_dashboard.html")
    html = build_dashboard(result)
    with open(dashboard_path, "w") as f:
        f.write(html)
    print(f"  {GREEN}✓{RESET} Dashboard saved: {dashboard_path}")

    # JSON
    json_path = os.path.join(out_dir, "vol_prediction_report.json")
    save_json(result, json_path)
    print(f"  {GREEN}✓{RESET} JSON report saved: {json_path}")

    print(f"\n{BOLD}{'=' * 78}{RESET}\n")


if __name__ == "__main__":
    main()
