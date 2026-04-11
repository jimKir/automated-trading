#!/usr/bin/env python3
"""
Daily Report Generator
======================
Run once per day (e.g. via cron at market close) to:
  - Snapshot current paper portfolio P&L
  - Generate an HTML performance report
  - Optionally email it via AWS SES or save to S3

Usage:
    python daily_report.py                 # save to results/daily/
    python daily_report.py --email         # also email via AWS SES
    python daily_report.py --s3            # also upload to S3
    python daily_report.py --email --s3    # both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("DailyReport")

STATE_FILE = Path("results/paper_state.json")


def load_paper_state() -> dict:
    """Load persisted paper portfolio state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "equity_history": [],
        "positions": {},
        "cash": None,
        "start_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "initial_equity": None,
    }


def save_paper_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def snapshot_portfolio(config: dict, state: dict) -> dict:
    """Fetch current prices and compute live P&L."""
    from execution.paper_broker import PaperBroker

    broker = PaperBroker(config)
    # Restore persisted cash + positions if they exist
    if state.get("cash") is not None:
        broker.cash = state["cash"]
    if state.get("positions"):
        broker._positions = state["positions"]

    account = broker.get_account()
    initial = state.get("initial_equity") or config.get("capital", {}).get("initial_equity", 25000)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    equity_history = state.get("equity_history", [])
    equity_history.append({"date": today, "equity": account.equity})

    # Keep last 365 days
    equity_history = equity_history[-365:]

    state["equity_history"] = equity_history
    state["cash"] = broker.cash
    state["positions"] = broker._positions
    state["initial_equity"] = initial
    save_paper_state(state)

    pnl_total = account.equity - initial
    pnl_pct = pnl_total / initial * 100

    prev_equity = equity_history[-2]["equity"] if len(equity_history) >= 2 else initial
    pnl_daily = account.equity - prev_equity
    pnl_daily_pct = pnl_daily / prev_equity * 100

    # Peak / drawdown
    equities = [e["equity"] for e in equity_history]
    peak = max(equities)
    drawdown_pct = (account.equity - peak) / peak * 100

    return {
        "date": today,
        "equity": account.equity,
        "cash": account.cash,
        "positions": account.positions,
        "initial_equity": initial,
        "pnl_total": pnl_total,
        "pnl_total_pct": pnl_pct,
        "pnl_daily": pnl_daily,
        "pnl_daily_pct": pnl_daily_pct,
        "peak_equity": peak,
        "drawdown_pct": drawdown_pct,
        "equity_history": equity_history,
    }


def generate_daily_html(snap: dict, out_dir: str = "results/daily") -> str:
    """Generate a compact daily HTML report."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pos_rows = ""
    for sym, pos in snap["positions"].items():
        qty = pos.get("quantity", 0)
        avg = pos.get("avg_price", 0)
        pos_rows += f"<tr><td>{sym}</td><td>{qty:.4f}</td><td>${avg:.4f}</td></tr>\n"

    # Mini equity chart data
    dates = [e["date"] for e in snap["equity_history"]]
    equities = [e["equity"] for e in snap["equity_history"]]
    chart_labels = json.dumps(dates[-90:])  # last 90 days
    chart_data = json.dumps(equities[-90:])

    pnl_class = "pos" if snap["pnl_daily"] >= 0 else "neg"
    tot_class = "pos" if snap["pnl_total"] >= 0 else "neg"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Daily Trading Report — {snap["date"]}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; margin:0; padding:20px; }}
  h1  {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom:10px; }}
  h2  {{ color: #79c0ff; margin-top:30px; }}
  .grid  {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:20px 0; }}
  .card  {{ background:#161b22; border-radius:8px; padding:16px; border:1px solid #30363d; }}
  .label {{ font-size:12px; color:#8b949e; margin-bottom:4px; }}
  .value {{ font-size:22px; font-weight:bold; }}
  .pos   {{ color:#3fb950; }}
  .neg   {{ color:#f85149; }}
  table  {{ border-collapse:collapse; width:100%; }}
  th     {{ background:#161b22; color:#58a6ff; padding:10px; text-align:left; }}
  td     {{ padding:8px 12px; border-bottom:1px solid #21262d; }}
  canvas {{ background:#161b22; border-radius:8px; padding:12px; }}
</style>
</head>
<body>
<h1>Paper Trading — Daily Report</h1>
<p style="color:#8b949e;">{snap["date"]} UTC | Strategy: Multi-Factor Momentum + Mean-Reversion</p>

<div class="grid">
  <div class="card">
    <div class="label">Portfolio Equity</div>
    <div class="value">${snap["equity"]:,.2f}</div>
  </div>
  <div class="card">
    <div class="label">Daily P&amp;L</div>
    <div class="value {pnl_class}">{snap["pnl_daily_pct"]:+.2f}%</div>
    <div class="{pnl_class}">${snap["pnl_daily"]:+,.2f}</div>
  </div>
  <div class="card">
    <div class="label">Total P&amp;L (since start)</div>
    <div class="value {tot_class}">{snap["pnl_total_pct"]:+.2f}%</div>
    <div class="{tot_class}">${snap["pnl_total"]:+,.2f}</div>
  </div>
  <div class="card">
    <div class="label">Drawdown from Peak</div>
    <div class="value neg">{snap["drawdown_pct"]:.2f}%</div>
    <div style="color:#8b949e;font-size:12px;">Peak: ${snap["peak_equity"]:,.2f}</div>
  </div>
</div>

<h2>Equity Curve (last 90 days)</h2>
<canvas id="equityChart" height="80"></canvas>
<script>
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: {chart_labels},
    datasets: [{{
      label: 'Portfolio Equity ($)',
      data: {chart_data},
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88,166,255,0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 2,
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color:'#c9d1d9' }} }} }},
    scales: {{
      x: {{ ticks: {{ color:'#8b949e', maxTicksLimit:10 }} }},
      y: {{ ticks: {{ color:'#8b949e' }} }}
    }}
  }}
}});
</script>

<h2>Open Positions</h2>
<div class="card">
  <table>
    <tr><th>Symbol</th><th>Quantity</th><th>Avg Entry Price</th></tr>
    {pos_rows or "<tr><td colspan='3' style='color:#8b949e;'>No open positions</td></tr>"}
  </table>
</div>

<p style="color:#8b949e;font-size:11px;margin-top:40px;">
  Generated by Trading System — Paper Mode | {snap["date"]}
</p>
</body>
</html>"""

    fname = f"daily_{snap['date']}.html"
    out_path = Path(out_dir) / fname
    with open(out_path, "w") as f:
        f.write(html)
    log.info(f"Daily report saved: {out_path}")
    return str(out_path)


def email_report(html_path: str, snap: dict) -> None:
    """Send report via AWS SES. Requires env vars: SES_SENDER, SES_RECIPIENT, AWS_REGION."""
    import boto3

    sender = os.environ.get("SES_SENDER")
    recipient = os.environ.get("SES_RECIPIENT")
    region = os.environ.get("AWS_REGION", "eu-west-1")

    if not sender or not recipient:
        log.warning("SES_SENDER / SES_RECIPIENT env vars not set — skipping email")
        return

    with open(html_path) as f:
        html_body = f.read()

    subject = (
        f"[Trading] Daily Report {snap['date']} | "
        f"P&L: {snap['pnl_daily_pct']:+.2f}% | "
        f"Equity: ${snap['equity']:,.0f}"
    )

    client = boto3.client("ses", region_name=region)
    client.send_email(
        Source=sender,
        Destination={"ToAddresses": [recipient]},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Html": {"Data": html_body}},
        },
    )
    log.info(f"Daily report emailed to {recipient}")


def upload_to_s3(html_path: str, snap: dict) -> None:
    """Upload report HTML to S3. Requires env var: S3_BUCKET."""
    import boto3

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        log.warning("S3_BUCKET env var not set — skipping S3 upload")
        return

    key = f"reports/daily_{snap['date']}.html"
    s3 = boto3.client("s3")
    s3.upload_file(
        html_path,
        bucket,
        key,
        ExtraArgs={"ContentType": "text/html"},
    )
    log.info(f"Report uploaded to s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Daily report generator")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--email", action="store_true", help="Email report via AWS SES")
    parser.add_argument("--s3", action="store_true", help="Upload report to S3")
    args = parser.parse_args()

    config = load_config(args.config)
    state = load_paper_state()
    snap = snapshot_portfolio(config, state)

    log.info(f"Date:         {snap['date']}")
    log.info(f"Equity:       ${snap['equity']:,.2f}")
    log.info(f"Daily P&L:    {snap['pnl_daily_pct']:+.2f}% (${snap['pnl_daily']:+,.2f})")
    log.info(f"Total P&L:    {snap['pnl_total_pct']:+.2f}% (${snap['pnl_total']:+,.2f})")
    log.info(f"Drawdown:     {snap['drawdown_pct']:.2f}%")
    log.info(f"Open pos:     {len(snap['positions'])}")

    html_path = generate_daily_html(snap)

    if args.email:
        email_report(html_path, snap)
    if args.s3:
        upload_to_s3(html_path, snap)


if __name__ == "__main__":
    main()
