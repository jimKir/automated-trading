#!/usr/bin/env python3
"""
run_v2.py — Production momentum scan runner
============================================

Run:
    python run_v2.py                   # normal scan
    python run_v2.py --force           # skip regime check
    python run_v2.py --symbols 100     # expand universe
    python run_v2.py --top 10          # fewer results
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── credentials (MUST come from environment variables) ──────────────────────
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError(
        "Missing credentials. Set environment variables:\n"
        "  export APCA_API_KEY_ID=your_key\n"
        "  export APCA_API_SECRET_KEY=your_secret"
    )

# ── symbol universe ────────────────────────────────────────────────────────
BASE_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "GOOG",
    "META",
    "AMZN",
    "TSLA",
    "AVGO",
    "JPM",
    "V",
    "WMT",
    "MA",
    "UNH",
    "XOM",
    "LLY",
    "JNJ",
    "PG",
    "COST",
    "HD",
    "MRK",
    "ABBV",
    "CVX",
    "BAC",
    "PEP",
    "KO",
    "TMO",
    "ACN",
    "MCD",
    "ABT",
    "CRM",
    "CSCO",
    "GE",
    "DHR",
    "LIN",
    "TXN",
    "AMD",
    "NKE",
    "HON",
    "SBUX",
    "QCOM",
    "UPS",
    "INTC",
    "LOW",
    "BA",
    "CAT",
    "GS",
    "AMAT",
    "BLK",
    "SCHW",
    "C",
    "WFC",
    "MS",
    "MU",
    "LRCX",
    "PANW",
    "NOW",
    "INTU",
    "ISRG",
    "REGN",
    "GILD",
    "BKNG",
    "ABNB",
    "TGT",
    "SLB",
    "MPC",
    "PSX",
    "OKE",
    "COP",
    "EOG",
    "XOM",
    "VLO",
    "DVN",
    "UNH",
    "MDT",
    "BMY",
    "PFE",
    "CVS",
    "JNJ",
    "PLD",
    "AMT",
    "CCI",
    "DLR",
    "EQIX",
    "NEE",
    "DUK",
    "SO",
    "NFLX",
    "DIS",
    "T",
    "MMM",
    "ROK",
    "RTX",
    "LMT",
    "NOC",
    "GD",
    "L3",
    "TDG",
    "PH",
    "ETN",
    "BRK.B",
    "AXP",
    "USB",
    "PNC",
    "TRV",
    "ALL",
    "CB",
    "MET",
    "PRU",
    "AFL",
]


def get_symbols(n: int) -> list:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from symbols import get_symbol_list

        return get_symbol_list("sp500")[:n]
    except Exception:
        return list(dict.fromkeys(BASE_UNIVERSE))[:n]  # deduplicated


# ===========================================================================
# Terminal output
# ===========================================================================

REGIME_COLOURS = {
    "TRENDING_UP": "\033[92m",  # green
    "TRENDING_DOWN": "\033[91m",  # red
    "TRANSITIONING": "\033[93m",  # yellow
    "CHOPPY": "\033[90m",  # grey
    "HIGH_FEAR": "\033[95m",  # magenta
    "ELEVATED_VOL": "\033[93m",  # yellow
    "UNKNOWN": "\033[97m",  # white
}
RESET = "\033[0m"
BOLD = "\033[1m"


def print_banner(result: dict):
    r = result["regime"]
    rc = REGIME_COLOURS.get(r["regime"], "")
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}  MOMENTUM SCANNER V2  ·  {ts}{RESET}")
    print(f"{'=' * 72}")
    print(
        f"  Regime   {rc}{BOLD}{r['regime']:16}{RESET}  "
        f"ADX {r['adx']:5.1f}  "
        f"SPY vs EMA20 {r['spy_vs_ema20_pct']:+.2f}%  "
        f"VIX {r['vix']:.1f}"
    )
    print(
        f"  Scanned  {result['symbols_scanned']} symbols  ·  "
        f"elapsed {result['elapsed']:.1f}s  ·  "
        f"consensus signals {len(result['consensus'])}"
    )
    if not r["tradeable"]:
        print(f"\n  ⚠  {rc}Market is {r['regime']} — signals may be unreliable{RESET}")
    print()


def print_table(title: str, rows: list, colour: str):
    if not rows:
        print(f"  {title}: none\n")
        return

    hdr = (
        f"{'Symbol':<7} {'Score':>7} {'Price':>8}  "
        f"{'VWAP Dev':>9} {'Rel Str':>8} {'Vol Surp':>9}  "
        f"{'Sector':<12} {'Direction'}"
    )
    sep = "-" * 80

    print(f"  {BOLD}{colour}{title}{RESET}")
    print(f"  {sep}")
    print(f"  {hdr}")
    print(f"  {sep}")

    for r in rows:
        flag = "★" if r["symbol"] in _consensus_set else " "
        print(
            f"  {flag}{r['symbol']:<6} "
            f"{r['score']:>7.3f} "
            f"${r['price']:>7.2f}  "
            f"{r.get('vwap_dev_pct', 0):>+8.3f}% "
            f"{r.get('rel_strength_pct', 0):>+7.3f}% "
            f"{r.get('vol_surprise', 0):>+8.3f}  "
            f"{r.get('sector', 'Other'):<12} "
            f"{r.get('direction', '')}"
        )
    print()


_consensus_set: set = set()


def print_consensus(consensus: list):
    if not consensus:
        print("  No consensus signals this scan.\n")
        return
    print(f"  {BOLD}★ CONSENSUS — all 3 factors agree:{RESET}")
    for sym in consensus:
        print(f"    • {sym}")
    print()


# ===========================================================================
# HTML dashboard
# ===========================================================================


def build_dashboard(result: dict, outfile: str = "momentum_v2_dashboard.html"):
    regime = result["regime"]
    top_long = result["top_long"]
    top_short = result["top_short"]
    consensus = result["consensus"]
    signals = result["signals"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    REGIME_BG = {
        "TRENDING_UP": "#1b5e20",
        "TRENDING_DOWN": "#b71c1c",
        "TRANSITIONING": "#e65100",
        "CHOPPY": "#37474f",
        "HIGH_FEAR": "#4a148c",
        "ELEVATED_VOL": "#f57f17",
        "UNKNOWN": "#263238",
    }
    regime_bg = REGIME_BG.get(regime["regime"], "#263238")

    def rows_html(items, colour):
        out = []
        for r in items:
            star = "★ " if r["symbol"] in consensus else ""
            sc = r.get("score", 0)
            sc_col = "#4CAF50" if sc > 0 else "#f44336"
            out.append(
                f"<tr>"
                f"<td><strong>{star}{r['symbol']}</strong></td>"
                f"<td style='color:{sc_col};font-weight:700'>{sc:+.3f}</td>"
                f"<td>${r.get('price', 0):.2f}</td>"
                f"<td style='color:{colour}'>{r.get('raw_return_pct', 0):+.2f}%</td>"
                f"<td>{r.get('vwap_dev_pct', 0):+.3f}%</td>"
                f"<td>{r.get('rel_strength_pct', 0):+.3f}%</td>"
                f"<td>{r.get('vol_surprise', 0):+.2f}</td>"
                f"<td>{r.get('volume', 0):,}</td>"
                f"<td>{r.get('sector', 'Other')}</td>"
                f"</tr>"
            )
        return "\n".join(out)

    # Build factor chart data from all signals
    chart_symbols = []
    chart_scores = []
    chart_colors = []
    if not signals.empty:
        top30 = signals.head(15).append(signals.tail(15)) if len(signals) >= 30 else signals
        for _, row in top30.iterrows():
            chart_symbols.append(row["symbol"])
            chart_scores.append(round(float(row["score"]), 3))
            chart_colors.append(
                "'rgba(76,175,80,0.8)'" if row["score"] > 0 else "'rgba(244,67,54,0.8)'"
            )

    consensus_badges = (
        "".join(f"<span class='badge'>{s}</span>" for s in consensus)
        or "<span style='color:#888'>None this scan</span>"
    )

    # Factor scatter data
    scatter_data = []
    if not signals.empty:
        for _, row in signals.iterrows():
            scatter_data.append(
                {
                    "x": round(float(row.get("rel_strength_pct", 0)), 3),
                    "y": round(float(row.get("vwap_dev_pct", 0)), 3),
                    "r": min(max(abs(float(row.get("vol_surprise", 0))) * 8 + 4, 4), 20),
                    "label": row["symbol"],
                    "score": round(float(row.get("score", 0)), 3),
                }
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum Scanner V2</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d0d0d;color:#e0e0e0;padding:20px}}
  h1{{font-size:1.6em;font-weight:700;margin-bottom:2px}}
  .sub{{color:#666;font-size:.85em;margin-bottom:24px}}
  .regime-bar{{background:{regime_bg};border-radius:8px;padding:14px 20px;
               margin-bottom:24px;display:flex;gap:32px;align-items:center;flex-wrap:wrap}}
  .regime-name{{font-size:1.4em;font-weight:800;letter-spacing:.05em}}
  .regime-stat{{font-size:.9em;color:rgba(255,255,255,.8)}}
  .regime-stat span{{font-weight:700;color:#fff}}
  .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}}
  .stat{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:16px;text-align:center}}
  .stat-val{{font-size:2em;font-weight:700;color:#4CAF50}}
  .stat-val.neg{{color:#f44336}}
  .stat-label{{font-size:.75em;color:#888;margin-top:4px;text-transform:uppercase;letter-spacing:.05em}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
  .grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}}
  @media(max-width:900px){{.grid2,.grid3{{grid-template-columns:1fr}}}}
  .card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:20px;overflow:hidden}}
  .card h2{{font-size:1em;font-weight:600;margin-bottom:14px;color:#aaa;text-transform:uppercase;letter-spacing:.06em}}
  table{{width:100%;border-collapse:collapse;font-size:.82em}}
  th{{text-align:left;padding:7px 8px;background:#222;color:#666;font-size:.78em;
      text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #333}}
  td{{padding:7px 8px;border-bottom:1px solid #1e1e1e}}
  tr:hover td{{background:#222}}
  .badge{{background:#1565c0;color:#fff;padding:4px 11px;border-radius:12px;
           margin:3px;display:inline-block;font-size:.82em;font-weight:600}}
  .consensus-box{{background:#0d2137;border:1px solid #1565c0;border-radius:8px;
                  padding:20px;margin-bottom:16px}}
  .consensus-box h2{{color:#64b5f6;font-size:1em;margin-bottom:12px}}
  canvas{{max-height:280px}}
  .full{{grid-column:1/-1}}
  .note{{color:#555;font-size:.75em;margin-top:8px}}
</style>
</head>
<body>
<h1>📊 Momentum Scanner V2</h1>
<p class="sub">Generated: {ts} &nbsp;·&nbsp; Signals use VWAP deviation + relative strength vs SPY + volume surprise</p>

<div class="regime-bar">
  <div class="regime-name">{regime["regime"].replace("_", " ")}</div>
  <div class="regime-stat">ADX <span>{regime["adx"]}</span></div>
  <div class="regime-stat">SPY vs EMA20 <span>{regime["spy_vs_ema20_pct"]:+.2f}%</span></div>
  <div class="regime-stat">VIX <span>{regime["vix"]}</span></div>
  <div class="regime-stat">SPY 5h drift <span>{regime["spy_drift_5h_pct"]:+.2f}%</span></div>
  <div class="regime-stat">Size multiplier <span>{regime["size_multiplier"]:.0%}</span></div>
</div>

<div class="stats">
  <div class="stat"><div class="stat-val">{result["symbols_scanned"]}</div><div class="stat-label">Symbols Scanned</div></div>
  <div class="stat"><div class="stat-val">{len(top_long)}</div><div class="stat-label">Long Signals</div></div>
  <div class="stat"><div class="stat-val neg">{len(top_short)}</div><div class="stat-label">Short Signals</div></div>
  <div class="stat"><div class="stat-val" style="color:#2196F3">{len(consensus)}</div><div class="stat-label">Consensus ★</div></div>
  <div class="stat"><div class="stat-val">{result["elapsed"]:.1f}s</div><div class="stat-label">Scan Time</div></div>
</div>

<div class="consensus-box">
  <h2>★ CONSENSUS SIGNALS — all 3 factors agree (highest confidence)</h2>
  <div>{consensus_badges}</div>
  <p class="note">A consensus signal requires VWAP deviation, relative strength vs SPY, and volume surprise to all point in the same direction, with |score| &gt; 0.5</p>
</div>

<div class="grid2">
  <div class="card">
    <h2>🚀 Top Long Candidates</h2>
    <table>
      <tr><th>Symbol</th><th>Score</th><th>Price</th><th>Return</th>
          <th>VWAP Dev</th><th>Rel Str</th><th>Vol Surp</th><th>Volume</th><th>Sector</th></tr>
      {rows_html(top_long, "#4CAF50")}
    </table>
  </div>
  <div class="card">
    <h2>📉 Top Short Candidates</h2>
    <table>
      <tr><th>Symbol</th><th>Score</th><th>Price</th><th>Return</th>
          <th>VWAP Dev</th><th>Rel Str</th><th>Vol Surp</th><th>Volume</th><th>Sector</th></tr>
      {rows_html(top_short, "#f44336")}
    </table>
  </div>
</div>

<div class="grid3">
  <div class="card full" style="grid-column:1/3">
    <h2>Composite Score — Top &amp; Bottom 30</h2>
    <canvas id="scoreChart"></canvas>
  </div>
  <div class="card">
    <h2>Rel Strength vs VWAP Dev (bubble = volume surprise)</h2>
    <canvas id="scatterChart"></canvas>
  </div>
</div>

<script>
// Score bar chart
const scoreCtx = document.getElementById('scoreChart').getContext('2d');
new Chart(scoreCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(chart_symbols)},
    datasets: [{{
      label: 'Composite Score',
      data:  {json.dumps(chart_scores)},
      backgroundColor: [{",".join(chart_colors)}],
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{legend:{{display:false}}}},
    scales: {{
      x: {{ticks:{{color:'#888',font:{{size:10}}}},grid:{{color:'#222'}}}},
      y: {{ticks:{{color:'#888'}},grid:{{color:'#222'}}}},
    }}
  }}
}});

// Scatter chart
const scatterCtx = document.getElementById('scatterChart').getContext('2d');
const scatterRaw = {json.dumps(scatter_data)};
new Chart(scatterCtx, {{
  type: 'bubble',
  data: {{
    datasets: [{{
      label: 'Symbols',
      data: scatterRaw.map(d => ({{x:d.x, y:d.y, r:d.r, label:d.label, score:d.score}})),
      backgroundColor: scatterRaw.map(d => d.score > 0
        ? 'rgba(76,175,80,0.7)' : 'rgba(244,67,54,0.7)'),
    }}]
  }},
  options: {{
    responsive:true,
    plugins: {{
      legend:{{display:false}},
      tooltip:{{callbacks:{{
        label: ctx => `${{ctx.raw.label}}  score:${{ctx.raw.score}}`
      }}}}
    }},
    scales: {{
      x: {{title:{{display:true,text:'Relative Strength vs SPY (%)',color:'#888'}},
           ticks:{{color:'#888'}},grid:{{color:'#222'}}}},
      y: {{title:{{display:true,text:'VWAP Deviation (%)',color:'#888'}},
           ticks:{{color:'#888'}},grid:{{color:'#222'}}}},
    }}
  }}
}});
</script>

<p class="note" style="margin-top:20px;text-align:center">
  ★ = consensus signal (all 3 factors agree) &nbsp;·&nbsp;
  Sector limit: max 3 signals per sector &nbsp;·&nbsp;
  Scores are cross-sectional Z-scores — comparable across symbols
</p>
</body>
</html>"""

    with open(outfile, "w") as f:
        f.write(html)
    logger.info(f"  Dashboard → {outfile}")
    return outfile


# ===========================================================================
# JSON report
# ===========================================================================


def save_json(result: dict, outfile: str = "momentum_v2_report.json"):
    payload = {
        "timestamp": result["timestamp"],
        "regime": result["regime"],
        "symbols_scanned": result["symbols_scanned"],
        "elapsed_sec": result["elapsed"],
        "consensus": result["consensus"],
        "top_long": [{k: v for k, v in r.items() if k != "signals"} for r in result["top_long"]],
        "top_short": [{k: v for k, v in r.items() if k != "signals"} for r in result["top_short"]],
    }
    # signals dataframe
    if not result["signals"].empty:
        payload["all_signals"] = result["signals"].to_dict("records")

    with open(outfile, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"  Report    → {outfile}")


# ===========================================================================
# Entry point
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Momentum Scanner V2")
    parser.add_argument("--symbols", type=int, default=60, help="Universe size (default 60)")
    parser.add_argument("--top", type=int, default=20, help="Top N results (default 20)")
    parser.add_argument("--sector", type=int, default=3, help="Max per sector (default 3)")
    parser.add_argument("--force", action="store_true", help="Ignore regime check")
    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print("  MOMENTUM SCANNER V2  —  starting")
    print(f"{'=' * 72}\n")

    # Import scanner
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scanner_v2 import MomentumScannerV2

    # Load symbols
    symbols = get_symbols(args.symbols)
    logger.info(f"Universe: {len(symbols)} symbols")

    # Run scan
    scanner = MomentumScannerV2(api_key=API_KEY, api_secret=API_SECRET)
    result = scanner.scan(
        symbols,
        top_n=args.top,
        max_per_sector=args.sector,
        force=args.force,
    )

    # Terminal output
    global _consensus_set
    _consensus_set = set(result["consensus"])

    print_banner(result)

    if result["top_long"] or result["top_short"]:
        print_table("TOP LONG SIGNALS", result["top_long"], "\033[92m")
        print_table("TOP SHORT SIGNALS", result["top_short"], "\033[91m")
        print_consensus(result["consensus"])
    else:
        print("  No signals generated.\n")
        if not result["regime"]["tradeable"]:
            print(f"  Reason: market regime is {result['regime']['regime']}")
            print("  Use --force to run anyway.\n")

    # Save outputs
    print(f"{'=' * 72}")
    print("  Saving outputs...")
    save_json(result)
    dash = build_dashboard(result)
    print(f"\n  ✅  Done — open {dash} in your browser")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
