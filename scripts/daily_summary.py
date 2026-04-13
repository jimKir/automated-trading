#!/usr/bin/env python3
"""
Daily Performance Summary
=========================
Comprehensive daily email summarising ALL performance, capital, and risk
metrics — not just the 6 go-live criteria.

Data sources:
  1. results/paper_state.json  — equity history (from daily_report.py / Alpaca)
  2. results/paper_monitor.json — latest go-live scorecard
  3. results/wf_12m_oos_results.json — backtest reference
  4. Alpaca paper account (live fallback)

Sends a rich HTML email to configured recipients via Gmail SMTP.

Usage:
    python scripts/daily_summary.py                     # generate + print
    python scripts/daily_summary.py --email             # generate + email
    python scripts/daily_summary.py --email --verbose   # detailed console output

Environment variables (for --email):
    SMTP_USER          — Gmail address to send from
    SMTP_APP_PASSWORD  — Gmail App Password (not regular password)

Run daily after US market close via GitHub Actions or local cron.
"""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAPER_STATE     = ROOT / "results" / "paper_state.json"
SCORECARD_FILE  = ROOT / "results" / "paper_monitor.json"
OOS_RESULTS     = ROOT / "results" / "wf_12m_oos_results.json"
OOS_RETURNS     = ROOT / "results" / "wf_12m_strat_returns.csv"
PERIODS_YEAR    = 252

RECIPIENTS = ["kiritsis.di@gmail.com", "o.zoumpou@gmail.com"]
INITIAL_CAPITAL = 25_000


# ═════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_equity_history() -> pd.DataFrame:
    """Load equity curve from paper_state.json."""
    if not PAPER_STATE.exists():
        return pd.DataFrame(columns=["equity"])

    state = json.load(open(PAPER_STATE))
    history = state.get("equity_history", [])
    if not history:
        return pd.DataFrame(columns=["equity"])

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")
    return df


def load_scorecard() -> dict | None:
    """Load the latest go-live scorecard."""
    if SCORECARD_FILE.exists():
        return json.load(open(SCORECARD_FILE))
    return None


def load_oos_reference() -> dict | None:
    """Load WF 12M OOS backtest results for comparison."""
    if OOS_RESULTS.exists():
        return json.load(open(OOS_RESULTS))
    return None


def load_oos_returns() -> pd.Series | None:
    """Load OOS strategy returns for correlation."""
    if not OOS_RETURNS.exists():
        return None
    df = pd.read_csv(OOS_RETURNS, parse_dates=["date"], index_col="date")
    return df["strategy"].dropna()


def try_alpaca_snapshot() -> dict | None:
    """Try to get live snapshot from Alpaca."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    if not api_key or not api_secret:
        return None
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
        account = client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
        }
    except Exception as e:
        print(f"Alpaca snapshot skipped: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  Metric Computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(equity_df: pd.DataFrame, alpaca: dict | None = None) -> dict:
    """Compute comprehensive performance, capital, and risk metrics."""

    result = {
        "report_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "report_time_utc": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        "has_data": False,
    }

    if len(equity_df) < 2:
        result["error"] = "Insufficient data (need >= 2 equity points)"
        return result

    equities = equity_df["equity"].astype(float)
    returns = equities.pct_change().dropna()
    n_days = len(returns)

    if n_days < 2:
        result["error"] = "Insufficient returns data"
        return result

    result["has_data"] = True
    result["error"] = None

    # ── Capital Metrics ──────────────────────────────────────────────────
    initial = INITIAL_CAPITAL
    current = float(equities.iloc[-1])
    peak = float(equities.max())
    trough = float(equities.min())
    prev_day = float(equities.iloc[-2]) if len(equities) >= 2 else initial

    result["capital"] = {
        "initial_equity": initial,
        "current_equity": current,
        "peak_equity": peak,
        "trough_equity": trough,
        "cash": alpaca["cash"] if alpaca else None,
        "buying_power": alpaca["buying_power"] if alpaca else None,
        "long_exposure": alpaca["long_market_value"] if alpaca else None,
        "short_exposure": alpaca["short_market_value"] if alpaca else None,
        "pnl_daily": round(current - prev_day, 2),
        "pnl_daily_pct": round((current / prev_day - 1) * 100, 2),
        "pnl_total": round(current - initial, 2),
        "pnl_total_pct": round((current / initial - 1) * 100, 2),
    }

    # ── Performance Metrics ──────────────────────────────────────────────
    ann_ret = (1 + returns).prod() ** (PERIODS_YEAR / n_days) - 1
    ann_vol = returns.std() * np.sqrt(PERIODS_YEAR)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(PERIODS_YEAR) if len(downside) > 1 else 0.001
    sortino = float(ann_ret / downside_std)

    # Calmar (CAGR / MaxDD)
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = abs(float(dd.min()))
    calmar = float(ann_ret / max_dd) if max_dd > 0 else float("inf")

    # Win rate
    win_days = int((returns > 0).sum())
    loss_days = int((returns < 0).sum())
    flat_days = int((returns == 0).sum())
    win_rate = win_days / n_days * 100

    # Best / worst day
    best_day = returns.idxmax()
    worst_day = returns.idxmin()

    # Average win / loss
    avg_win = float(returns[returns > 0].mean() * 100) if win_days > 0 else 0.0
    avg_loss = float(returns[returns < 0].mean() * 100) if loss_days > 0 else 0.0

    # Profit factor
    gross_profit = float(returns[returns > 0].sum()) if win_days > 0 else 0.0
    gross_loss = abs(float(returns[returns < 0].sum())) if loss_days > 0 else 0.001
    profit_factor = gross_profit / gross_loss

    # Rolling 5-day and 20-day return
    rolling_5d = float(equities.iloc[-1] / equities.iloc[-min(6, len(equities))] - 1) * 100
    rolling_20d = float(equities.iloc[-1] / equities.iloc[-min(21, len(equities))] - 1) * 100

    result["performance"] = {
        "n_trading_days": n_days,
        "start_date": str(equities.index.min().date()),
        "end_date": str(equities.index.max().date()),
        "total_return_pct": round((current / initial - 1) * 100, 2),
        "cagr_pct": round(float(ann_ret * 100), 2),
        "ann_volatility_pct": round(float(ann_vol * 100), 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "profit_factor": round(profit_factor, 2),
        "win_rate_pct": round(win_rate, 1),
        "win_days": win_days,
        "loss_days": loss_days,
        "flat_days": flat_days,
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "best_day_return_pct": round(float(returns.max()) * 100, 3),
        "best_day_date": str(best_day.date()) if hasattr(best_day, "date") else str(best_day),
        "worst_day_return_pct": round(float(returns.min()) * 100, 3),
        "worst_day_date": str(worst_day.date()) if hasattr(worst_day, "date") else str(worst_day),
        "rolling_5d_return_pct": round(rolling_5d, 2),
        "rolling_20d_return_pct": round(rolling_20d, 2),
    }

    # ── Risk Metrics ─────────────────────────────────────────────────────
    # Max drawdown
    max_dd_pct = round(max_dd * 100, 2)

    # Current drawdown
    current_dd = float((cum.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]) * 100
    current_dd = round(current_dd, 2)

    # Drawdown recovery episodes (>5% that recovered)
    dd_episodes = _count_dd_recoveries(dd, -0.05)

    # Value at Risk (95% parametric)
    var_95 = round(float(returns.quantile(0.05)) * 100, 3)

    # Conditional VaR (Expected Shortfall)
    cvar_95 = round(float(returns[returns <= returns.quantile(0.05)].mean()) * 100, 3)

    # Max consecutive losses
    is_loss = (returns < 0).astype(int)
    max_consec_loss = _max_consecutive(is_loss)
    max_consec_win = _max_consecutive((returns > 0).astype(int))

    # Uptime (trading days with data / expected business days)
    expected_days = pd.bdate_range(equities.index.min(), equities.index.max())
    uptime_pct = round(min(n_days / max(len(expected_days) - 1, 1) * 100, 100.0), 1)

    # Backtest correlation
    oos_ret = load_oos_returns()
    backtest_corr = _compute_correlation(returns, oos_ret)

    result["risk"] = {
        "max_drawdown_pct": max_dd_pct,
        "current_drawdown_pct": current_dd,
        "dd_recovery_episodes": dd_episodes,
        "var_95_pct": var_95,
        "cvar_95_pct": cvar_95,
        "max_consecutive_losses": max_consec_loss,
        "max_consecutive_wins": max_consec_win,
        "system_uptime_pct": uptime_pct,
        "backtest_correlation": backtest_corr,
        "days_since_peak": int((equities.index[-1] - equities.index[cum.idxmax()]).days)
            if hasattr(cum.idxmax(), "date") else 0,
    }

    return result


def _count_dd_recoveries(dd_series: pd.Series, threshold: float) -> int:
    in_episode = False
    count = 0
    for val in dd_series:
        if not in_episode and val <= threshold:
            in_episode = True
        elif in_episode and val >= -0.001:
            count += 1
            in_episode = False
    return count


def _max_consecutive(binary_series: pd.Series) -> int:
    groups = binary_series.ne(binary_series.shift()).cumsum()
    if binary_series.sum() == 0:
        return 0
    return int(binary_series.groupby(groups).sum().max())


def _compute_correlation(paper_ret: pd.Series, oos_ret: pd.Series | None) -> float | None:
    if oos_ret is None or len(oos_ret) < 10:
        return None
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
#  HTML Generation
# ═════════════════════════════════════════════════════════════════════════════

def _colour(val: float, good_positive: bool = True, threshold: float = 0) -> str:
    """Return CSS colour based on value."""
    if good_positive:
        return "#3fb950" if val > threshold else ("#f85149" if val < -threshold else "#c9d1d9")
    else:
        return "#3fb950" if val < threshold else ("#f85149" if val > threshold else "#c9d1d9")


def _card(label: str, value: str, colour: str = "#c9d1d9", sub: str = "") -> str:
    sub_html = f'<div style="color:#8b949e;font-size:12px;margin-top:2px;">{sub}</div>' if sub else ""
    return (
        f'<div style="background:#161b22;border-radius:8px;padding:16px;border:1px solid #30363d;">'
        f'<div style="font-size:12px;color:#8b949e;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:22px;font-weight:bold;color:{colour};">{value}</div>'
        f'{sub_html}</div>'
    )


def _metric_row(label: str, value: str, colour: str = "#c9d1d9") -> str:
    return (
        f'<tr style="border-bottom:1px solid #21262d;">'
        f'<td style="padding:6px 12px;color:#8b949e;">{label}</td>'
        f'<td style="padding:6px 12px;color:{colour};font-weight:500;text-align:right;">{value}</td></tr>'
    )


def generate_html(metrics: dict, scorecard: dict | None, oos_ref: dict | None) -> str:
    """Build the full daily summary HTML email."""
    report_date = metrics["report_date"]

    if not metrics["has_data"]:
        return f"""
        <html><body style="font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;">
        <h1 style="color:#58a6ff;">Daily Performance Summary — {report_date}</h1>
        <p>No trading data available yet. Paper trading has not started or no equity history found.</p>
        <p style="color:#8b949e;">{metrics.get("error", "")}</p>
        </body></html>
        """

    cap = metrics["capital"]
    perf = metrics["performance"]
    risk = metrics["risk"]

    # Top-level cards
    pnl_col = _colour(cap["pnl_daily"])
    tot_col = _colour(cap["pnl_total"])
    dd_col = _colour(risk["current_drawdown_pct"], good_positive=False)

    cards_html = (
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0;">'
        + _card("Portfolio Equity", f"${cap['current_equity']:,.2f}",
                sub=f"Peak: ${cap['peak_equity']:,.2f}")
        + _card("Daily P&L", f"{cap['pnl_daily_pct']:+.2f}%", pnl_col,
                sub=f"${cap['pnl_daily']:+,.2f}")
        + _card("Total P&L", f"{cap['pnl_total_pct']:+.2f}%", tot_col,
                sub=f"${cap['pnl_total']:+,.2f} from ${cap['initial_equity']:,.0f}")
        + _card("Current Drawdown", f"{risk['current_drawdown_pct']:.2f}%", dd_col,
                sub=f"Max: {risk['max_drawdown_pct']:.2f}% | {risk['days_since_peak']}d from peak")
        + '</div>'
    )

    # ── Go-Live Scorecard (if available) ──
    scorecard_html = ""
    if scorecard and "criteria" in scorecard:
        sc_rows = ""
        for c in scorecard["criteria"]:
            sc_col = "#f85149" if c["status"] == "FAILING" else (
                "#3fb950" if c["status"] == "PASSED" else "#8b949e"
            )
            icon = "\u2717" if c["status"] == "FAILING" else (
                "\u2713" if c["status"] == "PASSED" else "\u2014"
            )
            sc_rows += (
                f'<tr style="border-bottom:1px solid #30363d;">'
                f'<td style="padding:6px 8px;">{c["index"]}</td>'
                f'<td style="padding:6px 8px;">{c["metric"]}</td>'
                f'<td style="padding:6px 8px;">{c["threshold"]}</td>'
                f'<td style="padding:6px 8px;">{c["current"]}</td>'
                f'<td style="padding:6px 8px;color:{sc_col};font-weight:bold;">{icon} {c["status"]}</td>'
                f'</tr>'
            )
        scorecard_html = f"""
        <h2 style="color:#79c0ff;margin-top:28px;">Go-Live Scorecard</h2>
        <p style="color:#8b949e;">{scorecard.get("summary", "")} | Go-live ready: {"Yes" if scorecard.get("go_live_ready") else "No"}</p>
        <table style="border-collapse:collapse;width:100%;">
        <tr style="background:#161b22;">
          <th style="padding:8px;text-align:left;color:#58a6ff;">#</th>
          <th style="padding:8px;text-align:left;color:#58a6ff;">Metric</th>
          <th style="padding:8px;text-align:left;color:#58a6ff;">Threshold</th>
          <th style="padding:8px;text-align:left;color:#58a6ff;">Current</th>
          <th style="padding:8px;text-align:left;color:#58a6ff;">Status</th>
        </tr>
        {sc_rows}
        </table>
        """

    # ── Performance Table ──
    perf_rows = (
        _metric_row("Trading Days", str(perf["n_trading_days"]))
        + _metric_row("Period", f"{perf['start_date']} \u2192 {perf['end_date']}")
        + _metric_row("Total Return", f"{perf['total_return_pct']:+.2f}%",
                      _colour(perf["total_return_pct"]))
        + _metric_row("CAGR", f"{perf['cagr_pct']:+.2f}%",
                      _colour(perf["cagr_pct"]))
        + _metric_row("Ann. Volatility", f"{perf['ann_volatility_pct']:.2f}%")
        + _metric_row("Sharpe Ratio", f"{perf['sharpe_ratio']:.3f}",
                      _colour(perf["sharpe_ratio"] - 0.5))
        + _metric_row("Sortino Ratio", f"{perf['sortino_ratio']:.3f}",
                      _colour(perf["sortino_ratio"] - 0.5))
        + _metric_row("Calmar Ratio", f"{perf['calmar_ratio']:.3f}")
        + _metric_row("Profit Factor", f"{perf['profit_factor']:.2f}",
                      _colour(perf["profit_factor"] - 1.0))
        + _metric_row("Win Rate", f"{perf['win_rate_pct']:.1f}%  ({perf['win_days']}W / {perf['loss_days']}L / {perf['flat_days']}F)",
                      _colour(perf["win_rate_pct"] - 50))
        + _metric_row("Avg Win / Avg Loss",
                      f"{perf['avg_win_pct']:+.3f}% / {perf['avg_loss_pct']:+.3f}%")
        + _metric_row("Best Day",
                      f"{perf['best_day_return_pct']:+.3f}% ({perf['best_day_date']})",
                      "#3fb950")
        + _metric_row("Worst Day",
                      f"{perf['worst_day_return_pct']:+.3f}% ({perf['worst_day_date']})",
                      "#f85149")
        + _metric_row("Rolling 5d Return", f"{perf['rolling_5d_return_pct']:+.2f}%",
                      _colour(perf["rolling_5d_return_pct"]))
        + _metric_row("Rolling 20d Return", f"{perf['rolling_20d_return_pct']:+.2f}%",
                      _colour(perf["rolling_20d_return_pct"]))
    )

    # ── Risk Table ──
    corr_str = f"{risk['backtest_correlation']:.3f}" if risk["backtest_correlation"] is not None else "N/A"
    risk_rows = (
        _metric_row("Max Drawdown", f"{risk['max_drawdown_pct']:.2f}%",
                    _colour(risk["max_drawdown_pct"] - 15, good_positive=False))
        + _metric_row("Current Drawdown", f"{risk['current_drawdown_pct']:.2f}%",
                      _colour(risk["current_drawdown_pct"], good_positive=False))
        + _metric_row("Days Since Peak", str(risk["days_since_peak"]))
        + _metric_row("DD Recovery Episodes", f"{risk['dd_recovery_episodes']} (>5% recovered)")
        + _metric_row("VaR (95%)", f"{risk['var_95_pct']:.3f}%", "#f85149")
        + _metric_row("CVaR / Expected Shortfall (95%)", f"{risk['cvar_95_pct']:.3f}%", "#f85149")
        + _metric_row("Max Consecutive Losses", str(risk["max_consecutive_losses"]))
        + _metric_row("Max Consecutive Wins", str(risk["max_consecutive_wins"]))
        + _metric_row("System Uptime", f"{risk['system_uptime_pct']:.1f}%",
                      _colour(risk["system_uptime_pct"] - 95))
        + _metric_row("Backtest Correlation", corr_str,
                      _colour((risk["backtest_correlation"] or 0) - 0.6))
    )

    # ── Capital Table ──
    cap_rows = (
        _metric_row("Initial Capital", f"${cap['initial_equity']:,.2f}")
        + _metric_row("Current Equity", f"${cap['current_equity']:,.2f}")
        + _metric_row("Peak Equity", f"${cap['peak_equity']:,.2f}")
        + _metric_row("Trough Equity", f"${cap['trough_equity']:,.2f}")
    )
    if cap.get("cash") is not None:
        cap_rows += _metric_row("Cash", f"${cap['cash']:,.2f}")
    if cap.get("buying_power") is not None:
        cap_rows += _metric_row("Buying Power", f"${cap['buying_power']:,.2f}")
    if cap.get("long_exposure") is not None:
        cap_rows += _metric_row("Long Exposure", f"${cap['long_exposure']:,.2f}")
    if cap.get("short_exposure") is not None:
        cap_rows += _metric_row("Short Exposure", f"${cap['short_exposure']:,.2f}")

    # ── Backtest Reference ──
    bt_html = ""
    if oos_ref:
        bt = oos_ref.get("metrics", oos_ref)
        bt_html = f"""
        <h2 style="color:#79c0ff;margin-top:28px;">Backtest Reference (WF 12M OOS)</h2>
        <table style="border-collapse:collapse;width:100%;"><tbody>
        {_metric_row("Sharpe", str(bt.get("sharpe", "N/A")))}
        {_metric_row("CAGR", str(bt.get("cagr_pct", bt.get("cagr", "N/A"))) + "%")}
        {_metric_row("Max Drawdown", str(bt.get("max_drawdown_pct", bt.get("max_dd", "N/A"))) + "%")}
        {_metric_row("Win Rate", str(bt.get("win_rate_pct", bt.get("win_rate", "N/A"))) + "%")}
        </tbody></table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Daily Summary — {report_date}</title></head>
<body style="font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:20px;max-width:800px;">

<h1 style="color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:10px;">
  Daily Performance Summary
</h1>
<p style="color:#8b949e;">{report_date} | Strategy: Multi-Factor Momentum + Mean-Reversion | Paper Trading</p>

{cards_html}

{scorecard_html}

<h2 style="color:#79c0ff;margin-top:28px;">Performance Metrics</h2>
<table style="border-collapse:collapse;width:100%;"><tbody>
{perf_rows}
</tbody></table>

<h2 style="color:#79c0ff;margin-top:28px;">Risk Metrics</h2>
<table style="border-collapse:collapse;width:100%;"><tbody>
{risk_rows}
</tbody></table>

<h2 style="color:#79c0ff;margin-top:28px;">Capital Summary</h2>
<table style="border-collapse:collapse;width:100%;"><tbody>
{cap_rows}
</tbody></table>

{bt_html}

<p style="color:#8b949e;font-size:11px;margin-top:40px;border-top:1px solid #30363d;padding-top:12px;">
  Auto-generated by <a href="https://github.com/jimKir/automated-trading" style="color:#58a6ff;">Trading System</a>
  — Paper Mode | {metrics["report_time_utc"]}
</p>
</body></html>"""

    return html


# ═════════════════════════════════════════════════════════════════════════════
#  Email
# ═════════════════════════════════════════════════════════════════════════════

def send_email(html: str, report_date: str, metrics: dict) -> None:
    """Send the daily summary via Gmail SMTP."""
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_APP_PASSWORD", "")

    if not smtp_user or not smtp_pass:
        print("SMTP_USER / SMTP_APP_PASSWORD not set — skipping email")
        return

    if metrics["has_data"]:
        cap = metrics["capital"]
        subject = (
            f"[Trading] Daily Summary {report_date} | "
            f"P&L: {cap['pnl_daily_pct']:+.2f}% | "
            f"Equity: ${cap['current_equity']:,.0f}"
        )
    else:
        subject = f"[Trading] Daily Summary {report_date} | No data yet"

    for recipient in RECIPIENTS:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        print(f"Daily summary emailed to {recipient}")


# ═════════════════════════════════════════════════════════════════════════════
#  Console Output
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(metrics: dict, verbose: bool = False) -> None:
    """Print a compact summary to stdout."""
    if not metrics["has_data"]:
        print(f"\nDaily Summary — {metrics['report_date']}")
        print(f"  No data: {metrics.get('error', 'unknown')}\n")
        return

    cap = metrics["capital"]
    perf = metrics["performance"]
    risk = metrics["risk"]

    print(f"\n{'='*65}")
    print(f"  DAILY PERFORMANCE SUMMARY — {metrics['report_date']}")
    print(f"{'='*65}")
    print(f"  Equity:  ${cap['current_equity']:,.2f}  "
          f"(peak: ${cap['peak_equity']:,.2f})")
    print(f"  Daily:   {cap['pnl_daily_pct']:+.2f}%  (${cap['pnl_daily']:+,.2f})")
    print(f"  Total:   {cap['pnl_total_pct']:+.2f}%  (${cap['pnl_total']:+,.2f})")
    print(f"-"*65)
    print(f"  Sharpe: {perf['sharpe_ratio']:.3f}  |  "
          f"Sortino: {perf['sortino_ratio']:.3f}  |  "
          f"Calmar: {perf['calmar_ratio']:.3f}")
    print(f"  Win Rate: {perf['win_rate_pct']:.1f}%  |  "
          f"Profit Factor: {perf['profit_factor']:.2f}")
    print(f"  Max DD: {risk['max_drawdown_pct']:.2f}%  |  "
          f"Current DD: {risk['current_drawdown_pct']:.2f}%")
    print(f"  VaR(95%): {risk['var_95_pct']:.3f}%  |  "
          f"CVaR(95%): {risk['cvar_95_pct']:.3f}%")
    print(f"  Uptime: {risk['system_uptime_pct']:.1f}%  |  "
          f"BT Corr: {risk['backtest_correlation'] or 'N/A'}")

    if verbose:
        print(f"-"*65)
        print(f"  CAGR: {perf['cagr_pct']:+.2f}%  |  "
              f"Vol: {perf['ann_volatility_pct']:.2f}%")
        print(f"  Best:  {perf['best_day_return_pct']:+.3f}% ({perf['best_day_date']})")
        print(f"  Worst: {perf['worst_day_return_pct']:+.3f}% ({perf['worst_day_date']})")
        print(f"  5d: {perf['rolling_5d_return_pct']:+.2f}%  |  "
              f"20d: {perf['rolling_20d_return_pct']:+.2f}%")
        print(f"  Max consec loss: {risk['max_consecutive_losses']}  |  "
              f"Max consec win: {risk['max_consecutive_wins']}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Daily performance summary")
    parser.add_argument("--email", action="store_true", help="Send summary via email")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed console output")
    args = parser.parse_args()

    # Load data
    equity = load_equity_history()
    alpaca = try_alpaca_snapshot()
    scorecard = load_scorecard()
    oos_ref = load_oos_reference()

    # Compute all metrics
    metrics = compute_all_metrics(equity, alpaca)

    # Console output
    print_summary(metrics, verbose=args.verbose)

    # Save metrics JSON
    out_path = ROOT / "results" / "daily_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved → {out_path}")

    # Generate HTML and optionally email
    html = generate_html(metrics, scorecard, oos_ref)

    html_path = ROOT / "results" / "daily" / f"summary_{metrics['report_date']}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report → {html_path}")

    if args.email:
        send_email(html, metrics["report_date"], metrics)


if __name__ == "__main__":
    main()
