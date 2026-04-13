"""
Gmail Draft Lambda
==================
Fetches the latest daily_summary.json and paper_monitor.json from the
jimKir/automated-trading GitHub repo, formats a plain-text email, and
creates a Gmail draft in kiritsis.di@gmail.com so it can be reviewed
and forwarded manually.

If any go-live criterion is FAILING, a separate alert draft is created.

Environment variables (set in Lambda config or SSM Parameter Store):
    GITHUB_REPO        — e.g. jimKir/automated-trading (default)
    GITHUB_TOKEN       — fine-grained PAT with Contents:read scope
    GMAIL_CREDENTIALS  — SSM parameter name storing the Gmail OAuth
                         credentials JSON (client_id, client_secret,
                         refresh_token, token_uri)
    RECIPIENTS         — comma-separated emails (default: both addresses)
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from email.mime.text import MIMEText

import boto3

# ── Config ───────────────────────────────────────────────────────────────────
GITHUB_REPO = os.environ.get("GITHUB_REPO", "jimKir/automated-trading")
GITHUB_TOKEN_PARAM = os.environ.get("GITHUB_TOKEN_PARAM", "/trading/github-token")
SSM_CREDS_PARAM = os.environ.get("GMAIL_CREDENTIALS", "/trading/gmail-oauth")
RECIPIENTS = os.environ.get(
    "RECIPIENTS", "kiritsis.di@gmail.com,o.zoumpou@gmail.com"
).split(",")

SUMMARY_PATH = "results/daily_summary.json"
SCORECARD_PATH = "results/paper_monitor.json"

ssm = boto3.client("ssm")


# ── GitHub helpers ───────────────────────────────────────────────────────────

_github_token_cache: str | None = None


def _get_github_token() -> str:
    """Fetch GitHub PAT from SSM (cached for Lambda lifetime)."""
    global _github_token_cache
    if _github_token_cache is None:
        try:
            resp = ssm.get_parameter(Name=GITHUB_TOKEN_PARAM, WithDecryption=True)
            _github_token_cache = resp["Parameter"]["Value"]
        except Exception:
            _github_token_cache = ""
    return _github_token_cache


def github_fetch_json(path: str) -> dict | None:
    """Fetch a JSON file from the repo's main branch via GitHub raw content."""
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{path}"
    req = urllib.request.Request(url)
    token = _get_github_token()
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3.raw")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Failed to fetch {path}: {e}")
        return None


# ── Gmail helpers ────────────────────────────────────────────────────────────

def get_gmail_credentials() -> dict:
    """Load OAuth credentials from SSM Parameter Store."""
    resp = ssm.get_parameter(Name=SSM_CREDS_PARAM, WithDecryption=True)
    return json.loads(resp["Parameter"]["Value"])


def refresh_access_token(creds: dict) -> str:
    """Exchange refresh_token for a fresh access_token."""
    data = (
        f"client_id={creds['client_id']}"
        f"&client_secret={creds['client_secret']}"
        f"&refresh_token={creds['refresh_token']}"
        f"&grant_type=refresh_token"
    ).encode()
    req = urllib.request.Request(
        creds.get("token_uri", "https://oauth2.googleapis.com/token"),
        data=data,
        method="POST",
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())["access_token"]


def create_gmail_draft(access_token: str, to: list[str], subject: str, body: str) -> str:
    """Create a draft in the authenticated user's Gmail."""
    msg = MIMEText(body, "plain")
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    payload = json.dumps({"message": {"raw": raw}}).encode()
    req = urllib.request.Request(
        "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
        data=payload,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode())
    return result["id"]


# ── Email formatting ─────────────────────────────────────────────────────────

def format_daily_summary(m: dict) -> tuple[str, str]:
    """Format the daily summary JSON into subject + plain-text body."""
    date = m.get("report_date", "unknown")

    if not m.get("has_data"):
        return (
            f"[Trading] Daily Summary {date} | No data yet",
            f"Daily Performance Summary — {date}\n\n"
            f"No trading data available.\n{m.get('error', '')}\n"
        )

    cap = m["capital"]
    perf = m["performance"]
    risk = m["risk"]

    subject = (
        f"[Trading] Daily Summary {date} | "
        f"P&L: {cap['pnl_daily_pct']:+.2f}% | "
        f"Equity: ${cap['current_equity']:,.0f}"
    )

    corr_str = (
        f"{risk['backtest_correlation']:.3f}"
        if risk.get("backtest_correlation") is not None
        else "N/A"
    )

    body = f"""DAILY PERFORMANCE SUMMARY — {date}
Strategy: Multi-Factor Momentum + Mean-Reversion | Paper Trading
{'=' * 60}

CAPITAL
  Equity:       ${cap['current_equity']:,.2f}  (peak: ${cap['peak_equity']:,.2f})
  Daily P&L:    {cap['pnl_daily_pct']:+.2f}%  (${cap['pnl_daily']:+,.2f})
  Total P&L:    {cap['pnl_total_pct']:+.2f}%  (${cap['pnl_total']:+,.2f})
  Initial:      ${cap['initial_equity']:,.2f}"""

    if cap.get("cash") is not None:
        body += f"\n  Cash:         ${cap['cash']:,.2f}"
    if cap.get("buying_power") is not None:
        body += f"\n  Buying Power: ${cap['buying_power']:,.2f}"

    body += f"""

PERFORMANCE
  Trading Days: {perf['n_trading_days']}  ({perf['start_date']} → {perf['end_date']})
  Total Return: {perf['total_return_pct']:+.2f}%
  CAGR:         {perf['cagr_pct']:+.2f}%
  Volatility:   {perf['ann_volatility_pct']:.2f}%
  Sharpe:       {perf['sharpe_ratio']:.3f}
  Sortino:      {perf['sortino_ratio']:.3f}
  Calmar:       {perf['calmar_ratio']:.3f}
  Profit Factor:{perf['profit_factor']:.2f}
  Win Rate:     {perf['win_rate_pct']:.1f}%  ({perf['win_days']}W / {perf['loss_days']}L / {perf['flat_days']}F)
  Avg Win:      {perf['avg_win_pct']:+.3f}%
  Avg Loss:     {perf['avg_loss_pct']:+.3f}%
  Best Day:     {perf['best_day_return_pct']:+.3f}% ({perf['best_day_date']})
  Worst Day:    {perf['worst_day_return_pct']:+.3f}% ({perf['worst_day_date']})
  Rolling 5d:   {perf['rolling_5d_return_pct']:+.2f}%
  Rolling 20d:  {perf['rolling_20d_return_pct']:+.2f}%

RISK
  Max Drawdown:     {risk['max_drawdown_pct']:.2f}%
  Current Drawdown: {risk['current_drawdown_pct']:.2f}%
  Days Since Peak:  {risk['days_since_peak']}
  DD Recoveries:    {risk['dd_recovery_episodes']} episodes (>5%)
  VaR (95%):        {risk['var_95_pct']:.3f}%
  CVaR (95%):       {risk['cvar_95_pct']:.3f}%
  Max Consec Loss:  {risk['max_consecutive_losses']}
  Max Consec Win:   {risk['max_consecutive_wins']}
  System Uptime:    {risk['system_uptime_pct']:.1f}%
  BT Correlation:   {corr_str}

{'—' * 60}
Auto-generated by Trading System — Paper Mode
https://github.com/{GITHUB_REPO}
"""
    return subject, body


def format_alert(sc: dict) -> tuple[str, str] | None:
    """Format a FAILING-criteria alert. Returns None if no failures."""
    failing = [c for c in sc.get("criteria", []) if c["status"] == "FAILING"]
    if not failing:
        return None

    n_fail = len(failing)
    n_total = len(sc["criteria"])
    metrics = sc.get("metrics", {})
    eq = metrics.get("current_equity")
    dd = metrics.get("max_drawdown_pct")

    subject = f"\u26a0\ufe0f [Trading Alert] {n_fail}/{n_total} Go-Live Criteria FAILING"

    header = (
        f"PAPER TRADING ALERT — {n_fail} Criterion{'s' if n_fail > 1 else ''} FAILING\n"
        f"{'=' * 60}\n"
        f"Scorecard: {sc.get('summary', 'N/A')} | Run: {sc.get('run_date', 'N/A')}\n"
    )
    if eq is not None:
        header += f"Equity: ${eq:,.2f}"
    if dd is not None:
        header += f" | Max DD: {dd:.2f}%"
    header += "\n"

    table = f"\n{'#':<3} {'Metric':<24} {'Threshold':<16} {'Current':<14} {'Status':<10}\n"
    table += "-" * 60 + "\n"
    for c in sc["criteria"]:
        icon = "\u2717" if c["status"] == "FAILING" else (
            "\u2713" if c["status"] == "PASSED" else " "
        )
        table += (
            f"{icon} {c['index']:<2} {c['metric']:<24} "
            f"{c['threshold']:<16} {c['current']:<14} {c['status']:<10}\n"
        )

    failing_list = "\nFAILING:\n"
    for c in failing:
        failing_list += f"  - {c['metric']}: {c['current']} (need {c['threshold']})\n"

    footer = (
        f"\n{'—' * 60}\n"
        f"Auto-generated by Paper Trading Monitor\n"
        f"https://github.com/{GITHUB_REPO}\n"
    )

    return subject, header + table + failing_list + footer


# ── Lambda handler ───────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """Main entry point for EventBridge / manual invocation."""
    print(f"Event: {json.dumps(event)}")

    # 1. Fetch data from GitHub
    summary = github_fetch_json(SUMMARY_PATH)
    scorecard = github_fetch_json(SCORECARD_PATH)

    if not summary and not scorecard:
        print("No data files found in repo — nothing to draft")
        return {"statusCode": 200, "body": "No data available"}

    # 2. Get Gmail credentials and access token
    creds = get_gmail_credentials()
    access_token = refresh_access_token(creds)
    drafts_created = []

    # 3. Create daily summary draft (if summary exists)
    if summary:
        subj, body = format_daily_summary(summary)
        draft_id = create_gmail_draft(access_token, RECIPIENTS, subj, body)
        print(f"Daily summary draft created: {draft_id}")
        drafts_created.append({"type": "daily_summary", "draft_id": draft_id})

    # 4. Create alert draft if any go-live criteria are FAILING
    if scorecard:
        alert = format_alert(scorecard)
        if alert:
            subj, body = alert
            draft_id = create_gmail_draft(access_token, RECIPIENTS, subj, body)
            print(f"Alert draft created: {draft_id}")
            drafts_created.append({"type": "alert", "draft_id": draft_id})
        else:
            print("All go-live criteria PASSED or Pending — no alert draft")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "drafts_created": len(drafts_created),
            "details": drafts_created,
        }),
    }
