"""
Email Alerting for Anomaly Detector
====================================
Sends HTML-formatted alert emails when anomaly checks fail.
Uses stdlib smtplib + email.mime — no external dependencies.

Configuration via environment variables:
  ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASS,
  ALERT_EMAIL_FROM, ALERT_EMAIL_TO

Throttling: max 1 alert per anomaly type per hour (configurable).
Recovery: sends a RESOLVED email when a check passes after failing.
"""

from __future__ import annotations

import os
import smtplib
from datetime import UTC, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from utils.logger import get_logger

log = get_logger("AlertManager")


class AlertManager:
    """Manages email alerts with throttling and recovery notifications."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("monitoring", {})

        # SMTP config: env vars take precedence, then config dict
        self.smtp_host = os.environ.get("ALERT_SMTP_HOST", cfg.get("smtp_host", ""))
        self.smtp_port = int(os.environ.get("ALERT_SMTP_PORT", cfg.get("smtp_port", 587)))
        self.smtp_user = os.environ.get("ALERT_SMTP_USER", cfg.get("smtp_user", ""))
        self.smtp_pass = os.environ.get("ALERT_SMTP_PASS", cfg.get("smtp_pass", ""))
        self.email_from = os.environ.get("ALERT_EMAIL_FROM", cfg.get("email_from", ""))

        # Recipients: env var (comma-separated) or config list
        env_to = os.environ.get("ALERT_EMAIL_TO", "")
        if env_to:
            self.email_to = [e.strip() for e in env_to.split(",") if e.strip()]
        else:
            self.email_to = cfg.get("alert_emails", [])

        # Throttle: max 1 email per check type per cooldown period
        cooldown_min = cfg.get("alert_cooldown_minutes", 60)
        self._cooldown = timedelta(minutes=cooldown_min)
        self._last_alert: dict[str, datetime] = {}

        # Track which checks are currently in FAIL state (for recovery emails)
        self._active_failures: set[str] = set()

        self._configured = bool(self.smtp_host and self.email_from and self.email_to)
        if not self._configured:
            log.info("Email alerting not configured (missing SMTP env vars) — alerts disabled")

    @property
    def is_configured(self) -> bool:
        return self._configured

    def send_alert(self, report: dict[str, Any]) -> bool:
        """Send anomaly alert email. Returns True if sent, False if throttled/unconfigured."""
        if not self._configured:
            log.debug("Alert skipped — email not configured")
            return False

        failed = report.get("failed_checks", [])
        if not failed:
            return False

        # Throttle: skip if all failed checks were alerted recently
        now = datetime.now(UTC)
        new_failures = []
        for check in failed:
            last = self._last_alert.get(check)
            if last is None or (now - last) >= self._cooldown:
                new_failures.append(check)

        if not new_failures:
            log.debug(f"Alert throttled — all {len(failed)} failures alerted within cooldown")
            return False

        subject = f"[TRADING ALERT] Anomaly detected — {', '.join(new_failures)}"
        body = self._build_alert_html(report, new_failures)

        sent = self._send_email(subject, body)
        if sent:
            for check in new_failures:
                self._last_alert[check] = now
                self._active_failures.add(check)
        return sent

    def send_recovery(self, report: dict[str, Any]) -> bool:
        """Send recovery email for checks that passed after failing."""
        if not self._configured:
            return False

        failed = set(report.get("failed_checks", []))
        recovered = self._active_failures - failed
        if not recovered:
            return False

        subject = f"[TRADING RESOLVED] {', '.join(sorted(recovered))} recovered"
        body = self._build_recovery_html(report, sorted(recovered))

        sent = self._send_email(subject, body)
        if sent:
            self._active_failures -= recovered
        return sent

    def process_report(self, report: dict[str, Any]) -> None:
        """Handle a report: send alert if anomalies, send recovery if resolved."""
        failed = report.get("failed_checks", [])
        if failed:
            self.send_alert(report)
        # Always check for recoveries
        self.send_recovery(report)

    # ── Email building ─────────────────────────────────────────────────

    def _build_alert_html(self, report: dict[str, Any], new_failures: list[str]) -> str:
        checks = report.get("checks", {})
        snapshot = report.get("account_snapshot", {})
        version = report.get("version", "unknown")
        recent_orders = report.get("recent_orders", [])

        checks_html = ""
        for name, result in checks.items():
            status = result.get("status", "PASS")
            color = "#d32f2f" if status == "FAIL" else "#388e3c"
            marker = "&#10060;" if status == "FAIL" else "&#9989;"
            checks_html += (
                f"<tr>"
                f"<td>{marker} <b>{name}</b></td>"
                f"<td style='color:{color};font-weight:bold'>{status}</td>"
                f"<td>{result.get('value', 'N/A')}</td>"
                f"<td>{result.get('threshold', 'N/A')}</td>"
                f"</tr>"
            )

        snapshot_html = ""
        if snapshot:
            for k, v in snapshot.items():
                snapshot_html += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"

        orders_html = ""
        for o in recent_orders[-10:]:
            ts = o.get("ts", "")
            if hasattr(ts, "strftime"):
                ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            orders_html += (
                f"<tr>"
                f"<td>{ts}</td>"
                f"<td>{o.get('symbol', '')}</td>"
                f"<td>{o.get('side', '').upper()}</td>"
                f"<td>{o.get('qty', '')}</td>"
                f"<td>${o.get('dollar_value', 0):,.2f}</td>"
                f"</tr>"
            )

        return f"""
        <html><body style="font-family:monospace;font-size:14px;color:#222;">
        <h2 style="color:#d32f2f;">Trading Anomaly Detected</h2>
        <p><b>Time:</b> {report.get('timestamp', '')}</p>
        <p><b>Version:</b> {version}</p>
        <p><b>Failed checks:</b> {', '.join(new_failures)}</p>

        <h3>Health Check Results</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
        <tr style="background:#f5f5f5;">
            <th>Check</th><th>Status</th><th>Value</th><th>Threshold</th>
        </tr>
        {checks_html}
        </table>

        <h3>Account Snapshot</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
        {snapshot_html}
        </table>

        <h3>Recent Orders (last 10)</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
        <tr style="background:#f5f5f5;">
            <th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Value</th>
        </tr>
        {orders_html}
        </table>

        <h3>Recommended Action</h3>
        <p>Review the failed checks above. If the anomaly persists, consider
        pausing the trading engine and investigating root cause.</p>

        <hr><p style="color:#888;font-size:12px;">
        Automated alert from trading bot v{version}
        </p></body></html>
        """

    def _build_recovery_html(self, report: dict[str, Any], recovered: list[str]) -> str:
        version = report.get("version", "unknown")
        return f"""
        <html><body style="font-family:monospace;font-size:14px;color:#222;">
        <h2 style="color:#388e3c;">Trading Anomaly Resolved</h2>
        <p><b>Time:</b> {report.get('timestamp', '')}</p>
        <p><b>Version:</b> {version}</p>
        <p><b>Recovered checks:</b> {', '.join(recovered)}</p>
        <p>The following checks have returned to normal:
        <b>{', '.join(recovered)}</b></p>
        <hr><p style="color:#888;font-size:12px;">
        Automated alert from trading bot v{version}
        </p></body></html>
        """

    # ── SMTP sending ───────────────────────────────────────────────────

    def _send_email(self, subject: str, html_body: str) -> bool:
        """Send an HTML email via SMTP. Returns True on success."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                if self.smtp_user and self.smtp_pass:
                    server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.email_from, self.email_to, msg.as_string())

            log.info(f"Alert email sent: {subject}")
            return True
        except Exception as exc:
            log.error(f"Failed to send alert email: {exc}")
            return False
