"""Alert management: Slack, email, and webhook notifications."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert to be sent to configured channels."""

    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.WARNING
    source: str = "market-data-platform"
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


class AlertChannel:
    """Base alert channel."""

    def send(self, alert: Alert) -> bool:
        """Send an alert. Returns True on success."""
        raise NotImplementedError


class SlackChannel(AlertChannel):
    """Slack webhook alert channel.

    Args:
        webhook_url: Slack incoming webhook URL.
        channel: Optional channel override.
        username: Bot username.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        username: str = "Market Data Alerts",
    ) -> None:
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

    def send(self, alert: Alert) -> bool:
        """Send alert to Slack.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.CRITICAL: "#ff0000",
        }

        payload: dict[str, Any] = {
            "username": self.username,
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#cccccc"),
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp, "short": True},
                    ],
                    "footer": "Market Data Platform",
                }
            ],
        }
        if self.channel:
            payload["channel"] = self.channel

        if alert.details:
            detail_text = "\n".join(f"• {k}: {v}" for k, v in alert.details.items())
            payload["attachments"][0]["fields"].append(
                {"title": "Details", "value": detail_text, "short": False}
            )

        return self._post(payload)

    def _post(self, payload: dict[str, Any]) -> bool:
        """POST payload to Slack webhook."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(  # noqa: S310
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                return resp.status == 200
        except Exception as exc:
            logger.error("slack_alert_failed", error=str(exc))
            return False


class WebhookChannel(AlertChannel):
    """Generic HTTP webhook alert channel.

    Args:
        url: Webhook URL.
        headers: Optional extra headers.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.headers = headers or {}

    def send(self, alert: Alert) -> bool:
        """Send alert to webhook endpoint.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        payload = {
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.value,
            "source": alert.source,
            "details": alert.details,
            "timestamp": alert.timestamp,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json", **self.headers}
            req = urllib.request.Request(self.url, data=data, headers=headers)  # noqa: S310
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                return 200 <= resp.status < 300
        except Exception as exc:
            logger.error("webhook_alert_failed", url=self.url, error=str(exc))
            return False


class AlertManager:
    """Central alert manager distributing alerts to configured channels.

    Args:
        channels: List of alert channels.
        min_severity: Minimum severity to forward.
    """

    def __init__(
        self,
        channels: list[AlertChannel] | None = None,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
    ) -> None:
        self._channels: list[AlertChannel] = channels or []
        self._min_severity = min_severity
        self._history: list[Alert] = []

    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert channel.

        Args:
            channel: Channel to add.
        """
        self._channels.append(channel)

    def send_alert(self, alert: Alert) -> int:
        """Send an alert to all configured channels.

        Args:
            alert: Alert to send.

        Returns:
            Number of channels that successfully received the alert.
        """
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
        }

        if severity_order.get(alert.severity, 0) < severity_order.get(self._min_severity, 0):
            logger.debug("alert_below_threshold", title=alert.title, severity=alert.severity.value)
            return 0

        self._history.append(alert)

        sent = 0
        for channel in self._channels:
            try:
                if channel.send(alert):
                    sent += 1
            except Exception as exc:
                logger.error(
                    "alert_channel_error",
                    channel=type(channel).__name__,
                    error=str(exc),
                )

        logger.info(
            "alert_sent",
            title=alert.title,
            severity=alert.severity.value,
            channels_succeeded=sent,
            channels_total=len(self._channels),
        )
        return sent

    def alert_info(self, title: str, message: str, **details: Any) -> int:
        """Send an info alert.

        Args:
            title: Alert title.
            message: Alert message.
            **details: Extra details.

        Returns:
            Number of channels notified.
        """
        return self.send_alert(
            Alert(title=title, message=message, severity=AlertSeverity.INFO, details=details)
        )

    def alert_warning(self, title: str, message: str, **details: Any) -> int:
        """Send a warning alert.

        Args:
            title: Alert title.
            message: Alert message.
            **details: Extra details.

        Returns:
            Number of channels notified.
        """
        return self.send_alert(
            Alert(title=title, message=message, severity=AlertSeverity.WARNING, details=details)
        )

    def alert_critical(self, title: str, message: str, **details: Any) -> int:
        """Send a critical alert.

        Args:
            title: Alert title.
            message: Alert message.
            **details: Extra details.

        Returns:
            Number of channels notified.
        """
        return self.send_alert(
            Alert(title=title, message=message, severity=AlertSeverity.CRITICAL, details=details)
        )

    @property
    def history(self) -> list[Alert]:
        """Alert history."""
        return list(self._history)

    @property
    def channel_count(self) -> int:
        """Number of configured channels."""
        return len(self._channels)
