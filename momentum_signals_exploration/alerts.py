#!/usr/bin/env python3
"""
Alert system for momentum scanner results.

Send notifications via Slack, Email, or custom webhooks.
"""

import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage alert delivery."""

    def __init__(self, config: dict = None):
        """
        Initialize alert manager.

        Args:
            config: Configuration dict with alert settings
        """
        self.config = config or {}
        self.alerts_sent = []

    def send_slack_alert(
        self, gainers: list[tuple], losers: list[tuple], webhook_url: str = None
    ) -> bool:
        """
        Send alert to Slack.

        Args:
            gainers: Top gainers list
            losers: Top losers list
            webhook_url: Slack webhook URL

        Returns:
            True if successful
        """
        webhook_url = webhook_url or self.config.get("slack_webhook")

        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        try:
            message = self._format_slack_message(gainers, losers)
            response = requests.post(webhook_url, json=message, timeout=10)

            if response.status_code == 200:
                logger.info("✓ Slack alert sent")
                self.alerts_sent.append(("slack", datetime.now()))
                return True
            logger.error(f"✗ Slack alert failed: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"✗ Slack alert error: {e}")
            return False

    def _format_slack_message(self, gainers: list[tuple], losers: list[tuple]) -> dict:
        """Format message for Slack."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Top 5 gainers
        gainers_text = "\n".join(
            [
                f"• {symbol}: {metrics['intra_momentum'] * 100:+.2f}% (${metrics['price']:.2f})"
                for symbol, metrics in gainers[:5]
            ]
        )

        # Top 5 losers
        losers_text = "\n".join(
            [
                f"• {symbol}: {metrics['intra_momentum'] * 100:+.2f}% (${metrics['price']:.2f})"
                for symbol, metrics in losers[:5]
            ]
        )

        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📊 Hourly Momentum Scan - {timestamp}",
                    },
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Top 5 Gainers:*\n{gainers_text}"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Top 5 Losers:*\n{losers_text}"},
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Scanned {len(gainers) + len(losers)} symbols"}
                    ],
                },
            ]
        }

        return message

    def send_email_alert(
        self,
        gainers: list[tuple],
        losers: list[tuple],
        to_email: str = None,
        smtp_config: dict = None,
    ) -> bool:
        """
        Send alert via email.

        Args:
            gainers: Top gainers list
            losers: Top losers list
            to_email: Recipient email
            smtp_config: SMTP configuration

        Returns:
            True if successful
        """
        to_email = to_email or self.config.get("email_to")
        smtp_config = smtp_config or self.config.get("smtp_config", {})

        if not to_email or not smtp_config:
            logger.warning("Email not configured")
            return False

        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"Hourly Momentum Scan - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg["From"] = smtp_config.get("from_email")
            msg["To"] = to_email

            # HTML body
            html = self._format_email_html(gainers, losers)
            part = MIMEText(html, "html")
            msg.attach(part)

            # Send
            with smtplib.SMTP_SSL(smtp_config.get("host"), smtp_config.get("port")) as server:
                server.login(smtp_config.get("username"), smtp_config.get("password"))
                server.send_message(msg)

            logger.info(f"✓ Email alert sent to {to_email}")
            self.alerts_sent.append(("email", datetime.now()))
            return True

        except Exception as e:
            logger.error(f"✗ Email alert failed: {e}")
            return False

    def _format_email_html(self, gainers: list[tuple], losers: list[tuple]) -> str:
        """Format HTML email body."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        gainers_rows = "\n".join(
            [
                f"<tr><td>{symbol}</td><td>{metrics['intra_momentum'] * 100:+.2f}%</td>"
                f"<td>${metrics['price']:.2f}</td><td>{metrics['volume']:,.0f}</td></tr>"
                for symbol, metrics in gainers[:10]
            ]
        )

        losers_rows = "\n".join(
            [
                f"<tr><td>{symbol}</td><td>{metrics['intra_momentum'] * 100:+.2f}%</td>"
                f"<td>${metrics['price']:.2f}</td><td>{metrics['volume']:,.0f}</td></tr>"
                for symbol, metrics in losers[:10]
            ]
        )

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h2>📊 Hourly Momentum Scan</h2>
            <p><strong>Time:</strong> {timestamp}</p>

            <h3 style="color: green;">🚀 Top Gainers</h3>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Momentum</th>
                    <th>Price</th>
                    <th>Volume</th>
                </tr>
                {gainers_rows}
            </table>

            <h3 style="color: red;">📉 Top Losers</h3>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Momentum</th>
                    <th>Price</th>
                    <th>Volume</th>
                </tr>
                {losers_rows}
            </table>

            <p style="font-size: 0.9em; color: #666;">
                Automated hourly momentum scanner
            </p>
        </body>
        </html>
        """

        return html

    def send_webhook_alert(
        self, gainers: list[tuple], losers: list[tuple], webhook_url: str = None
    ) -> bool:
        """
        Send alert to custom webhook.

        Args:
            gainers: Top gainers list
            losers: Top losers list
            webhook_url: Webhook URL

        Returns:
            True if successful
        """
        webhook_url = webhook_url or self.config.get("webhook_url")

        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return False

        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "gainers": [
                    {
                        "symbol": symbol,
                        "momentum": metrics["intra_momentum"],
                        "price": metrics["price"],
                        "volume": metrics["volume"],
                    }
                    for symbol, metrics in gainers[:10]
                ],
                "losers": [
                    {
                        "symbol": symbol,
                        "momentum": metrics["intra_momentum"],
                        "price": metrics["price"],
                        "volume": metrics["volume"],
                    }
                    for symbol, metrics in losers[:10]
                ],
            }

            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code in [200, 201]:
                logger.info("✓ Webhook alert sent")
                self.alerts_sent.append(("webhook", datetime.now()))
                return True
            logger.error(f"✗ Webhook failed: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"✗ Webhook error: {e}")
            return False

    def print_alert(self, gainers: list[tuple], losers: list[tuple]) -> None:
        """Print results to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * 100)
        print(f"HOURLY MOMENTUM SCAN - {timestamp}")
        print("=" * 100)

        print("\n🚀 TOP GAINERS:")
        print("-" * 100)
        print(f"{'Symbol':<8} {'Momentum':<12} {'Hourly %':<12} {'Price':<10} {'Volume':<15}")
        print("-" * 100)
        for symbol, metrics in gainers[:10]:
            print(
                f"{symbol:<8} "
                f"{metrics['intra_momentum'] * 100:>10.2f}% "
                f"{metrics['hourly_return'] * 100:>10.2f}% "
                f"${metrics['price']:>8.2f} "
                f"{metrics['volume']:>13,.0f}"
            )

        print("\n📉 TOP LOSERS:")
        print("-" * 100)
        print(f"{'Symbol':<8} {'Momentum':<12} {'Hourly %':<12} {'Price':<10} {'Volume':<15}")
        print("-" * 100)
        for symbol, metrics in losers[:10]:
            print(
                f"{symbol:<8} "
                f"{metrics['intra_momentum'] * 100:>10.2f}% "
                f"{metrics['hourly_return'] * 100:>10.2f}% "
                f"${metrics['price']:>8.2f} "
                f"{metrics['volume']:>13,.0f}"
            )

        print("\n" + "=" * 100)

    def send_all_alerts(self, gainers: list[tuple], losers: list[tuple]) -> dict:
        """
        Send alerts via all configured channels.

        Args:
            gainers: Top gainers
            losers: Top losers

        Returns:
            Dict with status of each alert type
        """
        results = {}

        # Console
        self.print_alert(gainers, losers)
        results["console"] = True

        # Slack
        if self.config.get("slack_webhook"):
            results["slack"] = self.send_slack_alert(gainers, losers)

        # Email
        if self.config.get("email_to"):
            results["email"] = self.send_email_alert(gainers, losers)

        # Webhook
        if self.config.get("webhook_url"):
            results["webhook"] = self.send_webhook_alert(gainers, losers)

        logger.info(f"Alert status: {results}")
        return results
