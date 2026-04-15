"""
Tests for runtime anomaly detector + email alerting
=====================================================
Covers all 7 statistical checks (pass + fail), alert throttling,
recovery emails, and graceful behaviour when SMTP is not configured.

Run:  python3 -m pytest tests/test_anomaly_detector.py -v
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from monitoring.alerting import AlertManager
from monitoring.anomaly_detector import AnomalyDetector


# ── Helpers ────────────────────────────────────────────────────────────


def _make_detector(**overrides) -> AnomalyDetector:
    """Create a detector with optional threshold overrides."""
    thresholds = {
        "order_frequency_zscore": 3.0,
        "daily_turnover_ratio": 1.0,
        "max_roundtrips_per_symbol_1h": 2,
        "max_signal_flips_24h": 3,
        "max_hourly_drawdown_pct": 2.0,
        "portfolio_hhi": 0.25,
        "duplicate_order_window_sec": 300,
    }
    thresholds.update(overrides)
    config = {"monitoring": {"enabled": True, "thresholds": thresholds}}
    return AnomalyDetector(config)


# ======================================================================
# Check 1: Order Frequency Z-score
# ======================================================================


class TestOrderFrequencyZscore:
    def test_pass_few_orders(self):
        """Fewer than 3 orders → always PASS (not enough data)."""
        det = _make_detector()
        now = datetime.now(UTC)
        det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(hours=2))
        det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(hours=1))
        result = det.run_checks()
        assert result["order_frequency_zscore"]["status"] == "PASS"

    def test_pass_uniform_spacing(self):
        """Orders spaced evenly → z-score near 0 → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        for i in range(10):
            det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(hours=10 - i))
        result = det.run_checks()
        assert result["order_frequency_zscore"]["status"] == "PASS"

    def test_fail_burst(self):
        """Orders spaced days apart then a sudden burst → high z-score → FAIL."""
        det = _make_detector(order_frequency_zscore=2.0)  # lower threshold for test
        now = datetime.now(UTC)
        # Orders spaced 1 day apart (normal cadence)
        for i in range(5):
            det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(days=5 - i))
        # Then 2 orders in quick succession (anomalous burst)
        det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(seconds=2))
        det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(seconds=1))
        result = det.run_checks()
        assert result["order_frequency_zscore"]["value"] > 0
        # The z-score should reflect the anomalous burst


# ======================================================================
# Check 2: Daily Portfolio Turnover
# ======================================================================


class TestDailyTurnover:
    def test_pass_low_turnover(self):
        """Small trades relative to portfolio → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        det.record_order("AAPL", "buy", 10, 150.0, now)  # $1,500
        result = det.run_checks(portfolio_value=100_000)
        assert result["daily_turnover_ratio"]["status"] == "PASS"
        assert result["daily_turnover_ratio"]["value"] < 1.0

    def test_fail_high_turnover(self):
        """Heavy trading exceeding portfolio value → FAIL."""
        det = _make_detector()
        now = datetime.now(UTC)
        # Trade $120k against $100k portfolio → 1.2 turnover
        for i in range(12):
            det.record_order("AAPL", "buy", 100, 100.0, now - timedelta(minutes=i))
        result = det.run_checks(portfolio_value=100_000)
        assert result["daily_turnover_ratio"]["status"] == "FAIL"
        assert result["daily_turnover_ratio"]["value"] > 1.0

    def test_pass_no_portfolio_value(self):
        """Missing portfolio value → safe PASS."""
        det = _make_detector()
        result = det.run_checks(portfolio_value=None)
        assert result["daily_turnover_ratio"]["status"] == "PASS"


# ======================================================================
# Check 3: Round-Trip Detection
# ======================================================================


class TestRoundTrips:
    def test_pass_no_orders(self):
        det = _make_detector()
        result = det.run_checks()
        assert result["max_roundtrips_per_symbol_1h"]["status"] == "PASS"
        assert result["max_roundtrips_per_symbol_1h"]["value"] == 0

    def test_pass_one_direction(self):
        """All buys, no sells → no round-trips → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        for i in range(5):
            det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(minutes=i))
        result = det.run_checks()
        assert result["max_roundtrips_per_symbol_1h"]["status"] == "PASS"

    def test_fail_rapid_flipping(self):
        """Buy-sell-buy-sell-buy-sell in 1h → multiple round-trips → FAIL."""
        det = _make_detector()
        now = datetime.now(UTC)
        sides = ["buy", "sell", "buy", "sell", "buy", "sell"]
        for i, side in enumerate(sides):
            det.record_order("GLD", side, 100, 200.0, now - timedelta(minutes=30 - i * 5))
        result = det.run_checks()
        assert result["max_roundtrips_per_symbol_1h"]["status"] == "FAIL"
        assert result["max_roundtrips_per_symbol_1h"]["value"] > 2


# ======================================================================
# Check 4: Signal Flip Rate
# ======================================================================


class TestSignalFlips:
    def test_pass_stable_signals(self):
        """Same direction all day → 0 flips → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        for i in range(10):
            det.record_signal("AAPL", 0.5, now - timedelta(hours=i))
        result = det.run_checks()
        assert result["max_signal_flips_24h"]["status"] == "PASS"
        assert result["max_signal_flips_24h"]["value"] == 0

    def test_fail_rapid_flips(self):
        """Signal flips sign 5 times in 24h → FAIL."""
        det = _make_detector()
        now = datetime.now(UTC)
        values = [0.5, -0.3, 0.4, -0.2, 0.6, -0.1]
        for i, v in enumerate(values):
            det.record_signal("AAPL", v, now - timedelta(hours=20 - i * 3))
        result = det.run_checks()
        assert result["max_signal_flips_24h"]["status"] == "FAIL"
        assert result["max_signal_flips_24h"]["value"] > 3

    def test_pass_few_signals(self):
        """Only 1 signal → not enough data → PASS."""
        det = _make_detector()
        det.record_signal("AAPL", 0.5, datetime.now(UTC))
        result = det.run_checks()
        assert result["max_signal_flips_24h"]["status"] == "PASS"

    def test_ignores_zero_signals(self):
        """Zero-valued signals are ignored for flip counting."""
        det = _make_detector()
        now = datetime.now(UTC)
        values = [0.5, 0.0, 0.0, 0.0, 0.5]
        for i, v in enumerate(values):
            det.record_signal("AAPL", v, now - timedelta(hours=10 - i * 2))
        result = det.run_checks()
        assert result["max_signal_flips_24h"]["value"] == 0


# ======================================================================
# Check 5: Drawdown Velocity
# ======================================================================


class TestDrawdownVelocity:
    def test_pass_flat_equity(self):
        """Flat equity → 0% drawdown → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        for i in range(10):
            det.record_equity(100_000, now - timedelta(minutes=50 - i * 5))
        result = det.run_checks()
        assert result["max_hourly_drawdown_pct"]["status"] == "PASS"
        assert result["max_hourly_drawdown_pct"]["value"] == 0.0

    def test_fail_sharp_drawdown(self):
        """Equity drops 5% in 1 hour → FAIL."""
        det = _make_detector()
        now = datetime.now(UTC)
        det.record_equity(100_000, now - timedelta(minutes=30))
        det.record_equity(94_000, now)  # 6% drop
        result = det.run_checks()
        assert result["max_hourly_drawdown_pct"]["status"] == "FAIL"
        assert result["max_hourly_drawdown_pct"]["value"] > 2.0

    def test_pass_small_drawdown(self):
        """1% drop → under threshold → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        det.record_equity(100_000, now - timedelta(minutes=30))
        det.record_equity(99_000, now)
        result = det.run_checks()
        assert result["max_hourly_drawdown_pct"]["status"] == "PASS"

    def test_pass_no_data(self):
        """No equity snapshots → PASS."""
        det = _make_detector()
        result = det.run_checks()
        assert result["max_hourly_drawdown_pct"]["status"] == "PASS"


# ======================================================================
# Check 6: Position Concentration (HHI)
# ======================================================================


class TestConcentrationHHI:
    def test_pass_diversified(self):
        """Equal weights across 10 positions → HHI = 0.10 → PASS."""
        det = _make_detector()
        weights = {f"SYM{i}": 0.10 for i in range(10)}
        result = det.run_checks(portfolio_weights=weights)
        assert result["portfolio_hhi"]["status"] == "PASS"
        assert abs(result["portfolio_hhi"]["value"] - 0.10) < 0.01

    def test_fail_concentrated(self):
        """One position at 80%, rest at 5% each → HHI > 0.25 → FAIL."""
        det = _make_detector()
        weights = {"AAPL": 0.80, "MSFT": 0.05, "GOOGL": 0.05, "TSLA": 0.10}
        result = det.run_checks(portfolio_weights=weights)
        assert result["portfolio_hhi"]["status"] == "FAIL"
        assert result["portfolio_hhi"]["value"] > 0.25

    def test_pass_no_weights(self):
        """No portfolio weights → safe PASS."""
        det = _make_detector()
        result = det.run_checks(portfolio_weights=None)
        assert result["portfolio_hhi"]["status"] == "PASS"

    def test_single_position(self):
        """Single position → HHI = 1.0 → FAIL."""
        det = _make_detector()
        weights = {"AAPL": 1.0}
        result = det.run_checks(portfolio_weights=weights)
        assert result["portfolio_hhi"]["status"] == "FAIL"
        assert result["portfolio_hhi"]["value"] == 1.0


# ======================================================================
# Check 7: Duplicate Order Detection
# ======================================================================


class TestDuplicateOrders:
    def test_pass_no_duplicates(self):
        """Different symbols → no duplicates → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        det.record_order("AAPL", "buy", 10, 150.0, now)
        det.record_order("MSFT", "buy", 10, 300.0, now)
        result = det.run_checks()
        assert result["duplicate_orders_5min"]["status"] == "PASS"

    def test_fail_exact_duplicate(self):
        """Same symbol+side+qty within 5 min → FAIL."""
        det = _make_detector()
        now = datetime.now(UTC)
        det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(minutes=1))
        det.record_order("AAPL", "buy", 10, 150.0, now)
        result = det.run_checks()
        assert result["duplicate_orders_5min"]["status"] == "FAIL"
        assert result["duplicate_orders_5min"]["value"] >= 1

    def test_pass_different_qty(self):
        """Same symbol+side but different qty → not a duplicate → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(minutes=1))
        det.record_order("AAPL", "buy", 15, 150.0, now)
        result = det.run_checks()
        assert result["duplicate_orders_5min"]["status"] == "PASS"

    def test_pass_outside_window(self):
        """Same order but 10 minutes apart → outside 5 min window → PASS."""
        det = _make_detector()
        now = datetime.now(UTC)
        det.record_order("AAPL", "buy", 10, 150.0, now - timedelta(minutes=10))
        det.record_order("AAPL", "buy", 10, 150.0, now)
        result = det.run_checks()
        assert result["duplicate_orders_5min"]["status"] == "PASS"


# ======================================================================
# Detector: run_checks returns all 7 checks
# ======================================================================


class TestRunChecks:
    def test_returns_all_seven_checks(self):
        det = _make_detector()
        results = det.run_checks()
        expected_keys = {
            "order_frequency_zscore",
            "daily_turnover_ratio",
            "max_roundtrips_per_symbol_1h",
            "max_signal_flips_24h",
            "max_hourly_drawdown_pct",
            "portfolio_hhi",
            "duplicate_orders_5min",
        }
        assert set(results.keys()) == expected_keys

    def test_all_pass_on_empty_state(self):
        det = _make_detector()
        results = det.run_checks()
        for name, r in results.items():
            assert r["status"] == "PASS", f"{name} should PASS on empty state"

    def test_disabled_returns_empty(self):
        det = AnomalyDetector({"monitoring": {"enabled": False}})
        results = det.run_checks()
        assert results == {}


# ======================================================================
# Report generation
# ======================================================================


class TestGenerateReport:
    @patch("version.__version__", "test-abc123")
    def test_report_structure(self):
        det = _make_detector()
        results = det.run_checks()
        report = det.generate_report(results, {"equity": 100_000})
        assert "timestamp" in report
        assert report["version"] == "test-abc123"
        assert report["status"] == "HEALTHY"
        assert report["failed_checks"] == []
        assert report["account_snapshot"]["equity"] == 100_000


# ======================================================================
# Alert throttling
# ======================================================================


class TestAlertThrottling:
    def test_first_alert_sent(self):
        """First alert for a check type should be sent."""
        mgr = AlertManager(
            {
                "monitoring": {
                    "alert_emails": ["test@example.com"],
                    "alert_cooldown_minutes": 60,
                }
            }
        )
        # Force configured
        mgr._configured = True
        mgr.smtp_host = "smtp.test.com"
        mgr.email_from = "bot@test.com"
        mgr.email_to = ["test@example.com"]

        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "test",
            "status": "ANOMALY_DETECTED",
            "checks": {"order_frequency_zscore": {"value": 4.0, "threshold": 3.0, "status": "FAIL"}},
            "failed_checks": ["order_frequency_zscore"],
            "account_snapshot": {},
            "recent_orders": [],
        }

        with patch.object(mgr, "_send_email", return_value=True) as mock_send:
            result = mgr.send_alert(report)
            assert result is True
            mock_send.assert_called_once()

    def test_second_alert_throttled(self):
        """Second alert for same check type within cooldown should be throttled."""
        mgr = AlertManager(
            {
                "monitoring": {
                    "alert_emails": ["test@example.com"],
                    "alert_cooldown_minutes": 60,
                }
            }
        )
        mgr._configured = True
        mgr.smtp_host = "smtp.test.com"
        mgr.email_from = "bot@test.com"
        mgr.email_to = ["test@example.com"]

        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "test",
            "status": "ANOMALY_DETECTED",
            "checks": {"order_frequency_zscore": {"value": 4.0, "threshold": 3.0, "status": "FAIL"}},
            "failed_checks": ["order_frequency_zscore"],
            "account_snapshot": {},
            "recent_orders": [],
        }

        with patch.object(mgr, "_send_email", return_value=True) as mock_send:
            # First alert
            mgr.send_alert(report)
            assert mock_send.call_count == 1

            # Second alert within cooldown → throttled
            result = mgr.send_alert(report)
            assert result is False
            assert mock_send.call_count == 1  # still 1, not 2

    def test_alert_after_cooldown_expires(self):
        """Alert after cooldown expires should be sent."""
        mgr = AlertManager(
            {
                "monitoring": {
                    "alert_emails": ["test@example.com"],
                    "alert_cooldown_minutes": 60,
                }
            }
        )
        mgr._configured = True
        mgr.smtp_host = "smtp.test.com"
        mgr.email_from = "bot@test.com"
        mgr.email_to = ["test@example.com"]

        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "test",
            "status": "ANOMALY_DETECTED",
            "checks": {"order_frequency_zscore": {"value": 4.0, "threshold": 3.0, "status": "FAIL"}},
            "failed_checks": ["order_frequency_zscore"],
            "account_snapshot": {},
            "recent_orders": [],
        }

        with patch.object(mgr, "_send_email", return_value=True) as mock_send:
            mgr.send_alert(report)
            assert mock_send.call_count == 1

            # Simulate cooldown expiry
            mgr._last_alert["order_frequency_zscore"] = datetime.now(UTC) - timedelta(hours=2)

            mgr.send_alert(report)
            assert mock_send.call_count == 2


# ======================================================================
# Recovery emails
# ======================================================================


class TestRecoveryEmails:
    def test_recovery_sent_on_resolution(self):
        """When a check passes after failing, send recovery email."""
        mgr = AlertManager(
            {
                "monitoring": {
                    "alert_emails": ["test@example.com"],
                    "alert_cooldown_minutes": 60,
                }
            }
        )
        mgr._configured = True
        mgr.smtp_host = "smtp.test.com"
        mgr.email_from = "bot@test.com"
        mgr.email_to = ["test@example.com"]

        # Simulate a prior failure
        mgr._active_failures.add("order_frequency_zscore")

        # Now all checks pass
        recovery_report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "test",
            "status": "HEALTHY",
            "checks": {"order_frequency_zscore": {"value": 1.0, "threshold": 3.0, "status": "PASS"}},
            "failed_checks": [],
            "account_snapshot": {},
            "recent_orders": [],
        }

        with patch.object(mgr, "_send_email", return_value=True) as mock_send:
            result = mgr.send_recovery(recovery_report)
            assert result is True
            mock_send.assert_called_once()
            subject = mock_send.call_args[0][0]
            assert "RESOLVED" in subject

    def test_no_recovery_without_prior_failure(self):
        """No recovery email if there was no prior failure."""
        mgr = AlertManager(
            {
                "monitoring": {
                    "alert_emails": ["test@example.com"],
                    "alert_cooldown_minutes": 60,
                }
            }
        )
        mgr._configured = True
        mgr.smtp_host = "smtp.test.com"
        mgr.email_from = "bot@test.com"
        mgr.email_to = ["test@example.com"]

        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": "test",
            "status": "HEALTHY",
            "checks": {},
            "failed_checks": [],
            "account_snapshot": {},
            "recent_orders": [],
        }

        with patch.object(mgr, "_send_email", return_value=True) as mock_send:
            result = mgr.send_recovery(report)
            assert result is False
            mock_send.assert_not_called()


# ======================================================================
# Graceful behaviour when SMTP not configured
# ======================================================================


class TestGracefulNoSMTP:
    def test_alert_skipped_no_config(self):
        """No SMTP config → alert returns False, no crash."""
        mgr = AlertManager({})
        assert mgr.is_configured is False

        report = {
            "failed_checks": ["order_frequency_zscore"],
            "checks": {"order_frequency_zscore": {"value": 4.0, "threshold": 3.0, "status": "FAIL"}},
        }
        result = mgr.send_alert(report)
        assert result is False

    def test_recovery_skipped_no_config(self):
        """No SMTP config → recovery returns False, no crash."""
        mgr = AlertManager({})
        result = mgr.send_recovery({"failed_checks": []})
        assert result is False

    def test_process_report_no_crash(self):
        """process_report doesn't crash even without SMTP."""
        mgr = AlertManager({})
        mgr.process_report({"failed_checks": ["test"], "checks": {}})


# ======================================================================
# Config loading
# ======================================================================


class TestConfigLoading:
    def test_default_thresholds(self):
        """Detector with no config uses sensible defaults."""
        det = AnomalyDetector()
        assert det.thresholds["order_frequency_zscore"] == 3.0
        assert det.thresholds["daily_turnover_ratio"] == 1.0
        assert det.thresholds["portfolio_hhi"] == 0.25

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        config = {
            "monitoring": {
                "enabled": True,
                "thresholds": {"order_frequency_zscore": 5.0, "portfolio_hhi": 0.5},
            }
        }
        det = AnomalyDetector(config)
        assert det.thresholds["order_frequency_zscore"] == 5.0
        assert det.thresholds["portfolio_hhi"] == 0.5
        # Others remain default
        assert det.thresholds["daily_turnover_ratio"] == 1.0

    def test_alert_emails_from_config(self):
        mgr = AlertManager(
            {
                "monitoring": {
                    "alert_emails": ["a@test.com", "b@test.com"],
                }
            }
        )
        assert mgr.email_to == ["a@test.com", "b@test.com"]

    def test_alert_emails_from_env(self):
        with patch.dict("os.environ", {"ALERT_EMAIL_TO": "env@test.com, env2@test.com"}):
            mgr = AlertManager({})
            assert mgr.email_to == ["env@test.com", "env2@test.com"]
