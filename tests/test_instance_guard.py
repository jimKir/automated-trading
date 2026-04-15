"""
Tests for startup instance guard + version stamping
=====================================================
Feature 1: cancel_all_open_orders() in AlpacaBroker
Feature 2: _cleanup_stale_orders() called during LiveEngine.__init__
Feature 3: version.py resolution (env var -> git -> fallback)

Run:  python3 -m pytest tests/test_instance_guard.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── AlpacaBroker.cancel_all_open_orders ──────────────────────────────────


class TestCancelAllOpenOrders:
    """Tests for AlpacaBroker.cancel_all_open_orders()."""

    def _make_broker(self):
        from execution.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker({"brokers": {"alpaca": {"api_key": "k", "api_secret": "s"}}})
        broker.trading_client = MagicMock()
        return broker

    def test_cancels_and_returns_count(self):
        broker = self._make_broker()
        # Simulate 3 cancelled orders
        broker.trading_client.cancel_orders.return_value = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        assert broker.cancel_all_open_orders() == 3
        broker.trading_client.cancel_orders.assert_called_once()

    def test_returns_zero_when_no_orders(self):
        broker = self._make_broker()
        broker.trading_client.cancel_orders.return_value = []
        assert broker.cancel_all_open_orders() == 0

    def test_returns_zero_on_none(self):
        broker = self._make_broker()
        broker.trading_client.cancel_orders.return_value = None
        assert broker.cancel_all_open_orders() == 0

    def test_returns_zero_on_exception(self):
        broker = self._make_broker()
        broker.trading_client.cancel_orders.side_effect = Exception("API timeout")
        assert broker.cancel_all_open_orders() == 0


# ── LiveEngine._cleanup_stale_orders called on init ─────────────────────


class TestCleanupStaleOrdersOnInit:
    """Verify _cleanup_stale_orders() is wired into LiveEngine.__init__."""

    @patch("execution.live_engine.get_broker")
    @patch("execution.live_engine.DataFeed")
    @patch("execution.live_engine.SignalGenerator")
    @patch("execution.live_engine.RiskManager")
    def test_cleanup_called_during_init(self, mock_risk, mock_sig, mock_feed, mock_get_broker):
        """_cleanup_stale_orders should be called once during __init__."""
        broker = MagicMock()
        broker.cancel_all_open_orders.return_value = 2
        broker.get_last_filled_order_time.return_value = None
        mock_get_broker.return_value = broker

        config = {"system": {"mode": "paper"}, "strategy": {}}
        _make_engine(config, mock_get_broker, mock_feed, mock_sig, mock_risk)

        broker.cancel_all_open_orders.assert_called_once()

    @patch("execution.live_engine.get_broker")
    @patch("execution.live_engine.DataFeed")
    @patch("execution.live_engine.SignalGenerator")
    @patch("execution.live_engine.RiskManager")
    def test_cleanup_tolerates_broker_without_method(
        self, mock_risk, mock_sig, mock_feed, mock_get_broker
    ):
        """If broker lacks cancel_all_open_orders, init should not crash."""
        broker = MagicMock(spec=[])  # No methods at all
        mock_get_broker.return_value = broker

        config = {"system": {"mode": "paper"}, "strategy": {}}
        # Should not raise
        engine = _make_engine(config, mock_get_broker, mock_feed, mock_sig, mock_risk)
        assert engine is not None

    @patch("execution.live_engine.get_broker")
    @patch("execution.live_engine.DataFeed")
    @patch("execution.live_engine.SignalGenerator")
    @patch("execution.live_engine.RiskManager")
    def test_cleanup_exception_does_not_crash_init(
        self, mock_risk, mock_sig, mock_feed, mock_get_broker
    ):
        """If cancel_all_open_orders raises, init should not crash."""
        broker = MagicMock()
        broker.cancel_all_open_orders.side_effect = Exception("network down")
        broker.get_last_filled_order_time.return_value = None
        mock_get_broker.return_value = broker

        config = {"system": {"mode": "paper"}, "strategy": {}}
        engine = _make_engine(config, mock_get_broker, mock_feed, mock_sig, mock_risk)
        assert engine is not None


def _make_engine(config, mock_get_broker, mock_feed, mock_sig, mock_risk):
    """Helper to create a LiveEngine with all heavy subsystems mocked."""
    from execution.live_engine import LiveEngine

    return LiveEngine(config=config, dry_run=True)


# ── Version resolution ───────────────────────────────────────────────────


class TestVersionResolution:
    """Tests for version.py get_version() priority chain."""

    def test_env_var_takes_priority(self):
        with patch.dict("os.environ", {"BUILD_VERSION": "paper-abc1234-20260415"}):
            from version import get_version

            assert get_version() == "paper-abc1234-20260415"

    def test_git_sha_fallback(self):
        with (
            patch.dict("os.environ", {"BUILD_VERSION": ""}, clear=False),
            patch("subprocess.check_output", return_value=b"abc1234\n"),
        ):
            from version import get_version

            assert get_version() == "dev-abc1234"

    def test_dev_unknown_fallback(self):
        with (
            patch.dict("os.environ", {"BUILD_VERSION": ""}, clear=False),
            patch("subprocess.check_output", side_effect=FileNotFoundError),
        ):
            from version import get_version

            assert get_version() == "dev-unknown"

    def test_git_empty_output_falls_through(self):
        with (
            patch.dict("os.environ", {"BUILD_VERSION": ""}, clear=False),
            patch("subprocess.check_output", return_value=b"   \n"),
        ):
            from version import get_version

            # Empty git output should fall through to dev-unknown
            result = get_version()
            # strip() on whitespace yields "", which is falsy → fallback
            assert result == "dev-unknown"
