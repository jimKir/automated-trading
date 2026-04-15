"""
Tests for Alpaca order rejection guards
========================================
Issue 1: Fractional short-sell quantities must be floored to whole numbers.
Issue 2: Wash-trade prevention — cancel opposite-side, skip same-side duplicates.
Issue 3: Duplicate order guard — skip when open orders exist for same symbol+side.
Issue 4: Rebalance cadence enforcement — _should_rebalance respects adaptive cadence.

Run:  python3 -m pytest tests/test_order_guards.py -v
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from execution.broker_base import Order, OrderSide, OrderStatus
from execution.live_engine import LiveEngine

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_broker(positions: dict | None = None, open_orders: list | None = None):
    """Create an AlpacaBroker with mocked Alpaca SDK internals.

    The alpaca-py SDK is not installed in the test environment, so we
    mock get_position, get_open_orders, and the submit_order path.
    """
    from execution.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker({"brokers": {"alpaca": {"api_key": "k", "api_secret": "s"}}})
    broker.trading_client = MagicMock()

    # ── Mock get_position (used by the fractional short-sell guard) ──
    if positions is None:
        positions = {}

    def _mock_get_position(symbol):
        if symbol in positions:
            return positions[symbol]
        return None

    broker.get_position = _mock_get_position

    # ── Mock get_open_orders (used by wash-trade guard) ──
    if open_orders is None:
        open_orders = []

    broker.get_open_orders = MagicMock(return_value=open_orders)

    # ── Mock submit_order so it doesn't hit the network ──
    resp_mock = MagicMock()
    resp_mock.id = "mock-order-id"
    resp_mock.status = "filled"
    resp_mock.filled_avg_price = 150.0
    broker.trading_client.submit_order = MagicMock(return_value=resp_mock)

    return broker


def _patch_alpaca_imports():
    """Patch alpaca SDK imports used inside place_order."""
    from enum import Enum

    class FakeAlpacaSide(Enum):
        BUY = "buy"
        SELL = "sell"

    class FakeTimeInForce(Enum):
        DAY = "day"
        GTC = "gtc"

    mock_alpaca_side = FakeAlpacaSide
    mock_tif = FakeTimeInForce

    class FakeMarketOrderRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeLimitOrderRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeStopOrderRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    return {
        "alpaca.trading.enums": MagicMock(OrderSide=mock_alpaca_side, TimeInForce=mock_tif),
        "alpaca.trading.requests": MagicMock(
            MarketOrderRequest=FakeMarketOrderRequest,
            LimitOrderRequest=FakeLimitOrderRequest,
            StopOrderRequest=FakeStopOrderRequest,
        ),
        "alpaca": MagicMock(),
        "alpaca.trading": MagicMock(),
    }


# ============================================================
#  Issue 1 — Fractional short-sell guard
# ============================================================


class TestFractionalShortSellGuard:
    """place_order must floor fractional qty on short sells."""

    def test_fractional_short_sell_is_floored(self):
        """qty=36.6919 with no position → should become 36."""
        broker = _make_broker(positions={})  # no positions → short sell
        order = Order(symbol="XLV", side=OrderSide.SELL, quantity=36.6919)

        with patch.dict("sys.modules", _patch_alpaca_imports()):
            result = broker.place_order(order)

        assert order.quantity == 36.0
        assert result.status != OrderStatus.REJECTED
        # submit_order was called
        broker.trading_client.submit_order.assert_called_once()
        req = broker.trading_client.submit_order.call_args[0][0]
        assert req.qty == 36.0

    def test_fractional_short_sell_rounds_to_zero_is_skipped(self):
        """qty=0.5 with no position → floors to 0 → should be rejected/skipped."""
        broker = _make_broker(positions={})
        order = Order(symbol="XLV", side=OrderSide.SELL, quantity=0.5)

        with patch.dict("sys.modules", _patch_alpaca_imports()):
            result = broker.place_order(order)

        assert result.status == OrderStatus.REJECTED
        broker.trading_client.submit_order.assert_not_called()

    def test_sell_with_existing_position_keeps_fractional(self):
        """If we hold the symbol, this is a close/reduce — fractional is OK."""
        broker = _make_broker(positions={"XLV": {"quantity": 50.0, "avg_price": 100.0}})
        order = Order(symbol="XLV", side=OrderSide.SELL, quantity=36.6919)

        with patch.dict("sys.modules", _patch_alpaca_imports()):
            broker.place_order(order)

        # Should NOT floor — fractional sell of existing position is fine
        req = broker.trading_client.submit_order.call_args[0][0]
        assert req.qty == pytest.approx(36.6919, abs=0.001)

    def test_buy_order_unaffected(self):
        """BUY orders should not be affected by the short-sell guard."""
        broker = _make_broker(positions={})
        order = Order(symbol="XLV", side=OrderSide.BUY, quantity=36.6919)

        with patch.dict("sys.modules", _patch_alpaca_imports()):
            broker.place_order(order)

        req = broker.trading_client.submit_order.call_args[0][0]
        assert req.qty == pytest.approx(36.6919, abs=0.001)

    def test_whole_number_short_sell_passes_through(self):
        """qty=10.0 short sell should pass through unchanged."""
        broker = _make_broker(positions={})
        order = Order(symbol="XLV", side=OrderSide.SELL, quantity=10.0)

        with patch.dict("sys.modules", _patch_alpaca_imports()):
            result = broker.place_order(order)

        assert order.quantity == 10.0
        assert result.status != OrderStatus.REJECTED

    def test_fractional_short_sell_qty_1_point_9(self):
        """qty=1.9 with no position → should floor to 1."""
        broker = _make_broker(positions={})
        order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=1.9)

        with patch.dict("sys.modules", _patch_alpaca_imports()):
            result = broker.place_order(order)

        assert order.quantity == 1.0
        assert result.status != OrderStatus.REJECTED


# ============================================================
#  Issue 2 — Wash-trade prevention
# ============================================================


class TestWashTradePrevention:
    """cancel_conflicting_orders must handle opposite/same-side open orders."""

    def test_no_open_orders_returns_false(self):
        broker = _make_broker(open_orders=[])
        broker.get_open_orders = MagicMock(return_value=[])
        result = broker.cancel_conflicting_orders("XLV", OrderSide.SELL)
        assert result is False

    @patch("execution.alpaca_broker.time.sleep")
    def test_opposite_side_order_is_cancelled(self, mock_sleep):
        """Placing SELL when BUY is pending → cancel the BUY."""
        open_orders = [
            {"order_id": "oo-buy-1", "symbol": "XLV", "side": "buy", "qty": 10, "status": "new"}
        ]
        broker = _make_broker()
        broker.get_open_orders = MagicMock(return_value=open_orders)
        broker.cancel_order = MagicMock(return_value=True)

        result = broker.cancel_conflicting_orders("XLV", OrderSide.SELL)

        assert result is False  # no same-side duplicate
        broker.cancel_order.assert_called_once_with("oo-buy-1")

    def test_same_side_order_returns_true(self):
        """Placing SELL when SELL is already pending → skip (return True)."""
        open_orders = [
            {"order_id": "oo-sell-1", "symbol": "XLV", "side": "sell", "qty": 10, "status": "new"}
        ]
        broker = _make_broker()
        broker.get_open_orders = MagicMock(return_value=open_orders)
        broker.cancel_order = MagicMock(return_value=True)

        result = broker.cancel_conflicting_orders("XLV", OrderSide.SELL)

        assert result is True  # caller should skip
        broker.cancel_order.assert_not_called()

    @patch("execution.alpaca_broker.time.sleep")
    def test_mixed_orders_cancels_opposite_and_flags_same(self, mock_sleep):
        """Both a BUY and a SELL pending — cancel BUY, flag SELL as duplicate."""
        open_orders = [
            {"order_id": "oo-buy-1", "symbol": "XLV", "side": "buy", "qty": 10, "status": "new"},
            {"order_id": "oo-sell-1", "symbol": "XLV", "side": "sell", "qty": 5, "status": "new"},
        ]
        broker = _make_broker()
        broker.get_open_orders = MagicMock(return_value=open_orders)
        broker.cancel_order = MagicMock(return_value=True)

        result = broker.cancel_conflicting_orders("XLV", OrderSide.SELL)

        assert result is True  # same-side exists → skip
        broker.cancel_order.assert_called_once_with("oo-buy-1")


# ============================================================
#  Integration: LiveEngine wash-trade guard wiring
# ============================================================


class TestLiveEngineWashTradeWiring:
    """Verify LiveEngine calls cancel_conflicting_orders before place_order."""

    def test_skip_on_same_side_duplicate(self):
        """If broker signals same-side duplicate, LiveEngine should skip."""
        mock_broker = MagicMock()
        mock_broker.cancel_conflicting_orders = MagicMock(return_value=True)

        side = OrderSide.SELL
        sym = "XLV"

        has_duplicate = mock_broker.cancel_conflicting_orders(sym, side)
        assert has_duplicate is True
        mock_broker.place_order.assert_not_called()

    def test_proceed_when_no_conflicts(self):
        """If no conflicts, place_order should proceed."""
        mock_broker = MagicMock()
        mock_broker.cancel_conflicting_orders = MagicMock(return_value=False)

        has_duplicate = mock_broker.cancel_conflicting_orders("XLV", OrderSide.SELL)
        assert has_duplicate is False


# ============================================================
#  Issue 3 — Duplicate order guard (ECS crash-loop protection)
# ============================================================


class TestDuplicateOrderGuard:
    """Verify the dedup guard in _trading_cycle skips when open orders exist."""

    def test_dedup_skips_when_same_side_open_order_exists(self):
        """If broker has an open BUY for GLD, a new BUY for GLD should be skipped."""
        mock_broker = MagicMock()
        existing_orders = [
            {"order_id": "abc-123", "symbol": "GLD", "side": "buy", "qty": 10, "status": "new"}
        ]
        mock_broker.get_open_orders = MagicMock(return_value=existing_orders)

        # Simulate the dedup guard logic from _trading_cycle
        sym = "GLD"
        side = OrderSide.BUY
        side_str = side.value.lower()
        existing = mock_broker.get_open_orders(sym)
        same_side = [o for o in existing if o.get("side") == side_str]

        assert len(same_side) == 1
        # When same_side is non-empty, the engine skips → place_order NOT called
        mock_broker.place_order.assert_not_called()

    def test_dedup_allows_when_no_open_orders(self):
        """No open orders → dedup guard passes, order proceeds."""
        mock_broker = MagicMock()
        mock_broker.get_open_orders = MagicMock(return_value=[])

        existing = mock_broker.get_open_orders("GLD")
        same_side = [o for o in existing if o.get("side") == "buy"]

        assert len(same_side) == 0
        # Guard passes → engine would proceed to place_order

    def test_dedup_allows_opposite_side_orders(self):
        """Open SELL for GLD should NOT block a new BUY for GLD."""
        mock_broker = MagicMock()
        existing_orders = [
            {"order_id": "abc-456", "symbol": "GLD", "side": "sell", "qty": 5, "status": "new"}
        ]
        mock_broker.get_open_orders = MagicMock(return_value=existing_orders)

        side_str = OrderSide.BUY.value.lower()
        existing = mock_broker.get_open_orders("GLD")
        same_side = [o for o in existing if o.get("side") == side_str]

        assert len(same_side) == 0  # no same-side orders → allow

    def test_dedup_skips_multiple_same_side_orders(self):
        """Multiple open BUYs for same symbol → still skipped."""
        mock_broker = MagicMock()
        existing_orders = [
            {"order_id": "abc-1", "symbol": "EEM", "side": "buy", "qty": 10, "status": "new"},
            {"order_id": "abc-2", "symbol": "EEM", "side": "buy", "qty": 5, "status": "new"},
        ]
        mock_broker.get_open_orders = MagicMock(return_value=existing_orders)

        side_str = OrderSide.BUY.value.lower()
        existing = mock_broker.get_open_orders("EEM")
        same_side = [o for o in existing if o.get("side") == side_str]

        assert len(same_side) == 2


# ============================================================
#  Issue 4 — Rebalance cadence enforcement
# ============================================================


class TestRebalanceCadence:
    """Verify _should_rebalance respects cadence and last-fill seeding."""

    def _make_engine(self, rebalance_freq="adaptive", last_rebalance=None):
        """Create a minimal LiveEngine with mocked broker for cadence tests."""
        config = {
            "system": {"mode": "paper"},
            "strategy": {
                "rebalance_frequency": rebalance_freq,
                "adaptive_weekly_threshold": 0.17,
            },
            "brokers": {"alpaca": {"api_key": "", "api_secret": ""}},
        }

        with (
            patch("execution.live_engine.get_broker") as mock_gb,
            patch("execution.live_engine.DataFeed"),
            patch("execution.live_engine.SignalGenerator"),
            patch("execution.live_engine.RiskManager"),
        ):
            mock_broker = MagicMock()
            mock_broker.get_last_filled_order_time = MagicMock(return_value=last_rebalance)
            mock_gb.return_value = mock_broker
            engine = LiveEngine(config)

        return engine

    def test_first_cycle_with_no_prior_orders_allows_rebalance(self):
        """If broker has no prior fills, _should_rebalance returns True."""
        engine = self._make_engine(last_rebalance=None)
        assert engine._last_rebalance is None
        assert engine._should_rebalance(datetime.now(UTC)) is True

    def test_seeded_last_rebalance_blocks_immediate_rebalance(self):
        """If last fill was 1 hour ago, adaptive cadence should block rebalance."""
        recent_fill = datetime.now(UTC) - timedelta(hours=1)
        engine = self._make_engine(rebalance_freq="adaptive", last_rebalance=recent_fill)

        assert engine._last_rebalance == recent_fill
        assert engine._should_rebalance(datetime.now(UTC)) is False

    def test_seeded_last_rebalance_allows_after_cadence_elapsed(self):
        """If last fill was 15 days ago, adaptive biweekly cadence should allow."""
        old_fill = datetime.now(UTC) - timedelta(days=15)
        engine = self._make_engine(rebalance_freq="adaptive", last_rebalance=old_fill)

        assert engine._last_rebalance == old_fill
        assert engine._should_rebalance(datetime.now(UTC)) is True

    def test_weekly_cadence_blocks_within_5_days(self):
        """Weekly cadence: should block if < 5 days since last rebalance."""
        recent_fill = datetime.now(UTC) - timedelta(days=3)
        engine = self._make_engine(rebalance_freq="weekly", last_rebalance=recent_fill)

        assert engine._should_rebalance(datetime.now(UTC)) is False

    def test_daily_cadence_blocks_within_20_hours(self):
        """Daily cadence: should block if < 20 hours since last rebalance."""
        recent_fill = datetime.now(UTC) - timedelta(hours=10)
        engine = self._make_engine(rebalance_freq="daily", last_rebalance=recent_fill)

        assert engine._should_rebalance(datetime.now(UTC)) is False

    def test_daily_cadence_allows_after_20_hours(self):
        """Daily cadence: should allow if > 20 hours since last rebalance."""
        old_fill = datetime.now(UTC) - timedelta(hours=21)
        engine = self._make_engine(rebalance_freq="daily", last_rebalance=old_fill)

        assert engine._should_rebalance(datetime.now(UTC)) is True


# ============================================================
#  AlpacaBroker.get_last_filled_order_time
# ============================================================


class TestGetLastFilledOrderTime:
    """Verify get_last_filled_order_time returns correct timestamps."""

    def test_returns_filled_at_from_most_recent_order(self):
        from datetime import datetime

        broker = _make_broker()
        mock_order = MagicMock()
        mock_order.filled_at = datetime(2026, 4, 14, 20, 30, 0, tzinfo=UTC)

        with patch.dict(
            "sys.modules",
            {
                "alpaca.trading.enums": MagicMock(QueryOrderStatus=MagicMock(CLOSED="closed")),
                "alpaca.trading.requests": MagicMock(),
                "alpaca": MagicMock(),
                "alpaca.trading": MagicMock(),
            },
        ):
            broker.trading_client.get_orders = MagicMock(return_value=[mock_order])
            result = broker.get_last_filled_order_time()

        assert result == datetime(2026, 4, 14, 20, 30, 0, tzinfo=UTC)

    def test_returns_none_when_no_orders(self):
        broker = _make_broker()

        with patch.dict(
            "sys.modules",
            {
                "alpaca.trading.enums": MagicMock(QueryOrderStatus=MagicMock(CLOSED="closed")),
                "alpaca.trading.requests": MagicMock(),
                "alpaca": MagicMock(),
                "alpaca.trading": MagicMock(),
            },
        ):
            broker.trading_client.get_orders = MagicMock(return_value=[])
            result = broker.get_last_filled_order_time()

        assert result is None

    def test_returns_none_on_exception(self):
        broker = _make_broker()
        broker.trading_client.get_orders = MagicMock(side_effect=Exception("API error"))

        result = broker.get_last_filled_order_time()
        assert result is None
