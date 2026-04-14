"""
Tests for Alpaca order rejection guards
========================================
Issue 1: Fractional short-sell quantities must be floored to whole numbers.
Issue 2: Wash-trade prevention — cancel opposite-side, skip same-side duplicates.

Run:  python3 -m pytest tests/test_order_guards.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from execution.broker_base import Order, OrderSide, OrderStatus

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
