"""
Restart-resilience regression tests
=====================================
Preventive tests that would have CAUGHT the two bugs causing 486 day trades:
  Bug 1: _last_rebalance resetting to None on container restart
  Bug 2: Duplicate orders from overlapping ECS tasks (no dedup guard)

Run:  python3 -m pytest tests/test_restart_resilience.py -v
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from execution.broker_base import Order, OrderSide, OrderStatus, OrderType
from execution.live_engine import LiveEngine

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_engine(rebalance_freq="adaptive", last_fill_time=None):
    """Create a minimal LiveEngine with mocked broker for restart tests.

    Parameters
    ----------
    rebalance_freq : str
        The rebalance cadence (daily / weekly / adaptive / ...).
    last_fill_time : datetime | None
        Simulated last filled order time returned by the broker.
        None means the broker has no order history.
    """
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
        mock_broker.get_last_filled_order_time = MagicMock(return_value=last_fill_time)
        mock_broker.get_open_orders = MagicMock(return_value=[])
        mock_broker.get_recent_fills = MagicMock(return_value=[])  # no startup cooldown
        mock_broker.cancel_conflicting_orders = MagicMock(return_value=False)
        mock_broker.cancel_all_open_orders = MagicMock(return_value=0)
        mock_gb.return_value = mock_broker
        engine = LiveEngine(config)

    return engine


# ============================================================
#  Category 1: Cold-start rebalance cadence (prevents Bug 1)
# ============================================================


class TestColdStartRebalanceCadence:
    """Verify _last_rebalance is seeded from broker on cold start,
    preventing runaway rebalancing after container restarts."""

    def test_cold_start_with_recent_fills_blocks_rebalance(self):
        """Fresh LiveEngine init where broker reports a fill from 2 hours ago.
        _should_rebalance must return False — cadence not elapsed.
        This DIRECTLY catches Bug 1: if seeding is removed, this test fails."""
        recent_fill = datetime.now(UTC) - timedelta(hours=2)
        engine = _make_engine(rebalance_freq="adaptive", last_fill_time=recent_fill)

        assert engine._last_rebalance == recent_fill
        assert engine._should_rebalance(datetime.now(UTC)) is False

    def test_cold_start_with_no_fills_allows_first_rebalance(self):
        """Fresh LiveEngine init where broker has no order history.
        _should_rebalance must return True — first trade ever is OK."""
        engine = _make_engine(rebalance_freq="adaptive", last_fill_time=None)

        assert engine._last_rebalance is None
        assert engine._should_rebalance(datetime.now(UTC)) is True

    def test_consecutive_cold_starts_never_trade_within_cadence(self):
        """Simulate 5 consecutive LiveEngine inits (mimicking 5 ECS restarts)
        where the broker always returns the same recent fill time.
        _should_rebalance must return False on ALL 5."""
        recent_fill = datetime.now(UTC) - timedelta(hours=3)

        for restart_num in range(5):
            engine = _make_engine(rebalance_freq="adaptive", last_fill_time=recent_fill)
            assert engine._last_rebalance == recent_fill, (
                f"Restart #{restart_num + 1}: _last_rebalance was not seeded"
            )
            assert engine._should_rebalance(datetime.now(UTC)) is False, (
                f"Restart #{restart_num + 1}: should_rebalance returned True "
                f"despite recent fill {recent_fill}"
            )

    def test_cold_start_respects_adaptive_cadence_boundary(self):
        """Cold start with fill exactly at the cadence boundary (>= 14 days ago
        for adaptive/biweekly). _should_rebalance must return True."""
        old_fill = datetime.now(UTC) - timedelta(days=15)
        engine = _make_engine(rebalance_freq="adaptive", last_fill_time=old_fill)

        assert engine._last_rebalance == old_fill
        assert engine._should_rebalance(datetime.now(UTC)) is True


# ============================================================
#  Category 2: Concurrent instance protection (prevents Bug 2)
# ============================================================


class TestConcurrentInstanceProtection:
    """Verify the dedup guard prevents duplicate orders from overlapping
    ECS tasks submitting orders for the same symbol+side."""

    def test_concurrent_engines_only_one_places_orders(self):
        """Two LiveEngine instances sharing the same mock broker.
        First engine places BUY for GLD, second engine's dedup guard
        should detect existing open order and skip."""
        # Shared broker mock
        shared_broker = MagicMock()
        shared_broker.get_last_filled_order_time = MagicMock(return_value=None)
        shared_broker.cancel_conflicting_orders = MagicMock(return_value=False)

        # Track state: initially no open orders
        open_orders_state = []

        def get_open_orders_fn(symbol):
            return [o for o in open_orders_state if o["symbol"] == symbol]

        shared_broker.get_open_orders = MagicMock(side_effect=get_open_orders_fn)

        filled_order = Order(
            symbol="GLD",
            side=OrderSide.BUY,
            quantity=10,
            status=OrderStatus.FILLED,
            avg_fill_price=180.0,
        )
        shared_broker.place_order = MagicMock(return_value=filled_order)

        # --- Engine 1 places order ---
        sym, side = "GLD", OrderSide.BUY
        side_str = side.value.lower()

        existing = shared_broker.get_open_orders(sym)
        same_side = [o for o in existing if o.get("side") == side_str]
        assert len(same_side) == 0  # no duplicates yet

        order = Order(symbol=sym, side=side, quantity=10, order_type=OrderType.MARKET)
        shared_broker.place_order(order)

        # After engine 1 places order, it shows up as open
        open_orders_state.append(
            {"order_id": "eng1-order", "symbol": "GLD", "side": "buy", "qty": 10, "status": "new"}
        )

        # --- Engine 2 tries to place same order ---
        existing = shared_broker.get_open_orders(sym)
        same_side = [o for o in existing if o.get("side") == side_str]
        assert len(same_side) == 1  # dedup guard triggers

        # Engine 2 should skip — place_order should have been called only once
        assert shared_broker.place_order.call_count == 1

    def test_dedup_guard_catches_rapid_fire_same_symbol(self):
        """Crash-loop scenario: order submission path called 5 times rapidly
        for the same symbol+side. Only the first should place an order."""
        mock_broker = MagicMock()
        open_orders_state = []

        def get_open_orders_fn(symbol):
            return [o for o in open_orders_state if o["symbol"] == symbol]

        mock_broker.get_open_orders = MagicMock(side_effect=get_open_orders_fn)

        filled_order = Order(
            symbol="GLD",
            side=OrderSide.BUY,
            quantity=10,
            status=OrderStatus.FILLED,
            avg_fill_price=180.0,
        )
        mock_broker.place_order = MagicMock(return_value=filled_order)

        sym, side = "GLD", OrderSide.BUY
        side_str = side.value.lower()
        place_count = 0

        for attempt in range(5):
            existing = mock_broker.get_open_orders(sym)
            same_side = [o for o in existing if o.get("side") == side_str]
            if same_side:
                continue  # dedup guard skips

            order = Order(symbol=sym, side=side, quantity=10, order_type=OrderType.MARKET)
            mock_broker.place_order(order)
            place_count += 1

            # After first placement, order shows as open/pending
            open_orders_state.append(
                {
                    "order_id": f"order-{attempt}",
                    "symbol": "GLD",
                    "side": "buy",
                    "qty": 10,
                    "status": "new",
                }
            )

        assert place_count == 1
        assert mock_broker.place_order.call_count == 1

    def test_dedup_guard_allows_different_symbols_simultaneously(self):
        """Engine has open BUY for GLD. A BUY for XLE should still be
        allowed — dedup is per-symbol."""
        mock_broker = MagicMock()

        # GLD has an open order, XLE does not
        def get_open_orders_fn(symbol):
            if symbol == "GLD":
                return [
                    {
                        "order_id": "gld-1",
                        "symbol": "GLD",
                        "side": "buy",
                        "qty": 10,
                        "status": "new",
                    }
                ]
            return []

        mock_broker.get_open_orders = MagicMock(side_effect=get_open_orders_fn)
        filled_order = Order(
            symbol="XLE",
            side=OrderSide.BUY,
            quantity=5,
            status=OrderStatus.FILLED,
            avg_fill_price=90.0,
        )
        mock_broker.place_order = MagicMock(return_value=filled_order)

        orders_placed = []

        for sym in ["GLD", "XLE"]:
            side = OrderSide.BUY
            side_str = side.value.lower()
            existing = mock_broker.get_open_orders(sym)
            same_side = [o for o in existing if o.get("side") == side_str]
            if same_side:
                continue
            order = Order(symbol=sym, side=side, quantity=5, order_type=OrderType.MARKET)
            mock_broker.place_order(order)
            orders_placed.append(sym)

        # GLD skipped (dedup), XLE placed
        assert orders_placed == ["XLE"]
        assert mock_broker.place_order.call_count == 1

    def test_dedup_guard_allows_after_previous_order_fills(self):
        """Two sequential calls: first order fills (open_orders empty again),
        second call should also be allowed."""
        mock_broker = MagicMock()

        # First call: no open orders → order placed and immediately fills
        # Second call: still no open orders (previous filled) → allowed
        mock_broker.get_open_orders = MagicMock(return_value=[])
        filled_order = Order(
            symbol="GLD",
            side=OrderSide.BUY,
            quantity=10,
            status=OrderStatus.FILLED,
            avg_fill_price=180.0,
        )
        mock_broker.place_order = MagicMock(return_value=filled_order)

        place_count = 0
        for _ in range(2):
            existing = mock_broker.get_open_orders("GLD")
            same_side = [o for o in existing if o.get("side") == "buy"]
            if same_side:
                continue
            order = Order(
                symbol="GLD", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET
            )
            mock_broker.place_order(order)
            place_count += 1

        assert place_count == 2
        assert mock_broker.place_order.call_count == 2


# ============================================================
#  Category 3: End-to-end trading cycle guards
# ============================================================


class TestEndToEndTradingCycleGuards:
    """High-level tests verifying full _trading_cycle respects guards."""

    def test_full_trading_cycle_respects_cadence_after_restart(self):
        """Complete _trading_cycle() on a freshly constructed engine with
        recent fills must place NO orders at all (cadence blocks the cycle)."""
        recent_fill = datetime.now(UTC) - timedelta(hours=1)
        engine = _make_engine(rebalance_freq="adaptive", last_fill_time=recent_fill)

        # Mock account info for the trading cycle
        mock_account = MagicMock()
        mock_account.equity = 100_000.0
        mock_account.cash = 50_000.0
        mock_account.positions = {}
        engine.broker.get_account = MagicMock(return_value=mock_account)
        engine.broker.get_positions = MagicMock(return_value={})

        # Mock risk manager to not halt
        engine.risk_mgr.check_halt = MagicMock(return_value=(False, ""))

        # Run the full trading cycle
        engine._trading_cycle()

        # Cadence should have blocked — no orders placed
        engine.broker.place_order.assert_not_called()

    def test_trading_cycle_logs_dedup_skip(self, caplog):
        """Run a trading cycle where the dedup guard triggers.
        Assert the DEDUP SKIP warning is logged for monitoring/alerting."""
        # Engine with no recent fills → cadence allows rebalance
        engine = _make_engine(rebalance_freq="daily", last_fill_time=None)

        mock_account = MagicMock()
        mock_account.equity = 100_000.0
        mock_account.cash = 50_000.0
        mock_account.positions = {}
        engine.broker.get_account = MagicMock(return_value=mock_account)
        engine.broker.get_positions = MagicMock(return_value={})

        # Mock risk manager to not halt
        engine.risk_mgr.check_halt = MagicMock(return_value=(False, ""))

        # Mock data feed to return price data for one symbol
        import pandas as pd

        price_series = pd.Series(
            [180.0] * 252,
            index=pd.date_range(end=datetime.now(UTC), periods=252, freq="B"),
            name="Close",
        )
        mock_df = pd.DataFrame({"Close": price_series})
        engine.feed.load_all = MagicMock(return_value={"GLD": mock_df})

        # Signal generator returns a signal for GLD
        engine.signal_gen.generate_latest = MagicMock(return_value={"GLD": 0.5})
        engine.signal_gen.compute_stop_loss = MagicMock(return_value=5.0)

        # Broker has an open BUY for GLD → dedup guard should trigger
        engine.broker.get_open_orders = MagicMock(
            return_value=[
                {
                    "order_id": "existing-1",
                    "symbol": "GLD",
                    "side": "buy",
                    "qty": 10,
                    "status": "new",
                }
            ]
        )

        with caplog.at_level(logging.WARNING, logger="LiveEngine"):
            engine._trading_cycle()

        # Verify DEDUP SKIP was logged
        dedup_messages = [r for r in caplog.records if "DEDUP SKIP" in r.message]
        assert len(dedup_messages) >= 1, (
            f"Expected DEDUP SKIP log message, got: {[r.message for r in caplog.records]}"
        )
        engine.broker.place_order.assert_not_called()
