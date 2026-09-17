"""Live-engine short-selling tests (Gap B).

Covers the intraday hard-stop cover loop and the config wiring around it,
using a mocked broker — no network, no Alpaca credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from execution.broker_base import OrderSide, OrderStatus
from execution.live_engine import LiveEngine
from strategy.short_overlay import ShortingConfig


def _base_config(**overrides):
    cfg = {
        "system": {"mode": "paper"},
        "capital": {"initial_equity": 100000, "max_portfolio_heat": 0.95},
        "risk": {"max_position_pct": 0.15, "max_drawdown_halt": 0.15, "daily_loss_limit": 0.08},
        "strategy": {"rebalance_frequency": "daily"},
        "rebalance_guards": {
            "min_rebalance_interval_seconds": 86400,
            "min_order_delta_pct_of_position": 0.02,
            "min_order_delta_shares": 1.0,
        },
        "risk_limits": {"max_daily_turnover_x": 2.0, "persist_turnover_state": False},
        "signals": {"cache_per_session": True},
        "ews": {"enabled": False},
        "intraday_shock": {"enabled": False},
        "anomaly_layer": {"enabled": False},
        "position_anomaly": {"enabled": False},
        "monitoring": {"enabled": False},
        "shorting": {
            "enabled": True,
            "asset_classes": ["equity_etf"],
            "max_short_notional_pct": 0.30,
            "max_single_short_pct": 0.08,
            "min_signal_strength": 0.0,
            "universe": "liquid_etfs_only",
            "cover_on_regime_improvement": True,
            "hard_stop_pct": 0.08,
        },
    }
    cfg.update(overrides)
    return cfg


def _make_engine(config=None, broker=None, tmp_path=None):
    cfg = config or _base_config()
    broker = broker or _mock_broker()
    with (
        patch("execution.live_engine.get_broker", return_value=broker),
        patch("execution.live_engine.DataFeed"),
        patch("execution.live_engine.SignalGenerator"),
        patch("execution.live_engine.RiskManager"),
    ):
        engine = LiveEngine(cfg)
        if tmp_path is not None:
            engine._state_dir = tmp_path
        return engine


def _mock_broker():
    broker = MagicMock()
    broker.connect.return_value = True
    account = MagicMock()
    account.equity = 100000
    account.cash = 100000
    account.positions = {}
    broker.get_account.return_value = account
    broker.get_positions.return_value = {}
    broker.get_recent_fills.return_value = []
    broker.get_last_filled_order_time.return_value = None
    return broker


def _filled(qty, price):
    order = MagicMock()
    order.status = OrderStatus.FILLED
    order.avg_fill_price = price
    return order


class TestShortConfigWiring:
    def test_shorting_config_parsed(self, tmp_path):
        engine = _make_engine(tmp_path=tmp_path)
        assert engine._short_cfg.enabled is True
        assert engine._short_cfg.max_short_notional_pct == 0.30
        assert engine._short_cfg.hard_stop_pct == 0.08
        assert engine._short_targets_active == set()

    def test_shorting_disabled_by_default(self, tmp_path):
        cfg = _base_config()
        del cfg["shorting"]
        engine = _make_engine(config=cfg, tmp_path=tmp_path)
        assert engine._short_cfg.enabled is False


class TestHardStopLoop:
    def test_no_shorts_no_action(self, tmp_path):
        engine = _make_engine(tmp_path=tmp_path)
        account = MagicMock()
        account.positions = {"SPY": {"quantity": 10.0, "avg_price": 100.0}}
        engine._check_short_hard_stops(account)
        engine.broker.place_order.assert_not_called()

    def test_short_past_stop_gets_covered(self, tmp_path):
        broker = _mock_broker()
        broker.get_latest_prices.return_value = {"QQQ": 108.5}
        broker.place_order.return_value = _filled(50, 108.5)
        engine = _make_engine(broker=broker, tmp_path=tmp_path)
        engine._daily_gross_traded_usd = 0.0

        account = MagicMock()
        account.positions = {"QQQ": {"quantity": -50.0, "avg_price": 100.0}}
        engine._check_short_hard_stops(account)

        broker.place_order.assert_called_once()
        order = broker.place_order.call_args[0][0]
        assert order.symbol == "QQQ"
        assert order.side == OrderSide.BUY
        assert order.quantity == 50.0
        # Turnover tracked
        assert engine._daily_gross_traded_usd > 0

    def test_short_below_stop_not_covered(self, tmp_path):
        broker = _mock_broker()
        broker.get_latest_prices.return_value = {"QQQ": 107.0}
        engine = _make_engine(broker=broker, tmp_path=tmp_path)
        account = MagicMock()
        account.positions = {"QQQ": {"quantity": -50.0, "avg_price": 100.0}}
        engine._check_short_hard_stops(account)
        broker.place_order.assert_not_called()

    def test_price_fetch_failure_skips_safely(self, tmp_path):
        broker = _mock_broker()
        broker.get_latest_prices.side_effect = RuntimeError("api down")
        engine = _make_engine(broker=broker, tmp_path=tmp_path)
        account = MagicMock()
        account.positions = {"QQQ": {"quantity": -50.0, "avg_price": 100.0}}
        engine._check_short_hard_stops(account)  # must not raise
        broker.place_order.assert_not_called()

    def test_hard_stop_bypasses_turnover_cap(self, tmp_path):
        """Risk exits must execute even when the daily turnover cap is hit."""
        broker = _mock_broker()
        broker.get_latest_prices.return_value = {"QQQ": 110.0}
        broker.place_order.return_value = _filled(50, 110.0)
        engine = _make_engine(broker=broker, tmp_path=tmp_path)
        engine._daily_gross_traded_usd = 999_999_999  # way past the cap
        account = MagicMock()
        account.positions = {"QQQ": {"quantity": -50.0, "avg_price": 100.0}}
        engine._check_short_hard_stops(account)
        broker.place_order.assert_called_once()


class TestOverlayIntegrationPoints:
    def test_guard_blocks_crypto_shorts_via_is_shortable(self, tmp_path):
        engine = _make_engine(tmp_path=tmp_path)
        guard = engine._get_tradeable_guard()
        assert not guard.is_shortable("BTC-USD")
        assert not guard.is_shortable("ES=F")
        assert not guard.is_shortable("BNB-USD")
        assert guard.is_shortable("SPY")  # offline fallback: liquid ETF assumed

    def test_short_cfg_roundtrip_from_yaml(self, tmp_path):
        import yaml

        cfg_path = ROOT / "config" / "settings.yaml"
        with open(cfg_path) as f:
            full = yaml.safe_load(f)
        scfg = ShortingConfig.from_config(full)
        assert scfg.enabled is True
        assert scfg.asset_classes == ["equity_etf"]
        assert scfg.universe_symbols == ["SPY", "QQQ", "IWM", "DIA", "MDY", "EEM", "VGK", "EWJ",
                                         "XLE", "XLF", "XLV", "XLU", "XLP", "XLY", "XLK", "VNQ"]
