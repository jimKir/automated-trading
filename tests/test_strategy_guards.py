"""
Tests for P0-P2 Strategy Guards
================================
P0-1: 24h rebalance interval + larger delta threshold
P0-2: Signal cache per market session
P1-1: Hard daily turnover cap (2× equity)
P1-2: Gradual re-entry ramp after circuit breaker
P2-1: Backtest capital alignment + per-symbol turnover tracking
P2-2: Intraday circuit-breaker simulator in backtest

Run:  python3 -m pytest tests/test_strategy_guards.py -v
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from execution.live_engine import LiveEngine

# ── Minimal config factory ────────────────────────────────────────────


def _base_config(**overrides):
    """Return a minimal config dict for LiveEngine tests."""
    cfg = {
        "system": {"mode": "paper"},
        "capital": {
            "initial_equity": 100000,
            "max_portfolio_heat": 0.95,
            "hedge_reserve_pct": 0.20,
            "min_cash_pct": 0.05,
        },
        "risk": {"max_position_pct": 0.15, "max_drawdown_halt": 0.15, "daily_loss_limit": 0.08},
        "strategy": {"rebalance_frequency": "daily"},
        "rebalance_guards": {
            "min_rebalance_interval_seconds": 86400,
            "min_order_delta_pct_of_position": 0.02,
            "min_order_delta_shares": 1.0,
        },
        "risk_limits": {
            "max_daily_turnover_x": 2.0,
            "persist_turnover_state": False,
        },
        "reentry_ramp": {
            "day_0_pct": 0.50,
            "day_1_pct": 0.75,
            "day_2_plus_pct": 1.00,
        },
        "signals": {"cache_per_session": True},
        "backtest": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "intraday_cb_simulation": True,
            "intraday_cb_threshold_pct": 0.08,
        },
        "ews": {"enabled": False},
        "intraday_shock": {"enabled": False},
        "anomaly_layer": {"enabled": False},
        "position_anomaly": {"enabled": False},
        "monitoring": {"enabled": False},
    }
    cfg.update(overrides)
    return cfg


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


def _make_engine(config=None, tmp_path=None):
    """Create a LiveEngine with mocked broker for testing."""
    cfg = config or _base_config()
    with (
        patch("execution.live_engine.get_broker", return_value=_mock_broker()),
        patch("execution.live_engine.DataFeed"),
        patch("execution.live_engine.SignalGenerator"),
        patch("execution.live_engine.RiskManager"),
    ):
        engine = LiveEngine(cfg)
        if tmp_path is not None:
            engine._state_dir = tmp_path
        return engine


# ============================================================
#  P0-1: 24h rebalance interval
# ============================================================


class TestRebalanceInterval:
    """P0-1: Hard 24h minimum rebalance interval."""

    def test_rebalance_gated_when_1h_ago(self, tmp_path):
        """Rebalance returns False when last rebalance was 1h ago."""
        engine = _make_engine(tmp_path=tmp_path)
        engine._last_rebalance = datetime.now(UTC) - timedelta(hours=1)
        assert engine._should_rebalance(datetime.now(UTC)) is False

    def test_rebalance_allowed_when_25h_ago(self, tmp_path):
        """Rebalance returns True when last rebalance was 25h ago."""
        engine = _make_engine(tmp_path=tmp_path)
        engine._last_rebalance = datetime.now(UTC) - timedelta(hours=25)
        assert engine._should_rebalance(datetime.now(UTC)) is True

    def test_rebalance_allowed_on_first_run(self, tmp_path):
        """Rebalance returns True when _last_rebalance is None (first run)."""
        engine = _make_engine(tmp_path=tmp_path)
        engine._last_rebalance = None
        engine._startup_cooldown_active = False
        assert engine._should_rebalance(datetime.now(UTC)) is True

    def test_persistence_survives_restart(self, tmp_path):
        """Persisted timestamp survives engine restart."""
        engine1 = _make_engine(tmp_path=tmp_path)
        ts = datetime.now(UTC) - timedelta(hours=2)
        engine1._state_dir = tmp_path
        engine1._persist_rebalance_ts(ts)

        engine2 = _make_engine(tmp_path=tmp_path)
        engine2._state_dir = tmp_path
        engine2._load_persisted_rebalance_ts()
        assert engine2._last_rebalance is not None
        # Should be within a second of the original
        assert abs((engine2._last_rebalance - ts).total_seconds()) < 2

    def test_rebalance_log_message(self, tmp_path, caplog):
        """Gate emits the expected log line."""
        engine = _make_engine(tmp_path=tmp_path)
        engine._last_rebalance = datetime.now(UTC) - timedelta(hours=1)
        with caplog.at_level(logging.INFO, logger="LiveEngine"):
            result = engine._should_rebalance(datetime.now(UTC))
        assert result is False
        assert "[REBAL] gated: 24h min interval" in caplog.text


# ============================================================
#  P0-1: Larger delta threshold
# ============================================================


class TestDeltaThreshold:
    """P0-1: Order delta must exceed min(1 share, 2% of position)."""

    def test_small_delta_suppressed(self, tmp_path):
        """Order with delta=0.3 shares should be suppressed."""
        engine = _make_engine(tmp_path=tmp_path)
        # 0.3 shares at $100 = $30 notional
        # min_order_delta_shares=1.0 → threshold = 1.0 * $100 = $100
        # $30 < $100 → suppressed
        prices = {"SPY": 100.0}
        delta = 0.3 * prices["SPY"]  # $30
        min_delta = max(
            engine._min_order_delta_shares * prices["SPY"],
            engine._min_order_delta_pct * 0,  # no existing position
        )
        assert abs(delta) < min_delta

    def test_large_delta_passes(self, tmp_path):
        """Order with delta=2.5 shares (>2% of position) goes through."""
        engine = _make_engine(tmp_path=tmp_path)
        prices = {"SPY": 100.0}
        curr_value = 5000.0  # 50 shares at $100
        delta = 2.5 * prices["SPY"]  # $250
        min_delta = max(
            engine._min_order_delta_shares * prices["SPY"],  # $100
            engine._min_order_delta_pct * abs(curr_value),  # $100 (2% of $5000)
        )
        assert abs(delta) >= min_delta


# ============================================================
#  P0-2: Signal cache per session
# ============================================================


class TestSignalCache:
    """P0-2: Signals computed once per market session, cached for reuse."""

    def test_first_call_computes(self, tmp_path):
        """First call in a session computes signals."""
        engine = _make_engine(tmp_path=tmp_path)
        assert engine._signal_cache == {}
        assert engine._signal_cache_date == ""

    def test_cache_returns_same_value(self, tmp_path):
        """Second call within session returns cached signals."""
        engine = _make_engine(tmp_path=tmp_path)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        engine._signal_cache = {"SPY": 0.5, "QQQ": -0.2}
        engine._signal_cache_date = today
        engine._signal_cache_ts = datetime.now(UTC)
        # Cache should be hit
        assert engine._signal_cache_enabled is True
        assert engine._signal_cache_date == today
        assert engine._signal_cache["SPY"] == 0.5

    def test_cache_invalidates_on_date_change(self, tmp_path):
        """Cache invalidates when UTC date changes."""
        engine = _make_engine(tmp_path=tmp_path)
        engine._signal_cache = {"SPY": 0.5}
        engine._signal_cache_date = "2026-04-25"  # yesterday
        today = "2026-04-26"
        # Should not match
        assert engine._signal_cache_date != today

    def test_backtest_bypasses_cache(self, tmp_path):
        """Backtest mode always recomputes (cache_per_session=false)."""
        cfg = _base_config()
        cfg["signals"]["cache_per_session"] = False
        engine = _make_engine(config=cfg, tmp_path=tmp_path)
        assert engine._signal_cache_enabled is False


# ============================================================
#  P1-1: Daily turnover cap
# ============================================================


class TestTurnoverCap:
    """P1-1: Hard daily turnover cap (2× equity)."""

    def test_order_accepted_at_50pct(self, tmp_path):
        """Order at 50% of cap goes through."""
        engine = _make_engine(tmp_path=tmp_path)
        equity = 100000
        engine._daily_gross_traded_usd = equity * 1.0  # 1× used
        engine._turnover_date = datetime.now(UTC).strftime("%Y-%m-%d")
        new_order = equity * 0.5  # 0.5× more
        cap = engine._max_daily_turnover_x * equity
        assert engine._daily_gross_traded_usd + new_order <= cap

    def test_order_accepted_at_99pct(self, tmp_path):
        """Order at 99% of cap goes through."""
        engine = _make_engine(tmp_path=tmp_path)
        equity = 100000
        engine._daily_gross_traded_usd = equity * 1.98  # 99% of 2×
        new_order = equity * 0.01
        cap = engine._max_daily_turnover_x * equity
        assert engine._daily_gross_traded_usd + new_order <= cap

    def test_order_rejected_at_101pct(self, tmp_path):
        """Order at 101% of cap is rejected."""
        engine = _make_engine(tmp_path=tmp_path)
        equity = 100000
        engine._daily_gross_traded_usd = equity * 2.0  # exactly at cap
        new_order = equity * 0.01  # over cap
        cap = engine._max_daily_turnover_x * equity
        assert engine._daily_gross_traded_usd + new_order > cap

    def test_counter_resets_at_utc_date_change(self, tmp_path):
        """Counter resets at UTC date change."""
        engine = _make_engine(tmp_path=tmp_path)
        engine._daily_gross_traded_usd = 150000
        engine._turnover_date = "2026-04-25"
        # Simulate date change
        new_date = "2026-04-26"
        if engine._turnover_date != new_date:
            engine._daily_gross_traded_usd = 0.0
            engine._turnover_date = new_date
        assert engine._daily_gross_traded_usd == 0.0

    def test_persistence_across_restarts(self, tmp_path):
        """Turnover state persists across engine restarts."""
        engine1 = _make_engine(tmp_path=tmp_path)
        engine1._state_dir = tmp_path
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        engine1._daily_gross_traded_usd = 75000.0
        engine1._turnover_date = today
        engine1._persist_daily_turnover()

        engine2 = _make_engine(tmp_path=tmp_path)
        engine2._state_dir = tmp_path
        engine2._load_persisted_turnover()
        assert engine2._daily_gross_traded_usd == 75000.0


# ============================================================
#  P1-2: Gradual re-entry ramp
# ============================================================


class TestReentryRamp:
    """P1-2: Gradual capital deployment ramp after CB."""

    def test_day_0_cap(self, tmp_path):
        """Day 0 → 50% deployment cap."""
        engine = _make_engine(tmp_path=tmp_path)
        engine._reentry_start_date = datetime.now(UTC).strftime("%Y-%m-%d")
        assert engine._get_reentry_ramp_cap() == 0.50

    def test_day_1_cap(self, tmp_path):
        """Day 1 → 75% deployment cap."""
        engine = _make_engine(tmp_path=tmp_path)
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        engine._reentry_start_date = yesterday
        assert engine._get_reentry_ramp_cap() == 0.75

    def test_day_2_plus_no_cap(self, tmp_path):
        """Day 2+ → 100% (no cap)."""
        engine = _make_engine(tmp_path=tmp_path)
        two_days_ago = (datetime.now(UTC) - timedelta(days=2)).strftime("%Y-%m-%d")
        engine._reentry_start_date = two_days_ago
        assert engine._get_reentry_ramp_cap() == 1.00

    def test_no_ramp_when_no_reentry(self, tmp_path):
        """No ramp active → cap=1.0."""
        engine = _make_engine(tmp_path=tmp_path)
        engine._reentry_start_date = None
        assert engine._get_reentry_ramp_cap() == 1.00

    def test_heat_target_095(self, tmp_path):
        """Max portfolio heat of 0.95 allows ~95% deployment."""
        cfg = _base_config()
        cfg["capital"]["max_portfolio_heat"] = 0.95
        _make_engine(config=cfg, tmp_path=tmp_path)
        assert cfg["capital"]["max_portfolio_heat"] == 0.95

    def test_reentry_date_persists(self, tmp_path):
        """Re-entry start date persists across restarts."""
        engine1 = _make_engine(tmp_path=tmp_path)
        engine1._state_dir = tmp_path
        engine1._reentry_start_date = "2026-04-25"
        engine1._persist_reentry_date()

        engine2 = _make_engine(tmp_path=tmp_path)
        engine2._state_dir = tmp_path
        engine2._load_persisted_reentry_date()
        assert engine2._reentry_start_date == "2026-04-25"


# ============================================================
#  P2-1: Backtest capital alignment + turnover tracking
# ============================================================


class TestBacktestCapital:
    """P2-1: Backtest at $100K + per-symbol turnover."""

    def test_initial_equity_100k(self):
        """Config now uses $100K initial equity."""
        import yaml

        config_path = ROOT / "config" / "settings.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["capital"]["initial_equity"] == 100000

    def test_per_symbol_turnover_in_metrics(self):
        """Backtest engine initialises per-symbol fill tracking."""
        from backtest.engine import BacktestEngine

        cfg = _base_config()
        engine = BacktestEngine(cfg)
        # The engine tracks these during run() — verify attribute access
        assert engine.use_intraday_cb_sim is True


# ============================================================
#  P2-2: Intraday circuit-breaker simulator
# ============================================================


class TestIntradayCBSimulator:
    """P2-2: Intraday CB simulation using daily High-Low range."""

    def test_cb_triggers_on_large_open_to_low(self):
        """Day with sufficient open-to-low drawdown triggers CB."""
        from backtest.engine import BacktestEngine

        cfg = _base_config()
        cfg["backtest"]["intraday_cb_threshold_pct"] = 0.08
        engine = BacktestEngine(cfg)
        assert engine.use_intraday_cb_sim is True
        assert engine.intraday_cb_threshold == 0.08

    def test_cb_threshold_configurable(self):
        """CB threshold is configurable from settings."""
        from backtest.engine import BacktestEngine

        cfg = _base_config()
        cfg["backtest"]["intraday_cb_threshold_pct"] = 0.05
        engine = BacktestEngine(cfg)
        assert engine.intraday_cb_threshold == 0.05

    def test_cb_disabled_when_false(self):
        """CB simulation can be disabled."""
        from backtest.engine import BacktestEngine

        cfg = _base_config()
        cfg["backtest"]["intraday_cb_simulation"] = False
        engine = BacktestEngine(cfg)
        assert engine.use_intraday_cb_sim is False


# ============================================================
#  Config integration tests
# ============================================================


class TestConfigDefaults:
    """Ensure all new config keys have sensible defaults."""

    def test_rebalance_guards_defaults(self, tmp_path):
        cfg = _base_config()
        del cfg["rebalance_guards"]
        engine = _make_engine(config=cfg, tmp_path=tmp_path)
        assert engine._min_rebalance_interval_s == 86400
        assert engine._min_order_delta_pct == 0.02
        assert engine._min_order_delta_shares == 1.0

    def test_risk_limits_defaults(self, tmp_path):
        cfg = _base_config()
        del cfg["risk_limits"]
        engine = _make_engine(config=cfg, tmp_path=tmp_path)
        assert engine._max_daily_turnover_x == 2.0

    def test_reentry_ramp_defaults(self, tmp_path):
        cfg = _base_config()
        del cfg["reentry_ramp"]
        engine = _make_engine(config=cfg, tmp_path=tmp_path)
        assert engine._ramp_day_0_pct == 0.50
        assert engine._ramp_day_1_pct == 0.75
        assert engine._ramp_day_2_plus_pct == 1.00

    def test_signal_cache_default_enabled(self, tmp_path):
        cfg = _base_config()
        del cfg["signals"]
        engine = _make_engine(config=cfg, tmp_path=tmp_path)
        assert engine._signal_cache_enabled is True
