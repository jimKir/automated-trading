"""
Tests for EWS Series bug, Optimizer long-only clipping, and Re-entry guard
==========================================================================
Fix 1: EventShockDetector._fetch must handle MultiIndex columns from yfinance
Fix 2: PortfolioOptimizer must clip negative weights to 0 in long_only mode
Fix 3: LiveEngine re-entry guard gates first rebalance after circuit breaker

Run:  python3 -m pytest tests/test_ews_optimizer_reentry.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.optimizer import PortfolioOptimizer
from execution.live_engine import LiveEngine
from regime.event_shock import EventShockDetector

# ============================================================
#  Fix 1: EWS float(Series) bug — EventShockDetector._fetch
# ============================================================


class TestEventShockMultiIndex:
    """Verify EventShockDetector handles yfinance MultiIndex columns."""

    def test_fetch_flattens_multiindex(self):
        """_fetch must return a Series even when yfinance returns MultiIndex columns."""
        detector = EventShockDetector()

        # Simulate yfinance returning a DataFrame with MultiIndex columns
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        multi_cols = pd.MultiIndex.from_tuples(
            [
                ("Close", "^VIX"),
                ("Open", "^VIX"),
                ("High", "^VIX"),
                ("Low", "^VIX"),
                ("Volume", "^VIX"),
            ],
            names=["Price", "Ticker"],
        )
        data = np.random.default_rng(42).uniform(15, 25, (10, 5))
        mock_df = pd.DataFrame(data, index=dates, columns=multi_cols)

        with patch("yfinance.download", return_value=mock_df):
            result = detector._fetch("^VIX", "2026-01-01", "2026-02-01")

        assert isinstance(result, pd.Series), f"Expected Series, got {type(result).__name__}"
        assert len(result) == 10

    def test_fetch_normal_columns(self):
        """_fetch works normally with flat (non-MultiIndex) columns."""
        detector = EventShockDetector()

        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        mock_df = pd.DataFrame(
            {
                "Close": [20.0, 21.0, 22.0, 23.0, 24.0],
                "Open": [19.5] * 5,
                "High": [24.5] * 5,
                "Low": [19.0] * 5,
                "Volume": [1e6] * 5,
            },
            index=dates,
        )

        with patch("yfinance.download", return_value=mock_df):
            result = detector._fetch("^VIX", "2026-01-01", "2026-02-01")

        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_compute_series_with_multiindex_does_not_raise(self):
        """compute_series must not raise when yfinance returns MultiIndex data."""
        detector = EventShockDetector()

        dates = pd.date_range("2026-01-01", periods=90, freq="B")
        multi_cols = pd.MultiIndex.from_tuples(
            [("Close", "SYM"), ("Open", "SYM"), ("High", "SYM"), ("Low", "SYM"), ("Volume", "SYM")],
            names=["Price", "Ticker"],
        )
        rng = np.random.default_rng(42)

        def mock_download(symbol, **kwargs):
            data = rng.uniform(10, 100, (90, 5))
            return pd.DataFrame(data, index=dates, columns=multi_cols)

        all_prices = pd.DataFrame(
            {
                "SPY": rng.uniform(400, 500, 90),
                "TLT": rng.uniform(80, 100, 90),
                "GLD": rng.uniform(170, 200, 90),
            },
            index=dates,
        )

        with patch("yfinance.download", side_effect=mock_download):
            series = detector.compute_series("2026-01-01", "2026-04-22", all_prices)

        assert isinstance(series, pd.Series)
        assert not series.empty
        assert all(0.0 <= v <= 1.0 for v in series.dropna())

    def test_vix_velocity_score_with_series_input(self):
        """Verify _vix_velocity_score works with a proper Series (regression)."""
        detector = EventShockDetector()
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        vix = pd.Series([18, 19, 20, 21, 22, 30, 35, 40, 42, 44], index=dates, dtype=float)
        date = dates[-1]
        score = detector._vix_velocity_score(vix, date)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ============================================================
#  Fix 2: Optimizer long-only mode — negative weight clipping
# ============================================================


class TestOptimizerLongOnly:
    """Verify optimizer clips negative weights in long_only mode."""

    def _make_optimizer(self, long_only=True, **kwargs):
        config = {
            "optimizer": {
                "enabled": True,
                "method": "risk_parity",
                "long_only": long_only,
                "regime_scaling": False,
                **kwargs,
            }
        }
        return PortfolioOptimizer(config)

    def _make_price_history(self, symbols, n=100):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        return {
            sym: pd.DataFrame(
                {"Close": 100 + rng.standard_normal(n).cumsum()},
                index=dates,
            )
            for sym in symbols
        }

    def test_long_only_clips_negative_signals(self):
        """Negative signals must produce 0 weight in long_only mode."""
        opt = self._make_optimizer(long_only=True)
        signals = {"SPY": -0.15, "QQQ": -0.20, "GLD": 0.10, "XLE": 0.08}
        prices = self._make_price_history(signals.keys())

        weights = opt.compute_weights(
            signals, prices, max_position_pct=0.15, max_portfolio_heat=0.40
        )

        assert weights.get("SPY", 0) == 0.0, "SPY has negative signal — weight must be 0"
        assert weights.get("QQQ", 0) == 0.0, "QQQ has negative signal — weight must be 0"
        assert weights.get("GLD", 0) > 0.0, "GLD has positive signal — weight must be positive"
        assert weights.get("XLE", 0) > 0.0, "XLE has positive signal — weight must be positive"

    def test_long_only_all_negative_returns_zero(self):
        """When all signals are negative and long_only=True, all weights must be 0."""
        opt = self._make_optimizer(long_only=True)
        signals = {"SPY": -0.10, "QQQ": -0.20, "TLT": -0.15}
        prices = self._make_price_history(signals.keys())

        weights = opt.compute_weights(
            signals, prices, max_position_pct=0.15, max_portfolio_heat=0.40
        )

        for sym, w in weights.items():
            assert w == 0.0, f"{sym} should be 0 when all signals negative"

    def test_long_only_off_allows_negative_weights(self):
        """When long_only=False, negative signals produce negative weights."""
        opt = self._make_optimizer(long_only=False)
        signals = {"SPY": -0.15, "GLD": 0.10}
        prices = self._make_price_history(signals.keys())

        weights = opt.compute_weights(
            signals, prices, max_position_pct=0.15, max_portfolio_heat=0.40
        )

        assert weights.get("SPY", 0) < 0, "SPY should have negative weight when long_only=False"
        assert weights.get("GLD", 0) > 0, "GLD should have positive weight"

    def test_long_only_default_is_true(self):
        """Default long_only setting must be True."""
        opt = PortfolioOptimizer({"optimizer": {"enabled": True}})
        assert opt.long_only is True

    def test_from_cash_positive_signals_produce_buys(self):
        """From a zero-position state with positive signals, the optimizer
        must generate positive target weights (which become BUY orders)."""
        opt = self._make_optimizer(long_only=True)
        signals = {
            "GLD": 0.07,
            "SPY": -0.05,
            "QQQ": -0.17,
            "XLE": 0.063,
            "BTC-USD": -0.06,
            "ETH-USD": -0.08,
        }
        prices = self._make_price_history(signals.keys())

        weights = opt.compute_weights(
            signals, prices, max_position_pct=0.15, max_portfolio_heat=0.40
        )

        # GLD and XLE have positive signals — they should get positive weights
        assert weights.get("GLD", 0) > 0, "GLD (signal +0.07) must produce BUY"
        assert weights.get("XLE", 0) > 0, "XLE (signal +0.063) must produce BUY"
        # Negative signals should be 0
        assert weights.get("QQQ", 0) == 0.0
        assert weights.get("SPY", 0) == 0.0

    def test_weak_signals_filtered_by_threshold(self):
        """Signals below abs(0.05) threshold should not generate trades."""
        opt = self._make_optimizer(long_only=True)
        signals = {"SPY": 0.03, "GLD": 0.04}
        prices = self._make_price_history(signals.keys())

        weights = opt.compute_weights(
            signals, prices, max_position_pct=0.15, max_portfolio_heat=0.40
        )

        # All signals below 0.05 threshold → zero weights
        for w in weights.values():
            assert w == 0.0


# ============================================================
#  Fix 3: Post-circuit-breaker re-entry guard
# ============================================================


def _make_engine_for_reentry(
    positions=None,
    equity=100000.0,
    cash=100000.0,
    reentry_cfg=None,
):
    """Create a minimal LiveEngine with mocked broker for re-entry tests."""
    if positions is None:
        positions = {}
    config = {
        "system": {"mode": "paper"},
        "strategy": {"rebalance_frequency": "daily"},
        "brokers": {"alpaca": {"api_key": "", "api_secret": ""}},
    }
    if reentry_cfg:
        config["reentry_guard"] = reentry_cfg

    mock_account = MagicMock()
    mock_account.equity = equity
    mock_account.cash = cash
    mock_account.positions = positions

    with (
        patch("execution.live_engine.get_broker") as mock_gb,
        patch("execution.live_engine.DataFeed"),
        patch("execution.live_engine.SignalGenerator"),
        patch("execution.live_engine.RiskManager"),
    ):
        mock_broker = MagicMock()
        mock_broker.get_last_filled_order_time = MagicMock(return_value=None)
        mock_broker.get_open_orders = MagicMock(return_value=[])
        mock_broker.get_recent_fills = MagicMock(return_value=[])
        mock_broker.cancel_all_open_orders = MagicMock(return_value=0)
        mock_broker.get_account = MagicMock(return_value=mock_account)
        mock_broker.get_positions = MagicMock(return_value=positions)
        mock_gb.return_value = mock_broker
        engine = LiveEngine(config)

    return engine


class TestReentryDetection:
    """Verify cash-only re-entry state detection."""

    def test_cash_only_detected(self):
        """Engine detects cash-only state (0 positions, cash ≈ equity)."""
        engine = _make_engine_for_reentry(positions={}, equity=100000.0, cash=100000.0)
        assert engine._reentry_gate_active is True

    def test_with_positions_not_detected(self):
        """Engine with positions does NOT trigger re-entry gate."""
        positions = {"SPY": {"quantity": 100, "avg_price": 500.0}}
        engine = _make_engine_for_reentry(positions=positions, equity=100000.0, cash=50000.0)
        assert engine._reentry_gate_active is False

    def test_cash_not_equal_equity_not_detected(self):
        """When cash << equity (positions exist), re-entry gate is not set."""
        engine = _make_engine_for_reentry(positions={}, equity=100000.0, cash=50000.0)
        # cash/equity = 0.5 < 0.99, so NOT detected
        assert engine._reentry_gate_active is False


class TestReentryGates:
    """Verify re-entry gate conditions block/allow rebalance."""

    def test_reentry_blocks_on_red_regime(self):
        """Re-entry must block when EWS regime is RED."""
        engine = _make_engine_for_reentry()
        assert engine._reentry_gate_active is True

        signals = {"GLD": 0.10, "SPY": 0.08}
        allowed, _heat, reason = engine._check_reentry_gates(
            ews_colour="RED", signals=signals, max_heat=0.40
        )
        assert allowed is False
        assert "RED" in reason

    def test_reentry_blocks_on_orange_regime(self):
        """Re-entry must block when EWS regime is ORANGE."""
        engine = _make_engine_for_reentry()
        signals = {"GLD": 0.10}
        allowed, _, _reason = engine._check_reentry_gates(
            ews_colour="ORANGE", signals=signals, max_heat=0.40
        )
        assert allowed is False

    def test_reentry_blocks_no_positive_signals(self):
        """Re-entry must block when no tradeable symbol has signal >= 0.05."""
        engine = _make_engine_for_reentry()
        signals = {"SPY": -0.10, "QQQ": -0.20, "ES=F": 0.15}  # ES=F is non-tradeable
        allowed, _, reason = engine._check_reentry_gates(
            ews_colour="GREEN", signals=signals, max_heat=0.40
        )
        assert allowed is False
        assert "no tradeable symbol" in reason

    def test_reentry_caps_deployment_at_50pct(self):
        """First re-entry rebalance must cap heat at 50% (default)."""
        engine = _make_engine_for_reentry()
        signals = {"GLD": 0.10, "SPY": 0.08}
        allowed, heat, _ = engine._check_reentry_gates(
            ews_colour="GREEN", signals=signals, max_heat=0.75
        )
        assert allowed is True
        assert heat == pytest.approx(0.50), f"Heat should be capped at 50%, got {heat}"

    def test_reentry_respects_custom_deploy_cap(self):
        """Custom reentry_max_deploy_pct is respected."""
        engine = _make_engine_for_reentry(reentry_cfg={"max_first_deploy_pct": 0.30})
        signals = {"GLD": 0.10}
        _allowed, heat, _ = engine._check_reentry_gates(
            ews_colour="GREEN", signals=signals, max_heat=0.75
        )
        assert heat == pytest.approx(0.30)

    def test_reentry_flag_clears_after_pass(self):
        """Re-entry gate flag is checked but NOT cleared by _check_reentry_gates.
        It is cleared in _trading_cycle after orders are placed."""
        engine = _make_engine_for_reentry()
        signals = {"GLD": 0.10}
        engine._check_reentry_gates("GREEN", signals, 0.40)
        # Gate is still active — clearing happens in _trading_cycle
        assert engine._reentry_gate_active is True

    def test_not_active_allows_full_heat(self):
        """When re-entry gate is not active, full heat is returned."""
        engine = _make_engine_for_reentry(
            positions={"SPY": {"quantity": 100, "avg_price": 500}},
            equity=100000.0,
            cash=50000.0,
        )
        assert engine._reentry_gate_active is False
        allowed, heat, _ = engine._check_reentry_gates(
            ews_colour="GREEN", signals={"GLD": 0.10}, max_heat=0.75
        )
        assert allowed is True
        assert heat == pytest.approx(0.75)

    def test_reentry_allows_yellow_regime(self):
        """Re-entry is allowed when EWS is YELLOW (not just GREEN)."""
        engine = _make_engine_for_reentry()
        signals = {"GLD": 0.10}
        allowed, _, _ = engine._check_reentry_gates(
            ews_colour="YELLOW", signals=signals, max_heat=0.40
        )
        assert allowed is True

    def test_reentry_gate_manual_clear(self):
        """Simulating the flag clear that happens in _trading_cycle."""
        engine = _make_engine_for_reentry()
        assert engine._reentry_gate_active is True
        engine._reentry_gate_active = False
        # After clearing, subsequent rebalance uses full heat
        _allowed, heat, _ = engine._check_reentry_gates(
            ews_colour="GREEN", signals={"GLD": 0.10}, max_heat=0.75
        )
        assert heat == pytest.approx(0.75)

    def test_reentry_custom_signal_threshold(self):
        """Custom min_signal_threshold is respected."""
        engine = _make_engine_for_reentry(reentry_cfg={"min_signal_threshold": 0.10})
        # Signal 0.08 is below 0.10 threshold
        signals = {"GLD": 0.08, "SPY": -0.05}
        allowed, _, _reason = engine._check_reentry_gates(
            ews_colour="GREEN", signals=signals, max_heat=0.40
        )
        assert allowed is False

        # Signal 0.12 is above threshold
        signals2 = {"GLD": 0.12, "SPY": -0.05}
        allowed2, _, _ = engine._check_reentry_gates(
            ews_colour="GREEN", signals=signals2, max_heat=0.40
        )
        assert allowed2 is True
