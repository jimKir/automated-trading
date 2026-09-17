"""Unit tests for the bear-regime short overlay (Gap B).

Covers:
  - ShortingConfig parsing (defaults + config block)
  - bear regime detection (SPY vs 200d MA, fail-safe on insufficient data)
  - eligibility rules (bear regime OR negative signal + below own 200d MA)
  - sizing caps: per-name max_single_short_pct, aggregate
    max_short_notional_pct, portfolio-heat headroom
  - crypto / futures can never be shorted
  - cover-on-regime-improvement semantics (overlay returns empty -> targets 0)
  - hard-stop cover detection (incl. avg_price / avg_entry_price key variants)
  - portfolio accounting: short P&L sign, cash effects, gross exposure heat
"""

import numpy as np
import pandas as pd
import pytest

from strategy.short_overlay import (
    LIQUID_ETF_UNIVERSE,
    ShortingConfig,
    below_ma,
    compute_short_targets,
    hard_stop_covers,
    is_bear_regime,
    merge_short_targets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hist(start, end, n=300):
    """Downward or arbitrary close series with n business days."""
    return pd.DataFrame(
        {"Close": np.linspace(start, end, n)},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _cfg(**kw):
    base = dict(
        enabled=True,
        max_short_notional_pct=0.30,
        max_single_short_pct=0.08,
        min_signal_strength=0.0,
        hard_stop_pct=0.08,
    )
    base.update(kw)
    return ShortingConfig(**base)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestShortingConfig:
    def test_defaults_disabled(self):
        cfg = ShortingConfig.from_config({})
        assert cfg.enabled is False
        assert cfg.asset_classes == ["equity_etf"]

    def test_parses_settings_block(self):
        cfg = ShortingConfig.from_config(
            {
                "shorting": {
                    "enabled": True,
                    "asset_classes": ["equity_etf"],
                    "max_short_notional_pct": 0.30,
                    "max_single_short_pct": 0.08,
                    "min_signal_strength": 0.0,
                    "universe": "liquid_etfs_only",
                    "cover_on_regime_improvement": True,
                    "hard_stop_pct": 0.08,
                }
            }
        )
        assert cfg.enabled
        assert cfg.universe_symbols == LIQUID_ETF_UNIVERSE
        assert len(cfg.universe_symbols) == 16


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

class TestRegime:
    def test_bear_when_below_ma200(self):
        # First 200 days at 100, then drop to 80 → last < MA200
        close = pd.Series(
            np.concatenate([np.full(200, 100.0), np.full(100, 80.0)])
        )
        assert is_bear_regime(close) is True

    def test_bull_when_above_ma200(self):
        close = pd.Series(np.concatenate([np.full(200, 100.0), np.full(100, 120.0)]))
        assert is_bear_regime(close) is False

    def test_fail_safe_insufficient_history(self):
        assert is_bear_regime(pd.Series([100.0] * 100)) is False
        assert is_bear_regime(None) is False

    def test_below_ma_helper(self):
        falling = _hist(200, 100)["Close"]
        rising = _hist(100, 200)["Close"]
        assert below_ma(falling) is True
        assert below_ma(rising) is False


# ---------------------------------------------------------------------------
# Eligibility + sizing
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_disabled_returns_empty(self):
        out = compute_short_targets(
            {"SPY": -0.5},
            {"SPY": _hist(200, 100)},
            cfg=_cfg(enabled=False),
            bear_regime=True,
            long_gross=0.0,
            max_portfolio_heat=0.95,
        )
        assert out == {}

    def test_positive_signal_never_shorted(self):
        out = compute_short_targets(
            {"SPY": 0.5, "QQQ": -0.5},
            {"SPY": _hist(200, 100), "QQQ": _hist(200, 100)},
            cfg=_cfg(),
            bear_regime=True,
            long_gross=0.0,
            max_portfolio_heat=0.95,
        )
        assert "SPY" not in out
        assert "QQQ" in out

    def test_non_bear_requires_below_own_ma200(self):
        # QQQ falling (below own MA), SPY falling but above its own MA? use flat:
        hist = {
            "QQQ": _hist(200, 100),   # falling → below MA200
            "SPY": _hist(100, 110),   # gently rising → above MA200
        }
        sigs = {"QQQ": -0.3, "SPY": -0.3}
        out = compute_short_targets(
            sigs, hist, cfg=_cfg(), bear_regime=False,
            long_gross=0.0, max_portfolio_heat=0.95,
        )
        assert "QQQ" in out
        assert "SPY" not in out  # not below its own MA200 outside bear regime

    def test_bear_regime_relaxes_ma_requirement(self):
        hist = {"SPY": _hist(100, 110)}  # above its own MA200
        out = compute_short_targets(
            {"SPY": -0.3}, hist, cfg=_cfg(), bear_regime=True,
            long_gross=0.0, max_portfolio_heat=0.95,
        )
        assert "SPY" in out

    def test_only_universe_members_shortable(self):
        out = compute_short_targets(
            {"AAPL": -0.9, "SPY": -0.2},
            {"AAPL": _hist(200, 100), "SPY": _hist(200, 100)},
            cfg=_cfg(),
            bear_regime=True,
            long_gross=0.0,
            max_portfolio_heat=0.95,
        )
        assert "AAPL" not in out  # not in liquid ETF universe
        assert "SPY" in out

    def test_no_crypto_or_futures_shorts(self):
        out = compute_short_targets(
            {"BTC-USD": -0.9, "ES=F": -0.9, "XLE": -0.2},
            {"BTC-USD": _hist(200, 100), "ES=F": _hist(200, 100), "XLE": _hist(200, 100)},
            cfg=_cfg(),
            bear_regime=True,
            long_gross=0.0,
            max_portfolio_heat=0.95,
        )
        assert set(out) == {"XLE"}


class TestSizing:
    def test_aggregate_cap(self):
        sigs = {s: -0.2 for s in LIQUID_ETF_UNIVERSE}
        hist = {s: _hist(200, 100) for s in LIQUID_ETF_UNIVERSE}
        out = compute_short_targets(
            sigs, hist, cfg=_cfg(), bear_regime=True,
            long_gross=0.0, max_portfolio_heat=0.95,
        )
        total = sum(abs(w) for w in out.values())
        assert total <= 0.30 + 1e-9
        assert all(w < 0 for w in out.values())

    def test_single_name_cap(self):
        sigs = {"SPY": -0.9, "QQQ": -0.01}
        hist = {"SPY": _hist(200, 100), "QQQ": _hist(200, 100)}
        out = compute_short_targets(
            sigs, hist, cfg=_cfg(), bear_regime=True,
            long_gross=0.0, max_portfolio_heat=0.95,
        )
        assert abs(out["SPY"]) <= 0.08 + 1e-9

    def test_heat_headroom_limits_shorts(self):
        sigs = {s: -0.2 for s in LIQUID_ETF_UNIVERSE}
        hist = {s: _hist(200, 100) for s in LIQUID_ETF_UNIVERSE}
        out = compute_short_targets(
            sigs, hist, cfg=_cfg(), bear_regime=True,
            long_gross=0.80, max_portfolio_heat=0.95,
        )
        total = sum(abs(w) for w in out.values())
        assert total <= 0.15 + 1e-9  # only 15% heat headroom remains
        assert 0.80 + total <= 0.95 + 1e-9

    def test_no_heat_headroom_no_shorts(self):
        sigs = {s: -0.2 for s in LIQUID_ETF_UNIVERSE}
        hist = {s: _hist(200, 100) for s in LIQUID_ETF_UNIVERSE}
        out = compute_short_targets(
            sigs, hist, cfg=_cfg(), bear_regime=True,
            long_gross=0.95, max_portfolio_heat=0.95,
        )
        assert out == {}

    def test_guard_blocks_non_shortable(self):
        class _NoShortGuard:
            def is_shortable(self, sym):
                return sym != "QQQ"

        sigs = {"SPY": -0.3, "QQQ": -0.3}
        hist = {"SPY": _hist(200, 100), "QQQ": _hist(200, 100)}
        out = compute_short_targets(
            sigs, hist, cfg=_cfg(), bear_regime=True,
            long_gross=0.0, max_portfolio_heat=0.95, guard=_NoShortGuard(),
        )
        assert "SPY" in out
        assert "QQQ" not in out

    def test_weights_proportional_to_signal_magnitude(self):
        sigs = {"SPY": -0.6, "QQQ": -0.3}
        hist = {"SPY": _hist(200, 100), "QQQ": _hist(200, 100)}
        out = compute_short_targets(
            sigs, hist, cfg=_cfg(max_single_short_pct=0.30), bear_regime=True,
            long_gross=0.0, max_portfolio_heat=0.95,
        )
        assert abs(out["SPY"]) > abs(out["QQQ"])
        assert abs(out["SPY"]) / abs(out["QQQ"]) == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Merge + cover semantics
# ---------------------------------------------------------------------------

class TestMergeAndCover:
    def test_merge_replaces_long_target(self):
        longs = {"SPY": 0.10, "TLT": 0.05}
        shorts = {"SPY": -0.08}
        out = merge_short_targets(longs, shorts)
        assert out["SPY"] == -0.08
        assert out["TLT"] == 0.05

    def test_regime_improvement_produces_empty_overlay(self):
        # Regime flip bear→bull: no new targets, callers zero held shorts
        out = compute_short_targets(
            {"SPY": -0.3},
            {"SPY": _hist(100, 120)},  # above MA200
            cfg=_cfg(),
            bear_regime=False,
            long_gross=0.0,
            max_portfolio_heat=0.95,
        )
        assert out == {}


# ---------------------------------------------------------------------------
# Hard stops
# ---------------------------------------------------------------------------

class TestHardStops:
    def test_cover_when_up_past_stop(self):
        positions = {"QQQ": {"quantity": -10.0, "avg_price": 100.0}}
        covers = hard_stop_covers(positions, {"QQQ": 108.5}, 0.08)
        assert covers == ["QQQ"]

    def test_no_cover_below_stop(self):
        positions = {"QQQ": {"quantity": -10.0, "avg_price": 100.0}}
        covers = hard_stop_covers(positions, {"QQQ": 107.9}, 0.08)
        assert covers == []

    def test_winning_short_not_stopped(self):
        positions = {"QQQ": {"quantity": -10.0, "avg_price": 100.0}}
        covers = hard_stop_covers(positions, {"QQQ": 90.0}, 0.08)
        assert covers == []

    def test_longs_ignored(self):
        positions = {"SPY": {"quantity": 10.0, "avg_price": 100.0}}
        covers = hard_stop_covers(positions, {"SPY": 120.0}, 0.08)
        assert covers == []

    def test_avg_entry_price_key_accepted(self):
        positions = {"QQQ": {"quantity": -10.0, "avg_entry_price": 100.0}}
        covers = hard_stop_covers(positions, {"QQQ": 109.0}, 0.08)
        assert covers == ["QQQ"]

    def test_missing_price_skipped(self):
        positions = {"QQQ": {"quantity": -10.0, "avg_price": 100.0}}
        assert hard_stop_covers(positions, {}, 0.08) == []


# ---------------------------------------------------------------------------
# Portfolio accounting with shorts (core/portfolio.py)
# ---------------------------------------------------------------------------

class TestPortfolioShorts:
    def _portfolio(self):
        from core.portfolio import Portfolio

        return Portfolio({"capital": {"initial_equity": 100000}})

    def test_open_short_increases_cash(self):
        p = self._portfolio()
        p.execute_order("QQQ", -100, 100.0, pd.Timestamp("2026-01-05"), 0.0, 0.0)
        assert p.positions["QQQ"].quantity == pytest.approx(-100)
        # Cash rises by proceeds minus transaction costs
        assert p.cash > 100000 + 100 * 100.0 * 0.99
        # equity ≈ unchanged (cash + short liability cancel out, minus costs)
        p.update_prices({"QQQ": 100.0})
        assert p.equity == pytest.approx(100000, rel=0.01)
        assert p.equity < 100000  # only costs lost on entry day at same price

    def test_short_pnl_sign_correct(self):
        p = self._portfolio()
        p.execute_order("QQQ", -100, 100.0, pd.Timestamp("2026-01-05"), 0.0, 0.0)
        p.update_prices({"QQQ": 90.0})
        pos = p.positions["QQQ"]
        # gained ~$1000 as price fell (entry is fill-adjusted by half-spread)
        assert pos.unrealised_pnl == pytest.approx(1000.0, rel=0.02)
        assert pos.unrealised_pnl_pct > 0  # positive P&L % for a winning short
        p.update_prices({"QQQ": 110.0})
        assert pos.unrealised_pnl == pytest.approx(-1000.0, rel=0.02)
        assert pos.unrealised_pnl_pct < 0

    def test_cover_short_reduces_cash_and_flattens(self):
        p = self._portfolio()
        p.execute_order("QQQ", -100, 100.0, pd.Timestamp("2026-01-05"), 0.0, 0.0)
        p.update_prices({"QQQ": 95.0})
        p.execute_order("QQQ", 100, 95.0, pd.Timestamp("2026-01-06"), 0.0, 0.0)
        # Fully covered positions are removed from the book (flat)
        assert "QQQ" not in p.positions
        # profit locked into cash: 100*100 - 100*95 = +500 (minus tx costs)
        assert p.cash == pytest.approx(100500.0, rel=0.005)
        assert p.equity == pytest.approx(p.cash, rel=1e-6)
        assert p.equity > 100000  # net profit after costs

    def test_gross_exposure_counts_shorts(self):
        p = self._portfolio()
        p.execute_order("SPY", 100, 100.0, pd.Timestamp("2026-01-05"), 0.0, 0.0)
        p.execute_order("QQQ", -50, 100.0, pd.Timestamp("2026-01-05"), 0.0, 0.0)
        p.update_prices({"SPY": 100.0, "QQQ": 100.0})
        # |long 10k| + |short 5k| over equity
        assert p.gross_exposure == pytest.approx(15000.0 / p.equity)

    def test_compute_orders_opens_short_on_negative_target(self):
        p = self._portfolio()
        p.update_prices({"QQQ": 100.0})
        orders = p.compute_orders({"QQQ": -0.05}, {"QQQ": 100.0})
        assert orders["QQQ"] < 0
        assert abs(orders["QQQ"] * 100.0) == pytest.approx(0.05 * p.equity, rel=0.01)
