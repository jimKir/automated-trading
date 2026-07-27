"""
Regression tests for same-session marking of newly opened positions.

A fresh Position starts at current_price 0 and the backtest engine only calls
update_prices at the top of the next day. Before this fix a position therefore
contributed no market value for the remainder of its entry session: recorded equity
fell by the entire notional on every entry day and recovered the next one. Over the
2026 window that produced 103% annualised volatility and a -43% max drawdown on a
strategy whose live realised volatility is under 7%.

The same staleness also understated `equity` inside execute_order, which is what the
hedge-reserve cash check is computed from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from core.portfolio import Portfolio

CONFIG = {
    "capital": {
        "initial_equity": 100_000,
        "hedge_reserve_pct": 0.0,
        "min_cash_pct": 0.0,
    }
}

DATE = pd.Timestamp("2026-01-02", tz="UTC")


@pytest.fixture
def portfolio():
    return Portfolio(CONFIG)


def test_entry_does_not_destroy_equity(portfolio):
    """Buying an asset converts cash into position value; it does not burn it."""
    portfolio.execute_order("SPY", 100, 500.0, DATE)

    # Equity falls only by transaction costs, not by the $50k notional.
    assert portfolio.equity == pytest.approx(100_000, rel=2e-3)
    assert portfolio.equity < 100_000


def test_position_is_marked_at_the_traded_price(portfolio):
    portfolio.execute_order("SPY", 100, 500.0, DATE)
    assert portfolio.positions["SPY"].current_price == 500.0
    assert portfolio.positions["SPY"].market_value == pytest.approx(50_000)


def test_recorded_equity_is_flat_across_an_entry_day(portfolio):
    """
    The bug's signature: equity recorded on the entry day sat a full notional below
    the next day's, so every rebalance looked like a crash-and-rebound.
    """
    portfolio.record_equity(DATE)
    portfolio.execute_order("SPY", 100, 500.0, DATE)
    portfolio.record_equity(DATE + pd.Timedelta(days=1))

    curve = portfolio.get_equity_series()
    assert curve.pct_change().dropna().abs().max() < 0.01


def test_heat_reflects_positions_on_the_entry_day(portfolio):
    portfolio.execute_order("SPY", 100, 500.0, DATE)
    assert portfolio.gross_exposure == pytest.approx(0.5, abs=5e-3)


def test_gross_exposure_counts_shorts(portfolio):
    portfolio.execute_order("SPY", 100, 500.0, DATE)
    portfolio.execute_order("TLT", -100, 100.0, DATE)
    # 50k long + 10k short against ~100k equity.
    assert portfolio.gross_exposure == pytest.approx(0.6, abs=1e-2)


def test_gross_exposure_is_zero_with_no_positions(portfolio):
    assert portfolio.gross_exposure == 0.0


def test_later_price_updates_still_apply(portfolio):
    """Marking at fill must not shadow the daily mark-to-market."""
    portfolio.execute_order("SPY", 100, 500.0, DATE)
    portfolio.update_prices({"SPY": 550.0})
    assert portfolio.positions["SPY"].market_value == pytest.approx(55_000)
    assert portfolio.equity == pytest.approx(105_000, rel=2e-3)
