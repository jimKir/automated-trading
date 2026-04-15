"""
Tests for risk.capital_manager.CapitalManager
==============================================
Covers order validation, hedge reserve enforcement, cycle tracking,
capital health checks, and a simulated multi-order cycle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from risk.capital_manager import CapitalManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_account(equity: float = 100_000, cash: float = 40_000, buying_power: float = 80_000):
    """Return a minimal account-like object."""
    return SimpleNamespace(equity=equity, cash=cash, buying_power=buying_power)


def _make_account_dict(equity: float = 100_000, cash: float = 40_000, buying_power: float = 80_000):
    """Return a dict-style account snapshot."""
    return {"equity": equity, "cash": cash, "buying_power": buying_power}


# ---------------------------------------------------------------------------
# begin_cycle
# ---------------------------------------------------------------------------


class TestBeginCycle:
    def test_snapshots_values(self):
        cm = CapitalManager(hedge_reserve_pct=0.20, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))

        assert cm.equity == 100_000
        assert cm.cash == 40_000
        assert cm.hedge_reserve == 20_000  # 100k * 0.20
        assert cm.min_cash_floor == 5_000  # 100k * 0.05
        assert cm.available_for_trading == 15_000  # 40k - 20k - 5k
        assert cm.cycle_committed == 0.0

    def test_resets_cycle_committed(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account())
        cm.cycle_committed = 5_000
        cm.begin_cycle(_make_account())
        assert cm.cycle_committed == 0.0

    def test_dict_account_snapshot(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account_dict(equity=50_000, cash=20_000, buying_power=40_000))
        assert cm.equity == 50_000
        assert cm.cash == 20_000

    def test_available_never_negative(self):
        """When cash < reserves, available_for_trading should be 0."""
        cm = CapitalManager(hedge_reserve_pct=0.50, min_cash_pct=0.10)
        cm.begin_cycle(_make_account(equity=100_000, cash=10_000))
        # reserves = 50k + 10k = 60k, cash = 10k → available = 0
        assert cm.available_for_trading == 0.0


# ---------------------------------------------------------------------------
# validate_order — SELL always approved
# ---------------------------------------------------------------------------


class TestValidateOrderSell:
    def test_sell_always_approved(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=0))  # zero cash
        approved, qty, reason = cm.validate_order("AAPL", "SELL", 100, 150.0)
        assert approved is True
        assert qty == 100
        assert reason == "sell_approved"

    def test_sell_does_not_change_committed(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account())
        cm.validate_order("AAPL", "SELL", 50, 200.0)
        assert cm.cycle_committed == 0.0

    def test_sell_case_insensitive(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=0))
        approved, _, _ = cm.validate_order("AAPL", "sell", 10, 100.0)
        assert approved is True


# ---------------------------------------------------------------------------
# validate_order — BUY approved
# ---------------------------------------------------------------------------


class TestValidateOrderApprove:
    def test_normal_buy_approved(self):
        cm = CapitalManager(hedge_reserve_pct=0.20, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        # available = 15_000 (40k - 20k - 5k)
        approved, qty, reason = cm.validate_order("AAPL", "BUY", 50, 150.0)
        # order_cost = 50 * 150 = 7500 < 15000
        assert approved is True
        assert qty == 50
        assert reason == "approved"
        assert cm.cycle_committed == 7_500.0

    def test_buy_updates_committed(self):
        cm = CapitalManager(hedge_reserve_pct=0.10, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=50_000))
        # available = 50k - 10k - 5k = 35k
        cm.validate_order("AAPL", "BUY", 100, 100.0)  # costs 10k
        assert cm.cycle_committed == 10_000.0
        cm.validate_order("MSFT", "BUY", 50, 200.0)  # costs 10k
        assert cm.cycle_committed == 20_000.0


# ---------------------------------------------------------------------------
# validate_order — BUY rejected
# ---------------------------------------------------------------------------


class TestValidateOrderReject:
    def test_buy_rejected_when_capital_exhausted(self):
        cm = CapitalManager(hedge_reserve_pct=0.20, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=25_100))
        # available = 25100 - 20000 - 5000 = 100
        # Trying to buy 10 shares at $100 = $1000 > 100
        # remaining_capital = 100, price = 100, so 100 <= 100 — can't afford even 1 share
        approved, qty, reason = cm.validate_order("AAPL", "BUY", 10, 100.0)
        assert approved is False
        assert qty == 0.0
        assert "insufficient capital" in reason

    def test_buy_rejected_zero_available(self):
        cm = CapitalManager(hedge_reserve_pct=0.50, min_cash_pct=0.10)
        cm.begin_cycle(_make_account(equity=100_000, cash=10_000))
        # available = 0 (cash < reserves)
        approved, _qty, _reason = cm.validate_order("AAPL", "BUY", 1, 50.0)
        assert approved is False


# ---------------------------------------------------------------------------
# validate_order — BUY adjusted down
# ---------------------------------------------------------------------------


class TestValidateOrderAdjust:
    def test_buy_adjusted_when_partially_affordable(self):
        cm = CapitalManager(hedge_reserve_pct=0.20, min_cash_pct=0.05, max_single_order_pct=1.0)
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        # available = 15_000, max_single = 100k (effectively uncapped)
        # Try to buy 200 shares at $100 = $20_000 > 15_000
        # remaining_capital = 15_000, price = 100, so can afford 150 shares
        approved, qty, reason = cm.validate_order("AAPL", "BUY", 200, 100.0)
        assert approved is True
        assert qty == 150.0  # 15_000 / 100
        assert "adjusted_down" in reason

    def test_adjusted_qty_respects_max_single_order(self):
        """When adjusting down, the max_single_order cap should still apply."""
        cm = CapitalManager(hedge_reserve_pct=0.10, min_cash_pct=0.05, max_single_order_pct=0.10)
        cm.begin_cycle(_make_account(equity=100_000, cash=100_000))
        # available = 100k - 10k - 5k = 85k
        # max_single = 10k
        # Try to buy 2000 @ $100 = 200k → clamped to 100 shares (10k)
        approved, qty, _reason = cm.validate_order("AAPL", "BUY", 2000, 100.0)
        assert approved is True
        assert qty == 100.0  # 10_000 / 100 = 100


# ---------------------------------------------------------------------------
# max_single_order_pct clamping
# ---------------------------------------------------------------------------


class TestMaxSingleOrder:
    def test_clamp_to_max_single_order(self):
        cm = CapitalManager(hedge_reserve_pct=0.10, min_cash_pct=0.05, max_single_order_pct=0.10)
        cm.begin_cycle(_make_account(equity=100_000, cash=100_000))
        # max_single = 10k, try 200 @ $100 = 20k → clamped to 100 shares
        approved, qty, _reason = cm.validate_order("AAPL", "BUY", 200, 100.0)
        assert approved is True
        assert qty == 100.0
        assert cm.cycle_committed == 10_000.0


# ---------------------------------------------------------------------------
# cycle_committed tracking across multiple orders
# ---------------------------------------------------------------------------


class TestCycleCommitted:
    def test_accumulates_across_orders(self):
        cm = CapitalManager(hedge_reserve_pct=0.10, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=50_000))
        # available = 35k
        cm.validate_order("AAPL", "BUY", 100, 100.0)  # 10k
        cm.validate_order("MSFT", "BUY", 50, 200.0)  # 10k
        cm.validate_order("GOOGL", "BUY", 50, 100.0)  # 5k
        assert cm.cycle_committed == 25_000.0

    def test_eventual_rejection_after_exhaustion(self):
        cm = CapitalManager(hedge_reserve_pct=0.20, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        # available = 15k
        cm.validate_order("AAPL", "BUY", 100, 100.0)  # 10k committed
        cm.validate_order("MSFT", "BUY", 50, 100.0)  # 5k committed
        # Now exhausted (15k committed)
        approved, _, reason = cm.validate_order("GOOGL", "BUY", 10, 100.0)
        assert approved is False
        assert "insufficient capital" in reason

    def test_sells_do_not_affect_committed(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        cm.validate_order("AAPL", "SELL", 100, 150.0)
        assert cm.cycle_committed == 0.0


# ---------------------------------------------------------------------------
# get_capital_status
# ---------------------------------------------------------------------------


class TestGetCapitalStatus:
    def test_returns_all_keys(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        status = cm.get_capital_status()
        expected_keys = {
            "equity",
            "cash",
            "buying_power",
            "hedge_reserve",
            "min_cash_floor",
            "available_for_trading",
            "cycle_committed",
            "remaining_this_cycle",
            "deployed_pct",
            "cash_pct",
        }
        assert set(status.keys()) == expected_keys

    def test_deployed_pct_calculation(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        status = cm.get_capital_status()
        assert status["deployed_pct"] == pytest.approx(0.60)  # (100k - 40k) / 100k

    def test_cash_pct_calculation(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        status = cm.get_capital_status()
        assert status["cash_pct"] == pytest.approx(0.40)

    def test_remaining_updates_after_orders(self):
        cm = CapitalManager(hedge_reserve_pct=0.20, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        cm.validate_order("AAPL", "BUY", 50, 100.0)  # 5k
        status = cm.get_capital_status()
        assert status["remaining_this_cycle"] == pytest.approx(10_000.0)  # 15k - 5k


# ---------------------------------------------------------------------------
# check_capital_health
# ---------------------------------------------------------------------------


class TestCheckCapitalHealth:
    def test_all_pass_when_healthy(self):
        cm = CapitalManager(hedge_reserve_pct=0.20, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))
        results = cm.check_capital_health()
        assert len(results) == 3
        for r in results:
            assert r["status"] == "PASS"

    def test_cash_below_hedge_reserve_fails(self):
        cm = CapitalManager(hedge_reserve_pct=0.50, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=30_000))
        # hedge_reserve = 50k, cash = 30k → FAIL
        results = {r["check"]: r for r in cm.check_capital_health()}
        assert results["cash_below_hedge_reserve"]["status"] == "FAIL"

    def test_cash_below_min_floor_fails(self):
        cm = CapitalManager(hedge_reserve_pct=0.01, min_cash_pct=0.10)
        cm.begin_cycle(_make_account(equity=100_000, cash=5_000))
        # min_floor = 10k, cash = 5k → FAIL
        results = {r["check"]: r for r in cm.check_capital_health()}
        assert results["cash_below_min_floor"]["status"] == "FAIL"

    def test_deployed_ratio_extreme_fails(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=3_000))
        # deployed = 97% > 95% → FAIL
        results = {r["check"]: r for r in cm.check_capital_health()}
        assert results["deployed_ratio_extreme"]["status"] == "FAIL"

    def test_deployed_ratio_passes_at_boundary(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=5_000))
        # deployed = 95% → PASS (not strictly > 95%)
        results = {r["check"]: r for r in cm.check_capital_health()}
        assert results["deployed_ratio_extreme"]["status"] == "PASS"

    def test_result_structure(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account())
        results = cm.check_capital_health()
        for r in results:
            assert "check" in r
            assert "value" in r
            assert "threshold" in r
            assert "status" in r
            assert r["status"] in ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# Integration: simulated cycle prevents over-commitment
# ---------------------------------------------------------------------------


class TestIntegrationCycle:
    def test_prevents_over_commitment(self):
        """Simulate a full cycle with multiple BUY orders — capital manager
        should prevent committing more than available_for_trading."""
        cm = CapitalManager(
            hedge_reserve_pct=0.20,
            min_cash_pct=0.05,
            max_single_order_pct=1.0,  # effectively uncapped for this test
        )
        account = _make_account(equity=100_000, cash=40_000)
        cm.begin_cycle(account)

        # available = 40k - 20k - 5k = 15k
        orders = [
            ("AAPL", "BUY", 50, 100.0),  # 5k — approved
            ("MSFT", "BUY", 30, 200.0),  # 6k — approved
            ("GOOGL", "BUY", 20, 150.0),  # 3k — approved (4k remaining, 3k fits)
            ("TSLA", "BUY", 100, 300.0),  # 30k — remaining is 1k, price 300 → adjusted to ~3.33
            ("NVDA", "BUY", 10, 500.0),  # 5k — remaining nearly 0, can't afford 1 share
        ]

        results = []
        for sym, side, qty, price in orders:
            approved, adj_qty, reason = cm.validate_order(sym, side, qty, price)
            results.append((approved, adj_qty, reason))

        # First three fully approved
        assert results[0] == (True, 50, "approved")
        assert results[1] == (True, 30, "approved")
        assert results[2] == (True, 20, "approved")

        # Fourth: adjusted down (1k remaining / $300 ≈ 3.33 shares)
        assert results[3][0] is True
        assert results[3][1] == pytest.approx(1_000.0 / 300.0, rel=1e-4)
        assert "adjusted_down" in results[3][2]

        # Fifth: rejected (remaining ≈ 0, can't afford 1 share at $500)
        assert results[4][0] is False

        # Total committed should not exceed available
        assert cm.cycle_committed <= cm.available_for_trading + 0.01

    def test_mixed_buy_sell_cycle(self):
        """SELLs should not consume capital; BUYs should."""
        cm = CapitalManager(hedge_reserve_pct=0.20, min_cash_pct=0.05)
        cm.begin_cycle(_make_account(equity=100_000, cash=40_000))

        cm.validate_order("AAPL", "SELL", 50, 100.0)
        assert cm.cycle_committed == 0.0

        cm.validate_order("MSFT", "BUY", 100, 100.0)  # 10k
        assert cm.cycle_committed == 10_000.0

        cm.validate_order("GOOGL", "SELL", 20, 200.0)
        assert cm.cycle_committed == 10_000.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_equity(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=0, cash=0, buying_power=0))
        assert cm.available_for_trading == 0.0
        status = cm.get_capital_status()
        assert status["deployed_pct"] == 0.0
        assert status["cash_pct"] == 0.0

    def test_zero_price_order(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=50_000))
        # Zero price → order_cost = 0, should be approved
        approved, _qty, _reason = cm.validate_order("AAPL", "BUY", 100, 0.0)
        assert approved is True

    def test_very_small_order(self):
        cm = CapitalManager()
        cm.begin_cycle(_make_account(equity=100_000, cash=50_000))
        approved, qty, _reason = cm.validate_order("AAPL", "BUY", 0.001, 100.0)
        assert approved is True
        assert qty == 0.001
