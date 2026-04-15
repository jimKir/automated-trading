"""
Capital Manager
===============
Validates buying power before each order, enforces a cash reserve for
hedging, and monitors deployed-vs-available ratio.

Integrates with LiveEngine (real-time) and BacktestEngine (simulation)
to ensure OOS results reflect the same capital constraints as live.
"""

from __future__ import annotations

from utils.logger import get_logger

log = get_logger("CapitalManager")


class CapitalManager:
    """Track buying power, enforce hedge reserve and min cash floor."""

    def __init__(
        self,
        hedge_reserve_pct: float = 0.20,
        min_cash_pct: float = 0.05,
        max_single_order_pct: float = 0.15,
    ):
        self.hedge_reserve_pct = hedge_reserve_pct
        self.min_cash_pct = min_cash_pct
        self.max_single_order_pct = max_single_order_pct

        # Snapshot values (set by begin_cycle)
        self.equity: float = 0.0
        self.cash: float = 0.0
        self.buying_power: float = 0.0
        self.hedge_reserve: float = 0.0
        self.min_cash_floor: float = 0.0
        self.available_for_trading: float = 0.0

        # Tracks cumulative $ committed to BUY orders within a single cycle
        self.cycle_committed: float = 0.0

    # ------------------------------------------------------------------
    # Cycle management
    # ------------------------------------------------------------------

    def begin_cycle(self, account_snapshot) -> None:
        """Snapshot capital at the start of a trading cycle.

        Parameters
        ----------
        account_snapshot : object or dict
            Must expose ``equity``, ``cash``, and ``buying_power`` as
            attributes or dict keys.
        """
        if hasattr(account_snapshot, "equity"):
            self.equity = float(account_snapshot.equity)
            self.cash = float(account_snapshot.cash)
            self.buying_power = float(account_snapshot.buying_power)
        else:
            self.equity = float(account_snapshot["equity"])
            self.cash = float(account_snapshot["cash"])
            self.buying_power = float(account_snapshot["buying_power"])

        self.hedge_reserve = self.equity * self.hedge_reserve_pct
        self.min_cash_floor = self.equity * self.min_cash_pct
        self.available_for_trading = max(
            self.cash - self.hedge_reserve - self.min_cash_floor, 0.0
        )
        self.cycle_committed = 0.0

        log.info(
            f"[CAPITAL] equity=${self.equity:,.2f} cash=${self.cash:,.2f} "
            f"hedge_reserve=${self.hedge_reserve:,.2f} "
            f"available=${self.available_for_trading:,.2f}"
        )

    # ------------------------------------------------------------------
    # Order validation
    # ------------------------------------------------------------------

    def validate_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> tuple[bool, float, str]:
        """Validate a proposed order against available capital.

        Returns
        -------
        (approved, adjusted_qty, reason)
        """
        # SELLs always approved — selling frees capital
        if side.upper() == "SELL":
            return True, qty, "sell_approved"

        order_cost = qty * price
        remaining_capital = self.available_for_trading - self.cycle_committed

        # Check single-order size cap
        max_single = self.max_single_order_pct * self.equity
        if max_single > 0 and order_cost > max_single:
            clamped_qty = max_single / price
            log.info(
                f"[CAPITAL] {symbol}: clamped to max_single_order "
                f"qty {qty:.4f} -> {clamped_qty:.4f}"
            )
            qty = clamped_qty
            order_cost = qty * price

        if order_cost <= remaining_capital:
            self.cycle_committed += order_cost
            return True, qty, "approved"

        # Not enough for full order — try partial
        if remaining_capital > price:
            adjusted_qty = remaining_capital / price
            # Re-check single-order cap on the adjusted qty
            if max_single > 0 and adjusted_qty * price > max_single:
                adjusted_qty = max_single / price
            adjusted_cost = adjusted_qty * price
            self.cycle_committed += adjusted_cost
            return (
                True,
                adjusted_qty,
                f"adjusted_down: remaining_capital=${remaining_capital:,.2f}",
            )

        # Cannot afford even 1 share
        return (
            False,
            0.0,
            "insufficient capital after hedge reserve",
        )

    # ------------------------------------------------------------------
    # Status / health
    # ------------------------------------------------------------------

    def get_capital_status(self) -> dict:
        """Return a full capital breakdown."""
        deployed = self.equity - self.cash
        return {
            "equity": self.equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "hedge_reserve": self.hedge_reserve,
            "min_cash_floor": self.min_cash_floor,
            "available_for_trading": self.available_for_trading,
            "cycle_committed": self.cycle_committed,
            "remaining_this_cycle": max(
                self.available_for_trading - self.cycle_committed, 0.0
            ),
            "deployed_pct": (deployed / self.equity) if self.equity > 0 else 0.0,
            "cash_pct": (self.cash / self.equity) if self.equity > 0 else 0.0,
        }

    def check_capital_health(self) -> list[dict]:
        """Return anomaly-style results for capital health checks.

        Each result has the same shape the AnomalyDetector expects:
        ``{"check": str, "value": float, "threshold": float, "status": "PASS"|"FAIL"}``
        """
        results: list[dict] = []

        # 1. Cash below hedge reserve
        results.append(
            {
                "check": "cash_below_hedge_reserve",
                "value": self.cash,
                "threshold": self.hedge_reserve,
                "status": "FAIL" if self.cash < self.hedge_reserve else "PASS",
            }
        )

        # 2. Cash below minimum floor
        results.append(
            {
                "check": "cash_below_min_floor",
                "value": self.cash,
                "threshold": self.min_cash_floor,
                "status": "FAIL" if self.cash < self.min_cash_floor else "PASS",
            }
        )

        # 3. Deployed ratio extreme (> 95%)
        deployed_pct = (
            (self.equity - self.cash) / self.equity if self.equity > 0 else 0.0
        )
        results.append(
            {
                "check": "deployed_ratio_extreme",
                "value": round(deployed_pct, 4),
                "threshold": 0.95,
                "status": "FAIL" if deployed_pct > 0.95 else "PASS",
            }
        )

        return results
