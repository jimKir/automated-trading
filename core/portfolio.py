"""
Portfolio Manager
=================
Tracks positions, cash, P&L, and generates target weights from signals.
Uses CostModel for realistic transaction and carrying cost simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.cost_model import CostModel
from utils.logger import get_logger

log = get_logger("Portfolio")


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_entry_price)

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


class Portfolio:
    def __init__(self, config: dict):
        self.initial_equity = config.get("capital", {}).get("initial_equity", 25000)
        self.cash = float(self.initial_equity)
        self.positions: dict[str, Position] = {}
        self.trade_log: list = []
        self.equity_curve: list = []
        self._equity_curve_dates: list = []
        self.cost_model = CostModel(config)
        self._config = config
        # Futures roll tracking
        self._last_roll_date: dict[str, pd.Timestamp] = {}
        # Optional optimizer
        self._optimizer = None
        if config.get("optimizer", {}).get("enabled", False):
            try:
                from core.optimizer import PortfolioOptimizer

                self._optimizer = PortfolioOptimizer(config)
            except Exception as e:
                log.warning(f"Optimizer failed to load: {e}")

    # -----------------------------------------------------------------------
    # Equity
    # -----------------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def record_equity(self, date: pd.Timestamp) -> None:
        self._equity_curve_dates.append(date)
        self.equity_curve.append(self.equity)

    def get_equity_series(self) -> pd.Series:
        return pd.Series(self.equity_curve, index=self._equity_curve_dates, name="equity")

    # -----------------------------------------------------------------------
    # Target weights → orders
    # -----------------------------------------------------------------------

    def compute_target_weights(
        self,
        signals: dict[str, float],
        max_position_pct: float = 0.15,
        max_portfolio_heat: float = 0.40,
        price_history: dict[str, pd.DataFrame] | None = None,
        as_of_date: pd.Timestamp | None = None,
        spy_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Convert signal strengths to target portfolio weights.

        If optimizer is enabled, delegates to PortfolioOptimizer which uses
        risk parity or minimum variance sizing with crypto cap + regime scaling.

        Otherwise uses the original signal-proportional method.

        Returns dict of {symbol: weight}
        """
        if not signals:
            return {}

        # ── Optimized path ────────────────────────────────────────────────────
        if self._optimizer is not None and price_history is not None:
            return self._optimizer.compute_weights(
                signals=signals,
                price_history=price_history,
                max_position_pct=max_position_pct,
                max_portfolio_heat=max_portfolio_heat,
                as_of_date=as_of_date,
                spy_data=spy_data,
            )

        # ── Original signal-proportional path (fallback) ──────────────────────
        active = {k: v for k, v in signals.items() if abs(v) > 0.05}
        if not active:
            return dict.fromkeys(signals, 0.0)

        longs = {k: v for k, v in active.items() if v > 0}
        shorts = {k: v for k, v in active.items() if v < 0}
        weights: dict[str, float] = {}

        if longs:
            total_long_signal = sum(longs.values())
            long_budget = min(max_portfolio_heat * 0.7, 1.0)
            for sym, sig in longs.items():
                w = (sig / total_long_signal) * long_budget
                weights[sym] = min(w, max_position_pct)

        if shorts:
            total_short_signal = sum(abs(v) for v in shorts.values())
            short_budget = min(max_portfolio_heat * 0.3, 0.15)
            for sym, sig in shorts.items():
                w = (abs(sig) / total_short_signal) * short_budget
                weights[sym] = -min(w, max_position_pct * 0.5)

        for sym in signals:
            if sym not in weights:
                weights[sym] = 0.0

        return weights

    def compute_orders(
        self,
        target_weights: dict[str, float],
        current_prices: dict[str, float],
    ) -> dict[str, float]:
        """
        Compare target vs current positions to generate order quantities.
        Positive = buy, Negative = sell/short.

        Returns {symbol: quantity_change}
        """
        orders: dict[str, float] = {}
        equity = self.equity

        for sym, target_w in target_weights.items():
            price = current_prices.get(sym)
            if not price or price <= 0:
                continue

            target_value = target_w * equity
            current_value = self.positions[sym].market_value if sym in self.positions else 0.0
            delta_value = target_value - current_value

            # v15b: minimum trade threshold = max(1 unit, 0.5% of equity)
            # Prevents tiny rebalance trades that create turnover and vol
            # without meaningfully changing portfolio allocation.
            min_trade = max(price, equity * 0.005)
            if abs(delta_value) < min_trade:
                continue

            orders[sym] = delta_value / price  # positive=buy, negative=sell

        return orders

    # -----------------------------------------------------------------------
    # Trade execution (simulation)
    # -----------------------------------------------------------------------

    def execute_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        date: pd.Timestamp,
        commission_pct: float = None,  # kept for API compat; ignored — CostModel is used
        slippage_pct: float = None,  # kept for API compat; ignored
    ) -> dict | None:
        """
        Simulate order execution using the full CostModel.
        Costs: commission + bid-ask half-spread + market impact.
        Returns trade record dict.
        """
        if abs(quantity) < 1e-8:
            return None

        notional = abs(quantity * price)
        is_buy = quantity > 0

        # Full transaction cost from CostModel (one-way: entry leg)
        cost_breakdown = self.cost_model.transaction_cost(
            symbol=symbol,
            notional=notional,
            is_buy=is_buy,
        )
        tx_cost = cost_breakdown.total_transaction

        # Spread applied as price impact: buys fill slightly higher, sells slightly lower
        # Half-spread is already in tx_cost — apply it as fill price adjustment too
        spread_adj = (cost_breakdown.half_spread / notional) if notional > 0 else 0
        fill_price = price * (1 + spread_adj * np.sign(quantity))

        # Total cash impact = fill value + all transaction costs
        total_debit = notional + tx_cost if is_buy else -(notional - tx_cost)  # negative = cash inflow minus costs

        # Check cash sufficiency for buys
        if is_buy and total_debit > self.cash:
            max_affordable = self.cash / (1 + tx_cost / notional) if notional > 0 else 0
            quantity = max_affordable / price
            if quantity < 1e-8:
                log.debug(f"[{symbol}] Insufficient cash — order skipped")
                return None
            notional = abs(quantity * price)
            cost_breakdown = self.cost_model.transaction_cost(symbol, notional, is_buy)
            tx_cost = cost_breakdown.total_transaction
            total_debit = notional + tx_cost

        self.cash -= total_debit

        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        pos = self.positions[symbol]
        old_qty = pos.quantity

        if (old_qty >= 0 and quantity > 0) or (old_qty <= 0 and quantity < 0):
            new_qty = old_qty + quantity
            if new_qty != 0:
                pos.avg_entry_price = (
                    old_qty * pos.avg_entry_price + quantity * fill_price
                ) / new_qty
            pos.quantity = new_qty
        else:
            # Position reversal (long→short or short→long)
            pos.quantity += quantity
            if abs(pos.quantity) < 1e-8:
                del self.positions[symbol]
            else:
                # Direction flipped — reset avg_entry_price to the new fill
                pos.avg_entry_price = fill_price

        trade = {
            "date": date,
            "symbol": symbol,
            "quantity": quantity,
            "fill_price": fill_price,
            "commission": cost_breakdown.commission,
            "half_spread": cost_breakdown.half_spread,
            "market_impact": cost_breakdown.market_impact,
            "total_tx_cost": tx_cost,
            "total_debit": total_debit,
            "cash_after": self.cash,
            "equity_after": self.equity,
        }
        self.trade_log.append(trade)
        return trade

    def apply_daily_costs(self, date: pd.Timestamp) -> float:
        """
        Charge overnight carrying costs for all open positions.
        Called once per trading day AFTER prices are updated.
        Returns total $ cost charged.
        """
        total_daily_cost = 0.0
        for sym, pos in list(self.positions.items()):
            if pos.quantity == 0 or pos.current_price <= 0:
                continue

            position_value = abs(pos.market_value)
            is_long = pos.quantity > 0

            carrying = self.cost_model.daily_carrying_cost(
                symbol=sym,
                position_value=position_value,
                is_long=is_long,
                is_leveraged=False,  # assume fully-funded (no margin)
            )
            self.cash -= carrying.total_carrying
            total_daily_cost += carrying.total_carrying

            # Futures roll check: roll every ~63 trading days (quarterly)
            from core.cost_model import _classify

            if _classify(sym) == "future":
                last_roll = self._last_roll_date.get(sym)
                if last_roll is None or (date - last_roll).days >= 90:
                    roll_cost = self.cost_model.futures_roll_cost(sym, position_value)
                    self.cash -= roll_cost
                    total_daily_cost += roll_cost
                    self._last_roll_date[sym] = date
                    if roll_cost > 0:
                        log.debug(f"[{date.date()}] Futures roll {sym}: ${roll_cost:.2f}")

        return total_daily_cost

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for sym, price in prices.items():
            if sym in self.positions:
                self.positions[sym].current_price = price

    def get_trade_df(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log).set_index("date")
