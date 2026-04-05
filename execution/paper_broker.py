"""
Paper Trading Broker
====================
Simulates real broker execution locally for testing.
No internet connection needed — uses a DataFeed for prices.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from execution.broker_base import (
    BrokerBase, Order, OrderSide, OrderStatus, OrderType, AccountInfo
)
from utils.logger import get_logger

log = get_logger("PaperBroker")


class PaperBroker(BrokerBase):
    def __init__(self, config: dict):
        self.config = config
        self.equity = config.get("capital", {}).get("initial_equity", 25000)
        self.cash = float(self.equity)
        self._positions: Dict[str, dict] = {}
        self._orders: Dict[str, Order] = {}
        self._commission_pct = config.get("backtest", {}).get("commission_pct", 0.001)
        self._slippage_pct = config.get("backtest", {}).get("slippage_pct", 0.0005)

    def connect(self) -> bool:
        log.info("PaperBroker connected (simulation mode)")
        return True

    def disconnect(self) -> None:
        log.info("PaperBroker disconnected")

    def get_account(self) -> AccountInfo:
        mkt_value = sum(
            pos["quantity"] * self.get_latest_price(sym)
            for sym, pos in self._positions.items()
        )
        total_equity = self.cash + mkt_value
        return AccountInfo(
            account_id="paper-account-001",
            equity=total_equity,
            cash=self.cash,
            buying_power=self.cash,
            currency="USD",
            positions=self._positions.copy(),
        )

    def get_position(self, symbol: str) -> Optional[dict]:
        return self._positions.get(symbol)

    def get_positions(self) -> Dict[str, dict]:
        return self._positions.copy()

    def place_order(self, order: Order) -> Order:
        price = self.get_latest_price(order.symbol)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            log.warning(f"[{order.symbol}] Cannot get price — order rejected")
            return order

        # Apply slippage
        slippage = self._slippage_pct * (1 if order.side == OrderSide.BUY else -1)
        fill_price = price * (1 + slippage)
        commission = abs(order.quantity * fill_price) * self._commission_pct
        cost = order.quantity * fill_price

        if order.side == OrderSide.BUY:
            total_debit = cost + commission
            if total_debit > self.cash:
                max_qty = self.cash / (fill_price * (1 + self._commission_pct))
                order.quantity = max_qty
                if order.quantity < 0.001:
                    order.status = OrderStatus.REJECTED
                    return order
                commission = abs(order.quantity * fill_price) * self._commission_pct
                total_debit = order.quantity * fill_price + commission
            self.cash -= total_debit
            sym = order.symbol
            if sym not in self._positions:
                self._positions[sym] = {"quantity": 0, "avg_price": 0.0}
            pos = self._positions[sym]
            old_qty = pos["quantity"]
            new_qty = old_qty + order.quantity
            pos["avg_price"] = (old_qty * pos["avg_price"] + order.quantity * fill_price) / new_qty if new_qty else 0
            pos["quantity"] = new_qty

        else:  # SELL
            sym = order.symbol
            if sym not in self._positions or self._positions[sym]["quantity"] < 0.001:
                order.status = OrderStatus.REJECTED
                log.warning(f"[{sym}] No position to sell — order rejected")
                return order
            sell_qty = min(order.quantity, self._positions[sym]["quantity"])
            order.quantity = sell_qty
            commission = abs(sell_qty * fill_price) * self._commission_pct
            proceeds = sell_qty * fill_price - commission
            self.cash += proceeds
            self._positions[sym]["quantity"] -= sell_qty
            if abs(self._positions[sym]["quantity"]) < 0.001:
                del self._positions[sym]

        order_id = str(uuid.uuid4())[:8]
        order.order_id = order_id
        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        self._orders[order_id] = order

        log.info(
            f"[PAPER] {order.side.value.upper()} {order.symbol} "
            f"qty={order.quantity:.4f} @ ${fill_price:.4f} | "
            f"commission=${commission:.2f} | cash=${self.cash:.2f}"
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_latest_price(self, symbol: str) -> float:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d", auto_adjust=True)
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            log.warning(f"[{symbol}] Price fetch failed: {e}")
        return 0.0

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        prices = {}
        for sym in symbols:
            prices[sym] = self.get_latest_price(sym)
        return prices

    def is_market_open(self) -> bool:
        """Paper broker always trades."""
        return True
