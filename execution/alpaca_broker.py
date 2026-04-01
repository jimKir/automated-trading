"""
Alpaca Broker Adapter
======================
pip install alpaca-trade-api
Set env vars: ALPACA_API_KEY, ALPACA_API_SECRET
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from execution.broker_base import (
    BrokerBase, Order, OrderSide, OrderStatus, OrderType, AccountInfo
)
from utils.logger import get_logger

log = get_logger("AlpacaBroker")


class AlpacaBroker(BrokerBase):
    def __init__(self, config: dict):
        alpaca_cfg = config.get("brokers", {}).get("alpaca", {})
        self.api_key = alpaca_cfg.get("api_key") or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = alpaca_cfg.get("api_secret") or os.environ.get("ALPACA_API_SECRET", "")
        self.base_url = alpaca_cfg.get("base_url", "https://paper-api.alpaca.markets")
        self.api = None

    def connect(self) -> bool:
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(self.api_key, self.api_secret, base_url=self.base_url)
            account = self.api.get_account()
            log.info(f"Alpaca connected. Account status: {account.status}")
            return True
        except Exception as exc:
            log.error(f"Alpaca connection failed: {exc}")
            return False

    def disconnect(self) -> None:
        self.api = None

    def get_account(self) -> AccountInfo:
        acc = self.api.get_account()
        return AccountInfo(
            account_id=acc.id,
            equity=float(acc.equity),
            cash=float(acc.cash),
            buying_power=float(acc.buying_power),
            currency="USD",
            positions=self.get_positions(),
        )

    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            pos = self.api.get_position(symbol)
            return {"quantity": float(pos.qty), "avg_price": float(pos.avg_entry_price)}
        except Exception:
            return None

    def get_positions(self) -> Dict[str, dict]:
        return {
            p.symbol: {"quantity": float(p.qty), "avg_price": float(p.avg_entry_price)}
            for p in self.api.list_positions()
        }

    def place_order(self, order: Order) -> Order:
        side = "buy" if order.side == OrderSide.BUY else "sell"
        order_type = order.order_type.value

        kwargs = dict(
            symbol=order.symbol,
            qty=round(order.quantity, 6),
            side=side,
            type=order_type,
            time_in_force="gtc",
        )
        if order_type == "limit":
            kwargs["limit_price"] = str(order.limit_price)
        if order_type == "stop":
            kwargs["stop_price"] = str(order.stop_price)

        try:
            resp = self.api.submit_order(**kwargs)
            order.order_id = resp.id
            status_map = {
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIAL,
                "cancelled": OrderStatus.CANCELLED,
            }
            order.status = status_map.get(resp.status, OrderStatus.PENDING)
            if resp.filled_avg_price:
                order.avg_fill_price = float(resp.filled_avg_price)
            log.info(f"[Alpaca] {side.upper()} {order.symbol} qty={order.quantity} status={resp.status}")
        except Exception as exc:
            log.error(f"[Alpaca] Order error: {exc}")
            order.status = OrderStatus.REJECTED
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        try:
            o = self.api.get_order(order_id)
            order = Order(
                symbol=o.symbol,
                side=OrderSide.BUY if o.side == "buy" else OrderSide.SELL,
                quantity=float(o.qty),
            )
            order.order_id = order_id
            order.status = OrderStatus.FILLED if o.status == "filled" else OrderStatus.PENDING
            order.avg_fill_price = float(o.filled_avg_price or 0)
            return order
        except Exception:
            return None

    def get_latest_price(self, symbol: str) -> float:
        try:
            bars = self.api.get_bars(symbol, "1Min", limit=1).df
            return float(bars["close"].iloc[-1]) if not bars.empty else 0.0
        except Exception:
            return 0.0

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        return {sym: self.get_latest_price(sym) for sym in symbols}

    def is_market_open(self) -> bool:
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception:
            return False
