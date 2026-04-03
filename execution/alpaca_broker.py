"""
Alpaca Broker Adapter
======================
pip install alpaca-py
Set env vars: ALPACA_API_KEY, ALPACA_API_SECRET

SDK: uses alpaca-py (alpaca.trading / alpaca.data) — NOT the deprecated
alpaca_trade_api package.
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
        self.api_key    = alpaca_cfg.get("api_key")    or os.environ.get("ALPACA_API_KEY",    "")
        self.api_secret = alpaca_cfg.get("api_secret") or os.environ.get("ALPACA_API_SECRET", "")
        self.paper      = alpaca_cfg.get("paper", True)
        self.trading_client = None
        self.data_client    = None

    def connect(self) -> bool:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper,
            )
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
            )
            account = self.trading_client.get_account()
            log.info(f"Alpaca connected. Account status: {account.status}")
            return True
        except Exception as exc:
            log.error(f"Alpaca connection failed: {exc}")
            return False

    def disconnect(self) -> None:
        self.trading_client = None
        self.data_client    = None

    def get_account(self) -> AccountInfo:
        acc = self.trading_client.get_account()
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
            pos = self.trading_client.get_open_position(symbol)
            return {"quantity": float(pos.qty), "avg_price": float(pos.avg_entry_price)}
        except Exception:
            return None

    def get_positions(self) -> Dict[str, dict]:
        try:
            positions = self.trading_client.get_all_positions()
            return {
                p.symbol: {"quantity": float(p.qty), "avg_price": float(p.avg_entry_price)}
                for p in positions
            }
        except Exception:
            return {}

    def place_order(self, order: Order) -> Order:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
        order_type = order.order_type.value

        # ── Pre-trade safety limits ──────────────────────────────────────
        max_shares = 10000  # hard cap — configurable via config if needed
        if order.quantity > max_shares:
            log.warning(f"[Alpaca] Clamping {order.symbol} qty {order.quantity:.1f} → {max_shares}")
            order.quantity = max_shares
        if order.quantity < 0.001:
            order.status = OrderStatus.REJECTED
            log.warning(f"[Alpaca] {order.symbol} qty too small — rejected")
            return order

        try:
            if order_type == "market":
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=round(order.quantity, 6),
                    side=side,
                    time_in_force=TimeInForce.GTC,
                )
            elif order_type == "limit":
                req = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=round(order.quantity, 6),
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=order.limit_price,
                )
            elif order_type == "stop":
                req = StopOrderRequest(
                    symbol=order.symbol,
                    qty=round(order.quantity, 6),
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    stop_price=order.stop_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=round(order.quantity, 6),
                    side=side,
                    time_in_force=TimeInForce.GTC,
                )

            resp = self.trading_client.submit_order(req)
            order.order_id = str(resp.id)
            status_map = {
                "filled":           OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIAL,
                "cancelled":        OrderStatus.CANCELLED,
            }
            order.status = status_map.get(str(resp.status), OrderStatus.PENDING)
            if resp.filled_avg_price:
                order.avg_fill_price = float(resp.filled_avg_price)
            log.info(f"[Alpaca] {side.value.upper()} {order.symbol} qty={order.quantity} status={resp.status}")
        except Exception as exc:
            log.error(f"[Alpaca] Order error: {exc}")
            order.status = OrderStatus.REJECTED
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            import uuid
            self.trading_client.cancel_order_by_id(uuid.UUID(order_id))
            return True
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        try:
            import uuid
            o = self.trading_client.get_order_by_id(uuid.UUID(order_id))
            order = Order(
                symbol=o.symbol,
                side=OrderSide.BUY if str(o.side) == "buy" else OrderSide.SELL,
                quantity=float(o.qty),
            )
            order.order_id = order_id
            order.status = OrderStatus.FILLED if str(o.status) == "filled" else OrderStatus.PENDING
            order.avg_fill_price = float(o.filled_avg_price or 0)
            return order
        except Exception:
            return None

    def get_latest_price(self, symbol: str) -> float:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req  = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            resp = self.data_client.get_stock_latest_quote(req)
            if resp and symbol in resp:
                quote = resp[symbol]
                # Mid-price
                return float((quote.ask_price + quote.bid_price) / 2)
        except Exception:
            pass
        return 0.0

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req  = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            resp = self.data_client.get_stock_latest_quote(req)
            if resp:
                return {
                    sym: float((q.ask_price + q.bid_price) / 2)
                    for sym, q in resp.items()
                }
        except Exception:
            pass
        return {sym: 0.0 for sym in symbols}

    def is_market_open(self) -> bool:
        try:
            clock = self.trading_client.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False
