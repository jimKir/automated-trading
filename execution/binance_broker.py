"""
Binance Broker Adapter (Spot + Futures)
=========================================
Uses the `python-binance` library.
pip install python-binance

Set environment variables:
  BINANCE_API_KEY
  BINANCE_API_SECRET

Testnet URL: https://testnet.binance.vision  (set testnet=True)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from execution.broker_base import (
    BrokerBase, Order, OrderSide, OrderStatus, OrderType, AccountInfo
)
from utils.logger import get_logger

log = get_logger("BinanceBroker")


class BinanceBroker(BrokerBase):
    def __init__(self, config: dict):
        self.config = config
        binance_cfg = config.get("brokers", {}).get("binance", {})
        self.api_key = (
            binance_cfg.get("api_key") or os.environ.get("BINANCE_API_KEY", "")
        )
        self.api_secret = (
            binance_cfg.get("api_secret") or os.environ.get("BINANCE_API_SECRET", "")
        )
        self.testnet = binance_cfg.get("testnet", True)
        self.client = None

    def connect(self) -> bool:
        try:
            from binance.client import Client
            self.client = Client(
                self.api_key,
                self.api_secret,
                testnet=self.testnet,
            )
            # Ping
            self.client.ping()
            log.info(f"Binance connected (testnet={self.testnet})")
            return True
        except Exception as exc:
            log.error(f"Binance connection failed: {exc}")
            return False

    def disconnect(self) -> None:
        self.client = None
        log.info("Binance disconnected")

    def _to_binance_symbol(self, symbol: str) -> str:
        """Convert yfinance-style symbol (BTC-USD) to Binance style (BTCUSDT)."""
        return symbol.replace("-USD", "USDT").replace("-", "")

    def get_account(self) -> AccountInfo:
        info = self.client.get_account()
        balances = {
            b["asset"]: float(b["free"]) + float(b["locked"])
            for b in info["balances"]
            if float(b["free"]) + float(b["locked"]) > 0
        }
        usdt_balance = balances.get("USDT", 0.0)
        # Approximate equity (USD value of all holdings)
        equity = usdt_balance
        for asset, qty in balances.items():
            if asset == "USDT":
                continue
            try:
                p = self.get_latest_price(f"{asset}-USD")
                equity += qty * p
            except Exception:
                pass

        return AccountInfo(
            account_id=info.get("accountType", "SPOT"),
            equity=equity,
            cash=usdt_balance,
            buying_power=usdt_balance,
            currency="USDT",
            positions={k: {"quantity": v} for k, v in balances.items()},
        )

    def get_position(self, symbol: str) -> Optional[dict]:
        base = symbol.replace("-USD", "").replace("USDT", "")
        for b in self.client.get_account()["balances"]:
            if b["asset"] == base:
                qty = float(b["free"]) + float(b["locked"])
                if qty > 0:
                    return {"quantity": qty}
        return None

    def get_positions(self) -> Dict[str, dict]:
        result = {}
        for b in self.client.get_account()["balances"]:
            qty = float(b["free"]) + float(b["locked"])
            if qty > 0:
                result[b["asset"]] = {"quantity": qty}
        return result

    def place_order(self, order: Order) -> Order:
        from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT
        sym = self._to_binance_symbol(order.symbol)
        side = SIDE_BUY if order.side == OrderSide.BUY else SIDE_SELL

        try:
            if order.order_type == OrderType.MARKET:
                resp = self.client.create_order(
                    symbol=sym,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=round(order.quantity, 6),
                )
            elif order.order_type == OrderType.LIMIT:
                resp = self.client.create_order(
                    symbol=sym,
                    side=side,
                    type=ORDER_TYPE_LIMIT,
                    quantity=round(order.quantity, 6),
                    price=str(order.limit_price),
                    timeInForce="GTC",
                )
            else:
                resp = self.client.create_order(
                    symbol=sym,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=round(order.quantity, 6),
                )

            order.order_id = str(resp["orderId"])
            order.status = OrderStatus.FILLED if resp["status"] == "FILLED" else OrderStatus.PENDING
            fills = resp.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                order.avg_fill_price = total_cost / total_qty if total_qty > 0 else 0.0
                order.commission = sum(float(f["commission"]) for f in fills)
                order.filled_qty = total_qty

            log.info(f"[Binance] {side} {sym} qty={order.quantity:.6f} status={resp['status']}")

        except Exception as exc:
            log.error(f"[Binance] Order failed for {sym}: {exc}")
            order.status = OrderStatus.REJECTED
            order.metadata["error"] = str(exc)

        return order

    def cancel_order(self, order_id: str) -> bool:
        # Binance requires symbol to cancel — store in order metadata
        log.warning("cancel_order: symbol required for Binance cancellations")
        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        log.warning("get_order_status: symbol required for Binance query")
        return None

    def get_latest_price(self, symbol: str) -> float:
        sym = self._to_binance_symbol(symbol)
        try:
            ticker = self.client.get_symbol_ticker(symbol=sym)
            return float(ticker["price"])
        except Exception as e:
            log.warning(f"[Binance] Price error {sym}: {e}")
            return 0.0

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        prices = {}
        for sym in symbols:
            prices[sym] = self.get_latest_price(sym)
        return prices

    def is_market_open(self) -> bool:
        """Crypto trades 24/7."""
        return True
