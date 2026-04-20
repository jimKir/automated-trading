"""
Alpaca Broker Adapter
======================
pip install alpaca-py
Set env vars: ALPACA_API_KEY, ALPACA_API_SECRET

SDK: uses alpaca-py (alpaca.trading / alpaca.data) — NOT the deprecated
alpaca_trade_api package.
"""

from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime, timedelta

from execution.broker_base import AccountInfo, BrokerBase, Order, OrderSide, OrderStatus
from utils.logger import get_logger

log = get_logger("AlpacaBroker")


class AlpacaBroker(BrokerBase):
    def __init__(self, config: dict):
        alpaca_cfg = config.get("brokers", {}).get("alpaca", {})
        self.api_key = alpaca_cfg.get("api_key") or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = alpaca_cfg.get("api_secret") or os.environ.get("ALPACA_API_SECRET", "")
        self.paper = alpaca_cfg.get("paper", True)
        self.trading_client = None
        self.data_client = None

    def connect(self) -> bool:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient

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
        self.data_client = None

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

    def get_position(self, symbol: str) -> dict | None:
        try:
            pos = self.trading_client.get_open_position(symbol)
            return {"quantity": float(pos.qty), "avg_price": float(pos.avg_entry_price)}
        except Exception:
            return None

    def get_positions(self) -> dict[str, dict]:
        try:
            positions = self.trading_client.get_all_positions()
            return {
                p.symbol: {"quantity": float(p.qty), "avg_price": float(p.avg_entry_price)}
                for p in positions
            }
        except Exception:
            return {}

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Return open/pending orders, optionally filtered by symbol.

        Each dict has keys: order_id, symbol, side ("buy"/"sell"), qty, status.
        """
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            params = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.trading_client.get_orders(params)
            result = []
            for o in orders:
                if symbol and o.symbol != symbol:
                    continue
                result.append(
                    {
                        "order_id": str(o.id),
                        "symbol": o.symbol,
                        "side": str(o.side).lower(),
                        "qty": float(o.qty) if o.qty else 0,
                        "status": str(o.status),
                    }
                )
            return result
        except Exception as exc:
            log.error(f"[Alpaca] Failed to fetch open orders: {exc}")
            return []

    def cancel_all_open_orders(self) -> int:
        """Cancel ALL open/pending orders. Returns count of cancelled orders.

        Used on startup to clear stale orders from crashed instances.
        """
        if self.trading_client is None:
            log.warning("[Alpaca] cancel_all_open_orders skipped — client not connected yet")
            return 0
        try:
            statuses = self.trading_client.cancel_orders()
            cancelled = len(statuses) if statuses else 0
            if cancelled:
                log.warning(f"[Alpaca] Cancelled {cancelled} stale open order(s) on startup")
            else:
                log.info("[Alpaca] No open orders to cancel on startup")
            return cancelled
        except Exception as exc:
            log.error(f"[Alpaca] Failed to cancel all open orders: {exc}")
            return 0

    def cancel_conflicting_orders(self, symbol: str, new_side: OrderSide) -> bool:
        """Cancel open orders on the opposite side for *symbol* to avoid wash trades.

        Returns True if same-side orders already exist (caller should skip).
        """
        open_orders = self.get_open_orders(symbol)
        if not open_orders:
            return False

        new_side_str = new_side.value.lower()
        has_same_side = False

        for oo in open_orders:
            if oo["side"] == new_side_str:
                has_same_side = True
                log.info(
                    f"[Alpaca] Duplicate {new_side_str.upper()} already pending for "
                    f"{symbol} (order {oo['order_id']}) — will skip new order"
                )
            else:
                # Opposite side — cancel to prevent wash trade
                log.warning(
                    f"[Alpaca] Cancelling opposite-side {oo['side'].upper()} order "
                    f"{oo['order_id']} for {symbol} to avoid wash trade"
                )
                self.cancel_order(oo["order_id"])
                # Brief pause so Alpaca processes the cancellation
                time.sleep(0.3)

        return has_same_side

    def place_order(self, order: Order) -> Order:
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopOrderRequest

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

        # ── Fractional short-sell guard ──────────────────────────────────
        # Alpaca does not allow fractional quantities on short sells.
        # If this is a SELL and we have no position (i.e. a short sell),
        # round qty down to a whole number.
        if order.side == OrderSide.SELL:
            pos = self.get_position(order.symbol)
            if pos is None or pos.get("quantity", 0) <= 0:
                # No existing long position → this is a short sell
                rounded_qty = math.floor(order.quantity)
                if rounded_qty != int(order.quantity):
                    log.warning(
                        f"[Alpaca] Short sell fractional guard: {order.symbol} "
                        f"qty {order.quantity:.4f} → {rounded_qty} (floor)"
                    )
                if rounded_qty <= 0:
                    order.status = OrderStatus.REJECTED
                    log.warning(f"[Alpaca] {order.symbol} short sell qty rounds to 0 — skipping")
                    return order
                order.quantity = float(rounded_qty)

        try:
            # Alpaca requires TimeInForce.DAY for fractional share orders.
            # Use DAY for fractional quantities, GTC for whole shares.
            qty_rounded = round(order.quantity, 6)
            is_fractional = (qty_rounded % 1) != 0
            tif = TimeInForce.DAY if is_fractional else TimeInForce.GTC

            if order_type == "market":
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=qty_rounded,
                    side=side,
                    time_in_force=tif,
                )
            elif order_type == "limit":
                req = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=qty_rounded,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                )
            elif order_type == "stop":
                req = StopOrderRequest(
                    symbol=order.symbol,
                    qty=qty_rounded,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=qty_rounded,
                    side=side,
                    time_in_force=tif,
                )

            resp = self.trading_client.submit_order(req)
            order.order_id = str(resp.id)
            status_map = {
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIAL,
                "cancelled": OrderStatus.CANCELLED,
            }
            order.status = status_map.get(str(resp.status), OrderStatus.PENDING)
            if resp.filled_avg_price:
                order.avg_fill_price = float(resp.filled_avg_price)
            log.info(
                f"[Alpaca] {side.value.upper()} {order.symbol} qty={order.quantity} status={resp.status}"
            )
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

    def get_order_status(self, order_id: str) -> Order | None:
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

            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            resp = self.data_client.get_stock_latest_quote(req)
            if resp and symbol in resp:
                quote = resp[symbol]
                # Mid-price
                return float((quote.ask_price + quote.bid_price) / 2)
        except Exception:
            pass
        return 0.0

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest

            req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            resp = self.data_client.get_stock_latest_quote(req)
            if resp:
                return {sym: float((q.ask_price + q.bid_price) / 2) for sym, q in resp.items()}
        except Exception:
            pass
        return dict.fromkeys(symbols, 0.0)

    def get_last_filled_order_time(self) -> datetime | None:
        """Return the timestamp of the most recent filled order, or None.

        Used by LiveEngine to seed _last_rebalance so the adaptive cadence
        survives ECS container restarts without external storage.
        """
        if self.trading_client is None:
            log.warning("[Alpaca] get_last_filled_order_time skipped — client not connected yet")
            return None
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            params = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=1,
            )
            orders = self.trading_client.get_orders(params)
            if orders:
                filled_at = orders[0].filled_at
                if filled_at is not None:
                    if isinstance(filled_at, str):
                        return datetime.fromisoformat(filled_at)
                    return filled_at
        except Exception as exc:
            log.warning(f"[Alpaca] Could not fetch last filled order time: {exc}")
        return None

    def get_recent_fills(self, lookback_minutes: int = 10) -> list[dict]:
        """Return orders filled in the last *lookback_minutes*.

        Used to detect whether a previous instance already traded this cycle,
        preventing duplicate round-trips during overlapping start/stop windows.
        """
        if self.trading_client is None:
            return []
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            cutoff = datetime.now(UTC) - timedelta(minutes=lookback_minutes)
            params = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=50,
                after=cutoff,
            )
            orders = self.trading_client.get_orders(params)
            fills = []
            for o in orders:
                filled_at = o.filled_at
                if filled_at is None:
                    continue
                if isinstance(filled_at, str):
                    filled_at = datetime.fromisoformat(filled_at)
                if filled_at >= cutoff:
                    fills.append(
                        {
                            "order_id": str(o.id),
                            "symbol": o.symbol,
                            "side": str(o.side).lower(),
                            "qty": float(o.qty) if o.qty else 0,
                            "filled_at": filled_at.isoformat(),
                        }
                    )
            return fills
        except Exception as exc:
            log.warning(f"[Alpaca] Failed to fetch recent fills: {exc}")
            return []

    def is_market_open(self) -> bool:
        try:
            clock = self.trading_client.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False
