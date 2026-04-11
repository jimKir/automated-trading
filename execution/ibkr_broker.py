"""
Interactive Brokers (IBKR) Broker Adapter
==========================================
Uses the `ib_insync` library for TWS / IB Gateway connectivity.

Requirements:
  pip install ib_insync
  TWS or IB Gateway must be running with API access enabled.

Paper trading:  port=7497 (TWS) or 4002 (Gateway)
Live trading:   port=7496 (TWS) or 4001 (Gateway)
"""

from __future__ import annotations

from execution.broker_base import AccountInfo, BrokerBase, Order, OrderSide, OrderStatus, OrderType
from utils.logger import get_logger

log = get_logger("IBKRBroker")


class IBKRBroker(BrokerBase):
    def __init__(self, config: dict):
        self.config = config
        ibkr_cfg = config.get("brokers", {}).get("ibkr", {})
        self.host = ibkr_cfg.get("host", "127.0.0.1")
        self.port = ibkr_cfg.get("port", 7497)
        self.client_id = ibkr_cfg.get("client_id", 1)
        self.account_id = ibkr_cfg.get("account", "")
        self.paper = ibkr_cfg.get("paper_mode", True)
        self.ib = None

    def connect(self) -> bool:
        try:
            from ib_insync import IB

            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id, readonly=False)
            log.info(f"IBKR connected: {self.host}:{self.port} (paper={self.paper})")
            return True
        except Exception as exc:
            log.error(f"IBKR connection failed: {exc}")
            return False

    def disconnect(self) -> None:
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            log.info("IBKR disconnected")

    def get_account(self) -> AccountInfo:
        vals = {v.tag: v.value for v in self.ib.accountValues(self.account_id)}
        equity = float(vals.get("NetLiquidation", 0))
        cash = float(vals.get("AvailableFunds", 0))
        buying_power = float(vals.get("BuyingPower", cash))
        currency = vals.get("Currency", "USD")
        positions = self.get_positions()
        return AccountInfo(
            account_id=self.account_id,
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            currency=currency,
            positions=positions,
        )

    def get_position(self, symbol: str) -> dict | None:
        return self.get_positions().get(symbol)

    def get_positions(self) -> dict[str, dict]:
        positions = {}
        for pos in self.ib.positions(self.account_id):
            sym = pos.contract.symbol
            positions[sym] = {
                "quantity": pos.position,
                "avg_price": pos.avgCost,
                "contract": pos.contract,
            }
        return positions

    def _make_contract(self, symbol: str):
        """Build the appropriate IBKR contract for a symbol."""
        from ib_insync import Crypto, Future, Stock

        # Crypto
        if symbol.endswith("-USD") or symbol in ("BTC-USD", "ETH-USD", "SOL-USD"):
            base = symbol.replace("-USD", "")
            return Crypto(base, "PAXOS", "USD")
        # Futures (continuous front-month)
        futures_map = {
            "ES=F": ("ES", "CME", "USD"),
            "NQ=F": ("NQ", "CME", "USD"),
            "GC=F": ("GC", "COMEX", "USD"),
            "CL=F": ("CL", "NYMEX", "USD"),
        }
        if symbol in futures_map:
            sym, exchange, currency = futures_map[symbol]
            f = Future(sym, exchange=exchange, currency=currency)
            f.lastTradeDateOrContractMonth = ""  # let IB resolve front month
            return f
        # Default: US stock / ETF
        return Stock(symbol, "SMART", "USD")

    def place_order(self, order: Order) -> Order:
        from ib_insync import LimitOrder, MarketOrder, StopOrder

        contract = self._make_contract(order.symbol)
        self.ib.qualifyContracts(contract)

        action = "BUY" if order.side == OrderSide.BUY else "SELL"

        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(action, order.quantity, order.limit_price)
        elif order.order_type == OrderType.STOP:
            ib_order = StopOrder(action, order.quantity, order.stop_price)
        else:
            ib_order = MarketOrder(action, order.quantity)

        trade = self.ib.placeOrder(contract, ib_order)
        self.ib.sleep(1)  # allow fill

        order.order_id = str(trade.order.orderId)
        status = trade.orderStatus.status
        if status in ("Filled", "Submitted"):
            order.status = OrderStatus.FILLED
            order.avg_fill_price = trade.orderStatus.avgFillPrice or 0.0
            order.filled_qty = trade.orderStatus.filled or order.quantity
        elif status == "Cancelled":
            order.status = OrderStatus.CANCELLED
        else:
            order.status = OrderStatus.PENDING

        log.info(
            f"[IBKR] {action} {order.symbol} qty={order.quantity:.4f} "
            f"status={status} fill_px={order.avg_fill_price:.4f}"
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self.ib.cancelOrder(trade.order)
                log.info(f"[IBKR] Cancelled order {order_id}")
                return True
        return False

    def get_order_status(self, order_id: str) -> Order | None:
        for trade in self.ib.trades():
            if str(trade.order.orderId) == order_id:
                o = Order(
                    symbol=trade.contract.symbol,
                    side=OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL,
                    quantity=trade.order.totalQuantity,
                )
                o.order_id = order_id
                o.status = (
                    OrderStatus.FILLED
                    if trade.orderStatus.status == "Filled"
                    else OrderStatus.PENDING
                )
                o.avg_fill_price = trade.orderStatus.avgFillPrice
                return o
        return None

    def get_latest_price(self, symbol: str) -> float:
        contract = self._make_contract(symbol)
        try:
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, "", True, False)
            self.ib.sleep(2)
            price = ticker.last or ticker.bid or ticker.ask or 0.0
            self.ib.cancelMktData(contract)
            return float(price)
        except Exception as e:
            log.warning(f"[IBKR] Price error {symbol}: {e}")
            return 0.0

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        return {sym: self.get_latest_price(sym) for sym in symbols}

    def is_market_open(self) -> bool:
        from datetime import datetime, time as dtime

        import pytz

        et = pytz.timezone("America/New_York")
        now = datetime.now(et).time()
        return dtime(9, 30) <= now <= dtime(16, 0)
