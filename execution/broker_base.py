"""
Broker-Agnostic Execution Layer
================================
Abstract base class all broker adapters must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AccountInfo:
    account_id: str
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"
    positions: dict[str, dict] = None

    def __post_init__(self):
        if self.positions is None:
            self.positions = {}


class BrokerBase(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection. Returns True if successful."""
        ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_account(self) -> AccountInfo: ...

    @abstractmethod
    def get_position(self, symbol: str) -> dict | None: ...

    @abstractmethod
    def get_positions(self) -> dict[str, dict]: ...

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Place order and return filled/updated Order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order: ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float: ...

    @abstractmethod
    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]: ...

    def is_market_open(self) -> bool:
        """Override per broker. Default: assume open."""
        return True
