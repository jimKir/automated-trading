"""Token bucket rate limiter with cost tracking and priority queue."""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Priority(IntEnum):
    """Request priority levels."""

    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class PrioritizedRequest:
    """A request with priority for the priority queue."""

    priority: Priority
    timestamp: float = field(compare=True)
    symbol: str = field(compare=False)
    callback: Any = field(compare=False, repr=False)


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API calls.

    Implements a token bucket algorithm with configurable rate and burst.
    Thread-safe for concurrent access.

    Args:
        rate_per_minute: Maximum sustained requests per minute.
        burst_size: Maximum burst size (bucket capacity).
        vendor_name: Name of the vendor (for logging).
    """

    def __init__(
        self,
        rate_per_minute: int,
        burst_size: int | None = None,
        vendor_name: str = "unknown",
    ) -> None:
        self.rate_per_second = rate_per_minute / 60.0
        self.burst_size = burst_size or rate_per_minute
        self.tokens = float(self.burst_size)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()
        self.vendor_name = vendor_name
        self.logger = logger.bind(vendor=vendor_name)

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire.
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if tokens were acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            if deadline is not None and time.monotonic() >= deadline:
                return False

            wait_time = (tokens - self.tokens) / self.rate_per_second
            wait_time = min(wait_time, 0.1)
            time.sleep(wait_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate_per_second
        self.tokens = min(self.burst_size, self.tokens + new_tokens)
        self.last_refill = now


class CostTracker:
    """Track API costs per vendor per month.

    Monitors spending against budget thresholds and raises alerts.

    Args:
        vendor_name: Name of the vendor.
        monthly_budget: Monthly budget in dollars.
        alert_threshold: Fraction of budget at which to alert (0.0-1.0).
    """

    def __init__(
        self,
        vendor_name: str,
        monthly_budget: float = 1000.0,
        alert_threshold: float = 0.8,
    ) -> None:
        self.vendor_name = vendor_name
        self.monthly_budget = monthly_budget
        self.alert_threshold = alert_threshold
        self._costs: dict[str, float] = {}  # month_key -> total cost
        self._lock = threading.Lock()
        self.logger = logger.bind(vendor=vendor_name)

    def record_cost(self, amount: float, description: str = "") -> None:
        """Record an API cost.

        Args:
            amount: Cost in dollars.
            description: Description of the charge.
        """
        month_key = date.today().strftime("%Y-%m")
        with self._lock:
            self._costs[month_key] = self._costs.get(month_key, 0.0) + amount

        current = self.get_current_month_cost()
        self.logger.info(
            "cost_recorded",
            amount=amount,
            description=description,
            month_total=round(current, 2),
        )

        if current >= self.monthly_budget * self.alert_threshold:
            self.logger.warning(
                "budget_threshold_exceeded",
                current_cost=round(current, 2),
                budget=self.monthly_budget,
                threshold=self.alert_threshold,
            )

    def get_current_month_cost(self) -> float:
        """Get total cost for the current month."""
        month_key = date.today().strftime("%Y-%m")
        with self._lock:
            return self._costs.get(month_key, 0.0)

    def is_over_budget(self) -> bool:
        """Check if current month spending exceeds budget."""
        return self.get_current_month_cost() >= self.monthly_budget

    def get_all_costs(self) -> dict[str, float]:
        """Get cost breakdown by month."""
        with self._lock:
            return dict(self._costs)


class PriorityRequestQueue:
    """Priority queue for ingestion requests.

    Urgent symbols (e.g., actively traded) get higher priority.

    Args:
        rate_limiter: Rate limiter to use for requests.
    """

    def __init__(self, rate_limiter: TokenBucketRateLimiter) -> None:
        self.rate_limiter = rate_limiter
        self._queue: list[PrioritizedRequest] = []
        self._lock = threading.Lock()

    def submit(
        self,
        symbol: str,
        callback: Any,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """Submit a request to the priority queue.

        Args:
            symbol: Symbol to fetch.
            callback: Callable to execute when rate limit allows.
            priority: Request priority.
        """
        request = PrioritizedRequest(
            priority=priority,
            timestamp=time.monotonic(),
            symbol=symbol,
            callback=callback,
        )
        with self._lock:
            heapq.heappush(self._queue, request)

    def process_next(self, timeout: float = 30.0) -> bool:
        """Process the next highest-priority request.

        Args:
            timeout: Maximum time to wait for rate limit.

        Returns:
            True if a request was processed, False if queue empty.
        """
        with self._lock:
            if not self._queue:
                return False
            request = heapq.heappop(self._queue)

        if self.rate_limiter.acquire(timeout=timeout):
            request.callback()
            return True

        with self._lock:
            heapq.heappush(self._queue, request)
        return False

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        with self._lock:
            return len(self._queue)
