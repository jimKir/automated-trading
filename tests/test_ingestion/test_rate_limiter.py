"""Tests for rate limiter and cost tracking."""

from __future__ import annotations

import time

from market_data.ingestion.rate_limiter import (
    CostTracker,
    Priority,
    PriorityRequestQueue,
    TokenBucketRateLimiter,
)


class TestTokenBucketRateLimiter:
    def test_initial_tokens(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_minute=600, burst_size=10)
        assert limiter.acquire()

    def test_burst_limit(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_minute=60, burst_size=3)
        assert limiter.acquire()
        assert limiter.acquire()
        assert limiter.acquire()
        # 4th should fail with no wait
        assert not limiter.acquire(timeout=0)

    def test_token_refill(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_minute=6000, burst_size=1)
        assert limiter.acquire()
        assert not limiter.acquire(timeout=0)
        time.sleep(0.02)  # Wait for refill at 100/s
        assert limiter.acquire()


class TestCostTracker:
    def test_record_cost(self) -> None:
        tracker = CostTracker(vendor_name="databento", monthly_budget=1000.0)
        tracker.record_cost(50.0)
        assert tracker.get_current_month_cost() == 50.0

    def test_budget_check(self) -> None:
        tracker = CostTracker(vendor_name="databento", monthly_budget=100.0)
        tracker.record_cost(90.0)
        assert not tracker.is_over_budget()
        tracker.record_cost(20.0)
        assert tracker.is_over_budget()

    def test_get_all_costs(self) -> None:
        tracker = CostTracker(vendor_name="databento", monthly_budget=1000.0)
        tracker.record_cost(50.0)
        costs = tracker.get_all_costs()
        assert len(costs) == 1
        assert list(costs.values())[0] == 50.0


class TestPriorityRequestQueue:
    def test_submit_and_process(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_minute=600, burst_size=10)
        queue = PriorityRequestQueue(rate_limiter=limiter)
        results: list[str] = []

        queue.submit("AAPL", lambda: results.append("AAPL"), Priority.NORMAL)
        queue.submit("MSFT", lambda: results.append("MSFT"), Priority.URGENT)

        assert queue.pending_count == 2
        queue.process_next()
        queue.process_next()
        # Urgent should be processed first
        assert results[0] == "MSFT"
        assert results[1] == "AAPL"

    def test_empty_queue(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_minute=600, burst_size=10)
        queue = PriorityRequestQueue(rate_limiter=limiter)
        assert queue.pending_count == 0
        assert not queue.process_next()

    def test_pending_count(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_minute=600, burst_size=10)
        queue = PriorityRequestQueue(rate_limiter=limiter)
        queue.submit("A", lambda: None, Priority.NORMAL)
        queue.submit("B", lambda: None, Priority.HIGH)
        assert queue.pending_count == 2
