"""Health checks: liveness, readiness, and component status."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    healthy: bool
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Aggregated health status."""

    healthy: bool
    components: list[ComponentHealth] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "healthy": self.healthy,
            "timestamp": self.timestamp,
            "components": [
                {
                    "name": c.name,
                    "healthy": c.healthy,
                    "latency_ms": round(c.latency_ms, 2),
                    "details": c.details,
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """Health check manager for liveness and readiness probes.

    Registers component health check functions and runs them on demand.
    Supports Docker/Kubernetes health check endpoints.

    Args:
        service_name: Name of this service instance.
    """

    def __init__(self, service_name: str = "market-data-platform") -> None:
        self.service_name = service_name
        self._checks: dict[str, Callable[[], bool]] = {}
        self._start_time = time.monotonic()
        self._ready = False

    def register_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Register a component health check.

        Args:
            name: Component name.
            check_fn: Callable returning True if healthy.
        """
        self._checks[name] = check_fn
        logger.info("health_check_registered", component=name)

    def register_sqlite_check(self, name: str, db_path: str | Path) -> None:
        """Register a SQLite database health check.

        Args:
            name: Component name.
            db_path: Path to SQLite database.
        """

        def check() -> bool:
            try:
                with sqlite3.connect(str(db_path)) as conn:
                    conn.execute("SELECT 1")
                return True
            except Exception:
                return False

        self.register_check(name, check)

    def register_storage_check(self, name: str, check_path: str, storage_backend: Any) -> None:
        """Register a storage backend health check.

        Args:
            name: Component name.
            check_path: Path to check for existence.
            storage_backend: Storage backend instance.
        """

        def check() -> bool:
            try:
                storage_backend.list_files("")
                return True
            except Exception:
                return False

        self.register_check(name, check)

    def liveness(self) -> HealthStatus:
        """Liveness probe — is the service running?

        Returns:
            HealthStatus with basic liveness info.
        """
        uptime = time.monotonic() - self._start_time
        return HealthStatus(
            healthy=True,
            components=[
                ComponentHealth(
                    name="process",
                    healthy=True,
                    details={
                        "service": self.service_name,
                        "uptime_seconds": round(uptime, 1),
                    },
                )
            ],
        )

    def readiness(self) -> HealthStatus:
        """Readiness probe — is the service ready to handle requests?

        Runs all registered health checks and reports aggregate status.

        Returns:
            HealthStatus with all component statuses.
        """
        components: list[ComponentHealth] = []

        for name, check_fn in self._checks.items():
            start = time.monotonic()
            try:
                healthy = check_fn()
            except Exception as exc:
                healthy = False
                logger.warning("health_check_failed", component=name, error=str(exc))
            latency_ms = (time.monotonic() - start) * 1000

            components.append(
                ComponentHealth(
                    name=name,
                    healthy=healthy,
                    latency_ms=latency_ms,
                )
            )

        all_healthy = all(c.healthy for c in components)
        self._ready = all_healthy

        return HealthStatus(healthy=all_healthy, components=components)

    def mark_ready(self) -> None:
        """Manually mark the service as ready."""
        self._ready = True

    def mark_not_ready(self) -> None:
        """Manually mark the service as not ready."""
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """Whether the service is currently ready."""
        return self._ready

    @property
    def uptime_seconds(self) -> float:
        """Service uptime in seconds."""
        return time.monotonic() - self._start_time
