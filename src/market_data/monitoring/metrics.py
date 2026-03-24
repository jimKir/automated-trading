"""Prometheus metrics for ingestion, storage, and serving layers."""

from __future__ import annotations

from typing import Any

import structlog

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Prometheus metrics collector for the market data platform.

    Exposes counters, gauges, histograms, and summaries for
    ingestion, storage, feature computation, and data quality.

    Args:
        namespace: Metric name prefix.
        registry: Optional Prometheus registry. Uses default if None.
    """

    def __init__(
        self,
        namespace: str = "market_data",
        registry: Any = None,
    ) -> None:
        self.namespace = namespace

        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client_not_available")
            self._enabled = False
            return

        self._enabled = True
        self._registry = registry or CollectorRegistry()
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        ns = self.namespace
        reg = self._registry

        # Ingestion metrics
        self.records_ingested = Counter(
            f"{ns}_records_ingested_total",
            "Total records ingested",
            ["vendor", "schema", "symbol"],
            registry=reg,
        )
        self.ingestion_errors = Counter(
            f"{ns}_ingestion_errors_total",
            "Total ingestion errors",
            ["vendor", "error_type"],
            registry=reg,
        )
        self.ingestion_duration = Histogram(
            f"{ns}_ingestion_duration_seconds",
            "Time spent on ingestion operations",
            ["vendor", "operation"],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 300, 600],
            registry=reg,
        )
        self.ingestion_bytes = Counter(
            f"{ns}_ingestion_bytes_total",
            "Total bytes ingested",
            ["vendor"],
            registry=reg,
        )

        # Storage metrics
        self.storage_writes = Counter(
            f"{ns}_storage_writes_total",
            "Total storage write operations",
            ["layer", "format"],
            registry=reg,
        )
        self.storage_reads = Counter(
            f"{ns}_storage_reads_total",
            "Total storage read operations",
            ["layer"],
            registry=reg,
        )
        self.storage_bytes_written = Counter(
            f"{ns}_storage_bytes_written_total",
            "Total bytes written to storage",
            ["layer"],
            registry=reg,
        )
        self.storage_size_bytes = Gauge(
            f"{ns}_storage_size_bytes",
            "Current storage size",
            ["layer"],
            registry=reg,
        )

        # Feature computation metrics
        self.features_computed = Counter(
            f"{ns}_features_computed_total",
            "Total feature computations",
            ["version"],
            registry=reg,
        )
        self.feature_computation_duration = Histogram(
            f"{ns}_feature_computation_seconds",
            "Feature computation time",
            ["feature_set"],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
            registry=reg,
        )

        # Serving metrics
        self.api_requests = Counter(
            f"{ns}_api_requests_total",
            "Total API requests",
            ["endpoint", "status"],
            registry=reg,
        )
        self.api_latency = Histogram(
            f"{ns}_api_latency_seconds",
            "API response latency",
            ["endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
            registry=reg,
        )
        self.cache_hits = Counter(
            f"{ns}_cache_hits_total",
            "Cache hit count",
            registry=reg,
        )
        self.cache_misses = Counter(
            f"{ns}_cache_misses_total",
            "Cache miss count",
            registry=reg,
        )

        # Data quality metrics
        self.quality_checks_run = Counter(
            f"{ns}_quality_checks_total",
            "Total quality checks executed",
            ["check_name", "result"],
            registry=reg,
        )
        self.data_completeness = Gauge(
            f"{ns}_data_completeness_ratio",
            "Data completeness ratio",
            ["date"],
            registry=reg,
        )

        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            f"{ns}_rate_limit_hits_total",
            "Times rate limiter was hit",
            ["vendor"],
            registry=reg,
        )
        self.vendor_cost = Gauge(
            f"{ns}_vendor_cost_dollars",
            "Current vendor cost in dollars",
            ["vendor", "month"],
            registry=reg,
        )

        # System metrics
        self.active_tasks = Gauge(
            f"{ns}_active_tasks",
            "Currently active tasks",
            ["task_type"],
            registry=reg,
        )

    def record_ingestion(
        self,
        vendor: str,
        schema: str,
        symbol: str,
        records: int,
        bytes_count: int = 0,
        duration: float = 0.0,
    ) -> None:
        """Record an ingestion event.

        Args:
            vendor: Vendor name.
            schema: Data schema.
            symbol: Symbol ticker.
            records: Number of records ingested.
            bytes_count: Bytes ingested.
            duration: Duration in seconds.
        """
        if not self._enabled:
            return
        self.records_ingested.labels(vendor=vendor, schema=schema, symbol=symbol).inc(records)
        if bytes_count:
            self.ingestion_bytes.labels(vendor=vendor).inc(bytes_count)
        if duration:
            self.ingestion_duration.labels(vendor=vendor, operation="fetch").observe(duration)

    def record_ingestion_error(self, vendor: str, error_type: str) -> None:
        """Record an ingestion error.

        Args:
            vendor: Vendor name.
            error_type: Error classification.
        """
        if not self._enabled:
            return
        self.ingestion_errors.labels(vendor=vendor, error_type=error_type).inc()

    def record_storage_write(
        self, layer: str, fmt: str, bytes_count: int
    ) -> None:
        """Record a storage write.

        Args:
            layer: Storage layer (raw/analytics).
            fmt: File format (parquet/dbn/json).
            bytes_count: Bytes written.
        """
        if not self._enabled:
            return
        self.storage_writes.labels(layer=layer, format=fmt).inc()
        self.storage_bytes_written.labels(layer=layer).inc(bytes_count)

    def record_quality_check(self, check_name: str, passed: bool) -> None:
        """Record a quality check result.

        Args:
            check_name: Check name.
            passed: Whether the check passed.
        """
        if not self._enabled:
            return
        result = "passed" if passed else "failed"
        self.quality_checks_run.labels(check_name=check_name, result=result).inc()

    def get_metrics(self) -> bytes:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-format metrics bytes.
        """
        if not self._enabled:
            return b""
        return generate_latest(self._registry)

    @property
    def enabled(self) -> bool:
        """Whether Prometheus metrics are available."""
        return self._enabled
