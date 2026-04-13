"""Tests for data quality checker."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa

from market_data.serving.quality import DataQualityChecker, QualityReport
from market_data.storage.analytics_lake import OHLCV_SCHEMA, AnalyticsLake

if TYPE_CHECKING:
    from market_data.storage.symbol_master import SymbolMaster


class TestDataQualityChecker:
    def test_check_negative_prices(
        self, analytics_lake: AnalyticsLake, symbol_master: SymbolMaster
    ) -> None:
        checker = DataQualityChecker(
            analytics_lake=analytics_lake,
            symbol_master=symbol_master,
        )
        # Write data with negative prices
        now = datetime.now(tz=UTC)
        sid = symbol_master.get_symbol_id("AAPL")
        table = pa.Table.from_pylist(
            [
                {
                    "timestamp_utc": 1704067200000000000,
                    "symbol_id": sid,
                    "open": -10.0,
                    "high": 155.0,
                    "low": 149.0,
                    "close": 153.0,
                    "volume": 1000000,
                    "vwap": 152.0,
                    "trade_count": 5000,
                    "ingestion_time": now,
                }
            ],
            schema=OHLCV_SCHEMA,
        )
        analytics_lake.write_table(
            table=table,
            asset_class="equity",
            schema_name="ohlcv-1d",
            symbol_id=sid,
            year=2024,
            month=1,
        )

        result = checker.check_negative_prices("2024-01-01", ["AAPL"])
        assert not result.passed
        assert result.details["negative_count"] == 1

    def test_check_completeness_pass(
        self, analytics_lake: AnalyticsLake, symbol_master: SymbolMaster
    ) -> None:
        checker = DataQualityChecker(
            analytics_lake=analytics_lake,
            symbol_master=symbol_master,
            completeness_threshold=0.5,
        )

        # Write data for AAPL
        now = datetime.now(tz=UTC)
        sid = symbol_master.get_symbol_id("AAPL")
        table = pa.Table.from_pylist(
            [
                {
                    "timestamp_utc": 1704067200000000000,
                    "symbol_id": sid,
                    "open": 150.0,
                    "high": 155.0,
                    "low": 149.0,
                    "close": 153.0,
                    "volume": 1000000,
                    "vwap": 152.0,
                    "trade_count": 5000,
                    "ingestion_time": now,
                }
            ],
            schema=OHLCV_SCHEMA,
        )
        analytics_lake.write_table(
            table=table,
            asset_class="equity",
            schema_name="ohlcv-1d",
            symbol_id=sid,
            year=2024,
            month=1,
        )

        result = checker.check_completeness("2024-01-01", ["AAPL"])
        assert result.passed

    def test_run_daily_checks_no_symbols(
        self, analytics_lake: AnalyticsLake, symbol_master: SymbolMaster
    ) -> None:
        checker = DataQualityChecker(
            analytics_lake=analytics_lake,
            symbol_master=symbol_master,
        )
        report = checker.run_daily_checks("2024-01-01", symbols=[])
        assert report.total_checks == 0

    def test_generate_dashboard_data(self) -> None:
        from market_data.serving.quality import DataQualityChecker, QualityCheckResult

        reports = [
            QualityReport(
                date="2024-01-01",
                total_checks=3,
                passed_checks=2,
                failed_checks=1,
                results=[
                    QualityCheckResult(check_name="completeness", passed=True),
                    QualityCheckResult(check_name="negative_prices", passed=True),
                    QualityCheckResult(check_name="outliers", passed=False),
                ],
            ),
        ]

        # Use a dummy instance just for the method
        class FakeLake:
            pass

        class FakeSM:
            pass

        checker = DataQualityChecker.__new__(DataQualityChecker)
        dashboard = checker.generate_dashboard_data(reports)
        assert dashboard["total_reports"] == 1
        assert len(dashboard["dates"]) == 1
        assert "check_summary" in dashboard
