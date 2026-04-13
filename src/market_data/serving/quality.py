"""Data quality checks: completeness, outliers, cross-validation, and dashboards."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from market_data.storage.analytics_lake import AnalyticsLake
    from market_data.storage.symbol_master import SymbolMaster

logger = structlog.get_logger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""

    check_name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


@dataclass
class QualityReport:
    """Aggregated quality report for a day."""

    date: str
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    results: list[QualityCheckResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of checks that passed."""
        return self.passed_checks / self.total_checks if self.total_checks > 0 else 0.0


class DataQualityChecker:
    """Daily data quality checker for the analytics lake.

    Runs completeness checks, outlier detection, cross-vendor validation,
    and generates dashboard data. Alerts if S&P 500 completeness < 99.5%.

    Args:
        analytics_lake: Analytics lake to check.
        symbol_master: Symbol master for universe lookups.
        outlier_std_threshold: Standard deviation threshold for price outlier detection.
        completeness_threshold: Minimum completeness ratio for alerts.
        cross_validation_sample_size: Number of random symbols for cross-vendor checks.
    """

    def __init__(
        self,
        analytics_lake: AnalyticsLake,
        symbol_master: SymbolMaster,
        outlier_std_threshold: float = 10.0,
        completeness_threshold: float = 0.995,
        cross_validation_sample_size: int = 10,
    ) -> None:
        self.lake = analytics_lake
        self.symbol_master = symbol_master
        self.outlier_std_threshold = outlier_std_threshold
        self.completeness_threshold = completeness_threshold
        self.cross_validation_sample_size = cross_validation_sample_size

    def run_daily_checks(
        self,
        check_date: str | date,
        symbols: list[str] | None = None,
    ) -> QualityReport:
        """Run all daily quality checks.

        Args:
            check_date: Date to check (YYYY-MM-DD).
            symbols: Symbol list to check. Defaults to all active symbols.

        Returns:
            Aggregated quality report.
        """
        if isinstance(check_date, date):
            check_date = check_date.isoformat()

        report = QualityReport(date=check_date)

        if symbols is None:
            records = self.symbol_master.list_symbols(asset_class="equity", active_only=True)
            symbols = [r.ticker for r in records]

        if not symbols:
            logger.warning("no_symbols_for_quality_check", date=check_date)
            return report

        # Run individual checks
        checks = [
            self.check_missing_timestamps(check_date, symbols),
            self.check_price_outliers(check_date, symbols),
            self.check_zero_volume(check_date, symbols),
            self.check_negative_prices(check_date, symbols),
            self.check_completeness(check_date, symbols),
        ]

        for result in checks:
            report.results.append(result)
            report.total_checks += 1
            if result.passed:
                report.passed_checks += 1
            else:
                report.failed_checks += 1

        logger.info(
            "daily_quality_checks_complete",
            date=check_date,
            total=report.total_checks,
            passed=report.passed_checks,
            failed=report.failed_checks,
        )
        return report

    def check_missing_timestamps(self, check_date: str, symbols: list[str]) -> QualityCheckResult:
        """Check for missing timestamps (gaps in expected data).

        Args:
            check_date: Date to check.
            symbols: List of symbols.

        Returns:
            Quality check result.
        """
        dt = datetime.fromisoformat(check_date)
        missing_symbols: list[str] = []

        for ticker in symbols:
            symbol_id = self.symbol_master.get_symbol_id(ticker)
            if symbol_id is None:
                missing_symbols.append(ticker)
                continue

            table = self.lake.read_table(
                asset_class="equity",
                schema_name="ohlcv-1d",
                symbol_id=symbol_id,
                year=dt.year,
                month=dt.month,
            )
            if table is None or table.num_rows == 0:
                missing_symbols.append(ticker)

        passed = len(missing_symbols) == 0
        return QualityCheckResult(
            check_name="missing_timestamps",
            passed=passed,
            details={
                "missing_count": len(missing_symbols),
                "total_symbols": len(symbols),
                "missing_symbols": missing_symbols[:20],  # Limit output
            },
        )

    def check_price_outliers(self, check_date: str, symbols: list[str]) -> QualityCheckResult:
        """Detect price outliers exceeding N standard deviations.

        Args:
            check_date: Date to check.
            symbols: List of symbols.

        Returns:
            Quality check result.
        """
        dt = datetime.fromisoformat(check_date)
        outliers: list[dict[str, Any]] = []

        for ticker in symbols:
            symbol_id = self.symbol_master.get_symbol_id(ticker)
            if symbol_id is None:
                continue

            table = self.lake.read_table(
                asset_class="equity",
                schema_name="ohlcv-1d",
                symbol_id=symbol_id,
                year=dt.year,
                month=dt.month,
                columns=["close"],
            )
            if table is None or table.num_rows < 5:
                continue

            prices = table.column("close").to_pandas()
            returns = prices.pct_change().dropna()
            if len(returns) < 2:
                continue

            mean = returns.mean()
            std = returns.std()
            if std == 0:
                continue

            z_scores = ((returns - mean) / std).abs()
            max_z = z_scores.max()
            if max_z > self.outlier_std_threshold:
                outliers.append(
                    {
                        "ticker": ticker,
                        "max_z_score": round(float(max_z), 2),
                    }
                )

        passed = len(outliers) == 0
        return QualityCheckResult(
            check_name="price_outliers",
            passed=passed,
            details={
                "outlier_count": len(outliers),
                "threshold_std": self.outlier_std_threshold,
                "outliers": outliers[:20],
            },
        )

    def check_zero_volume(self, check_date: str, symbols: list[str]) -> QualityCheckResult:
        """Check for zero-volume bars which may indicate stale data.

        Args:
            check_date: Date to check.
            symbols: List of symbols.

        Returns:
            Quality check result.
        """
        dt = datetime.fromisoformat(check_date)
        zero_volume_symbols: list[str] = []

        for ticker in symbols:
            symbol_id = self.symbol_master.get_symbol_id(ticker)
            if symbol_id is None:
                continue

            table = self.lake.read_table(
                asset_class="equity",
                schema_name="ohlcv-1d",
                symbol_id=symbol_id,
                year=dt.year,
                month=dt.month,
                columns=["volume"],
            )
            if table is None or table.num_rows == 0:
                continue

            volumes = table.column("volume").to_pandas()
            if (volumes == 0).any():
                zero_volume_symbols.append(ticker)

        passed = len(zero_volume_symbols) == 0
        return QualityCheckResult(
            check_name="zero_volume",
            passed=passed,
            details={
                "zero_volume_count": len(zero_volume_symbols),
                "symbols": zero_volume_symbols[:20],
            },
        )

    def check_negative_prices(self, check_date: str, symbols: list[str]) -> QualityCheckResult:
        """Check for negative prices which are always invalid.

        Args:
            check_date: Date to check.
            symbols: List of symbols.

        Returns:
            Quality check result.
        """
        dt = datetime.fromisoformat(check_date)
        negative_price_symbols: list[str] = []

        for ticker in symbols:
            symbol_id = self.symbol_master.get_symbol_id(ticker)
            if symbol_id is None:
                continue

            table = self.lake.read_table(
                asset_class="equity",
                schema_name="ohlcv-1d",
                symbol_id=symbol_id,
                year=dt.year,
                month=dt.month,
                columns=["open", "high", "low", "close"],
            )
            if table is None or table.num_rows == 0:
                continue

            df = table.to_pandas()
            if (df[["open", "high", "low", "close"]] < 0).any().any():
                negative_price_symbols.append(ticker)

        passed = len(negative_price_symbols) == 0
        return QualityCheckResult(
            check_name="negative_prices",
            passed=passed,
            details={
                "negative_count": len(negative_price_symbols),
                "symbols": negative_price_symbols[:20],
            },
        )

    def check_completeness(self, check_date: str, symbols: list[str]) -> QualityCheckResult:
        """Check data completeness against expected symbol universe.

        Alerts if completeness falls below threshold (default 99.5%).

        Args:
            check_date: Date to check.
            symbols: Expected symbols.

        Returns:
            Quality check result.
        """
        dt = datetime.fromisoformat(check_date)
        found = 0

        for ticker in symbols:
            symbol_id = self.symbol_master.get_symbol_id(ticker)
            if symbol_id is None:
                continue

            table = self.lake.read_table(
                asset_class="equity",
                schema_name="ohlcv-1d",
                symbol_id=symbol_id,
                year=dt.year,
                month=dt.month,
            )
            if table is not None and table.num_rows > 0:
                found += 1

        completeness = found / len(symbols) if symbols else 0.0
        passed = completeness >= self.completeness_threshold

        if not passed:
            logger.warning(
                "completeness_below_threshold",
                date=check_date,
                completeness=round(completeness, 4),
                threshold=self.completeness_threshold,
            )

        return QualityCheckResult(
            check_name="completeness",
            passed=passed,
            details={
                "expected": len(symbols),
                "found": found,
                "completeness": round(completeness, 4),
                "threshold": self.completeness_threshold,
            },
        )

    def cross_validate_vendors(
        self,
        check_date: str,
        vendor_a_lake: AnalyticsLake,
        vendor_b_lake: AnalyticsLake,
        tolerance: float = 0.01,
    ) -> QualityCheckResult:
        """Cross-validate data between two vendor sources.

        Picks random symbols and compares close prices between vendors.

        Args:
            check_date: Date to compare.
            vendor_a_lake: First vendor's analytics lake.
            vendor_b_lake: Second vendor's analytics lake.
            tolerance: Maximum allowed price difference ratio.

        Returns:
            Quality check result.
        """
        dt = datetime.fromisoformat(check_date)
        records = self.symbol_master.list_symbols(asset_class="equity", active_only=True)

        sample_size = min(self.cross_validation_sample_size, len(records))
        sample = random.sample(records, sample_size) if records else []

        mismatches: list[dict[str, Any]] = []
        compared = 0

        for record in sample:
            table_a = vendor_a_lake.read_table(
                asset_class="equity",
                schema_name="ohlcv-1d",
                symbol_id=record.symbol_id,
                year=dt.year,
                month=dt.month,
                columns=["close"],
            )
            table_b = vendor_b_lake.read_table(
                asset_class="equity",
                schema_name="ohlcv-1d",
                symbol_id=record.symbol_id,
                year=dt.year,
                month=dt.month,
                columns=["close"],
            )

            if table_a is None or table_b is None:
                continue
            if table_a.num_rows == 0 or table_b.num_rows == 0:
                continue

            price_a = table_a.column("close").to_pylist()[-1]
            price_b = table_b.column("close").to_pylist()[-1]
            compared += 1

            if price_a == 0:
                continue

            diff_ratio = abs(price_a - price_b) / abs(price_a)
            if diff_ratio > tolerance:
                mismatches.append(
                    {
                        "ticker": record.ticker,
                        "price_a": round(price_a, 4),
                        "price_b": round(price_b, 4),
                        "diff_ratio": round(diff_ratio, 4),
                    }
                )

        passed = len(mismatches) == 0
        return QualityCheckResult(
            check_name="cross_vendor_validation",
            passed=passed,
            details={
                "compared": compared,
                "mismatches": len(mismatches),
                "tolerance": tolerance,
                "mismatch_details": mismatches,
            },
        )

    def generate_dashboard_data(self, reports: list[QualityReport]) -> dict[str, Any]:
        """Generate data quality dashboard summary.

        Args:
            reports: List of quality reports over time.

        Returns:
            Dashboard data dictionary.
        """
        if not reports:
            return {"dates": [], "pass_rates": [], "check_details": {}}

        dates = [r.date for r in reports]
        pass_rates = [r.pass_rate for r in reports]

        # Aggregate check-level pass rates
        check_details: dict[str, list[bool]] = {}
        for report in reports:
            for result in report.results:
                if result.check_name not in check_details:
                    check_details[result.check_name] = []
                check_details[result.check_name].append(result.passed)

        check_summary = {
            name: {
                "total": len(results),
                "passed": sum(results),
                "pass_rate": round(sum(results) / len(results), 4) if results else 0.0,
            }
            for name, results in check_details.items()
        }

        return {
            "dates": dates,
            "pass_rates": [round(r, 4) for r in pass_rates],
            "overall_pass_rate": round(sum(pass_rates) / len(pass_rates), 4) if pass_rates else 0.0,
            "check_summary": check_summary,
            "total_reports": len(reports),
        }
