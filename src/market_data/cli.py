"""CLI interface for the Market Data Platform."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta

import click
import structlog

logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(package_name="market-data-platform")
@click.option("--config", "-c", default="config/config.yaml", help="Config file path.")
@click.option("--env", "-e", default=None, help="Environment config overlay path.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, config: str, env: str | None, verbose: bool) -> None:
    """Market Data Platform — Ingest, store, and serve market data."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["env_path"] = env
    ctx.obj["verbose"] = verbose

    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0),
        )


@cli.command()
@click.option("--symbols", "-s", multiple=True, help="Symbols to backfill.")
@click.option("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD).")
@click.option("--end-date", type=str, default=None, help="End date (YYYY-MM-DD).")
@click.option("--vendor", type=click.Choice(["databento", "alpaca", "all"]), default="all")
@click.option("--schema", type=str, default="ohlcv-1d", help="Data schema.")
@click.pass_context
def backfill(
    ctx: click.Context,
    symbols: tuple[str, ...],
    start_date: str | None,
    end_date: str | None,
    vendor: str,
    schema: str,
) -> None:
    """Run historical data backfill."""
    from market_data.config import get_settings

    settings = get_settings(ctx.obj["config_path"], ctx.obj.get("env_path"))
    symbol_list = list(symbols) or settings.ingestion.backfill.symbols or ["AAPL"]

    if start_date is None:
        start_date = (date.today() - timedelta(days=730)).isoformat()
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    vendors = ["databento", "alpaca"] if vendor == "all" else [vendor]

    click.echo(f"Starting backfill: {len(symbol_list)} symbols, {start_date} to {end_date}")
    click.echo(f"Vendors: {', '.join(vendors)}, Schema: {schema}")

    try:
        from flows.backfill_flow import backfill_flow

        results = backfill_flow(
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            vendors=vendors,
            schemas=[schema],
        )
        for v, summary in results.items():
            click.echo(
                f"  {v}: {summary.completed_chunks}/{summary.total_chunks} chunks "
                f"({summary.success_rate:.0%} success)"
            )
    except Exception as exc:
        click.echo(f"Backfill failed: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--symbols", "-s", multiple=True, help="Symbols to update.")
@click.option("--vendor", type=click.Choice(["databento", "alpaca", "all"]), default="all")
@click.option("--schema", type=str, default="ohlcv-1d", help="Data schema.")
@click.option("--skip-market-check", is_flag=True, help="Skip market calendar check.")
@click.pass_context
def update(
    ctx: click.Context,
    symbols: tuple[str, ...],
    vendor: str,
    schema: str,
    skip_market_check: bool,
) -> None:
    """Run incremental data update."""
    from market_data.config import get_settings

    settings = get_settings(ctx.obj["config_path"], ctx.obj.get("env_path"))
    symbol_list = list(symbols) or settings.ingestion.incremental.symbols or ["AAPL"]
    vendors = ["databento", "alpaca"] if vendor == "all" else [vendor]

    click.echo(f"Starting incremental update: {len(symbol_list)} symbols")

    try:
        from flows.incremental_flow import incremental_flow

        results = incremental_flow(
            symbols=symbol_list,
            vendors=vendors,
            schemas=[schema],
            skip_market_check=skip_market_check,
        )
        for v, summary in results.items():
            click.echo(
                f"  {v}: {summary.symbols_updated} symbols, {summary.new_records} new records"
            )
    except Exception as exc:
        click.echo(f"Update failed: {exc}", err=True)
        sys.exit(1)


@cli.command("generate-features")
@click.option("--symbols", "-s", multiple=True, help="Symbols to compute features for.")
@click.option("--year", type=int, default=None, help="Year to process.")
@click.option("--month", type=int, default=None, help="Month to process.")
@click.option("--version", type=str, default="1.0.0", help="Feature version.")
@click.option("--name", type=str, default="daily_features", help="Feature set name.")
@click.pass_context
def generate_features(
    ctx: click.Context,
    symbols: tuple[str, ...],
    year: int | None,
    month: int | None,
    version: str,
    name: str,
) -> None:
    """Compute and store features from OHLCV data."""
    symbol_list = list(symbols) if symbols else None
    today = date.today()

    click.echo(
        f"Computing features: version={version}, "
        f"period={year or today.year}-{month or today.month:02d}"
    )

    try:
        from flows.feature_flow import feature_flow

        count = feature_flow(
            symbols=symbol_list,
            year=year,
            month=month,
            version=version,
            feature_set_name=name,
        )
        click.echo(f"Computed {count} feature rows")
    except Exception as exc:
        click.echo(f"Feature generation failed: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--date", "check_date", type=str, default=None, help="Date to validate (YYYY-MM-DD).")
@click.option("--symbols", "-s", multiple=True, help="Symbols to validate.")
@click.pass_context
def validate(ctx: click.Context, check_date: str | None, symbols: tuple[str, ...]) -> None:
    """Run data quality checks."""
    from market_data.config import get_settings
    from market_data.serving.quality import DataQualityChecker
    from market_data.storage.analytics_lake import AnalyticsLake
    from market_data.storage.cloud_storage import CloudStorageFactory
    from market_data.storage.symbol_master import SymbolMaster

    settings = get_settings(ctx.obj["config_path"], ctx.obj.get("env_path"))

    storage = CloudStorageFactory.create(
        provider=settings.storage.provider,
        local_path=settings.storage.local_path,
    )
    lake = AnalyticsLake(storage=storage)
    symbol_master = SymbolMaster(db_path=settings.storage.local_path + "/symbol_master.db")

    checker = DataQualityChecker(
        analytics_lake=lake,
        symbol_master=symbol_master,
        outlier_std_threshold=settings.quality.outlier_std_threshold,
        completeness_threshold=settings.quality.completeness_threshold,
    )

    if check_date is None:
        check_date = (date.today() - timedelta(days=1)).isoformat()

    symbol_list = list(symbols) if symbols else None
    click.echo(f"Running quality checks for {check_date}")

    report = checker.run_daily_checks(check_date=check_date, symbols=symbol_list)
    click.echo(f"Results: {report.passed_checks}/{report.total_checks} passed")

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        click.echo(f"  [{status}] {result.check_name}")
        if not result.passed and result.details:
            for k, v in result.details.items():
                if k != "symbols" and k != "missing_symbols" and k != "outliers":
                    click.echo(f"    {k}: {v}")


@cli.command()
@click.option("--host", type=str, default="0.0.0.0", help="Server host.")
@click.option("--port", type=int, default=8000, help="Server port.")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the HTTP health/metrics server."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from market_data.monitoring.health import HealthChecker
    from market_data.monitoring.metrics import MetricsCollector

    health = HealthChecker()
    metrics = MetricsCollector()
    health.mark_ready()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                status = health.liveness()
                self._respond(200, json.dumps(status.to_dict()))
            elif self.path == "/ready":
                status = health.readiness()
                code = 200 if status.healthy else 503
                self._respond(code, json.dumps(status.to_dict()))
            elif self.path == "/metrics":
                self._respond(200, metrics.get_metrics().decode(), "text/plain")
            else:
                self._respond(404, '{"error": "not found"}')

        def _respond(self, code: int, body: str, content_type: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, format: str, *args: object) -> None:
            pass  # Suppress default logging

    click.echo(f"Starting server on {host}:{port}")
    click.echo("Endpoints: /health, /ready, /metrics")
    server = HTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        server.shutdown()


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show platform status and statistics."""
    from market_data.config import get_settings
    from market_data.storage.cloud_storage import CloudStorageFactory
    from market_data.storage.symbol_master import SymbolMaster

    settings = get_settings(ctx.obj["config_path"], ctx.obj.get("env_path"))

    click.echo("Market Data Platform Status")
    click.echo(f"  Storage: {settings.storage.provider}")
    click.echo(f"  Data Path: {settings.storage.local_path}")

    try:
        symbol_master = SymbolMaster(db_path=settings.storage.local_path + "/symbol_master.db")
        click.echo(f"  Symbols: {symbol_master.symbol_count}")
    except Exception:
        click.echo("  Symbols: N/A")

    try:
        storage = CloudStorageFactory.create(
            provider=settings.storage.provider,
            local_path=settings.storage.local_path,
        )
        raw_files = storage.list_files("raw")
        analytics_files = storage.list_files("analytics")
        click.echo(f"  Raw files: {len(raw_files)}")
        click.echo(f"  Analytics files: {len(analytics_files)}")
    except Exception:
        click.echo("  Storage: N/A")


if __name__ == "__main__":
    cli()
