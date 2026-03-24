"""Prefect flow for feature computation and storage."""

from __future__ import annotations

from datetime import date

import pandas as pd
import structlog
from prefect import flow, task

from market_data.config import get_settings
from market_data.serving.feature_store import FeatureStore
from market_data.storage.analytics_lake import AnalyticsLake
from market_data.storage.cloud_storage import CloudStorageFactory
from market_data.storage.symbol_master import SymbolMaster
from market_data.transforms.features import FeatureEngineer

logger = structlog.get_logger(__name__)


@task(retries=1, retry_delay_seconds=30)
def compute_features_for_symbol(
    symbol_id: int,
    ticker: str,
    analytics_lake: AnalyticsLake,
    feature_engineer: FeatureEngineer,
    year: int,
    month: int,
) -> pd.DataFrame | None:
    """Compute features for a single symbol and month.

    Args:
        symbol_id: Internal symbol ID.
        ticker: Symbol ticker.
        analytics_lake: Analytics lake to read from.
        feature_engineer: Feature engineer instance.
        year: Data year.
        month: Data month.

    Returns:
        DataFrame with computed features, or None.
    """
    table = analytics_lake.read_table(
        asset_class="equity",
        schema_name="ohlcv-1d",
        symbol_id=symbol_id,
        year=year,
        month=month,
    )

    if table is None or table.num_rows == 0:
        logger.debug("no_data_for_features", ticker=ticker, year=year, month=month)
        return None

    df = table.to_pandas()
    featured = feature_engineer.compute_all_features(df)

    logger.info(
        "features_computed_for_symbol",
        ticker=ticker,
        rows=len(featured),
        year=year,
        month=month,
    )
    return featured


@task
def store_feature_set(
    feature_store: FeatureStore,
    df: pd.DataFrame,
    name: str,
    version: str,
) -> str:
    """Store computed features in the feature store.

    Args:
        feature_store: Feature store instance.
        df: DataFrame with features.
        name: Feature set name.
        version: Version string.

    Returns:
        Storage path.
    """
    metadata = feature_store.compute_and_store(df, name, version)
    logger.info("feature_set_stored", name=name, version=version, rows=metadata.row_count)
    return f"{name}/v{version}"


@flow(name="Feature Computation", log_prints=True)
def feature_flow(
    symbols: list[str] | None = None,
    year: int | None = None,
    month: int | None = None,
    version: str | None = None,
    feature_set_name: str = "daily_features",
) -> int:
    """Compute features for all symbols and store in feature store.

    Reads OHLCV data from the analytics lake, computes technical,
    volume, and volatility features, then stores in the feature store.

    Args:
        symbols: Symbols to compute features for. Defaults to all active.
        year: Year to process. Defaults to current year.
        month: Month to process. Defaults to current month.
        version: Feature version. Defaults to config.
        feature_set_name: Name for the feature set.

    Returns:
        Number of features computed.
    """
    settings = get_settings()
    today = date.today()

    if year is None:
        year = today.year
    if month is None:
        month = today.month
    if version is None:
        version = "1.0.0"

    # Initialize components
    storage = CloudStorageFactory.create(
        provider=settings.storage.provider,
        local_path=settings.storage.local_path,
    )
    analytics_lake = AnalyticsLake(storage=storage)
    feature_engineer = FeatureEngineer(version=version)
    feature_store = FeatureStore(
        storage=storage,
        feature_engineer=feature_engineer,
    )
    symbol_master = SymbolMaster(db_path=settings.storage.local_path + "/symbol_master.db")

    # Get symbols
    if symbols is None:
        records = symbol_master.list_symbols(asset_class="equity", active_only=True)
        symbols_with_ids = [(r.ticker, r.symbol_id) for r in records]
    else:
        symbols_with_ids = []
        for ticker in symbols:
            sid = symbol_master.get_symbol_id(ticker)
            if sid is not None:
                symbols_with_ids.append((ticker, sid))

    if not symbols_with_ids:
        logger.warning("no_symbols_for_feature_computation")
        return 0

    logger.info(
        "feature_flow_started",
        symbols=len(symbols_with_ids),
        year=year,
        month=month,
        version=version,
    )

    # Compute features for each symbol
    all_frames: list[pd.DataFrame] = []
    for ticker, symbol_id in symbols_with_ids:
        result = compute_features_for_symbol(
            symbol_id=symbol_id,
            ticker=ticker,
            analytics_lake=analytics_lake,
            feature_engineer=feature_engineer,
            year=year,
            month=month,
        )
        if result is not None:
            all_frames.append(result)

    if not all_frames:
        logger.warning("no_features_computed")
        return 0

    combined = pd.concat(all_frames, ignore_index=True)
    store_feature_set(
        feature_store=feature_store,
        df=combined,
        name=feature_set_name,
        version=version,
    )

    logger.info(
        "feature_flow_complete",
        total_rows=len(combined),
        symbols_processed=len(all_frames),
    )
    return len(combined)


if __name__ == "__main__":
    feature_flow()
