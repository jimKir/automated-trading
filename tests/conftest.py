"""Shared pytest fixtures for the market data platform tests."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from market_data.storage.cloud_storage import LocalStorageBackend
from market_data.storage.analytics_lake import AnalyticsLake, OHLCV_SCHEMA
from market_data.storage.symbol_master import SymbolMaster, SymbolRecord
from market_data.transforms.corporate_actions import CorporateActionsManager
from market_data.transforms.features import FeatureEngineer


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def local_storage(tmp_dir: Path) -> LocalStorageBackend:
    """Provide a local storage backend rooted in a temp dir."""
    return LocalStorageBackend(base_path=str(tmp_dir / "storage"))


@pytest.fixture
def analytics_lake(local_storage: LocalStorageBackend) -> AnalyticsLake:
    """Provide an analytics lake with local storage."""
    return AnalyticsLake(storage=local_storage)


@pytest.fixture
def symbol_master(tmp_dir: Path) -> SymbolMaster:
    """Provide a fresh symbol master."""
    sm = SymbolMaster(db_path=tmp_dir / "symbol_master.db")
    # Pre-populate with test symbols
    sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="AAPL", asset_class="equity", name="Apple Inc."))
    sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="MSFT", asset_class="equity", name="Microsoft Corp."))
    sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="GOOGL", asset_class="equity", name="Alphabet Inc."))
    return sm


@pytest.fixture
def corporate_actions(tmp_dir: Path) -> CorporateActionsManager:
    """Provide a corporate actions manager."""
    return CorporateActionsManager(db_path=tmp_dir / "corporate_actions.db")


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """Provide a feature engineer."""
    return FeatureEngineer(version="1.0.0")


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    # Generate realistic-ish prices
    base = 150.0
    returns = np.random.normal(0.001, 0.02, n)
    close = base * np.exp(np.cumsum(returns))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.01, n)))
    volume = np.random.randint(1_000_000, 50_000_000, n)

    return pd.DataFrame({
        "timestamp_utc": [int(d.timestamp() * 1e9) for d in dates],
        "symbol_id": [1] * n,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": (high + low + close) / 3,
        "trade_count": np.random.randint(100, 10000, n),
    })


@pytest.fixture
def sample_ohlcv_table(sample_ohlcv_df: pd.DataFrame) -> pa.Table:
    """Create a sample OHLCV PyArrow table."""
    now = datetime.now(tz=timezone.utc)
    df = sample_ohlcv_df.copy()
    df["ingestion_time"] = now
    return pa.Table.from_pandas(df, schema=OHLCV_SCHEMA)
