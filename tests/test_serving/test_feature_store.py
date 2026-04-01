"""Tests for feature store."""

from __future__ import annotations

import pandas as pd

from market_data.serving.feature_store import (
    FeatureStore,
    TrainingDatasetConfig,
)
from market_data.storage.cloud_storage import LocalStorageBackend
from market_data.transforms.features import FeatureEngineer


class TestFeatureStore:
    def test_compute_and_store(
        self, local_storage: LocalStorageBackend, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        store = FeatureStore(storage=local_storage)
        metadata = store.compute_and_store(sample_ohlcv_df, "test_features")
        assert metadata.row_count == len(sample_ohlcv_df)
        assert metadata.version == "1.0.0"
        assert len(metadata.features) > 0

    def test_load_features(
        self, local_storage: LocalStorageBackend, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        store = FeatureStore(storage=local_storage)
        store.compute_and_store(sample_ohlcv_df, "test_features")
        loaded = store.load_features("test_features", "1.0.0")
        assert len(loaded) == len(sample_ohlcv_df)
        assert "sma_20" in loaded.columns

    def test_load_features_with_column_filter(
        self, local_storage: LocalStorageBackend, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        store = FeatureStore(storage=local_storage)
        store.compute_and_store(sample_ohlcv_df, "test_features")
        loaded = store.load_features(
            "test_features", "1.0.0", columns=["close", "sma_20"]
        )
        assert list(loaded.columns) == ["close", "sma_20"]

    def test_get_metadata(
        self, local_storage: LocalStorageBackend, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        store = FeatureStore(storage=local_storage)
        store.compute_and_store(sample_ohlcv_df, "test_features")
        meta = store.get_metadata("test_features", "1.0.0")
        assert meta is not None
        assert meta.name == "test_features"

    def test_generate_training_dataset(
        self, local_storage: LocalStorageBackend, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        store = FeatureStore(storage=local_storage)
        store.compute_and_store(sample_ohlcv_df, "test_features")

        config = TrainingDatasetConfig(
            features=["sma_20", "ema_50", "rsi_14"],
            target="return_1d",
            lookback_window=1,
            prediction_horizon=1,
        )
        dataset = store.generate_training_dataset("test_features", "1.0.0", config)
        assert len(dataset.X_train) > 0
        assert len(dataset.X_test) > 0
        assert len(dataset.feature_names) == 3
        total = len(dataset.X_train) + len(dataset.X_val) + len(dataset.X_test)
        assert total > 0

    def test_list_versions(
        self, local_storage: LocalStorageBackend, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        store = FeatureStore(storage=local_storage)
        store.compute_and_store(sample_ohlcv_df, "test_features", version="1.0.0")
        versions = store.list_versions("test_features")
        assert "1.0.0" in versions
