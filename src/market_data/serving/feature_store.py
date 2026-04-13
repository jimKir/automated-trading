"""Feature store for precomputed features, training dataset generation, and versioning."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from market_data.transforms.features import FeatureEngineer

if TYPE_CHECKING:
    from market_data.storage.cloud_storage import StorageBackend

logger = structlog.get_logger(__name__)


@dataclass
class FeatureSetMetadata:
    """Metadata for a stored feature set."""

    name: str
    version: str
    features: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    row_count: int = 0
    symbol_count: int = 0
    date_range: tuple[str, str] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDatasetConfig:
    """Configuration for training dataset generation.

    Args:
        features: List of feature column names to include.
        target: Target column name.
        lookback_window: Number of rows for lookback context.
        prediction_horizon: Number of rows ahead for target.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
    """

    features: list[str]
    target: str = "return_1d"
    lookback_window: int = 20
    prediction_horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class TrainingDataset:
    """Generated training dataset with train/val/test splits."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    metadata: dict[str, Any]


class FeatureStore:
    """Feature store for precomputed features and training dataset generation.

    Stores versioned feature sets as Parquet files. Supports feature selection,
    lookback windows, prediction horizons, and train/val/test splits for ML.

    Args:
        storage: Storage backend.
        base_path: Base path for feature store files.
        feature_engineer: Feature engineer for computing features.
    """

    def __init__(
        self,
        storage: StorageBackend,
        base_path: str = "feature_store",
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        self.storage = storage
        self.base_path = base_path
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self._metadata_cache: dict[str, FeatureSetMetadata] = {}

    def compute_and_store(
        self,
        df: pd.DataFrame,
        name: str,
        version: str | None = None,
    ) -> FeatureSetMetadata:
        """Compute features and store as a versioned feature set.

        Args:
            df: Input OHLCV DataFrame.
            name: Feature set name.
            version: Version string. Defaults to engineer's version.

        Returns:
            Metadata for the stored feature set.
        """
        version = version or self.feature_engineer.version
        features_df = self.feature_engineer.compute_all_features(df)

        # Store the feature data
        path = self._feature_path(name, version)
        buf = io.BytesIO()
        table = pa.Table.from_pandas(features_df)
        pq.write_table(table, buf, compression="snappy")
        self.storage.write(path, buf.getvalue())

        # Build metadata
        feature_cols = [
            c
            for c in features_df.columns
            if c not in ("timestamp_utc", "symbol_id", "ingestion_time")
        ]
        date_range = None
        if "timestamp_utc" in features_df.columns and len(features_df) > 0:
            date_range = (
                str(features_df["timestamp_utc"].min()),
                str(features_df["timestamp_utc"].max()),
            )

        metadata = FeatureSetMetadata(
            name=name,
            version=version,
            features=feature_cols,
            row_count=len(features_df),
            symbol_count=features_df["symbol_id"].nunique()
            if "symbol_id" in features_df.columns
            else 1,
            date_range=date_range,
        )

        # Store metadata
        meta_path = self._metadata_path(name, version)
        meta_dict = {
            "name": metadata.name,
            "version": metadata.version,
            "features": metadata.features,
            "created_at": metadata.created_at,
            "row_count": metadata.row_count,
            "symbol_count": metadata.symbol_count,
            "date_range": list(metadata.date_range) if metadata.date_range else None,
            "parameters": metadata.parameters,
        }
        self.storage.write(meta_path, json.dumps(meta_dict, indent=2).encode())

        self._metadata_cache[f"{name}/{version}"] = metadata

        logger.info(
            "feature_set_stored",
            name=name,
            version=version,
            rows=metadata.row_count,
            features=len(feature_cols),
        )
        return metadata

    def load_features(
        self,
        name: str,
        version: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load a stored feature set.

        Args:
            name: Feature set name.
            version: Version string.
            columns: Optional column subset.

        Returns:
            DataFrame with features.
        """
        path = self._feature_path(name, version)
        if not self.storage.exists(path):
            raise FileNotFoundError(f"Feature set not found: {name} v{version}")

        data = self.storage.read(path)
        table = pq.read_table(io.BytesIO(data), columns=columns)
        return table.to_pandas()

    def generate_training_dataset(
        self,
        name: str,
        version: str,
        config: TrainingDatasetConfig,
    ) -> TrainingDataset:
        """Generate a training dataset from stored features.

        Creates features with lookback window context and forward-looking
        targets, then splits into train/val/test sets chronologically.

        Args:
            name: Feature set name.
            version: Feature set version.
            config: Training dataset configuration.

        Returns:
            TrainingDataset with splits.
        """
        df = self.load_features(name, version)

        # Ensure features exist
        available_features = [f for f in config.features if f in df.columns]
        if not available_features:
            raise ValueError(f"None of the requested features found. Available: {list(df.columns)}")

        # Create target column (forward return)
        if config.target not in df.columns:
            if "close" in df.columns:
                df[config.target] = np.log(
                    df["close"].shift(-config.prediction_horizon) / df["close"]
                )
            else:
                raise ValueError(f"Target column '{config.target}' not found")

        # Build feature matrix with lookback
        feature_df = df[available_features].copy()
        target_series = df[config.target].copy()

        # Add lagged features for lookback
        if config.lookback_window > 1:
            lagged_frames = [feature_df]
            for lag in range(1, config.lookback_window):
                lagged = feature_df.shift(lag)
                lagged.columns = [f"{c}_lag{lag}" for c in feature_df.columns]
                lagged_frames.append(lagged)
            feature_df = pd.concat(lagged_frames, axis=1)

        # Drop rows with NaN from lookback/forward
        valid_mask = feature_df.notna().all(axis=1) & target_series.notna()
        feature_df = feature_df[valid_mask].reset_index(drop=True)
        target_series = target_series[valid_mask].reset_index(drop=True)

        # Chronological train/val/test split
        n = len(feature_df)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        dataset = TrainingDataset(
            X_train=feature_df.iloc[:train_end],
            y_train=target_series.iloc[:train_end],
            X_val=feature_df.iloc[train_end:val_end],
            y_val=target_series.iloc[train_end:val_end],
            X_test=feature_df.iloc[val_end:],
            y_test=target_series.iloc[val_end:],
            feature_names=list(feature_df.columns),
            metadata={
                "source_name": name,
                "source_version": version,
                "config": {
                    "features": config.features,
                    "target": config.target,
                    "lookback_window": config.lookback_window,
                    "prediction_horizon": config.prediction_horizon,
                    "train_ratio": config.train_ratio,
                    "val_ratio": config.val_ratio,
                    "test_ratio": config.test_ratio,
                },
                "total_rows": n,
                "train_rows": train_end,
                "val_rows": val_end - train_end,
                "test_rows": n - val_end,
                "feature_count": len(feature_df.columns),
            },
        )

        logger.info(
            "training_dataset_generated",
            name=name,
            version=version,
            train=train_end,
            val=val_end - train_end,
            test=n - val_end,
            features=len(feature_df.columns),
        )
        return dataset

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a feature set.

        Args:
            name: Feature set name.

        Returns:
            List of version strings.
        """
        prefix = f"{self.base_path}/{name}/"
        files = self.storage.list_files(prefix)
        versions = set()
        for f in files:
            parts = f.replace(prefix, "").split("/")
            if parts:
                versions.add(parts[0])
        return sorted(versions)

    def get_metadata(self, name: str, version: str) -> FeatureSetMetadata | None:
        """Get metadata for a feature set.

        Args:
            name: Feature set name.
            version: Version string.

        Returns:
            Metadata or None if not found.
        """
        cache_key = f"{name}/{version}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        meta_path = self._metadata_path(name, version)
        if not self.storage.exists(meta_path):
            return None

        data = self.storage.read(meta_path)
        meta_dict = json.loads(data)
        metadata = FeatureSetMetadata(
            name=meta_dict["name"],
            version=meta_dict["version"],
            features=meta_dict["features"],
            created_at=meta_dict.get("created_at", ""),
            row_count=meta_dict.get("row_count", 0),
            symbol_count=meta_dict.get("symbol_count", 0),
            date_range=tuple(meta_dict["date_range"]) if meta_dict.get("date_range") else None,
            parameters=meta_dict.get("parameters", {}),
        )
        self._metadata_cache[cache_key] = metadata
        return metadata

    def get_lineage(self) -> list[dict[str, Any]]:
        """Get feature lineage from the underlying feature engineer.

        Returns:
            List of version lineage records.
        """
        return self.feature_engineer.get_feature_lineage()

    def _feature_path(self, name: str, version: str) -> str:
        """Build path for feature Parquet file."""
        return f"{self.base_path}/{name}/{version}/features.parquet"

    def _metadata_path(self, name: str, version: str) -> str:
        """Build path for feature metadata JSON."""
        return f"{self.base_path}/{name}/{version}/metadata.json"
