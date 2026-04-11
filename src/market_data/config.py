"""Configuration management using Pydantic settings with YAML + env var support."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR:-default} patterns in config values."""
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def load_yaml_config(
    base_path: str | Path = "config/config.yaml",
    env_override_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load YAML config with optional environment-specific overlay."""
    base_path = Path(base_path)
    if not base_path.exists():
        return {}

    with open(base_path) as f:
        config = yaml.safe_load(f) or {}

    if env_override_path:
        override_path = Path(env_override_path)
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
            config = _deep_merge(config, overrides)

    return _resolve_env_vars(config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class DatabentoCfg(BaseModel):
    """Databento vendor configuration."""

    api_key: str = ""
    dataset: str = "GLBX.MDP3"
    rate_limit_per_min: int = 1000
    batch_threshold_gb: int = 5


class AlpacaCfg(BaseModel):
    """Alpaca vendor configuration."""

    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://data.alpaca.markets"
    rate_limit_per_min: int = 10000


class VendorsCfg(BaseModel):
    """All vendor configurations."""

    databento: DatabentoCfg = Field(default_factory=DatabentoCfg)
    alpaca: AlpacaCfg = Field(default_factory=AlpacaCfg)


class StorageCfg(BaseModel):
    """Storage configuration."""

    provider: str = "local"
    raw_lake_path: str = "/data/raw"
    analytics_lake_path: str = "/data/analytics"
    feature_store_path: str = "/data/features"
    format: str = "parquet"
    compression: str = "snappy"
    row_group_size_mb: int = 128


class BackfillCfg(BaseModel):
    """Backfill ingestion configuration."""

    start_date: str = "2019-01-01"
    end_date: str = "2026-03-24"
    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])
    schemas: list[str] = Field(default_factory=lambda: ["trades", "quotes", "ohlcv-1m", "ohlcv-1d"])
    chunk_size_days: int = 30
    max_retries: int = 5
    checkpoint_every_records: int = 1000
    checkpoint_every_mb: int = 100


class IncrementalCfg(BaseModel):
    """Incremental ingestion configuration."""

    schedule: str = "0 17 * * 1-5"
    lookback_hours: int = 24
    symbols_per_batch: int = 100


class IngestionCfg(BaseModel):
    """Ingestion configuration."""

    backfill: BackfillCfg = Field(default_factory=BackfillCfg)
    incremental: IncrementalCfg = Field(default_factory=IncrementalCfg)


class TechnicalIndicator(BaseModel):
    """Single technical indicator config."""

    name: str
    type: str
    window: int | None = None
    fast: int | None = None
    slow: int | None = None
    signal: int | None = None
    std: int | None = None


class FeaturesCfg(BaseModel):
    """Feature engineering configuration."""

    technical_indicators: list[TechnicalIndicator] = Field(default_factory=list)
    volume_indicators: list[TechnicalIndicator] = Field(default_factory=list)
    volatility_indicators: list[TechnicalIndicator] = Field(default_factory=list)
    version: str = "1.0.0"


class QualityCfg(BaseModel):
    """Data quality configuration."""

    completeness_threshold: float = 0.995
    outlier_std_threshold: int = 10
    cross_validation_sample_size: int = 10
    alert_channels: list[str] = Field(default_factory=lambda: ["slack", "email"])


class MonitoringCfg(BaseModel):
    """Monitoring configuration."""

    prometheus_port: int = 9090
    health_check_port: int = 8080
    slack_webhook_url: str = ""
    alert_email: str = ""


class Settings(BaseSettings):
    """Root application settings.

    Loads from config.yaml with env var overrides.
    """

    environment: str = "dev"
    log_level: str = "INFO"
    vendors: VendorsCfg = Field(default_factory=VendorsCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    ingestion: IngestionCfg = Field(default_factory=IngestionCfg)
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    quality: QualityCfg = Field(default_factory=QualityCfg)
    monitoring: MonitoringCfg = Field(default_factory=MonitoringCfg)

    model_config = {"env_prefix": "", "extra": "ignore"}


def get_settings(
    config_path: str | Path | None = None,
    environment: str | None = None,
) -> Settings:
    """Load settings from YAML config + environment variables.

    Args:
        config_path: Path to base config.yaml.
        environment: Environment name (dev/test/prod) for overlay config.

    Returns:
        Fully resolved Settings object.
    """
    env = environment or os.environ.get("ENVIRONMENT", "dev")
    base_path = Path(config_path) if config_path else Path("config/config.yaml")
    env_overlay = base_path.parent / f"config.{env}.yaml"

    yaml_data = load_yaml_config(
        base_path=base_path,
        env_override_path=env_overlay if env_overlay.exists() else None,
    )
    yaml_data["environment"] = env
    return Settings(**yaml_data)
