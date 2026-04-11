"""Feature engineering pipeline with technical indicators, volume, and volatility features."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureVersion:
    """Feature version metadata for lineage tracking."""

    version: str
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    features: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    parent_version: str | None = None


class FeatureEngineer:
    """Feature engineering pipeline for market data.

    Computes technical indicators, volume features, and volatility measures.
    Supports feature versioning with semantic versioning and lineage tracking.

    Features computed:
    - Technical: SMA, EMA, RSI, MACD, Bollinger Bands
    - Volume: VWAP, OBV, volume ratio
    - Volatility: Garman-Klass, Parkinson, ATR
    - Returns: 1d, 5d, 20d log returns

    Args:
        version: Feature version string (semantic versioning).
    """

    def __init__(self, version: str = "1.0.0") -> None:
        self.version = version
        self._versions: list[FeatureVersion] = []

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for a DataFrame of OHLCV data.

        Expected columns: open, high, low, close, volume, timestamp_utc, symbol_id.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            DataFrame with all computed features appended.
        """
        result = df.copy()

        # Returns
        result["return_1d"] = self._log_returns(result["close"], 1)
        result["return_5d"] = self._log_returns(result["close"], 5)
        result["return_20d"] = self._log_returns(result["close"], 20)

        # Technical indicators
        result["sma_20"] = self.sma(result["close"], 20)
        result["ema_50"] = self.ema(result["close"], 50)
        result["rsi_14"] = self.rsi(result["close"], 14)

        macd_line, signal_line, histogram = self.macd(result["close"])
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = histogram

        upper, middle, lower = self.bollinger_bands(result["close"], 20, 2)
        result["bb_upper"] = upper
        result["bb_middle"] = middle
        result["bb_lower"] = lower

        # Volume features
        if "vwap" not in result.columns or result["vwap"].isna().all():
            result["vwap_computed"] = self.compute_vwap(
                result["high"], result["low"], result["close"], result["volume"]
            )
        result["obv"] = self.on_balance_volume(result["close"], result["volume"])
        result["volume_ratio"] = self.volume_ratio(result["volume"], 20)

        # Volatility features
        result["realized_vol_20"] = self.garman_klass_volatility(
            result["open"], result["high"], result["low"], result["close"], 20
        )
        result["parkinson_vol_20"] = self.parkinson_volatility(result["high"], result["low"], 20)
        result["atr_14"] = self.atr(result["high"], result["low"], result["close"], 14)

        # Record version
        feature_cols = [
            "return_1d",
            "return_5d",
            "return_20d",
            "sma_20",
            "ema_50",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "obv",
            "volume_ratio",
            "realized_vol_20",
            "parkinson_vol_20",
            "atr_14",
        ]
        self._versions.append(
            FeatureVersion(
                version=self.version,
                features=feature_cols,
            )
        )

        logger.info(
            "features_computed",
            version=self.version,
            rows=len(result),
            feature_count=len(feature_cols),
        )
        return result

    @staticmethod
    def _log_returns(series: pd.Series, periods: int) -> pd.Series:
        """Compute log returns over N periods."""
        return np.log(series / series.shift(periods))

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index.

        Args:
            series: Price series.
            window: RSI lookback window.

        Returns:
            RSI values (0-100).
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence).

        Args:
            series: Price series.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal line period.

        Returns:
            Tuple of (MACD line, signal line, histogram).
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.

        Args:
            series: Price series.
            window: Moving average window.
            num_std: Number of standard deviations.

        Returns:
            Tuple of (upper band, middle band, lower band).
        """
        middle = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    @staticmethod
    def compute_vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price (VWAP).

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            volume: Volume.

        Returns:
            VWAP series.
        """
        typical_price = (high + low + close) / 3.0
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)

    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume (OBV).

        Args:
            close: Close prices.
            volume: Volume.

        Returns:
            OBV series.
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        obv = (direction * volume).cumsum()
        return obv

    @staticmethod
    def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume ratio vs N-day average.

        Args:
            volume: Volume series.
            window: Lookback window for average.

        Returns:
            Ratio of current volume to average volume.
        """
        avg_vol = volume.rolling(window=window, min_periods=1).mean()
        return volume / avg_vol.replace(0, np.nan)

    @staticmethod
    def garman_klass_volatility(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Garman-Klass volatility estimator.

        More efficient than close-to-close estimator as it uses OHLC data.

        Args:
            open_: Open prices.
            high: High prices.
            low: Low prices.
            close: Close prices.
            window: Rolling window.

        Returns:
            Annualized volatility series.
        """
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
        gk = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
        return np.sqrt(gk.rolling(window=window, min_periods=1).mean() * 252)

    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """Parkinson volatility estimator using high-low range.

        Args:
            high: High prices.
            low: Low prices.
            window: Rolling window.

        Returns:
            Annualized volatility series.
        """
        log_hl_sq = np.log(high / low) ** 2
        factor = 1.0 / (4.0 * np.log(2.0))
        return np.sqrt(factor * log_hl_sq.rolling(window=window, min_periods=1).mean() * 252)

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Average True Range.

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            window: ATR period.

        Returns:
            ATR series.
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.ewm(alpha=1.0 / window, min_periods=window).mean()

    def get_feature_lineage(self) -> list[dict[str, Any]]:
        """Get lineage information for all computed feature versions.

        Returns:
            List of version metadata dictionaries.
        """
        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "features": v.features,
                "parameters": v.parameters,
                "parent_version": v.parent_version,
            }
            for v in self._versions
        ]

    def save_lineage(self, path: str | Path) -> None:
        """Save feature lineage to JSON file.

        Args:
            path: Output path for lineage JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_feature_lineage(), f, indent=2)
