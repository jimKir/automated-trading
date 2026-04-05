"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data.transforms.features import FeatureEngineer


class TestFeatureEngineer:
    def test_compute_all_features(self, sample_ohlcv_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(version="1.0.0")
        result = fe.compute_all_features(sample_ohlcv_df)

        # Check that all expected columns are present
        expected = [
            "return_1d", "return_5d", "return_20d",
            "sma_20", "ema_50", "rsi_14",
            "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower",
            "obv", "volume_ratio",
            "realized_vol_20", "parkinson_vol_20", "atr_14",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

        # Original columns preserved
        assert "close" in result.columns
        assert "volume" in result.columns
        assert len(result) == len(sample_ohlcv_df)

    def test_sma(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = FeatureEngineer.sma(series, window=3)
        assert abs(result.iloc[-1] - 4.0) < 1e-10

    def test_ema(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = FeatureEngineer.ema(series, window=3)
        assert len(result) == 5
        # EMA should be close to recent values
        assert result.iloc[-1] > result.iloc[0]

    def test_rsi_bounds(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = FeatureEngineer.rsi(sample_ohlcv_df["close"], window=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_returns_three_series(self, sample_ohlcv_df: pd.DataFrame) -> None:
        macd_line, signal, histogram = FeatureEngineer.macd(sample_ohlcv_df["close"])
        assert len(macd_line) == len(sample_ohlcv_df)
        assert len(signal) == len(sample_ohlcv_df)
        assert len(histogram) == len(sample_ohlcv_df)

    def test_bollinger_bands(self, sample_ohlcv_df: pd.DataFrame) -> None:
        upper, middle, lower = FeatureEngineer.bollinger_bands(sample_ohlcv_df["close"])
        # Upper should be above middle, middle above lower
        valid_mask = upper.notna() & middle.notna() & lower.notna()
        assert (upper[valid_mask] >= middle[valid_mask]).all()
        assert (middle[valid_mask] >= lower[valid_mask]).all()

    def test_atr_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = FeatureEngineer.atr(
            sample_ohlcv_df["high"],
            sample_ohlcv_df["low"],
            sample_ohlcv_df["close"],
        )
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_feature_lineage(self, sample_ohlcv_df: pd.DataFrame) -> None:
        fe = FeatureEngineer(version="2.0.0")
        fe.compute_all_features(sample_ohlcv_df)
        lineage = fe.get_feature_lineage()
        assert len(lineage) == 1
        assert lineage[0]["version"] == "2.0.0"
        assert len(lineage[0]["features"]) > 0


class TestCorporateActions:
    def test_split_adjustment(self, corporate_actions) -> None:
        from market_data.transforms.corporate_actions import SplitRecord

        corporate_actions.add_split(SplitRecord(
            symbol_id=1,
            ex_date="2024-06-15",
            split_from=1,
            split_to=4,
        ))

        factor = corporate_actions.get_split_adjustment_factor(
            symbol_id=1,
            as_of_date="2024-12-31",
            target_date="2024-01-01",
        )
        assert factor == 4.0

    def test_dividend_adjustment(self, corporate_actions) -> None:
        from market_data.transforms.corporate_actions import DividendRecord

        corporate_actions.add_dividend(DividendRecord(
            symbol_id=1,
            ex_date="2024-03-15",
            amount=0.50,
        ))
        corporate_actions.add_dividend(DividendRecord(
            symbol_id=1,
            ex_date="2024-06-15",
            amount=0.50,
        ))

        total = corporate_actions.get_dividend_adjustment(
            symbol_id=1,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert total == 1.0

    def test_adjust_prices_raw(self, corporate_actions) -> None:
        prices = np.array([100.0, 101.0, 102.0])
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        result = corporate_actions.adjust_prices(
            prices=prices,
            dates=dates,
            symbol_id=1,
            adjustment_mode="raw",
        )
        np.testing.assert_array_equal(result, prices)
