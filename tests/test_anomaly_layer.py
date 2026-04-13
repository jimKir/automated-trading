"""
Unit tests for the multi-source anomaly detection layer.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from regime.anomaly_layer import AnomalyRegimeLayer, AnomalyScore


@pytest.fixture
def layer():
    """Create a default AnomalyRegimeLayer."""
    return AnomalyRegimeLayer()


@pytest.fixture
def sample_prices():
    """Create synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    syms = ["SPY", "QQQ", "TLT", "GLD", "IWM"]
    data = {}
    for sym in syms:
        base = 100 + np.random.randn() * 10
        returns = np.random.randn(500) * 0.01
        prices = base * np.exp(np.cumsum(returns))
        data[sym] = prices
    return pd.DataFrame(data, index=dates)


class TestAnomalyRegimeLayerInstantiation:
    """Test that AnomalyRegimeLayer can be created without errors."""

    def test_default_instantiation(self):
        layer = AnomalyRegimeLayer()
        assert layer is not None

    def test_with_config(self):
        config = {
            "anomaly_layer": {
                "enabled": True,
                "macro_weight": 0.25,
                "sentiment_weight": 0.35,
                "fx_weight": 0.20,
                "isolation_weight": 0.20,
            }
        }
        layer = AnomalyRegimeLayer(config)
        assert layer.weights["macro"] == 0.25
        assert layer.weights["sentiment"] == 0.35

    def test_empty_config(self):
        layer = AnomalyRegimeLayer(config={})
        assert layer.weights == AnomalyRegimeLayer.DEFAULT_WEIGHTS


class TestAnomalyScoreOutput:
    """Test that compute() returns proper AnomalyScore."""

    def test_compute_returns_anomaly_score(self, layer, sample_prices):
        result = layer.compute(sample_prices)
        assert isinstance(result, AnomalyScore)

    def test_composite_in_range(self, layer, sample_prices):
        result = layer.compute(sample_prices)
        assert 0.0 <= result.composite <= 1.0

    def test_label_is_valid(self, layer, sample_prices):
        result = layer.compute(sample_prices)
        assert result.label in ["NORMAL", "ELEVATED", "STRESSED", "CRISIS"]

    def test_position_scale_is_valid(self, layer, sample_prices):
        result = layer.compute(sample_prices)
        assert result.position_scale in [1.0, 0.85, 0.65, 0.40]

    def test_source_scores_is_dict(self, layer, sample_prices):
        result = layer.compute(sample_prices)
        assert isinstance(result.source_scores, dict)

    def test_source_scores_in_range(self, layer, sample_prices):
        result = layer.compute(sample_prices)
        for src, val in result.source_scores.items():
            assert 0.0 <= val <= 1.0, f"Source {src} score {val} out of range"


class TestRegimeLabels:
    """Test regime label mapping."""

    def test_normal(self):
        assert AnomalyRegimeLayer.get_regime_label(0.10) == "NORMAL"

    def test_normal_boundary(self):
        assert AnomalyRegimeLayer.get_regime_label(0.19) == "NORMAL"

    def test_elevated(self):
        assert AnomalyRegimeLayer.get_regime_label(0.25) == "ELEVATED"

    def test_elevated_boundary(self):
        assert AnomalyRegimeLayer.get_regime_label(0.20) == "ELEVATED"

    def test_stressed(self):
        assert AnomalyRegimeLayer.get_regime_label(0.40) == "STRESSED"

    def test_stressed_boundary(self):
        assert AnomalyRegimeLayer.get_regime_label(0.35) == "STRESSED"

    def test_crisis(self):
        assert AnomalyRegimeLayer.get_regime_label(0.60) == "CRISIS"

    def test_crisis_boundary(self):
        assert AnomalyRegimeLayer.get_regime_label(0.50) == "CRISIS"

    def test_extreme_crisis(self):
        assert AnomalyRegimeLayer.get_regime_label(1.0) == "CRISIS"

    def test_zero(self):
        assert AnomalyRegimeLayer.get_regime_label(0.0) == "NORMAL"


class TestPositionScale:
    """Test position scale multiplier."""

    def test_normal_scale(self, layer):
        assert layer.get_position_scale(0.10) == 1.0

    def test_elevated_scale(self, layer):
        assert layer.get_position_scale(0.25) == 0.85

    def test_stressed_scale(self, layer):
        assert layer.get_position_scale(0.40) == 0.65

    def test_crisis_scale(self, layer):
        assert layer.get_position_scale(0.60) == 0.40


class TestGracefulDegradation:
    """Test that the layer works when sources fail."""

    def test_no_sources_returns_normal(self):
        """When all sources fail, should return NORMAL with composite=0."""
        layer = AnomalyRegimeLayer()
        layer._initialised = True
        # All detectors set to None = all fail
        layer._macro_scorer = None
        layer._fx_detector = None
        layer._sentiment_detector = None
        layer._isolation_detector = None

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.DataFrame({"SPY": np.random.randn(100).cumsum() + 100}, index=dates)

        result = layer.compute(prices)
        assert result.composite == 0.0
        assert result.label == "NORMAL"
        assert result.position_scale == 1.0
        assert result.source_scores == {}

    def test_partial_sources(self):
        """When some sources fail, remaining should be re-weighted."""
        layer = AnomalyRegimeLayer()
        layer._initialised = True
        layer._macro_scorer = None  # macro fails
        layer._isolation_detector = None  # isolation fails
        # fx and sentiment should still work if data is available

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        prices = pd.DataFrame({"SPY": np.random.randn(200).cumsum() + 100}, index=dates)
        result = layer.compute(prices)
        # Should still return a valid score even with missing sources
        assert 0.0 <= result.composite <= 1.0


class TestWeights:
    """Test weight configuration."""

    def test_default_weights_sum_to_one(self):
        total = sum(AnomalyRegimeLayer.DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Default weights sum to {total}, expected 1.0"

    def test_custom_weights(self):
        config = {
            "anomaly_layer": {
                "macro_weight": 0.40,
                "sentiment_weight": 0.20,
                "fx_weight": 0.25,
                "isolation_weight": 0.15,
            }
        }
        layer = AnomalyRegimeLayer(config)
        assert layer.weights["macro"] == 0.40
        assert layer.weights["sentiment"] == 0.20

    def test_default_weights_keys(self):
        expected = {"macro", "sentiment", "fx", "isolation"}
        assert set(AnomalyRegimeLayer.DEFAULT_WEIGHTS.keys()) == expected


class TestComputeSeries:
    """Test series computation for backtest."""

    def test_compute_series_returns_dataframe(self, layer, sample_prices):
        result = layer.compute_series(sample_prices)
        assert isinstance(result, pd.DataFrame)
        assert "composite" in result.columns
        assert "label" in result.columns
        assert "scale" in result.columns

    def test_composite_series_in_range(self, layer, sample_prices):
        result = layer.compute_series(sample_prices)
        assert (result["composite"] >= 0).all()
        assert (result["composite"] <= 1).all()

    def test_labels_valid(self, layer, sample_prices):
        result = layer.compute_series(sample_prices)
        valid_labels = {"NORMAL", "ELEVATED", "STRESSED", "CRISIS"}
        assert set(result["label"].unique()).issubset(valid_labels)


class TestThresholds:
    """Test threshold constants."""

    def test_thresholds_ordering(self):
        t = AnomalyRegimeLayer.THRESHOLDS
        assert t["elevated"] < t["stressed"] < t["crisis"]

    def test_scale_map_ordering(self):
        s = AnomalyRegimeLayer.SCALE_MAP
        assert s["NORMAL"] > s["ELEVATED"] > s["STRESSED"] > s["CRISIS"]

    def test_all_scales_positive(self):
        for label, scale in AnomalyRegimeLayer.SCALE_MAP.items():
            assert scale > 0, f"Scale for {label} is {scale}"
