"""Unit tests for ChoppyRegimeDetector v4."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from regime.choppy_regime import ChoppyRegimeDetector


@pytest.fixture
def detector():
    return ChoppyRegimeDetector(mode='backtest')


@pytest.fixture
def mock_data():
    """30 days of synthetic daily data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    spy = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, 30))), index=dates)
    hyg = pd.Series(80 * np.exp(np.cumsum(np.random.normal(0.0002, 0.004, 30))), index=dates)
    lqd = pd.Series(90 * np.exp(np.cumsum(np.random.normal(0.0001, 0.003, 30))), index=dates)
    vix = pd.Series(np.random.uniform(12, 20, 30), index=dates)
    spy_df = pd.DataFrame({
        'close': spy,
        'volume': np.random.randint(50_000_000, 200_000_000, 30)
    }, index=dates)
    return {'spy_daily': spy_df, 'HYG': hyg, 'LQD': lqd, 'vix': vix}


class TestInitialization:
    def test_has_nine_feature_groups(self, detector):
        assert len(detector.feature_groups) == 9, \
            f"Expected 9 feature groups (v4), got {len(detector.feature_groups)}"

    def test_thresholds_loaded(self, detector):
        assert hasattr(detector, 'GREEN_MAX'), "GREEN_MAX threshold not set"
        assert hasattr(detector, 'YELLOW_MAX'), "YELLOW_MAX threshold not set"
        assert hasattr(detector, 'ORANGE_MAX'), "ORANGE_MAX threshold not set"

    def test_v4_thresholds_values(self, detector):
        """v4 thresholds from regime_params_validated.json."""
        # The JSON has green_ceiling=0.17, yellow_ceiling=0.27, orange_ceiling=0.40
        assert pytest.approx(0.17, abs=0.03) == detector.GREEN_MAX
        assert pytest.approx(0.27, abs=0.03) == detector.YELLOW_MAX
        assert pytest.approx(0.40, abs=0.05) == detector.ORANGE_MAX

    def test_order_flow_group_present(self, detector):
        assert 'order_flow' in detector.feature_groups, \
            "order_flow group missing — ChoppyDetector is not v4"

    def test_credit_group_present(self, detector):
        assert 'credit_stress' in detector.feature_groups, \
            "credit_stress group missing — ChoppyDetector is not v3+"

    def test_order_flow_detector_exists(self, detector):
        assert hasattr(detector, 'order_flow_detector'), \
            "OrderFlowAnomalyDetector not wired"


class TestScoring:
    def test_regime_label(self, detector):
        assert detector.get_regime(0.10) == 'GREEN'
        assert detector.get_regime(0.20) == 'YELLOW'
        assert detector.get_regime(0.35) == 'ORANGE'
        assert detector.get_regime(0.50) == 'RED'

    def test_score_to_scale_returns_tuple(self, detector):
        scale, colour = detector.score_to_scale(0.10)
        assert isinstance(scale, float)
        assert isinstance(colour, str)
        assert 0 < scale <= 1.0

    def test_current_score_default(self, detector):
        """current_score() should return 0.0 before any computation."""
        assert detector.current_score() == 0.0
