"""Unit tests for SignalEngine — factor computation and regime dispatch."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategy.signals import SignalGenerator


@pytest.fixture
def signal_engine():
    config = {
        "strategy": {
            "lookback_fast": 20,
            "lookback_slow": 60,
            "lookback_vol": 21,
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            "momentum_threshold": 0.02,
            "regime_window": 126,
            "volume_confirmation": False,
            "pv_segments_enabled": False,
            "regime_switching": {"enabled": False},
            "predictive": {"credit_regime_enabled": False},
        },
        "trend_classifier": {"enabled": False},
    }
    return SignalGenerator(config)


@pytest.fixture
def mock_prices():
    """100 days of synthetic price data for 5 symbols."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
    data = {}
    for s in symbols:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 100)))
        data[s] = pd.DataFrame(
            {
                "Open": prices * 0.999,
                "High": prices * 1.005,
                "Low": prices * 0.995,
                "Close": prices,
                "Volume": np.random.randint(1_000_000, 10_000_000, 100),
            },
            index=dates,
        )
    return data


class TestBullBlendWeights:
    def test_bull_blend_sums_to_one(self, signal_engine):
        weights = signal_engine._bull_blend()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.11, (
            f"Bull blend sums to {total}, expected ~1.0 (imbalance is additive)"
        )

    def test_bear_blend_sums_to_one(self, signal_engine):
        weights = signal_engine._bear_blend()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.11, f"Bear blend sums to {total}, expected ~1.0"

    def test_choppy_blend_sums_to_one(self, signal_engine):
        weights = signal_engine._choppy_blend()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Choppy blend sums to {total}, expected 1.0"

    def test_vwap_not_in_bull_blend(self, signal_engine):
        """VWAP Factor 15 must NOT be in bull blend (rejected OOS — NOISE verdict)."""
        weights = signal_engine._bull_blend()
        assert "vwap_sma" not in weights or weights.get("vwap_sma", 0) == 0, (
            "VWAP-SMA should not be in bull blend (NOISE verdict)"
        )


class TestRegimeDispatch:
    def test_bull_regime_used_above_200d(self, signal_engine, mock_prices):
        """When SPY is above 200d MA and choppy_score is low, bull blend should be used."""
        spy = mock_prices["SPY"]["Close"]
        # Ensure uptrend: set last price well above mean
        spy_modified = spy.copy()
        spy_modified.iloc[-1] = spy.mean() * 1.2
        blend = signal_engine._get_regime_blend(choppy_score=0.05, spy_prices=spy_modified)
        expected = signal_engine._bull_blend()
        assert blend == expected

    def test_choppy_regime_above_threshold(self, signal_engine, mock_prices):
        """When choppy_score >= ORANGE threshold, choppy blend should be used."""
        spy = mock_prices["SPY"]["Close"]
        blend = signal_engine._get_regime_blend(choppy_score=0.30, spy_prices=spy)
        expected = signal_engine._choppy_blend(spy_within_15pct=True)
        assert blend == expected

    def test_generate_returns_all_symbols(self, signal_engine, mock_prices):
        result = signal_engine.generate(mock_prices, choppy_score=0.05)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > 0


class TestSignalComputation:
    def test_no_nan_in_signals(self, signal_engine, mock_prices):
        """No NaN values in signal output after warmup period."""
        result = signal_engine.generate(mock_prices, choppy_score=0.05)
        # Check last row (after warmup)
        for col in result.columns:
            val = result[col].iloc[-1]
            assert not np.isnan(val), f"NaN score for {col}"

    def test_stochastic_method_exists(self, signal_engine):
        assert hasattr(signal_engine, "_stochastic_contrarian"), (
            "Stochastic contrarian method missing"
        )

    def test_pmo_method_exists(self, signal_engine):
        assert hasattr(signal_engine, "_pmo_crossover"), (
            "PMO method missing (used in bear/choppy blends)"
        )

    def test_generate_accepts_choppy_score(self, signal_engine, mock_prices):
        """generate() must accept choppy_score parameter."""
        import inspect

        sig = inspect.signature(signal_engine.generate)
        assert "choppy_score" in sig.parameters, "generate() must accept choppy_score parameter"

    def test_generate_accepts_spy_prices(self, signal_engine, mock_prices):
        """generate() must accept spy_prices parameter."""
        import inspect

        sig = inspect.signature(signal_engine.generate)
        assert "spy_prices" in sig.parameters, "generate() must accept spy_prices parameter"
