"""Unit tests for PositionAnomalyScorer."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.position_anomaly import (
    _CLASS_CONFIG,
    _DD_CEILING_CRYPTO,
    _DD_CEILING_EQUITY,
    _VOL_SPIKE_CEILING,
    AssetClass,
    PositionAnomalyScorer,
    classify,
)


@pytest.fixture
def scorer():
    return PositionAnomalyScorer()


class TestConfiguration:
    def test_crypto_floor(self, scorer):
        cfg = _CLASS_CONFIG[AssetClass.CRYPTO]
        assert cfg["floor"] == pytest.approx(0.10), (
            f"Crypto floor should be 0.10, got {cfg['floor']}"
        )

    def test_equity_floor(self, scorer):
        cfg = _CLASS_CONFIG[AssetClass.EQUITY]
        assert cfg["floor"] == pytest.approx(0.40), (
            f"Equity floor should be 0.40, got {cfg['floor']}"
        )

    def test_g1_ceiling_corrected(self, scorer):
        """G1 ceiling must be 1.55 (corrected from broken 3.0)."""
        assert pytest.approx(1.55) == _VOL_SPIKE_CEILING, (
            f"G1_ceiling should be 1.55 (WF-calibrated), got {_VOL_SPIKE_CEILING}"
        )

    def test_g3_dd_ceil_crypto(self, scorer):
        assert pytest.approx(0.25) == _DD_CEILING_CRYPTO

    def test_g3_dd_ceil_equity(self, scorer):
        assert pytest.approx(0.15) == _DD_CEILING_EQUITY


class TestClassification:
    def test_btc_is_crypto(self):
        assert classify("BTC-USD") == AssetClass.CRYPTO

    def test_spy_is_etf_equity(self):
        assert classify("SPY") == AssetClass.ETF_EQUITY

    def test_tlt_is_etf_hedge(self):
        assert classify("TLT") == AssetClass.ETF_HEDGE

    def test_gld_is_etf_hedge(self):
        assert classify("GLD") == AssetClass.ETF_HEDGE

    def test_etf_hedge_never_cut(self):
        """ETF hedges should have sensitivity=0 and floor=1.0."""
        cfg = _CLASS_CONFIG[AssetClass.ETF_HEDGE]
        assert cfg["sensitivity"] == 0.0
        assert cfg["floor"] == 1.0


class TestScaling:
    def test_crypto_scales_down_on_anomaly(self, scorer):
        """BTC should scale down more aggressively than equity on same anomaly score."""
        # _score_to_scale uses sensitivity and floor to compute scale
        btc_scale = scorer._score_to_scale(0.8, AssetClass.CRYPTO)
        spy_scale = scorer._score_to_scale(0.8, AssetClass.ETF_EQUITY)
        assert btc_scale <= spy_scale, "Crypto should scale down more than equity"

    def test_crypto_never_below_floor(self, scorer):
        """Crypto position should never go below floor."""
        scale = scorer._score_to_scale(1.0, AssetClass.CRYPTO)
        floor = _CLASS_CONFIG[AssetClass.CRYPTO]["floor"]
        assert scale >= floor

    def test_equity_never_below_floor(self, scorer):
        scale = scorer._score_to_scale(1.0, AssetClass.EQUITY)
        floor = _CLASS_CONFIG[AssetClass.EQUITY]["floor"]
        assert scale >= floor

    def test_hedge_always_full(self, scorer):
        """ETF hedges should always return 1.0 scale."""
        scale = scorer._score_to_scale(1.0, AssetClass.ETF_HEDGE)
        assert scale == 1.0
