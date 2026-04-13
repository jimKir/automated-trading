"""Unit tests for HourlyEntryTimer — entry rule validation."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from execution.hourly_entry_timer import HourlyEntryTimer


@pytest.fixture
def timer():
    return HourlyEntryTimer()


@pytest.fixture
def mock_hourly_bars():
    dates = pd.date_range('2024-01-15 09:30', periods=30, freq='1h')
    prices = 400 + np.cumsum(np.random.normal(0, 1, 30))
    return pd.DataFrame({
        'open': prices - 0.5,
        'high': prices + 1,
        'low': prices - 1,
        'close': prices,
        'volume': np.random.randint(500_000, 2_000_000, 30)
    }, index=dates)


class TestBypassSymbols:
    def test_gld_bypassed(self, timer, mock_hourly_bars):
        result = timer.should_enter_now('GLD', pd.Series(), mock_hourly_bars, 1)
        enter = result['enter'] if isinstance(result, dict) else result
        assert enter, "GLD should always bypass timing (no intraday momentum)"

    def test_tlt_bypassed(self, timer, mock_hourly_bars):
        result = timer.should_enter_now('TLT', pd.Series(), mock_hourly_bars, 1)
        enter = result['enter'] if isinstance(result, dict) else result
        assert enter

    def test_shy_bypassed(self, timer, mock_hourly_bars):
        result = timer.should_enter_now('SHY', pd.Series(), mock_hourly_bars, 1)
        enter = result['enter'] if isinstance(result, dict) else result
        assert enter


class TestEquityTimingRules:
    def test_bypass_symbols_list(self, timer):
        assert 'GLD' in timer.BYPASS_SYMBOLS
        assert 'TLT' in timer.BYPASS_SYMBOLS
        assert 'SHY' in timer.BYPASS_SYMBOLS
        assert 'XLU' in timer.BYPASS_SYMBOLS
        assert 'XLP' in timer.BYPASS_SYMBOLS
        # Should NOT contain wrong symbols from old version
        assert 'AGG' not in timer.BYPASS_SYMBOLS
        assert 'IEF' not in timer.BYPASS_SYMBOLS
        assert 'BND' not in timer.BYPASS_SYMBOLS

    def test_fallback_hour_is_13(self, timer):
        assert timer.EQUITY_FALLBACK_HOUR == 13


class TestCryptoTimingRules:
    def test_btc_window(self, timer):
        assert timer.CRYPTO_WINDOWS.get('BTC/USD') == (14, 17) or \
               timer.CRYPTO_WINDOWS.get('BTCUSD') == (14, 17)

    def test_eth_window(self, timer):
        assert timer.CRYPTO_WINDOWS.get('ETH/USD') == (17, 20) or \
               timer.CRYPTO_WINDOWS.get('ETHUSD') == (17, 20)

    def test_crypto_hard_stop(self, timer):
        assert timer.CRYPTO_HARD_STOP_UTC == 20


class TestVWAPComputation:
    def test_vwap_computation(self, timer, mock_hourly_bars):
        vwap = timer.compute_vwap(mock_hourly_bars)
        assert isinstance(vwap, float)
        assert vwap > 0
        # VWAP should be close to mean close price
        assert abs(vwap - mock_hourly_bars['close'].mean()) < mock_hourly_bars['close'].std() * 3
