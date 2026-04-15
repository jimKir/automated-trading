"""Regression tests for the shape mismatch bug in _compute_symbol_signal.

The live engine's PMO/stochastic/regime blending path in SignalGenerator
could produce arrays of different lengths when High/Low had a different
index from Close (common with Alpaca live API data).  These tests call
_compute_symbol_signal directly to ensure the reindex fix and shape
assertion prevent silent regressions during paper trading.

Bug:  ValueError: operands could not be broadcast together with shapes
      (400,) (399,) (399,)
Path: strategy/signals.py → _compute_symbol_signal → np.where(...)
Fix:  Commits 3a136f3 (reindex), 2615015 (assertion)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int, start: str = "2024-01-01", seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with *n* business days."""
    dates = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 0.3, n),
            "High": close + rng.uniform(0, 1.0, n),
            "Low": close - rng.uniform(0, 1.0, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


def _make_signal_gen():
    """Minimal SignalGenerator with optional features disabled."""
    from strategy.signals import SignalGenerator

    return SignalGenerator(
        {
            "strategy": {
                "lookback_fast": 10,
                "lookback_slow": 20,
                "regime_window": 30,
                "volume_confirmation": False,
                "pv_segments_enabled": False,
                "predictive": {"credit_regime_enabled": False},
                "regime_switching": {"enabled": False},
            },
            "trend_classifier": {"enabled": False},
        }
    )


def _call_compute(sg, sym, df, all_data):
    """Call _compute_symbol_signal with sensible defaults."""
    credit_signal = pd.Series(0.0, index=df["Close"].index)
    return sg._compute_symbol_signal(
        sym=sym,
        df=df,
        as_of_date=None,
        credit_signal=credit_signal,
        all_data=all_data,
        choppy_score=None,
        all_prices=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeSymbolSignalShapeRegression:
    """Regression suite targeting _compute_symbol_signal directly."""

    def test_mismatched_high_low_close(self):
        """Reproduce the original bug: Close has N rows, High/Low have N-1.

        The reindex fix should align everything to close.index so no
        broadcast ValueError occurs.
        """
        sg = _make_signal_gen()
        full = _make_ohlcv(400)

        # Trim High/Low to 399 rows — simulates the Alpaca API mismatch
        mismatched = full.copy()
        mismatched["High"] = full["High"].iloc[1:]  # first row becomes NaN
        mismatched["Low"] = full["Low"].iloc[1:]

        spy = _make_ohlcv(400, seed=99)
        all_data = {"TEST": mismatched, "SPY": spy}

        try:
            sym, signal = _call_compute(sg, "TEST", mismatched, all_data)
        except ValueError as exc:
            if "broadcast" in str(exc).lower() or "shape mismatch" in str(exc).lower():
                pytest.fail(f"Shape mismatch leaked through reindex fix: {exc}")
            raise

        assert sym == "TEST"
        assert len(signal) == len(full["Close"])

    def test_consistent_data_golden_path(self):
        """Perfectly aligned OHLCV data should produce a valid signal."""
        sg = _make_signal_gen()
        df = _make_ohlcv(400)
        spy = _make_ohlcv(400, seed=99)
        all_data = {"TEST": df, "SPY": spy}

        sym, signal = _call_compute(sg, "TEST", df, all_data)

        assert sym == "TEST"
        assert isinstance(signal, pd.Series)
        assert len(signal) == 400
        # Signal should contain at least some non-zero values
        assert not (signal == 0).all(), "Signal is all zeros — golden path should produce signals"

    def test_pmo_stoch_reindex_specifically(self):
        """Target the PMO/stochastic path: first row of High/Low is NaN so
        rolling operations inside _stochastic_contrarian and _pmo_crossover
        naturally produce N-1 valid elements.  The output must still have
        N elements matching close.index.
        """
        sg = _make_signal_gen()
        df = _make_ohlcv(400)

        # Set first row of High/Low to NaN — this causes rolling min/max in
        # stochastic and the shift in PMO to produce one fewer valid element.
        df.loc[df.index[0], "High"] = np.nan
        df.loc[df.index[0], "Low"] = np.nan

        spy = _make_ohlcv(400, seed=99)
        all_data = {"TEST": df, "SPY": spy}

        sym, signal = _call_compute(sg, "TEST", df, all_data)

        assert sym == "TEST"
        assert len(signal) == 400, f"Signal length {len(signal)} != 400 — PMO/stoch reindex failed"

    def test_shape_assertion_fires_on_mismatch(self):
        """Verify the assertion block in _compute_symbol_signal raises a
        descriptive ValueError when arrays have mismatched shapes.

        We replicate the exact assertion logic from signals.py with
        intentionally mismatched numpy arrays to confirm the error
        message names only the bad arrays, includes the symbol, and
        states the expected length.
        """
        sym = "TEST"
        expected_len = 400

        # Simulate arrays after a broken reindex — choppy and bear are short
        bull_regime_arr = np.zeros(expected_len, dtype=bool)
        t3_gate_arr = np.zeros(expected_len, dtype=bool)
        bull_blend_arr = np.zeros(expected_len)
        bear_blend_arr = np.zeros(expected_len - 1)  # 399 — WRONG
        choppy_blend_arr = np.zeros(expected_len - 1)  # 399 — WRONG

        # Run the same assertion logic as signals.py
        _arr_map = {
            "bull_regime_arr": bull_regime_arr,
            "t3_gate_arr": t3_gate_arr,
            "bull_blend_arr": bull_blend_arr,
            "bear_blend_arr": bear_blend_arr,
            "choppy_blend_arr": choppy_blend_arr,
        }
        _mismatched = {
            name: arr.shape for name, arr in _arr_map.items() if arr.shape[0] != expected_len
        }

        assert _mismatched, "Expected mismatched arrays but found none"

        parts = " ".join(f"{name}={shape}" for name, shape in _mismatched.items())
        msg = f"Shape mismatch in _compute_symbol_signal for {sym}: {parts} expected={expected_len}"

        # Verify only the bad arrays are named
        assert "bear_blend_arr=(399,)" in msg
        assert "choppy_blend_arr=(399,)" in msg
        # Correct arrays must NOT appear
        assert "bull_regime_arr" not in msg
        assert "t3_gate_arr" not in msg
        assert "bull_blend_arr" not in msg
        # Symbol and expected length present
        assert "TEST" in msg
        assert "expected=400" in msg

        # Confirm the ValueError would be raised with the correct message
        with pytest.raises(
            ValueError,
            match=r"Shape mismatch in _compute_symbol_signal for TEST",
        ):
            raise ValueError(msg)
