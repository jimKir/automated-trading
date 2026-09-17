"""Unit tests for the execution-guard tradeable universe classifier.

Covers:
  - classify() for all five AssetClass values
  - crypto variant normalisation (BTC/USD, BTCUSD, BTC_USD, BTC-USDT, ...)
  - is_tradeable / filter_tradeable static helpers
  - TradeableUniverse.check with config overrides (allowed_crypto, block_futures)
  - Alpaca runtime verification (tradable / shortable / easy_to_borrow, caching)
  - DynamicUniverseSelector top-N skips phantom instruments and fills the
    slot with the next-ranked tradeable name
"""

import numpy as np
import pandas as pd
import pytest

from execution.tradeable_universe import (
    ALPACA_CRYPTO_TRADEABLE,
    KNOWN_FUTURES,
    AssetClass,
    TradeableUniverse,
    classification_reason,
    classify,
    filter_tradeable,
    is_tradeable,
    normalize_crypto_symbol,
)

# ---------------------------------------------------------------------------
# classify() — all five classes
# ---------------------------------------------------------------------------


class TestClassify:
    @pytest.mark.parametrize("sym", ["ES=F", "NQ=F", "GC=F", "CL=F", "SI=F", "ZB=F", "NG=F"])
    def test_futures_suffix(self, sym):
        assert classify(sym) is AssetClass.FUTURES
        assert sym in KNOWN_FUTURES

    @pytest.mark.parametrize("sym", ["MNQ=F", "RTY=F"])  # not in KNOWN list
    def test_unknown_futures_suffix_still_futures(self, sym):
        assert classify(sym) is AssetClass.FUTURES

    @pytest.mark.parametrize("sym", ["SPY", "QQQ", "IWM", "AAPL", "BRK-B", "BF.B", "VNQ"])
    def test_equities_and_etfs(self, sym):
        assert classify(sym) is AssetClass.EQUITY_ETF

    @pytest.mark.parametrize(
        "sym",
        [
            "BTC-USD",
            "BTC/USD",
            "BTCUSD",
            "BTC_USD",
            "BTC-USDT",
            "btc-usd",
            "ETH-USD",
            "ETH/USD",
            "ETHUSD",
            "SOL-USD",
            "SOL/USD",
            "SOLUSD",
        ],
    )
    def test_alpaca_crypto_variants(self, sym):
        assert classify(sym) is AssetClass.CRYPTO_ALPACA

    @pytest.mark.parametrize(
        "sym",
        [
            "BNB-USD",
            "ADA-USD",
            "AVAX-USD",
            "DOT-USD",
            "LINK-USD",
            "BNB/USD",
            "ADAUSD",
            "AVAX_USD",
            "DOGE-USD",
            "XRPUSDT",
        ],
    )
    def test_unsupported_crypto(self, sym):
        assert classify(sym) is AssetClass.CRYPTO_UNSUPPORTED

    @pytest.mark.parametrize("sym", ["^VIX", "^GSPC", "", "   ", "ES=F  X", "DOW1234567*"])
    def test_unknown(self, sym):
        assert classify(sym) is AssetClass.UNKNOWN


class TestNormalizeCrypto:
    def test_canonical(self):
        assert normalize_crypto_symbol("BTC-USD") == "BTC-USD"
        assert normalize_crypto_symbol("BTC/USD") == "BTC-USD"
        assert normalize_crypto_symbol("BTCUSD") == "BTC-USD"
        assert normalize_crypto_symbol("BTC_USD") == "BTC-USD"
        assert normalize_crypto_symbol("BTC-USDT") == "BTC-USD"
        assert normalize_crypto_symbol("BTCUSDT") == "BTC-USD"
        assert normalize_crypto_symbol("ethusd") == "ETH-USD"

    def test_non_crypto(self):
        assert normalize_crypto_symbol("SPY") is None
        assert normalize_crypto_symbol("ES=F") is None
        assert normalize_crypto_symbol("USD") is None
        assert normalize_crypto_symbol(None) is None


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    def test_is_tradeable(self):
        assert is_tradeable("SPY")
        assert is_tradeable("BTC-USD")
        assert not is_tradeable("ES=F")
        assert not is_tradeable("BNB-USD")
        assert not is_tradeable("^VIX")

    def test_filter_tradeable_preserves_order(self):
        cands = ["ES=F", "SPY", "BNB-USD", "BTC-USD", "NQ=F", "QQQ"]
        assert filter_tradeable(cands) == ["SPY", "BTC-USD", "QQQ"]

    def test_reason_strings(self):
        assert "futures" in classification_reason("ES=F")
        assert "not supported" in classification_reason("BNB-USD")
        assert classification_reason("SPY") == "tradeable"


# ---------------------------------------------------------------------------
# TradeableUniverse — config-driven behaviour
# ---------------------------------------------------------------------------


class TestTradeableUniverseConfig:
    def test_defaults_enabled_no_config(self):
        g = TradeableUniverse({})
        assert g.enabled is True
        assert g.allowed_crypto == ALPACA_CRYPTO_TRADEABLE

    def test_allowed_crypto_override_blocks_sol(self):
        g = TradeableUniverse({"execution_guards": {"allowed_crypto": ["BTC-USD", "ETH-USD"]}})
        assert g.is_tradeable("BTC-USD")
        ok, reason = g.check("SOL-USD")
        assert not ok
        assert "allowed_crypto" in reason

    def test_disabled_guards_use_legacy_suffix_filter(self):
        g = TradeableUniverse({"execution_guards": {"enabled": False}})
        assert g.enabled is False
        assert not g.is_tradeable("ES=F")
        assert not g.is_tradeable("^VIX")
        assert g.is_tradeable("BNB-USD")  # legacy filter passes it (matches old code)

    def test_block_futures_disabled_passthrough(self):
        g = TradeableUniverse({"execution_guards": {"block_futures": False}})
        assert g.is_tradeable("ES=F")

    def test_filter_logs_and_skips(self, caplog):
        g = TradeableUniverse({})
        with caplog.at_level("INFO"):
            out = g.filter_tradeable(["SPY", "ES=F", "BNB-USD", "BTC-USD"])
        assert out == ["SPY", "BTC-USD"]
        text = caplog.text
        assert "[GUARD] Skipping ES=F" in text
        assert "[GUARD] Skipping BNB-USD" in text


# ---------------------------------------------------------------------------
# TradeableUniverse — Alpaca runtime verification
# ---------------------------------------------------------------------------


class _FakeAsset:
    def __init__(self, tradable=True, shortable=True, easy_to_borrow=True):
        self.tradable = tradable
        self.shortable = shortable
        self.easy_to_borrow = easy_to_borrow


class _FakeClient:
    def __init__(self, assets: dict[str, _FakeAsset]):
        self.assets = assets
        self.calls = []

    def get_asset(self, sym):
        self.calls.append(sym)
        if sym not in self.assets:
            raise KeyError(f"asset not found: {sym}")
        return self.assets[sym]


class TestAlpacaVerification:
    def test_not_tradable_equity_blocked(self):
        client = _FakeClient({"HALT": _FakeAsset(tradable=False)})
        g = TradeableUniverse({}, trading_client=client)
        ok, reason = g.check("HALT")
        assert not ok
        assert "not tradable" in reason

    def test_shortable_requires_short_and_etb(self):
        client = _FakeClient(
            {
                "SPY": _FakeAsset(shortable=True, easy_to_borrow=True),
                "HTB": _FakeAsset(shortable=True, easy_to_borrow=False),
                "NS": _FakeAsset(shortable=False, easy_to_borrow=False),
            }
        )
        g = TradeableUniverse({}, trading_client=client)
        assert g.is_shortable("SPY")
        assert not g.is_shortable("HTB")
        assert not g.is_shortable("NS")

    def test_crypto_never_shortable(self):
        client = _FakeClient({"BTC/USD": _FakeAsset(shortable=True, easy_to_borrow=True)})
        g = TradeableUniverse({}, trading_client=client)
        assert not g.is_shortable("BTC-USD")
        assert not g.is_shortable("BNB-USD")
        # crypto classified unsupported must not even hit the client
        assert "BNB/USD" not in client.calls

    def test_crypto_symbol_mapping_slash_format(self):
        client = _FakeClient({"BTC/USD": _FakeAsset()})
        g = TradeableUniverse({}, trading_client=client)
        assert g.is_tradeable("BTC-USD")
        assert "BTC/USD" in client.calls

    def test_cache_per_session(self):
        client = _FakeClient({"SPY": _FakeAsset()})
        g = TradeableUniverse({}, trading_client=client)
        g.is_shortable("SPY")
        g.is_shortable("SPY")
        g.check("SPY")
        assert client.calls.count("SPY") == 1

    def test_lookup_failure_cached_and_fallback(self):
        client = _FakeClient({})  # everything raises
        g = TradeableUniverse({}, trading_client=client)
        assert g.is_tradeable("SPY")  # static fallback
        assert g.is_shortable("SPY")  # offline fallback: assume liquid ETF shortable
        g.is_shortable("SPY")
        assert client.calls.count("SPY") == 1  # failure cached

    def test_verify_with_alpaca_false_uses_static(self):
        client = _FakeClient({"SPY": _FakeAsset(tradable=False)})
        g = TradeableUniverse(
            {"execution_guards": {"verify_with_alpaca": False}}, trading_client=client
        )
        assert g.is_tradeable("SPY")
        assert client.calls == []


# ---------------------------------------------------------------------------
# DynamicUniverseSelector integration — guard skips phantom top ranks
# ---------------------------------------------------------------------------


def _synthetic_data(symbols, drifts, n=320):
    idx = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    data = {}
    for sym, drift in zip(symbols, drifts):
        rng = np.random.default_rng(42)
        rets = drift + rng.normal(0, 0.005, n)
        close = 100 * np.cumprod(1 + rets)
        data[sym] = pd.DataFrame({"Close": close}, index=idx)
    return data


def _selector_config(top_n=3, guards_on=True):
    return {
        "execution_guards": {"enabled": guards_on},
        "dynamic_universe": {
            "enabled": True,
            "top_n": top_n,
            "momentum_window": 63,
            "min_history_days": 252,
            "adaptive_caps": False,
            "candidates": {
                "equities": ["SPY", "QQQ", "IWM", "TLT", "GLD", "SHY"],
                "futures": ["ES=F"],
                "crypto": ["BTC-USD", "BNB-USD"],
            },
        },
    }


class TestSelectorGuard:
    def test_guard_skips_futures_and_phantom_crypto_in_top_n(self):
        from strategy.universe import DynamicUniverseSelector

        syms = ["ES=F", "BNB-USD", "SPY", "QQQ", "IWM", "TLT", "GLD", "SHY", "BTC-USD"]
        # Futures + phantom crypto get the highest drift → top-ranked
        drifts = [0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0005, 0.0001]
        data = _synthetic_data(syms, drifts)
        sel = DynamicUniverseSelector(_selector_config(top_n=3, guards_on=True))
        picked = sel.select(data, data["SPY"].index[-1])
        assert len(picked) == 3
        assert "ES=F" not in picked
        assert "BNB-USD" not in picked
        # Slots filled by next-ranked tradeable names
        assert "SPY" in picked

    def test_guard_off_preserves_legacy_selection(self):
        from strategy.universe import DynamicUniverseSelector

        syms = ["ES=F", "SPY", "QQQ", "IWM", "TLT", "GLD", "SHY", "BTC-USD", "BNB-USD"]
        drifts = [0.004, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.0005, 0.0001, 0.0]
        data = _synthetic_data(syms, drifts)
        sel = DynamicUniverseSelector(_selector_config(top_n=4, guards_on=False))
        picked = sel.select(data, data["SPY"].index[-1])
        # Without the guard the top-ranked futures contract must appear
        assert "ES=F" in picked
