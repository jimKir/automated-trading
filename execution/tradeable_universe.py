"""
Tradeable Universe Classifier
=============================
Decides which candidate symbols Alpaca can actually execute, so that
data-feed symbols used only for signal/regime generation never reach the
order boundary as phantom trades.

Asset classes
-------------
  EQUITY_ETF          — US equities/ETFs; tradeable long AND short
                        (shortable / easy_to_borrow verified against the
                        Alpaca assets endpoint at runtime, cached per session).
  CRYPTO_ALPACA       — crypto Alpaca supports: BTC, ETH, SOL (USD pairs).
                        Long-only (no crypto shorts).
  CRYPTO_UNSUPPORTED  — any other crypto (BNB, ADA, AVAX, DOT, LINK, ...).
                        NOT tradeable on Alpaca.
  FUTURES             — *=F tickers (ES=F, NQ=F, GC=F, CL=F, SI=F, ZB=F, NG=F).
                        Alpaca has no futures. NEVER tradeable — but these stay
                        in the data feed for signal/regime generation.
  UNKNOWN             — anything else (^VIX indices, unrecognised patterns).
                        Treated as NOT tradeable (fail-safe).

Static helpers (is_tradeable / classify / filter_tradeable) need no network.
The TradeableUniverse class adds optional runtime verification via a
connected alpaca TradingClient, degrading gracefully to the static rules
when no client/credentials are available.
"""

from __future__ import annotations

import re
from enum import Enum

from utils.logger import get_logger

log = get_logger("TradeableUniverse")


class AssetClass(Enum):
    EQUITY_ETF = "equity_etf"
    CRYPTO_ALPACA = "crypto_alpaca"
    CRYPTO_UNSUPPORTED = "crypto_unsupported"
    FUTURES = "futures"
    UNKNOWN = "unknown"


# Crypto Alpaca actually supports for paper trading (long only).
ALPACA_CRYPTO_TRADEABLE: set[str] = {"BTC-USD", "ETH-USD", "SOL-USD"}

# Known CME/ICE futures tickers seen in data feeds — kept explicit so new
# futures are caught even before the generic *=F suffix rule runs.
KNOWN_FUTURES: set[str] = {"ES=F", "NQ=F", "GC=F", "CL=F", "SI=F", "ZB=F", "NG=F"}

_FUTURES_SUFFIX = "=F"
_INDEX_PREFIX = "^"  # ^VIX etc. — data only
_CRYPTO_RE = re.compile(r"^([A-Z0-9]{2,12})[-/_]?(USD[T]?)$")
_EQUITY_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]|-[A-Z])?$")


def normalize_crypto_symbol(symbol: str) -> str | None:
    """Normalise crypto variants to the canonical ``BASE-USD`` form.

    Accepts BTC-USD, BTC/USD, BTCUSD, BTC_USD, BTC-USDT, BTCUSDT.
    Returns None if the symbol is not a recognisable crypto pair.
    """
    if not symbol:
        return None
    s = symbol.upper().strip()
    m = _CRYPTO_RE.match(s)
    if not m:
        return None
    base = m.group(1)
    # Guard against plain equities that coincidentally end in USD (e.g. "USD" itself)
    if base in {"USD", "USDT"}:
        return None
    return f"{base}-USD"


DEFAULT_ALLOWED_CRYPTO: list[str] = sorted(ALPACA_CRYPTO_TRADEABLE)


def classify(symbol: str) -> AssetClass:
    """Static classification of a symbol. No network required."""
    if not symbol:
        return AssetClass.UNKNOWN
    s = symbol.strip()

    if s.startswith(_INDEX_PREFIX):
        return AssetClass.UNKNOWN
    if s.endswith(_FUTURES_SUFFIX) or s in KNOWN_FUTURES:
        return AssetClass.FUTURES

    crypto = normalize_crypto_symbol(s)
    if crypto is not None:
        if crypto in ALPACA_CRYPTO_TRADEABLE:
            return AssetClass.CRYPTO_ALPACA
        return AssetClass.CRYPTO_UNSUPPORTED

    if _EQUITY_RE.match(s.upper()):
        return AssetClass.EQUITY_ETF

    return AssetClass.UNKNOWN


_TRADEABLE_CLASSES = {AssetClass.EQUITY_ETF, AssetClass.CRYPTO_ALPACA}


def classification_reason(symbol: str) -> str:
    """Human-readable reason used in [GUARD] log lines and health checks."""
    cls = classify(symbol)
    if cls is AssetClass.FUTURES:
        return "futures contract — Alpaca has no futures (data/signals only)"
    if cls is AssetClass.CRYPTO_UNSUPPORTED:
        return (
            f"crypto pair not supported by Alpaca "
            f"(allowed: {', '.join(sorted(ALPACA_CRYPTO_TRADEABLE))})"
        )
    if cls is AssetClass.UNKNOWN:
        return "unrecognised symbol class (index/data-only or malformed)"
    return "tradeable"


def is_tradeable(symbol: str) -> bool:
    """Static tradeability check (no Alpaca verification).

    True only for EQUITY_ETF and CRYPTO_ALPACA classes.
    """
    return classify(symbol) in _TRADEABLE_CLASSES


def filter_tradeable(candidates: list[str]) -> list[str]:
    """Return the input list with non-tradeable symbols removed, order preserved.

    Pure filter — no logging. Callers that need audit logs should use
    TradeableUniverse.filter_tradeable or iterate with classification_reason.
    """
    return [c for c in candidates if is_tradeable(c)]


class TradeableUniverse:
    """Session-scoped tradeability/shortability guard.

    Wraps the static classifier and (optionally) verifies equity assets
    against the Alpaca ``/v2/assets`` endpoint at runtime:

      - ``tradable``     → gates long entries
      - ``shortable`` and ``easy_to_borrow`` → gate short entries

    Asset lookups are cached for the lifetime of the instance (one session).
    If no client is available or a lookup fails, the guard falls back to the
    static class rules (equities assumed tradeable+shortable, so the bot
    keeps working offline; failures are logged).
    """

    def __init__(self, config: dict | None = None, trading_client=None):
        cfg = (config or {}).get("execution_guards", {})
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.block_futures: bool = bool(cfg.get("block_futures", True))
        allowed = cfg.get("allowed_crypto", DEFAULT_ALLOWED_CRYPTO)
        self.allowed_crypto: set[str] = {
            c for c in (normalize_crypto_symbol(a) for a in allowed) if c
        } or set(ALPACA_CRYPTO_TRADEABLE)
        self.verify_with_alpaca: bool = bool(cfg.get("verify_with_alpaca", True))
        self._client = trading_client
        self._asset_cache: dict[str, dict] = {}  # symbol → attrs (per session)

    # ------------------------------------------------------------------
    # Alpaca asset lookup (cached per session)
    # ------------------------------------------------------------------

    @staticmethod
    def _alpaca_symbol(symbol: str) -> str:
        """Alpaca symbol format: equities plain, crypto as BASE/USD."""
        crypto = normalize_crypto_symbol(symbol)
        if crypto is not None:
            return crypto.replace("-", "/")
        return symbol.upper()

    def _lookup_asset(self, symbol: str) -> dict | None:
        """Fetch asset attrs from Alpaca once per session; None if unavailable."""
        if symbol in self._asset_cache:
            return self._asset_cache[symbol]
        if self._client is None or not self.verify_with_alpaca:
            return None
        try:
            asset = self._client.get_asset(self._alpaca_symbol(symbol))
            attrs = {
                "tradable": bool(getattr(asset, "tradable", False)),
                "shortable": bool(getattr(asset, "shortable", False)),
                "easy_to_borrow": bool(getattr(asset, "easy_to_borrow", False)),
            }
            self._asset_cache[symbol] = attrs
            return attrs
        except Exception as e:
            log.debug(f"[GUARD] Alpaca asset lookup failed for {symbol}: {e}")
            # Cache the failure so we don't hammer the API every cycle
            self._asset_cache[symbol] = None
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, symbol: str) -> tuple[bool, str]:
        """Return (tradeable, reason). Respects config + Alpaca verification."""
        cls = classify(symbol)

        if cls is AssetClass.FUTURES:
            reason = classification_reason(symbol)
            if not self.block_futures and self.enabled:
                return True, f"block_futures disabled — passing through ({reason})"
            return False, reason
        if cls in (AssetClass.CRYPTO_UNSUPPORTED, AssetClass.UNKNOWN):
            return False, classification_reason(symbol)
        if cls is AssetClass.CRYPTO_ALPACA:
            crypto = normalize_crypto_symbol(symbol)
            if crypto not in self.allowed_crypto:
                return False, f"crypto {crypto} not in execution_guards.allowed_crypto"
            asset = self._lookup_asset(symbol)
            if asset is not None and not asset["tradable"]:
                return False, f"crypto {crypto} not tradable on Alpaca account"
            return True, "tradeable"

        # EQUITY_ETF
        asset = self._lookup_asset(symbol)
        if asset is not None and not asset["tradable"]:
            return False, f"{symbol} not tradable on Alpaca account"
        return True, "tradeable"

    def is_tradeable(self, symbol: str) -> bool:
        """Full tradeability check (static classes + Alpaca verification)."""
        if not self.enabled:
            # Guards disabled → preserve legacy behaviour: block futures and
            # index-style symbols only.
            s = symbol.strip()
            return not s.endswith(_FUTURES_SUFFIX) and not s.startswith(_INDEX_PREFIX)
        ok, _reason = self.check(symbol)
        return ok

    def is_shortable(self, symbol: str) -> bool:
        """True only for shortable, easy-to-borrow US equities/ETFs.

        Crypto (even Alpaca-supported) is NEVER shortable through this bot —
        Alpaca crypto is long-only. Returns False when in doubt.
        """
        if classify(symbol) is not AssetClass.EQUITY_ETF:
            return False
        asset = self._lookup_asset(symbol)
        if asset is None:
            # Offline fallback: static class says equity/ETF; the caller must
            # additionally restrict to the liquid easy-to-borrow universe.
            return True
        return asset["tradable"] and asset["shortable"] and asset["easy_to_borrow"]

    def filter_tradeable(self, candidates: list[str], log_skips: bool = True) -> list[str]:
        """Order-preserving filter with explicit [GUARD] audit logging."""
        out = []
        for c in candidates:
            if self.is_tradeable(c):
                out.append(c)
            elif log_skips:
                _ok, reason = self.check(c) if self.enabled else (False, "legacy guard")
                log.info(f"[GUARD] Skipping {c} — not tradeable on Alpaca ({reason})")
        return out
