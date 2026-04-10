"""
Hourly Entry Timer
===================
Implements intraday entry timing logic for the live/paper engine.

Concept:
  Instead of executing all orders at market open, this module evaluates
  hourly bars and determines optimal entry points based on:
    - VWAP position (price below VWAP = better entry for longs)
    - Momentum confirmation (short-term positive)
    - Time-of-day rules (12:00 ET for equities, session windows for crypto)
    - Hard fallback at 13:05 ET to avoid missing the day entirely

OOS verdict: NO_EDGE — hourly timing adds minimal value on a daily strategy.
Wired but not critical. Defaults to True (enter now) if disabled or erroring.

Usage:
    timer = HourlyEntryTimer()
    should_enter = timer.should_enter_now(
        symbol="SPY",
        hourly_bars=hourly_df,
        current_time=datetime_now,
    )
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("HourlyEntryTimer")

# Symbols that bypass timing entirely (always enter immediately)
_BYPASS_SYMBOLS = {"GLD", "TLT", "SHY", "AGG", "IEF", "BND"}

# Crypto session window (UTC): only enter BTC/ETH during 14:00-17:00 UTC
# This corresponds to US market open overlap with European close
_CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "BTC/USD", "ETH/USD", "BTCUSD", "ETHUSD",
                   "BTC", "ETH", "SOL-USD", "SOL"}
_CRYPTO_WINDOW_START = time(14, 0)   # 14:00 UTC
_CRYPTO_WINDOW_END   = time(17, 0)   # 17:00 UTC

# Equity timing: prefer 12:00 ET (16:00 UTC in winter, 17:00 UTC in summer)
# Fallback: 13:05 ET regardless of VWAP position
_EQUITY_PREFERRED_HOUR = 12   # noon ET
_EQUITY_FALLBACK_HOUR  = 13   # 1:05pm ET
_EQUITY_FALLBACK_MIN   = 5


class HourlyEntryTimer:
    """
    Determines whether to enter a position NOW based on intraday timing.

    The timer uses a simple VWAP-relative + momentum check for equities
    and a session window + RSI check for crypto.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def should_enter_now(
        self,
        symbol: str,
        hourly_bars: Optional[pd.DataFrame] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Determine whether to execute an entry for the given symbol right now.

        Parameters
        ----------
        symbol       : Instrument symbol
        hourly_bars  : DataFrame with OHLCV hourly data (at least 10 bars)
        current_time : Current datetime (UTC). If None, uses utcnow().

        Returns
        -------
        True if should enter now, False if should wait.
        """
        if not self.enabled:
            return True

        if current_time is None:
            current_time = datetime.utcnow()

        sym_upper = symbol.upper().replace("-", "").replace("/", "")

        # Bypass symbols: always enter immediately
        if symbol.upper() in _BYPASS_SYMBOLS:
            log.debug(f"{symbol}: bypass symbol — enter now")
            return True

        # Crypto path
        if symbol.upper() in _CRYPTO_SYMBOLS or sym_upper in {"BTCUSD", "ETHUSD", "SOLUSD"}:
            return self._crypto_timing(symbol, hourly_bars, current_time)

        # Equity path
        return self._equity_timing(symbol, hourly_bars, current_time)

    def _equity_timing(
        self,
        symbol: str,
        hourly_bars: Optional[pd.DataFrame],
        current_time: datetime,
    ) -> bool:
        """
        Equity entry timing:
          - At 12:00 ET: enter if price below VWAP and momentum positive
          - At 13:05 ET: enter regardless (fallback)
          - Before 12:00 ET: wait
        """
        # Convert UTC to ET (approximate: UTC-4 for EDT, UTC-5 for EST)
        # For simplicity, use UTC-4 (EDT, valid Apr-Nov)
        et_hour = (current_time.hour - 4) % 24

        # Hard fallback: 13:05 ET or later → enter now regardless
        if et_hour > _EQUITY_FALLBACK_HOUR or (
            et_hour == _EQUITY_FALLBACK_HOUR and current_time.minute >= _EQUITY_FALLBACK_MIN
        ):
            log.debug(f"{symbol}: fallback time reached ({et_hour}:{current_time.minute:02d} ET) — enter now")
            return True

        # Before preferred hour: wait
        if et_hour < _EQUITY_PREFERRED_HOUR:
            log.debug(f"{symbol}: too early ({et_hour}:00 ET < 12:00 ET) — wait")
            return False

        # At preferred hour (12:00 ET): check VWAP and momentum
        if hourly_bars is None or len(hourly_bars) < 5:
            log.debug(f"{symbol}: no hourly bars available — enter now (default)")
            return True

        try:
            bars = hourly_bars.copy()
            if isinstance(bars.columns, pd.MultiIndex):
                bars.columns = [c[0] for c in bars.columns]
            bars.columns = [c.capitalize() for c in bars.columns]

            close = bars["Close"].iloc[-1]
            volume = bars["Volume"] if "Volume" in bars.columns else None

            # Compute VWAP from hourly bars
            if volume is not None and volume.sum() > 0:
                typical_price = (bars["High"] + bars["Low"] + bars["Close"]) / 3
                vwap = (typical_price * volume).cumsum() / volume.cumsum()
                current_vwap = vwap.iloc[-1]
                below_vwap = close < current_vwap
            else:
                below_vwap = True  # no volume data → assume OK

            # Short-term momentum: last 3 bars rising
            if len(bars) >= 3:
                momentum_positive = bars["Close"].iloc[-1] > bars["Close"].iloc[-3]
            else:
                momentum_positive = True

            should_enter = below_vwap and momentum_positive
            log.debug(
                f"{symbol}: VWAP check — below_vwap={below_vwap}, "
                f"momentum={momentum_positive} → enter={should_enter}"
            )
            return should_enter

        except Exception as e:
            log.debug(f"{symbol}: timing check failed ({e}) — enter now (default)")
            return True

    def _crypto_timing(
        self,
        symbol: str,
        hourly_bars: Optional[pd.DataFrame],
        current_time: datetime,
    ) -> bool:
        """
        Crypto entry timing:
          - Only enter during 14:00-17:00 UTC window
          - Within window: enter if RSI < 45 (slightly oversold)
        """
        current_utc_time = current_time.time()

        # Outside session window: wait
        if current_utc_time < _CRYPTO_WINDOW_START or current_utc_time >= _CRYPTO_WINDOW_END:
            log.debug(
                f"{symbol}: outside crypto window "
                f"({current_utc_time} not in {_CRYPTO_WINDOW_START}-{_CRYPTO_WINDOW_END}) — wait"
            )
            return False

        # Inside window: check RSI
        if hourly_bars is None or len(hourly_bars) < 14:
            log.debug(f"{symbol}: inside crypto window, no bars — enter now")
            return True

        try:
            bars = hourly_bars.copy()
            if isinstance(bars.columns, pd.MultiIndex):
                bars.columns = [c[0] for c in bars.columns]
            bars.columns = [c.capitalize() for c in bars.columns]

            close = bars["Close"]
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]

            should_enter = rsi < 45
            log.debug(f"{symbol}: crypto RSI={rsi:.1f} → enter={should_enter}")
            return should_enter

        except Exception as e:
            log.debug(f"{symbol}: crypto timing failed ({e}) — enter now")
            return True
