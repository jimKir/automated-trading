#!/usr/bin/env python3
"""
Crypto Data Fetcher using yfinance
===================================
Fallback for crypto pairs not available via Alpaca.
Provides BTC-USD, ETH-USD, SOL-USD and other major crypto.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger("crypto_fetcher")

# Crypto symbols available via yfinance
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]


class YFinanceCryptoFetcher:
    """Fetch crypto OHLCV data via yfinance."""

    def __init__(self):
        logger.info("YFinance crypto fetcher initialized")

    def fetch_bars(self, sym: str, start: str, end: str) -> pd.DataFrame | None:
        """
        Fetch daily OHLCV bars for a crypto symbol.

        Args:
            sym: Symbol (e.g., "BTC-USD", "ETH-USD")
            start: Start date YYYY-MM-DD
            end: End date YYYY-MM-DD

        Returns:
            DataFrame with OHLCV data, or None if fetch fails
        """
        try:
            # yfinance handles BTC-USD, ETH-USD, SOL-USD natively
            df = yf.download(sym, start=start, end=end, progress=False)

            if df is None or df.empty:
                logger.debug(f"{sym}: No data fetched from yfinance")
                return None

            # Handle MultiIndex columns from yfinance >= 0.2.40
            # (returns MultiIndex like ("Open", "BTC-USD") for single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # yfinance returns UTC times, ensure consistent timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            logger.debug(f"{sym}: Fetched {len(df)} bars ({start} to {end})")
            return df

        except Exception as ex:
            logger.debug(f"{sym}: yfinance error - {type(ex).__name__}: {ex}")
            return None

    def is_available(self, sym: str) -> bool:
        """Check if symbol is available via yfinance."""
        return sym in CRYPTO_SYMBOLS
