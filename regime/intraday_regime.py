"""
Layer E — Intraday Regime Score
================================
Classifies the current intraday market regime and converts it to a stress
score in [0.0, 1.0] for use in the Early Warning System (EWS).

Regime → EWS stress mapping
----------------------------
  TRENDING_UP    →  0.0   (no stress — momentum working)
  TRENDING_DOWN  →  0.3   (mild stress — momentum may reverse)
  TRANSITIONING  →  0.4   (moderate stress)
  CHOPPY         →  0.7   (high stress — momentum unreliable)
  HIGH_FEAR      →  0.9   (acute stress)
  ELEVATED_VOL   →  0.5   (elevated vol but not extreme)
  UNKNOWN        →  0.0   (insufficient data — assume benign)

Detection logic is identical to RegimeDetector.detect() in
momentum_signals_exploration/scanner_v2.py:
  • SPY price vs 20-bar EMA
  • ADX (14-bar directional movement index)
  • 5-bar net return (drift direction)
  • VIX level threshold check

Config key (in main config dict):
  ews:
    use_intraday: true   # toggle Layer E on/off
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("EWS.LayerE")

# ── Regime label → EWS stress score ────────────────────────────────────────
REGIME_SCORES: dict[str, float] = {
    "TRENDING_UP":   0.0,
    "TRENDING_DOWN": 0.3,
    "TRANSITIONING": 0.4,
    "ELEVATED_VOL":  0.5,
    "CHOPPY":        0.7,
    "HIGH_FEAR":     0.9,
    "UNKNOWN":       0.0,
}

# Default lookback for SPY bars when fetching live
_LIVE_LOOKBACK_DAYS = 5
# Minimum number of hourly bars required before the detector fires
_MIN_BARS = 6


class IntradayRegimeScorer:
    """
    Layer E of the Early Warning System.

    Detects the intraday market regime from SPY hourly bars + VIX level and
    returns a normalised stress score in [0.0, 1.0].

    Parameters
    ----------
    None — all data fetching is handled internally via yfinance or the
    caller can inject pre-fetched bars.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_today(
        self,
        spy_bars: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None,
    ) -> float:
        """
        Compute the Layer-E stress score for today.

        Parameters
        ----------
        spy_bars : pd.DataFrame, optional
            Hourly OHLCV bars for SPY (columns: open, high, low, close, volume).
            If omitted, the last ``_LIVE_LOOKBACK_DAYS`` days are fetched from
            yfinance automatically.
        vix_level : float, optional
            Current VIX level.  If omitted, fetched from yfinance (^VIX).

        Returns
        -------
        float
            Stress score in [0.0, 1.0].  0.0 = no stress, 1.0 = acute stress.
            Returns 0.0 on any error to avoid false positives that would
            incorrectly gate the strategy.
        """
        try:
            bars = self._ensure_spy_bars(spy_bars)
            vix  = self._ensure_vix(vix_level)
            regime = self._detect_regime(bars, vix)
            score  = REGIME_SCORES.get(regime, 0.0)
            log.info(
                f"EWS Layer E | regime={regime} vix={vix:.1f} "
                f"bars={len(bars)} → score={score:.2f}"
            )
            return float(score)
        except Exception as exc:
            log.warning(f"EWS Layer E score_today failed ({exc}); returning 0.0")
            return 0.0

    def compute_series(
        self,
        start: str,
        end:   str,
        spy_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute a daily regime-stress series for backtesting.

        For each business day in [start, end] the method classifies the
        regime using the hourly SPY bars available *up to and including* that
        day (walk-forward — no look-ahead).

        Parameters
        ----------
        start : str
            Backtest start date, e.g. "2022-01-01".
        end : str
            Backtest end date, e.g. "2024-12-31".
        spy_data : pd.DataFrame, optional
            Pre-fetched daily SPY data (columns: Open/High/Low/Close/Volume or
            open/high/low/close/volume).  If omitted, yfinance is used to
            download the full range.

        Returns
        -------
        pd.Series
            Daily series indexed by date, values in [0.0, 1.0].
            Any day that cannot be computed is filled with 0.0.
        """
        try:
            biz_days = pd.date_range(start, end, freq="B")
            if biz_days.empty:
                return pd.Series(dtype=float)

            prices = self._load_daily_spy(start, end, spy_data)

            # Also grab VIX daily close for the period
            vix_series = self._load_daily_vix(start, end)

            scores = {}
            for day in biz_days:
                try:
                    day_str = day.strftime("%Y-%m-%d")
                    hist = prices.loc[:day]
                    if len(hist) < _MIN_BARS:
                        scores[day] = 0.0
                        continue

                    vix_val = float(
                        vix_series.asof(day)
                        if not vix_series.empty
                        else 18.0
                    )
                    if np.isnan(vix_val):
                        vix_val = 18.0

                    regime = self._detect_regime(hist, vix_val)
                    scores[day] = REGIME_SCORES.get(regime, 0.0)
                except Exception as day_exc:
                    log.debug(f"EWS Layer E skipping {day_str}: {day_exc}")
                    scores[day] = 0.0

            series = pd.Series(scores)
            series.index = pd.DatetimeIndex(series.index)
            log.info(
                f"EWS Layer E compute_series: {start}→{end}  "
                f"mean={series.mean():.3f}  max={series.max():.3f}"
            )
            return series

        except Exception as exc:
            log.warning(f"EWS Layer E compute_series failed ({exc}); returning zeros")
            idx = pd.date_range(start, end, freq="B")
            return pd.Series(0.0, index=idx)

    # ------------------------------------------------------------------
    # Regime detection — mirrors scanner_v2.RegimeDetector.detect()
    # ------------------------------------------------------------------

    def _detect_regime(
        self,
        bars: pd.DataFrame,
        vix_level: float = 18.0,
    ) -> str:
        """
        Classify market regime from OHLC bars + VIX.

        Replicates the logic from
        ``momentum_signals_exploration/scanner_v2.RegimeDetector.detect()``.

        Parameters
        ----------
        bars : pd.DataFrame
            OHLCV with columns: open, high, low, close (case-insensitive).
            Minimum 6 rows required.
        vix_level : float
            Current VIX reading.

        Returns
        -------
        str
            One of: TRENDING_UP, TRENDING_DOWN, TRANSITIONING, CHOPPY,
            HIGH_FEAR, ELEVATED_VOL, UNKNOWN.
        """
        bars = self._normalise_columns(bars)
        if bars is None or len(bars) < _MIN_BARS:
            return "UNKNOWN"

        closes = bars["close"].astype(float).values
        highs  = bars["high"].astype(float).values
        lows   = bars["low"].astype(float).values

        # 20-bar EMA
        ema20    = self._ema(closes, 20)
        above_ma = bool(closes[-1] > ema20)

        # ADX (14-bar directional movement)
        adx      = self._adx(highs, lows, closes, 14)
        trending = bool(adx > 20)

        # 5-bar drift
        drift_5 = float(
            (closes[-1] - closes[-5]) / (closes[-5] + 1e-9)
        ) if len(closes) >= 5 else 0.0

        # Classify — same thresholds as scanner_v2
        if vix_level > 35:
            return "HIGH_FEAR"
        elif vix_level > 28:
            return "ELEVATED_VOL"
        elif trending and above_ma and drift_5 > 0:
            return "TRENDING_UP"
        elif trending and (not above_ma) and drift_5 < 0:
            return "TRENDING_DOWN"
        elif not trending:
            return "CHOPPY"
        else:
            return "TRANSITIONING"

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _ensure_spy_bars(
        self, spy_bars: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Return provided bars or fetch from yfinance."""
        if spy_bars is not None and not spy_bars.empty:
            return spy_bars
        return self._fetch_yfinance(
            "SPY",
            (datetime.utcnow() - timedelta(days=_LIVE_LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
            datetime.utcnow().strftime("%Y-%m-%d"),
            interval="1h",
        )

    def _ensure_vix(self, vix_level: Optional[float]) -> float:
        """Return provided VIX or fetch latest reading from yfinance."""
        if vix_level is not None:
            return float(vix_level)
        try:
            import yfinance as yf
            vix_df = yf.Ticker("^VIX").history(period="5d", interval="1d")
            if not vix_df.empty:
                return float(vix_df["Close"].iloc[-1])
        except Exception as exc:
            log.debug(f"EWS Layer E: VIX fetch failed ({exc}); using default 18.0")
        return 18.0

    def _load_daily_spy(
        self,
        start: str,
        end:   str,
        spy_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Return daily SPY OHLCV for the range, normalised to lowercase cols."""
        if spy_data is not None and not spy_data.empty:
            df = self._normalise_columns(spy_data.copy())
            if df is not None:
                # Ensure DatetimeIndex is tz-naive for .loc slicing
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df

        # Extend window by 60 days to ensure enough bars at start
        fetch_start = (
            pd.Timestamp(start) - timedelta(days=60)
        ).strftime("%Y-%m-%d")
        df = self._fetch_yfinance("SPY", fetch_start, end, interval="1d")
        if df is not None and not df.empty:
            return df

        return pd.DataFrame()

    def _load_daily_vix(self, start: str, end: str) -> pd.Series:
        """Return daily VIX close as a pd.Series indexed by date."""
        try:
            fetch_start = (
                pd.Timestamp(start) - timedelta(days=60)
            ).strftime("%Y-%m-%d")
            df = self._fetch_yfinance("^VIX", fetch_start, end, interval="1d")
            if df is not None and not df.empty:
                return df["close"].astype(float)
        except Exception as exc:
            log.debug(f"EWS Layer E: VIX series fetch failed: {exc}")
        return pd.Series(dtype=float)

    @staticmethod
    def _fetch_yfinance(
        ticker: str,
        start:  str,
        end:    str,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV from yfinance and normalise column names to lowercase.
        Returns None on failure.
        """
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval=interval,
            )
            if df is None or df.empty:
                return None
            df = df.rename(columns={
                "Open":   "open",
                "High":   "high",
                "Low":    "low",
                "Close":  "close",
                "Volume": "volume",
            })
            # Strip timezone so comparisons with naive Timestamps work
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df[["open", "high", "low", "close"]].copy()
        except Exception as exc:
            log.debug(f"EWS Layer E: yfinance fetch {ticker} failed: {exc}")
            return None

    @staticmethod
    def _normalise_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Accept DataFrames with either Title-case or lower-case OHLC columns.
        Returns the frame with lower-case column names, or None if required
        columns are missing.
        """
        if df is None or df.empty:
            return None
        rename = {c: c.lower() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=rename)
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                return None
        return df

    # ------------------------------------------------------------------
    # Statistical helpers (identical to scanner_v2.RegimeDetector)
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> float:
        """Exponential moving average — last value."""
        if len(arr) < 1:
            return 0.0
        if len(arr) < period:
            return float(np.mean(arr))
        k = 2.0 / (period + 1)
        v = float(arr[0])
        for x in arr[1:]:
            v = float(x) * k + v * (1 - k)
        return v

    @staticmethod
    def _adx(
        highs:  np.ndarray,
        lows:   np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """
        Simplified Average Directional Index.

        Returns 25.0 (trending) when there is insufficient history so that a
        neutral default does not erroneously classify the market as CHOPPY.
        """
        n = len(highs)
        if n < period + 2:
            return 25.0

        tr_list, dm_p, dm_m = [], [], []
        for i in range(1, n):
            hl  = highs[i]  - lows[i]
            hpc = abs(highs[i]  - closes[i - 1])
            lpc = abs(lows[i]   - closes[i - 1])
            tr_list.append(max(hl, hpc, lpc))

            up   = highs[i]  - highs[i - 1]
            down = lows[i - 1] - lows[i]
            dm_p.append(up   if (up   > down and up   > 0) else 0.0)
            dm_m.append(down if (down > up   and down > 0) else 0.0)

        atr = float(np.mean(tr_list[-period:]))
        if atr == 0:
            return 0.0
        di_p  = float(np.mean(dm_p[-period:])) / atr
        di_m  = float(np.mean(dm_m[-period:])) / atr
        denom = di_p + di_m
        return 0.0 if denom == 0 else abs(di_p - di_m) / denom * 100
