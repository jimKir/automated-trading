#!/usr/bin/env python3
"""
scanner_v2.py — Evidence-based Signal Engine
=============================================

Factor validation status (OOS 2023-2026, IC analysis):

VALIDATED SIGNALS (in production):
  - VWAP intraday deviation (1-min bars): IC +0.054, p=0.007 ✅
  - PMO crossover (contrarian, flip sign): IC -0.032, p=0.005 ✅
  - Databento closing imbalance (real): pending full validation
  - Volume surprise: IC -0.018, p=0.053 ⚠️ weak, reduced weight

VALIDATED FILTERS:
  - ADX(14) > 20: Sharpe +0.71 OOS, 4/4 positive walk-forward years ✅

REMOVED (zero IC):
  - Relative strength: IC +0.001, p=0.963 ❌
  - Daily VWAP proxy: IC -0.020, p=0.081 ❌
  - Imbalance price proxy: IC +0.011, p=0.255 ❌
  - RSI, MACD, Stochastic, TS Momentum, CS Momentum: all insignificant ❌
  - XGBoost ML: IC -0.009 ❌
  - LSTM: IC -0.000 ❌

Architecture:
  1. Batch data fetch   — 1 API call for all symbols (not 50 sequential calls)
  2. Z-score normalise  — cross-sectional, so scores are comparable
  3. Regime detection   — ADX + SPY trend; skip scan on choppy days
  4. ADX position gate  — ADX < 20 halves position size (walk-forward validated)
  5. Sector limits      — max 3 signals per sector to avoid concentration
  6. Composite score    — one number per symbol, directly actionable

SDK: uses alpaca-py (alpaca.trading / alpaca.data) — NOT the deprecated
alpaca_trade_api package.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_closing_imbalance_signal = None  # module-level singleton to avoid per-symbol init

# ---------------------------------------------------------------------------
# Sector map — used for concentration limits and sector-relative strength
# ---------------------------------------------------------------------------
SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","GOOGL":"Tech","GOOG":"Tech",
    "META":"Tech","AVGO":"Tech","AMD":"Tech","INTC":"Tech","QCOM":"Tech",
    "CRM":"Tech","ADBE":"Tech","TXN":"Tech","AMAT":"Tech","MU":"Tech",
    "LRCX":"Tech","PANW":"Tech","INTU":"Tech","NOW":"Tech","ORCL":"Tech",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","SLB":"Energy",
    "MPC":"Energy","PSX":"Energy","OKE":"Energy","VLO":"Energy","DVN":"Energy",
    # Financials
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","GS":"Financials",
    "MS":"Financials","BLK":"Financials","SCHW":"Financials","C":"Financials",
    "AXP":"Financials","USB":"Financials","PNC":"Financials",
    # Healthcare
    "UNH":"Health","LLY":"Health","JNJ":"Health","MRK":"Health","ABBV":"Health",
    "TMO":"Health","ABT":"Health","DHR":"Health","ISRG":"Health","PFE":"Health",
    "GILD":"Health","REGN":"Health","BMY":"Health","MDT":"Health","CVS":"Health",
    # Consumer Discretionary
    "AMZN":"ConDisc","TSLA":"ConDisc","HD":"ConDisc","MCD":"ConDisc",
    "NKE":"ConDisc","LOW":"ConDisc","SBUX":"ConDisc","TGT":"ConDisc",
    "BKNG":"ConDisc","ABNB":"ConDisc",
    # Consumer Staples
    "WMT":"ConStap","PG":"ConStap","COST":"ConStap","PEP":"ConStap",
    "KO":"ConStap","PM":"ConStap","MDLZ":"ConStap","CL":"ConStap",
    # Industrials
    "CAT":"Indust","HON":"Indust","BA":"Indust","GE":"Indust",
    "LMT":"Indust","RTX":"Indust","MMM":"Indust","ROK":"Indust","UPS":"Indust",
    # Real Estate
    "PLD":"REIT","AMT":"REIT","CCI":"REIT","DLR":"REIT","EQIX":"REIT",
    # Utilities / Comm
    "NEE":"Util","DUK":"Util","SO":"Util","NFLX":"Comm","DIS":"Comm","T":"Comm",
}


# ===========================================================================
# 1. DATA LAYER
# ===========================================================================

class MarketData:
    """
    All data fetching in one place.

    Priority order:
      0. Databento DBEQ.BASIC (1-min bars — fastest, most accurate, already paid)
      1. Alpaca batch          (1 API call — free IEX feed)
      2. yfinance bulk         (yf.download — parallel, free)

    Accepts alpaca-py clients:
      data_client   — StockHistoricalDataClient  (market data)
    """

    def __init__(self, data_client):
        self.data_client = data_client

    # ------------------------------------------------------------------
    def fetch_bars_batch(
        self,
        symbols: List[str],
        lookback_days: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch hourly OHLCV for all symbols in a SINGLE API call.
        Returns a DataFrame indexed by (symbol, timestamp).
        Falls back to yfinance if Alpaca returns nothing.
        """
        end   = datetime.now()
        start = end - timedelta(days=lookback_days)

        # Try Databento first (most accurate, already paid)
        df = self._databento_batch(symbols, start, end)
        if df is not None and not df.empty:
            n = df.index.get_level_values(0).nunique()
            logger.info(f"  Databento: {n}/{len(symbols)} symbols")
            return df

        # Then Alpaca...
        df = self._alpaca_batch(symbols, start, end)
        if df is not None and not df.empty:
            n = df.index.get_level_values(0).nunique()
            logger.info(f"  Alpaca batch: {n}/{len(symbols)} symbols")
            return df

        logger.info("  Alpaca returned no data — trying yfinance...")
        df = self._yfinance_bulk(symbols, start, end)
        if df is not None and not df.empty:
            n = df.index.get_level_values(0).nunique()
            logger.info(f"  yfinance bulk: {n}/{len(symbols)} symbols")
            return df

        logger.warning("  No data from either source.")
        return pd.DataFrame()

    def fetch_single(self, symbol: str, lookback_days: int = 5) -> pd.DataFrame:
        """Fetch a single ticker (SPY, VIX reference)."""
        end   = datetime.now()
        start = end - timedelta(days=lookback_days)

        def _unwrap(df: pd.DataFrame, sym: str) -> pd.DataFrame:
            """Unwrap single-symbol MultiIndex → plain DatetimeIndex."""
            if hasattr(df.index, "levels"):
                syms = df.index.get_level_values(0).unique()
                if sym in syms:
                    return df.xs(sym, level=0)
                return df.droplevel(0)
            return df

        # Databento (Priority 0)
        try:
            df = self._databento_batch([symbol], start, end)
            if df is not None and not df.empty:
                return _unwrap(df, symbol)
        except Exception as e:
            logger.debug(f"  Databento single {symbol}: {e}")

        # Alpaca
        try:
            df = self._alpaca_batch([symbol], start, end)
            if df is not None and not df.empty:
                return _unwrap(df, symbol)
        except Exception as e:
            logger.debug(f"  Alpaca single {symbol}: {e}")

        # yfinance
        try:
            import yfinance as yf
            df = yf.Ticker(symbol).history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1h",
            )
            if df is not None and not df.empty:
                return df.rename(columns={"Open":"open","High":"high","Low":"low",
                                           "Close":"close","Volume":"volume"})
        except Exception as e:
            logger.debug(f"  yfinance single {symbol}: {e}")

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _databento_batch(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 1-min OHLCV from Databento DBEQ.BASIC (Priority 0).
        Returns MultiIndex (symbol, timestamp) DataFrame or None.
        """
        try:
            import databento as db
            import os
            key = os.environ.get("DATABENTO_KEY", "")
            if not key:
                return None
            client = db.Historical(key=key)
            store = client.timeseries.get_range(
                dataset="DBEQ.BASIC",
                symbols=symbols,
                schema="ohlcv-1m",
                start=start,
                end=end,
            )
            df = store.to_df()
            if df.empty:
                return None
            # Rename columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            col_map = {"open_price": "open", "high_price": "high",
                       "low_price": "low", "close_price": "close"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            needed = ["open", "high", "low", "close", "volume"]
            available = [c for c in needed if c in df.columns]
            df = df[available]
            # Ensure MultiIndex (symbol, timestamp)
            if "symbol" in df.columns:
                df = df.set_index(["symbol", df.index])
            elif not isinstance(df.index, pd.MultiIndex):
                return None
            df.index.names = ["symbol", "timestamp"]
            # Strip timezone
            if (hasattr(df.index.get_level_values(1), "tz")
                    and df.index.get_level_values(1).tz is not None):
                df.index = df.index.set_levels(
                    df.index.get_level_values(1).tz_localize(None), level=1
                )
            n = df.index.get_level_values(0).nunique()
            logger.info(f"  Databento DBEQ.BASIC: {n}/{len(symbols)} symbols")
            return df
        except Exception as e:
            logger.debug(f"  Databento batch error: {e}")
            return None

    def _alpaca_batch(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch hourly bars for *symbols* via alpaca-py StockHistoricalDataClient.

        Returns a MultiIndex DataFrame (symbol, timestamp) or None on failure.
        """
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=start,
                end=end,
                adjustment="all",
            )
            resp = self.data_client.get_stock_bars(req)

            # resp.data is Dict[str, List[Bar]]
            if not resp or not resp.data:
                return None

            frames = []
            for sym, bars in resp.data.items():
                if not bars:
                    continue
                rows = [
                    {
                        "open":      bar.open,
                        "high":      bar.high,
                        "low":       bar.low,
                        "close":     bar.close,
                        "volume":    bar.volume,
                        "timestamp": bar.timestamp,
                        "_sym":      sym,
                    }
                    for bar in bars
                ]
                df = pd.DataFrame(rows).set_index(["_sym", "timestamp"])
                frames.append(df)

            if not frames:
                return None

            out = pd.concat(frames)
            out.index.names = ["symbol", "timestamp"]
            return out

        except Exception as e:
            logger.debug(f"  Alpaca batch error: {e}")
            return None

    def _yfinance_bulk(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            raw = yf.download(
                symbols,
                start=(start - timedelta(days=2)).strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1h",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw is None or raw.empty:
                return None

            frames = []
            for sym in symbols:
                try:
                    df = raw[sym].copy() if len(symbols) > 1 else raw.copy()
                    df = df.dropna(subset=["Close"])
                    if len(df) < 2:
                        continue
                    df = df.rename(columns={
                        "Open":"open","High":"high","Low":"low",
                        "Close":"close","Volume":"volume",
                    })[["open","high","low","close","volume"]]
                    df.index.name = "timestamp"
                    df["_sym"] = sym
                    frames.append(df.reset_index().set_index(["_sym","timestamp"]))
                except Exception:
                    continue

            if not frames:
                return None
            out = pd.concat(frames)
            out.index.names = ["symbol","timestamp"]
            return out
        except Exception as e:
            logger.debug(f"  yfinance bulk error: {e}")
            return None


# ===========================================================================
# 2. REGIME DETECTION
# ===========================================================================

class RegimeDetector:
    """
    Classify the current market regime.

    TRENDING_UP   — momentum signals are reliable, full size
    TRENDING_DOWN — short-side signals are reliable, full size
    TRANSITIONING — mixed; trade top-5 only, half size
    CHOPPY        — mean-reverting; skip momentum entirely
    HIGH_FEAR     — VIX > 30; trade only 1-3 highest-conviction signals

    Uses:
      • SPY position vs 20-bar EMA
      • ADX (simplified directional movement index)
      • 5-bar net return direction
      • VIX level if available

    ADX threshold validated via walk-forward analysis 2023-2026:
      4/4 positive OOS years, combined Sharpe +0.71.
      ADX > 20 confirmed as regime filter — strategy performance collapses
      when ADX < 20 (no trend). See also: adx_position_multiplier().
    """

    def detect(
        self,
        spy_bars: pd.DataFrame,
        vix_level: float = 18.0,
    ) -> dict:
        if spy_bars is None or len(spy_bars) < 6:
            return {"regime":"UNKNOWN","tradeable":True,"top_n_limit":20,
                    "size_multiplier":1.0,"reason":"Insufficient SPY data"}

        closes = spy_bars["close"].astype(float).values
        highs  = spy_bars["high"].astype(float).values
        lows   = spy_bars["low"].astype(float).values

        # 20-bar EMA
        ema20    = self._ema(closes, 20)
        above_ma = closes[-1] > ema20

        # ADX (14-bar)
        adx      = self._adx(highs, lows, closes, 14)
        trending = adx > 20

        # 5-bar drift
        drift_5  = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0

        # Classify
        if vix_level > 35:
            regime, tradeable, top_n, mult = "HIGH_FEAR",    True,  5,  0.5
        elif vix_level > 28:
            regime, tradeable, top_n, mult = "ELEVATED_VOL", True,  10, 0.75
        elif trending and above_ma and drift_5 > 0:
            regime, tradeable, top_n, mult = "TRENDING_UP",  True,  20, 1.0
        elif trending and (not above_ma) and drift_5 < 0:
            regime, tradeable, top_n, mult = "TRENDING_DOWN",True,  20, 1.0
        elif not trending:
            regime, tradeable, top_n, mult = "CHOPPY",       False, 0,  0.0
        else:
            regime, tradeable, top_n, mult = "TRANSITIONING",True,  10, 0.6

        return {
            "regime":          regime,
            "tradeable":       tradeable,
            "top_n_limit":     top_n,
            "size_multiplier": mult,
            "adx":             round(float(adx), 1),
            "spy_vs_ema20_pct":round((closes[-1] / ema20 - 1) * 100, 2),
            "spy_drift_5h_pct":round(drift_5 * 100, 2),
            "vix":             round(vix_level, 1),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> float:
        if len(arr) < period:
            return float(np.mean(arr))
        k = 2 / (period + 1)
        v = arr[0]
        for x in arr[1:]:
            v = x * k + v * (1 - k)
        return float(v)

    @staticmethod
    def _adx(highs, lows, closes, period: int = 14) -> float:
        n = len(highs)
        if n < period + 2:
            return 25.0

        tr_list, dm_p, dm_m = [], [], []
        for i in range(1, n):
            hl  = highs[i]  - lows[i]
            hpc = abs(highs[i]  - closes[i-1])
            lpc = abs(lows[i]   - closes[i-1])
            tr_list.append(max(hl, hpc, lpc))

            up   = highs[i]  - highs[i-1]
            down = lows[i-1] - lows[i]
            dm_p.append(up   if up   > down and up   > 0 else 0.0)
            dm_m.append(down if down > up   and down > 0 else 0.0)

        atr  = float(np.mean(tr_list[-period:]))
        if atr == 0:
            return 0.0
        di_p = float(np.mean(dm_p[-period:])) / atr
        di_m = float(np.mean(dm_m[-period:])) / atr
        denom = di_p + di_m
        return 0.0 if denom == 0 else abs(di_p - di_m) / denom * 100


# ===========================================================================
# 2b. ADX POSITION GATE
# ===========================================================================

ADX_THRESHOLD: float = 20.0  # walk-forward validated: Sharpe +0.71 OOS 2023-2026


def adx_position_multiplier(adx_value: float) -> float:
    """
    Returns position size multiplier based on ADX regime.

    ADX >= 20: full size (1.0) — trending regime, signals are reliable
    ADX <  20: half size (0.5) — weak trend, strategy performance collapses

    Validated via walk-forward 2023-2026: 4/4 positive OOS years,
    combined Sharpe +0.71 when ADX > 20 filter is applied.
    """
    return 1.0 if adx_value >= ADX_THRESHOLD else 0.5


# ===========================================================================
# 3. SIGNAL ENGINE
# ===========================================================================

class SignalEngine:
    """
    Multi-factor signal computation + cross-sectional Z-score normalisation.

    Evidence-based factor weights (IC analysis, OOS 2023-2026):
      vwap_dev_intraday  35%  — 1-min VWAP distance, IC +0.054 p=0.007 ✅
      pmo_crossover      20%  — contrarian (sign flipped), IC -0.032 p=0.005 ✅
      vol_surprise       15%  — log volume vs avg, IC -0.018 p=0.053 ⚠️ weak
      imbalance_real     30%  — Databento closing imbalance, pending full validation

    REMOVED (zero IC, pure noise):
      rel_strength  — IC +0.001 p=0.963
      vwap_dev (daily OHLC proxy) — IC -0.020 p=0.081
      imbalance (price-based proxy) — IC +0.011 p=0.255

    Each factor is Z-scored cross-sectionally before weighting so a 1-point
    difference in score means the same thing regardless of the factor scale.
    """

    WEIGHTS = {
        "vwap_dev_intraday": 0.35,  # IC +0.054 p=0.007 ✅ — requires 1-min bars from Alpaca
        "pmo_crossover": 0.20,      # IC -0.032 p=0.005 ✅ contrarian — FLIP SIGN in scoring
        "vol_surprise": 0.15,       # IC -0.018 p=0.053 ⚠️ weak but kept at reduced weight
        "imbalance_real": 0.30,     # Databento real closing imbalance — pending full validation
    }
    # REMOVED: rel_strength (IC +0.001, pure noise)
    # REMOVED: vwap_dev (daily OHLC proxy, IC -0.020, noise)
    # REMOVED: imbalance proxy (price-only, IC +0.011, noise)

    def compute(
        self,
        all_bars: pd.DataFrame,
        spy_bars: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute composite signal for every symbol in all_bars.
        Returns a DataFrame sorted by score descending.
        """
        spy_return = self._latest_return(spy_bars)
        symbols    = all_bars.index.get_level_values(0).unique().tolist()

        rows = []
        for sym in symbols:
            row = self._compute_one(sym, all_bars, spy_return)
            if row:
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Cross-sectional Z-score each factor
        for f in self.WEIGHTS:
            mu, sd   = df[f].mean(), df[f].std()
            df[f+"_z"] = (df[f] - mu) / (sd + 1e-9)

        # Composite score
        df["score"] = sum(
            w * df[f+"_z"] for f, w in self.WEIGHTS.items()
        )

        df["direction"] = np.where(df["score"] > 0, "LONG", "SHORT")

        # ADX position multiplier — halve size when trend is weak
        df["adx_multiplier"] = df["adx"].apply(adx_position_multiplier)

        # Readable pct columns
        df["vwap_dev_intraday_pct"] = (df["vwap_dev_intraday"] * 100).round(3)
        df["raw_return_pct"]        = (df["raw_return"]         * 100).round(3)
        df["imbalance_real_pct"]    = (df["imbalance_real"]     * 100).round(3)
        df["score"]                 = df["score"].round(4)
        df["sector"]                = df["symbol"].map(lambda s: SECTOR_MAP.get(s, "Other"))

        return df.sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def _compute_one(
        self, symbol: str, all_bars: pd.DataFrame, spy_return: float
    ) -> Optional[dict]:
        try:
            bars = all_bars.xs(symbol, level=0)
            if len(bars) < 3:
                return None

            close  = bars["close"].astype(float)
            high   = bars["high"].astype(float)
            low    = bars["low"].astype(float)
            volume = bars["volume"].astype(float)

            price       = float(close.iloc[-1])
            raw_return  = self._latest_return(bars)

            # Factor 1 — VWAP deviation (intraday 1-min bars only)
            # Daily OHLC proxy has IC -0.020 (noise). Only 1-min intraday
            # VWAP is valid (IC +0.054, p=0.007, validated in
            # alpaca_microstructure.py). If the data source provides 1-min
            # bars (e.g. Databento DBEQ.BASIC), compute real intraday VWAP.
            # Otherwise return 0 (neutral) — do NOT fall back to daily proxy.
            vwap_dev_intraday = 0.0
            bar_count = len(bars)
            if bar_count >= 60:
                # Likely intraday 1-min bars — compute real VWAP
                typical = (high + low + close) / 3
                vwap    = float((typical * volume).sum() / (volume.sum() + 1e-9))
                vwap_dev_intraday = (price - vwap) / (vwap + 1e-9)
            # else: hourly or daily data — return 0 (neutral, don't use daily proxy)

            # Factor 2 — PMO crossover (contrarian, sign flipped)
            # IC -0.032, p=0.005 — only statistically significant technical indicator.
            # Negative IC means contrarian: flip sign when scoring.
            pmo_crossover = self.compute_pmo(close)

            # Factor 3 — Volume surprise (log ratio vs rolling avg)
            vol_vals    = volume.values
            vol_avg     = float(np.mean(vol_vals[:-1])) if len(vol_vals) > 1 else float(vol_vals[-1])
            vol_current = float(vol_vals[-1])
            vol_surprise = float(np.log(max(vol_current, 1) / max(vol_avg, 1)))

            # Factor 4 — Real closing imbalance (Databento)
            # Replaces old price-based imbalance proxy (IC +0.011, noise).
            # Real Databento imbalance validation is pending; slot reserved.
            # Returns 0 (neutral) if unavailable.
            imbalance_real = 0.0
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from strategy.databento_imbalance import ClosingImbalanceSignal, _prev_trading_day
                from datetime import date
                global _closing_imbalance_signal
                if _closing_imbalance_signal is None:
                    _closing_imbalance_signal = ClosingImbalanceSignal()
                sig = _closing_imbalance_signal
                daily_sigs = sig.compute_daily([symbol], _prev_trading_day(date.today()))
                imbalance_real = float(daily_sigs.get(symbol, 0.0))
            except Exception:
                pass  # Databento unavailable — neutral contribution (0)

            # ADX for position sizing gate (not a scoring signal)
            adx = RegimeDetector._adx(
                high.values, low.values, close.values, 14
            )

            return {
                "symbol":           symbol,
                "price":            round(price, 2),
                "volume":           int(vol_current),
                "vwap_dev_intraday":vwap_dev_intraday,
                "pmo_crossover":    pmo_crossover,
                "vol_surprise":     vol_surprise,
                "raw_return":       raw_return,
                "imbalance_real":   imbalance_real,
                "adx":              round(float(adx), 1),
            }
        except Exception as e:
            logger.debug(f"  Signal error {symbol}: {e}")
            return None

    @staticmethod
    def compute_pmo(close: pd.Series) -> float:
        """
        Price Momentum Oscillator (PMO) crossover value.

        PMO is a double-smoothed rate of change:
            ROC_1      = (close / close.shift(1) - 1) * 100
            EMA1       = ROC_1.ewm(span=35).mean() * 20
            PMO        = EMA1.ewm(span=20).mean()
            PMO_signal = PMO.ewm(span=10).mean()
            crossover  = PMO - PMO_signal

        IC = -0.032 (p=0.005) — statistically significant but CONTRARIAN.
        The sign is flipped when used in scoring (negative crossover = buy).

        Returns the raw crossover value (caller flips sign for scoring).
        Returns 0.0 if insufficient data.
        """
        if close is None or len(close) < 40:
            return 0.0
        try:
            roc_1      = (close / close.shift(1) - 1) * 100
            ema1       = roc_1.ewm(span=35, min_periods=10).mean() * 20
            pmo        = ema1.ewm(span=20, min_periods=5).mean()
            pmo_signal = pmo.ewm(span=10, min_periods=3).mean()
            crossover  = float(pmo.iloc[-1] - pmo_signal.iloc[-1])
            # Flip sign: contrarian indicator (negative IC)
            return -crossover
        except Exception:
            return 0.0

    @staticmethod
    def _latest_return(bars: pd.DataFrame) -> float:
        if bars is None or len(bars) < 2:
            return 0.0
        c = bars["close"].astype(float).values
        return float((c[-1] - c[-2]) / (c[-2] + 1e-9))


# ===========================================================================
# 4. SCANNER ORCHESTRATOR
# ===========================================================================

class MomentumScannerV2:
    """
    Top-level orchestrator.

    Usage:
        scanner = MomentumScannerV2(api_key="...", api_secret="...")
        result  = scanner.scan(symbols, top_n=20)

    Result keys:
        regime       — dict with regime name and tradeable flag
        signals      — full scored DataFrame for every symbol
        top_long     — list[dict] top long candidates (sector-limited)
        top_short    — list[dict] top short candidates (sector-limited)
        consensus    — symbols where all 3 factors agree (highest confidence)
        elapsed      — seconds taken

    SDK: alpaca-py (alpaca.trading / alpaca.data) — NOT alpaca_trade_api.
    Environment variables (checked in priority order):
        ALPACA_API_KEY     / APCA_API_KEY_ID
        ALPACA_API_SECRET  / APCA_API_SECRET_KEY
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        key    = api_key    or os.getenv("ALPACA_API_KEY",    "") or os.getenv("APCA_API_KEY_ID",     "")
        secret = api_secret or os.getenv("ALPACA_API_SECRET", "") or os.getenv("APCA_API_SECRET_KEY", "")

        self.trading_client = TradingClient(api_key=key, secret_key=secret, paper=True)
        self.data_client    = StockHistoricalDataClient(api_key=key, secret_key=secret)

        self.data    = MarketData(self.data_client)
        self.regime  = RegimeDetector()
        self.signals = SignalEngine()

        # Test connection
        try:
            acct = self.trading_client.get_account()
            logger.info(f"✓ Alpaca connected — status:{acct.status}  cash:${float(acct.cash):,.0f}")
        except Exception as e:
            logger.warning(f"Alpaca connection warning: {e}")

    # ------------------------------------------------------------------
    def scan(
        self,
        symbols:         List[str],
        top_n:           int  = 20,
        max_per_sector:  int  = 3,
        force:           bool = False,   # ignore regime check
    ) -> dict:
        t0 = datetime.now()
        logger.info(f"V2 scan starting — {len(symbols)} symbols")

        # ── 1. SPY + VIX reference ────────────────────────────────────
        spy_bars = self.data.fetch_single("SPY", lookback_days=3)
        vix_level = 18.0
        try:
            vix_bars  = self.data.fetch_single("VIX", lookback_days=2)
            if not vix_bars.empty:
                vix_level = float(vix_bars["close"].iloc[-1])
        except Exception:
            pass

        # ── 2. Regime ─────────────────────────────────────────────────
        regime = self.regime.detect(spy_bars, vix_level)
        logger.info(
            f"Regime: {regime['regime']}  ADX={regime['adx']}  "
            f"SPY vs EMA20={regime['spy_vs_ema20_pct']:+.2f}%  VIX={regime['vix']}"
        )

        if not regime["tradeable"] and not force:
            logger.warning(f"Market is {regime['regime']} — skipping scan (use force=True to override)")
            return self._empty_result(regime, t0)

        # ── 3. Batch data fetch ───────────────────────────────────────
        all_bars = self.data.fetch_bars_batch(symbols, lookback_days=3)
        if all_bars.empty:
            logger.error("No market data available.")
            return self._empty_result(regime, t0)

        fetched = all_bars.index.get_level_values(0).nunique()
        logger.info(f"✓ Bars fetched: {fetched}/{len(symbols)} symbols")

        # ── 4. Signals ────────────────────────────────────────────────
        sig_df = self.signals.compute(all_bars, spy_bars)
        if sig_df.empty:
            logger.error("Signal computation returned nothing.")
            return self._empty_result(regime, t0)

        logger.info(f"✓ Signals computed: {len(sig_df)} symbols")

        # ── 5. Select top candidates with sector limits ───────────────
        limit     = min(top_n, regime["top_n_limit"])
        top_long  = self._select(sig_df, "LONG",  limit, max_per_sector)
        top_short = self._select(sig_df, "SHORT", limit, max_per_sector)

        # ── 6. Consensus — all 4 factor z-scores agree ────────────────
        # A consensus signal requires:
        #   • |score| > 0.5  (meaningfully above average)
        #   • all 4 z-scores positive (LONG) or all negative (SHORT)
        mask_agree = (
            (sig_df["vwap_dev_intraday_z"] * sig_df["pmo_crossover_z"] > 0) &
            (sig_df["pmo_crossover_z"]     * sig_df["vol_surprise_z"]  > 0) &
            (sig_df["vol_surprise_z"]      * sig_df["imbalance_real_z"] > 0) &
            (sig_df["score"].abs() > 0.5)
        )
        consensus = sig_df.loc[mask_agree, "symbol"].tolist()

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f"✓ Done in {elapsed:.1f}s  "
            f"longs:{len(top_long)}  shorts:{len(top_short)}  consensus:{len(consensus)}"
        )

        return {
            "regime":    regime,
            "signals":   sig_df,
            "top_long":  top_long,
            "top_short": top_short,
            "consensus": consensus,
            "spy_return":self.signals._latest_return(spy_bars),
            "elapsed":   elapsed,
            "timestamp": datetime.now().isoformat(),
            "symbols_scanned": fetched,
        }

    # ------------------------------------------------------------------
    def _select(
        self,
        df:             pd.DataFrame,
        direction:      str,
        top_n:          int,
        max_per_sector: int,
    ) -> List[dict]:
        sub = df[df["direction"] == direction].copy()
        sub = sub.sort_values("score", ascending=(direction == "SHORT"))

        result, sector_counts = [], {}
        for _, row in sub.iterrows():
            if len(result) >= top_n:
                break
            sec = row.get("sector", "Other")
            if sector_counts.get(sec, 0) >= max_per_sector:
                continue
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            result.append(row.to_dict())
        return result

    @staticmethod
    def _empty_result(regime: dict, t0: datetime) -> dict:
        return {
            "regime": regime, "signals": pd.DataFrame(),
            "top_long": [], "top_short": [], "consensus": [],
            "spy_return": 0.0, "elapsed": (datetime.now() - t0).total_seconds(),
            "timestamp": datetime.now().isoformat(), "symbols_scanned": 0,
        }
