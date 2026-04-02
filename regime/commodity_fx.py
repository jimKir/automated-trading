"""
Commodity & FX Stress Signals
==============================
Extracts stress signals from commodities and FX markets:

  Commodities:
    - Oil velocity (>15% in 2 weeks = supply shock)
    - Gold/SPY ratio trend (rising = flight to safety)
    - Copper trend (Dr. Copper — leading economic indicator)
    - Oil/Gold ratio (inflation vs deflation regime)

  FX:
    - DXY momentum (USD strengthening = risk-off)
    - USD/JPY direction (JPY strengthening = carry unwind)
    - EUR/USD trend (European risk appetite)
    - EM stress proxy (DXY vs EEM correlation)

Score: 0.0 (risk-on, benign) → 1.0 (maximum commodity/FX stress)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("CommodityFX")

# Symbols
OIL_SYM   = "CL=F"      # Crude oil futures
GOLD_SYM  = "GC=F"      # Gold futures
COPPER_SYM = "HG=F"     # Copper futures
SPY_SYM   = "SPY"
DXY_SYM   = "DX-Y.NYB"  # US Dollar Index
JPY_SYM   = "JPY=X"     # USD/JPY
EUR_SYM   = "EURUSD=X"  # EUR/USD
EEM_SYM   = "EEM"       # Emerging markets ETF

# Thresholds (economic-logic based)
OIL_SPIKE_THRESHOLD      = 0.12   # 12% rise in 10 days = supply shock signal
OIL_CRASH_THRESHOLD      = -0.15  # 15% drop = demand shock / recession signal
GOLD_SPY_TREND_WINDOW    = 20     # 20d MA of gold/SPY ratio
COPPER_TREND_WINDOW      = 60     # 60d MA for Dr. Copper
DXY_MOMENTUM_WINDOW      = 10     # 10d momentum
JPY_STRENGTHENING_THRESH = -0.02  # USD/JPY down 2% in 10 days = carry unwind


class CommodityFXScorer:
    """
    Computes stress/risk-off scores from commodity and FX markets.
    All signals are price-derived — always available, no revision risk.
    """

    def __init__(self):
        self._cache: Dict[str, pd.Series] = {}

    def _fetch(self, symbol: str, start: str, end: str) -> pd.Series:
        key = f"{symbol}_{start}_{end}"
        if key in self._cache:
            return self._cache[key]
        try:
            import yfinance as yf
            df = yf.download(symbol, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                return pd.Series(dtype=float)
            # FIX: handle both single-level and multi-level column DataFrames
            # yfinance can return MultiIndex columns like ('Close', 'CL=F')
            if isinstance(df.columns, pd.MultiIndex):
                if "Close" in df.columns.get_level_values(0):
                    s = df["Close"].iloc[:, 0]
                else:
                    return pd.Series(dtype=float)
            else:
                if "Close" not in df.columns:
                    return pd.Series(dtype=float)
                s = df["Close"]
            # Always squeeze to a 1-D Series and strip tz
            s = s.squeeze().dropna()
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            self._cache[key] = s
            return s
        except Exception as e:
            log.warning(f"CommodityFX fetch failed {symbol}: {e}")
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    def _oil_score(self, oil: pd.Series, date: pd.Timestamp) -> float:
        """
        Oil price shock detector.
        Both rapid spikes (supply shock) and rapid crashes (demand shock)
        signal stress for risk assets.
        """
        window = oil[oil.index <= date].tail(10)
        if len(window) < 2:
            return 0.0
        chg = (float(window.iloc[-1]) - float(window.iloc[0])) / max(float(window.iloc[0]), 1e-6)

        spike_score = float(np.clip((chg - OIL_SPIKE_THRESHOLD) / 0.20, 0, 1)) if chg > OIL_SPIKE_THRESHOLD else 0.0
        crash_score = float(np.clip((abs(chg) - abs(OIL_CRASH_THRESHOLD)) / 0.15, 0, 1)) if chg < OIL_CRASH_THRESHOLD else 0.0
        return max(spike_score, crash_score)

    def _gold_spy_score(self, gold: pd.Series, spy: pd.Series,
                        date: pd.Timestamp) -> float:
        """
        Gold/SPY ratio rising = flight to safety.
        Uses 20d MA crossover to avoid noise.
        """
        try:
            g_window = gold[gold.index <= date].tail(GOLD_SPY_TREND_WINDOW + 5)
            s_window = spy[spy.index <= date].reindex(g_window.index, method="ffill")
            ratio    = (g_window / s_window.replace(0, np.nan)).dropna()
            if len(ratio) < GOLD_SPY_TREND_WINDOW:
                return 0.0
            ma = ratio.rolling(GOLD_SPY_TREND_WINDOW).mean()
            if ma.isna().all():
                return 0.0
            # If ratio is above its MA and rising, signal safety flight
            current_ratio = float(ratio.iloc[-1])
            ma_val        = float(ma.iloc[-1])
            above_ma      = current_ratio > ma_val
            trend_up      = float(ratio.iloc[-1]) > float(ratio.iloc[-5]) if len(ratio) >= 5 else False
            return 0.5 if (above_ma and trend_up) else (0.2 if above_ma else 0.0)
        except Exception:
            return 0.0

    def _copper_score(self, copper: pd.Series, date: pd.Timestamp) -> float:
        """
        Dr. Copper: copper below 60d MA and declining = economic slowdown.
        Slow signal — confirms macro deterioration.
        """
        window = copper[copper.index <= date].tail(COPPER_TREND_WINDOW + 5)
        if len(window) < COPPER_TREND_WINDOW:
            return 0.0
        ma = window.rolling(COPPER_TREND_WINDOW).mean()
        if ma.isna().all():
            return 0.0
        current = float(window.iloc[-1])
        ma_val  = float(ma.iloc[-1])
        below_ma = current < ma_val
        pct_below = (ma_val - current) / ma_val if ma_val > 0 else 0
        if below_ma:
            return float(np.clip(pct_below * 10, 0, 0.8))
        return 0.0

    def _dxy_score(self, dxy: pd.Series, date: pd.Timestamp) -> float:
        """DXY momentum — rapid USD strengthening = risk-off."""
        window = dxy[dxy.index <= date].tail(DXY_MOMENTUM_WINDOW)
        if len(window) < 2:
            return 0.0
        chg = (float(window.iloc[-1]) - float(window.iloc[0])) / max(float(window.iloc[0]), 1e-6)
        return float(np.clip(chg * 10, 0, 1)) if chg > 0.01 else 0.0

    def _jpy_score(self, usdjpy: pd.Series, date: pd.Timestamp) -> float:
        """
        Yen strengthening (USD/JPY falling) = carry trade unwinding.
        Carry unwind = forced liquidation of risk assets.
        """
        window = usdjpy[usdjpy.index <= date].tail(10)
        if len(window) < 2:
            return 0.0
        chg = (float(window.iloc[-1]) - float(window.iloc[0])) / max(float(window.iloc[0]), 1e-6)
        # USD/JPY falling means JPY strengthening = risk-off
        if chg < JPY_STRENGTHENING_THRESH:
            return float(np.clip(abs(chg) * 15, 0, 1))
        return 0.0

    def _eur_score(self, eurusd: pd.Series, date: pd.Timestamp) -> float:
        """EUR/USD rapid decline = European stress, USD safe haven demand."""
        window = eurusd[eurusd.index <= date].tail(10)
        if len(window) < 2:
            return 0.0
        chg = (float(window.iloc[-1]) - float(window.iloc[0])) / max(float(window.iloc[0]), 1e-6)
        if chg < -0.015:  # EUR/USD falling >1.5% in 10 days
            return float(np.clip(abs(chg) * 20, 0, 0.6))
        return 0.0

    # ------------------------------------------------------------------
    def compute_series(self, start: str, end: str) -> pd.Series:
        """
        Compute daily commodity/FX stress score for the full backtest period.
        """
        log.info("CommodityFX: fetching commodity + FX data...")

        oil    = self._fetch(OIL_SYM,    start, end)
        gold   = self._fetch(GOLD_SYM,   start, end)
        copper = self._fetch(COPPER_SYM, start, end)
        spy    = self._fetch(SPY_SYM,    start, end)
        dxy    = self._fetch(DXY_SYM,    start, end)
        usdjpy = self._fetch(JPY_SYM,    start, end)
        eurusd = self._fetch(EUR_SYM,    start, end)

        biz_days = pd.date_range(start, end, freq="B")
        scores   = pd.Series(index=biz_days, dtype=float)

        weights = {
            "oil":       0.20,
            "gold_spy":  0.20,
            "copper":    0.15,
            "dxy":       0.20,
            "jpy":       0.15,
            "eur":       0.10,
        }

        for date in biz_days:
            oil_s  = self._oil_score(oil, date)
            gs_s   = self._gold_spy_score(gold, spy, date)
            cu_s   = self._copper_score(copper, date)
            dxy_s  = self._dxy_score(dxy, date)
            jpy_s  = self._jpy_score(usdjpy, date)
            eur_s  = self._eur_score(eurusd, date)

            scores[date] = (
                weights["oil"]      * oil_s  +
                weights["gold_spy"] * gs_s   +
                weights["copper"]   * cu_s   +
                weights["dxy"]      * dxy_s  +
                weights["jpy"]      * jpy_s  +
                weights["eur"]      * eur_s
            )

        return scores.ffill().fillna(0.0)

    def score_today(self) -> float:
        """Score for live/paper trading."""
        end   = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=120)).strftime("%Y-%m-%d")
        series = self.compute_series(start, end)
        return float(series.iloc[-1]) if not series.empty else 0.0
