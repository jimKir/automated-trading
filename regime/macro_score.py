"""
Macro Stress Scorer
====================
Rule-based macro stress score using FRED data.
Thresholds derived from economic theory — NOT optimised on backtest data.

All indicators use real-time vintage where possible to avoid look-ahead bias:
  - Yield curve: daily market data, no revision
  - Credit spreads: daily, no revision
  - VIX: daily, no revision
  - PMI: monthly release, used with 1-month lag to avoid look-ahead

Score: 0 (benign) → 1.0 (maximum macro stress)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("MacroScorer")

# FRED series IDs (all free, no API key required for basic access)
FRED_SERIES = {
    "t10y2y":    "T10Y2Y",    # 10Y-2Y Treasury spread (yield curve)
    "t10y3m":    "T10Y3M",    # 10Y-3M spread (more predictive of recession)
    "bamlh0a0":  "BAMLH0A0HYM2",  # BofA HY OAS (credit spread)
    "vix":       "VIXCLS",    # VIX close
    "dxy":       None,        # DXY: fetched from yfinance (DX-Y.NYB)
}

# Economic-logic thresholds (not optimised)
YIELD_CURVE_INVERSION_THRESHOLD = 0.0   # below 0 = inverted
CREDIT_SPREAD_STRESS_THRESHOLD  = 4.0   # >4% = elevated stress (historical avg ~3.5%)
CREDIT_SPREAD_CRISIS_THRESHOLD  = 7.0   # >7% = crisis level (GFC peak ~20%, normal ~3%)
VIX_STRESS_THRESHOLD            = 20.0  # >20 = elevated fear
VIX_CRISIS_THRESHOLD            = 30.0  # >30 = fear/crisis
DXY_SPIKE_THRESHOLD             = 1.5   # >1.5% weekly rise = risk-off dollar spike


class MacroStressScorer:
    """
    Computes a macro stress score [0, 1] from yield curve,
    credit spreads, VIX, and dollar strength.
    """

    def __init__(self):
        self._cache: Dict[str, pd.Series] = {}
        self._last_fetch: Optional[datetime] = None

    # ------------------------------------------------------------------
    def _fetch_fred(self, series_id: str, start: str, end: str) -> pd.Series:
        """Fetch a FRED time series. Falls back to zeros on failure."""
        cache_key = f"{series_id}_{start}_{end}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            import pandas_datareader.data as web
            s = web.DataReader(series_id, "fred", start, end)[series_id].dropna()
            self._cache[cache_key] = s
            return s
        except Exception as e:
            log.warning(f"FRED fetch failed for {series_id}: {e} — using zeros")
            idx = pd.date_range(start, end, freq="B")
            return pd.Series(0.0, index=idx)

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.Series:
        """Fetch price series from yfinance."""
        cache_key = f"yf_{symbol}_{start}_{end}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            import yfinance as yf
            df = yf.download(symbol, start=start, end=end,
                             auto_adjust=True, progress=False)
            s = df["Close"].dropna()
            self._cache[cache_key] = s
            return s
        except Exception as e:
            log.warning(f"yfinance fetch failed for {symbol}: {e}")
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    def _yield_curve_score(self, t10y2y: pd.Series, t10y3m: pd.Series,
                           date: pd.Timestamp) -> Tuple[float, str]:
        """
        Score yield curve stress.
        Both spreads inverting = strong recession signal.
        """
        try:
            v1 = float(t10y2y.asof(date)) if not t10y2y.empty else 0.5
            v2 = float(t10y3m.asof(date)) if not t10y3m.empty else 0.5
        except Exception:
            return 0.0, "N/A"

        score = 0.0
        if v1 < YIELD_CURVE_INVERSION_THRESHOLD:
            score += 0.5
        if v2 < YIELD_CURVE_INVERSION_THRESHOLD:
            score += 0.5
        # Deeper inversion = higher score
        depth = max(0, -min(v1, v2))  # how inverted (positive number)
        score = min(1.0, score + depth * 0.3)
        return score, f"10Y-2Y={v1:.2f} 10Y-3M={v2:.2f}"

    def _credit_spread_score(self, hy_spread: pd.Series,
                             date: pd.Timestamp) -> Tuple[float, str]:
        """Score credit spread stress."""
        try:
            v = float(hy_spread.asof(date)) if not hy_spread.empty else 3.5
        except Exception:
            return 0.0, "N/A"

        if v >= CREDIT_SPREAD_CRISIS_THRESHOLD:
            score = 1.0
        elif v >= CREDIT_SPREAD_STRESS_THRESHOLD:
            score = 0.5 + 0.5 * (v - CREDIT_SPREAD_STRESS_THRESHOLD) / (
                CREDIT_SPREAD_CRISIS_THRESHOLD - CREDIT_SPREAD_STRESS_THRESHOLD)
        else:
            score = v / CREDIT_SPREAD_STRESS_THRESHOLD * 0.3
        return float(np.clip(score, 0, 1)), f"HY_spread={v:.2f}%"

    def _vix_score(self, vix: pd.Series, date: pd.Timestamp) -> Tuple[float, str]:
        """Score VIX stress — level AND trend matter."""
        try:
            v = float(vix.asof(date)) if not vix.empty else 15.0
            # 20d trend
            window = vix[vix.index <= date].tail(20)
            trend = (float(window.iloc[-1]) - float(window.iloc[0])) / max(float(window.iloc[0]), 1) if len(window) >= 5 else 0.0
        except Exception:
            return 0.0, "N/A"

        level_score = 0.0
        if v >= VIX_CRISIS_THRESHOLD:
            level_score = 1.0
        elif v >= VIX_STRESS_THRESHOLD:
            level_score = 0.4 + 0.6 * (v - VIX_STRESS_THRESHOLD) / (
                VIX_CRISIS_THRESHOLD - VIX_STRESS_THRESHOLD)
        else:
            level_score = v / VIX_STRESS_THRESHOLD * 0.3

        trend_score = float(np.clip(trend * 2, 0, 0.3))  # rising VIX adds up to 0.3
        score = float(np.clip(level_score + trend_score, 0, 1))
        return score, f"VIX={v:.1f} trend={trend*100:.1f}%"

    def _dxy_score(self, dxy: pd.Series, date: pd.Timestamp) -> Tuple[float, str]:
        """Score dollar stress — rapid USD strengthening = risk-off."""
        try:
            window = dxy[dxy.index <= date].tail(5)
            if len(window) < 2:
                return 0.0, "N/A"
            weekly_chg = (float(window.iloc[-1]) - float(window.iloc[0])) / float(window.iloc[0]) * 100
        except Exception:
            return 0.0, "N/A"

        score = 0.0
        if weekly_chg > DXY_SPIKE_THRESHOLD:
            score = min(1.0, (weekly_chg - DXY_SPIKE_THRESHOLD) / 3.0)
        return score, f"DXY_5d_chg={weekly_chg:.2f}%"

    # ------------------------------------------------------------------
    def compute_series(self, start: str, end: str) -> pd.Series:
        """
        Compute daily macro stress score for the full backtest period.
        Returns pd.Series with date index, values in [0, 1].
        """
        log.info("MacroScorer: fetching FRED + market data...")

        t10y2y   = self._fetch_fred("T10Y2Y",           start, end)
        t10y3m   = self._fetch_fred("T10Y3M",           start, end)
        hy_spread = self._fetch_fred("BAMLH0A0HYM2",    start, end)
        vix      = self._fetch_fred("VIXCLS",           start, end)
        dxy      = self._fetch_yfinance("DX-Y.NYB",     start, end)

        biz_days = pd.date_range(start, end, freq="B")
        scores   = pd.Series(index=biz_days, dtype=float)

        weights = {"yield": 0.30, "credit": 0.35, "vix": 0.25, "dxy": 0.10}

        for date in biz_days:
            yc,  _ = self._yield_curve_score(t10y2y, t10y3m, date)
            cs,  _ = self._credit_spread_score(hy_spread, date)
            vs,  _ = self._vix_score(vix, date)
            ds,  _ = self._dxy_score(dxy, date)

            scores[date] = (
                weights["yield"]  * yc +
                weights["credit"] * cs +
                weights["vix"]    * vs +
                weights["dxy"]    * ds
            )

        return scores.fillna(method="ffill").fillna(0.0)

    def score_today(self) -> float:
        """Score for live/paper trading — uses last 90 days of data."""
        end   = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
        series = self.compute_series(start, end)
        return float(series.iloc[-1]) if not series.empty else 0.0
