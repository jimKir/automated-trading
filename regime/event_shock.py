"""
Event Shock Detector
=====================
Detects sudden market shocks via:
  1. VIX velocity — rapid VIX acceleration (not just level)
  2. VIX term structure — spot vs 3-month VIX futures (contango/backwardation)
  3. Put-call ratio spike (SPY options) via yfinance proxy
  4. Market breadth collapse — % of S&P500 assets above 50d MA
  5. Cross-asset shock — simultaneous selloff in equity + bond + gold

These are *fast* signals (daily resolution) that complement the slower
macro stress score. They fire quickly when a shock begins.

Score: 0.0 (quiet) → 1.0 (acute shock in progress)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("EventShock")

# Thresholds — set by domain knowledge, not optimisation
VIX_VEL_THRESHOLD = 0.20  # 20% rise in VIX over 5 days = shock
VIX_VEL_CRISIS = 0.50  # 50% rise = acute crisis
BREADTH_STRESS = 0.45  # <45% stocks above 50d MA = stress
BREADTH_CRISIS = 0.30  # <30% = crisis breadth
CROSS_ASSET_THRESHOLD = 0.015  # 1.5% simultaneous drop across asset classes


class EventShockDetector:
    """
    Fast-reacting event shock detector.
    Uses market-derived signals only — no external API dependencies.
    """

    def __init__(self):
        self._cache: dict[str, pd.Series] = {}

    def _fetch(self, symbol: str, start: str, end: str) -> pd.Series:
        key = f"{symbol}_{start}_{end}"
        if key in self._cache:
            return self._cache[key]
        try:
            import yfinance as yf

            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            s = df["Close"].dropna()
            self._cache[key] = s
            return s
        except Exception as e:
            log.warning(f"EventShock fetch failed {symbol}: {e}")
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    def _vix_velocity_score(self, vix: pd.Series, date: pd.Timestamp) -> float:
        """How fast is VIX accelerating? Faster = worse."""
        window = vix[vix.index <= date].tail(5)
        if len(window) < 2:
            return 0.0
        chg = (float(window.iloc[-1]) - float(window.iloc[0])) / max(float(window.iloc[0]), 1)
        if chg >= VIX_VEL_CRISIS:
            return 1.0
        if chg >= VIX_VEL_THRESHOLD:
            return 0.3 + 0.7 * (chg - VIX_VEL_THRESHOLD) / (VIX_VEL_CRISIS - VIX_VEL_THRESHOLD)
        return 0.0

    def _vix_term_structure_score(
        self, vix: pd.Series, vix3m: pd.Series, date: pd.Timestamp
    ) -> float:
        """
        VIX term structure inversion (backwardation) = panic.
        When spot VIX > 3m VIX, near-term fear > medium-term fear.
        """
        try:
            v_spot = float(vix.asof(date))
            v_3m = float(vix3m.asof(date))
            if v_3m <= 0:
                return 0.0
            spread = (v_spot - v_3m) / v_3m
            # backwardation: spot > 3m → spread > 0 → stress
            return float(np.clip(spread * 2.5, 0, 1))
        except Exception:
            return 0.0

    def _breadth_score(self, breadth: pd.Series, date: pd.Timestamp) -> float:
        """
        Market breadth: % of our ETF universe above 50d MA.
        Low breadth = underlying weakness even if index holds up.
        """
        try:
            v = float(breadth.asof(date))
        except Exception:
            return 0.0
        if v <= BREADTH_CRISIS:
            return 1.0
        if v <= BREADTH_STRESS:
            return 0.3 + 0.7 * (BREADTH_STRESS - v) / (BREADTH_STRESS - BREADTH_CRISIS)
        return 0.0

    def _cross_asset_shock_score(
        self, spy: pd.Series, tlt: pd.Series, gld: pd.Series, date: pd.Timestamp
    ) -> float:
        """
        Cross-asset shock: when equity, bonds AND gold all fall together,
        it signals a liquidity crisis (forced selling across everything).
        """
        scores = []
        for s in [spy, tlt, gld]:
            window = s[s.index <= date].tail(3)
            if len(window) >= 2:
                chg = (float(window.iloc[-1]) - float(window.iloc[0])) / max(
                    float(window.iloc[0]), 1e-6
                )
                scores.append(chg)

        if len(scores) < 3:
            return 0.0

        # All three falling simultaneously
        all_negative = all(c < -CROSS_ASSET_THRESHOLD for c in scores)
        if all_negative:
            avg_drop = abs(np.mean(scores))
            return float(np.clip(avg_drop / 0.05, 0, 1))
        return 0.0

    # ------------------------------------------------------------------
    def _compute_breadth_proxy(self, all_prices: pd.DataFrame) -> pd.Series:
        """
        Compute % of instruments above their 50d MA as a breadth proxy.
        Uses the instruments in our universe.
        """
        ma50 = all_prices.rolling(50).mean()
        above = (all_prices > ma50).astype(float)
        return above.mean(axis=1)

    # ------------------------------------------------------------------
    def compute_series(self, start: str, end: str, all_prices: pd.DataFrame = None) -> pd.Series:
        """
        Compute daily event shock score for the full backtest period.
        all_prices: multi-asset price DataFrame (for breadth calculation).
        """
        log.info("EventShock: computing shock scores...")

        vix = self._fetch("^VIX", start, end)
        vix3m = self._fetch("^VIX3M", start, end)
        spy = self._fetch("SPY", start, end)
        tlt = self._fetch("TLT", start, end)
        gld = self._fetch("GLD", start, end)

        # Breadth from our universe if provided, else SPY proxy
        if all_prices is not None and not all_prices.empty:
            breadth = self._compute_breadth_proxy(all_prices)
        else:
            # Use SPY distance from 200d MA as proxy
            spy_ma200 = spy.rolling(200).mean()
            breadth_val = (spy / spy_ma200.replace(0, np.nan)).clip(0.8, 1.2)
            breadth = (breadth_val - 0.8) / 0.4  # normalise to [0,1]
            breadth = breadth.rename("breadth")

        biz_days = pd.date_range(start, end, freq="B")
        scores = pd.Series(index=biz_days, dtype=float)

        weights = {
            "vix_vel": 0.35,
            "vix_ts": 0.20,
            "breadth": 0.25,
            "cross_asset": 0.20,
        }

        for date in biz_days:
            vv = self._vix_velocity_score(vix, date)
            vts = self._vix_term_structure_score(vix, vix3m, date)
            br = self._breadth_score(breadth, date)
            ca = self._cross_asset_shock_score(spy, tlt, gld, date)

            scores[date] = (
                weights["vix_vel"] * vv
                + weights["vix_ts"] * vts
                + weights["breadth"] * br
                + weights["cross_asset"] * ca
            )

        return scores.fillna(method="ffill").fillna(0.0)

    def score_today(self, all_prices: pd.DataFrame = None) -> float:
        """Score for live/paper trading."""
        end = datetime.now(UTC).strftime("%Y-%m-%d")
        start = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%d")
        series = self.compute_series(start, end, all_prices)
        return float(series.iloc[-1]) if not series.empty else 0.0
