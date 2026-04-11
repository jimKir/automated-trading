#!/usr/bin/env python3
"""
Momentum Scanner V2 — Production Grade
=======================================

Key improvements over V1:
  1. Batch data fetch   — 1 API call for all symbols (not 50 sequential calls)
  2. VWAP deviation     — price vs volume-weighted fair value (40% weight)
  3. Relative strength  — stock alpha vs SPY, strips market noise (35% weight)
  4. Volume surprise    — log-normalised volume vs rolling average (25% weight)
  5. Z-score normalise  — cross-sectional, so scores are comparable
  6. Regime detection   — ADX + SPY trend; skip scan on choppy days
  7. Sector limits      — max 3 signals per sector to avoid concentration
  8. Composite score    — one number per symbol, directly actionable
"""

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector map — used for concentration limits and sector-relative strength
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Tech",
    "MSFT": "Tech",
    "NVDA": "Tech",
    "GOOGL": "Tech",
    "GOOG": "Tech",
    "META": "Tech",
    "AVGO": "Tech",
    "AMD": "Tech",
    "INTC": "Tech",
    "QCOM": "Tech",
    "CRM": "Tech",
    "ADBE": "Tech",
    "TXN": "Tech",
    "AMAT": "Tech",
    "MU": "Tech",
    "LRCX": "Tech",
    "PANW": "Tech",
    "INTU": "Tech",
    "NOW": "Tech",
    "ORCL": "Tech",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "EOG": "Energy",
    "SLB": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "OKE": "Energy",
    "VLO": "Energy",
    "DVN": "Energy",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    "C": "Financials",
    "AXP": "Financials",
    "USB": "Financials",
    "PNC": "Financials",
    # Healthcare
    "UNH": "Health",
    "LLY": "Health",
    "JNJ": "Health",
    "MRK": "Health",
    "ABBV": "Health",
    "TMO": "Health",
    "ABT": "Health",
    "DHR": "Health",
    "ISRG": "Health",
    "PFE": "Health",
    "GILD": "Health",
    "REGN": "Health",
    "BMY": "Health",
    "MDT": "Health",
    "CVS": "Health",
    # Consumer Discretionary
    "AMZN": "ConDisc",
    "TSLA": "ConDisc",
    "HD": "ConDisc",
    "MCD": "ConDisc",
    "NKE": "ConDisc",
    "LOW": "ConDisc",
    "SBUX": "ConDisc",
    "TGT": "ConDisc",
    "BKNG": "ConDisc",
    "ABNB": "ConDisc",
    # Consumer Staples
    "WMT": "ConStap",
    "PG": "ConStap",
    "COST": "ConStap",
    "PEP": "ConStap",
    "KO": "ConStap",
    "PM": "ConStap",
    "MDLZ": "ConStap",
    "CL": "ConStap",
    # Industrials
    "CAT": "Indust",
    "HON": "Indust",
    "BA": "Indust",
    "GE": "Indust",
    "LMT": "Indust",
    "RTX": "Indust",
    "MMM": "Indust",
    "ROK": "Indust",
    "UPS": "Indust",
    # Real Estate
    "PLD": "REIT",
    "AMT": "REIT",
    "CCI": "REIT",
    "DLR": "REIT",
    "EQIX": "REIT",
    # Utilities / Comm
    "NEE": "Util",
    "DUK": "Util",
    "SO": "Util",
    "NFLX": "Comm",
    "DIS": "Comm",
    "T": "Comm",
}


# ===========================================================================
# 1. DATA LAYER
# ===========================================================================


class MarketData:
    """
    All data fetching in one place.

    Priority order:
      1. Alpaca batch  (1 API call — fastest, free IEX feed)
      2. yfinance bulk (yf.download — parallel, free)
    """

    def __init__(self, api):
        self.api = api

    # ------------------------------------------------------------------
    def fetch_bars_batch(
        self,
        symbols: list[str],
        lookback_days: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch hourly OHLCV for all symbols in a SINGLE API call.
        Returns a DataFrame indexed by (symbol, timestamp).
        Falls back to yfinance if Alpaca returns nothing.
        """
        end = datetime.now()
        start = end - timedelta(days=lookback_days)

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
        end = datetime.now()
        start = end - timedelta(days=lookback_days)

        # Alpaca
        try:
            from alpaca_trade_api.rest import TimeFrame

            raw = self.api.get_bars(
                symbol,
                TimeFrame.Hour,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                adjustment="raw",
                feed="iex",
            ).df
            if raw is not None and not raw.empty:
                if hasattr(raw.index, "levels"):
                    raw = (
                        raw.xs(symbol, level=0)
                        if symbol in raw.index.get_level_values(0)
                        else raw.droplevel(0)
                    )
                return raw
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
                return df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
        except Exception as e:
            logger.debug(f"  yfinance single {symbol}: {e}")

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _alpaca_batch(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> pd.DataFrame | None:
        try:
            from alpaca_trade_api.rest import TimeFrame

            raw = self.api.get_bars(
                symbols,
                TimeFrame.Hour,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                adjustment="raw",
                feed="iex",
            ).df
            if raw is None or raw.empty:
                return None
            # Ensure two-level index (symbol, timestamp)
            if not hasattr(raw.index, "levels"):
                return None
            raw.index.names = ["symbol", "timestamp"]
            return raw
        except Exception as e:
            logger.debug(f"  Alpaca batch error: {e}")
            return None

    def _yfinance_bulk(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> pd.DataFrame | None:
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
                    df = df.rename(
                        columns={
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )[["open", "high", "low", "close", "volume"]]
                    df.index.name = "timestamp"
                    df["_sym"] = sym
                    frames.append(df.reset_index().set_index(["_sym", "timestamp"]))
                except Exception:
                    continue

            if not frames:
                return None
            out = pd.concat(frames)
            out.index.names = ["symbol", "timestamp"]
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
    """

    def detect(
        self,
        spy_bars: pd.DataFrame,
        vix_level: float = 18.0,
    ) -> dict:
        if spy_bars is None or len(spy_bars) < 6:
            return {
                "regime": "UNKNOWN",
                "tradeable": True,
                "top_n_limit": 20,
                "size_multiplier": 1.0,
                "reason": "Insufficient SPY data",
            }

        closes = spy_bars["close"].astype(float).values
        highs = spy_bars["high"].astype(float).values
        lows = spy_bars["low"].astype(float).values

        # 20-bar EMA
        ema20 = self._ema(closes, 20)
        above_ma = closes[-1] > ema20

        # ADX (14-bar)
        adx = self._adx(highs, lows, closes, 14)
        trending = adx > 20

        # 5-bar drift
        drift_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0

        # Classify
        if vix_level > 35:
            regime, tradeable, top_n, mult = "HIGH_FEAR", True, 5, 0.5
        elif vix_level > 28:
            regime, tradeable, top_n, mult = "ELEVATED_VOL", True, 10, 0.75
        elif trending and above_ma and drift_5 > 0:
            regime, tradeable, top_n, mult = "TRENDING_UP", True, 20, 1.0
        elif trending and (not above_ma) and drift_5 < 0:
            regime, tradeable, top_n, mult = "TRENDING_DOWN", True, 20, 1.0
        elif not trending:
            regime, tradeable, top_n, mult = "CHOPPY", False, 0, 0.0
        else:
            regime, tradeable, top_n, mult = "TRANSITIONING", True, 10, 0.6

        return {
            "regime": regime,
            "tradeable": tradeable,
            "top_n_limit": top_n,
            "size_multiplier": mult,
            "adx": round(float(adx), 1),
            "spy_vs_ema20_pct": round((closes[-1] / ema20 - 1) * 100, 2),
            "spy_drift_5h_pct": round(drift_5 * 100, 2),
            "vix": round(vix_level, 1),
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
            hl = highs[i] - lows[i]
            hpc = abs(highs[i] - closes[i - 1])
            lpc = abs(lows[i] - closes[i - 1])
            tr_list.append(max(hl, hpc, lpc))

            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            dm_p.append(up if up > down and up > 0 else 0.0)
            dm_m.append(down if down > up and down > 0 else 0.0)

        atr = float(np.mean(tr_list[-period:]))
        if atr == 0:
            return 0.0
        di_p = float(np.mean(dm_p[-period:])) / atr
        di_m = float(np.mean(dm_m[-period:])) / atr
        denom = di_p + di_m
        return 0.0 if denom == 0 else abs(di_p - di_m) / denom * 100


# ===========================================================================
# 3. SIGNAL ENGINE
# ===========================================================================


class SignalEngine:
    """
    Multi-factor signal computation + cross-sectional Z-score normalisation.

    Factor weights (chosen to be ~orthogonal):
      VWAP deviation    40%  — is price dislocated from fair value?
      Relative strength 35%  — is this stock moving vs the market?
      Volume surprise   25%  — is there real participation behind the move?

    Each factor is Z-scored cross-sectionally before weighting so a 1-point
    difference in score means the same thing regardless of the factor scale.
    """

    # Evidence-based weights (IC analysis Apr 2026, 793 trading days):
    #   imbalance_real: IC +0.16 (VIX>25), +0.05 (VIX 18-25) — regime-gated ✅
    #   pmo_crossover:  IC -0.032 p=0.005 contrarian ✅
    #   vwap_dev_intraday: IC +0.054 p=0.007 (1-min bars) ✅
    #   vol_surprise:   IC -0.018 weak negative ⚠️  (kept at minimal weight)
    #   REMOVED: vwap_dev daily proxy (IC ≈ 0), rel_strength (IC ≈ 0)
    WEIGHTS = {
        "vwap_dev_intraday": 0.35,  # IC +0.054 p=0.007 ✅ — requires 1-min bars
        "pmo_crossover": 0.25,  # IC -0.032 p=0.005 ✅ contrarian (sign flipped)
        "imbalance_real": 0.25,  # IC +0.16 (VIX>25) ✅ — regime-gated (0 when VIX<18)
        "vol_surprise": 0.15,  # IC -0.018 ⚠️ weak — kept at minimal weight
    }

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
        symbols = all_bars.index.get_level_values(0).unique().tolist()

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
            mu, sd = df[f].mean(), df[f].std()
            df[f + "_z"] = (df[f] - mu) / (sd + 1e-9)

        # Composite score
        df["score"] = sum(w * df[f + "_z"] for f, w in self.WEIGHTS.items())

        df["direction"] = np.where(df["score"] > 0, "LONG", "SHORT")

        # Readable pct columns
        df["vwap_dev_pct"] = (df["vwap_dev"] * 100).round(3)
        df["rel_strength_pct"] = (df["rel_strength"] * 100).round(3)
        df["raw_return_pct"] = (df["raw_return"] * 100).round(3)
        df["score"] = df["score"].round(4)
        df["sector"] = df["symbol"].map(lambda s: SECTOR_MAP.get(s, "Other"))

        return df.sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def _compute_one(self, symbol: str, all_bars: pd.DataFrame, spy_return: float) -> dict | None:
        try:
            bars = all_bars.xs(symbol, level=0)
            if len(bars) < 3:
                return None

            close = bars["close"].astype(float)
            high = bars["high"].astype(float)
            low = bars["low"].astype(float)
            volume = bars["volume"].astype(float)

            # Factor 1 — VWAP deviation
            typical = (high + low + close) / 3
            vwap = float((typical * volume).sum() / (volume.sum() + 1e-9))
            price = float(close.iloc[-1])
            vwap_dev = (price - vwap) / (vwap + 1e-9)

            # Factor 2 — Relative strength vs SPY
            raw_return = self._latest_return(bars)
            rel_strength = raw_return - spy_return

            # Factor 3 — Volume surprise (log ratio vs rolling avg)
            vol_vals = volume.values
            vol_avg = float(np.mean(vol_vals[:-1])) if len(vol_vals) > 1 else float(vol_vals[-1])
            vol_current = float(vol_vals[-1])
            vol_surprise = float(np.log(max(vol_current, 1) / max(vol_avg, 1)))

            return {
                "symbol": symbol,
                "price": round(price, 2),
                "volume": int(vol_current),
                "vwap": round(vwap, 2),
                "vwap_dev": vwap_dev,
                "rel_strength": rel_strength,
                "vol_surprise": vol_surprise,
                "raw_return": raw_return,
            }
        except Exception as e:
            logger.debug(f"  Signal error {symbol}: {e}")
            return None

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
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        from alpaca_trade_api import REST

        key = api_key or os.getenv("APCA_API_KEY_ID", "")
        secret = api_secret or os.getenv("APCA_API_SECRET_KEY", "")
        base = "https://paper-api.alpaca.markets"

        self.api = REST(key, secret, base_url=base)
        self.data = MarketData(self.api)
        self.regime = RegimeDetector()
        self.signals = SignalEngine()

        # Test connection
        try:
            acct = self.api.get_account()
            logger.info(f"✓ Alpaca connected — status:{acct.status}  cash:${float(acct.cash):,.0f}")
        except Exception as e:
            logger.warning(f"Alpaca connection warning: {e}")

    # ------------------------------------------------------------------
    def scan(
        self,
        symbols: list[str],
        top_n: int = 20,
        max_per_sector: int = 3,
        force: bool = False,  # ignore regime check
    ) -> dict:
        t0 = datetime.now()
        logger.info(f"V2 scan starting — {len(symbols)} symbols")

        # ── 1. SPY + VIX reference ────────────────────────────────────
        spy_bars = self.data.fetch_single("SPY", lookback_days=3)
        vix_level = 18.0
        try:
            vix_bars = self.data.fetch_single("VIX", lookback_days=2)
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
            logger.warning(
                f"Market is {regime['regime']} — skipping scan (use force=True to override)"
            )
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
        limit = min(top_n, regime["top_n_limit"])
        top_long = self._select(sig_df, "LONG", limit, max_per_sector)
        top_short = self._select(sig_df, "SHORT", limit, max_per_sector)

        # ── 6. Consensus — all 3 factor z-scores agree ────────────────
        # A consensus signal requires:
        #   • |score| > 0.5  (meaningfully above average)
        #   • all 3 z-scores positive (LONG) or all negative (SHORT)
        mask_agree = (
            (sig_df["vwap_dev_z"] * sig_df["rel_strength_z"] > 0)
            & (sig_df["rel_strength_z"] * sig_df["vol_surprise_z"] > 0)
            & (sig_df["score"].abs() > 0.5)
        )
        consensus = sig_df.loc[mask_agree, "symbol"].tolist()

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f"✓ Done in {elapsed:.1f}s  "
            f"longs:{len(top_long)}  shorts:{len(top_short)}  consensus:{len(consensus)}"
        )

        return {
            "regime": regime,
            "signals": sig_df,
            "top_long": top_long,
            "top_short": top_short,
            "consensus": consensus,
            "spy_return": self.signals._latest_return(spy_bars),
            "elapsed": elapsed,
            "timestamp": datetime.now().isoformat(),
            "symbols_scanned": fetched,
        }

    # ------------------------------------------------------------------
    def _select(
        self,
        df: pd.DataFrame,
        direction: str,
        top_n: int,
        max_per_sector: int,
    ) -> list[dict]:
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
            "regime": regime,
            "signals": pd.DataFrame(),
            "top_long": [],
            "top_short": [],
            "consensus": [],
            "spy_return": 0.0,
            "elapsed": (datetime.now() - t0).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "symbols_scanned": 0,
        }
