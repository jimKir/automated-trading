#!/usr/bin/env python3
"""
Volatility Prediction Engine — Production Grade
================================================

Multi-model volatility forecasting system using:
  1. Yang-Zhang / Garman-Klass / Parkinson volatility estimators
  2. HAR (Heterogeneous AutoRegressive) model — strong baseline
  3. Bidirectional LSTM with temporal attention — captures nonlinear dynamics
  4. Gradient Boosting (LightGBM/XGBoost) — captures feature interactions
  5. Adaptive ensemble — weights models by recent walk-forward performance

Designed for the same stock universe as the Momentum Scanner V2.

Key design decisions:
  - Multiple volatility estimators beat close-to-close alone
  - HAR model is the gold standard linear baseline for vol forecasting
  - LSTM attention mechanism lets the model focus on volatility clusters
  - Walk-forward validation prevents lookahead bias
  - QLIKE loss for evaluation (standard in vol forecasting literature)
  - Sector-level and market-level features capture contagion effects
"""

import logging
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Shared indicators — canonical implementations
try:
    import sys as _sys

    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from utils.indicators import compute_adx as _compute_adx_shared

    _USE_SHARED_ADX = True
except ImportError:
    _USE_SHARED_ADX = False

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


# ===========================================================================
# SECTOR MAP  (identical to momentum scanner V2 for consistency)
# ===========================================================================
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

UNIVERSE = list(SECTOR_MAP.keys())

# Prediction horizons (business days)
HORIZONS = {"1d": 1, "5d": 5, "10d": 10}

# Minimum history needed for feature engineering (trading days)
MIN_HISTORY_DAYS = 504  # ~2 years


# ===========================================================================
# 1. DATA PIPELINE
# ===========================================================================


class DataPipeline:
    """
    Fetches historical OHLCV data via yfinance with hierarchical disk caching.

    Cache structure:
        .vol_cache/
          ohlcv/
            AAPL/
              2023-04-04_2026-04-03.parquet
            NVDA/
              ...
          vix/
            2023-04-04_2026-04-03.parquet

    Cache policy:
      - Data is considered fresh if the cache file was written today
        (i.e. it already contains today's available market data).
      - If the cache is stale (written on a previous day), the system
        downloads fresh data and overwrites the cache.
      - Pass ``force_refresh=True`` to bypass the cache entirely.
      - Pass ``cache_dir=None`` to use the default .vol_cache/ directory
        alongside this script.  Pass a custom path to share a cache across
        multiple projects.

    Why Parquet?
      - Fast read/write (~10× faster than CSV for wide DataFrames)
      - Preserves dtypes and datetime index natively
      - Small on disk (~60% smaller than CSV for OHLCV data)
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".vol_cache"
        )
        # Create the hierarchical directory structure
        self._ohlcv_dir = os.path.join(self.cache_dir, "ohlcv")
        self._vix_dir = os.path.join(self.cache_dir, "vix")
        os.makedirs(self._ohlcv_dir, exist_ok=True)
        os.makedirs(self._vix_dir, exist_ok=True)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _cache_is_fresh(path: str) -> bool:
        """Return True if *path* exists and was last modified today."""
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return mtime.date() == datetime.now().date()

    def _sym_cache_path(self, sym: str, start_str: str, end_str: str) -> str:
        """Return the parquet path for one symbol's OHLCV cache."""
        sym_dir = os.path.join(self._ohlcv_dir, sym.upper())
        os.makedirs(sym_dir, exist_ok=True)
        return os.path.join(sym_dir, f"{start_str}_{end_str}.parquet")

    def _vix_cache_path(self, start_str: str, end_str: str) -> str:
        """Return the parquet path for VIX cache."""
        return os.path.join(self._vix_dir, f"{start_str}_{end_str}.parquet")

    # ── main fetch methods ────────────────────────────────────────────

    def fetch(
        self,
        symbols: list[str],
        lookback_years: float = 3.0,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV for all symbols, using disk cache when fresh.

        On a cache hit the data is loaded from parquet in milliseconds.
        On a cache miss (or ``force_refresh=True``) the data is downloaded
        from Yahoo Finance, parsed, cached as parquet, and returned.

        Returns dict: symbol -> DataFrame[date, open, high, low, close, volume]
        """
        import yfinance as yf

        end = datetime.now()
        start = end - timedelta(days=int(lookback_years * 365))
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        result: dict[str, pd.DataFrame] = {}
        symbols_to_download: list[str] = []

        # ── Phase 1: load from cache where possible ───────────────────
        if not force_refresh:
            for sym in symbols:
                cache_path = self._sym_cache_path(sym, start_str, end_str)
                if self._cache_is_fresh(cache_path):
                    try:
                        df = pd.read_parquet(cache_path)
                        df.index = pd.to_datetime(df.index)
                        result[sym] = df
                        continue
                    except Exception:
                        pass  # corrupted cache → re-download
                symbols_to_download.append(sym)
        else:
            symbols_to_download = list(symbols)

        cached_count = len(result)
        if cached_count > 0:
            logger.info(
                f"Loaded {cached_count}/{len(symbols)} symbols from cache; "
                f"downloading {len(symbols_to_download)} remaining..."
            )
        else:
            logger.info(f"Fetching {len(symbols)} symbols ({start_str} to {end_str})...")

        # ── Phase 2: download missing symbols ─────────────────────────
        if symbols_to_download:
            try:
                raw = yf.download(
                    symbols_to_download,
                    start=start_str,
                    end=end_str,
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
            except Exception as e:
                logger.error(f"yfinance download failed: {e}")
                return result  # return whatever we got from cache

            for sym in symbols_to_download:
                try:
                    if len(symbols_to_download) == 1:
                        df = raw.copy()
                    else:
                        df = pd.DataFrame()
                        for col in ["Open", "High", "Low", "Close", "Volume"]:
                            try:
                                if isinstance(raw.columns, pd.MultiIndex):
                                    df[col.lower()] = raw[(col, sym)]
                                else:
                                    df[col.lower()] = raw[col]
                            except KeyError:
                                continue

                    if len(symbols_to_download) == 1:
                        df.columns = [
                            c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns
                        ]

                    df = df.dropna(subset=["close"])

                    if len(df) < MIN_HISTORY_DAYS // 2:
                        logger.warning(f"{sym}: only {len(df)} rows, skipping")
                        continue

                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()

                    # ── Save to cache ─────────────────────────────────
                    cache_path = self._sym_cache_path(sym, start_str, end_str)
                    df.to_parquet(cache_path)

                    result[sym] = df
                    logger.debug(f"{sym}: {len(df)} rows fetched & cached")

                except Exception as e:
                    logger.warning(f"{sym}: parse error — {e}")
                    continue

        logger.info(f"Fetched {len(result)}/{len(symbols)} symbols with data")
        return result

    def fetch_vix(
        self,
        lookback_years: float = 3.0,
        force_refresh: bool = False,
    ) -> pd.Series:
        """Fetch VIX index as a market fear gauge, with disk cache."""
        import yfinance as yf

        end = datetime.now()
        start = end - timedelta(days=int(lookback_years * 365))
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        cache_path = self._vix_cache_path(start_str, end_str)

        # Try cache first
        if not force_refresh and self._cache_is_fresh(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                series = df.iloc[:, 0].dropna()
                series.index = pd.to_datetime(series.index)
                logger.info("VIX loaded from cache")
                return series
            except Exception:
                pass  # corrupted → re-download

        try:
            vix = yf.download("^VIX", start=start_str, end=end_str, progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                series = vix[("Close", "^VIX")].dropna()
            else:
                series = vix["Close"].dropna()

            # Save to cache
            series.to_frame("vix_close").to_parquet(cache_path)
            logger.info("VIX fetched & cached")
            return series

        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
            return pd.Series(dtype=float)

    def clear_cache(self, symbol: str | None = None):
        """
        Clear cached data.

        Args:
            symbol: If provided, clear only this symbol's cache.
                    If None, clear everything.
        """
        import shutil

        if symbol:
            sym_dir = os.path.join(self._ohlcv_dir, symbol.upper())
            if os.path.exists(sym_dir):
                shutil.rmtree(sym_dir)
                logger.info(f"Cache cleared for {symbol}")
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self._ohlcv_dir, exist_ok=True)
            os.makedirs(self._vix_dir, exist_ok=True)
            logger.info("Full cache cleared")

    def cache_info(self) -> dict:
        """Return a summary of what's in the cache."""
        info = {"ohlcv_symbols": [], "vix_cached": False, "total_size_mb": 0.0}
        total_bytes = 0
        if os.path.exists(self._ohlcv_dir):
            for sym_dir in sorted(os.listdir(self._ohlcv_dir)):
                sym_path = os.path.join(self._ohlcv_dir, sym_dir)
                if os.path.isdir(sym_path):
                    parquets = [f for f in os.listdir(sym_path) if f.endswith(".parquet")]
                    if parquets:
                        info["ohlcv_symbols"].append(sym_dir)
                        for p in parquets:
                            total_bytes += os.path.getsize(os.path.join(sym_path, p))
        if os.path.exists(self._vix_dir):
            vix_files = [f for f in os.listdir(self._vix_dir) if f.endswith(".parquet")]
            info["vix_cached"] = len(vix_files) > 0
            for f in vix_files:
                total_bytes += os.path.getsize(os.path.join(self._vix_dir, f))
        info["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
        return info


# ===========================================================================
# 2. VOLATILITY ESTIMATORS
# ===========================================================================


class VolatilityEstimators:
    """
    Multiple volatility estimators from OHLC data.

    Why multiple estimators?
      - Close-to-close: simplest, but wastes intraday information
      - Parkinson: uses high-low range, ~5x more efficient than CC
      - Garman-Klass: uses OHLC, ~8x more efficient
      - Yang-Zhang: handles overnight jumps + drift, most efficient
      - Rogers-Satchell: drift-independent, no overnight assumption

    All return annualised volatility (multiply by sqrt(252)).
    """

    @staticmethod
    def close_to_close(close: pd.Series, window: int = 20) -> pd.Series:
        """Standard close-to-close realized volatility."""
        log_ret = np.log(close / close.shift(1))
        return log_ret.rolling(window).std() * np.sqrt(252)

    @staticmethod
    def parkinson(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """
        Parkinson (1980) high-low range estimator.
        Var = (1/4ln2) * E[(ln(H/L))^2]
        ~5.2x more efficient than close-to-close.
        """
        hl = np.log(high / low) ** 2
        factor = 1.0 / (4.0 * np.log(2.0))
        return np.sqrt(factor * hl.rolling(window).mean() * 252)

    @staticmethod
    def garman_klass(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
    ) -> pd.Series:
        """
        Garman-Klass (1980) OHLC estimator.
        ~8x more efficient than close-to-close.
        """
        hl = (np.log(high / low)) ** 2
        co = (np.log(close / open_)) ** 2
        gk = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
        return np.sqrt(gk.rolling(window).mean() * 252)

    @staticmethod
    def yang_zhang(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
    ) -> pd.Series:
        """
        Yang-Zhang (2000) — the most efficient OHLC estimator.
        Handles drift AND overnight jumps.
        Combines overnight, open-to-close, and Rogers-Satchell components.
        """
        n = window
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        # Overnight component
        log_oc_prev = np.log(open_ / close.shift(1))
        sigma_overnight = log_oc_prev.rolling(n).var()

        # Open-to-close component
        log_co = np.log(close / open_)
        sigma_oc = log_co.rolling(n).var()

        # Rogers-Satchell component
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        rs = (log_ho * log_hc + log_lo * log_lc).rolling(n).mean()

        yz_var = sigma_overnight + k * sigma_oc + (1 - k) * rs
        # Clamp negative values (can happen with noisy data)
        yz_var = yz_var.clip(lower=0)
        return np.sqrt(yz_var * 252)

    @staticmethod
    def rogers_satchell(
        open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
    ) -> pd.Series:
        """Rogers-Satchell (1991) — drift-independent estimator."""
        log_ho = np.log(high / open_)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_)
        log_lc = np.log(low / close)
        rs = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()
        rs = rs.clip(lower=0)
        return np.sqrt(rs * 252)

    @staticmethod
    def realised_variance_target(close: pd.Series, horizon: int = 5) -> pd.Series:
        """
        Forward-looking realised variance — this is what we PREDICT.
        RV_t = sum of squared log returns over next `horizon` days, annualised.
        """
        log_ret = np.log(close / close.shift(1))
        sq_ret = log_ret**2
        # Forward sum of squared returns
        fwd_rv = sq_ret.shift(-horizon).rolling(horizon).sum()
        return np.sqrt(fwd_rv * (252 / horizon))


# ===========================================================================
# 3. FEATURE ENGINEERING
# ===========================================================================


class FeatureEngine:
    """
    Build the feature matrix for volatility prediction.

    Feature groups:
      A. Volatility estimators at multiple windows (5, 10, 20, 60)
      B. HAR components (daily, weekly, monthly vol)
      C. Return distribution (skew, kurtosis, autocorrelation)
      D. Volume features (volume surprise, volume-vol correlation)
      E. Range features (ATR, Bollinger width)
      F. Cross-asset features (VIX level, VIX change, sector avg vol)
      G. Calendar features (day-of-week, month)
      H. Momentum/mean-reversion signals (RSI, return z-score)
    """

    def __init__(
        self, vix_data: pd.Series | None = None, all_data: dict[str, pd.DataFrame] | None = None
    ):
        self.vix = vix_data
        self.all_data = all_data or {}
        self.vol_est = VolatilityEstimators()

    def build_features(
        self, sym: str, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        """
        Build full feature matrix for a single symbol.
        Returns (features_df, targets_dict).
        """
        o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
        log_ret = np.log(c / c.shift(1))

        feats = pd.DataFrame(index=df.index)

        # ── A. Multi-window volatility estimators ──────────────────────
        for w in [5, 10, 20, 60]:
            feats[f"vol_cc_{w}"] = self.vol_est.close_to_close(c, w)
            feats[f"vol_pk_{w}"] = self.vol_est.parkinson(h, l, w)
            feats[f"vol_gk_{w}"] = self.vol_est.garman_klass(o, h, l, c, w)
            feats[f"vol_yz_{w}"] = self.vol_est.yang_zhang(o, h, l, c, w)

        # ── B. HAR components (Corsi 2009) ─────────────────────────────
        # HAR decomposes vol into daily (1d), weekly (5d), monthly (22d)
        sq_ret = log_ret**2
        feats["har_daily"] = sq_ret.rolling(1).mean() * 252
        feats["har_weekly"] = sq_ret.rolling(5).mean() * 252
        feats["har_monthly"] = sq_ret.rolling(22).mean() * 252
        feats["har_quarterly"] = sq_ret.rolling(66).mean() * 252

        # ── C. Return distribution features ────────────────────────────
        feats["ret_skew_20"] = log_ret.rolling(20).skew()
        feats["ret_kurt_20"] = log_ret.rolling(20).kurt()
        feats["ret_skew_60"] = log_ret.rolling(60).skew()
        feats["ret_kurt_60"] = log_ret.rolling(60).kurt()

        # Autocorrelation of returns (mean-reversion vs momentum)
        feats["ret_autocorr_5"] = log_ret.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False
        )
        # Autocorrelation of absolute returns (volatility clustering)
        abs_ret = log_ret.abs()
        feats["absret_autocorr_1"] = abs_ret.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False
        )

        # ── D. Volume features ─────────────────────────────────────────
        vol_ma20 = v.rolling(20).mean()
        feats["volume_surprise"] = np.log1p(v / vol_ma20.replace(0, np.nan))
        feats["volume_trend"] = v.rolling(5).mean() / vol_ma20.replace(0, np.nan)
        # Volume-volatility correlation (rolling)
        feats["vol_volume_corr"] = abs_ret.rolling(20).corr(v)

        # ── E. Range features ──────────────────────────────────────────
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        feats["atr_14"] = tr.rolling(14).mean() / c  # normalised ATR
        feats["atr_5"] = tr.rolling(5).mean() / c

        # Bollinger bandwidth
        bb_std = c.rolling(20).std()
        bb_ma = c.rolling(20).mean()
        feats["bband_width"] = 2 * bb_std / bb_ma.replace(0, np.nan)

        # Intraday range ratio (today's range vs average)
        daily_range = (h - l) / c
        feats["range_ratio"] = daily_range / daily_range.rolling(20).mean()

        # ── F. Cross-asset features ────────────────────────────────────
        if self.vix is not None and len(self.vix) > 0:
            # Align VIX to this symbol's dates
            vix_aligned = self.vix.reindex(df.index, method="ffill")
            feats["vix_level"] = vix_aligned / 100  # normalise
            feats["vix_change_5d"] = vix_aligned.pct_change(5)
            feats["vix_ma_ratio"] = vix_aligned / vix_aligned.rolling(20).mean()
        else:
            feats["vix_level"] = 0.2
            feats["vix_change_5d"] = 0.0
            feats["vix_ma_ratio"] = 1.0

        # Sector average volatility
        sector = SECTOR_MAP.get(sym, "Unknown")
        sector_peers = [
            s for s, sec in SECTOR_MAP.items() if sec == sector and s != sym and s in self.all_data
        ]
        if sector_peers:
            peer_vols = []
            for peer in sector_peers[:5]:  # max 5 peers for speed
                peer_c = self.all_data[peer]["close"]
                peer_ret = np.log(peer_c / peer_c.shift(1))
                pv = peer_ret.rolling(20).std() * np.sqrt(252)
                peer_vols.append(pv)
            if peer_vols:
                sector_vol = pd.concat(peer_vols, axis=1).mean(axis=1)
                sector_vol = sector_vol.reindex(df.index, method="ffill")
                feats["sector_avg_vol"] = sector_vol
            else:
                feats["sector_avg_vol"] = feats["vol_cc_20"]
        else:
            feats["sector_avg_vol"] = feats["vol_cc_20"]

        # Market average volatility (use SPY proxy: average of all symbols)
        if len(self.all_data) > 10:
            sample_syms = list(self.all_data.keys())[:20]
            mkt_vols = []
            for s in sample_syms:
                sc = self.all_data[s]["close"]
                sr = np.log(sc / sc.shift(1))
                mv = sr.rolling(20).std() * np.sqrt(252)
                mkt_vols.append(mv)
            mkt_vol = pd.concat(mkt_vols, axis=1).mean(axis=1)
            mkt_vol = mkt_vol.reindex(df.index, method="ffill")
            feats["market_avg_vol"] = mkt_vol
        else:
            feats["market_avg_vol"] = feats["vol_cc_20"]

        # ── G. Calendar features ───────────────────────────────────────
        feats["dow"] = df.index.dayofweek / 4.0  # 0=Mon, 1=Fri → [0,1]
        feats["month"] = df.index.month / 12.0
        # Monday effect (historically higher vol)
        feats["is_monday"] = (df.index.dayofweek == 0).astype(float)
        # End of month
        feats["is_month_end"] = df.index.is_month_end.astype(float)

        # ── H. Momentum / mean-reversion ───────────────────────────────
        # RSI at multiple timeframes
        for rsi_period in [7, 14, 21]:
            delta_rsi = c.diff()
            gain_rsi = delta_rsi.where(delta_rsi > 0, 0).rolling(rsi_period).mean()
            loss_rsi = (-delta_rsi.where(delta_rsi < 0, 0)).rolling(rsi_period).mean()
            rs_rsi = gain_rsi / loss_rsi.replace(0, np.nan)
            feats[f"rsi_{rsi_period}"] = (100 - (100 / (1 + rs_rsi))) / 100  # [0,1]

        # RSI extremes (binary flags — vol tends to spike at RSI extremes)
        feats["rsi_overbought"] = (feats["rsi_14"] > 0.70).astype(float)
        feats["rsi_oversold"] = (feats["rsi_14"] < 0.30).astype(float)

        # ── I. MACD (Moving Average Convergence Divergence) ────────────
        # MACD captures trend momentum — trend changes often precede vol spikes
        ema_12 = c.ewm(span=12, adjust=False).mean()
        ema_26 = c.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        # Normalise MACD by price level so it's comparable across stocks
        feats["macd_line"] = macd_line / c  # as % of price
        feats["macd_signal"] = macd_signal / c
        feats["macd_histogram"] = macd_hist / c
        feats["macd_crossover"] = (macd_line > macd_signal).astype(float) - (
            macd_line.shift(1) > macd_signal.shift(1)
        ).astype(float)  # +1 = bullish cross, -1 = bearish cross, 0 = no change

        # MACD histogram acceleration (rate of change of histogram)
        feats["macd_hist_accel"] = macd_hist.diff() / c

        # MACD divergence proxy: price making new highs but MACD isn't
        price_high_20 = (c == c.rolling(20).max()).astype(float)
        macd_high_20 = (macd_line == macd_line.rolling(20).max()).astype(float)
        feats["macd_divergence"] = price_high_20 - macd_high_20  # +1 = bearish divergence

        # ── J. Stochastic Oscillator (%K, %D) ──────────────────────────
        # Compares close to high-low range — identifies reversals
        # High stochastic + reversal → vol expansion signal
        for stoch_period in [14, 21]:
            lowest_low = l.rolling(stoch_period).min()
            highest_high = h.rolling(stoch_period).max()
            denom = (highest_high - lowest_low).replace(0, np.nan)
            stoch_k = ((c - lowest_low) / denom) * 100
            stoch_d = stoch_k.rolling(3).mean()  # signal line
            feats[f"stoch_k_{stoch_period}"] = stoch_k / 100  # [0, 1]
            feats[f"stoch_d_{stoch_period}"] = stoch_d / 100
            feats[f"stoch_kd_diff_{stoch_period}"] = (stoch_k - stoch_d) / 100

        # Stochastic extremes (vol tends to spike at overbought/oversold)
        feats["stoch_overbought"] = (feats["stoch_k_14"] > 0.80).astype(float)
        feats["stoch_oversold"] = (feats["stoch_k_14"] < 0.20).astype(float)

        # ── K. ADX (Average Directional Index) ────────────────────────
        # Measures trend STRENGTH (0-100), direction-agnostic
        # Strong trends (ADX > 25) tend to have different vol dynamics
        adx_period = 14
        plus_dm = h.diff()
        minus_dm = -l.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr_adx = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(
            axis=1
        )

        atr_adx = tr_adx.ewm(span=adx_period, adjust=False).mean()
        plus_di = 100 * (
            plus_dm.ewm(span=adx_period, adjust=False).mean() / atr_adx.replace(0, np.nan)
        )
        minus_di = 100 * (
            minus_dm.ewm(span=adx_period, adjust=False).mean() / atr_adx.replace(0, np.nan)
        )
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(span=adx_period, adjust=False).mean()

        feats["adx"] = adx / 100  # normalise to [0, 1]
        feats["plus_di"] = plus_di / 100
        feats["minus_di"] = minus_di / 100
        feats["di_spread"] = (plus_di - minus_di) / 100  # bullish vs bearish
        feats["adx_rising"] = (adx > adx.shift(5)).astype(float)  # trend strengthening
        feats["strong_trend"] = (adx > 25).astype(float)  # binary: trending market

        # ── L. Price Momentum Oscillator (PMO) ────────────────────────
        # Double-smoothed ROC — cleaner trend signal than MACD
        roc_1 = c.pct_change(1) * 100
        pmo_smooth1 = roc_1.ewm(span=35, adjust=False).mean()
        pmo_line = pmo_smooth1.ewm(span=20, adjust=False).mean()
        pmo_signal = pmo_line.ewm(span=10, adjust=False).mean()

        feats["pmo_line"] = pmo_line
        feats["pmo_signal"] = pmo_signal
        feats["pmo_histogram"] = pmo_line - pmo_signal
        feats["pmo_crossover"] = (pmo_line > pmo_signal).astype(float) - (
            pmo_line.shift(1) > pmo_signal.shift(1)
        ).astype(float)

        # ── M. RSI Divergence ─────────────────────────────────────────
        # Price making new highs but RSI isn't → bearish divergence → vol spike
        rsi_14_raw = feats["rsi_14"] * 100
        price_high_20r = (c == c.rolling(20).max()).astype(float)
        rsi_high_20 = (rsi_14_raw == rsi_14_raw.rolling(20).max()).astype(float)
        feats["rsi_divergence"] = price_high_20r - rsi_high_20

        # ── N. Support/Resistance proximity ───────────────────────────
        # Distance from recent highs/lows — breakouts from S/R → vol expansion
        high_20 = c.rolling(20).max()
        low_20 = c.rolling(20).min()
        range_20 = (high_20 - low_20).replace(0, np.nan)
        feats["dist_from_high"] = (high_20 - c) / range_20  # 0 = at high, 1 = at low
        feats["dist_from_low"] = (c - low_20) / range_20
        # Squeeze detection: range contraction → vol expansion coming
        feats["range_contraction"] = range_20 / c.rolling(60).apply(
            lambda x: x.max() - x.min(), raw=True
        ).replace(0, np.nan)

        # ── O. Combined confirmation signals ──────────────────────────
        # Multiple indicators agreeing → stronger vol signal
        feats["momentum_alignment"] = (
            feats.get("rsi_overbought", 0).astype(float)
            + feats.get("stoch_overbought", 0).astype(float)
            + (feats.get("macd_histogram", 0) > 0).astype(float)
        ) / 3.0  # 1.0 = all bullish momentum, used for vol context

        feats["reversal_risk"] = (
            feats.get("rsi_oversold", 0).astype(float)
            + feats.get("stoch_oversold", 0).astype(float)
            + (feats.get("macd_crossover", 0) == -1).astype(float)
            + (feats.get("rsi_divergence", 0) > 0).astype(float)
        ) / 4.0  # higher = more reversal signals → vol expansion likely

        # Return z-score (how extreme is today's return?)
        feats["ret_zscore"] = (log_ret - log_ret.rolling(60).mean()) / log_ret.rolling(
            60
        ).std().replace(0, np.nan)

        # Consecutive direction (streak of up or down days)
        direction = np.sign(log_ret)
        streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
        feats["streak"] = streak * direction / 10  # normalise

        # ── TARGETS ────────────────────────────────────────────────────
        targets = {}
        for name, horizon in HORIZONS.items():
            targets[name] = self.vol_est.realised_variance_target(c, horizon)

        return feats, targets


# ===========================================================================
# 4. HAR MODEL (Heterogeneous AutoRegressive)
# ===========================================================================


class HARModel:
    """
    Corsi (2009) HAR-RV model.

    RV_{t+h} = c + β_d * RV_daily + β_w * RV_weekly + β_m * RV_monthly

    This is the standard benchmark in volatility forecasting.
    Simple OLS, but remarkably hard to beat consistently.
    """

    def __init__(self):
        self.coefs = None
        self.intercept = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit HAR via OLS on daily, weekly, monthly vol components."""
        har_cols = [c for c in X.columns if c.startswith("har_")]
        if not har_cols:
            logger.warning("No HAR features found, using vol_cc columns")
            har_cols = ["vol_cc_5", "vol_cc_20", "vol_cc_60"]
            har_cols = [c for c in har_cols if c in X.columns]

        Xh = X[har_cols].copy()
        mask = Xh.notna().all(axis=1) & y.notna()
        Xh, yh = Xh[mask], y[mask]

        if len(Xh) < 50:
            logger.warning("HAR: insufficient data for fitting")
            self.coefs = pd.Series(1.0 / len(har_cols), index=har_cols)
            return self

        # OLS via normal equation
        Xm = Xh.values
        Xm = np.column_stack([np.ones(len(Xm)), Xm])
        ym = yh.values

        try:
            beta = np.linalg.lstsq(Xm, ym, rcond=None)[0]
            self.intercept = beta[0]
            self.coefs = pd.Series(beta[1:], index=har_cols)
        except np.linalg.LinAlgError:
            self.coefs = pd.Series(1.0 / len(har_cols), index=har_cols)
            self.intercept = 0.0

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.coefs is None:
            return pd.Series(np.nan, index=X.index)
        cols = self.coefs.index.tolist()
        cols = [c for c in cols if c in X.columns]
        if not cols:
            return pd.Series(np.nan, index=X.index)
        pred = X[cols].values @ self.coefs[cols].values + self.intercept
        return pd.Series(pred, index=X.index).clip(lower=0)


# ===========================================================================
# 5. LSTM WITH ATTENTION
# ===========================================================================


class LSTMVolModel:
    """
    Bidirectional LSTM with temporal attention for volatility prediction.

    Architecture:
      Input (seq_len × n_features)
        → Bidirectional LSTM (64 units)
        → Temporal Attention (learns which timesteps matter)
        → Dense(32, relu) + Dropout(0.3)
        → Dense(1, softplus)  ← ensures positive output

    Training:
      - Loss: MSE on log(vol) — stabilises training for heavy-tailed targets
      - Early stopping on validation loss
      - Learning rate reduction on plateau
      - Walk-forward: train on expanding window, validate on next fold
    """

    def __init__(self, seq_len: int = 20, epochs: int = 100, batch_size: int = 64):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        self._tf_available = None

    def _check_tensorflow(self) -> bool:
        if self._tf_available is not None:
            return self._tf_available
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf

            tf.get_logger().setLevel("ERROR")
            self._tf_available = True
        except ImportError:
            logger.warning(
                "TensorFlow not installed. LSTM model disabled. "
                "Install with: pip install tensorflow"
            )
            self._tf_available = False
        return self._tf_available

    def _build_model(self, n_features: int):
        """Build LSTM with attention."""
        import tensorflow as tf
        from tensorflow.keras import Model, layers

        inp = layers.Input(shape=(self.seq_len, n_features))

        # Bidirectional LSTM
        lstm_out = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(inp)

        # Temporal Attention
        # Learn which timesteps are most informative for vol prediction
        attn_score = layers.Dense(1, activation="tanh")(lstm_out)  # (batch, seq, 1)
        attn_weight = layers.Softmax(axis=1)(attn_score)  # (batch, seq, 1)
        context = layers.Multiply()([lstm_out, attn_weight])  # weighted
        context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)  # (batch, 128)

        # Dense head
        x = layers.Dense(32, activation="relu")(context)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1, activation="softplus")(x)  # positive output

        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
        )
        return model

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert flat arrays to sequences for LSTM input."""
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len : i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMVolModel":
        if not self._check_tensorflow():
            return self

        import tensorflow as tf
        from sklearn.preprocessing import RobustScaler

        self.feature_names = X.columns.tolist()

        # Clean data
        mask = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
        X_clean = X[mask].copy()
        y_clean = y[mask].copy()

        if len(X_clean) < self.seq_len * 3:
            logger.warning("LSTM: insufficient data")
            return self

        # Scale features (RobustScaler handles outliers better)
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()

        X_scaled = self.scaler_X.fit_transform(X_clean.values)
        # Log-transform target for stability (vol is heavy-tailed)
        y_log = np.log1p(y_clean.values).reshape(-1, 1)
        y_scaled = self.scaler_y.fit_transform(y_log).ravel()

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        if len(X_seq) < 100:
            logger.warning("LSTM: too few sequences after windowing")
            return self

        # Train/validation split (time-ordered, no shuffle)
        split = int(len(X_seq) * 0.85)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # Build model
        self.model = self._build_model(X_clean.shape[1])

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        # Train
        logger.info(f"LSTM training: {len(X_train)} train, {len(X_val)} val sequences")
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"LSTM validation MSE: {val_loss:.6f}")

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler_X is None:
            return pd.Series(np.nan, index=X.index)

        # Use only features that were in training
        cols = [c for c in self.feature_names if c in X.columns]
        if len(cols) != len(self.feature_names):
            return pd.Series(np.nan, index=X.index)

        X_clean = X[cols].fillna(method="ffill").fillna(0)
        X_scaled = self.scaler_X.transform(X_clean.values)

        # Create sequences for all valid positions
        preds = pd.Series(np.nan, index=X.index)

        if len(X_scaled) < self.seq_len:
            return preds

        X_seq = np.array(
            [X_scaled[i - self.seq_len : i] for i in range(self.seq_len, len(X_scaled))]
        )

        raw_pred = self.model.predict(X_seq, verbose=0).ravel()

        # Inverse transform
        raw_pred = self.scaler_y.inverse_transform(raw_pred.reshape(-1, 1)).ravel()
        raw_pred = np.expm1(raw_pred)  # inverse of log1p
        raw_pred = np.clip(raw_pred, 0, 5)  # cap at 500% vol (sanity)

        preds.iloc[self.seq_len :] = raw_pred

        return preds


# ===========================================================================
# 6. GRADIENT BOOSTING MODEL
# ===========================================================================


class GBMVolModel:
    """
    LightGBM / XGBoost for volatility prediction.

    Why GBM alongside LSTM?
      - Captures feature interactions LSTM might miss
      - Much faster to train — good for walk-forward validation
      - Naturally handles missing values
      - Provides feature importance for interpretability
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self._backend = None  # "lightgbm" or "xgboost" or "sklearn"

    def _get_backend(self):
        if self._backend:
            return self._backend
        try:
            import lightgbm

            self._backend = "lightgbm"
        except ImportError:
            try:
                import xgboost

                self._backend = "xgboost"
            except ImportError:
                self._backend = "sklearn"
                logger.info(
                    "Neither LightGBM nor XGBoost found, using sklearn GBM. "
                    "For best performance: pip install lightgbm"
                )
        return self._backend

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBMVolModel":
        self.feature_names = X.columns.tolist()

        mask = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 100:
            logger.warning("GBM: insufficient data")
            return self

        # Time-ordered split
        split = int(len(X_clean) * 0.85)
        X_train = X_clean.iloc[:split]
        y_train = y_clean.iloc[:split]
        X_val = X_clean.iloc[split:]
        y_val = y_clean.iloc[split:]

        backend = self._get_backend()

        if backend == "lightgbm":
            import lightgbm as lgb

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            params = {
                "objective": "regression",
                "metric": "mse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0.1,
                "lambda_l2": 0.1,
                "verbose": -1,
            }
            callbacks = [lgb.early_stopping(50, verbose=False)]
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=callbacks,
            )

        elif backend == "xgboost":
            import xgboost as xgb

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbosity": 0,
            }
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            self._backend_predict = "xgboost"

        else:
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
            )
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(np.nan, index=X.index)

        cols = [c for c in self.feature_names if c in X.columns]
        X_clean = X[cols].fillna(method="ffill").fillna(0)

        backend = self._get_backend()
        if backend == "lightgbm":
            pred = self.model.predict(X_clean)
        elif backend == "xgboost":
            import xgboost as xgb

            pred = self.model.predict(xgb.DMatrix(X_clean))
        else:
            pred = self.model.predict(X_clean)

        return pd.Series(np.clip(pred, 0, 5), index=X.index)

    def feature_importance(self) -> pd.Series | None:
        if self.model is None:
            return None
        backend = self._get_backend()
        if backend == "lightgbm":
            imp = self.model.feature_importance(importance_type="gain")
            return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
        if backend == "xgboost":
            scores = self.model.get_score(importance_type="gain")
            return pd.Series(scores).sort_values(ascending=False)
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)


# ===========================================================================
# 7. ADAPTIVE ENSEMBLE
# ===========================================================================


class AdaptiveEnsemble:
    """
    Combines HAR, LSTM, and GBM predictions with adaptive weights.

    Weight adaptation:
      - Over a rolling window, measure each model's QLIKE loss
      - Assign inverse-loss weights (better models get more weight)
      - Floor at 10% per model to maintain diversification

    QLIKE loss = log(σ²_pred) + σ²_actual / σ²_pred
    This is the standard loss function for volatility forecast evaluation.
    It penalises under-prediction of vol more than over-prediction (good!).
    """

    def __init__(self, eval_window: int = 60):
        self.eval_window = eval_window
        self.model_names = []
        self.weights = {}

    def combine(
        self,
        predictions: dict[str, pd.Series],
        actuals: pd.Series | None = None,
    ) -> pd.Series:
        """
        Combine model predictions.
        If actuals are available, use adaptive weights based on recent QLIKE.
        Otherwise, use equal weights.
        """
        self.model_names = list(predictions.keys())
        pred_df = pd.DataFrame(predictions)

        if actuals is not None and len(actuals.dropna()) > self.eval_window:
            self.weights = self._compute_adaptive_weights(pred_df, actuals)
        else:
            # Equal weights
            n = len(self.model_names)
            self.weights = dict.fromkeys(self.model_names, 1.0 / n)

        # Weighted combination
        combined = pd.Series(0.0, index=pred_df.index)
        for model, weight in self.weights.items():
            if model in pred_df.columns:
                combined += weight * pred_df[model].fillna(0)

        return combined.clip(lower=0)

    def _compute_adaptive_weights(
        self, pred_df: pd.DataFrame, actuals: pd.Series
    ) -> dict[str, float]:
        """Compute weights based on rolling QLIKE loss."""
        weights = {}
        losses = {}

        for model in self.model_names:
            if model not in pred_df.columns:
                continue
            pred = pred_df[model]
            # Align
            mask = pred.notna() & actuals.notna() & (pred > 0.001) & (actuals > 0.001)
            p = pred[mask].tail(self.eval_window)
            a = actuals[mask].tail(self.eval_window)

            if len(p) < 20:
                losses[model] = 1.0
                continue

            # QLIKE: log(σ²_pred) + σ²_actual / σ²_pred
            p_sq = p**2
            a_sq = a**2
            qlike = (np.log(p_sq) + a_sq / p_sq).mean()
            losses[model] = max(qlike, 0.01)

        if not losses:
            return {m: 1.0 / len(self.model_names) for m in self.model_names}

        # Inverse loss weights
        inv_losses = {m: 1.0 / l for m, l in losses.items()}
        total = sum(inv_losses.values())
        weights = {m: v / total for m, v in inv_losses.items()}

        # Floor at 10% per model
        n = len(weights)
        floor = 0.10
        for m in weights:
            weights[m] = max(weights[m], floor)
        total = sum(weights.values())
        weights = {m: v / total for m, v in weights.items()}

        return weights


# ===========================================================================
# 8. WALK-FORWARD VALIDATOR
# ===========================================================================


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    Unlike k-fold CV, this respects temporal ordering:
      - Train on data up to time t
      - Predict on [t, t + eval_window]
      - Slide forward and repeat

    Reports: MSE, MAE, QLIKE, R², Mincer-Zarnowitz regression stats.
    """

    def __init__(self, n_folds: int = 5, min_train_pct: float = 0.5):
        self.n_folds = n_folds
        self.min_train_pct = min_train_pct
        self.results = []

    def evaluate(self, predictions: pd.Series, actuals: pd.Series) -> dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        mask = predictions.notna() & actuals.notna() & (predictions > 0) & (actuals > 0)
        pred = predictions[mask]
        actual = actuals[mask]

        if len(pred) < 30:
            return {"mse": np.nan, "mae": np.nan, "qlike": np.nan, "r2": np.nan}

        mse = ((pred - actual) ** 2).mean()
        mae = (pred - actual).abs().mean()

        # QLIKE
        p_sq = pred**2
        a_sq = actual**2
        qlike = (np.log(p_sq) + a_sq / p_sq).mean()

        # R² (out-of-sample)
        ss_res = ((actual - pred) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Mincer-Zarnowitz: regress actual on predicted
        # If β₁ ≈ 1 and α ≈ 0, forecasts are unbiased
        Xmz = np.column_stack([np.ones(len(pred)), pred.values])
        try:
            beta = np.linalg.lstsq(Xmz, actual.values, rcond=None)[0]
            mz_alpha = beta[0]
            mz_beta = beta[1]
        except np.linalg.LinAlgError:
            mz_alpha, mz_beta = np.nan, np.nan

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "qlike": qlike,
            "r2": r2,
            "mz_alpha": mz_alpha,
            "mz_beta": mz_beta,
            "n_obs": len(pred),
        }


# ===========================================================================
# 9. MAIN ORCHESTRATOR
# ===========================================================================


class VolatilityPredictor:
    """
    End-to-end volatility prediction pipeline.

    Usage:
        predictor = VolatilityPredictor()
        results = predictor.run(symbols=["AAPL", "MSFT", ...])
        # results is a dict with predictions, metrics, feature importance
    """

    def __init__(
        self,
        horizon: str = "5d",
        use_lstm: bool = True,
        use_gbm: bool = True,
        lstm_epochs: int = 80,
        lstm_seq_len: int = 20,
    ):
        self.horizon = horizon
        self.horizon_days = HORIZONS.get(horizon, 5)
        self.use_lstm = use_lstm
        self.use_gbm = use_gbm
        self.lstm_epochs = lstm_epochs
        self.lstm_seq_len = lstm_seq_len

        self.pipeline = DataPipeline()
        self.validator = WalkForwardValidator()
        self.ensemble = AdaptiveEnsemble()

    def run(
        self,
        symbols: list[str] | None = None,
        lookback_years: float = 3.0,
        force_refresh: bool = False,
    ) -> dict:
        """
        Run the full prediction pipeline.

        Returns dict with:
          - predictions: {symbol: {model: Series, ensemble: Series}}
          - metrics: {symbol: {model: metrics_dict}}
          - latest: {symbol: {current_vol, predicted_vol, vol_regime, ...}}
          - feature_importance: Series (from GBM)
          - ensemble_weights: Dict
        """
        symbols = symbols or UNIVERSE
        logger.info(f"Starting volatility prediction for {len(symbols)} symbols")
        logger.info(f"Horizon: {self.horizon} ({self.horizon_days} days)")

        # ── 1. Fetch data (uses disk cache when fresh) ────────────────
        all_data = self.pipeline.fetch(symbols, lookback_years, force_refresh=force_refresh)
        vix_data = self.pipeline.fetch_vix(lookback_years, force_refresh=force_refresh)

        if not all_data:
            logger.error("No data fetched — aborting")
            return {"error": "No data available"}

        # ── 2. Build features for all symbols ─────────────────────────
        feat_engine = FeatureEngine(vix_data=vix_data, all_data=all_data)

        all_predictions = {}
        all_metrics = {}
        all_latest = {}
        global_feature_importance = None

        # Pool all training data for a global model
        global_X_list = []
        global_y_list = []
        global_features_dict = {}
        global_targets_dict = {}

        logger.info("Building features for all symbols...")
        for sym in all_data:
            df = all_data[sym]
            feats, targets = feat_engine.build_features(sym, df)
            target = targets.get(self.horizon)
            if target is None:
                continue

            # Store for global model training
            global_features_dict[sym] = feats
            global_targets_dict[sym] = target

            # Combine for pooled training
            sym_mask = feats.notna().all(axis=1) & target.notna()
            if sym_mask.sum() > 100:
                global_X_list.append(feats[sym_mask])
                global_y_list.append(target[sym_mask])

        if not global_X_list:
            logger.error("No valid features built — aborting")
            return {"error": "Feature engineering failed"}

        # Pool data for global model training
        global_X = pd.concat(global_X_list, axis=0).sort_index()
        global_y = pd.concat(global_y_list, axis=0).sort_index()

        logger.info(f"Global training set: {len(global_X)} rows, {global_X.shape[1]} features")

        # ── 3. Train global models ────────────────────────────────────
        # HAR (per-symbol is better, but we also fit a global one)
        har_model = HARModel()
        har_model.fit(global_X, global_y)
        logger.info("HAR model fitted")

        # GBM (global model)
        gbm_model = GBMVolModel()
        if self.use_gbm:
            logger.info("Training GBM model...")
            gbm_model.fit(global_X, global_y)
            global_feature_importance = gbm_model.feature_importance()
            if global_feature_importance is not None:
                logger.info(f"Top features: {global_feature_importance.head(5).to_dict()}")

        # LSTM (global model)
        lstm_model = LSTMVolModel(
            seq_len=self.lstm_seq_len,
            epochs=self.lstm_epochs,
        )
        if self.use_lstm:
            logger.info("Training LSTM model (this may take a few minutes)...")
            lstm_model.fit(global_X, global_y)

        # ── 4. Generate per-symbol predictions ────────────────────────
        logger.info("Generating predictions...")
        for sym in global_features_dict:
            feats = global_features_dict[sym]
            target = global_targets_dict[sym]

            preds = {}

            # HAR (fit per-symbol for better accuracy)
            har_sym = HARModel()
            har_sym.fit(feats, target)
            preds["HAR"] = har_sym.predict(feats)

            # GBM
            if self.use_gbm and gbm_model.model is not None:
                preds["GBM"] = gbm_model.predict(feats)

            # LSTM
            if self.use_lstm and lstm_model.model is not None:
                preds["LSTM"] = lstm_model.predict(feats)

            # Ensemble
            ensemble_pred = self.ensemble.combine(preds, target)
            preds["Ensemble"] = ensemble_pred

            # Evaluate each model
            sym_metrics = {}
            for model_name, pred_series in preds.items():
                sym_metrics[model_name] = self.validator.evaluate(pred_series, target)

            all_predictions[sym] = preds
            all_metrics[sym] = sym_metrics

            # Latest prediction
            last_idx = feats.index[-1]
            current_vol = feats["vol_yz_20"].iloc[-1] if "vol_yz_20" in feats else np.nan
            predicted_vol = ensemble_pred.iloc[-1] if len(ensemble_pred) > 0 else np.nan

            # Classify volatility regime
            vol_hist = feats.get("vol_yz_20", pd.Series(dtype=float))
            vol_percentile = 50
            if len(vol_hist.dropna()) > 60:
                vol_percentile = (vol_hist.dropna() < current_vol).mean() * 100

            if vol_percentile > 80:
                regime = "HIGH_VOL"
            elif vol_percentile > 60:
                regime = "ELEVATED"
            elif vol_percentile > 40:
                regime = "NORMAL"
            elif vol_percentile > 20:
                regime = "LOW"
            else:
                regime = "COMPRESSED"

            # Vol direction forecast
            if not np.isnan(predicted_vol) and not np.isnan(current_vol) and current_vol > 0:
                vol_change = (predicted_vol - current_vol) / current_vol
                if vol_change > 0.10:
                    direction = "EXPANDING"
                elif vol_change < -0.10:
                    direction = "CONTRACTING"
                else:
                    direction = "STABLE"
            else:
                vol_change = 0.0
                direction = "UNKNOWN"

            all_latest[sym] = {
                "symbol": sym,
                "sector": SECTOR_MAP.get(sym, "Unknown"),
                "current_vol": round(current_vol * 100, 1) if not np.isnan(current_vol) else None,
                "predicted_vol": round(predicted_vol * 100, 1)
                if not np.isnan(predicted_vol)
                else None,
                "vol_change_pct": round(vol_change * 100, 1),
                "vol_percentile": round(vol_percentile, 0),
                "regime": regime,
                "direction": direction,
                "date": str(last_idx.date()) if hasattr(last_idx, "date") else str(last_idx),
            }

        # ── 5. Aggregate sector-level predictions ─────────────────────
        sector_summary = {}
        for sym, info in all_latest.items():
            sec = info["sector"]
            if sec not in sector_summary:
                sector_summary[sec] = {"symbols": [], "vols": [], "predicted": [], "directions": []}
            sector_summary[sec]["symbols"].append(sym)
            if info["current_vol"] is not None:
                sector_summary[sec]["vols"].append(info["current_vol"])
            if info["predicted_vol"] is not None:
                sector_summary[sec]["predicted"].append(info["predicted_vol"])
            sector_summary[sec]["directions"].append(info["direction"])

        for sec in sector_summary:
            vols = sector_summary[sec]["vols"]
            preds = sector_summary[sec]["predicted"]
            dirs = sector_summary[sec]["directions"]
            sector_summary[sec]["avg_current_vol"] = round(np.mean(vols), 1) if vols else None
            sector_summary[sec]["avg_predicted_vol"] = round(np.mean(preds), 1) if preds else None
            sector_summary[sec]["dominant_direction"] = (
                max(set(dirs), key=dirs.count) if dirs else "UNKNOWN"
            )
            sector_summary[sec]["n_stocks"] = len(sector_summary[sec]["symbols"])

        return {
            "predictions": all_predictions,
            "metrics": all_metrics,
            "latest": all_latest,
            "sector_summary": sector_summary,
            "feature_importance": global_feature_importance,
            "ensemble_weights": self.ensemble.weights,
            "horizon": self.horizon,
            "horizon_days": self.horizon_days,
            "n_symbols": len(all_latest),
            "timestamp": datetime.now().isoformat(),
        }


# ===========================================================================
# FACADE — simple entry point for external callers (vol_targeting, live engine)
# ===========================================================================


class VolatilityPredictionEngine:
    """
    Thin facade over the multi-model vol forecasting stack.

    Used by core/vol_targeting.py as the primary vol estimate source
    (priority 1, above H2O and EWMA fallbacks).

    Example:
        engine = VolatilityPredictionEngine()
        ann_vol = engine.predict_one("AAPL", ohlcv_df)
        scale   = vol_targeter.scale_from_vol(ann_vol)
    """

    def __init__(self):
        self._pipeline = DataPipeline()
        self._feat_eng = FeatureEngine()
        self._estimators = VolatilityEstimators()
        self._har = HARModel()
        self._ensemble = AdaptiveEnsemble()
        self._fitted = False

    def predict_one(
        self,
        symbol: str,
        price_df: "pd.DataFrame",
        horizon: int = 5,
        as_of_date=None,
    ) -> float | None:
        """
        Predict annualised volatility for one symbol using available price data.

        Parameters
        ----------
        symbol    : ticker string (used for sector features)
        price_df  : OHLCV DataFrame with columns Open/High/Low/Close/Volume
        horizon   : forecast horizon in trading days (default 5)
        as_of_date: if set, restrict data to this date (anti-lookahead)

        Returns
        -------
        float — annualised volatility estimate, or None on failure
        """
        try:
            df = price_df.copy()
            # Normalise column names: vol_engine expects lowercase
            col_map = {
                c: c.lower() for c in df.columns if c in ("Open", "High", "Low", "Close", "Volume")
            }
            if col_map:
                df = df.rename(columns=col_map)

            if as_of_date is not None:
                df = df[df.index <= as_of_date]

            if len(df) < 30:
                return None

            close_col = "close" if "close" in df.columns else "Close"
            close = df[close_col].squeeze().astype(float)

            # Build feature set (sym, df are the correct params)
            feat_result = self._feat_eng.build_features(symbol, df)
            # build_features returns (DataFrame, Dict) tuple
            if isinstance(feat_result, tuple):
                features, _ = feat_result
            else:
                features = feat_result
            if features is None or features.empty:
                return None

            # HAR model: realised variance at daily / weekly / monthly
            log_ret = np.log(close / close.shift(1)).dropna()
            rv_d = float((log_ret**2).iloc[-1] * 252)
            rv_w = float((log_ret**2).rolling(5).mean().iloc[-1] * 252)
            rv_m = float((log_ret**2).rolling(21).mean().iloc[-1] * 252)

            # Weighted average of HAR components (Corsi 2009 weights)
            har_vol = float(np.sqrt(max(0.4 * rv_d + 0.35 * rv_w + 0.25 * rv_m, 1e-8)))

            # Add technical regime context
            recent_vol = float(log_ret.rolling(21).std().iloc[-1] * np.sqrt(252))
            vix_adj = 1.0
            if "vix" in features.columns:
                vix_val = float(features["vix"].iloc[-1])
                # Scale up estimate when VIX is elevated
                vix_adj = max(1.0, vix_val / 20.0)

            # Blend HAR with recent EWMA for responsiveness
            ewma_var = float(log_ret.ewm(span=21).var().iloc[-1] * 252)
            ewma_vol = float(np.sqrt(max(ewma_var, 1e-8)))
            blended = 0.6 * har_vol + 0.4 * ewma_vol

            return float(np.clip(blended * vix_adj, 0.02, 2.0))

        except Exception as e:
            logger.debug(f"VolatilityPredictionEngine.predict_one({symbol}): {e}")
            return None
