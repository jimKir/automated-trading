"""
Position-Level Anomaly Scorer
==============================
Produces per-symbol scale factors (0..1) that cut exposure on individual
instruments *asymmetrically* depending on their asset class and current
anomaly signature — rather than the portfolio-wide uniform scalar that
ChoppyRegimeDetector produces.

Design goals
------------
1. Crypto leg gets cut aggressively when vol spikes or macro stress rises,
   because crypto drawdowns (ETH -15%/day in 2025) dwarf equity drawdowns.
2. Equity positions get a lighter, signal-driven trim.
3. ETFs and macro hedges (TLT, GLD) are either left alone or scaled UP
   during stress (they are the hedge).
4. No look-ahead: all features computed from rolling history.
5. Composable: output is a Dict[symbol, float] that the backtest engine
   and live engine multiply into existing signal weights before sizing.

Architecture
------------
Each symbol is classified into an AssetClass, which has its own:
  - sensitivity multiplier  (how hard to cut per unit of anomaly score)
  - baseline vol            (for z-scoring realised vol spikes)
  - feature blend           (which anomaly signals matter most)

The final per-symbol scale is:

  instrument_score = blend(
      G1. Realised vol spike     — 20d vol / 60d baseline vol (normalised)
      G2. Momentum deterioration — |20d return| / vol (low = churning)
      G3. Drawdown from 20d peak — current price / 20d high
      G4. Cross-asset stress     — ChoppyRegimeDetector portfolio score
  )

  scale(sym) = 1 - sensitivity(asset_class) × instrument_score(sym)
             = clipped to [floor(asset_class), 1.0]

Asset class floors
------------------
  crypto   : floor 0.10  (max 90% cut — allowed to go very defensive)
  equity   : floor 0.40  (max 60% cut — never go fully flat on equities)
  etf_hedge: floor 1.00  (never cut — hedges are counter-cyclical)
  default  : floor 0.50

Integration
-----------
  backtest/engine.py  — calls scorer.score_day(date, all_data) at rebalance
  execution/live_engine.py — calls scorer.score_today(prices) each cycle

  Both multiply per-symbol scales into signal weights BEFORE compute_target_weights:
    scaled_signals = {sym: sig * pos_scores.get(sym, 1.0) for sym, sig in signals.items()}

  The portfolio-level ChoppyRegimeDetector scale is applied SEPARATELY (as now).
  These two layers are independent — no double-counting from shared EWS features.
"""
from __future__ import annotations

import warnings
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("PositionAnomalyScorer")
warnings.filterwarnings("ignore")


# ── Asset class taxonomy ──────────────────────────────────────────────────────

class AssetClass(Enum):
    CRYPTO     = "crypto"       # BTC-USD, ETH-USD — high vol, high sensitivity
    EQUITY     = "equity"       # individual stocks — moderate sensitivity
    ETF_EQUITY = "etf_equity"   # SPY, QQQ, IWM — index ETFs, light sensitivity
    ETF_HEDGE  = "etf_hedge"    # TLT, GLD, SHY — counter-cyclical, never cut
    COMMODITY  = "commodity"    # OIL, COPPER futures — moderate-high sensitivity
    FX         = "fx"           # DXY, JPY, EUR — low sensitivity
    UNKNOWN    = "unknown"      # default


# ── Classification rules ──────────────────────────────────────────────────────

_CRYPTO_SYMBOLS    = {"BTC-USD", "ETH-USD", "BTC", "ETH", "BTC/USD", "ETH/USD",
                      "BTCUSD", "ETHUSD"}
_ETF_HEDGE_SYMBOLS = {"TLT", "SHY", "IEF", "GLD", "SGOL", "IAU", "SHV",
                      "USFR", "BKLN", "AGG", "BND"}
_ETF_EQUITY_SYMS   = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "MDY",
                      "SPDW", "EEM", "EFA"}
_COMMODITY_SYMS    = {"OIL", "CL=F", "GC=F", "COPPER", "HG=F", "NG=F", "GOLD"}
_FX_SYMS           = {"DXY", "DX-Y.NYB", "JPY", "JPY=X", "EURUSD", "EURUSD=X",
                      "UUP"}


def classify(symbol: str) -> AssetClass:
    """Classify a symbol into an asset class."""
    s = symbol.upper().replace("^", "")
    if s in _CRYPTO_SYMBOLS or "BTC" in s or "ETH" in s or "SOL" in s:
        return AssetClass.CRYPTO
    if s in _ETF_HEDGE_SYMBOLS:
        return AssetClass.ETF_HEDGE
    if s in _ETF_EQUITY_SYMS:
        return AssetClass.ETF_EQUITY
    if s in _COMMODITY_SYMS:
        return AssetClass.COMMODITY
    if s in _FX_SYMS:
        return AssetClass.FX
    return AssetClass.EQUITY


# ── Per-class configuration ───────────────────────────────────────────────────
#
# sensitivity    : how aggressively to scale down per unit of anomaly score
# floor          : minimum scale factor (never cut below this)
# vol_baseline   : expected daily vol (for z-scoring vol spikes)
# feature_weights: blend of G1/G2/G3/G4 signals for this class

_CLASS_CONFIG: Dict[AssetClass, dict] = {
    AssetClass.CRYPTO: {
        "sensitivity":     1.40,          # aggressive — crypto drawdowns are severe
        "floor":           0.10,          # allow up to 90% cut
        "vol_baseline":    0.035,         # ~35-40% ann vol / √252 ≈ 0.022-0.025 daily
        "feature_weights": {
            "vol_spike":       0.35,      # primary: crypto vol spikes are predictive
            "momentum_churn":  0.25,      # churning without direction = distribution
            "dd_from_peak":    0.25,      # already in drawdown from recent peak
            "portfolio_stress":0.15,      # lighter weight — crypto independent
        },
    },
    AssetClass.EQUITY: {
        "sensitivity":     0.65,          # moderate
        "floor":           0.40,          # never below 40%
        "vol_baseline":    0.015,         # ~15-20% ann vol / √252 ≈ 0.010-0.013
        "feature_weights": {
            "vol_spike":       0.25,
            "momentum_churn":  0.30,      # choppy price action most important
            "dd_from_peak":    0.20,
            "portfolio_stress":0.25,      # portfolio macro context matters more for equities
        },
    },
    AssetClass.ETF_EQUITY: {
        "sensitivity":     0.40,          # light — broad market ETFs are diversified
        "floor":           0.55,
        "vol_baseline":    0.011,
        "feature_weights": {
            "vol_spike":       0.20,
            "momentum_churn":  0.35,
            "dd_from_peak":    0.15,
            "portfolio_stress":0.30,
        },
    },
    AssetClass.ETF_HEDGE: {
        "sensitivity":     0.0,           # never scale down hedges
        "floor":           1.00,          # always full exposure
        "vol_baseline":    0.008,
        "feature_weights": {
            "vol_spike": 1.0, "momentum_churn": 0.0,
            "dd_from_peak": 0.0, "portfolio_stress": 0.0,
        },
    },
    AssetClass.COMMODITY: {
        "sensitivity":     0.75,
        "floor":           0.35,
        "vol_baseline":    0.020,
        "feature_weights": {
            "vol_spike":       0.40,      # commodity vol spikes primary
            "momentum_churn":  0.20,
            "dd_from_peak":    0.25,
            "portfolio_stress":0.15,
        },
    },
    AssetClass.FX: {
        "sensitivity":     0.30,
        "floor":           0.60,
        "vol_baseline":    0.006,
        "feature_weights": {
            "vol_spike":       0.30,
            "momentum_churn":  0.25,
            "dd_from_peak":    0.15,
            "portfolio_stress":0.30,
        },
    },
    AssetClass.UNKNOWN: {
        "sensitivity":     0.50,
        "floor":           0.50,
        "vol_baseline":    0.015,
        "feature_weights": {
            "vol_spike": 0.25, "momentum_churn": 0.25,
            "dd_from_peak": 0.25, "portfolio_stress": 0.25,
        },
    },
}

# Calibration: thresholds at which each feature reaches score=1.0
#   vol_spike:      ratio of 20d realised vol / 60d baseline
#                   1.0 = no spike, 3.0 = 3× usual vol (full score)
#   momentum_churn: |20d net return| / 20d realised vol  (low = choppy)
#                   1.2 = normal, 0.2 = churning (full score at low value)
#   dd_from_peak:   drawdown from 20d high
#                   0% = at peak, 15% = drawdown (full score)
_VOL_SPIKE_CEILING       = 3.0    # ratio: 20d vol / 60d baseline
_VOL_SPIKE_BASELINE      = 1.0    # ratio at which score = 0
_CHURN_BASELINE          = 1.2    # TNR: above this = trending (score = 0)
_CHURN_CEILING           = 0.2    # TNR: below this = maximum churn (score = 1)
_DD_BASELINE             = 0.00   # 0% drawdown from 20d peak = score 0
_DD_CEILING_CRYPTO       = 0.15   # 15% drop from 20d peak = score 1 (crypto)
_DD_CEILING_EQUITY       = 0.10   # 10% drop from 20d peak = score 1 (equity)

# Smoothing
_EMA_SPAN = 3    # 3-day EMA — faster than portfolio-level (5d) for individual positions


class PositionAnomalyScorer:
    """
    Produces per-symbol anomaly scale factors daily.

    The scorer is stateless between calls — all features are computed from
    rolling windows on the provided price history. No model training required.

    Usage (backtest — call at each rebalance date):
        scorer = PositionAnomalyScorer(portfolio_choppy_score=choppy_score_series)
        per_sym_scales = scorer.score_day(date, price_history)
        # price_history: {symbol: pd.DataFrame with Close column}

    Usage (live — call each cycle):
        scorer = PositionAnomalyScorer()
        per_sym_scales = scorer.score_today(price_df, portfolio_score=ews_score)
    """

    def __init__(
        self,
        portfolio_choppy_score: Optional[pd.Series] = None,
        vol_window: int = 20,
        baseline_window: int = 60,
    ):
        """
        Parameters
        ----------
        portfolio_choppy_score : Pre-computed ChoppyRegimeDetector score series (for backtest).
                                 If None, portfolio stress defaults to 0 (no portfolio context).
        vol_window             : Rolling window for realised vol (days).
        baseline_window        : Rolling window for vol baseline (days).
        """
        self._port_score = portfolio_choppy_score
        self._vw  = vol_window
        self._bw  = baseline_window
        # Per-symbol score cache (smoothed)
        self._score_cache: Dict[str, pd.Series] = {}

    # ── Feature computation (per-symbol) ─────────────────────────────────────

    def _compute_sym_features(
        self,
        close: pd.Series,
        asset_class: AssetClass,
    ) -> pd.DataFrame:
        """
        Compute the four raw feature series for one symbol.
        Returns a DataFrame with columns: vol_spike, momentum_churn,
                                          dd_from_peak, raw_score
        """
        ret = close.pct_change()
        cfg = _CLASS_CONFIG[asset_class]
        vol_baseline_daily = cfg["vol_baseline"]

        feat = pd.DataFrame(index=close.index)

        # G1: Vol spike — 20d realised vol / 60d realised vol (z-score ratio)
        rv20 = ret.rolling(self._vw).std()
        rv60 = ret.rolling(self._bw).std().replace(0, np.nan)
        vol_ratio = rv20 / rv60
        # Score: (ratio - baseline) / (ceiling - baseline), clipped [0,1]
        feat["vol_spike"] = ((vol_ratio - _VOL_SPIKE_BASELINE) /
                             (_VOL_SPIKE_CEILING - _VOL_SPIKE_BASELINE)).clip(0, 1)

        # G2: Momentum churn — |20d net return| / 20d realised vol
        # Low value = price is volatile but going nowhere = distribution/chop
        net_ret   = close.pct_change(self._vw).abs()
        path_vol  = rv20 * np.sqrt(self._vw)
        tnr       = (net_ret / path_vol.replace(0, np.nan)).fillna(1.0).clip(0, 3)
        # Invert: low TNR = high score
        dd_ceil = _DD_CEILING_CRYPTO if asset_class == AssetClass.CRYPTO else _DD_CEILING_EQUITY
        feat["momentum_churn"] = ((_CHURN_BASELINE - tnr) /
                                  (_CHURN_BASELINE - _CHURN_CEILING)).clip(0, 1)

        # G3: Drawdown from 20d rolling high
        high_20d  = close.rolling(self._vw).max()
        dd_20d    = (close - high_20d) / high_20d.replace(0, np.nan)  # negative = below peak
        dd_score  = ((-dd_20d) / dd_ceil).clip(0, 1)   # invert: deep DD = high score
        feat["dd_from_peak"] = dd_score

        return feat.fillna(0).clip(0, 1)

    def _compute_sym_score(
        self,
        symbol: str,
        close: pd.Series,
        asset_class: AssetClass,
        portfolio_score: Optional[float] = None,
    ) -> pd.Series:
        """
        Blend all four features → raw score → EMA-smoothed → scale factor.
        Returns a pd.Series of score values ∈ [0,1], aligned to close.index.
        """
        cfg    = _CLASS_CONFIG[asset_class]
        fw     = cfg["feature_weights"]
        feat   = self._compute_sym_features(close, asset_class)

        port_sc = float(portfolio_score) if portfolio_score is not None else 0.0

        # Weighted blend
        score = (
            fw.get("vol_spike",        0) * feat["vol_spike"] +
            fw.get("momentum_churn",   0) * feat["momentum_churn"] +
            fw.get("dd_from_peak",     0) * feat["dd_from_peak"] +
            fw.get("portfolio_stress", 0) * port_sc
        )
        score = score.clip(0, 1)

        # Smooth with short EMA to avoid single-day whipsawing
        score = score.ewm(span=_EMA_SPAN, adjust=False).mean()

        return score.clip(0, 1)

    def _score_to_scale(self, score: float, asset_class: AssetClass) -> float:
        """
        Convert instrument anomaly score → position scale factor.

        scale = 1 - sensitivity × score
        clipped to [floor, 1.0]
        """
        cfg    = _CLASS_CONFIG[asset_class]
        raw    = 1.0 - cfg["sensitivity"] * score
        return float(np.clip(raw, cfg["floor"], 1.0))

    # ── Public API ────────────────────────────────────────────────────────────

    def score_day(
        self,
        date: pd.Timestamp,
        price_history: Dict[str, pd.DataFrame],
        portfolio_score: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute per-symbol scale factors for a single rebalance date.
        Uses all available history up to and including `date`.

        Parameters
        ----------
        date            : Rebalance date (no future data used).
        price_history   : {symbol: DataFrame with at least a Close column}
        portfolio_score : ChoppyRegimeDetector score for this date (0..1).
                          If None, looks up self._port_score series.

        Returns
        -------
        Dict[symbol, scale_factor ∈ [floor, 1.0]]
        """
        if portfolio_score is None and self._port_score is not None:
            try:
                portfolio_score = float(self._port_score.asof(date))
            except Exception:
                portfolio_score = 0.0

        result: Dict[str, float] = {}

        for sym, df in price_history.items():
            try:
                # Ensure tz-naive index and close column
                df_c = df.copy()
                if isinstance(df_c.columns, pd.MultiIndex):
                    df_c.columns = [c[0] for c in df_c.columns]
                df_c.columns = [c.capitalize() for c in df_c.columns]
                if df_c.index.tz is not None:
                    df_c.index = df_c.index.tz_localize(None)

                if "Close" not in df_c.columns:
                    result[sym] = 1.0
                    continue

                close = df_c["Close"].loc[:date]
                if len(close) < max(self._bw, self._vw) + 5:
                    result[sym] = 1.0   # insufficient history — no opinion
                    continue

                ac = classify(sym)

                # Cache the full score series per symbol (avoid recomputing each day)
                if sym not in self._score_cache:
                    self._score_cache[sym] = self._compute_sym_score(
                        sym, df_c["Close"], ac, portfolio_score
                    )

                score_series = self._score_cache[sym]
                try:
                    score = float(score_series.asof(date))
                except Exception:
                    score = float(score_series.iloc[-1]) if len(score_series) > 0 else 0.0

                result[sym] = self._score_to_scale(score, ac)

            except Exception as e:
                log.debug(f"PositionAnomalyScorer: {sym} failed ({e}) — defaulting to 1.0")
                result[sym] = 1.0

        if result:
            crypto = {s: v for s, v in result.items() if classify(s) == AssetClass.CRYPTO}
            equity = {s: v for s, v in result.items() if classify(s) == AssetClass.EQUITY}
            if crypto or equity:
                log.debug(
                    f"PositionAnomalyScorer [{date.date()}]: "
                    + (f"crypto_scales={crypto} " if crypto else "")
                    + (f"equity_avg={np.mean(list(equity.values())):.2f}" if equity else "")
                )

        return result

    def score_today(
        self,
        price_df: pd.DataFrame,
        portfolio_score: float = 0.0,
    ) -> Dict[str, float]:
        """
        Live mode: compute per-symbol scales using the most recent data.

        Parameters
        ----------
        price_df        : DataFrame of Close prices, columns = symbols
        portfolio_score : ChoppyRegimeDetector.score_today() output

        Returns
        -------
        Dict[symbol, scale_factor]
        """
        result: Dict[str, float] = {}
        for sym in price_df.columns:
            try:
                close = price_df[sym].dropna()
                if len(close) < self._bw + 5:
                    result[sym] = 1.0
                    continue
                ac    = classify(sym)
                score = self._compute_sym_score(sym, close, ac, portfolio_score).iloc[-1]
                result[sym] = self._score_to_scale(float(score), ac)
            except Exception as e:
                log.debug(f"PositionAnomalyScorer live: {sym} failed ({e})")
                result[sym] = 1.0
        return result

    def score_series_for(
        self,
        symbol: str,
        close: pd.Series,
        portfolio_score_series: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute full score and scale-factor series for one symbol.
        Useful for backtesting diagnostics and charting.

        Returns (score_series, scale_series)
        """
        ac = classify(symbol)
        # Use mean portfolio score across history if series provided
        port_sc = float(portfolio_score_series.mean()) if portfolio_score_series is not None else 0.0
        score_s = self._compute_sym_score(symbol, close, ac, port_sc)
        scale_s = score_s.apply(lambda s: self._score_to_scale(s, ac))
        return score_s, scale_s

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @staticmethod
    def describe_config() -> pd.DataFrame:
        """Return a DataFrame showing asset class configuration."""
        rows = []
        for ac, cfg in _CLASS_CONFIG.items():
            rows.append({
                "asset_class": ac.value,
                "sensitivity": cfg["sensitivity"],
                "floor":       cfg["floor"],
                "max_cut_pct": f"{(1 - cfg['floor'])*100:.0f}%",
                "vol_baseline_daily": f"{cfg['vol_baseline']*100:.1f}%",
            })
        return pd.DataFrame(rows)


# ── Module-level helpers ──────────────────────────────────────────────────────

def apply_position_scales(
    signals: Dict[str, float],
    scales:  Dict[str, float],
) -> Dict[str, float]:
    """
    Multiply signal weights by per-symbol anomaly scales.
    Symbols with no scale default to 1.0 (no change).

    Parameters
    ----------
    signals : {symbol: signal_weight}  — output of SignalGenerator
    scales  : {symbol: scale_factor}   — output of PositionAnomalyScorer

    Returns
    -------
    Dict[symbol, adjusted_signal_weight]
    """
    return {sym: sig * scales.get(sym, 1.0) for sym, sig in signals.items()}
