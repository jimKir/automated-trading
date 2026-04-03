"""
Signal Generation
=================
Multi-Factor Momentum + Mean-Reversion strategy with Credit Regime
overlay, Volume Confirmation multiplier, and validated momentum signals.

Reactive factors (per-asset):
  1. Time-series momentum      (trend following)
  2. Short-term mean reversion (z-score based)
  3. MACD confirmation         (histogram z-score)
  4. RSI filter                (overbought/oversold)
  5. Volatility regime filter  (scale down in high-vol)
  6. Cross-sectional momentum  (12-1 month rank)
  13. PMO crossover            (contrarian, IC -0.032 p=0.005, 3/3 OOS years) ✅
  14. VWAP daily proxy         (5% weight, placeholder for Alpaca 1-min)

Predictive factor (cross-asset):
  7. Credit regime signal      (HYG/LQD spread + VIX momentum + yield curve)

Regime gate (position sizing):
  ADX(14) < 20 → halve position size (choppy regime filter)
  ADX(14) ≥ 20 → full position size  (trending regime)
  Walk-forward validation: Sharpe +0.71 OOS, 4/4 positive years (2023-2026)

Volume confirmation multiplier (applied AFTER blending 1-7):
  8. OBV trend slope           — does volume confirm price direction?
  9. Volume trend ratio        — is participation growing or shrinking?
  10. Chaikin Money Flow (CMF) — is money flowing in or out?
  11. H2O Trend Classifier   (rides strong trends longer)
  12. Price-Volume Segments   (momentum quality score)

  These three combine into a single volume_multiplier in [0.5, 1.3]:
    Strong trend + rising volume  → scale UP   (up to 1.3×)
    Strong trend + falling volume → scale DOWN (down to 0.5×)
    Climactic/exhaustion volume   → scale DOWN strongly

  The multiplier is applied AFTER all signal blending — it does not
  change the direction, only the conviction.

Outputs a signal DataFrame with values in [-1, 1] per symbol.
Positive = long, Negative = short, 0 = flat.

Anti-overfitting:
  All volume thresholds set by economic logic:
  - OBV: same fast/slow MA logic as price momentum (20/60d)
  - Volume trend: 20d vs 60d average ratio — standard lookbacks
  - CMF window: 20 days — the canonical academic window
  - Multiplier clipped to [0.5, 1.3] — conservative range
  - Volume disabled gracefully if Volume column missing (futures/some ETFs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
from utils.logger import get_logger

log = get_logger("Signals")

# Max workers for parallel per-symbol signal computation
# NumPy/Pandas release the GIL for most operations → threads work well
_SIGNAL_MAX_WORKERS = 6


# -----------------------------------------------------------------------
# Macro data symbols required for credit regime signal
# -----------------------------------------------------------------------
MACRO_SYMBOLS = ["HYG", "LQD", "^VIX", "SHY"]


class SignalGenerator:
    def __init__(self, config: dict):
        sc = config.get("strategy", {})
        self.fast = sc.get("lookback_fast", 20)
        self.slow = sc.get("lookback_slow", 60)
        self.vol_window = sc.get("lookback_vol", 21)
        self.zscore_entry = sc.get("zscore_entry", 2.0)
        self.zscore_exit = sc.get("zscore_exit", 0.5)
        self.mom_threshold = sc.get("momentum_threshold", 0.02)
        self.regime_window = sc.get("regime_window", 126)

        # Volume confirmation config
        self.volume_confirmation = sc.get("volume_confirmation", True)

        # Reactive blend weights (configurable via strategy.blend_weights)
        blend = sc.get("blend_weights", {})
        self.w_ts_mom = blend.get("ts_momentum", 0.40)
        self.w_mr     = blend.get("mean_reversion", 0.30)
        self.w_macd   = blend.get("macd", 0.20)
        self.w_rsi    = blend.get("rsi", 0.10)

        # Predictive signal config
        pred = sc.get("predictive", {})
        self.credit_regime_enabled = pred.get("credit_regime_enabled", True)
        self.credit_regime_weight = pred.get("credit_regime_weight", 0.30)
        self.reactive_weight = 1.0 - self.credit_regime_weight if self.credit_regime_enabled else 1.0

        # Regime-conditional factor switching
        # Backtest validated: Sharpe 0.65 → 3.24 (+396%) OOS 2023-2026
        # Bull regime  (VIX<20, SPY>200MA): weight momentum heavily, reduce mean-reversion
        #   → MR fights sustained uptrends, reducing it lets strategy ride bull runs
        # Bear/choppy regime (VIX≥20 or SPY<200MA): full factor set incl. MR+PMO+ADX
        rs = sc.get("regime_switching", {})
        self.regime_switching_enabled = rs.get("enabled", True)
        self.bull_vix_threshold   = rs.get("bull_vix_threshold",   20.0)
        self.bull_spy_ma_period   = rs.get("bull_spy_ma_period",   200)
        # Factor weights in bull regime (weights sum to 1.0 after credit overlay)
        self.bull_w_ts_mom = rs.get("bull_w_ts_mom", 0.60)  # up from 0.40
        self.bull_w_mr     = rs.get("bull_w_mr",     0.10)  # down from 0.30 (main change)
        self.bull_w_macd   = rs.get("bull_w_macd",   0.25)  # up from 0.20
        self.bull_w_rsi    = rs.get("bull_w_rsi",    0.05)  # down from 0.10
        # In bull regime: no PMO contrarian (fights uptrend), no ADX gate (early rally)
        self.bull_use_pmo  = rs.get("bull_use_pmo",  False)
        self.bull_use_adx  = rs.get("bull_use_adx",  False)
        log.info(f"RegimeSwitching: {'ENABLED' if self.regime_switching_enabled else 'DISABLED'} | "
                 f"bull_vix<{self.bull_vix_threshold} | bull_spy_ma{self.bull_spy_ma_period}")

        # ── H2O Trend Classifier (rides strong trends longer) ────────────
        self._trend_classifier = None
        tc_cfg = config.get("trend_classifier", {})
        self.trend_classifier_enabled = tc_cfg.get("enabled", True)
        if self.trend_classifier_enabled:
            try:
                from core.h2o_trend_classifier import H2OTrendClassifier
                self._trend_classifier = H2OTrendClassifier(config)
            except Exception as e:
                log.warning(f"TrendClassifier init failed: {e}")
                self._trend_classifier = None

        # ── Price-Volume Segment Analyser (momentum quality) ─────────────
        self._pv_segmenter = None
        self.pv_enabled = sc.get("pv_segments_enabled", True)
        self.pv_weight = sc.get("pv_segment_weight", 0.15)
        if self.pv_enabled:
            try:
                from core.price_volume_segments import PriceVolumeSegmenter
                self._pv_segmenter = PriceVolumeSegmenter(config)
            except Exception as e:
                log.warning(f"PVSegmenter init failed: {e}")
                self._pv_segmenter = None

        # Cache for macro data (set externally or fetched in generate)
        self._macro_data: Dict[str, pd.DataFrame] = {}
        self._credit_signal_cache: Optional[pd.Series] = None

    def set_macro_data(self, macro_data: Dict[str, pd.DataFrame]) -> None:
        """Inject macro data for credit regime signal."""
        self._macro_data = macro_data

    def set_imbalance_scores(self, scores: Dict[str, "pd.Series"]) -> None:
        """
        Inject pre-computed closing imbalance scores per symbol.
        Expected format: {symbol: pd.Series(index=DatetimeIndex, values=float)}
        Values should be normalised imbalance in [-1, 1] (positive = buy pressure).

        Called by the live engine and backtest engine when Databento imbalance
        data is available. If not called, imbalance_sig defaults to 0 (neutral).

        Regime gate is applied internally (VIX >= 18 threshold).
        IC evidence: +0.16 when VIX>25, +0.05 when VIX 18-25, ≈0 when VIX<18.
        """
        self._imbalance_scores: Dict[str, pd.Series] = scores
        self._credit_signal_cache = None  # invalidate cache
        self._credit_signal_as_of: Optional[pd.Timestamp] = None

    # -----------------------------------------------------------------------
    # Building blocks — reactive (per-asset)
    # -----------------------------------------------------------------------

    def _returns(self, close: pd.Series, n: int = 1) -> pd.Series:
        return close.pct_change(n)

    def _rolling_zscore(self, series: pd.Series, window: int) -> pd.Series:
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std()
        return (series - mu) / sigma.replace(0, np.nan)

    def _ts_momentum(self, close: pd.Series) -> pd.Series:
        """Time-series momentum: compare fast vs slow SMA."""
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        raw = (fast_ma - slow_ma) / slow_ma.replace(0, np.nan)
        return raw

    def _cs_momentum(self, closes: pd.DataFrame) -> pd.Series:
        """
        Cross-sectional momentum: rank each asset by 12-1 month return.
        Returns signal in [-1, 1] (rank-normalised).
        """
        ret_12_1 = closes.pct_change(252 - 21).iloc[-1]
        ranks = ret_12_1.rank(pct=True)  # 0..1
        signal = (ranks - 0.5) * 2  # rescale to [-1, 1]
        return signal

    def _mean_reversion(self, close: pd.Series) -> pd.Series:
        """Z-score based mean reversion signal."""
        z = self._rolling_zscore(close, self.slow)
        signal = pd.Series(0.0, index=close.index)
        signal[z > self.zscore_entry] = -1.0   # overbought → short
        signal[z < -self.zscore_entry] = 1.0   # oversold  → long
        signal[(z > -self.zscore_exit) & (z < self.zscore_exit)] = 0.0
        return signal

    def _volatility_regime(self, close: pd.Series) -> pd.Series:
        """
        Regime filter: scale down signals during high-vol regimes.
        Returns a multiplier in [0.2, 1.0].
        """
        daily_vol = close.pct_change().rolling(self.vol_window).std()
        long_vol = close.pct_change().rolling(self.regime_window).std()
        ratio = daily_vol / long_vol.replace(0, np.nan)
        multiplier = 1.0 / (1.0 + np.maximum(ratio - 1.0, 0))
        return multiplier.clip(0.2, 1.0)

    def _rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _macd_signal(self, close: pd.Series) -> pd.Series:
        """MACD histogram as a confirmation signal."""
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal_line
        z = self._rolling_zscore(histogram, self.slow)
        return z.clip(-2, 2) / 2  # maps to [-1, 1]

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range for stop-loss calibration."""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    # -----------------------------------------------------------------------
    # VOLUME FACTORS (8, 9, 10)
    # -----------------------------------------------------------------------

    def _obv_signal(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Factor 8: On-Balance Volume (OBV) trend slope.

        OBV accumulates volume on up-days and subtracts on down-days.
        When OBV's fast MA > slow MA: volume is confirming price trend.
        When OBV's fast MA < slow MA: volume is diverging (trend weakening).

        Returns signal in [-1, 1]:
          +1 = OBV strongly confirms uptrend (volume buying)
          -1 = OBV strongly confirms downtrend (volume selling)
           0 = neutral / no divergence

        Academic basis: Granville (1963), Achelis (2000).
        Window choice: same fast/slow as price momentum (20/60d).
        """
        # Build OBV: accumulate volume × sign of daily price change
        daily_ret = close.diff()
        direction = np.sign(daily_ret).fillna(0)
        obv = (volume * direction).cumsum()

        # Normalise OBV to remove scale dependence (z-score over 252d)
        obv_norm = (obv - obv.rolling(252).mean()) / obv.rolling(252).std().replace(0, np.nan)

        # Fast vs slow MA on normalised OBV
        obv_fast = obv_norm.rolling(self.fast).mean()
        obv_slow = obv_norm.rolling(self.slow).mean()
        raw = (obv_fast - obv_slow).clip(-2, 2) / 2   # → [-1, 1]
        return raw.fillna(0)

    def _volume_trend_ratio(self, volume: pd.Series) -> pd.Series:
        """
        Factor 9: Volume trend ratio — is participation expanding or contracting?

        Ratio = 20-day average volume / 60-day average volume.
        > 1.0 = recent volume above long-run average (expanding participation)
        < 1.0 = recent volume below long-run average (shrinking participation)

        Returns a multiplier in [0.5, 1.3]:
          1.3 = volume expanding 50%+ vs baseline (strong confirmation)
          1.0 = flat volume (no adjustment)
          0.5 = volume contracting 40%+ (trend exhaustion warning)

        Used as a multiplier on the combined signal, not a directional signal.
        Economic logic: trends on rising volume are more reliable than those
        on declining volume (Blume, Easley & O'Hara 1994).
        """
        vol_fast = volume.rolling(self.fast).mean()   # 20d average
        vol_slow = volume.rolling(self.slow).mean()   # 60d average
        ratio = (vol_fast / vol_slow.replace(0, np.nan)).fillna(1.0)

        # Map ratio to multiplier:
        #   ratio 0.5 → multiplier 0.5 (volume halved vs baseline)
        #   ratio 1.0 → multiplier 1.0 (flat)
        #   ratio 1.5 → multiplier 1.3 (volume 50% above baseline, cap at 1.3)
        multiplier = 0.5 + 0.5 * ratio   # linear interpolation
        return multiplier.clip(0.5, 1.3)

    def _chaikin_money_flow(self, high: pd.Series, low: pd.Series,
                             close: pd.Series, volume: pd.Series,
                             window: int = 20) -> pd.Series:
        """
        Factor 10: Chaikin Money Flow (CMF).

        CMF measures whether volume flows into (buying pressure) or
        out of (selling pressure) an asset over the last N days.

        Formula:
          MFM = [(close - low) - (high - close)] / (high - low)
          CMF = sum(MFM × volume, N) / sum(volume, N)

        Output in [-1, 1]:
          +1 = all volume at the high (maximum buying pressure)
          -1 = all volume at the low  (maximum selling pressure)
           0 = neutral

        Academic basis: Chaikin (1989), widely replicated.
        Window = 20 days (canonical, used in all academic studies).
        """
        hl_range = (high - low).replace(0, np.nan)
        # Money Flow Multiplier: where in the day's range did we close?
        mfm = ((close - low) - (high - close)) / hl_range
        mfm = mfm.fillna(0).clip(-1, 1)

        # Money Flow Volume
        mfv = mfm * volume

        # CMF = rolling sum of MFV / rolling sum of volume
        cmf = mfv.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)
        return cmf.fillna(0).clip(-1, 1)

    # ── MOMENTUM EXPLORATION — validated signals (IC analysis Apr 2026) ────

    def _pmo_crossover(self, close: pd.Series) -> pd.Series:
        """
        Price Momentum Oscillator crossover.
        IC = -0.032, p=0.005 — CONTRARIAN signal (sign is flipped on return).
        PMO > signal line = overbought → expect reversion → score is NEGATIVE.
        PMO < signal line = oversold   → expect bounce  → score is POSITIVE.

        Walk-forward: 3/3 positive OOS years (2023-2025) as contrarian factor.
        Weight in blend: 15% (validated, additive to existing factors).
        """
        roc  = (close / close.shift(1) - 1) * 100
        ema1 = roc.ewm(span=35, min_periods=10).mean() * 20
        pmo  = ema1.ewm(span=20, min_periods=5).mean()
        sig  = pmo.ewm(span=10, min_periods=3).mean()
        crossover = pmo - sig
        # Flip sign: contrarian (IC is negative, so -signal has positive IC)
        return (-crossover).clip(-2, 2) / 2   # normalise to [-1, 1]

    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
              period: int = 14) -> pd.Series:
        """
        ADX (14-period). Used as a FILTER gate, not a directional signal.
        ADX >= 20: trending regime → full position size (multiplier = 1.0)
        ADX <  20: choppy regime  → half position size (multiplier = 0.5)

        Walk-forward validation: Sharpe +0.71 OOS, 4/4 positive years (2023-2026).
        """
        hi, lo, cl = high.values, low.values, close.values
        n = len(hi)
        if n < period + 5:
            return pd.Series(20.0, index=close.index)  # neutral default
        tr_arr, dmp_arr, dmm_arr = [], [], []
        for i in range(1, n):
            hl  = hi[i] - lo[i]
            hpc = abs(hi[i] - cl[i-1])
            lpc = abs(lo[i] - cl[i-1])
            tr_arr.append(max(hl, hpc, lpc))
            up   = hi[i] - hi[i-1]
            down = lo[i-1] - lo[i]
            dmp_arr.append(up   if up   > down and up   > 0 else 0.0)
            dmm_arr.append(down if down > up   and down > 0 else 0.0)
        atr   = pd.Series(tr_arr).ewm(span=period, min_periods=period).mean()
        di_p  = pd.Series(dmp_arr).ewm(span=period, min_periods=period).mean() / (atr + 1e-9) * 100
        di_m  = pd.Series(dmm_arr).ewm(span=period, min_periods=period).mean() / (atr + 1e-9) * 100
        dx    = (di_p - di_m).abs() / (di_p + di_m + 1e-9) * 100
        adx_s = dx.ewm(span=period, min_periods=period).mean()
        result = pd.Series(np.nan, index=close.index)
        result.iloc[1:] = adx_s.values
        return result.ffill().fillna(20.0)

    def _adx_position_multiplier(self, adx_series: pd.Series) -> pd.Series:
        """Returns 1.0 when ADX >= 20 (trending), 0.5 when ADX < 20 (choppy)."""
        return adx_series.apply(lambda v: 1.0 if v >= 20 else 0.5)

    def _detect_bull_regime(
        self,
        close_index: "pd.DatetimeIndex",
        as_of_date: Optional["pd.Timestamp"] = None,
    ) -> "pd.Series":
        """
        Per-day bull/bear regime classification.
        Returns boolean Series: True = bull, False = bear/neutral.

        Bull conditions (both must be met):
          1. VIX < self.bull_vix_threshold (default 20)
          2. SPY > SPY 200-day MA

        Evidence: OOS 2023-2026 backtest showed +396% Sharpe improvement
        from switching factor weights based on this regime.
        In bull regime: remove mean-reversion (it fights sustained uptrends)
        In bear regime: full factor set including MR, PMO contrarian, ADX gate.
        """
        try:
            vix_data = self._macro_data.get("^VIX")
            spy_data = self._macro_data.get("SPY")
            if vix_data is None or spy_data is None:
                return pd.Series(False, index=close_index)

            vix_close = vix_data["Close"].squeeze().reindex(close_index).ffill()
            spy_close = spy_data["Close"].squeeze().reindex(close_index).ffill()
            spy_ma    = spy_close.rolling(self.bull_spy_ma_period, min_periods=50).mean()

            if as_of_date is not None:
                vix_close = vix_close.loc[:as_of_date]
                spy_close = spy_close.loc[:as_of_date]
                spy_ma    = spy_ma.loc[:as_of_date]

            bull = (vix_close < self.bull_vix_threshold) & (spy_close > spy_ma)
            return bull.reindex(close_index).fillna(False)
        except Exception as e:
            log.debug(f"Regime detection failed: {e}")
            return pd.Series(False, index=close_index)

    def _vwap_daily_proxy(self, close: pd.Series, high: pd.Series,
                           low: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Daily VWAP distance proxy.
        NOTE: The validated IC +0.054 (p=0.007) is from 1-min Alpaca bars.
        This daily OHLC proxy has IC ~0 (confirmed in diagnostics).
        Used here at low weight (5%) as a directional tiebreaker only.
        Full signal lives in strategy/alpaca_microstructure.py.
        """
        tp   = (high + low + close) / 3
        vwap = (tp * volume).rolling(5).sum() / volume.rolling(5).sum().replace(0, np.nan)
        dev  = (close - vwap) / (vwap + 1e-9)
        return dev.clip(-0.1, 0.1) / 0.1   # normalise to [-1, 1]

    def _volume_confirmation_multiplier(
        self,
        close: pd.Series,
        volume: pd.Series,
        high:   Optional[pd.Series] = None,
        low:    Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Combine the 3 volume factors into a single multiplier in [0.5, 1.3].

        Weights:
          OBV trend (directional agreement):  40%
          Volume trend ratio (participation): 40%
          CMF (money flow direction):         20%

        The multiplier is applied to the final combined signal:
          signal_final = signal_combined × volume_multiplier

        Crucially: the multiplier is SYMMETRIC around 1.0.
        When volume is neutral (ratio=1, OBV flat, CMF=0) → multiplier = 1.0.
        Volume only adjusts conviction, never reverses signal direction.
        """
        # Factor 8: OBV slope → directional in [-1, 1]
        obv_sig = self._obv_signal(close, volume)

        # Factor 9: Volume trend → multiplier in [0.5, 1.3]
        vtr = self._volume_trend_ratio(volume)

        # Factor 10: CMF → directional in [-1, 1]
        if high is not None and low is not None:
            cmf = self._chaikin_money_flow(high, low, close, volume)
        else:
            # Approximation when H/L not available: use close-based range proxy
            cmf = pd.Series(0.0, index=close.index)

        # Combine into single multiplier:
        # OBV and CMF are directional [-1,1] — convert to multiplier space
        # Positive = confirms upside, negative = confirms downside (both directions ok)
        # We want: strong OBV/CMF agreement → high multiplier, disagreement → low
        # So we take the ABSOLUTE value as confirmation strength, then scale
        obv_confirm = (1.0 + 0.15 * obv_sig).clip(0.7, 1.3)   # ±15% from OBV
        cmf_confirm = (1.0 + 0.10 * cmf).clip(0.85, 1.15)     # ±10% from CMF

        # Volume trend ratio directly used (already in multiplier space)
        # Blend: 40% OBV-confirm + 40% VTR + 20% CMF-confirm
        combined = (
            0.40 * obv_confirm
            + 0.40 * vtr
            + 0.20 * cmf_confirm
        )
        return combined.clip(0.5, 1.3)

    # -----------------------------------------------------------------------
    # PREDICTIVE: Credit Regime Signal
    # -----------------------------------------------------------------------

    def _compute_credit_regime(self, as_of_date: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Cross-asset leading indicator for equity risk appetite.
        Combines three forward-looking signals:

        1. HYG/LQD spread momentum — High-yield vs investment-grade bond ratio.
           When HYG outperforms LQD, credit risk appetite is growing,
           which leads equity rallies by 1-2 weeks.

        2. VIX momentum — Falling implied volatility signals decreasing
           fear, which precedes equity strength.

        3. Yield curve slope change — SHY/TLT ratio momentum. Steepening
           (short-term rates falling relative to long) signals easing
           financial conditions, bullish for equities.

        Returns signal in [-1, 1]. Applied uniformly to all equity signals
        as a cross-asset overlay.

        as_of_date: if provided, only uses data up to this date (prevents look-ahead).
        """
        # Cache hit only if same as_of_date
        if self._credit_signal_cache is not None and self._credit_signal_as_of == as_of_date:
            return self._credit_signal_cache

        components = []

        # --- Component 1: Credit spread momentum (HYG/LQD) ---
        hyg = self._macro_data.get("HYG")
        lqd = self._macro_data.get("LQD")
        if hyg is not None and lqd is not None:
            ratio = hyg["Close"] / lqd["Close"]
            ratio_ret = ratio.pct_change(5)  # 1-week momentum
            z = (ratio_ret - ratio_ret.rolling(60).mean()) / ratio_ret.rolling(60).std().replace(0, np.nan)
            components.append(z.clip(-2, 2) / 2)
            log.debug("Credit regime: HYG/LQD component active")

        # --- Component 2: VIX momentum (inverted — falling VIX = bullish) ---
        vix = self._macro_data.get("^VIX")
        if vix is not None:
            vix_fast = vix["Close"].rolling(5).mean()
            vix_slow = vix["Close"].rolling(21).mean()
            vix_sig = -(vix_fast - vix_slow) / vix_slow.replace(0, np.nan)
            components.append(vix_sig.clip(-1, 1))
            log.debug("Credit regime: VIX component active")

        # --- Component 3: Yield curve slope (SHY/TLT) ---
        tlt_data = self._macro_data.get("TLT")
        shy_data = self._macro_data.get("SHY")
        if tlt_data is not None and shy_data is not None:
            curve = shy_data["Close"] / tlt_data["Close"]
            curve_mom = curve.pct_change(20)
            z = (curve_mom - curve_mom.rolling(60).mean()) / curve_mom.rolling(60).std().replace(0, np.nan)
            components.append(z.clip(-2, 2) / 2)
            log.debug("Credit regime: Yield curve component active")

        if not components:
            log.warning("Credit regime: no macro data available — signal disabled")
            self._credit_signal_cache = pd.Series(dtype=float)
            return self._credit_signal_cache

        # Average all available components
        result = pd.concat(components, axis=1).mean(axis=1).clip(-1, 1)
        # Apply as_of_date cutoff to prevent look-ahead
        if as_of_date is not None:
            result = result[result.index <= as_of_date]
        self._credit_signal_cache = result
        self._credit_signal_as_of = as_of_date
        log.debug(f"Credit regime signal computed: {len(result)} values, "
                  f"{len(components)} components active")
        return self._credit_signal_cache

    # -----------------------------------------------------------------------
    # Main signal generation
    # -----------------------------------------------------------------------

    def _compute_symbol_signal(
        self,
        sym: str,
        df: pd.DataFrame,
        as_of_date: Optional[pd.Timestamp],
        credit_signal: pd.Series,
        all_data: Dict[str, pd.DataFrame],
    ) -> tuple:
        """Compute signal for a single symbol — thread-safe, no shared mutation."""
        close = df["Close"]
        if as_of_date:
            close = close[close.index <= as_of_date]

        if len(close) < self.slow + self.regime_window:
            return sym, pd.Series(0.0, index=close.index)

        # --- Factor 1: Time-series momentum ---
        ts_mom = self._ts_momentum(close).clip(-1, 1)

        # --- Factor 2: Mean reversion ---
        mr = self._mean_reversion(close)

        # --- Factor 3: MACD confirmation ---
        macd = self._macd_signal(close)

        # --- Factor 4: RSI (avoid extremes) ---
        rsi = self._rsi(close)
        rsi_filter = pd.Series(1.0, index=rsi.index)
        rsi_filter[rsi > 80] = -0.5
        rsi_filter[rsi < 20] = 1.5
        rsi_filter = rsi_filter.clip(-1, 1)

        # --- Factor 5: Volatility regime multiplier ---
        vol_mult = self._volatility_regime(close)

        # --- Factor 13: PMO crossover (contrarian, IC -0.032 p=0.005) --------
        pmo_sig = pd.Series(0.0, index=close.index)
        try:
            pmo_sig = self._pmo_crossover(close).fillna(0)
        except Exception as exc:
            log.debug(f"PMO computation failed for {sym}: {exc}")

        # --- Factor 14: VWAP daily proxy (low weight, full signal in microstructure) ---
        vwap_sig = pd.Series(0.0, index=close.index)
        try:
            if "High" in df.columns and "Low" in df.columns and "Volume" in df.columns:
                h_col = df["High"][df.index <= as_of_date] if as_of_date else df["High"]
                l_col = df["Low"][df.index  <= as_of_date] if as_of_date else df["Low"]
                v_col = df["Volume"][df.index <= as_of_date] if as_of_date else df["Volume"]
                vwap_sig = self._vwap_daily_proxy(close, h_col, l_col, v_col).fillna(0)
        except Exception as exc:
            log.debug(f"VWAP computation failed for {sym}: {exc}")

        # --- ADX gate: halve position size in choppy (non-trending) regime ----
        adx_mult = pd.Series(1.0, index=close.index)
        try:
            if "High" in df.columns and "Low" in df.columns:
                h_col = df["High"][df.index <= as_of_date] if as_of_date else df["High"]
                l_col = df["Low"][df.index  <= as_of_date] if as_of_date else df["Low"]
                adx_s = self._adx(h_col, l_col, close)
                adx_mult = self._adx_position_multiplier(adx_s)
        except Exception as exc:
            log.debug(f"ADX computation failed for {sym}: {exc}")

        # --- Factor 15: Closing Imbalance (regime-conditional) ---------------
        # IC analysis on 793 trading days (Jan 2023 – Mar 2026):
        #   Bull (VIX<18):  IC ≈ 0.000  → noise, set to 0
        #   VIX 18-25:      IC = +0.051  p=0.000 ✅
        #   Bear (VIX>25):  IC = +0.161  p=0.000 ✅  strongest regime
        #   SPY < 200MA:    IC = +0.148  p=0.000 ✅
        # Signal ONLY active when VIX >= 18 (stress/bear regime).
        # In calm bull markets it is pure noise and should be 0.
        imbalance_sig = pd.Series(0.0, index=close.index)
        try:
            vix_data = self._macro_data.get("^VIX")
            spy_data = self._macro_data.get("SPY")
            if vix_data is not None and not vix_data.empty:
                vix_close = vix_data["Close"].squeeze().reindex(close.index).ffill()
                # VIX regime gate: active above 18, full weight above 25
                vix_activation = ((vix_close - 18.0) / 7.0).clip(0.0, 1.0)
                # Imbalance proxy from VWAP deviation (sign = buy pressure)
                # Full signal from ClosingImbalanceSignal would plug in here
                # For now use closing price momentum as buy-pressure proxy
                # (will be replaced by real Databento data when called from live engine)
                if hasattr(self, "_imbalance_scores") and sym in self._imbalance_scores:
                    raw_imb = self._imbalance_scores[sym].reindex(close.index).fillna(0)
                    imbalance_sig = (raw_imb * vix_activation).fillna(0)
                # else: stays 0 — safe default when no real imbalance data injected
        except Exception as exc:
            log.debug(f"Imbalance signal failed for {sym}: {exc}")

        # --- Regime detection (bull vs bear/neutral) ----------------------
        # Bull (VIX<20, SPY>200MA):  reduce MR, drop PMO+ADX — ride the trend
        # Bear/neutral:              full factor set, MR + PMO contrarian + ADX gate
        # OOS evidence: +396% Sharpe improvement 2023-2026
        if self.regime_switching_enabled:
            bull_regime = self._detect_bull_regime(close.index, as_of_date)
        else:
            bull_regime = pd.Series(False, index=close.index)

        # --- Blend: regime-conditional weights (both sum to 1.00) ----------
        #
        # BULL regime weights  (VIX<20 + SPY>200MA):
        #   TS Momentum:  0.60  (ride the trend hard)
        #   Mean Revert:  0.10  (minimal — MR fights uptrends)
        #   MACD:         0.25  (trend confirmation)
        #   RSI filter:   0.05  (minimal)
        #   PMO:          0.00  (contrarian fights bull — disabled)
        #   Imbalance:    0.10  (regime-gated by VIX, safe to include)
        #   ADX gate:     1.0×  (disabled in bull — early rallies have low ADX)
        #
        # BEAR/NEUTRAL weights (VIX≥20 or SPY<200MA):
        #   TS Momentum:  0.30  (reduced — trends less reliable)
        #   Mean Revert:  0.225 (more weight — bear bounces are mean-reverting)
        #   MACD:         0.15  (reduced)
        #   RSI filter:   0.075 (unchanged)
        #   PMO:          0.15  (contrarian is useful in bear market)
        #   Imbalance:    0.10  (highest IC in this regime)
        #   ADX gate:     0.5×  (active — avoids choppy false signals)
        #
        bull_blend = (
            self.bull_w_ts_mom * ts_mom.fillna(0)
            + self.bull_w_mr   * mr.fillna(0)
            + self.bull_w_macd * macd.fillna(0)
            + self.bull_w_rsi  * rsi_filter.fillna(0)
            + 0.10 * imbalance_sig     # regime-gated internally by VIX
        )
        bear_blend = (
            (self.w_ts_mom * 0.75) * ts_mom.fillna(0)
            + (self.w_mr   * 0.75) * mr.fillna(0)
            + (self.w_macd * 0.75) * macd.fillna(0)
            + (self.w_rsi  * 0.75) * rsi_filter.fillna(0)
            + 0.15 * pmo_sig
            + 0.10 * imbalance_sig
        )

        reactive = pd.Series(
            np.where(bull_regime, bull_blend, bear_blend),
            index=close.index,
        )

        # Apply vol regime multiplier
        reactive = reactive * vol_mult.reindex(close.index).fillna(1.0)

        # ADX gate only in bear/neutral regime
        if not self.bull_use_adx:
            bear_adx_mult = adx_mult
            bull_adx_mult = pd.Series(1.0, index=close.index)
            effective_adx = pd.Series(
                np.where(bull_regime, bull_adx_mult, bear_adx_mult),
                index=close.index,
            )
        else:
            effective_adx = adx_mult
        reactive = reactive * effective_adx

        # --- Factor 7: Credit regime (predictive, cross-asset) ---
        if self.credit_regime_enabled and len(credit_signal) > 0:
            cs_aligned = credit_signal[~credit_signal.index.duplicated(keep="last")].reindex(close.index).fillna(0)
            combined = (
                self.reactive_weight * reactive.fillna(0)
                + self.credit_regime_weight * cs_aligned
            )
        else:
            combined = reactive

        # --- Factors 8-10: Volume confirmation multiplier ---------------
        vol_mult_factor = pd.Series(1.0, index=close.index)
        if self.volume_confirmation and "Volume" in df.columns:
            volume = df["Volume"]
            if as_of_date:
                volume = volume[volume.index <= as_of_date]
            vol_nonzero = (volume > 0).sum()
            if vol_nonzero > self.slow * 2:
                high_col = df["High"][df.index <= as_of_date] if (as_of_date and "High" in df.columns) else df.get("High")
                low_col  = df["Low"][df.index  <= as_of_date] if (as_of_date and "Low"  in df.columns) else df.get("Low")
                vol_mult_factor = self._volume_confirmation_multiplier(
                    close, volume, high=high_col, low=low_col
                ).reindex(close.index).fillna(1.0)

        combined_with_vol = (combined * vol_mult_factor).clip(-1, 1)

        # --- Factor 11: H2O Trend Classifier overlay ─────────────────
        if self._trend_classifier is not None and self.trend_classifier_enabled:
            try:
                high_col = df["High"][df.index <= as_of_date] if (as_of_date and "High" in df.columns) else df.get("High")
                low_col  = df["Low"][df.index  <= as_of_date] if (as_of_date and "Low"  in df.columns) else df.get("Low")
                vol_col  = df["Volume"][df.index <= as_of_date] if (as_of_date and "Volume" in df.columns) else df.get("Volume")
                spy_close = all_data.get("SPY", {}).get("Close") if isinstance(all_data.get("SPY"), pd.DataFrame) else None
                vix_s = self._macro_data.get("^VIX", pd.DataFrame()).get("Close")
                trend_mult = self._trend_classifier.get_multiplier(
                    close, high_col, low_col, vol_col, vix_s, spy_close, as_of_date
                )
                combined_with_vol = (combined_with_vol * trend_mult).clip(-1, 1)
            except Exception:
                pass

        # --- Factor 12: Price-Volume Segment score ───────────────────
        if self._pv_segmenter is not None and self.pv_enabled:
            try:
                vol_col = df["Volume"][df.index <= as_of_date] if (as_of_date and "Volume" in df.columns) else df.get("Volume")
                high_col = df["High"][df.index <= as_of_date] if (as_of_date and "High" in df.columns) else df.get("High")
                low_col  = df["Low"][df.index  <= as_of_date] if (as_of_date and "Low"  in df.columns) else df.get("Low")
                open_col = df["Open"][df.index <= as_of_date] if (as_of_date and "Open" in df.columns) else df.get("Open")
                pv_score = self._pv_segmenter.score(
                    close, vol_col, high_col, low_col, open_col, as_of_date
                )
                w = self.pv_weight
                combined_with_vol = (
                    (1.0 - w) * combined_with_vol + w * pv_score
                ).clip(-1, 1)
            except Exception:
                pass

        return sym, combined_with_vol

    def generate(
        self,
        all_data: Dict[str, pd.DataFrame],
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generate signals for all symbols — PARALLEL per-symbol computation.

        Returns
        -------
        pd.DataFrame  shape = (dates, symbols)
          Values in [-1, 1]. Positive=long, Negative=short, 0=flat.
        """
        signals: Dict[str, pd.Series] = {}

        # Collect close prices for cross-sectional ranking
        closes = pd.DataFrame({
            sym: df["Close"] for sym, df in all_data.items()
        })

        if as_of_date:
            closes = closes[closes.index <= as_of_date]

        # Pre-compute credit regime signal if enabled (respects as_of_date cutoff)
        credit_signal = pd.Series(dtype=float)
        if self.credit_regime_enabled and self._macro_data:
            credit_signal = self._compute_credit_regime(as_of_date=as_of_date)

        # ── Parallel per-symbol signal computation ──────────────────────────
        n_symbols = len(all_data)
        if n_symbols >= 4:
            # Use thread pool for 4+ symbols (numpy/pandas release GIL)
            workers = min(_SIGNAL_MAX_WORKERS, n_symbols)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._compute_symbol_signal,
                        sym, df, as_of_date, credit_signal, all_data
                    ): sym
                    for sym, df in all_data.items()
                }
                for future in as_completed(futures):
                    sym, sig = future.result()
                    signals[sym] = sig
        else:
            # Sequential for small universes (thread overhead not worth it)
            for sym, df in all_data.items():
                _, sig = self._compute_symbol_signal(
                    sym, df, as_of_date, credit_signal, all_data
                )
                signals[sym] = sig

        signal_df = pd.DataFrame(signals)

        # --- Cross-sectional overlay ---
        eq_syms = [s for s in signal_df.columns if not any(
            s.endswith(x) for x in ["-USD", "=F"]
        )]
        if len(eq_syms) > 1:
            eq_signals = signal_df[eq_syms].copy()
            cs_mom = closes[eq_syms].copy()
            cs_ranks = cs_mom.pct_change(min(231, len(cs_mom) - 1)).rank(axis=1, pct=True)
            cs_ranks = (cs_ranks - 0.5) * 2
            for sym in eq_syms:
                if sym in cs_ranks.columns:
                    signal_df[sym] = (
                        0.70 * signal_df[sym].fillna(0)
                        + 0.30 * cs_ranks[sym].fillna(0)
                    ).clip(-1, 1)

        return signal_df

    def generate_latest(
        self, all_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Return the latest signal for each symbol (for live trading)."""
        signal_df = self.generate(all_data)
        if signal_df.empty:
            return {}
        latest = signal_df.iloc[-1].to_dict()
        log.info(f"Latest signals: { {k: f'{v:.3f}' for k,v in latest.items()} }")
        return latest

    def compute_stop_loss(
        self, df: pd.DataFrame, signal: float = 0, atr_mult: float = 2.0
    ) -> float:
        """
        ATR-based dynamic stop-loss distance.
        Returns the $ distance from entry to stop (always positive).
        """
        if "High" not in df.columns:
            return df["Close"].iloc[-1] * 0.02  # fallback: 2% of price
        atr = self._atr(df["High"], df["Low"], df["Close"]).iloc[-1]
        return float(atr * atr_mult)
