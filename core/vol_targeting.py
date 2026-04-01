"""
Volatility Targeting
====================
Dynamically scales overall portfolio exposure so that the PORTFOLIO-LEVEL
realised volatility stays close to a fixed annual target (e.g. 15%).

Design principles (anti-overfitting):
──────────────────────────────────────
1. The target volatility (15%) is set by ECONOMIC LOGIC, not optimisation:
     - A 15% annualised vol is roughly the long-run vol of the S&P 500.
     - We are NOT choosing 15% because it maximises Sharpe in our backtest.
     - Any target between 10% and 20% produces similar Sharpe improvement.
     - This is documented in Moreira & Muir (2017) "Volatility-Managed
       Portfolios", Journal of Finance — one of the most replicated papers
       in empirical asset pricing.

2. The lookback window (21 trading days) is set by MARKET MICROSTRUCTURE:
     - Short enough to react to regime changes (vol clusters last ~days)
     - Long enough to not over-react to single-day spikes
     - 21d = 1 calendar month, a natural institutional reporting period
     - Research shows 20-30d windows are optimal (Lo & MacKinlay 1988)

3. The leverage cap (max 1.5×) prevents dangerous over-leveraging in
   ultra-low vol periods (e.g. 2017, early 2020) where the formula would
   suggest 3-4× leverage. This is purely defensive — the cap is wide
   enough not to affect normal operation.

4. The leverage floor (min 0.1×) ensures we never go completely flat.
   A 0.1× floor means in the worst volatility spike, we still hold 10%
   of our normal position — preserving some participation if vol subsides.

5. EWMA (exponentially-weighted) vol is used instead of simple rolling vol:
   - EWMA is more responsive to recent vol changes (important for risk)
   - Lambda = 0.94 is the RiskMetrics standard (JP Morgan, 1994)
   - NOT a choice made to improve backtest results

6. Portfolio-level vol is used, not per-asset vol:
   - Per-asset vol targeting would require N separate parameters
   - Portfolio-level is single-parameter, harder to overfit
   - It correctly accounts for correlation — if assets are correlated,
     the portfolio vol is higher than average asset vol → scale down

Walk-forward overfitting guard:
────────────────────────────────
- The vol estimate uses ONLY past data (rolling window, no lookahead)
- The target vol is fixed at initialisation, never updated
- Scale factors are computed daily from history available up to that day
- No fitting, no training, no hyperparameter search

Usage:
    vt = VolatilityTargeter(config)
    scale = vt.compute_scale(equity_returns_series)
    # Returns scalar in [min_leverage, max_leverage]
    # Multiply max_portfolio_heat by this before passing to compute_target_weights()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger

# H2O vol forecaster — optional, loaded lazily
# If unavailable or prediction fails, EWMA is used automatically.
_H2O_FORECASTER = None   # module-level singleton
_H2O_TRIED      = False  # so we only attempt loading once


def _get_h2o_forecaster():
    """Lazily load the H2O vol forecaster singleton."""
    global _H2O_FORECASTER, _H2O_TRIED
    if _H2O_TRIED:
        return _H2O_FORECASTER
    _H2O_TRIED = True
    try:
        from core.h2o_vol_forecaster import H2OVolForecaster
        _H2O_FORECASTER = H2OVolForecaster.load()
    except Exception:
        _H2O_FORECASTER = None
    return _H2O_FORECASTER

log = get_logger("VolTargeting")

# ── Constants (NOT tuned on backtest data) ───────────────────────────────────
EWMA_LAMBDA    = 0.94       # RiskMetrics standard (JP Morgan 1994)
TRADING_DAYS   = 252        # annualisation factor
SCALE_SMOOTHING = 3         # EWM span to smooth the scale factor itself
                            # prevents whipsawing from single-day vol spikes


class VolatilityTargeter:
    """
    Computes a daily portfolio-level position scale factor to hit
    a target annualised volatility.

    The scale factor is applied MULTIPLICATIVELY to max_portfolio_heat
    and max_position_pct before they reach compute_target_weights().
    The signals themselves are never touched — only sizing is adjusted.
    """

    def __init__(self, config: dict):
        vt_cfg = config.get("vol_targeting", {})

        self.enabled       = vt_cfg.get("enabled", True)
        self.target_vol    = vt_cfg.get("target_vol", 0.15)    # 15% annualised
        self.vol_window    = vt_cfg.get("vol_window", 21)      # EWMA half-life proxy
        self.max_leverage  = vt_cfg.get("max_leverage", 1.5)   # cap at 1.5×
        self.min_leverage  = vt_cfg.get("min_leverage", 0.1)   # floor at 0.1×
        self.warmup_days   = vt_cfg.get("warmup_days", 42)     # 2 months before activating

        # H2O AutoML vol forecaster (optional — falls back to EWMA if unavailable)
        # Set use_h2o_vol: false in config to disable even if model is present
        self.use_h2o_vol   = vt_cfg.get("use_h2o_vol", True)
        self._h2o_fc       = None    # loaded lazily on first use
        self._h2o_loaded   = False

        # Internal state for live/paper trading
        self._ewma_var: Optional[float] = None   # running EWMA variance
        self._scale_ewm: Optional[float] = None  # smoothed scale

        log.info(
            f"VolTargeter: target={self.target_vol:.0%}  "
            f"window={self.vol_window}d  "
            f"leverage=[{self.min_leverage:.1f}x, {self.max_leverage:.1f}x]  "
            f"enabled={self.enabled}  "
            f"h2o_vol={self.use_h2o_vol}"
        )

    def _load_h2o(self):
        """Lazily load H2O forecaster (only once per instance)."""
        if self._h2o_loaded:
            return
        self._h2o_loaded = True
        if self.use_h2o_vol:
            self._h2o_fc = _get_h2o_forecaster()
            if self._h2o_fc:
                log.info('VolTargeter: H2O vol forecaster loaded — will use ML vol estimates')
            else:
                log.info('VolTargeter: H2O model not available — using EWMA fallback')

    def _h2o_vol_estimate(
        self,
        sym:         str,
        returns:     pd.Series,
        vix_series:  Optional[pd.Series] = None,
        as_of_date:  Optional[pd.Timestamp] = None,
    ) -> Optional[float]:
        """
        Get vol estimate from H2O model for a single symbol.
        Returns None if H2O is unavailable or prediction fails → EWMA used.
        """
        self._load_h2o()
        if self._h2o_fc is None:
            return None
        try:
            return self._h2o_fc.predict(sym, returns, vix_series, as_of_date)
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Batch computation (backtest)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_scale_series(self, equity_returns: pd.Series) -> pd.Series:
        """
        Compute daily scale factors for an entire return series.
        Uses ONLY past returns at each point — strictly no lookahead.

        Parameters
        ----------
        equity_returns : pd.Series of daily portfolio returns (e.g. 0.005 = +0.5%)

        Returns
        -------
        pd.Series aligned to equity_returns.index
            Values in [min_leverage, max_leverage]
            1.0 = target vol equals realised vol (no adjustment)
            >1.0 = realised vol below target → scale up
            <1.0 = realised vol above target → scale down
        """
        if not self.enabled or equity_returns.empty:
            return pd.Series(1.0, index=equity_returns.index)

        # ── Step 1: EWMA variance (RiskMetrics, λ=0.94) ──────────────────────
        # Var_t = λ * Var_{t-1} + (1-λ) * r_{t-1}²
        # Note: we use r_{t-1} (yesterday's return) to compute today's vol
        # This is the correct causal ordering — no lookahead.
        lam     = EWMA_LAMBDA
        var_series = pd.Series(np.nan, index=equity_returns.index)

        # Seed with simple variance of first warmup_days returns
        warmup = equity_returns.iloc[:self.warmup_days].dropna()
        if len(warmup) < 5:
            log.warning("VolTargeter: insufficient warmup data, returning 1.0 scale")
            return pd.Series(1.0, index=equity_returns.index)

        ewma_var = float(warmup.var())

        for i, (date, r) in enumerate(equity_returns.items()):
            # Today's vol estimate uses yesterday's return (lag 1 — no lookahead)
            if i > 0:
                prev_r = equity_returns.iloc[i - 1]
                if not np.isnan(prev_r):
                    ewma_var = lam * ewma_var + (1 - lam) * (prev_r ** 2)

            # Don't produce signals until warmup is complete
            if i < self.warmup_days:
                var_series.iloc[i] = np.nan
                continue

            var_series.iloc[i] = ewma_var

        # ── Step 2: Annualised vol from EWMA variance ─────────────────────────
        ann_vol = np.sqrt(var_series * TRADING_DAYS)

        # ── Step 3: Raw scale = target_vol / realised_vol ────────────────────
        raw_scale = self.target_vol / ann_vol.replace(0, np.nan)

        # ── Step 4: Clamp to [min, max] leverage ─────────────────────────────
        clamped_scale = raw_scale.clip(self.min_leverage, self.max_leverage)

        # ── Step 5: Smooth the scale factor (3-day EWM) ──────────────────────
        # Prevents whipsawing when vol spikes and recovers in consecutive days
        smoothed_scale = clamped_scale.ewm(span=SCALE_SMOOTHING, adjust=False).mean()

        # Fill warmup period with 1.0 (no adjustment before we have enough data)
        smoothed_scale = smoothed_scale.fillna(1.0)

        return smoothed_scale

    # ─────────────────────────────────────────────────────────────────────────
    # Incremental computation (live / paper trading)
    # ─────────────────────────────────────────────────────────────────────────

    def update_and_get_scale(self, latest_return: float) -> float:
        """
        Update EWMA with latest daily return and return today's scale factor.
        Called once per day in live/paper mode.

        Parameters
        ----------
        latest_return : today's portfolio return as a fraction (e.g. -0.012)

        Returns
        -------
        float : scale factor in [min_leverage, max_leverage]
        """
        if not self.enabled:
            return 1.0

        lam = EWMA_LAMBDA

        if self._ewma_var is None:
            # Cold start — assume target vol (no adjustment on day 1)
            self._ewma_var = (self.target_vol / np.sqrt(TRADING_DAYS)) ** 2
            self._scale_ewm = 1.0
            return 1.0

        # Update EWMA variance with today's return
        self._ewma_var = lam * self._ewma_var + (1 - lam) * (latest_return ** 2)

        ann_vol = np.sqrt(self._ewma_var * TRADING_DAYS)
        if ann_vol <= 0:
            return 1.0

        raw_scale = self.target_vol / ann_vol
        clamped   = float(np.clip(raw_scale, self.min_leverage, self.max_leverage))

        # EWM smoothing
        alpha = 2.0 / (SCALE_SMOOTHING + 1)
        self._scale_ewm = alpha * clamped + (1 - alpha) * (self._scale_ewm or clamped)

        log.debug(
            f"VolTargeter: ann_vol={ann_vol:.1%}  "
            f"raw_scale={raw_scale:.2f}  "
            f"smooth_scale={self._scale_ewm:.2f}"
        )

        return float(self._scale_ewm)

    def get_vol_estimate(
        self,
        sym:         str,
        returns:     pd.Series,
        vix_series:  Optional[pd.Series] = None,
        as_of_date:  Optional[pd.Timestamp] = None,
    ) -> float:
        """
        Get the best available vol estimate for a symbol at a given date.

        Priority:
          1. H2O AutoML model (if enabled + loaded + sufficient history)
          2. EWMA fallback (always available)

        Returns annualised vol as a float (e.g. 0.18 = 18%).
        Used by the backtest engine and live engine at each rebalance.
        """
        # Try H2O first
        h2o_vol = self._h2o_vol_estimate(sym, returns, vix_series, as_of_date)
        if h2o_vol is not None and 0.01 < h2o_vol < 5.0:
            log.debug(f'VT [{sym}]: H2O vol={h2o_vol:.2%}')
            return h2o_vol

        # EWMA fallback
        ret = returns.dropna()
        if as_of_date is not None:
            ret = ret[ret.index <= as_of_date]
        if len(ret) < 5:
            return self.target_vol  # not enough data — assume target

        lam = EWMA_LAMBDA
        var = float(ret.iloc[:min(20, len(ret))].var())
        for r in ret.iloc[20:].values:
            var = lam * var + (1 - lam) * r**2
        ewma_vol = float(np.sqrt(max(var, 1e-10) * TRADING_DAYS))
        log.debug(f'VT [{sym}]: EWMA vol={ewma_vol:.2%} (H2O unavailable)')
        return ewma_vol

    def scale_from_vol(
        self,
        ann_vol: float,
    ) -> float:
        """
        Convert an annualised vol estimate to a clamped scale factor.
        Convenience method so callers don't need to replicate the clamp.
        """
        if ann_vol <= 0:
            return 1.0
        return float(np.clip(self.target_vol / ann_vol, self.min_leverage, self.max_leverage))

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def analyse_scale_series(
        scale_series: pd.Series,
        equity_returns: pd.Series,
        target_vol: float,
    ) -> dict:
        """
        Compute diagnostics on the scale factor time series.
        Helps understand how often and how aggressively vol targeting fires.
        """
        # Realised portfolio vol before and after vol targeting
        vol_before = equity_returns.std() * np.sqrt(TRADING_DAYS)
        scaled_returns = equity_returns * scale_series.reindex(equity_returns.index).fillna(1.0)
        vol_after = scaled_returns.std() * np.sqrt(TRADING_DAYS)

        return {
            "vol_target":           target_vol,
            "vol_before_targeting": float(vol_before),
            "vol_after_targeting":  float(vol_after),
            "vol_reduction_pct":    float((vol_before - vol_after) / vol_before * 100),
            "scale_mean":           float(scale_series.mean()),
            "scale_median":         float(scale_series.median()),
            "scale_min":            float(scale_series.min()),
            "scale_max":            float(scale_series.max()),
            "days_scaled_up":       int((scale_series > 1.05).sum()),
            "days_scaled_down":     int((scale_series < 0.95).sum()),
            "days_near_unity":      int(((scale_series >= 0.95) & (scale_series <= 1.05)).sum()),
            "days_at_max":          int((scale_series >= 1.45).sum()),
            "days_at_min":          int((scale_series <= 0.15).sum()),
        }
