"""
kelly_sizer.py
==============
Kelly-fractional position sizing per signal.

Diagnostic findings:
- Long side and short side individually have NEGATIVE Sharpe.
- The edge is entirely in **relative value** (cross-sectional dispersion).
- Uniform heat allocation leaves alpha on the table because positions are
  sized identically regardless of signal quality.

This module replaces uniform allocation with per-symbol fractional Kelly
sizing derived from the rolling 63-day information coefficient (IC) between
historical signals and their realised forward returns.

Kelly formula (per position)
----------------------------
    p      = rolling signal accuracy (from IC, mapped to [0, 1])
    b      = mean_win / mean_loss  (payoff ratio from return history)
    kelly_raw  = (p * b - (1 - p)) / b
    kelly_size = kelly_raw * kelly_fraction   (default 0.25 — quarter Kelly)
    clipped    = clip(kelly_size, 0, max_position_pct)

Portfolio normalisation
-----------------------
All kelly_size values are scaled so the total gross heat ≤ max_portfolio_heat.

Fallback
--------
If a symbol has < min_history_days of data, the raw signal magnitude is used
directly (replicates the current system behaviour).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class KellyConfig:
    """Configuration for KellySizer.

    YAML key: ``kelly``
    """
    enabled: bool = True
    fraction: float = 0.25          # fraction of full Kelly to use
    min_history_days: int = 63      # rolling window; fallback below this
    max_position_pct: float = 0.15  # per-symbol cap as fraction of portfolio
    max_portfolio_heat: float = 1.0 # gross exposure cap (100 % of capital)
    ic_window: int = 63             # rolling IC estimation window (days)
    ic_lag: int = 10                # forward-return horizon matching signal IC peak

    @classmethod
    def from_dict(cls, cfg: dict) -> "KellyConfig":
        section = cfg.get("kelly", cfg)
        return cls(
            enabled=bool(section.get("enabled", True)),
            fraction=float(section.get("fraction", 0.25)),
            min_history_days=int(section.get("min_history_days", 63)),
            max_position_pct=float(section.get("max_position_pct", 0.15)),
            max_portfolio_heat=float(section.get("max_portfolio_heat", 1.0)),
            ic_window=int(section.get("ic_window", 63)),
            ic_lag=int(section.get("ic_lag", 10)),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KellySizer:
    """Kelly-fractional position sizer for a cross-sectional signal portfolio.

    Parameters
    ----------
    config : KellyConfig | dict, optional
        Configuration.  If a dict, must have a ``kelly`` top-level key or be
        the kelly section itself.

    Examples
    --------
    >>> sizer = KellySizer()
    >>> weights = sizer.compute_weights(signals, returns_history, as_of_date)
    >>> # weights is a dict[symbol → position weight], summing ≤ max_portfolio_heat
    """

    def __init__(self, config: Optional[KellyConfig | dict] = None) -> None:
        if config is None:
            self.cfg = KellyConfig()
        elif isinstance(config, dict):
            self.cfg = KellyConfig.from_dict(config)
        else:
            self.cfg = config

        logger.info(
            "KellySizer initialised | enabled=%s | fraction=%.2f | "
            "min_history=%dd | max_pos=%.2f | max_heat=%.2f",
            self.cfg.enabled,
            self.cfg.fraction,
            self.cfg.min_history_days,
            self.cfg.max_position_pct,
            self.cfg.max_portfolio_heat,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_weights(
        self,
        signals: Dict[str, float],
        returns_history: Dict[str, pd.Series],
        as_of_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """Compute Kelly-fractional position weights.

        Parameters
        ----------
        signals : dict[str, float]
            Current signal per symbol, values in [-1, +1].
        returns_history : dict[str, pd.Series]
            Daily return series per symbol, datetime-indexed.
            Used to estimate p and b for each symbol.
        as_of_date : pd.Timestamp
            Evaluation date — used to slice history causally.

        Returns
        -------
        dict[str, float]
            Position weights in the same sign as ``signals``.
            The *magnitude* is the Kelly-fractional size.
            Sum of absolute weights ≤ cfg.max_portfolio_heat.
        """
        if not self.cfg.enabled:
            # Disabled: return uniform weights scaled to max_heat / n
            n = max(len(signals), 1)
            uniform = self.cfg.max_portfolio_heat / n
            return {sym: sig * uniform for sym, sig in signals.items()}

        raw_weights: Dict[str, float] = {}

        for symbol, signal in signals.items():
            if abs(signal) < 1e-9:
                raw_weights[symbol] = 0.0
                continue

            hist = returns_history.get(symbol)
            size = self._kelly_size_for_symbol(symbol, signal, hist, as_of_date)
            # Preserve signal direction
            raw_weights[symbol] = np.sign(signal) * size

        # Portfolio-level normalisation
        normalised = self._normalise(raw_weights)

        logger.debug(
            "as_of=%s | weights: %s",
            as_of_date.date(),
            {k: round(v, 4) for k, v in normalised.items()},
        )
        return normalised

    # ------------------------------------------------------------------
    # Per-symbol sizing
    # ------------------------------------------------------------------

    def _kelly_size_for_symbol(
        self,
        symbol: str,
        signal: float,
        hist: Optional[pd.Series],
        as_of_date: pd.Timestamp,
    ) -> float:
        """Return the unsigned Kelly-fractional size for one symbol."""
        cfg = self.cfg

        # Causal slice
        if hist is not None:
            hist = hist[hist.index <= as_of_date]

        if hist is None or len(hist) < cfg.min_history_days:
            # Fallback: use signal magnitude directly
            size = abs(signal) * cfg.max_position_pct
            logger.debug(
                "%s: insufficient history → fallback size=%.4f", symbol, size
            )
            return float(np.clip(size, 0.0, cfg.max_position_pct))

        # --- Estimate p from rolling IC ---
        p = self._estimate_p_from_ic(hist, signal, cfg.ic_window, cfg.ic_lag)

        # --- Estimate b (win/loss ratio) from recent return history ---
        b = self._estimate_payoff_ratio(hist, cfg.ic_window)

        # --- Raw Kelly fraction ---
        kelly_raw = self._kelly_formula(p, b)
        kelly_size = kelly_raw * cfg.fraction

        # Clip to [0, max_position_pct]
        kelly_size = float(np.clip(kelly_size, 0.0, cfg.max_position_pct))

        logger.debug(
            "%s: p=%.3f b=%.3f kelly_raw=%.4f kelly_size=%.4f",
            symbol, p, b, kelly_raw, kelly_size,
        )
        return kelly_size

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kelly_formula(p: float, b: float) -> float:
        """Standard Kelly formula: (p*b - (1-p)) / b.

        Returns 0 if b == 0 or formula yields negative value.
        """
        if b <= 0:
            return 0.0
        raw = (p * b - (1.0 - p)) / b
        return max(raw, 0.0)

    def _estimate_p_from_ic(
        self,
        returns: pd.Series,
        signal: float,
        window: int,
        lag: int,
    ) -> float:
        """Estimate signal accuracy p from the rolling IC.

        Method
        ------
        We use the last ``window`` return observations to compute a proxy IC:
        the rank correlation between lagged returns (as a stand-in for signals
        from ``lag`` days prior) and the next ``lag``-day forward returns.

        When actual historical signal data is unavailable, we approximate using
        the autocorrelation structure of returns to estimate the IC magnitude,
        then convert IC → accuracy p via:
            p = 0.5 + IC / 2   (maps IC ∈ [-1,1] → p ∈ [0,1])

        If computation fails, defaults to p = 0.5 (coin-flip, Kelly → 0).
        """
        try:
            tail = returns.iloc[-window - lag:]
            if len(tail) < window:
                return 0.5

            # Proxy: use lagged return as signal surrogate
            past_rets = tail.iloc[:-lag]
            fwd_rets = tail.iloc[lag:].values

            if len(past_rets) < 10:
                return 0.5

            # Spearman rank correlation (IC proxy)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(past_rets.values, fwd_rets)

            if np.isnan(ic):
                return 0.5

            # Map IC → accuracy probability
            p = 0.5 + float(ic) / 2.0
            p = float(np.clip(p, 0.05, 0.95))
            return p

        except Exception as exc:
            logger.warning("IC estimation failed for signal=%.3f: %s", signal, exc)
            return 0.5

    @staticmethod
    def _estimate_payoff_ratio(returns: pd.Series, window: int) -> float:
        """Estimate b = mean_win / mean_loss from the last ``window`` returns.

        Returns 1.0 as a neutral default if insufficient data.
        """
        try:
            tail = returns.iloc[-window:]
            wins = tail[tail > 0]
            losses = tail[tail < 0].abs()

            mean_win = float(wins.mean()) if len(wins) > 0 else 0.0
            mean_loss = float(losses.mean()) if len(losses) > 0 else 1.0

            if mean_loss < 1e-9:
                return 1.0

            b = mean_win / mean_loss
            return float(np.clip(b, 0.1, 10.0))  # sanity bounds

        except Exception as exc:
            logger.warning("Payoff ratio estimation failed: %s", exc)
            return 1.0

    # ------------------------------------------------------------------
    # Portfolio normalisation
    # ------------------------------------------------------------------

    def _normalise(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """Scale weights so total gross exposure ≤ max_portfolio_heat."""
        total_gross = sum(abs(w) for w in raw_weights.values())

        if total_gross < 1e-9:
            return {sym: 0.0 for sym in raw_weights}

        if total_gross <= self.cfg.max_portfolio_heat:
            return dict(raw_weights)

        scale = self.cfg.max_portfolio_heat / total_gross
        return {sym: w * scale for sym, w in raw_weights.items()}

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"KellySizer(enabled={self.cfg.enabled}, "
            f"fraction={self.cfg.fraction}, "
            f"max_pos={self.cfg.max_position_pct}, "
            f"max_heat={self.cfg.max_portfolio_heat})"
        )
