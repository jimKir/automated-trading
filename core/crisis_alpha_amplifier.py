"""
crisis_alpha_amplifier.py
=========================
VIX-regime-aware position scale amplifier.

Diagnostic finding: composite Sharpe = 2.03 in HIGH-VOL (VIX-spike) weeks
vs 0.128 in LOW-VOL (calm) weeks.  The original vol-targeting framework
*reduces* exposure when volatility rises, which is backwards for this signal
structure.  This module INCREASES position sizes in the regimes where the
signal-to-noise is highest and reduces them in suppressed-vol environments.

Regime map
----------
CRISIS    VIX > 30 AND persistently rising   → scale 1.60
ELEVATED  VIX 20-30                          → scale 1.25
NORMAL    VIX 15-20                          → scale 1.00
SUPPRESSED VIX < 15                          → scale 0.80

"Rising" = 5d MA > 10d MA > 20d MA (momentum structure, not noise)
Anti-whipsaw = regime must persist for min_days_in_regime consecutive
               days before the amplifier state changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CrisisAlphaConfig:
    """Runtime configuration for CrisisAlphaAmplifier.

    All fields have sensible defaults derived from the diagnostic findings.
    Override via dict / yaml loader before constructing the amplifier.
    """

    enabled: bool = True

    # Scale factors per regime
    crisis_scale: float = 1.60
    elevated_scale: float = 1.25
    normal_scale: float = 1.00
    suppressed_scale: float = 0.80

    # VIX level thresholds
    vix_crisis_threshold: float = 30.0
    vix_elevated_threshold: float = 20.0
    vix_normal_threshold: float = 15.0  # below this → SUPPRESSED

    # Anti-whipsaw: require this many consecutive days before flipping state
    min_days_in_regime: int = 3

    # Hard bounds on the returned scale factor
    scale_min: float = 0.5
    scale_max: float = 2.0

    @classmethod
    def from_dict(cls, cfg: dict) -> CrisisAlphaConfig:
        """Build from a plain dict (e.g. parsed from YAML ``crisis_alpha`` key)."""
        section = cfg.get("crisis_alpha", cfg)
        return cls(
            enabled=section.get("enabled", True),
            crisis_scale=float(section.get("crisis_scale", 1.60)),
            elevated_scale=float(section.get("elevated_scale", 1.25)),
            normal_scale=float(section.get("normal_scale", 1.00)),
            suppressed_scale=float(section.get("suppressed_scale", 0.80)),
            vix_crisis_threshold=float(section.get("vix_crisis_threshold", 30.0)),
            vix_elevated_threshold=float(section.get("vix_elevated_threshold", 20.0)),
            vix_normal_threshold=float(section.get("vix_normal_threshold", 15.0)),
            min_days_in_regime=int(section.get("min_days_in_regime", 3)),
            scale_min=float(section.get("scale_min", 0.5)),
            scale_max=float(section.get("scale_max", 2.0)),
        )


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------


class VIXRegime(StrEnum):
    CRISIS = "CRISIS"
    ELEVATED = "ELEVATED"
    NORMAL = "NORMAL"
    SUPPRESSED = "SUPPRESSED"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CrisisAlphaAmplifier:
    """Computes a position-size scale factor based on the VIX regime.

    This class is **strictly causal**: ``get_scale`` never looks beyond
    ``as_of_date``.

    Parameters
    ----------
    config : CrisisAlphaConfig | dict, optional
        Configuration object or raw dict.  Defaults to diagnostic-tuned values.

    Examples
    --------
    >>> amp = CrisisAlphaAmplifier()
    >>> scale = amp.get_scale(vix_series, spy_returns, as_of_date=pd.Timestamp("2025-03-01"))
    >>> position_size *= scale
    """

    def __init__(self, config: CrisisAlphaConfig | dict | None = None) -> None:
        if config is None:
            self.cfg = CrisisAlphaConfig()
        elif isinstance(config, dict):
            self.cfg = CrisisAlphaConfig.from_dict(config)
        else:
            self.cfg = config

        # Persistent state — tracks the *confirmed* (anti-whipsawed) regime
        self._confirmed_regime: VIXRegime = VIXRegime.NORMAL
        self._candidate_regime: VIXRegime = VIXRegime.NORMAL
        self._candidate_streak: int = 0

        logger.info(
            "CrisisAlphaAmplifier initialised | enabled=%s | thresholds: "
            "crisis=%.0f, elevated=%.0f, normal_floor=%.0f",
            self.cfg.enabled,
            self.cfg.vix_crisis_threshold,
            self.cfg.vix_elevated_threshold,
            self.cfg.vix_normal_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scale(
        self,
        vix_series: pd.Series,
        spy_returns: pd.Series,
        as_of_date: pd.Timestamp,
    ) -> float:
        """Return the position-size multiplier for ``as_of_date``.

        Parameters
        ----------
        vix_series : pd.Series
            Daily VIX closing levels, datetime-indexed.  Any frequency works;
            the method slices up to ``as_of_date`` internally.
        spy_returns : pd.Series
            Daily SPY total returns (used as a market-stress cross-check).
            Must share the same datetime index as ``vix_series``.
        as_of_date : pd.Timestamp
            The evaluation date.  Only data **on or before** this date is used.

        Returns
        -------
        float
            Scale factor clamped to [cfg.scale_min, cfg.scale_max].
            Returns 1.0 if the amplifier is disabled.
        """
        if not self.cfg.enabled:
            return 1.0

        # --- 1. Slice to causal window -----------------------------------------
        vix = vix_series[vix_series.index <= as_of_date].copy()
        spy_returns[spy_returns.index <= as_of_date].copy()

        if vix.empty:
            logger.warning("VIX series is empty up to %s; returning neutral scale", as_of_date)
            return 1.0

        current_vix = float(vix.iloc[-1])

        # --- 2. Detect VIX trend (5d MA > 10d MA > 20d MA) -------------------
        vix_rising = self._is_vix_rising(vix)

        # --- 3. Classify raw regime for today ---------------------------------
        raw_regime = self._classify_regime(current_vix, vix_rising)

        # --- 4. Apply anti-whipsaw filter -------------------------------------
        confirmed = self._update_regime_state(raw_regime)

        # --- 5. Map regime → scale --------------------------------------------
        scale = self._regime_to_scale(confirmed)

        # --- 6. Clamp and return ----------------------------------------------
        scale = float(np.clip(scale, self.cfg.scale_min, self.cfg.scale_max))

        logger.debug(
            "as_of=%s  vix=%.1f  rising=%s  raw=%s  confirmed=%s  scale=%.3f",
            as_of_date.date(),
            current_vix,
            vix_rising,
            raw_regime.value,
            confirmed.value,
            scale,
        )
        return scale

    def get_regime_series(
        self,
        vix_series: pd.Series,
        spy_returns: pd.Series,
    ) -> pd.DataFrame:
        """Batch-compute regime and scale for every date in ``vix_series``.

        Useful for backtesting / diagnostics.  Internally resets state so it
        can be called independently of ``get_scale`` call history.

        Returns
        -------
        pd.DataFrame with columns: [regime, scale]
        """
        # Reset internal state for a clean walk-forward
        self._confirmed_regime = VIXRegime.NORMAL
        self._candidate_regime = VIXRegime.NORMAL
        self._candidate_streak = 0

        records = []
        for date in vix_series.index:
            scale = self.get_scale(vix_series, spy_returns, as_of_date=date)
            records.append(
                {
                    "date": date,
                    "regime": self._confirmed_regime.value,
                    "scale": scale,
                }
            )
        df = pd.DataFrame(records).set_index("date")
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_vix_rising(self, vix: pd.Series) -> bool:
        """Return True iff 5d MA > 10d MA > 20d MA on the most recent data.

        Requires at least 20 observations; returns False if insufficient data.
        """
        if len(vix) < 20:
            return False

        ma5 = vix.rolling(5).mean().iloc[-1]
        ma10 = vix.rolling(10).mean().iloc[-1]
        ma20 = vix.rolling(20).mean().iloc[-1]

        return bool(ma5 > ma10 > ma20)

    def _classify_regime(self, vix_level: float, vix_rising: bool) -> VIXRegime:
        """Map a VIX level + trend to a VIXRegime (un-filtered / raw)."""
        cfg = self.cfg
        if vix_level > cfg.vix_crisis_threshold and vix_rising:
            return VIXRegime.CRISIS
        if vix_level > cfg.vix_elevated_threshold:
            return VIXRegime.ELEVATED
        if vix_level > cfg.vix_normal_threshold:
            return VIXRegime.NORMAL
        return VIXRegime.SUPPRESSED

    def _update_regime_state(self, raw_regime: VIXRegime) -> VIXRegime:
        """Anti-whipsaw state machine.

        A regime flip is only confirmed after ``min_days_in_regime`` consecutive
        days in the candidate regime.  Until then, the *previous* confirmed
        regime is returned.
        """
        if raw_regime == self._candidate_regime:
            self._candidate_streak += 1
        else:
            # Potential new regime — restart streak counter
            self._candidate_regime = raw_regime
            self._candidate_streak = 1

        if self._candidate_streak >= self.cfg.min_days_in_regime:
            if raw_regime != self._confirmed_regime:
                logger.info(
                    "Regime change confirmed: %s → %s (streak=%d)",
                    self._confirmed_regime.value,
                    raw_regime.value,
                    self._candidate_streak,
                )
            self._confirmed_regime = raw_regime

        return self._confirmed_regime

    def _regime_to_scale(self, regime: VIXRegime) -> float:
        """Map confirmed regime to raw scale factor (before clamping)."""
        mapping = {
            VIXRegime.CRISIS: self.cfg.crisis_scale,
            VIXRegime.ELEVATED: self.cfg.elevated_scale,
            VIXRegime.NORMAL: self.cfg.normal_scale,
            VIXRegime.SUPPRESSED: self.cfg.suppressed_scale,
        }
        return mapping[regime]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CrisisAlphaAmplifier("
            f"enabled={self.cfg.enabled}, "
            f"confirmed_regime={self._confirmed_regime.value}, "
            f"scales=[crisis={self.cfg.crisis_scale}, "
            f"elevated={self.cfg.elevated_scale}, "
            f"normal={self.cfg.normal_scale}, "
            f"suppressed={self.cfg.suppressed_scale}])"
        )
