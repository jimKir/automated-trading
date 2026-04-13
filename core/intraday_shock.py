"""
Intraday Shock Detector
========================
Closes the gap between the 5-minute trading loop and actual intraday events.
Runs on EVERY loop iteration (every 5 minutes in live/paper mode).
In backtest, simulates daily using end-of-day data as a proxy.

Four detection mechanisms:
───────────────────────────
1. VIX INTRADAY SPIKE
   VIX up >15% intraday → SHOCK (0.25×)
   VIX up >10% intraday → CAUTION (0.60×)

2. PORTFOLIO EQUITY DROP (intraday)
   Portfolio down >3% from morning open → SHOCK
   Portfolio down >2% → CAUTION

3. VOLUME SHOCK (real-time, this is where volume is genuinely useful)
   A) PANIC VOLUME: today's volume > 2× 20-day average AND price down
      → CAUTION (0.60×). Forced liquidation in progress.
   B) CLIMACTIC VOLUME: today's volume > 3× average AND price down >1%
      → SHOCK (0.25×). Exhaustion selling / institutional capitulation.

   Why volume works HERE but not in weekly signals:
   - Volume spikes are same-day events — detected and acted on immediately
   - No weekly decay, no aggregation that destroys the signal
   - Academic basis: Jones, Kaul & Lipson (1994) — high volume predicts
     continuation of the current price move on the same day

4. RECOVERY
   5-day gradual ramp: 30% → 50% → 75% → 90% → 100%
   RECOVERY is immune to re-triggering until ramp completes.

Scale factors:
   CLEAR:    1.00×
   CAUTION:  0.60×
   SHOCK:    0.25×
   RECOVERY: 0.30 → 0.50 → 0.75 → 0.90 → 1.00 (5-day ramp)

In LIVE: volume from broker feed, checked every 5 minutes.
In BACKTEST: day-over-day SPY volume used as portfolio-level proxy.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.logger import get_logger

if TYPE_CHECKING:
    from datetime import date, datetime

log = get_logger("IntradayShock")

# ── Thresholds (economic logic, not optimised) ────────────────────────────────
VIX_SPIKE_CAUTION = 0.10  # VIX up 10% intraday
VIX_SPIKE_SHOCK = 0.15  # VIX up 15% intraday
EQUITY_DROP_CAUTION = 0.02  # portfolio down 2% from open
EQUITY_DROP_SHOCK = 0.03  # portfolio down 3% from open
VIX_RECOVERY_LEVEL = 1.05
EQUITY_RECOVERY_LVL = -0.015

# ── Volume shock thresholds ───────────────────────────────────────────────────
VOL_PANIC_MULTIPLE = 2.0  # volume > 2× 20d avg AND price down = panic
VOL_CLIMACTIC_MULTIPLE = 3.0  # volume > 3× 20d avg AND price down >1% = climax
VOL_PRICE_DOWN_MIN = 0.01  # price must be down ≥1% for climactic trigger
VOL_LOOKBACK = 20  # rolling average window for volume baseline

# ── Scale factors ─────────────────────────────────────────────────────────────
SCALE_CLEAR = 1.00
SCALE_CAUTION = 0.70  # v15: raised from 0.60 (less punishing)
SCALE_SHOCK = 0.35  # v15: raised from 0.25 (maintain some participation)
SCALE_RECOVERY = [0.50, 0.80, 1.00]  # v15: 3-day ramp (was 5-day)


class ShockState(Enum):
    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    SHOCK = "SHOCK"
    RECOVERY = "RECOVERY"


class IntradayShockDetector:
    def __init__(self, config: dict):
        isd_cfg = config.get("intraday_shock", {})
        self.enabled = isd_cfg.get("enabled", True)
        self.vix_spike_caution = isd_cfg.get("vix_spike_caution", VIX_SPIKE_CAUTION)
        self.vix_spike_shock = isd_cfg.get("vix_spike_shock", VIX_SPIKE_SHOCK)
        self.equity_drop_caution = isd_cfg.get("equity_drop_caution", EQUITY_DROP_CAUTION)
        self.equity_drop_shock = isd_cfg.get("equity_drop_shock", EQUITY_DROP_SHOCK)
        self.vol_panic_mult = isd_cfg.get("vol_panic_multiple", VOL_PANIC_MULTIPLE)
        self.vol_climactic_mult = isd_cfg.get("vol_climactic_multiple", VOL_CLIMACTIC_MULTIPLE)
        self.vol_enabled = isd_cfg.get("volume_shock", True)

        # Daily snapshots
        self._morning_vix: float | None = None
        self._morning_equity: float | None = None
        self._today: date | None = None

        # Rolling volume history for baseline (live mode)
        self._volume_history: list[float] = []

        # State machine
        self._state: ShockState = ShockState.CLEAR
        self._recovery_day: int = 0
        self._shock_start_date: date | None = None
        self._state_log: list = []

    # ─────────────────────────────────────────────────────────────────────────
    # Live / paper mode
    # ─────────────────────────────────────────────────────────────────────────

    def reset_day(
        self,
        vix_open: float,
        equity_open: float,
        today: date,
        prev_volume: float | None = None,
    ) -> None:
        if not self.enabled:
            return

        # Advance recovery ramp
        if self._state == ShockState.RECOVERY:
            self._recovery_day += 1
            if self._recovery_day >= len(SCALE_RECOVERY):
                self._state = ShockState.CLEAR
                self._recovery_day = 0
                log.info(f"[{today}] IntradayShock: RECOVERY complete → CLEAR")

        # Update volume baseline
        if prev_volume is not None and prev_volume > 0:
            self._volume_history.append(prev_volume)
            if len(self._volume_history) > VOL_LOOKBACK:
                self._volume_history = self._volume_history[-VOL_LOOKBACK:]

        self._morning_vix = vix_open
        self._morning_equity = equity_open
        self._today = today

    def _check_volume_shock(
        self,
        current_volume: float | None,
        price_chg: float,
    ) -> tuple[ShockState, str]:
        """Volume-based sub-detector. Returns (state, reason)."""
        if not self.vol_enabled or current_volume is None or current_volume <= 0:
            return ShockState.CLEAR, ""
        if len(self._volume_history) < 5:
            return ShockState.CLEAR, ""
        vol_avg = float(np.mean(self._volume_history))
        if vol_avg <= 0:
            return ShockState.CLEAR, ""

        vol_ratio = current_volume / vol_avg

        if vol_ratio >= self.vol_climactic_mult and price_chg <= -VOL_PRICE_DOWN_MIN:
            reason = f"CLIMACTIC VOLUME: {vol_ratio:.1f}× avg, price {price_chg:+.1%}"
            log.warning(f"[{self._today}] ISD: {reason}")
            return ShockState.SHOCK, reason

        if vol_ratio >= self.vol_panic_mult and price_chg < 0:
            reason = f"PANIC VOLUME: {vol_ratio:.1f}× avg, price {price_chg:+.1%}"
            log.info(f"[{self._today}] ISD: {reason}")
            return ShockState.CAUTION, reason

        return ShockState.CLEAR, ""

    def check(
        self,
        current_vix: float,
        current_equity: float,
        now: datetime | None = None,
        current_volume: float | None = None,
        prev_close: float | None = None,
        current_close: float | None = None,
    ) -> tuple[float, ShockState, str]:
        """
        Check all four shock conditions. Called every 5 minutes in live mode.
        Returns (scale_factor, state, reason).
        """
        if not self.enabled:
            return 1.0, ShockState.CLEAR, "disabled"
        if self._morning_vix is None or self._morning_equity is None:
            return 1.0, ShockState.CLEAR, "no morning snapshot"

        reason = ""

        vix_chg = (
            (current_vix - self._morning_vix) / self._morning_vix if self._morning_vix > 0 else 0
        )
        equity_chg = (
            (current_equity - self._morning_equity) / self._morning_equity
            if self._morning_equity > 0
            else 0
        )

        # Volume signal
        price_chg = 0.0
        if prev_close and current_close and prev_close > 0:
            price_chg = (current_close - prev_close) / prev_close
        vol_state, vol_reason = self._check_volume_shock(current_volume, price_chg)

        prev_state = self._state

        if self._state in (ShockState.CLEAR, ShockState.CAUTION):
            vix_shock = vix_chg >= self.vix_spike_shock
            equity_shock = equity_chg <= -self.equity_drop_shock
            vol_shock = vol_state == ShockState.SHOCK

            if vix_shock or equity_shock or vol_shock:
                self._state = ShockState.SHOCK
                self._recovery_day = 0
                if self._shock_start_date != self._today:
                    self._shock_start_date = self._today
                    if vol_shock and not vix_shock and not equity_shock:
                        reason = vol_reason
                    elif vix_shock:
                        reason = f"SHOCK: VIX +{vix_chg:.1%} intraday"
                    else:
                        reason = f"SHOCK: portfolio {equity_chg:.1%} intraday"
                    log.warning(f"[{self._today}] IntradayShock: {reason}")

            elif (
                vix_chg >= self.vix_spike_caution
                or equity_chg <= -self.equity_drop_caution
                or vol_state == ShockState.CAUTION
            ):
                self._state = ShockState.CAUTION
                if vol_state == ShockState.CAUTION and vix_chg < self.vix_spike_caution:
                    reason = vol_reason
                elif vix_chg >= self.vix_spike_caution:
                    reason = f"CAUTION: VIX +{vix_chg:.1%}"
                else:
                    reason = f"CAUTION: portfolio {equity_chg:.1%}"
            else:
                self._state = ShockState.CLEAR

        elif self._state == ShockState.SHOCK:
            if vix_chg <= 0.05:
                self._state = ShockState.RECOVERY
                self._recovery_day = 0
                reason = "Recovery starting — 5-day gradual ramp"
                log.info(f"[{self._today}] IntradayShock: SHOCK → RECOVERY")
            else:
                reason = f"SHOCK active: VIX +{vix_chg:.1%}"

        elif self._state == ShockState.RECOVERY:
            reason = f"RECOVERY day {self._recovery_day + 1}/{len(SCALE_RECOVERY)}"

        if self._state != prev_state:
            vol_ratio = None
            if current_volume and self._volume_history:
                avg = float(np.mean(self._volume_history))
                vol_ratio = round(current_volume / avg, 3) if avg > 0 else None
            self._state_log.append(
                {
                    "date": str(self._today),
                    "from_state": prev_state.value,
                    "to_state": self._state.value,
                    "vix_chg": round(vix_chg, 4),
                    "equity_chg": round(equity_chg, 4),
                    "vol_ratio": vol_ratio,
                    "reason": reason,
                }
            )

        return self._get_scale(), self._state, reason

    def _get_scale(self) -> float:
        if self._state == ShockState.CLEAR:
            return SCALE_CLEAR
        if self._state == ShockState.CAUTION:
            return SCALE_CAUTION
        if self._state == ShockState.SHOCK:
            return SCALE_SHOCK
        if self._state == ShockState.RECOVERY:
            return SCALE_RECOVERY[min(self._recovery_day, len(SCALE_RECOVERY) - 1)]
        return 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Backtest mode — daily simulation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_backtest_scales(
        self,
        vix_series: pd.Series,
        equity_series: pd.Series,
        spy_volume: pd.Series | None = None,
    ) -> pd.Series:
        """
        Simulate all four shock mechanisms for a full backtest period.
        Uses day-over-day changes as proxy for intraday moves.

        spy_volume: daily SPY volume series aligned to equity_series dates.
        """
        if not self.enabled:
            return pd.Series(1.0, index=equity_series.index)

        scales = pd.Series(1.0, index=equity_series.index, dtype=float)

        vix_aligned = vix_series.reindex(equity_series.index).ffill().bfill()
        vol_aligned = (
            spy_volume.reindex(equity_series.index).ffill() if spy_volume is not None else None
        )

        state = ShockState.CLEAR
        recovery_day = 0
        vol_history: list[float] = []

        for i in range(1, len(equity_series)):
            vix_prev = (
                float(vix_aligned.iloc[i - 1]) if not pd.isna(vix_aligned.iloc[i - 1]) else 15.0
            )
            vix_curr = float(vix_aligned.iloc[i]) if not pd.isna(vix_aligned.iloc[i]) else 15.0
            eq_prev = float(equity_series.iloc[i - 1])
            eq_curr = float(equity_series.iloc[i])

            vix_chg = (vix_curr - vix_prev) / vix_prev if vix_prev > 0 else 0
            equity_chg = (eq_curr - eq_prev) / eq_prev if eq_prev > 0 else 0
            price_chg = equity_chg  # use equity as price proxy

            # Build volume ratio
            vol_ratio = 0.0
            vol_state_bt = ShockState.CLEAR
            if vol_aligned is not None:
                v_curr = float(vol_aligned.iloc[i]) if not pd.isna(vol_aligned.iloc[i]) else 0
                v_prev = (
                    float(vol_aligned.iloc[i - 1]) if not pd.isna(vol_aligned.iloc[i - 1]) else 0
                )
                if v_prev > 0:
                    vol_history.append(v_prev)
                    if len(vol_history) > VOL_LOOKBACK:
                        vol_history = vol_history[-VOL_LOOKBACK:]
                if len(vol_history) >= 5 and v_curr > 0:
                    vol_avg = float(np.mean(vol_history))
                    if vol_avg > 0:
                        vol_ratio = v_curr / vol_avg
                        if (
                            vol_ratio >= self.vol_climactic_mult
                            and price_chg <= -VOL_PRICE_DOWN_MIN
                        ):
                            vol_state_bt = ShockState.SHOCK
                        elif vol_ratio >= self.vol_panic_mult and price_chg < 0:
                            vol_state_bt = ShockState.CAUTION

            # Advance recovery
            if state == ShockState.RECOVERY:
                recovery_day += 1
                if recovery_day >= len(SCALE_RECOVERY):
                    state = ShockState.CLEAR
                    recovery_day = 0

            # Transitions
            if state in (ShockState.CLEAR, ShockState.CAUTION):
                if (
                    vix_chg >= self.vix_spike_shock
                    or equity_chg <= -self.equity_drop_shock
                    or vol_state_bt == ShockState.SHOCK
                ):
                    state = ShockState.SHOCK
                    recovery_day = 0
                elif (
                    vix_chg >= self.vix_spike_caution
                    or equity_chg <= -self.equity_drop_caution
                    or vol_state_bt == ShockState.CAUTION
                ):
                    state = ShockState.CAUTION
                else:
                    state = ShockState.CLEAR

            elif state == ShockState.SHOCK and vix_chg <= 0.05:
                state = ShockState.RECOVERY
                recovery_day = 0

            # Assign scale
            if state == ShockState.CLEAR:
                scales.iloc[i] = SCALE_CLEAR
            elif state == ShockState.CAUTION:
                scales.iloc[i] = SCALE_CAUTION
            elif state == ShockState.SHOCK:
                scales.iloc[i] = SCALE_SHOCK
            elif state == ShockState.RECOVERY:
                scales.iloc[i] = SCALE_RECOVERY[min(recovery_day, len(SCALE_RECOVERY) - 1)]

        log.info(
            f"IntradayShock (with volume): "
            f"CLEAR={int((scales == SCALE_CLEAR).sum())} | "
            f"CAUTION={int((scales == SCALE_CAUTION).sum())} | "
            f"SHOCK={int((scales == SCALE_SHOCK).sum())} | "
            f"RECOVERY={(scales.isin(SCALE_RECOVERY[:-1])).sum()}"
        )
        return scales

    # ─────────────────────────────────────────────────────────────────────────

    def get_state_log(self) -> pd.DataFrame:
        if not self._state_log:
            return pd.DataFrame()
        return pd.DataFrame(self._state_log)

    @property
    def current_state(self) -> ShockState:
        return self._state

    @property
    def current_scale(self) -> float:
        return self._get_scale()
