"""
engine_rebalance_patch.py
=========================
Drop-in replacement for ``BacktestEngine._rebalance_schedule``.

Diagnostic motivation
---------------------
Signal IC peaks at 10 TRADING DAYS (not 5).  The current weekly schedule
misses this peak.  A biweekly (every-10-trading-day) option is added along
with two intelligent overrides:

1. **Signal-change threshold**: if today's portfolio-level signal has shifted
   by less than ``signal_change_threshold`` (default 0.15) vs the last
   rebalance, skip the trade even if it falls on a scheduled date.
   Reduces unnecessary turnover in calm, low-IC periods.

2. **Forced VIX-spike rebalance**: if VIX jumps >20 % in a single day,
   force a rebalance regardless of the schedule.
   Ensures the system reacts quickly in crisis regimes where Sharpe = 2.03.

Usage (inside BacktestEngine)
------------------------------
Replace the existing method body with this function, or call it as a
standalone helper:

    from engine_rebalance_patch import build_rebalance_schedule

    rebal_dates = build_rebalance_schedule(
        dates=all_dates,
        rebalance_freq=self.rebalance_freq,
        signal_series=self.signal_series,        # pd.Series, date-indexed
        vix_series=self.vix_series,              # pd.Series, date-indexed
        signal_change_threshold=self.cfg.get("signal_change_threshold", 0.15),
        vix_spike_threshold=self.cfg.get("vix_spike_threshold", 0.20),
    )
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API (standalone helper — mirrors BacktestEngine method signature)
# ---------------------------------------------------------------------------

def build_rebalance_schedule(
    dates: list,
    rebalance_freq: str = "weekly",
    signal_series: pd.Series | None = None,
    vix_series: pd.Series | None = None,
    signal_change_threshold: float = 0.15,
    vix_spike_threshold: float = 0.20,
    biweekly_n_trading_days: int = 10,
    # ── Adaptive scheduler ────────────────────────────────────────────
    # rebalance_freq="adaptive" uses choppy_score_series to switch:
    #   score < adaptive_weekly_threshold  → biweekly (GREEN regime)
    #   score >= adaptive_weekly_threshold → weekly   (YELLOW/ORANGE/RED)
    choppy_score_series: pd.Series | None = None,
    adaptive_weekly_threshold: float = 0.17,  # YELLOW onset in ChoppyDetector
    adaptive_smoothing: int = 3,              # days of score EMA before switching
) -> set:
    """Build the set of dates on which the portfolio should rebalance.

    This is a **drop-in replacement** for the original
    ``BacktestEngine._rebalance_schedule`` method.  It extends it with:

    * ``"biweekly"`` frequency — every ``biweekly_n_trading_days`` trading
      days (default 10), capturing the signal IC peak identified in the
      diagnostic.
    * Signal-change skip — if the composite signal has barely moved since the
      last rebalance, skip the scheduled date.
    * Forced VIX-spike rebalance — large intraday VIX jumps trigger
      an immediate out-of-schedule rebalance.

    Parameters
    ----------
    dates : list of pd.Timestamp
        Full chronological list of trading dates in the backtest.
    rebalance_freq : str
        One of ``"daily"``, ``"weekly"``, ``"biweekly"``, ``"monthly"``.
    signal_series : pd.Series, optional
        Portfolio-level composite signal (scalar per date).  Required for the
        signal-change threshold logic.  Skipped if None.
    vix_series : pd.Series, optional
        Daily VIX closing levels.  Required for forced-rebalance logic.
        Skipped if None.
    signal_change_threshold : float
        Minimum absolute change in signal required to trigger a scheduled
        rebalance.  0.15 means 15 % of the [-1, +1] scale.
    vix_spike_threshold : float
        VIX single-day percentage jump (e.g. 0.20 = 20 %) that forces an
        out-of-schedule rebalance.
    biweekly_n_trading_days : int
        Number of trading days between biweekly rebalances (default 10).

    Returns
    -------
    set of pd.Timestamp
        Dates on which a rebalance should occur.
    """
    if not dates:
        return set()

    s = pd.Series(dates, index=dates)

    # ------------------------------------------------------------------
    # 1. Base schedule from frequency
    # ------------------------------------------------------------------
    if rebalance_freq == "adaptive" and choppy_score_series is not None:
        base_schedule = _adaptive_schedule(
            dates,
            choppy_score_series,
            adaptive_weekly_threshold,
            adaptive_smoothing,
            biweekly_n_trading_days,
        )
    else:
        base_schedule = _base_schedule(s, rebalance_freq, biweekly_n_trading_days)

    logger.info(
        "Base schedule: freq='%s', %d rebalance dates from %s to %s",
        rebalance_freq,
        len(base_schedule),
        min(dates).date(),
        max(dates).date(),
    )

    # ------------------------------------------------------------------
    # 2. Signal-change filter — drop scheduled dates where signal barely moved
    # ------------------------------------------------------------------
    if signal_series is not None and not signal_series.empty:
        base_schedule = _apply_signal_filter(
            base_schedule,
            signal_series,
            threshold=signal_change_threshold,
        )
        logger.info(
            "After signal-change filter (threshold=%.2f): %d rebalance dates remain",
            signal_change_threshold,
            len(base_schedule),
        )

    # ------------------------------------------------------------------
    # 3. Forced VIX-spike rebalances — add any date with a large VIX jump
    # ------------------------------------------------------------------
    if vix_series is not None and not vix_series.empty:
        forced = _vix_forced_dates(
            dates,
            vix_series,
            spike_threshold=vix_spike_threshold,
        )
        if forced:
            logger.info(
                "VIX spike forced rebalance on %d additional date(s): %s",
                len(forced),
                sorted(d.date() for d in forced),
            )
        base_schedule |= forced

    return base_schedule


# ---------------------------------------------------------------------------
# BacktestEngine method version
# (paste directly into BacktestEngine class body)
# ---------------------------------------------------------------------------

def _rebalance_schedule(self, dates: list) -> set:
    """Updated BacktestEngine._rebalance_schedule — paste to replace original.

    Adds:
    * ``biweekly``  — every 10 trading days (matches IC peak at day 10)
    * Signal-change skip  — avoids churn when signal barely moves
    * Forced VIX-spike rebalance — crisis reactivity (Sharpe 2.03 in VIX weeks)

    Config keys consumed from ``self`` (all optional with defaults):
        rebalance_freq                  str   "weekly"
        signal_change_threshold         float 0.15
        vix_spike_threshold             float 0.20
        biweekly_n_trading_days         int   10
    """
    # Pull optional config; guard for engines that don't expose these
    freq = getattr(self, "rebalance_freq", "weekly")
    sig_threshold = float(
        getattr(self, "signal_change_threshold", 0.15)
    )
    vix_thresh = float(getattr(self, "vix_spike_threshold", 0.20))
    biweekly_n = int(getattr(self, "biweekly_n_trading_days", 10))

    signal_series: Optional[pd.Series] = getattr(self, "signal_series", None)
    vix_series: Optional[pd.Series] = getattr(self, "vix_series", None)

    return build_rebalance_schedule(
        dates=dates,
        rebalance_freq=freq,
        signal_series=signal_series,
        vix_series=vix_series,
        signal_change_threshold=sig_threshold,
        vix_spike_threshold=vix_thresh,
        biweekly_n_trading_days=biweekly_n,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _base_schedule(
    s: pd.Series,
    freq: str,
    biweekly_n: int,
) -> set:
    """Return base rebalance dates from frequency string."""
    freq_lower = freq.strip().lower()

    if freq_lower == "daily":
        return set(s.index)

    elif freq_lower == "weekly":
        # Last trading day of each calendar week (Friday anchor)
        return set(s.resample("W-FRI").last().dropna())

    elif freq_lower == "biweekly":
        # Every ``biweekly_n`` *trading* days, starting from the first date
        sorted_dates = sorted(s.index.tolist())
        schedule = set()
        for i in range(0, len(sorted_dates), biweekly_n):
            schedule.add(sorted_dates[i])
        return schedule

    elif freq_lower == "monthly":
        # Last business day of each calendar month
        return set(s.resample("BME").last().dropna())

    else:
        logger.warning(
            "Unknown rebalance_freq '%s'; defaulting to all dates.", freq
        )
        return set(s.index)


def _apply_signal_filter(
    schedule: set,
    signal_series: pd.Series,
    threshold: float,
) -> set:
    """Remove scheduled rebalances where the signal change is below threshold.

    Parameters
    ----------
    schedule : set of timestamps
        Candidate rebalance dates.
    signal_series : pd.Series
        Composite signal, date-indexed.
    threshold : float
        Minimum absolute signal change required to keep the rebalance.

    Returns
    -------
    set of timestamps — filtered schedule (always retains the first date).
    """
    sorted_sched = sorted(schedule)
    if not sorted_sched:
        return schedule

    filtered = {sorted_sched[0]}   # always keep the first rebalance
    last_signal = float(
        signal_series.reindex(method="ffill").loc[
            signal_series.index[signal_series.index <= sorted_sched[0]][-1]
        ]
        if any(signal_series.index <= sorted_sched[0])
        else 0.0
    )

    for date in sorted_sched[1:]:
        # Find most recent signal value on or before this date
        available = signal_series[signal_series.index <= date]
        if available.empty:
            filtered.add(date)
            continue

        current_signal = float(available.iloc[-1])
        change = abs(current_signal - last_signal)

        if change >= threshold:
            filtered.add(date)
            last_signal = current_signal
        else:
            logger.debug(
                "Skipping scheduled rebalance on %s: signal_change=%.4f < threshold=%.4f",
                date.date(),
                change,
                threshold,
            )

    return filtered


def _adaptive_schedule(
    dates: list,
    choppy_score: pd.Series,
    weekly_threshold: float,
    smoothing: int,
    biweekly_n: int,
) -> set:
    """Adaptive rebalance schedule driven by ChoppyRegimeDetector score.

    Logic (per trading day):
      - Smooth choppy_score with ``smoothing``-day EMA (prevents single-day
        whipsawing between regimes).
      - If smoothed score >= weekly_threshold (YELLOW/ORANGE/RED):
          schedule every Friday  (weekly — react fast in choppy/crisis regime)
      - If smoothed score < weekly_threshold (GREEN):
          schedule every ``biweekly_n`` trading days (biweekly — save turnover
          cost in calm trending regime)

    This directly targets the three failure folds:
      COVID 2020   — score spiked to ORANGE in Feb, switched to weekly.
      GFC 2007-08  — score elevated through bear cascade, stayed weekly.
      Bull 2013-19 — score mostly GREEN, biweekly throughout (no regression).

    Parameters
    ----------
    dates           : trading dates in the period
    choppy_score    : ChoppyRegimeDetector score series (full history, causal)
    weekly_threshold: score above which we switch to weekly (default: 0.17,
                      matching the YELLOW onset in CHOPPY_SCALE_THRESHOLDS)
    smoothing       : EMA span to avoid single-day regime flips
    biweekly_n      : trading days between biweekly rebalances

    Returns
    -------
    set of pd.Timestamp — rebalance dates for the period
    """
    sorted_dates = sorted(dates)
    if not sorted_dates:
        return set()

    # Smooth the score to avoid whipsawing (3-day EMA default)
    score_aligned = (
        choppy_score
        .reindex(sorted_dates, method="ffill")
        .fillna(0.0)
    )
    score_smooth = score_aligned.ewm(span=smoothing, adjust=False).mean()

    schedule: set = set()
    last_biweekly_idx = 0   # tracks position in biweekly cadence
    in_weekly_mode = score_smooth.iloc[0] >= weekly_threshold

    for i, date in enumerate(sorted_dates):
        score_today = float(score_smooth.iloc[i])
        was_weekly  = in_weekly_mode
        in_weekly_mode = score_today >= weekly_threshold

        if in_weekly_mode:
            # Weekly mode: rebalance on Fridays (same as _base_schedule)
            if date.weekday() == 4:   # 4 = Friday
                schedule.add(date)
                last_biweekly_idx = i  # reset biweekly counter on switch-back
        else:
            # Biweekly mode: rebalance every biweekly_n trading days
            # On first switch back to GREEN, start fresh biweekly counter
            if was_weekly and not in_weekly_mode:
                last_biweekly_idx = i  # reset counter at GREEN re-entry
            if (i - last_biweekly_idx) % biweekly_n == 0:
                schedule.add(date)

    # Always include the first date as initialisation rebalance
    if sorted_dates:
        schedule.add(sorted_dates[0])

    logger.debug(
        "Adaptive schedule: %d rebalance dates | weekly_days=%d, biweekly_days=%d",
        len(schedule),
        sum(1 for d in schedule
            if float(score_smooth.reindex([d], method="ffill").iloc[0]) >= weekly_threshold),
        sum(1 for d in schedule
            if float(score_smooth.reindex([d], method="ffill").iloc[0]) < weekly_threshold),
    )
    return schedule


def _vix_forced_dates(
    all_dates: list,
    vix_series: pd.Series,
    spike_threshold: float,
) -> set:
    """Return dates where VIX spiked > spike_threshold in a single session.

    Only dates that are in ``all_dates`` are returned (must be a backtest
    trading day).
    """
    date_set = set(all_dates)
    forced: set = set()

    vix_aligned = vix_series.reindex(sorted(all_dates), method="ffill")
    vix_pct_change = vix_aligned.pct_change()

    spike_dates = vix_pct_change[vix_pct_change.abs() > spike_threshold].index

    for d in spike_dates:
        if d in date_set:
            forced.add(d)

    return forced


# ---------------------------------------------------------------------------
# Quick self-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate two years of fake trading dates
    idx = pd.bdate_range("2023-01-01", "2024-12-31")
    _dates = idx.tolist()

    # Fake signal — slow random walk
    rng = np.random.default_rng(42)
    _signal = pd.Series(
        np.cumsum(rng.normal(0, 0.03, len(_dates))).clip(-1, 1),
        index=idx,
    )

    # Fake VIX — mean-reverting around 18, with a few spikes
    _vix = pd.Series(
        18 + np.cumsum(rng.normal(0, 0.5, len(_dates))).clip(-8, 20),
        index=idx,
    )
    # Inject a large spike
    _vix.iloc[100] = _vix.iloc[99] * 1.35

    for _freq in ("daily", "weekly", "biweekly", "monthly"):
        _sched = build_rebalance_schedule(
            _dates,
            rebalance_freq=_freq,
            signal_series=_signal,
            vix_series=_vix,
            signal_change_threshold=0.15,
            vix_spike_threshold=0.20,
        )
        print(f"{_freq:12s}: {len(_sched):4d} rebalance dates")
