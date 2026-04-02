"""
NASDAQ Opening Cross Volume Anomaly Signal
==========================================
Signal 3 of 3 — Cao, Ghysels & Hatheway (2000) opening auction volume anomaly.

NASDAQ publishes two types of statistics events via XNAS.ITCH statistics schema:
  - stat_type = 1:  Opening cross price and volume
  - stat_type = 11: Closing cross price and volume

The opening cross volume is the total number of shares matched at the opening
auction. When it is unusually high (>1.5× the 20-day average), it signals that
institutions had large overnight orders queued. The direction is then confirmed
by the gap between the opening cross price and the prior close.

Academic basis:
  Cao, Ghysels & Hatheway (2000): "Price Discovery Without Trading:
  Evidence from the Nasdaq Preopening". Journal of Finance.
  Opening auction imbalance and volume predict next-day to 5-day returns
  with IC 0.05–0.09.

Signal composition:
  PRIMARY — Opening cross (stat_type = 1):
    volume_anomaly  = open_vol / rolling_mean(open_vol, 20d)
    gap             = (open_cross_price − prev_close) / prev_close
    signal          = sign(gap) × clip(volume_anomaly, 0, 2) / 2
    Threshold 1.5×: anomaly ≥ 1.5 → full signal; below → 0.3× attenuation.

  SECONDARY — Closing cross (stat_type = 11):  [weight 20%]
    close_vol_anomaly = close_vol / rolling_mean(close_vol, 20d)
    Direction from intraday return sign (open→close).
    Contrarian at weekly horizon: large close vol + down day = likely reversal.

Config (settings.yaml):
  opening_cross_signal:
    enabled: true
    weight: 0.25
    lookback_days: 5
    decay_halflife: 2
    high_volume_threshold: 1.5
    volume_baseline_days: 20

Anti-lookahead:
  compute_weekly(symbols, as_of_date) uses data strictly BEFORE as_of_date.
  Opening cross data on day T is available at T-open, but we use it for T+1
  signals to avoid look-ahead in weekly aggregation. A 1-day shift is applied
  internally before weekly aggregation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from src.market_data.catalogue import get_catalogue as _get_catalogue
    _CATALOGUE_AVAILABLE = True
except ImportError:
    _CATALOGUE_AVAILABLE = False

log = logging.getLogger("OpeningCrossSignal")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DATABENTO_KEY = os.environ.get("DATABENTO_KEY", "db-SpVxiQLLTdDe9iD3sLwTpiqgBjtxk")

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "databento"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 24 * 365  # historical data — cache permanently
# NASDAQ opening cross time window: 09:00–09:35 ET = 14:00–14:35 UTC
OPEN_CROSS_START_UTC = (14, 0)   # (hour, minute)
OPEN_CROSS_END_UTC   = (14, 35)  # includes all pre-open auction prints

# NASDAQ closing cross time window: 15:58–16:01 ET = 20:58–21:01 UTC
CLOSE_CROSS_START_UTC = (20, 55)
CLOSE_CROSS_END_UTC   = (21,  5)

# Rough list of US equity market holidays (NYSE/NASDAQ) for busday calculations.
_US_HOLIDAYS: List[str] = [
    # 2022
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30",
    "2022-06-19", "2022-06-20", "2022-07-04", "2022-09-05",
    "2022-11-24", "2022-11-25", "2022-12-26",
    # 2023
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07",
    "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04",
    "2023-11-23", "2023-11-24", "2023-12-25",
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-11-29", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17",
    "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-11-27", "2026-12-25",
]
_HOLIDAY_DATES = np.array(_US_HOLIDAYS, dtype="datetime64[D]")


# ── TRADING CALENDAR HELPERS ──────────────────────────────────────────────────

def _is_trading_day(d: date) -> bool:
    """Return True if d is a US equity market trading day."""
    np_date = np.datetime64(d, "D")
    return bool(np.is_busday(np_date, holidays=_HOLIDAY_DATES))


def _prev_trading_day(d: date) -> date:
    """Return the most recent trading day strictly before d."""
    np_date = np.datetime64(d, "D")
    result = np.busday_offset(np_date, -1, roll="backward", holidays=_HOLIDAY_DATES)
    return pd.Timestamp(result).date()


def _get_trading_days(start: date, end: date) -> List[date]:
    """Return all trading days in [start, end] inclusive, oldest first."""
    np_start = np.datetime64(start, "D")
    np_end   = np.datetime64(end,   "D")
    bdays    = np.busdaycalendar(weekmask="Mon Tue Wed Thu Fri", holidays=_HOLIDAY_DATES)
    all_days = np.arange(np_start, np_end + np.timedelta64(1, "D"), dtype="datetime64[D]")
    mask     = np.is_busday(all_days, busdaycal=bdays)
    return [pd.Timestamp(d).date() for d in all_days[mask]]


def _lookback_trading_days(as_of: date, n: int) -> List[date]:
    """Return n trading days ending strictly before as_of (oldest first)."""
    start       = as_of - timedelta(days=n * 2 + 20)
    days_before = _get_trading_days(start, as_of - timedelta(days=1))
    return days_before[-n:]


def _weekly_rebalance_dates(start: date, end: date) -> List[date]:
    """Return the last trading day of each calendar week between start and end."""
    all_days = _get_trading_days(start, end)
    seen: Dict[tuple, date] = {}
    for d in all_days:
        iso = (d.isocalendar()[0], d.isocalendar()[1])
        seen[iso] = d  # overwrite → keeps last day of each week
    return sorted(seen.values())


# ── CACHE HELPERS ─────────────────────────────────────────────────────────────

def _cache_path(*parts) -> Path:
    """
    Human-readable cache filename: {schema}_{date}_{syms_short}_{hash8}.json
    Examples:
      imbalance_2024-06-15_20syms_a1b2c3d4.json
      opra-trades_2025-03-20_AAPL_e5f6g7h8.json
    The hash ensures uniqueness even if the description collides.
    """
    import hashlib as _hl
    raw = "|".join(str(p) for p in parts)
    h8  = _hl.md5(raw.encode()).hexdigest()[:8]

    # Build readable prefix from parts
    readable_parts = []
    for p in parts:
        s = str(p)
        if s.startswith("["):          # symbol list like "['AAPL', 'MSFT', ...]"
            syms = [x.strip().strip("'") for x in s.strip("[]").split(",")]
            if len(syms) <= 3:
                readable_parts.append("-".join(syms))
            else:
                readable_parts.append(f"{len(syms)}syms")
        elif re.match(r"\d{4}-\d{2}-\d{2}", s):  # date
            readable_parts.append(s)
        else:
            # schema / type label — clean up special chars
            readable_parts.append(re.sub(r"[^a-zA-Z0-9_-]", "", s)[:20])

    prefix = "_".join(p for p in readable_parts if p)
    filename = f"{prefix}_{h8}.json"
    # Keep filenames filesystem-safe (max 200 chars)
    if len(filename) > 200:
        filename = f"{prefix[:180]}_{h8}.json"
    return CACHE_DIR / filename


def get_cache_path_for(*parts) -> Path:
    """Public alias — lets callers use the same key as the module."""
    return _cache_path(*parts)


def build_signal(config: Optional[dict] = None) -> OpeningCrossSignal:
    """
    Factory function for OpeningCrossSignal.

    Parameters
    ----------
    config : dict, optional
        Full strategy config dict (reads config['opening_cross_signal']).

    Returns
    -------
    OpeningCrossSignal
    """
    return OpeningCrossSignal(config=config)


# ── SETTINGS YAML TEMPLATE ────────────────────────────────────────────────────
# Add to settings.yaml under the top-level key:
#
# opening_cross_signal:
#   enabled: true
#   weight: 0.25           # weight in Databento composite
#   lookback_days: 5       # trading days to aggregate in weekly signal
#   decay_halflife: 2      # exponential decay half-life in days
#   high_volume_threshold: 1.5   # volume anomaly ratio for high-conviction flag
#   volume_baseline_days: 20     # rolling window for average opening cross volume
