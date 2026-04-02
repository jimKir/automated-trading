"""
OPRA Options Order Flow Signal
==============================
Signal 2 of 3 — Pan & Poteshman (2006) options order flow imbalance.

Uses Databento OPRA.PILLAR tick-level options trades from all US options
exchanges (2013–present). Each trade record carries price, size, aggressor
side (buy- vs sell-initiated), and the full OCC contract symbol.

Economic basis:
  Pan & Poteshman (2006): buy-initiated call volume minus buy-initiated put
  volume, normalised by total volume, predicts underlying stock returns at
  1–5 day horizon with IC 0.08–0.12.

  Informed traders systematically prefer options to equities for directional
  bets (leverage, limited loss). The asymmetry in call vs put buying pressure
  — net of noise trading — carries forward-looking information unavailable in
  price and volume data alone.

  OTM contracts carry stronger information content (informed traders prefer
  OTM for higher leverage), so a 1.5× weight multiplier is applied to OTM
  contracts when computing call_vol / put_vol.

Signal computation (Pan & Poteshman):
  call_vol  = Σ size  where contract_type='C' and side='A' (ask-side = buy-initiated)
  put_vol   = Σ size  where contract_type='P' and side='A'
  total_vol = call_vol + put_vol
  OFI       = (call_vol − put_vol) / max(total_vol, 1)   → clip to [−1, +1]

OTM weighting:
  Contracts with delta_proxy < 0.40 (i.e. >5% OTM from underlying close)
  receive a 1.5× size multiplier before call_vol / put_vol accumulation.
  When no underlying close is available, OTM proxy is |moneyness| > 0.05.

IV skew supplement:
  When ohlcv-1d fallback data is available, we compute a near-ATM put/call
  price-range ratio as a rough skew proxy. Values are averaged with the
  OFI score (50/50) to produce a combined daily signal.

Fallback chain:
  1. OPRA.PILLAR trades schema  — full tick-level buy/sell attribution
  2. OPRA.PILLAR ohlcv-1d schema — daily aggregated volume per contract
  3. Return 0.0 for symbol (graceful degradation)

Anti-lookahead:
  compute_weekly(symbols, as_of_date) uses trade data strictly BEFORE
  as_of_date. Options flow on day T is available at T-close → used for T+1.
  A one-day shift is applied before aggregation.

Config (settings.yaml):
  opra_flow_signal:
    enabled: true
    weight: 0.40
    lookback_days: 5
    decay_halflife: 2
    otm_weight_mult: 1.5
    min_daily_volume: 100
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Data catalogue — tracks all fetched data (source/schema/date/path)
try:
    from src.market_data.catalogue import get_catalogue as _get_catalogue
    _CATALOGUE_AVAILABLE = True
except ImportError:
    _CATALOGUE_AVAILABLE = False

log = logging.getLogger("OPRAOptionsFlow")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DATABENTO_KEY = os.environ.get("DATABENTO_KEY", "db-SpVxiQLLTdDe9iD3sLwTpiqgBjtxk")

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "databento"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 24 * 365  # historical data — cache permanently
# Rate limit: 1 request per second
_RATE_LIMIT_SLEEP = 1.0

# OCC symbol regex:  ROOT YYMMDD C/P STRIKEX1000 (padded to 8 digits)
# Example: AAPL260418C00250000
_OCC_RE = re.compile(
    r"^([A-Z1-9]{1,6})(\d{6})([CP])(\d{8})$"
)

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

def _get_trading_days(start: date, end: date) -> List[date]:
    """Return all trading days in [start, end] inclusive, oldest first."""
    np_start = np.datetime64(start, "D")
    np_end   = np.datetime64(end,   "D")
    bdays    = np.busdaycalendar(weekmask="Mon Tue Wed Thu Fri", holidays=_HOLIDAY_DATES)
    all_days = np.arange(np_start, np_end + np.timedelta64(1, "D"), dtype="datetime64[D]")
    mask     = np.is_busday(all_days, busdaycal=bdays)
    return [pd.Timestamp(d).date() for d in all_days[mask]]


def _prev_trading_day(d: date) -> date:
    """Return the most recent trading day strictly before d."""
    np_date = np.datetime64(d, "D")
    result  = np.busday_offset(np_date, -1, roll="backward", holidays=_HOLIDAY_DATES)
    return pd.Timestamp(result).date()


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


def build_signal(config: Optional[dict] = None) -> OPRAOptionsFlowSignal:
    """
    Factory function for OPRAOptionsFlowSignal.

    Parameters
    ----------
    config : dict, optional
        Full strategy config dict (reads config['opra_flow_signal']).

    Returns
    -------
    OPRAOptionsFlowSignal
    """
    return OPRAOptionsFlowSignal(config=config)


# ── SETTINGS YAML TEMPLATE ────────────────────────────────────────────────────
# Add to settings.yaml under the top-level key:
#
# opra_flow_signal:
#   enabled: true
#   weight: 0.40           # weight in Databento composite
#   lookback_days: 5       # trading days of OFI to aggregate (options decay fast)
#   decay_halflife: 2      # exponential decay half-life in days
#   otm_weight_mult: 1.5   # multiplier for OTM contracts (delta_proxy < 0.40)
#   min_daily_volume: 100  # minimum total options volume to trust the signal
