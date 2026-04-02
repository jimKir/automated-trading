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

CACHE_DIR = Path.home() / ".databento_cache" / "stats"
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
    key = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _cache_load(path: Path, ttl_hours: float = CACHE_TTL_HOURS) -> Optional[object]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        if time.time() - raw.get("_ts", 0) < ttl_hours * 3600:
            return raw.get("v")
    except Exception:
        pass
    return None


def _cache_save(path: Path, value: object) -> None:
    try:
        path.write_text(json.dumps({"v": value, "_ts": time.time()}, default=str))
    except Exception as e:
        log.debug(f"Cache write failed: {e}")


# ── DATA FETCHER ──────────────────────────────────────────────────────────────

class _StatisticsFetcher:
    """
    Thin wrapper around databento.Historical for XNAS.ITCH statistics schema.

    Statistics schema records carry:
      - ts_recv:    timestamp (UTC)
      - symbol:     instrument ticker
      - stat_type:  1 = opening cross, 11 = closing cross
      - price:      auction reference/match price (in fixed-point, /1e9 for USD)
      - quantity:   matched volume (shares)

    All API calls are wrapped in try/except; failures return empty DataFrame.
    Results are cached for CACHE_TTL_HOURS to avoid repeated API charges.
    """

    def __init__(self, key: str = DATABENTO_KEY) -> None:
        self._key    = key
        self._client = None
        self._last_request_ts: float = 0.0
        self._init_client()

    def _init_client(self) -> None:
        try:
            import databento
            self._client = databento.Historical(key=self._key)
            log.info("Databento Historical client initialised (statistics/XNAS.ITCH)")
        except Exception as e:
            log.warning(f"Databento client init failed: {e}")

    def _rate_limit(self, min_interval: float = 1.0) -> None:
        """Enforce a minimum interval between API requests."""
        elapsed = time.time() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_ts = time.time()

    def fetch_statistics_day(
        self,
        symbols: List[str],
        trading_date: date,
    ) -> pd.DataFrame:
        """
        Fetch XNAS.ITCH statistics schema records for the given trading day.

        Covers the full trading day (00:00–23:59 UTC) to capture pre-open
        and post-close auction statistics publications.

        Returns a DataFrame with columns:
            symbol, stat_type, price, quantity, ts_recv (UTC, tz-aware)

        The price field is returned in USD (divided by 1e9 from fixed-point).
        Returns empty DataFrame on any error or missing data.

        Parameters
        ----------
        symbols : list of str
            List of ticker symbols (e.g. ['AAPL', 'MSFT']).
        trading_date : date
            The trading day to fetch statistics data for.
        """
        ck = _cache_path("stats_day", sorted(symbols), str(trading_date))
        cached = _cache_load(ck)
        if cached is not None:
            try:
                if not cached:
                    return pd.DataFrame()
                df = pd.DataFrame.from_records(cached)
                if not df.empty and "ts_recv" in df.columns:
                    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
                return df
            except Exception:
                pass

        if self._client is None:
            log.debug("Databento client not available; returning empty DataFrame")
            return pd.DataFrame()

        # Full day window — statistics events may arrive any time during the session
        start_dt = datetime(trading_date.year, trading_date.month, trading_date.day,  0,  0, 0)
        end_dt   = datetime(trading_date.year, trading_date.month, trading_date.day, 23, 59, 59)

        self._rate_limit()
        try:
            store = self._client.timeseries.get_range(
                dataset="XNAS.ITCH",
                schema="statistics",
                start=start_dt,
                end=end_dt,
                symbols=symbols,
            )
            df = store.to_df(pretty_ts=True, map_symbols=True, tz="UTC")

            if df.empty:
                _cache_save(ck, [])
                return pd.DataFrame()

            df.columns = [c.lower() for c in df.columns]

            # Normalise timestamp column
            if "ts_recv" not in df.columns:
                for candidate in ("ts_event", "ts_out", "timestamp"):
                    if candidate in df.columns:
                        df = df.rename(columns={candidate: "ts_recv"})
                        break

            # Normalise symbol column
            if "symbol" not in df.columns:
                for candidate in ("raw_symbol", "instrument_id"):
                    if candidate in df.columns:
                        df = df.rename(columns={candidate: "symbol"})
                        break

            # Map Databento field names → our standard names
            col_aliases = {
                "stat_type": ["stat_type", "type", "event_type"],
                "price":     ["price", "ref_price", "cross_price", "stat_price"],
                "quantity":  ["quantity", "stat_quantity", "size", "quantity_total"],
            }
            for target, candidates in col_aliases.items():
                if target not in df.columns:
                    for c in candidates:
                        if c in df.columns:
                            df = df.rename(columns={c: target})
                            break

            required = ["symbol", "stat_type", "price", "quantity"]
            missing  = [c for c in required if c not in df.columns]
            if missing:
                log.warning(
                    f"Statistics fetch {trading_date}: missing columns {missing}; "
                    f"available: {list(df.columns)}"
                )
                _cache_save(ck, [])
                return pd.DataFrame()

            df = df[["symbol", "stat_type", "price", "quantity", "ts_recv"]].copy()

            # Convert fixed-point price (Databento stores price × 1e9)
            df["price"]    = pd.to_numeric(df["price"],    errors="coerce").fillna(0) / 1e9
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
            df["stat_type"] = pd.to_numeric(df["stat_type"], errors="coerce").fillna(0).astype(int)

            if df["ts_recv"].dtype == object:
                df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)

            records = df.reset_index(drop=True).to_dict("records")
            _cache_save(ck, records)

            log.debug(
                f"Fetched {len(df)} statistics records for {len(symbols)} symbols "
                f"on {trading_date}"
            )
            return df

        except Exception as e:
            log.warning(f"Statistics fetch failed for {trading_date}: {e}")
            return pd.DataFrame()


# ── ROLLING BASELINE STORE ─────────────────────────────────────────────────────

class _VolumeBaseline:
    """
    Maintains an in-memory rolling 20-day baseline of opening and closing
    cross volumes per symbol.

    Baseline is populated lazily as compute_daily is called across dates.
    For backtesting, histories are built up as days are processed in order.
    """

    def __init__(self, window: int = 20) -> None:
        self._window = window
        # {symbol: [vol0, vol1, ...]} — newest last
        self._open_history:  Dict[str, List[float]] = defaultdict(list)
        self._close_history: Dict[str, List[float]] = defaultdict(list)

    def update_open(self, symbol: str, volume: float) -> None:
        hist = self._open_history[symbol]
        hist.append(volume)
        if len(hist) > self._window * 2:  # trim old entries
            self._open_history[symbol] = hist[-self._window * 2:]

    def update_close(self, symbol: str, volume: float) -> None:
        hist = self._close_history[symbol]
        hist.append(volume)
        if len(hist) > self._window * 2:
            self._close_history[symbol] = hist[-self._window * 2:]

    def mean_open(self, symbol: str) -> Optional[float]:
        hist = self._open_history.get(symbol, [])
        if len(hist) < 2:
            return None
        return float(np.mean(hist[-self._window:]))

    def mean_close(self, symbol: str) -> Optional[float]:
        hist = self._close_history.get(symbol, [])
        if len(hist) < 2:
            return None
        return float(np.mean(hist[-self._window:]))


# ── SIGNAL COMPUTATION ────────────────────────────────────────────────────────

class OpeningCrossSignal:
    """
    NASDAQ Opening Cross Volume Anomaly Signal.

    Uses XNAS.ITCH statistics schema (stat_type=1 for opening cross,
    stat_type=11 for closing cross) to compute a volume-weighted directional
    signal anchored to the Cao, Ghysels & Hatheway (2000) opening auction
    anomaly framework.

    Opening cross signal (80% weight):
      volume_anomaly = open_vol / 20d_rolling_mean(open_vol)
      gap            = (open_cross_price − prev_close) / prev_close
      High volume + positive gap  → strong bullish (+1 direction)
      High volume + negative gap  → strong bearish (-1 direction)
      Normal volume (< threshold) → attenuation (× 0.3)

    Closing cross signal (20% weight, contrarian):
      close_vol_anomaly = close_vol / 20d_rolling_mean(close_vol)
      Direction from intraday return sign (prior open-to-close).
      Contrarian: large closing auction volume on a down day suggests
      institutional rebalancing at a temporarily depressed price,
      predicting short-term reversal.

    Parameters
    ----------
    config : dict, optional
        Strategy config dict. Reads from config['opening_cross_signal'].
    key : str, optional
        Databento API key. Falls back to DATABENTO_KEY env var.

    Attributes
    ----------
    weight : float
        Signal weight in the Databento composite (default 0.25).
    lookback_days : int
        Trading days to aggregate in compute_weekly (default 5).
    decay_halflife : float
        Exponential decay half-life in days (default 2).
    high_volume_threshold : float
        Volume anomaly ratio above which the signal is treated as
        high conviction (default 1.5×). Below this, signal is attenuated.
    volume_baseline_days : int
        Rolling window for computing average opening cross volume (default 20).
    """

    # Weight of closing cross component in the composite daily signal
    _CLOSE_CROSS_WEIGHT: float = 0.20
    _OPEN_CROSS_WEIGHT:  float = 0.80

    # Attenuation multiplier for normal-volume days
    _LOW_VOLUME_ATTENUATION: float = 0.3

    def __init__(
        self,
        config: Optional[dict] = None,
        key: str = DATABENTO_KEY,
    ) -> None:
        cfg = (config or {}).get("opening_cross_signal", {})
        self.enabled:               bool  = cfg.get("enabled",               True)
        self.weight:                float = cfg.get("weight",                0.25)
        self.lookback_days:         int   = cfg.get("lookback_days",            5)
        self.decay_halflife:        float = cfg.get("decay_halflife",          2.0)
        self.high_volume_threshold: float = cfg.get("high_volume_threshold",  1.5)
        self.volume_baseline_days:  int   = cfg.get("volume_baseline_days",   20)

        self._fetcher  = _StatisticsFetcher(key=key)
        self._baseline = _VolumeBaseline(window=self.volume_baseline_days)

        log.info(
            f"OpeningCrossSignal: enabled={self.enabled} "
            f"weight={self.weight} lookback={self.lookback_days}d "
            f"halflife={self.decay_halflife}d "
            f"high_vol_thresh={self.high_volume_threshold}x "
            f"baseline_days={self.volume_baseline_days}"
        )

    # ── PUBLIC: FETCH STATISTICS ───────────────────────────────────────────────

    def fetch_statistics(
        self,
        symbols: List[str],
        date_: date,
    ) -> pd.DataFrame:
        """
        Fetch XNAS.ITCH statistics schema for the given date.

        Filters to stat_type == 1 (opening cross) only.

        Parameters
        ----------
        symbols : list of str
            Ticker symbols to fetch.
        date_ : date
            Trading date.

        Returns
        -------
        pd.DataFrame
            Columns: symbol, price, quantity, timestamp (ts_recv).
            Returns empty DataFrame if no data or an error occurs.
        """
        if not symbols:
            return pd.DataFrame()

        df = self._fetcher.fetch_statistics_day(symbols, date_)
        if df.empty:
            return pd.DataFrame()

        # Filter to opening cross only (stat_type == 1)
        open_df = df[df["stat_type"] == 1].copy()
        if open_df.empty:
            log.debug(f"No opening cross records (stat_type=1) for {date_}")
            return pd.DataFrame()

        # Rename ts_recv → timestamp for cleaner public API
        open_df = open_df.rename(columns={"ts_recv": "timestamp"})
        return open_df[["symbol", "price", "quantity", "timestamp"]].reset_index(drop=True)

    # ── INTERNAL: EXTRACT CROSS DATA ──────────────────────────────────────────

    def _extract_cross(
        self,
        df: pd.DataFrame,
        stat_type: int,
        symbols: List[str],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Extract (price, volume) for each symbol from a statistics DataFrame.

        Takes the LAST record for each symbol (most authoritative final print).

        Parameters
        ----------
        df : pd.DataFrame
            Raw statistics DataFrame with columns: symbol, stat_type, price, quantity.
        stat_type : int
            1 for opening cross, 11 for closing cross.
        symbols : list of str
            Symbols to extract. Missing symbols return (0.0, 0.0).

        Returns
        -------
        Dict[str, Tuple[float, float]]
            {symbol: (price, volume)}
        """
        result: Dict[str, Tuple[float, float]] = {}

        if df.empty:
            return {s: (0.0, 0.0) for s in symbols}

        filtered = df[df["stat_type"] == stat_type]

        for sym in symbols:
            sym_df = filtered[filtered["symbol"] == sym] if "symbol" in filtered.columns else pd.DataFrame()
            if sym_df.empty:
                result[sym] = (0.0, 0.0)
                continue

            # Take the LAST record per symbol (final authoritative publication)
            ts_col = "ts_recv" if "ts_recv" in sym_df.columns else (
                "timestamp" if "timestamp" in sym_df.columns else None
            )
            if ts_col:
                last = sym_df.sort_values(ts_col).iloc[-1]
            else:
                last = sym_df.iloc[-1]

            price = float(last.get("price",    0) or 0)
            qty   = float(last.get("quantity", 0) or 0)
            result[sym] = (price, qty)

        return result

    # ── PUBLIC: COMPUTE DAILY ─────────────────────────────────────────────────

    def compute_daily(
        self,
        symbols: List[str],
        date_: date,
        prev_close_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute the opening cross volume anomaly signal for each symbol.

        Fetches XNAS.ITCH statistics schema for date_, extracts opening cross
        (stat_type=1) and closing cross (stat_type=11) data, then:

        Opening cross component (80% weight):
          volume_anomaly = open_vol / 20d_rolling_mean(open_vol)
            → updates rolling baseline for future calls
          gap = (open_cross_price − prev_close) / prev_close   [if prev_close given]
          signal direction:
            volume_anomaly ≥ high_volume_threshold:
              positive gap → +1.0 (strong bullish)
              negative gap → −1.0 (strong bearish)
              no prev_close → use clipped anomaly as signal
            volume_anomaly < high_volume_threshold:
              → attenuate by × 0.3 (normal volume, low conviction)

        Closing cross component (20% weight, contrarian):
          close_vol_anomaly = close_vol / 20d_rolling_mean(close_vol)
          direction from intraday return sign:
            large close vol + down day → +contrarian signal (reversal expected)
            large close vol + up day   → −contrarian signal (rebalancing fade)
          Contrarian component is only applied when close_vol_anomaly ≥ threshold.

        Both components are combined as a weighted composite and clipped to [−1, +1].

        The rolling baseline (20-day average) updates on each call with the
        current day's actual volume. This means the signal is only meaningful
        after at least 2 days of observations; earlier calls return 0.0.

        Parameters
        ----------
        symbols : list of str
            Ticker symbols to compute signal for.
        date_ : date
            Trading day.
        prev_close_prices : dict, optional
            {symbol: close_price_on_prior_day} for gap computation.
            If None, gap direction is omitted and only volume anomaly is used.

        Returns
        -------
        Dict[str, float]
            {symbol: signal} in [−1, +1].
            Symbols with no data receive 0.0.
        """
        if not symbols:
            return {}

        df = self._fetcher.fetch_statistics_day(symbols, date_)

        # Extract opening cross (stat_type=1) and closing cross (stat_type=11)
        open_data:  Dict[str, Tuple[float, float]] = self._extract_cross(df, 1,  symbols)
        close_data: Dict[str, Tuple[float, float]] = self._extract_cross(df, 11, symbols)

        result: Dict[str, float] = {}

        for sym in symbols:
            try:
                open_price, open_vol   = open_data.get(sym,  (0.0, 0.0))
                close_price, close_vol = close_data.get(sym, (0.0, 0.0))

                # ── Opening cross component ──────────────────────────────────
                open_signal = self._compute_open_signal(
                    symbol=sym,
                    open_price=open_price,
                    open_vol=open_vol,
                    prev_close=( prev_close_prices or {}).get(sym),
                )

                # ── Closing cross component (contrarian) ─────────────────────
                # Intraday return direction: from opening cross price to prior close
                # (we use open_price as proxy for today's open direction vs prior close)
                close_signal = self._compute_close_signal(
                    symbol=sym,
                    close_vol=close_vol,
                    intraday_return_sign=self._intraday_sign(open_price, prev_close_prices, sym),
                )

                # ── Composite ────────────────────────────────────────────────
                composite = (
                    self._OPEN_CROSS_WEIGHT  * open_signal
                    + self._CLOSE_CROSS_WEIGHT * close_signal
                )
                clipped = float(np.clip(composite, -1.0, 1.0))
                result[sym] = clipped

                log.debug(
                    f"{sym} {date_}: open_vol={open_vol:.0f} open_px={open_price:.4f} "
                    f"open_sig={open_signal:.4f} close_vol={close_vol:.0f} "
                    f"close_sig={close_signal:.4f} composite={clipped:.4f}"
                )

            except Exception as e:
                log.debug(f"compute_daily error {sym} {date_}: {e}")
                result[sym] = 0.0

        return result

    def _compute_open_signal(
        self,
        symbol: str,
        open_price: float,
        open_vol: float,
        prev_close: Optional[float],
    ) -> float:
        """
        Compute the opening cross signal for a single symbol.

        Updates the rolling volume baseline with the current day's open_vol,
        then computes the volume anomaly and gap-direction signal.

        Returns float in [−1, +1].
        """
        # Update rolling baseline (even with zero volume, to maintain continuity)
        if open_vol > 0:
            self._baseline.update_open(symbol, open_vol)

        mean_open_vol = self._baseline.mean_open(symbol)

        if mean_open_vol is None or mean_open_vol <= 0 or open_vol <= 0:
            return 0.0

        volume_anomaly = open_vol / mean_open_vol

        # Determine gap direction
        if prev_close is not None and prev_close > 0 and open_price > 0:
            gap = (open_price - prev_close) / prev_close
            gap_direction = 1.0 if gap > 0 else (-1.0 if gap < 0 else 0.0)
        else:
            # No prior close available — use volume anomaly only (no direction)
            gap_direction = None

        if volume_anomaly >= self.high_volume_threshold:
            # High conviction: direction × clipped anomaly ratio
            if gap_direction is not None:
                # Full signal: anomaly above threshold + gap direction
                # Clip anomaly to [1.5, 3.0] for stable scaling → [0.5, 1.0] range
                magnitude = float(np.clip(volume_anomaly / (self.high_volume_threshold * 2), 0.5, 1.0))
                signal = gap_direction * magnitude
            else:
                # No price direction — use anomaly magnitude alone
                signal = float(np.clip(volume_anomaly / (self.high_volume_threshold * 2), 0.0, 1.0))
        else:
            # Low conviction: attenuate
            if gap_direction is not None:
                signal = gap_direction * self._LOW_VOLUME_ATTENUATION * (volume_anomaly / self.high_volume_threshold)
            else:
                signal = 0.0

        return float(np.clip(signal, -1.0, 1.0))

    def _compute_close_signal(
        self,
        symbol: str,
        close_vol: float,
        intraday_return_sign: float,
    ) -> float:
        """
        Compute the contrarian closing cross signal for a single symbol.

        Updates the rolling closing volume baseline, then computes the
        closing volume anomaly and applies contrarian direction logic.

        Returns float in [−1, +1].
        """
        if close_vol > 0:
            self._baseline.update_close(symbol, close_vol)

        mean_close_vol = self._baseline.mean_close(symbol)

        if mean_close_vol is None or mean_close_vol <= 0 or close_vol <= 0:
            return 0.0

        close_vol_anomaly = close_vol / mean_close_vol

        if close_vol_anomaly < self.high_volume_threshold:
            return 0.0  # Normal closing volume — no contrarian signal

        # Contrarian: large closing auction volume predicts short-term reversal
        # Large close vol + down day → institutions bought the dip → bullish
        # Large close vol + up day  → institutions rebalanced into strength → bearish fade
        magnitude = float(np.clip(
            close_vol_anomaly / (self.high_volume_threshold * 2), 0.5, 1.0
        ))
        # Negate the intraday direction for contrarian interpretation
        signal = -intraday_return_sign * magnitude

        return float(np.clip(signal, -1.0, 1.0))

    @staticmethod
    def _intraday_sign(
        open_price: float,
        prev_close_prices: Optional[Dict[str, float]],
        symbol: str,
    ) -> float:
        """
        Compute the sign of the intraday move.

        When prev_close is available, sign is derived from the gap:
          open_cross_price vs prior_close → proxy for early-session direction.
        Returns 0.0 if prices are unavailable or equal.
        """
        if prev_close_prices is None or open_price <= 0:
            return 0.0
        prev_close = prev_close_prices.get(symbol)
        if prev_close is None or prev_close <= 0:
            return 0.0
        diff = open_price - prev_close
        if diff > 0:
            return 1.0
        elif diff < 0:
            return -1.0
        return 0.0

    # ── SAFE WRAPPER ──────────────────────────────────────────────────────────

    def _compute_daily_safe(
        self,
        symbols: List[str],
        date_: date,
        prev_close_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """compute_daily with guaranteed fallback to 0.0 for all symbols."""
        try:
            result = self.compute_daily(symbols, date_, prev_close_prices)
            for s in symbols:
                result.setdefault(s, 0.0)
            return result
        except Exception as e:
            log.warning(f"compute_daily failed for {date_}: {e}")
            return {s: 0.0 for s in symbols}

    # ── PUBLIC: COMPUTE WEEKLY ────────────────────────────────────────────────

    def compute_weekly(
        self,
        symbols: List[str],
        as_of_date: date,
        lookback_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute the weekly opening cross signal for each symbol.

        Aggregates daily opening cross signals over the last `lookback_days`
        trading days strictly BEFORE as_of_date (anti-lookahead enforced).
        Applies exponential decay weighting and cross-sectional z-score
        normalisation clipped to [−1, +1].

        Anti-lookahead detail:
            Opening cross data on day T is available at T open (09:30 ET).
            To enforce strict anti-lookahead for weekly rebalancing, we use
            days {T−lookback_days, …, T−1} where T = as_of_date. This means
            we never use the current day's opening cross in the weekly signal.

        Decay weighting:
            weight_i = 2^(−lag / half_life)
            where lag = 0 for the most recent day, n−1 for the oldest.
            Half-life = 2 days → recent opening cross prints dominate.

        Cross-sectional z-score:
            After weighting, scores are standardised across the symbol
            universe and clipped to [−1, +1].

        Parameters
        ----------
        symbols : list of str
            Ticker symbols to compute signal for.
        as_of_date : date
            Signal computation date. All data used is strictly prior to this.
        lookback_days : int, optional
            Override for self.lookback_days.

        Returns
        -------
        Dict[str, float]
            {symbol: signal} in [−1, +1].
            Symbols with no data receive 0.0.
        """
        if not symbols:
            return {}

        n = lookback_days if lookback_days is not None else self.lookback_days

        # Anti-lookahead: use data strictly before as_of_date
        lookback_end  = _prev_trading_day(as_of_date)
        trading_days  = _lookback_trading_days(lookback_end + timedelta(days=1), n)

        if not trading_days:
            log.warning(f"No trading days found before {as_of_date}")
            return {s: 0.0 for s in symbols}

        # Compute daily signals for each lookback day
        daily_signals: List[Dict[str, float]] = []
        valid_dates:   List[date]             = []

        for td in trading_days:
            day_sig = self._compute_daily_safe(symbols, td)
            daily_signals.append(day_sig)
            valid_dates.append(td)

        if not daily_signals:
            return {s: 0.0 for s in symbols}

        # Build matrix: rows=dates (oldest→newest), cols=symbols
        sig_df = pd.DataFrame(daily_signals, index=valid_dates, columns=symbols)
        sig_df = sig_df.fillna(0.0)

        # Exponential decay weights: lag=0 → most recent, lag=n-1 → oldest
        n_days  = len(sig_df)
        lags    = np.arange(n_days - 1, -1, -1, dtype=float)  # [n-1, n-2, ..., 0]
        weights = np.power(2.0, -lags / self.decay_halflife)
        weights /= weights.sum()

        # Weighted average across dates per symbol
        weighted   = sig_df.values * weights[:, np.newaxis]
        raw_scores = pd.Series(weighted.sum(axis=0), index=symbols)

        # Cross-sectional z-score normalisation
        mu  = raw_scores.mean()
        std = raw_scores.std(ddof=1)

        if std > 0:
            normalised = (raw_scores - mu) / std
        else:
            normalised = raw_scores * 0.0  # all identical → zero signal

        clipped = normalised.clip(-1.0, 1.0)
        return {sym: float(clipped.get(sym, 0.0)) for sym in symbols}

    # ── PUBLIC: COMPUTE SERIES ────────────────────────────────────────────────

    def compute_series(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Walk-forward computation of the weekly signal over a date range.

        For each Friday (or the last trading day of each week) between start
        and end, calls compute_weekly(symbols, as_of_date) and collects the
        result into a time-indexed DataFrame.

        Used for backtesting and information coefficient (IC) validation.
        The rolling volume baseline is updated in chronological order across
        the series, so early dates will have a shorter baseline window and
        may produce attenuated signals until 20+ days have accumulated.

        Parameters
        ----------
        symbols : list of str
            Ticker symbols to compute signal for.
        start : date
            Start of the backtest range (inclusive).
        end : date
            End of the backtest range (inclusive).

        Returns
        -------
        pd.DataFrame
            index   = weekly rebalance dates (DatetimeIndex, weekly Friday-ish),
            columns = symbols,
            values  = signal in [−1, +1].
        """
        rebalance_dates = _weekly_rebalance_dates(start, end)

        if not rebalance_dates:
            log.warning(f"No rebalance dates found between {start} and {end}")
            return pd.DataFrame(columns=symbols)

        rows: Dict[date, Dict[str, float]] = {}

        for rb_date in rebalance_dates:
            log.info(f"compute_series: computing weekly signal for {rb_date}")
            try:
                weekly = self.compute_weekly(symbols, rb_date)
                rows[rb_date] = weekly
            except Exception as e:
                log.warning(f"compute_series error on {rb_date}: {e}")
                rows[rb_date] = {s: 0.0 for s in symbols}

        df = pd.DataFrame.from_dict(rows, orient="index", columns=symbols)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df


# ── MODULE-LEVEL CONVENIENCE FUNCTION ─────────────────────────────────────────

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
