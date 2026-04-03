"""
Databento Closing Auction Imbalance Signal
==========================================
Signal 1 of 3 — NASDAQ closing auction order imbalance.

NASDAQ publishes closing auction imbalance data every 10 seconds starting
at 3:50 PM ET via the XNAS.ITCH feed. The data shows the net excess of
buy vs sell orders at the current reference price heading into the close.
This provides a pre-close view of institutional demand that predicts
next-day and next-week directional bias.

Economic basis:
  - Closing auction imbalance reflects institutional programme trading
    demand that cannot be filled in the continuous session.
  - A persistent buy imbalance at 3:55 PM signals net institutional
    demand that will be filled at the close and often persists T+1.
  - Korajczyk & Murphy (2019): institutional demand imbalances around
    the close predict 1-5 day future returns with IC 0.06-0.10.

Fields used from XNAS.ITCH imbalance schema:
  - total_imbalance_qty: unsigned size of imbalance
  - side: 'A' (buy) / 'B' (sell) / 'N' (none)
  - ref_price: reference price for the auction
  - paired_qty: already matched (bilateral) quantity

Config (settings.yaml):
  imbalance_signal:
    enabled: true
    weight: 0.35           # weight in Databento composite
    lookback_days: 10      # days of imbalance to aggregate
    decay_halflife: 3      # exponential decay half-life in days
    min_paired_qty: 1000   # minimum paired qty to trust the signal

Anti-lookahead:
  compute_weekly(symbols, as_of_date) uses data strictly BEFORE as_of_date.
  The imbalance at 3:55 PM on day T is available at T-close → used for T+1.
  A 1-day shift is applied internally before aggregation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Data catalogue — tracks all fetched data (source/schema/date/path)
try:
    from src.market_data.catalogue import get_catalogue as _get_catalogue
    _CATALOGUE_AVAILABLE = True
except ImportError:
    _CATALOGUE_AVAILABLE = False

log = logging.getLogger("DatabentoImbalance")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DATABENTO_KEY = os.environ.get("DATABENTO_KEY", "db-SpVxiQLLTdDe9iD3sLwTpiqgBjtxk")

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "databento"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 24 * 365  # historical data — cache permanently

# 4:00 PM ET is 20:00 UTC (valid year-round; NASDAQ close is always 20:00 UTC)
CLOSE_UTC_HOUR = 20
# Start of imbalance window: 3:50 PM ET = 19:50 UTC
IMBALANCE_START_UTC_HOUR = 19
IMBALANCE_START_UTC_MINUTE = 50

# Rough list of US equity market holidays (NYSE/NASDAQ) for busday calculations.
# We keep 5 years of known dates; the code gracefully handles missing ones.
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
    if np.is_busday(np_date, holidays=_HOLIDAY_DATES):
        return True
    return False


def _prev_trading_day(d: date) -> date:
    """Return the most recent trading day strictly before d."""
    np_date = np.datetime64(d, "D")
    result = np.busday_offset(np_date, -1, roll="backward", holidays=_HOLIDAY_DATES)
    return pd.Timestamp(result).date()


def _get_trading_days(start: date, end: date) -> List[date]:
    """
    Return all trading days in [start, end] inclusive, oldest first.
    end is inclusive only if it is itself a trading day.
    """
    np_start = np.datetime64(start, "D")
    np_end = np.datetime64(end, "D")
    bdays = np.busdaycalendar(weekmask="Mon Tue Wed Thu Fri", holidays=_HOLIDAY_DATES)
    all_days = np.arange(np_start, np_end + np.timedelta64(1, "D"), dtype="datetime64[D]")
    mask = np.is_busday(all_days, busdaycal=bdays)
    return [pd.Timestamp(d).date() for d in all_days[mask]]


def _lookback_trading_days(as_of: date, n: int) -> List[date]:
    """
    Return the n trading days ending strictly before as_of (oldest first).
    These are the days whose imbalance data is available at as_of open.
    """
    # Start far enough back to find n days even with holidays
    start = as_of - timedelta(days=n * 2 + 20)
    # All trading days in range [start, day_before_as_of]
    days_before = _get_trading_days(start, as_of - timedelta(days=1))
    return days_before[-n:]  # most recent n, oldest first


# ── CACHE HELPERS ─────────────────────────────────────────────────────────────

def _cache_path(*parts) -> Path:
    """
    Human-readable cache filename: {schema}_{date}_{syms_short}_{hash8}.json
    Examples:
      imbalance_2024-06-15_20syms_a1b2c3d4.json
      opra-trades_2025-03-20_AAPL_e5f6g7h8.json
    The hash ensures uniqueness even if the description collides.
    """
    import hashlib as _hl, re
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


def get_cache_path_for(*parts) -> Path:
    """Public alias — lets the validation script use the exact same key."""
    return _cache_path(*parts)


def _cache_load(path: Path, ttl_hours: float = CACHE_TTL_HOURS) -> Optional[dict]:
    """
    Load cached data. Returns:
      None — file missing, expired, or corrupt  → caller should re-fetch
      {}   — confirmed "no data" from Databento → caller must NOT re-fetch
      dict — real data rows                     → caller uses directly

    IMPORTANT: v={} is a permanent sentinel meaning "we already asked Databento
    and it returned nothing for this day." Treat it as a valid cached result,
    NOT as a miss. Deleting it would trigger an API call that returns the same
    empty result, wasting money and time.
    """
    if not path.exists():
        return None
    # Files < 100 bytes = v={} sentinel written by _cache_save — valid "no data"
    if path.stat().st_size < 100:
        return {}   # confirmed no-data — do NOT delete, do NOT re-fetch
    try:
        raw = json.loads(path.read_text())
        if time.time() - raw.get("_ts", 0) > ttl_hours * 3600:
            return None  # expired — allow re-fetch
        v = raw.get("v", {})
        if not isinstance(v, dict):
            return None  # corrupt structure — allow re-fetch
        # v={} in a larger file = also a valid "no data" sentinel
        return v  # callers get {} for no-data, or populated dict for real data
    except Exception:
        return None

def _cache_save(path: Path, value) -> None:
    try:
        path.write_text(json.dumps({"v": value, "_ts": time.time()}, default=str))
    except Exception as e:
        log.debug(f"Cache write failed: {e}")


# ── DATA FETCHER ──────────────────────────────────────────────────────────────

class _DatabentoFetcher:
    """
    Thin wrapper around databento.Historical for XNAS.ITCH imbalance data.

    All API calls are wrapped in try/except; failures return empty DataFrame.
    Results are cached for CACHE_TTL_HOURS to avoid repeated API charges.
    """

    def __init__(self, key: str = DATABENTO_KEY):
        self._key = key
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            import databento
            self._client = databento.Historical(key=self._key)
            log.info("Databento Historical client initialised")
        except Exception as e:
            log.warning(f"Databento client init failed: {e}")

    def fetch_imbalance_day(
        self,
        symbols: List[str],
        trading_date: date,
    ) -> pd.DataFrame:
        """
        Fetch XNAS.ITCH imbalance records for the given trading day.

        Returns a DataFrame with columns:
            symbol, ts_recv (UTC), total_imbalance_qty, side,
            ref_price, paired_qty
        Indexed by ts_recv (UTC, tz-aware).

        Returns empty DataFrame on any error or missing data.
        """
        ck = _cache_path("imbalance", sorted(symbols), str(trading_date))
        cached = _cache_load(ck)
        if cached is not None:
            # {} = confirmed no-data sentinel — return empty DataFrame immediately
            if len(cached) == 0:
                return pd.DataFrame()
            try:
                # orient='index': keys are row indices, values are {col: val} dicts
                # DO NOT use pd.DataFrame(cached) — that treats keys as COLUMN names
                # (transposed), causing pd.to_datetime(index) to fail on field names
                # like "symbol", which then silently falls through to an API re-fetch.
                df = pd.DataFrame.from_dict(cached, orient="index")
                if df.empty:
                    return pd.DataFrame()
                # Convert string row indices back to integers, then drop (not timestamps)
                df = df.reset_index(drop=True)
                return df
            except Exception as e:
                log.debug(f"Cache reconstruct failed: {e}")
                # Do NOT fall through to API — return empty to avoid charges
                return pd.DataFrame()

        if self._client is None:
            log.debug("Databento client not available; returning empty DataFrame")
            return pd.DataFrame()

        # NASDAQ closing imbalance window: 3:50 PM – 4:00 PM ET = 19:50–20:01 UTC
        start_dt = datetime(
            trading_date.year, trading_date.month, trading_date.day,
            IMBALANCE_START_UTC_HOUR, IMBALANCE_START_UTC_MINUTE, 0
        )
        # End just after market close (inclusive of 4:00 PM prints)
        end_dt = datetime(
            trading_date.year, trading_date.month, trading_date.day,
            CLOSE_UTC_HOUR, 1, 0
        )

        try:
            store = self._client.timeseries.get_range(
                dataset="XNAS.ITCH",
                schema="imbalance",
                start=start_dt,
                end=end_dt,
                symbols=symbols,
            )
            df = store.to_df(
                pretty_ts=True,
                map_symbols=True,
                tz="UTC",
            )
            if df.empty:
                _cache_save(ck, {})
                return pd.DataFrame()

            # Normalise column names (databento may use different casing)
            df.columns = [c.lower() for c in df.columns]

            # Keep only the columns we need
            needed = ["symbol", "total_imbalance_qty", "side", "ref_price", "paired_qty"]
            missing_cols = [c for c in needed if c not in df.columns]
            if missing_cols:
                log.warning(
                    f"Imbalance fetch {trading_date}: missing columns {missing_cols}; "
                    f"available: {list(df.columns)}"
                )
                _cache_save(ck, {})
                return pd.DataFrame()

            df = df[needed].copy()
            df.index = df.index.tz_localize("UTC") if df.index.tzinfo is None else df.index

            # FIX: index may be non-unique (multiple snapshots per symbol per day)
            # Reset to integer index before saving to avoid orient='index' error
            df = df.reset_index(drop=True)

            # Persist to cache
            cache_dict = df.to_dict("index")
            _cache_save(ck, {str(k): v for k, v in cache_dict.items()})
            # Record in data catalogue
            if _CATALOGUE_AVAILABLE:
                try:
                    _get_catalogue().record(
                        source="databento", dataset="XNAS.ITCH",
                        schema="imbalance", symbols=list(symbols),
                        frequency="snapshot", start=str(trading_date),
                        end=str(trading_date), rows=len(df),
                        cache_path=str(ck),
                        notes="closing auction window 19:50-19:59 UTC",
                        tags=["signal", "microstructure", "imbalance"],
                    )
                except Exception:
                    pass
            log.debug(
                f"Fetched {len(df)} imbalance records for {len(symbols)} symbols "
                f"on {trading_date}"
            )
            return df

        except Exception as e:
            log.warning(f"Imbalance fetch failed for {trading_date}: {e}")
            return pd.DataFrame()


# ── SIGNAL COMPUTATION ────────────────────────────────────────────────────────

class ClosingImbalanceSignal:
    """
    NASDAQ Closing Auction Imbalance Signal.

    Uses XNAS.ITCH imbalance schema data published every 10 seconds from
    3:50 PM ET. Takes the last snapshot before 4:00 PM ET and computes
    a signed, normalised imbalance score per symbol.

    Parameters
    ----------
    config : dict, optional
        Strategy configuration dict. Reads from config['imbalance_signal'].
    key : str, optional
        Databento API key. Falls back to DATABENTO_KEY env var.

    Attributes
    ----------
    lookback_days : int
        Number of trading days to aggregate in compute_weekly (default 10).
    decay_halflife : float
        Exponential decay half-life in days for weekly aggregation (default 3).
    min_paired_qty : int
        Minimum paired_qty to trust a daily imbalance snapshot (default 1000).
    weight : float
        Signal weight in the Databento composite (informational, default 0.35).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        key: str = DATABENTO_KEY,
    ) -> None:
        cfg = (config or {}).get("imbalance_signal", {})
        self.enabled: bool         = cfg.get("enabled",        True)
        self.weight: float         = cfg.get("weight",         0.35)
        self.lookback_days: int    = cfg.get("lookback_days",  10)
        self.decay_halflife: float = cfg.get("decay_halflife", 3.0)
        self.min_paired_qty: int   = cfg.get("min_paired_qty", 1000)

        self._fetcher = _DatabentoFetcher(key=key)

        log.info(
            f"ClosingImbalanceSignal: enabled={self.enabled} "
            f"lookback={self.lookback_days}d halflife={self.decay_halflife}d "
            f"min_paired={self.min_paired_qty}"
        )

    # ── DAILY ─────────────────────────────────────────────────────────────────

    def compute_daily(
        self,
        symbols: List[str],
        trading_date: date,
    ) -> Dict[str, float]:
        """
        Compute the closing imbalance signal for each symbol on a single day.

        Fetches XNAS.ITCH imbalance data for trading_date, takes the LAST
        snapshot for each symbol (closest to 4:00 PM ET / 20:00 UTC), then:

            signed_imbalance = total_imbalance_qty × sign(side)
                where sign: 'A'→+1, 'B'→-1, 'N'→0

            relative_imbalance = signed_imbalance / max(paired_qty, 1)

        A min_paired_qty guard is applied: if paired_qty < min_paired_qty,
        the signal is treated as 0.0 (insufficient auction liquidity to trust
        the imbalance direction).

        The raw signal is in (-∞, +∞) but typically stays in [-5, +5];
        cross-sectional normalisation in compute_weekly clips to [-1, +1].

        Parameters
        ----------
        symbols : list of str
            List of ticker symbols (e.g. ['AAPL', 'MSFT']).
        trading_date : date
            The trading day to fetch imbalance data for.

        Returns
        -------
        Dict[str, float]
            {symbol: relative_imbalance} for all symbols with valid data.
            Missing symbols (no data or error) are absent from the dict.
        """
        if not symbols:
            return {}

        df = self._fetcher.fetch_imbalance_day(symbols, trading_date)
        if df.empty:
            log.debug(f"No imbalance data for {trading_date}; returning neutral")
            return {s: 0.0 for s in symbols}

        result: Dict[str, float] = {}

        for sym in symbols:
            try:
                sym_df = df[df["symbol"] == sym] if "symbol" in df.columns else pd.DataFrame()
                if sym_df.empty:
                    result[sym] = 0.0
                    continue

                # Take the LAST snapshot (closest to 4:00 PM ET)
                last = sym_df.sort_index().iloc[-1]

                total_imb = float(last.get("total_imbalance_qty", 0) or 0)
                side       = str(last.get("side", "N") or "N").upper()
                paired     = float(last.get("paired_qty", 0) or 0)

                # Paired quantity guard
                if paired < self.min_paired_qty:
                    log.debug(
                        f"{sym} {trading_date}: paired_qty={paired:.0f} < "
                        f"min={self.min_paired_qty}; treating as neutral"
                    )
                    result[sym] = 0.0
                    continue

                # Sign the imbalance
                if side == "A":
                    direction = 1.0
                elif side == "B":
                    direction = -1.0
                else:
                    direction = 0.0

                signed_imbalance = total_imb * direction
                relative = signed_imbalance / max(paired, 1.0)

                result[sym] = float(relative)
                log.debug(
                    f"{sym} {trading_date}: side={side} total={total_imb:.0f} "
                    f"paired={paired:.0f} rel={relative:.4f}"
                )

            except Exception as e:
                log.debug(f"compute_daily error {sym} {trading_date}: {e}")
                result[sym] = 0.0

        return result

    # ── WEEKLY ────────────────────────────────────────────────────────────────

    def compute_weekly(
        self,
        symbols: List[str],
        as_of_date: date,
        lookback_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute the weekly closing imbalance signal for each symbol.

        Aggregates daily imbalance over the last `lookback_days` trading days
        strictly BEFORE as_of_date (anti-lookahead enforced). Applies
        exponential decay (higher weight to more recent days) and cross-
        sectional z-score normalisation clipped to [-1, +1].

        Anti-lookahead detail:
            The imbalance printed at 3:55 PM on day T is data that becomes
            available at day T close. It is therefore valid as a signal
            input starting day T+1. We shift the usable day range by 1 day:
            days used = {T-lookback_days, …, T-1} where T = as_of_date.
            This means we NEVER use today's imbalance (it may not yet exist
            and would introduce lookahead).

        Parameters
        ----------
        symbols : list of str
        as_of_date : date
            Signal computation date. All data used is strictly prior to this.
        lookback_days : int, optional
            Override for self.lookback_days.

        Returns
        -------
        Dict[str, float]
            {symbol: signal} in [-1, +1].
            Symbols with no data receive 0.0.
        """
        if not symbols:
            return {}

        n = lookback_days if lookback_days is not None else self.lookback_days

        # Get the trading days available as of as_of_date (shift by 1 = anti-lookahead)
        lookback_end  = _prev_trading_day(as_of_date)   # last day whose data is available
        trading_days  = _lookback_trading_days(lookback_end + timedelta(days=1), n)

        if not trading_days:
            log.warning(f"No trading days found before {as_of_date}")
            return {s: 0.0 for s in symbols}

        # Compute daily signal for each day
        daily_signals: List[Dict[str, float]] = []
        valid_dates: List[date] = []

        for td in trading_days:
            day_sig = self._compute_daily_safe(symbols, td)
            daily_signals.append(day_sig)
            valid_dates.append(td)

        if not daily_signals:
            return {s: 0.0 for s in symbols}

        # Build DataFrame: rows=dates, cols=symbols
        sig_df = pd.DataFrame(daily_signals, index=valid_dates, columns=symbols)
        sig_df = sig_df.fillna(0.0)

        # Exponential decay weights: more recent days weighted higher
        # weight_i = 2^(-lag / half_life), where lag = days from end (0 = most recent)
        n_days = len(sig_df)
        lags = np.arange(n_days - 1, -1, -1, dtype=float)  # [n-1, n-2, ..., 0]
        decay_weights = np.power(2.0, -lags / self.decay_halflife)
        decay_weights /= decay_weights.sum()  # normalise to sum=1

        # Weighted average across days per symbol
        weighted = sig_df.values * decay_weights[:, np.newaxis]
        raw_scores = pd.Series(weighted.sum(axis=0), index=symbols)

        # Cross-sectional z-score normalisation
        mu  = raw_scores.mean()
        std = raw_scores.std(ddof=1)

        if std > 0:
            normalised = (raw_scores - mu) / std
        else:
            normalised = raw_scores * 0.0  # all identical → zero signal

        # Clip to [-1, +1]
        clipped = normalised.clip(-1.0, 1.0)
        return {sym: float(clipped.get(sym, 0.0)) for sym in symbols}

    def _compute_daily_safe(
        self,
        symbols: List[str],
        trading_date: date,
    ) -> Dict[str, float]:
        """compute_daily with guaranteed fallback to 0.0 for all symbols."""
        try:
            result = self.compute_daily(symbols, trading_date)
            # Fill any missing symbols with 0.0
            for s in symbols:
                if s not in result:
                    result[s] = 0.0
            return result
        except Exception as e:
            log.warning(f"compute_daily failed for {trading_date}: {e}")
            return {s: 0.0 for s in symbols}

    # ── SERIES (BACKTESTING) ──────────────────────────────────────────────────

    def compute_series(
        self,
        symbols: List[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Walk-forward computation of the weekly signal over a date range.

        For each Friday (or the last trading day of the week) from start to end,
        calls compute_weekly(symbols, as_of_date) and collects the result.

        Used for backtesting and information coefficient (IC) validation.

        Parameters
        ----------
        symbols : list of str
        start : date
            Start of the backtest range (inclusive).
        end : date
            End of the backtest range (inclusive).

        Returns
        -------
        pd.DataFrame
            index = weekly rebalance dates (every Friday or last trading day),
            columns = symbols,
            values  = signal in [-1, +1]
        """
        rebalance_dates = self._weekly_rebalance_dates(start, end)

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

    @staticmethod
    def _weekly_rebalance_dates(start: date, end: date) -> List[date]:
        """
        Return the last trading day of each calendar week (typically Friday)
        between start and end inclusive.
        """
        all_trading_days = _get_trading_days(start, end)
        if not all_trading_days:
            return []

        # Group by ISO week, take the last trading day of each week
        seen_weeks: Dict[tuple, date] = {}
        for d in all_trading_days:
            iso = (d.isocalendar()[0], d.isocalendar()[1])  # (year, week)
            seen_weeks[iso] = d  # overwrite → keeps last day of each week

        return sorted(seen_weeks.values())


# ── MODULE-LEVEL CONVENIENCE FUNCTION ─────────────────────────────────────────

def build_signal(config: Optional[dict] = None) -> ClosingImbalanceSignal:
    """
    Factory function for ClosingImbalanceSignal.

    Parameters
    ----------
    config : dict, optional
        Full strategy config dict (reads config['imbalance_signal']).

    Returns
    -------
    ClosingImbalanceSignal
    """
    return ClosingImbalanceSignal(config=config)


# ── SETTINGS YAML TEMPLATE ────────────────────────────────────────────────────
# Add to settings.yaml under the top-level key:
#
# imbalance_signal:
#   enabled: true
#   weight: 0.35           # weight in Databento composite
#   lookback_days: 10      # trading days of imbalance to aggregate
#   decay_halflife: 3      # exponential decay half-life in days
#   min_paired_qty: 1000   # minimum paired qty to trust the signal
