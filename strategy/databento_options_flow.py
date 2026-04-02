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

log = logging.getLogger("OPRAOptionsFlow")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DATABENTO_KEY = os.environ.get("DATABENTO_KEY", "db-SpVxiQLLTdDe9iD3sLwTpiqgBjtxk")

CACHE_DIR = Path("/tmp/databento_cache/opra")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 24

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
    key = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _cache_load(path: Path, ttl_hours: float = CACHE_TTL_HOURS) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        if time.time() - raw.get("_ts", 0) < ttl_hours * 3600:
            return raw.get("v")
    except Exception:
        pass
    return None


def _cache_save(path: Path, value) -> None:
    try:
        path.write_text(json.dumps({"v": value, "_ts": time.time()}, default=str))
    except Exception as e:
        log.debug(f"Cache write failed: {e}")


# ── OCC SYMBOL PARSER ─────────────────────────────────────────────────────────

def _parse_occ(symbol: str) -> Optional[Tuple[str, str, str, float, date]]:
    """
    Parse an OCC option symbol into its components.

    OCC format: ROOT YYMMDD C/P STRIKEX1000 (strike padded to 8 digits)
    Example: AAPL260418C00250000

    Returns
    -------
    (root, underlying, option_type, strike, expiry)
        option_type: 'C' or 'P'
        strike: float (e.g. 250.0)
        expiry: date
    or None on parse failure.
    """
    m = _OCC_RE.match(symbol.strip().upper())
    if not m:
        return None
    root, yymmdd, opt_type, strike_str = m.groups()
    try:
        expiry = datetime.strptime(yymmdd, "%y%m%d").date()
        strike = int(strike_str) / 1000.0
        return root, root, opt_type, strike, expiry
    except (ValueError, OverflowError):
        return None


def _is_otm(opt_type: str, strike: float, underlying_close: float, otm_threshold: float = 0.05) -> bool:
    """
    Determine if the contract is OTM using moneyness.

    For calls: OTM when strike > close × (1 + threshold)
    For puts:  OTM when strike < close × (1 - threshold)
    Delta proxy < 0.40 ~ moneyness > 5% OTM.
    """
    if underlying_close <= 0:
        return False
    moneyness = (strike - underlying_close) / underlying_close
    if opt_type == "C":
        return moneyness > otm_threshold
    else:  # put
        return moneyness < -otm_threshold


def _delta_proxy(opt_type: str, strike: float, underlying_close: float) -> float:
    """
    Simple delta proxy using strike / underlying ratio.

    For a call: delta ≈ N(d1) ≈ 0.5 when ATM. We approximate with a
    logistic function of moneyness so that deep OTM → 0 and deep ITM → 1.

    Returns value in [0, 1].
    """
    if underlying_close <= 0 or strike <= 0:
        return 0.5
    moneyness = (underlying_close - strike) / underlying_close  # positive = ITM for call
    if opt_type == "P":
        moneyness = -moneyness
    # Logistic approximation: delta_proxy = 1 / (1 + exp(-3 * moneyness))
    try:
        return float(1.0 / (1.0 + np.exp(-3.0 * moneyness)))
    except Exception:
        return 0.5


# ── DATA FETCHER ──────────────────────────────────────────────────────────────

class _OPRAFetcher:
    """
    Thin wrapper around databento.Historical for OPRA.PILLAR data.

    Supports two schemas:
      - 'trades':   tick-level records with aggressor side attribution
      - 'ohlcv-1d': daily aggregated OHLCV per contract (fallback)

    All API calls are wrapped in try/except; failures return empty DataFrame.
    Results are cached for CACHE_TTL_HOURS. Rate-limited to ≤1 req/sec.
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
            log.info("Databento Historical client initialised (OPRA)")
        except Exception as e:
            log.warning(f"Databento client init failed: {e}")

    def _rate_limit(self) -> None:
        """Enforce max 1 request per second."""
        elapsed = time.time() - self._last_request_ts
        if elapsed < _RATE_LIMIT_SLEEP:
            time.sleep(_RATE_LIMIT_SLEEP - elapsed)
        self._last_request_ts = time.time()

    # ── TRADES SCHEMA ─────────────────────────────────────────────────────────

    def fetch_trades_day(
        self,
        underlying: str,
        trading_date: date,
    ) -> pd.DataFrame:
        """
        Fetch OPRA.PILLAR tick-level trades for all options on `underlying`.

        Two-step approach (required by Databento OPRA symbology):
          Step 1: Get instrument definitions to find instrument_ids for this underlying
          Step 2: Fetch trades filtered by those instrument_ids

        Returns DataFrame with columns: symbol, price, size, side
        Empty DataFrame on any error.
        """
        ck = _cache_path("trades", underlying, str(trading_date))
        cached = _cache_load(ck)
        if cached is not None:
            try:
                if not cached:
                    return pd.DataFrame()
                return pd.DataFrame(list(cached.values()))
            except Exception:
                pass

        if self._client is None:
            return pd.DataFrame()

        start_dt = datetime(trading_date.year, trading_date.month, trading_date.day, 13, 30, 0)
        end_dt   = datetime(trading_date.year, trading_date.month, trading_date.day, 20,  0, 0)

        try:
            # Step 1: get definitions (all OPRA instruments for the day, limit to first 2000)
            self._rate_limit()
            def_store = self._client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="definition",
                start=start_dt,
                end=end_dt,
                limit=2000,
            )
            df_def = def_store.to_df()
            if df_def.empty or "underlying" not in df_def.columns:
                _cache_save(ck, {})
                return pd.DataFrame()

            # Filter to our underlying
            mask = df_def["underlying"].str.upper() == underlying.upper()
            relevant = df_def[mask]
            if relevant.empty:
                _cache_save(ck, {})
                return pd.DataFrame()

            # Get instrument_ids
            inst_ids = relevant["instrument_id"].unique().tolist()
            inst_ids = [str(i) for i in inst_ids[:500]]  # cap at 500 contracts

            # Build symbol lookup from definitions
            sym_map = {}
            if "raw_symbol" in relevant.columns:
                for _, row in relevant.iterrows():
                    iid = str(row["instrument_id"])
                    rs  = str(row.get("raw_symbol", ""))
                    if rs:
                        sym_map[iid] = rs

            # Step 2: fetch trades for those instrument_ids
            self._rate_limit()
            trade_store = self._client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="trades",
                start=start_dt,
                end=end_dt,
                symbols=inst_ids,
                stype_in="instrument_id",
                limit=100000,
            )
            df = trade_store.to_df()
            if df.empty:
                _cache_save(ck, {})
                return pd.DataFrame()

            # Add symbol column from map
            df["symbol"] = df["instrument_id"].astype(str).map(sym_map)
            df = df.dropna(subset=["symbol"])

            result = df[["symbol", "price", "size", "side"]].copy()
            _cache_save(ck, result.to_dict(orient="index"))
            return result

        except Exception as e:
            log.debug(f"OPRA fetch {underlying} {trading_date}: {e}")
            _cache_save(ck, {})
            return pd.DataFrame()


    def fetch_ohlcv_day(
        self,
        underlying: str,
        trading_date: date,
    ) -> pd.DataFrame:
        """
        Fetch OPRA.PILLAR ohlcv-1d data for all options on `underlying`
        for the given trading day.

        Returns a DataFrame with columns:
            symbol, open, high, low, close, volume
        Empty DataFrame on any error or missing data.
        """
        ck = _cache_path("ohlcv1d", underlying, str(trading_date))
        cached = _cache_load(ck)
        if cached is not None:
            try:
                if not cached:
                    return pd.DataFrame()
                df = pd.DataFrame.from_records(list(cached.values()))
                return df
            except Exception:
                pass

        if self._client is None:
            return pd.DataFrame()

        # Use next-day boundary for ohlcv-1d (Databento convention: end exclusive)
        start_dt = datetime(trading_date.year, trading_date.month, trading_date.day, 0, 0, 0)
        next_day = trading_date + timedelta(days=1)
        end_dt   = datetime(next_day.year, next_day.month, next_day.day, 0, 0, 0)

        self._rate_limit()
        try:
            store = self._client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="ohlcv-1d",
                start=start_dt,
                end=end_dt,
                symbols=[f"{underlying}.OPT"],
                stype_in="parent",
                stype_out="raw_symbol",
            )
            df = store.to_df(pretty_ts=True, map_symbols=True, tz="UTC")

            if df.empty:
                _cache_save(ck, {})
                return pd.DataFrame()

            df.columns = [c.lower() for c in df.columns]

            col_map = {"raw_symbol": "symbol"}
            for old, new in col_map.items():
                if old in df.columns and new not in df.columns:
                    df = df.rename(columns={old: new})

            needed = ["symbol", "open", "high", "low", "close", "volume"]
            available = [c for c in needed if c in df.columns]
            df = df[available].copy()

            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            records = df.reset_index(drop=True).to_dict("index")
            _cache_save(ck, records)
            log.debug(f"OPRA ohlcv-1d: {len(df)} contracts for {underlying} on {trading_date}")
            return df

        except Exception as e:
            log.warning(f"OPRA ohlcv-1d fetch failed for {underlying} on {trading_date}: {e}")
            return pd.DataFrame()


# ── SIGNAL COMPUTATION ────────────────────────────────────────────────────────

class OPRAOptionsFlowSignal:
    """
    Pan & Poteshman (2006) options order flow signal using OPRA.PILLAR data.

    Measures buy-initiated call volume minus buy-initiated put volume,
    normalised by total options volume, with OTM contract weighting.
    Predicts underlying equity returns at 1–5 day horizon.

    Parameters
    ----------
    config : dict, optional
        Strategy configuration dict. Reads from config['opra_flow_signal'].
    key : str, optional
        Databento API key. Falls back to DATABENTO_KEY env var.

    Attributes
    ----------
    weight : float
        Signal weight in the Databento composite (default 0.40).
    lookback_days : int
        Trading days of OFI to aggregate in compute_weekly (default 5).
    decay_halflife : float
        Exponential decay half-life in days — options signals decay fast (default 2).
    otm_weight_mult : float
        Multiplier for OTM contract volume (default 1.5).
    min_daily_volume : int
        Minimum total options volume on a day to trust the signal (default 100).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        key: str = DATABENTO_KEY,
    ) -> None:
        cfg = (config or {}).get("opra_flow_signal", {})
        self.enabled:         bool  = cfg.get("enabled",          True)
        self.weight:          float = cfg.get("weight",           0.40)
        self.lookback_days:   int   = cfg.get("lookback_days",       5)
        self.decay_halflife:  float = cfg.get("decay_halflife",     2.0)
        self.otm_weight_mult: float = cfg.get("otm_weight_mult",   1.5)
        self.min_daily_volume: int  = cfg.get("min_daily_volume",  100)

        self._fetcher = _OPRAFetcher(key=key)

        log.info(
            f"OPRAOptionsFlowSignal: enabled={self.enabled} "
            f"lookback={self.lookback_days}d halflife={self.decay_halflife}d "
            f"otm_mult={self.otm_weight_mult}x min_vol={self.min_daily_volume}"
        )

    # ── INTERNAL: OFI FROM TRADES ──────────────────────────────────────────

    def _compute_ofi_from_trades(
        self,
        underlying: str,
        df: pd.DataFrame,
        close_price: Optional[float] = None,
    ) -> float:
        """
        Compute Pan & Poteshman OFI from tick-level trade records.

        Parameters
        ----------
        underlying : str
            Underlying ticker (used for logging only).
        df : pd.DataFrame
            Trades DataFrame with columns: symbol, size, side.
        close_price : float, optional
            Previous close price for OTM determination. If None, falls back
            to 5% moneyness threshold without price anchor.

        Returns
        -------
        float
            OFI in [−1, +1].  Positive = net call buying, Negative = net put buying.
        """
        if df.empty:
            return 0.0

        call_vol: float = 0.0
        put_vol:  float = 0.0

        for _, row in df.iterrows():
            sym = str(row.get("symbol", "") or "")
            parsed = _parse_occ(sym)
            if parsed is None:
                continue

            root, _, opt_type, strike, expiry = parsed
            size = float(row.get("size", 0) or 0)
            side = str(row.get("side", "") or "").upper()

            # Only buy-initiated (ask-side aggressor)
            if side != "A":
                continue
            if size <= 0:
                continue

            # OTM weight multiplier
            weight = 1.0
            if close_price is not None and close_price > 0:
                proxy = _delta_proxy(opt_type, strike, close_price)
                if proxy < 0.40:
                    weight = self.otm_weight_mult
            else:
                # Fallback: use pure moneyness when no close available.
                # We have no price anchor, so we cannot determine OTM without it.
                # Use unweighted (no adjustment).
                weight = 1.0

            weighted_size = size * weight

            if opt_type == "C":
                call_vol += weighted_size
            else:  # 'P'
                put_vol  += weighted_size

        total_vol = call_vol + put_vol
        if total_vol < self.min_daily_volume:
            log.debug(
                f"{underlying}: total_vol={total_vol:.0f} < "
                f"min={self.min_daily_volume}; returning 0.0"
            )
            return 0.0

        ofi = (call_vol - put_vol) / max(total_vol, 1.0)
        return float(np.clip(ofi, -1.0, 1.0))

    # ── INTERNAL: OFI FROM OHLCV-1D (FALLBACK) ────────────────────────────

    def _compute_ofi_from_ohlcv(
        self,
        underlying: str,
        df: pd.DataFrame,
        close_price: Optional[float] = None,
    ) -> float:
        """
        Compute a best-effort OFI signal from daily OHLCV aggregates.

        Without aggressor side attribution, we proxy buy-initiated volume
        by the fraction of a session's volume occurring near the ask:
          buy_proxy_vol = volume × (close - low) / max(high - low, 1 tick)

        This is the Chaikin Money Flow multiplier applied per contract.
        Less accurate than tick-level side attribution, but still informative.

        IV skew component:
          Also computes near-ATM put/call price range ratio:
            skew = ATM_put_HL_range / ATM_call_HL_range
            skew_signal = clip((1 - skew) / 2, -1, 1)
          Elevated put skew (skew > 1) → bearish; call skew (skew < 1) → bullish.

        Returns combined OFI (70%) + skew_signal (30%), clipped to [−1, +1].
        """
        if df.empty:
            return 0.0

        call_buy_vol: float = 0.0
        put_buy_vol:  float = 0.0

        # For IV skew: accumulate ATM contract price ranges
        atm_call_ranges: List[float] = []
        atm_put_ranges:  List[float] = []

        for _, row in df.iterrows():
            sym = str(row.get("symbol", "") or "")
            parsed = _parse_occ(sym)
            if parsed is None:
                continue

            root, _, opt_type, strike, expiry = parsed
            volume = float(row.get("volume", 0) or 0)
            high   = float(row.get("high",   0) or 0)
            low    = float(row.get("low",    0) or 0)
            close  = float(row.get("close",  0) or 0)

            if volume <= 0:
                continue

            # OTM weight
            weight = 1.0
            if close_price is not None and close_price > 0:
                proxy = _delta_proxy(opt_type, strike, close_price)
                if proxy < 0.40:
                    weight = self.otm_weight_mult

            # Chaikin money flow multiplier (fraction of bar range closed near ask)
            hl_range = max(high - low, 1e-8)
            cmf_mult = max((close - low) / hl_range, 0.0)  # in [0, 1]
            buy_proxy = volume * cmf_mult * weight

            if opt_type == "C":
                call_buy_vol += buy_proxy
            else:
                put_buy_vol  += buy_proxy

            # Near-ATM for skew: within 10% moneyness of underlying close
            if close_price is not None and close_price > 0:
                moneyness = abs(strike - close_price) / close_price
                if moneyness < 0.10 and high > low:
                    price_range = high - low
                    if opt_type == "C":
                        atm_call_ranges.append(price_range)
                    else:
                        atm_put_ranges.append(price_range)

        total_vol = call_buy_vol + put_buy_vol
        if total_vol < self.min_daily_volume * 0.1:  # scaled for buy-proxy volume
            return 0.0

        ofi = (call_buy_vol - put_buy_vol) / max(total_vol, 1.0)
        ofi = float(np.clip(ofi, -1.0, 1.0))

        # IV skew supplement
        skew_signal = 0.0
        if atm_call_ranges and atm_put_ranges:
            avg_call_range = np.mean(atm_call_ranges)
            avg_put_range  = np.mean(atm_put_ranges)
            if avg_call_range > 0:
                skew_ratio = avg_put_range / avg_call_range
                # skew > 1 = elevated put premium = bearish skew → negative signal
                # skew < 1 = call premium elevated = bullish skew → positive signal
                skew_signal = float(np.clip((1.0 - skew_ratio) / 2.0, -1.0, 1.0))

        combined = 0.70 * ofi + 0.30 * skew_signal
        return float(np.clip(combined, -1.0, 1.0))

    # ── DAILY ─────────────────────────────────────────────────────────────────

    def compute_daily(
        self,
        symbols: List[str],
        trading_date: date,
        close_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute the Pan & Poteshman OFI signal for each symbol on a single day.

        Fetches OPRA.PILLAR trades for each symbol's options (e.g. AAPL.OPT),
        parses each OCC contract symbol (type, strike, expiry), then computes:

            call_vol  = Σ size×weight for calls  where side='A' (buy-initiated)
            put_vol   = Σ size×weight for puts   where side='A'
            OFI       = (call_vol − put_vol) / max(call_vol + put_vol, 1)

        OTM contracts (delta_proxy < 0.40) receive a 1.5× size multiplier.

        Falls back to ohlcv-1d aggregate if tick-level trades are unavailable.

        Parameters
        ----------
        symbols : list of str
            List of underlying tickers (e.g. ['AAPL', 'MSFT']).
        trading_date : date
            The trading day for which to compute the signal.
        close_prices : dict, optional
            {symbol: prev_close_price} for OTM determination. When provided,
            uses delta_proxy < 0.40 (moneyness-based) rather than the 5%
            flat threshold.

        Returns
        -------
        Dict[str, float]
            {symbol: OFI} in [−1, +1].
            Missing / zero-volume symbols receive 0.0.
        """
        if not symbols:
            return {}

        result: Dict[str, float] = {}

        for sym in symbols:
            close_px = (close_prices or {}).get(sym)

            # ── Attempt 1: tick-level trades ──────────────────────────────
            try:
                trades_df = self._fetcher.fetch_trades_day(sym, trading_date)
                if not trades_df.empty:
                    ofi = self._compute_ofi_from_trades(sym, trades_df, close_px)
                    result[sym] = ofi
                    log.debug(f"{sym} {trading_date}: OFI={ofi:.4f} (trades, n={len(trades_df)})")
                    continue
            except Exception as e:
                log.debug(f"Trades path failed for {sym} {trading_date}: {e}")

            # ── Attempt 2: ohlcv-1d fallback ──────────────────────────────
            try:
                ohlcv_df = self._fetcher.fetch_ohlcv_day(sym, trading_date)
                if not ohlcv_df.empty:
                    ofi = self._compute_ofi_from_ohlcv(sym, ohlcv_df, close_px)
                    result[sym] = ofi
                    log.debug(f"{sym} {trading_date}: OFI={ofi:.4f} (ohlcv-1d fallback, n={len(ohlcv_df)})")
                    continue
            except Exception as e:
                log.debug(f"OHLCV-1d path failed for {sym} {trading_date}: {e}")

            # ── Fallback: no data ──────────────────────────────────────────
            log.debug(f"{sym} {trading_date}: no OPRA data, returning 0.0")
            result[sym] = 0.0

        return result

    def _compute_daily_safe(
        self,
        symbols: List[str],
        trading_date: date,
        close_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """compute_daily with guaranteed fallback to 0.0 for all symbols."""
        try:
            result = self.compute_daily(symbols, trading_date, close_prices)
            for s in symbols:
                if s not in result:
                    result[s] = 0.0
            return result
        except Exception as e:
            log.warning(f"compute_daily failed for {trading_date}: {e}")
            return {s: 0.0 for s in symbols}

    # ── WEEKLY ────────────────────────────────────────────────────────────────

    def compute_weekly(
        self,
        symbols: List[str],
        as_of_date: date,
        lookback_days: Optional[int] = None,
        close_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute the weekly OFI signal for each symbol.

        Aggregates daily OFI over the last `lookback_days` trading days
        strictly BEFORE as_of_date (anti-lookahead enforced). Applies
        exponential decay (options signals decay fast; half-life = 2 days)
        and cross-sectional z-score normalisation clipped to [−1, +1].

        Anti-lookahead:
            Options flow on day T is observable at T-close → valid for T+1.
            Days used = {T−lookback_days, …, T−1} where T = as_of_date.

        Parameters
        ----------
        symbols : list of str
        as_of_date : date
            Signal computation date. All data strictly prior to this date.
        lookback_days : int, optional
            Override for self.lookback_days.
        close_prices : dict, optional
            {symbol: price} for OTM weighting (latest available close).

        Returns
        -------
        Dict[str, float]
            {symbol: OFI_weekly} in [−1, +1].
            Symbols with no data receive 0.0.
        """
        if not symbols:
            return {}

        n = lookback_days if lookback_days is not None else self.lookback_days

        # Anti-lookahead: shift by 1 day — options data available from T-close
        lookback_end  = _prev_trading_day(as_of_date)
        trading_days  = _lookback_trading_days(lookback_end + timedelta(days=1), n)

        if not trading_days:
            log.warning(f"No trading days found before {as_of_date}")
            return {s: 0.0 for s in symbols}

        # Collect daily OFI per day
        daily_signals: List[Dict[str, float]] = []
        valid_dates:   List[date]             = []

        for td in trading_days:
            day_sig = self._compute_daily_safe(symbols, td, close_prices)
            daily_signals.append(day_sig)
            valid_dates.append(td)

        if not daily_signals:
            return {s: 0.0 for s in symbols}

        # Build DataFrame: rows = dates, cols = symbols
        sig_df = pd.DataFrame(daily_signals, index=valid_dates, columns=symbols).fillna(0.0)

        # Exponential decay: half-life = self.decay_halflife days (options decay fast)
        # weight_i = 2^(-lag / hl), lag=0 for most recent day
        n_days = len(sig_df)
        lags   = np.arange(n_days - 1, -1, -1, dtype=float)   # [n-1, ..., 0]
        decay_weights = np.power(2.0, -lags / self.decay_halflife)
        decay_weights /= decay_weights.sum()  # normalise

        weighted    = sig_df.values * decay_weights[:, np.newaxis]
        raw_scores  = pd.Series(weighted.sum(axis=0), index=symbols)

        # Cross-sectional z-score
        mu  = raw_scores.mean()
        std = raw_scores.std(ddof=1)

        if std > 1e-8:
            normalised = (raw_scores - mu) / std
        else:
            normalised = raw_scores * 0.0  # all identical → zero signal

        clipped = normalised.clip(-1.0, 1.0)
        return {sym: float(clipped.get(sym, 0.0)) for sym in symbols}

    # ── SERIES (BACKTESTING) ──────────────────────────────────────────────────

    def compute_series(
        self,
        symbols: List[str],
        start: date,
        end: date,
        close_prices_ts: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """
        Walk-forward computation of the weekly OFI signal over a date range.

        For each Friday (or last trading day of the week) from start to end,
        calls compute_weekly(symbols, as_of_date) and collects the result.
        Used for backtesting and IC validation against Pan & Poteshman (2006).

        Parameters
        ----------
        symbols : list of str
        start : date
            Start of the backtest range (inclusive).
        end : date
            End of the backtest range (inclusive).
        close_prices_ts : dict, optional
            {symbol: pd.Series} of daily closing prices indexed by date.
            If provided, per-rebalance close prices are extracted for OTM
            weighting.

        Returns
        -------
        pd.DataFrame
            index   = weekly rebalance dates (last trading day of each week),
            columns = symbols,
            values  = OFI signal in [−1, +1].
        """
        rebalance_dates = _weekly_rebalance_dates(start, end)

        if not rebalance_dates:
            log.warning(f"No rebalance dates found between {start} and {end}")
            return pd.DataFrame(columns=symbols)

        rows: Dict[date, Dict[str, float]] = {}

        for rb_date in rebalance_dates:
            log.info(f"compute_series: OFI signal for {rb_date}")

            # Extract close prices as of rb_date - 1 for OTM weighting
            close_px: Optional[Dict[str, float]] = None
            if close_prices_ts is not None:
                prev_close_date = _prev_trading_day(rb_date)
                close_px = {}
                for sym, price_series in close_prices_ts.items():
                    try:
                        # Strict: only data up to prev_close_date
                        available = price_series[
                            price_series.index <= pd.Timestamp(prev_close_date)
                        ]
                        if len(available) > 0:
                            close_px[sym] = float(available.iloc[-1])
                    except Exception:
                        pass

            try:
                weekly = self.compute_weekly(
                    symbols, rb_date, close_prices=close_px
                )
                rows[rb_date] = weekly
            except Exception as e:
                log.warning(f"compute_series error on {rb_date}: {e}")
                rows[rb_date] = {s: 0.0 for s in symbols}

        df = pd.DataFrame.from_dict(rows, orient="index", columns=symbols)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df


# ── MODULE-LEVEL CONVENIENCE FUNCTION ─────────────────────────────────────────

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
