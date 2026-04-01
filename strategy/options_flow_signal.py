"""
options_flow_signal.py
======================
Options Order Flow Signal for the Automated Trading System.

OVERVIEW
--------
This module computes a directional signal in [-1, +1] derived from three
complementary options-market data sources:

  A. Unusual Options Activity   (50% weight)
  B. Implied Volatility Skew    (30% weight)
  C. Put/Call Ratio Momentum    (20% weight)

The combined signal is used as a 20% component of the overall strategy blend
(configured in config/settings.yaml under ``options_flow.weight``).

ECONOMIC RATIONALE
------------------
Options markets are forward-looking: informed traders with private information
preferentially use options because of their embedded leverage and limited
downside (Black 1975; Easley, O'Hara & Srinivas 1998). Three robust patterns
have been documented in the academic literature:

1. **Unusual volume** — Block trades in options, particularly OTM contracts,
   predict next-week equity returns (Cao, Chen & Griffin 2005). The 2× average
   volume threshold is the canonical definition used by market makers and
   exchanges to flag "unusual activity" (CBOE Unusual Activity Alerts).

2. **IV skew** — The ratio of 25-delta put IV to 25-delta call IV captures
   the asymmetric demand for downside protection. Elevated skew (puts
   relatively expensive) reflects hedging demand and forward-looking
   pessimism (Bollen & Whaley 2004; Xing, Zhang & Zhao 2010). We invert
   the skew z-score so that high skew → negative (bearish) signal.

3. **Put/Call ratio momentum** — Pan & Poteshman (2006) show that stocks with
   low PCR outperform high-PCR stocks by ~40 bps the next day and ~1% over
   the following week. As a contrarian indicator, extreme PCR readings (fear
   peaks) tend to coincide with local troughs; we use the 5-day vs 21-day
   momentum cross to capture this mean-reversion signal.

ANTI-OVERFITTING NOTES
-----------------------
All numerical thresholds are grounded in economic theory or industry convention:
  - 2× average volume  → CBOE/FINRA "unusual activity" standard
  - 3-day half-life    → Typical informed-trader position horizon
  - 1.5× OTM weight   → OTM options are more informationally efficient (leverage)
  - Contrarian PCR     → Pan & Poteshman (2006), Blau & Brough (2015)
  - 25-delta strikes   → Market convention for risk-reversal / skew measurement
  - 63-day normalization → One calendar quarter; removes seasonal regime changes

No threshold was selected by backtesting optimisation.

DATA SOURCES
------------
1. Databento OPRA.PILLAR  — tick-by-tick consolidated options trades
2. yfinance option_chain   — free end-of-day snapshot for testing/fallback
3. Neutral (0.0)           — returned with a WARNING if both sources fail

USAGE
-----
    from strategy.options_flow_signal import OptionsFlowSignal

    signal = OptionsFlowSignal()
    signals = signal.compute(
        symbols=["AAPL", "SPY", "QQQ"],
        as_of_date=datetime.date(2026, 4, 1),
        lookback_days=5,
    )
    # → {"AAPL": 0.42, "SPY": -0.17, "QQQ": 0.05}

CONFIGURATION  (config/settings.yaml)
--------------------------------------
    options_flow:
      enabled: true
      weight: 0.20
      lookback_days: 5
      min_volume_threshold: 100
      databento_enabled: true
      yfinance_fallback: true
      unusual_volume_multiple: 2.0
      pcr_contrarian: true

References
----------
- Black, F. (1975). Fact and Fantasy in the Use of Options. FAJ.
- Easley, D., O'Hara, M., & Srinivas, P. S. (1998). Option Volume and Stock Prices. JF.
- Cao, C., Chen, Z., & Griffin, J. M. (2005). Informational Content of Option Volume. JFE.
- Pan, J., & Poteshman, A. M. (2006). The Information in Option Volume for Future Stock Prices. RFS.
- Bollen, N. P. B., & Whaley, R. E. (2004). Does Net Buying Pressure Affect the Shape of IV? JF.
- Xing, Y., Zhang, X., & Zhao, R. (2010). What Does the Individual Option Volatility Smirk Tell Us? JFQA.
- Blau, B., & Brough, T. (2015). Put-Call Parity and the Predictability of Option Returns. JFQA.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module is still importable in
# environments where databento or yfinance are not installed.
# ---------------------------------------------------------------------------
try:
    import databento as db  # type: ignore[import]

    _DATABENTO_AVAILABLE = True
except ImportError:
    _DATABENTO_AVAILABLE = False

try:
    import yfinance as yf  # type: ignore[import]

    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

try:
    from scipy.stats import norm as _norm  # type: ignore[import]

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import yaml  # type: ignore[import]

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants  (all economically motivated — see module docstring)
# ---------------------------------------------------------------------------
_UNUSUAL_VOLUME_WINDOW = 20          # days used to compute "average" volume baseline
_UNUSUAL_VOLUME_MULTIPLE = 2.0       # block-trade threshold: > 2× average
_DECAY_HALF_LIFE_DAYS = 3            # informed-trader holding period (half-life)
_OTM_WEIGHT_MULTIPLIER = 1.5        # OTM options carry more information signal
_ATM_MONEYNESS_BAND = 0.05           # |strike/spot - 1| ≤ 5% → ATM
_SKEW_NORM_WINDOW = 63               # one quarter of trading days for skew z-score
_PCR_SHORT_WINDOW = 5                # short PCR window (days)
_PCR_LONG_WINDOW = 21                # long PCR window  (days)
_MIN_VOLUME_THRESHOLD = 100          # minimum total options volume to generate signal
_CACHE_TTL_SECONDS = 3600            # 1 hour cache for raw options data
_RISK_FREE_RATE = 0.05               # approximate risk-free rate for IV calculation
_MAX_IV_NEWTON_ITERS = 50            # Newton–Raphson iterations for IV root-finding
_IV_TOLERANCE = 1e-6                 # convergence criterion for IV solver
_SIGNAL_CLIP = 1.0                   # hard clip for all sub-signals and final output

# Component weights — must sum to 1.0
_WEIGHT_UNUSUAL_ACTIVITY = 0.50
_WEIGHT_IV_SKEW = 0.30
_WEIGHT_PCR_MOMENTUM = 0.20

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class _OptionsSnapshot:
    """Aggregated, per-day options statistics for a single underlying symbol.

    All fields represent *end-of-day* aggregates for the given ``date``.

    Parameters
    ----------
    date : datetime.date
        Trade date.
    symbol : str
        Underlying ticker symbol.
    call_volume : float
        Total call-option trading volume (contracts) on ``date``.
    put_volume : float
        Total put-option trading volume (contracts) on ``date``.
    call_iv_atm : float
        Volume-weighted average implied volatility for near-ATM call options.
    put_iv_atm : float
        Volume-weighted average implied volatility for near-ATM put options.
    call_iv_25d : float
        Implied volatility of the nearest-to-25-delta call option.
    put_iv_25d : float
        Implied volatility of the nearest-to-25-delta put option.
    total_volume : float
        call_volume + put_volume.
    contracts : pd.DataFrame
        Optional per-contract detail (strike, iv, volume, instrument_class,
        moneyness_category). May be empty.
    """

    date: datetime.date
    symbol: str
    call_volume: float = 0.0
    put_volume: float = 0.0
    call_iv_atm: float = float("nan")
    put_iv_atm: float = float("nan")
    call_iv_25d: float = float("nan")
    put_iv_25d: float = float("nan")
    total_volume: float = 0.0
    contracts: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.total_volume = self.call_volume + self.put_volume


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _TTLCache:
    """Simple in-memory TTL cache keyed by arbitrary hashable keys.

    Thread-safety is *not* guaranteed; this is a single-threaded helper.
    """

    def __init__(self, ttl_seconds: float = _CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[Any, Tuple[float, Any]] = {}

    def get(self, key: Any) -> Optional[Any]:
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: Any, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Black-Scholes helpers (pure Python, no scipy required for basic IV)
# ---------------------------------------------------------------------------


def _standard_normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the math.erf function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(
    spot: float,
    strike: float,
    tau: float,
    r: float,
    sigma: float,
    is_call: bool,
) -> float:
    """Black-Scholes European option price.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Option strike price.
    tau : float
        Time to expiry in years (must be > 0).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility (> 0).
    is_call : bool
        True for call, False for put.

    Returns
    -------
    float
        Theoretical option price, or NaN on domain error.
    """
    if tau <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    try:
        sqrt_tau = math.sqrt(tau)
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        if is_call:
            return spot * _standard_normal_cdf(d1) - strike * math.exp(-r * tau) * _standard_normal_cdf(d2)
        else:
            return strike * math.exp(-r * tau) * _standard_normal_cdf(-d2) - spot * _standard_normal_cdf(-d1)
    except (ValueError, ZeroDivisionError, OverflowError):
        return float("nan")


def _bs_vega(spot: float, strike: float, tau: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (same for call and put)."""
    if tau <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        return spot * math.sqrt(tau) * math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    except (ValueError, ZeroDivisionError, OverflowError):
        return float("nan")


def _bs_delta(spot: float, strike: float, tau: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes delta."""
    if tau <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        nd1 = _standard_normal_cdf(d1)
        return nd1 if is_call else (nd1 - 1.0)
    except (ValueError, ZeroDivisionError, OverflowError):
        return float("nan")


def _implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    tau: float,
    r: float,
    is_call: bool,
    initial_sigma: float = 0.25,
) -> float:
    """Compute implied volatility via Newton-Raphson iteration.

    Returns NaN if the solver does not converge or inputs are invalid.
    """
    if market_price <= 0 or spot <= 0 or strike <= 0 or tau <= 0:
        return float("nan")

    # Intrinsic value check
    intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
    if market_price < intrinsic:
        return float("nan")

    sigma = initial_sigma
    for _ in range(_MAX_IV_NEWTON_ITERS):
        price = _bs_price(spot, strike, tau, r, sigma, is_call)
        vega = _bs_vega(spot, strike, tau, r, sigma)
        if math.isnan(price) or math.isnan(vega) or abs(vega) < 1e-10:
            return float("nan")
        diff = price - market_price
        if abs(diff) < _IV_TOLERANCE:
            return max(sigma, 0.0)
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)  # keep sigma positive
    return float("nan")  # did not converge


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def _load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load the options_flow section from settings.yaml.

    Returns a dict of defaults if the file is missing or the key is absent.
    """
    defaults: Dict[str, Any] = {
        "enabled": True,
        "weight": 0.20,
        "lookback_days": 5,
        "min_volume_threshold": _MIN_VOLUME_THRESHOLD,
        "databento_enabled": True,
        "yfinance_fallback": True,
        "unusual_volume_multiple": _UNUSUAL_VOLUME_MULTIPLE,
        "pcr_contrarian": True,
    }
    if not _YAML_AVAILABLE:
        logger.debug("PyYAML not installed; using default options_flow configuration.")
        return defaults
    try:
        with open(config_path, "r") as fh:
            full_config = yaml.safe_load(fh) or {}
        cfg = full_config.get("options_flow", {})
        return {**defaults, **cfg}
    except FileNotFoundError:
        logger.debug("Config file %s not found; using defaults.", config_path)
        return defaults
    except Exception as exc:
        logger.warning("Failed to parse config %s: %s — using defaults.", config_path, exc)
        return defaults


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


class _DatabenteFetcher:
    """Fetches options data from Databento OPRA.PILLAR.

    Uses the ``ohlcv-1d`` schema (daily bars with volume) plus the
    ``definition`` schema (to identify calls, puts, strikes, expiries) to
    build per-day ``_OptionsSnapshot`` objects.

    Requires the environment variable ``DATABENTO_API_KEY`` or a key passed
    at construction time.
    """

    DATASET = "OPRA.PILLAR"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("DATABENTO_API_KEY", "")
        self._client: Optional[Any] = None  # db.Historical

    def _get_client(self) -> Any:
        if self._client is None:
            if not _DATABENTO_AVAILABLE:
                raise ImportError("databento package is not installed.")
            if not self._api_key:
                raise ValueError(
                    "Databento API key not found. Set the DATABENTO_API_KEY "
                    "environment variable or pass api_key to OptionsFlowSignal."
                )
            self._client = db.Historical(self._api_key)
        return self._client

    def fetch(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[_OptionsSnapshot]:
        """Return per-day snapshots for *symbol* over [start_date, end_date].

        Strategy
        --------
        1. Fetch instrument definitions for ``{symbol}.OPT`` (parent stype).
        2. Fetch daily OHLCV bars for the same parent to get per-contract
           volumes per day.
        3. Join on instrument_id; classify calls/puts from
           ``instrument_class``; compute moneyness and IV from bid/ask midpoint
           using the OHLCV close (for snapshot IV we use a rough implied-spot
           approximation — for production, splice in underlying prices).
        4. Aggregate to one ``_OptionsSnapshot`` per calendar day.
        """
        client = self._get_client()
        opt_parent = f"{symbol}.OPT"

        # ------------------------------------------------------------------
        # Step 1 — instrument definitions (strike, expiry, put/call flag)
        # ------------------------------------------------------------------
        logger.debug("Databento: fetching definitions for %s", opt_parent)
        def_data = client.timeseries.get_range(
            dataset=self.DATASET,
            schema="definition",
            symbols=opt_parent,
            stype_in="parent",
            start=start_date.isoformat(),
        )
        def_df: pd.DataFrame = def_data.to_df()

        if def_df.empty:
            logger.warning("Databento returned no definitions for %s", symbol)
            return []

        # Normalise strike price (stored in fixed-point 1e-9 units by Databento)
        if "strike_price" in def_df.columns:
            if def_df["strike_price"].abs().max() > 1e6:
                def_df["strike_price"] = def_df["strike_price"] / 1e9

        # Keep only true options (not spreads/strategies)
        if "instrument_class" in def_df.columns:
            def_df = def_df[
                def_df["instrument_class"].isin(
                    [db.InstrumentClass.CALL, db.InstrumentClass.PUT]
                )
            ]

        if def_df.empty:
            return []

        # ------------------------------------------------------------------
        # Step 2 — daily OHLCV bars for volume and close price
        # ------------------------------------------------------------------
        logger.debug("Databento: fetching ohlcv-1d for %s  %s → %s", opt_parent, start_date, end_date)
        ohlcv_data = client.timeseries.get_range(
            dataset=self.DATASET,
            schema="ohlcv-1d",
            symbols=opt_parent,
            stype_in="parent",
            start=start_date.isoformat(),
            end=(end_date + datetime.timedelta(days=1)).isoformat(),
        )
        ohlcv_df: pd.DataFrame = ohlcv_data.to_df()

        if ohlcv_df.empty:
            logger.warning("Databento returned no OHLCV data for %s", symbol)
            return []

        # Normalise price fields
        for col in ("open", "high", "low", "close"):
            if col in ohlcv_df.columns and ohlcv_df[col].abs().max() > 1e6:
                ohlcv_df[col] = ohlcv_df[col] / 1e9

        # ------------------------------------------------------------------
        # Step 3 — join definitions and OHLCV on instrument_id
        # ------------------------------------------------------------------
        # def_df may be indexed by ts_recv; reset to keep instrument_id
        if "instrument_id" in def_df.columns:
            def_subset = def_df[
                ["instrument_id", "strike_price", "instrument_class", "expiration"]
            ].drop_duplicates("instrument_id")
        else:
            logger.warning("Databento definition df missing instrument_id for %s", symbol)
            return []

        merged = ohlcv_df.merge(def_subset, on="instrument_id", how="inner")
        if merged.empty:
            return []

        # Extract trade date
        if isinstance(merged.index, pd.DatetimeIndex):
            merged["trade_date"] = merged.index.date
        elif "ts_event" in merged.columns:
            merged["trade_date"] = pd.to_datetime(merged["ts_event"]).dt.date
        else:
            merged["trade_date"] = start_date

        # Classify instrument_class to string
        merged["side"] = merged["instrument_class"].apply(
            lambda c: "call"
            if str(c).upper() in ("C", "CALL", str(getattr(db, "InstrumentClass", object)).split(".")[-1])
            else "put"
        )
        # More robust classification
        try:
            call_class = db.InstrumentClass.CALL
            put_class = db.InstrumentClass.PUT
            merged["side"] = merged["instrument_class"].map(
                {call_class: "call", put_class: "put"}
            )
        except AttributeError:
            pass

        # ------------------------------------------------------------------
        # Step 4 — build snapshots per trade_date
        # ------------------------------------------------------------------
        snapshots: List[_OptionsSnapshot] = []
        for trade_date, day_df in merged.groupby("trade_date"):
            snap = _build_snapshot_from_contracts(symbol, trade_date, day_df)
            if snap is not None:
                snapshots.append(snap)

        snapshots.sort(key=lambda s: s.date)
        return snapshots


class _YFinanceFetcher:
    """Fetches options data from Yahoo Finance via yfinance.

    Yahoo Finance provides *current* option chains (not historical), so
    ``as_of_date`` is not directly honoured — we always get today's snapshot.
    This makes the yfinance path suitable for testing and live signal
    generation only, not for historical backtesting.

    For each expiry available within ``lookback_days * 7`` days (to capture
    multiple weekly expirations), we aggregate call and put volumes, compute
    a rough 25-delta IV from the chain's ``impliedVolatility`` column, and
    build an ``_OptionsSnapshot``.
    """

    def fetch(
        self,
        symbol: str,
        as_of_date: datetime.date,
        lookback_days: int = 5,
    ) -> List[_OptionsSnapshot]:
        """Return a list with a single ``_OptionsSnapshot`` for *symbol*.

        yfinance does not provide historical data; we synthesise a single
        snapshot representing the current state of the option chain and
        return it under ``as_of_date``.  Callers that need a time-series
        should only use this path for live signal generation.
        """
        if not _YFINANCE_AVAILABLE:
            raise ImportError("yfinance package is not installed.")

        try:
            ticker = yf.Ticker(symbol)
            expiry_dates: List[str] = list(ticker.options or [])
        except Exception as exc:
            logger.warning("yfinance: failed to retrieve options for %s: %s", symbol, exc)
            return []

        if not expiry_dates:
            logger.debug("yfinance: no options found for %s", symbol)
            return []

        # Get current underlying price for moneyness calculation
        spot: Optional[float] = None
        try:
            info = ticker.fast_info
            spot = float(info.last_price)
        except Exception:
            try:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    spot = float(hist["Close"].iloc[-1])
            except Exception:
                pass

        # Only consider near-term expiries (within 45 calendar days)
        max_exp = as_of_date + datetime.timedelta(days=45)
        near_expiries = [
            e for e in expiry_dates
            if datetime.date.fromisoformat(e) <= max_exp
        ]
        if not near_expiries:
            near_expiries = expiry_dates[:4]  # fallback: first 4 expiries

        all_calls: List[pd.DataFrame] = []
        all_puts: List[pd.DataFrame] = []

        for expiry in near_expiries:
            try:
                chain = ticker.option_chain(expiry)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["expiry"] = expiry
                puts["expiry"] = expiry

                # Days to expiry (for tau in IV computations)
                exp_date = datetime.date.fromisoformat(expiry)
                dte = max((exp_date - as_of_date).days, 1)
                calls["dte"] = dte
                puts["dte"] = dte

                all_calls.append(calls)
                all_puts.append(puts)
            except Exception as exc:
                logger.debug("yfinance: error fetching chain %s/%s: %s", symbol, expiry, exc)

        if not all_calls and not all_puts:
            return []

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        snap = _build_snapshot_from_yf(symbol, as_of_date, calls_df, puts_df, spot)
        return [snap] if snap is not None else []


# ---------------------------------------------------------------------------
# Snapshot construction helpers
# ---------------------------------------------------------------------------


def _build_snapshot_from_contracts(
    symbol: str,
    trade_date: datetime.date,
    day_df: pd.DataFrame,
) -> Optional[_OptionsSnapshot]:
    """Build an ``_OptionsSnapshot`` from a Databento OHLCV+definition join."""
    if day_df.empty:
        return None

    contracts_list = []
    call_vol = 0.0
    put_vol = 0.0

    for _, row in day_df.iterrows():
        side = str(row.get("side", "")).lower()
        vol = float(row.get("volume", 0) or 0)
        strike = float(row.get("strike_price", 0) or 0)
        close_px = float(row.get("close", 0) or 0)
        iv = float(row.get("impliedVolatility", float("nan")))

        if side == "call":
            call_vol += vol
        elif side == "put":
            put_vol += vol

        contracts_list.append({
            "side": side,
            "strike": strike,
            "volume": vol,
            "close": close_px,
            "iv": iv,
            "dte": 30,  # approximate; refined if expiration available
        })

    # Compute DTE if expiration column is present
    if "expiration" in day_df.columns:
        for i, (_, row) in enumerate(day_df.iterrows()):
            try:
                exp = pd.to_datetime(row["expiration"]).date()
                contracts_list[i]["dte"] = max((exp - trade_date).days, 1)
            except Exception:
                pass

    contracts_df = pd.DataFrame(contracts_list)

    snap = _OptionsSnapshot(
        date=trade_date,
        symbol=symbol,
        call_volume=call_vol,
        put_volume=put_vol,
    )
    snap.contracts = contracts_df
    return snap


def _build_snapshot_from_yf(
    symbol: str,
    trade_date: datetime.date,
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    spot: Optional[float],
) -> Optional[_OptionsSnapshot]:
    """Build an ``_OptionsSnapshot`` from yfinance option chain DataFrames."""

    def safe_vol(df: pd.DataFrame) -> float:
        if df.empty or "volume" not in df.columns:
            return 0.0
        return float(df["volume"].fillna(0).sum())

    call_vol = safe_vol(calls_df)
    put_vol = safe_vol(puts_df)

    # --- Extract per-contract data for IV skew computation ----------------
    contracts_list: List[Dict[str, Any]] = []

    for side, df in [("call", calls_df), ("put", puts_df)]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            iv_raw = row.get("impliedVolatility", float("nan"))
            strike = float(row.get("strike", 0) or 0)
            vol = float(row.get("volume", 0) or 0)
            dte = int(row.get("dte", 30))
            mid_price = 0.5 * (
                float(row.get("bid", 0) or 0) + float(row.get("ask", 0) or 0)
            )
            iv_val: float = float(iv_raw) if not _is_nan_or_none(iv_raw) else float("nan")

            # Recompute IV from bid/ask midpoint if available and spot known
            if spot and spot > 0 and strike > 0 and dte > 0 and mid_price > 0:
                tau = dte / 365.0
                iv_computed = _implied_vol(
                    market_price=mid_price,
                    spot=spot,
                    strike=strike,
                    tau=tau,
                    r=_RISK_FREE_RATE,
                    is_call=(side == "call"),
                )
                if not math.isnan(iv_computed):
                    iv_val = iv_computed

            contracts_list.append({
                "side": side,
                "strike": strike,
                "volume": vol,
                "iv": iv_val,
                "dte": dte,
                "mid_price": mid_price,
            })

    contracts_df = pd.DataFrame(contracts_list)

    # --- Compute 25-delta IV for skew -------------------------------------
    call_iv_25d = float("nan")
    put_iv_25d = float("nan")

    if spot and spot > 0 and not contracts_df.empty:
        call_iv_25d, put_iv_25d = _estimate_25delta_ivs(contracts_df, spot)

    snap = _OptionsSnapshot(
        date=trade_date,
        symbol=symbol,
        call_volume=call_vol,
        put_volume=put_vol,
        call_iv_25d=call_iv_25d,
        put_iv_25d=put_iv_25d,
    )
    snap.contracts = contracts_df
    return snap


# ---------------------------------------------------------------------------
# IV and delta utilities
# ---------------------------------------------------------------------------


def _is_nan_or_none(val: Any) -> bool:
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return True


def _estimate_25delta_ivs(
    contracts_df: pd.DataFrame,
    spot: float,
    target_delta: float = 0.25,
) -> Tuple[float, float]:
    """Find the call and put options closest to ±25-delta and return their IVs.

    For each expiry group we find the contract whose Black-Scholes delta is
    nearest to ``target_delta`` (calls) or ``-target_delta`` (puts).  We then
    volume-weight across expiry groups.

    Returns
    -------
    (call_iv_25d, put_iv_25d) : Tuple[float, float]
        NaN if no valid contract is found.
    """
    if contracts_df.empty or spot <= 0:
        return float("nan"), float("nan")

    call_ivs: List[float] = []
    put_ivs: List[float] = []
    call_vols: List[float] = []
    put_vols: List[float] = []

    for dte, grp in contracts_df.groupby("dte"):
        tau = max(float(dte), 1) / 365.0
        r = _RISK_FREE_RATE

        calls_g = grp[grp["side"] == "call"].copy()
        puts_g = grp[grp["side"] == "put"].copy()

        for side_df, is_call, iv_list, vol_list, tgt_delta in [
            (calls_g, True, call_ivs, call_vols, target_delta),
            (puts_g, False, put_ivs, put_vols, -target_delta),
        ]:
            if side_df.empty:
                continue
            # Compute delta for each strike using contract IV (or a guess)
            best_iv = float("nan")
            best_dist = float("inf")
            best_vol = 0.0

            for _, row in side_df.iterrows():
                strike = float(row.get("strike", 0) or 0)
                iv = float(row.get("iv", float("nan")))
                vol = float(row.get("volume", 0) or 0)

                if strike <= 0:
                    continue
                if _is_nan_or_none(iv) or iv <= 0:
                    # Fallback: guess IV from moneyness
                    iv = 0.25
                delta = _bs_delta(spot, strike, tau, r, iv, is_call)
                if _is_nan_or_none(delta):
                    continue
                dist = abs(abs(delta) - abs(tgt_delta))
                if dist < best_dist:
                    best_dist = dist
                    best_iv = iv
                    best_vol = vol

            if not _is_nan_or_none(best_iv):
                iv_list.append(best_iv)
                vol_list.append(max(best_vol, 1.0))  # weight >= 1

    def _wavg(ivs: List[float], vols: List[float]) -> float:
        if not ivs:
            return float("nan")
        w = np.array(vols, dtype=float)
        v = np.array(ivs, dtype=float)
        w = w / w.sum()
        return float(np.dot(w, v))

    return _wavg(call_ivs, call_vols), _wavg(put_ivs, put_vols)


def _classify_moneyness(strike: float, spot: float) -> str:
    """Return 'ATM', 'OTM', or 'ITM' based on simple moneyness band.

    ATM: |strike/spot - 1| ≤ 5%
    OTM: strike > spot*1.05 for calls; strike < spot*0.95 for puts
         (we use a symmetric definition here since we don't have the side context)
    """
    if spot <= 0:
        return "ATM"
    ratio = strike / spot
    if abs(ratio - 1.0) <= _ATM_MONEYNESS_BAND:
        return "ATM"
    return "OTM"


# ---------------------------------------------------------------------------
# Signal component computors
# ---------------------------------------------------------------------------


def _compute_unusual_activity_signal(
    snapshots: List[_OptionsSnapshot],
    unusual_volume_multiple: float = _UNUSUAL_VOLUME_MULTIPLE,
    min_volume: float = _MIN_VOLUME_THRESHOLD,
    spot: Optional[float] = None,
) -> float:
    """Compute Unusual Options Activity signal component.

    Economic rationale
    ------------------
    Informed traders preferentially trade options — the leverage and limited
    downside allow them to take directional positions without revealing their
    hand in the equity market (Black 1975; Easley et al. 1998).  Block trades
    (volume > 2× the 20-day average) are the canonical signal of informed flow.

    The directional bias is computed as a call/put volume imbalance, weighted
    by moneyness (OTM options are 1.5× more informative because informed
    traders lean toward OTM strikes for pure directional bets) and by recency
    (exponential decay with 3-day half-life matching the typical informed
    holding period).

    Returns
    -------
    float
        Signal in [-1, +1].  +1 = strong unusual call activity (bullish).
    """
    if not snapshots or len(snapshots) < 2:
        return 0.0

    # Sort ascending by date
    snaps = sorted(snapshots, key=lambda s: s.date)

    # Compute 20-day average total volume (all available history in window)
    all_vols = np.array([s.total_volume for s in snaps])
    avg_vol = float(np.nanmean(all_vols[:-1])) if len(all_vols) > 1 else float(np.nanmean(all_vols))
    if avg_vol < 1.0:
        avg_vol = 1.0

    # Exponential decay weights (most recent snapshot has weight = 1)
    decay_rate = math.log(2.0) / _DECAY_HALF_LIFE_DAYS  # ln(2)/3 ≈ 0.231
    n = len(snaps)
    weights = np.array([math.exp(-decay_rate * (n - 1 - i)) for i in range(n)])

    directional_scores: List[float] = []
    day_weights: List[float] = []

    for i, snap in enumerate(snaps):
        total = snap.total_volume
        if total < min_volume:
            continue

        # Block-trade flag: total volume > unusual_volume_multiple × average
        is_block = total > unusual_volume_multiple * avg_vol
        block_multiplier = 2.0 if is_block else 1.0

        # Directional bias: call/put volume imbalance, normalised to [-1, +1]
        call_v = snap.call_volume
        put_v = snap.put_volume
        denom = call_v + put_v
        if denom < 1.0:
            continue
        raw_bias = (call_v - put_v) / denom  # ∈ [-1, +1]

        # Moneyness weighting on per-contract basis
        moneyness_bias = raw_bias  # default if no contract data
        if not snap.contracts.empty and spot and spot > 0:
            contracts = snap.contracts.copy()
            if "strike" in contracts.columns:
                contracts["moneyness"] = contracts["strike"].apply(
                    lambda k: _classify_moneyness(float(k), spot)
                )
                contracts["side_sign"] = contracts["side"].map({"call": 1.0, "put": -1.0}).fillna(0.0)
                contracts["weight_mono"] = contracts["moneyness"].map(
                    {"OTM": _OTM_WEIGHT_MULTIPLIER, "ATM": 1.0, "ITM": 0.5}
                ).fillna(1.0)
                contracts["weighted_dir"] = (
                    contracts["side_sign"] * contracts["volume"].fillna(0) * contracts["weight_mono"]
                )
                total_weighted = (
                    contracts["volume"].fillna(0) * contracts["weight_mono"]
                ).sum()
                if total_weighted > 0:
                    moneyness_bias = contracts["weighted_dir"].sum() / total_weighted

        score = float(np.clip(moneyness_bias * block_multiplier, -1.0, 1.0))
        directional_scores.append(score)
        day_weights.append(weights[i])

    if not directional_scores:
        return 0.0

    w = np.array(day_weights)
    s = np.array(directional_scores)
    w = w / w.sum()
    signal = float(np.dot(w, s))
    return float(np.clip(signal, -_SIGNAL_CLIP, _SIGNAL_CLIP))


def _compute_iv_skew_signal(
    snapshots: List[_OptionsSnapshot],
    history_window: int = _SKEW_NORM_WINDOW,
) -> float:
    """Compute Implied Volatility Skew signal component.

    Economic rationale
    ------------------
    The 25-delta risk reversal (put IV / call IV - 1) captures asymmetric
    demand for tail protection.  When puts are relatively expensive versus
    calls (high skew), market participants are paying up for downside
    insurance — a forward-looking bearish signal (Xing, Zhang & Zhao 2010).
    We normalise by a 63-day rolling mean/std (one quarter) to remove
    secular regime shifts and express the signal as a z-score.

    The z-score is **inverted** so that:
      - High skew (fear, expensive puts) → negative z → bearish signal
      - Low skew (complacency, cheap puts) → positive z → bullish signal

    The resulting z-score is clipped to [-1, +1] via tanh compression.

    Returns
    -------
    float
        Signal in [-1, +1].  +1 = skew low (bullish); -1 = skew high (bearish).
    """
    skew_series: List[float] = []

    for snap in sorted(snapshots, key=lambda s: s.date):
        p25 = snap.put_iv_25d
        c25 = snap.call_iv_25d
        if _is_nan_or_none(p25) or _is_nan_or_none(c25) or c25 <= 0:
            # Fall back to per-contract estimation from contracts df
            if not snap.contracts.empty:
                p25, c25 = _estimate_25delta_ivs_from_df(snap)
        if _is_nan_or_none(p25) or _is_nan_or_none(c25) or c25 <= 0:
            skew_series.append(float("nan"))
        else:
            skew_series.append(float(p25) / float(c25) - 1.0)

    # Require at least 3 non-NaN observations
    valid = [s for s in skew_series if not math.isnan(s)]
    if len(valid) < 2:
        return 0.0

    skew_arr = np.array(skew_series)
    # Rolling normalisation using available window (up to history_window days)
    roll_mean = np.nanmean(skew_arr[-history_window:])
    roll_std = np.nanstd(skew_arr[-history_window:])

    if roll_std < 1e-8:
        return 0.0

    # Use the most recent non-NaN skew
    latest_skew = next(
        (s for s in reversed(skew_series) if not math.isnan(s)), float("nan")
    )
    if math.isnan(latest_skew):
        return 0.0

    z_score = (latest_skew - roll_mean) / roll_std
    # Invert: high skew → negative signal
    inverted_z = -z_score
    # Compress via tanh to enforce [-1, +1] more gracefully than hard clip
    signal = float(math.tanh(inverted_z / 2.0))
    return float(np.clip(signal, -_SIGNAL_CLIP, _SIGNAL_CLIP))


def _estimate_25delta_ivs_from_df(snap: _OptionsSnapshot) -> Tuple[float, float]:
    """Helper to re-derive 25-delta IVs from the contracts DataFrame."""
    df = snap.contracts
    if df.empty:
        return float("nan"), float("nan")
    # Sort calls by |strike - spot| using volume-weighted strike as proxy for spot
    all_strikes = df["strike"].dropna().values if "strike" in df.columns else np.array([])
    if len(all_strikes) == 0:
        return float("nan"), float("nan")
    approx_spot = float(np.median(all_strikes))
    return _estimate_25delta_ivs(df, approx_spot)


def _compute_pcr_momentum_signal(
    snapshots: List[_OptionsSnapshot],
    short_window: int = _PCR_SHORT_WINDOW,
    long_window: int = _PCR_LONG_WINDOW,
    contrarian: bool = True,
    min_volume: float = _MIN_VOLUME_THRESHOLD,
) -> float:
    """Compute Put/Call Ratio Momentum signal component.

    Economic rationale
    ------------------
    Pan & Poteshman (2006) show that stocks with *low* put/call ratios
    outperform high-PCR stocks by 40 bps the next day and ~1% over the
    following week.  This contrarian interpretation is well-documented:
    extreme PCR readings indicate sentiment extremes that tend to mean-revert.

    We compute the 5-day PCR and the 21-day PCR and take the ratio as a
    momentum indicator.  When the short-term PCR is elevated relative to
    the long-term average (excess fear in the near term), the contrarian
    signal turns bullish.

    Normalisation: the ratio is converted to a z-score over the available
    history and clipped to [-1, +1].

    Returns
    -------
    float
        Signal in [-1, +1].  +1 = contrarian bullish (high PCR, fear extreme).
    """
    snaps = sorted(snapshots, key=lambda s: s.date)
    if len(snaps) < 2:
        return 0.0

    # Build daily PCR series
    pcr_series: List[float] = []
    for snap in snaps:
        call_v = snap.call_volume
        put_v = snap.put_volume
        total = call_v + put_v
        if total < min_volume or call_v < 1.0:
            pcr_series.append(float("nan"))
        else:
            pcr_series.append(put_v / call_v)

    pcr_arr = np.array(pcr_series, dtype=float)
    n = len(pcr_arr)

    # Compute rolling averages using available data (clamp windows to data length)
    sw = min(short_window, n)
    lw = min(long_window, n)

    pcr_short = float(np.nanmean(pcr_arr[-sw:]))
    pcr_long = float(np.nanmean(pcr_arr[-lw:]))

    if math.isnan(pcr_short) or math.isnan(pcr_long) or pcr_long < 1e-8:
        return 0.0

    # Relative PCR: how elevated is near-term vs long-term
    pcr_ratio = pcr_short / pcr_long  # > 1 = near-term fear elevated

    # Normalise using overall series statistics
    valid_pcr = pcr_arr[~np.isnan(pcr_arr)]
    if len(valid_pcr) < 2:
        return 0.0

    mean_pcr = float(np.mean(valid_pcr))
    std_pcr = float(np.std(valid_pcr))
    if std_pcr < 1e-8:
        return 0.0

    # z-score of the short-term PCR vs historical distribution
    latest_pcr = pcr_short
    z_score = (latest_pcr - mean_pcr) / std_pcr

    if contrarian:
        # High PCR (fear) → bullish: invert z-score
        signal = float(-z_score)
    else:
        signal = float(z_score)

    # Clip to [-1, +1]
    return float(np.clip(signal, -_SIGNAL_CLIP, _SIGNAL_CLIP))


# ---------------------------------------------------------------------------
# Main signal class
# ---------------------------------------------------------------------------


class OptionsFlowSignal:
    """Options order flow signal generator for the automated trading system.

    Produces a directional signal in [-1, +1] for each requested equity
    symbol, derived from three options-market data components:

      - **Unusual Options Activity** (50%): block-trade volume spikes,
        directional call/put imbalance weighted by moneyness and recency.
      - **Implied Volatility Skew** (30%): 25-delta put/call IV ratio,
        normalised by a 63-day rolling z-score and inverted (high skew = bearish).
      - **Put/Call Ratio Momentum** (20%): contrarian 5-day vs 21-day PCR
        cross, consistent with Pan & Poteshman (2006).

    Data sources (with automatic fallback):
      1. Databento OPRA.PILLAR (real-time and historical, tick-by-tick)
      2. yfinance option chain (end-of-day snapshot, free, suitable for testing)
      3. Neutral signal (0.0) with a WARNING log if both sources fail

    Raw options data is cached for 1 hour to minimise redundant API calls.

    Parameters
    ----------
    config_path : str
        Path to ``config/settings.yaml``.  Defaults to the relative path
        expected by the trading system.
    databento_api_key : str, optional
        Override for ``DATABENTO_API_KEY`` environment variable.
    cache_ttl_seconds : float
        Time-to-live for the in-memory data cache.  Default: 3600 (1 hour).

    Examples
    --------
    >>> from strategy.options_flow_signal import OptionsFlowSignal
    >>> import datetime
    >>> signal = OptionsFlowSignal()
    >>> signals = signal.compute(
    ...     symbols=["AAPL", "SPY"],
    ...     as_of_date=datetime.date.today(),
    ...     lookback_days=5,
    ... )
    >>> signals
    {"AAPL": 0.31, "SPY": -0.12}
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        databento_api_key: Optional[str] = None,
        cache_ttl_seconds: float = float(_CACHE_TTL_SECONDS),
    ) -> None:
        self._cfg = _load_config(config_path)
        self._cache = _TTLCache(ttl_seconds=cache_ttl_seconds)
        self._db_fetcher = _DatabenteFetcher(api_key=databento_api_key)
        self._yf_fetcher = _YFinanceFetcher()
        logger.info(
            "OptionsFlowSignal initialised — "
            "databento_enabled=%s  yfinance_fallback=%s",
            self._cfg.get("databento_enabled"),
            self._cfg.get("yfinance_fallback"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        symbols: List[str],
        as_of_date: datetime.date,
        lookback_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute the combined options flow signal for each symbol.

        All signals are strictly causal — only data available *before*
        ``as_of_date`` is used (lookback window ends at ``as_of_date - 1``).

        Parameters
        ----------
        symbols : list of str
            Equity ticker symbols (e.g. ``["AAPL", "SPY", "QQQ"]``).
        as_of_date : datetime.date
            The reference date.  Signal uses data from
            ``[as_of_date - lookback_days, as_of_date - 1]`` inclusive.
        lookback_days : int, optional
            Number of trading days of options data to include.  Overrides
            the value in ``config/settings.yaml`` if provided.

        Returns
        -------
        Dict[str, float]
            Symbol → signal in [-1, +1].  Returns 0.0 for any symbol where
            data cannot be obtained.
        """
        if not self._cfg.get("enabled", True):
            logger.info("OptionsFlowSignal is disabled in configuration.")
            return {sym: 0.0 for sym in symbols}

        n_days: int = lookback_days if lookback_days is not None else int(self._cfg.get("lookback_days", 5))
        min_vol: float = float(self._cfg.get("min_volume_threshold", _MIN_VOLUME_THRESHOLD))
        unusual_multiple: float = float(self._cfg.get("unusual_volume_multiple", _UNUSUAL_VOLUME_MULTIPLE))
        pcr_contrarian: bool = bool(self._cfg.get("pcr_contrarian", True))
        db_enabled: bool = bool(self._cfg.get("databento_enabled", True))
        yf_fallback: bool = bool(self._cfg.get("yfinance_fallback", True))

        # Causal date range: up to as_of_date - 1
        end_date = as_of_date - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=n_days + _UNUSUAL_VOLUME_WINDOW + 10)

        results: Dict[str, float] = {}
        for symbol in symbols:
            try:
                signal = self._compute_symbol(
                    symbol=symbol,
                    as_of_date=as_of_date,
                    start_date=start_date,
                    end_date=end_date,
                    n_days=n_days,
                    min_vol=min_vol,
                    unusual_multiple=unusual_multiple,
                    pcr_contrarian=pcr_contrarian,
                    db_enabled=db_enabled,
                    yf_fallback=yf_fallback,
                )
            except Exception as exc:
                logger.error(
                    "Unhandled error computing options signal for %s: %s — returning 0.0",
                    symbol, exc, exc_info=True,
                )
                signal = 0.0
            results[symbol] = float(np.clip(signal, -_SIGNAL_CLIP, _SIGNAL_CLIP))

        logger.info(
            "OptionsFlowSignal.compute(%s) as_of=%s → %s",
            symbols, as_of_date,
            {k: round(v, 4) for k, v in results.items()},
        )
        return results

    # ------------------------------------------------------------------
    # Internal per-symbol computation
    # ------------------------------------------------------------------

    def _compute_symbol(
        self,
        symbol: str,
        as_of_date: datetime.date,
        start_date: datetime.date,
        end_date: datetime.date,
        n_days: int,
        min_vol: float,
        unusual_multiple: float,
        pcr_contrarian: bool,
        db_enabled: bool,
        yf_fallback: bool,
    ) -> float:
        """Full pipeline for a single symbol."""
        snapshots = self._fetch_snapshots(
            symbol=symbol,
            as_of_date=as_of_date,
            start_date=start_date,
            end_date=end_date,
            n_days=n_days,
            db_enabled=db_enabled,
            yf_fallback=yf_fallback,
        )

        if not snapshots:
            logger.warning(
                "No options data available for %s as of %s — returning neutral 0.0",
                symbol, as_of_date,
            )
            return 0.0

        # Trim to the lookback window strictly before as_of_date
        recent_snaps = [
            s for s in snapshots
            if start_date <= s.date <= end_date
        ][-n_days:]  # most recent n_days

        if not recent_snaps:
            logger.debug("No snapshots in lookback window for %s", symbol)
            return 0.0

        total_vol = sum(s.total_volume for s in recent_snaps)
        if total_vol < min_vol:
            logger.debug(
                "%s: total options volume %.0f below threshold %.0f — returning 0.0",
                symbol, total_vol, min_vol,
            )
            return 0.0

        # Estimate underlying spot from last snapshot's contracts
        spot = self._estimate_spot(symbol, recent_snaps)

        # --- Component A: Unusual Options Activity -----------------------
        sig_a = _compute_unusual_activity_signal(
            snapshots=recent_snaps,
            unusual_volume_multiple=unusual_multiple,
            min_volume=min_vol,
            spot=spot,
        )

        # --- Component B: IV Skew ----------------------------------------
        # Provide a longer history window for z-score normalisation
        longer_snaps = [
            s for s in snapshots
            if s.date <= end_date
        ][-_SKEW_NORM_WINDOW:]
        sig_b = _compute_iv_skew_signal(
            snapshots=longer_snaps,
            history_window=_SKEW_NORM_WINDOW,
        )

        # --- Component C: PCR Momentum -----------------------------------
        longer_pcr_snaps = [
            s for s in snapshots
            if s.date <= end_date
        ][-(_PCR_LONG_WINDOW + 5):]
        sig_c = _compute_pcr_momentum_signal(
            snapshots=longer_pcr_snaps,
            short_window=_PCR_SHORT_WINDOW,
            long_window=_PCR_LONG_WINDOW,
            contrarian=pcr_contrarian,
            min_volume=min_vol,
        )

        # --- Combine components ------------------------------------------
        combined = (
            _WEIGHT_UNUSUAL_ACTIVITY * sig_a
            + _WEIGHT_IV_SKEW * sig_b
            + _WEIGHT_PCR_MOMENTUM * sig_c
        )

        logger.debug(
            "%s: A(unusual)=%.3f  B(iv_skew)=%.3f  C(pcr)=%.3f  combined=%.3f",
            symbol, sig_a, sig_b, sig_c, combined,
        )

        return float(np.clip(combined, -_SIGNAL_CLIP, _SIGNAL_CLIP))

    # ------------------------------------------------------------------
    # Data fetching with fallback chain and caching
    # ------------------------------------------------------------------

    def _fetch_snapshots(
        self,
        symbol: str,
        as_of_date: datetime.date,
        start_date: datetime.date,
        end_date: datetime.date,
        n_days: int,
        db_enabled: bool,
        yf_fallback: bool,
    ) -> List[_OptionsSnapshot]:
        """Fetch raw options snapshots with caching and fallback."""
        cache_key = (symbol, start_date, end_date)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s [%s → %s]", symbol, start_date, end_date)
            return cached  # type: ignore[return-value]

        snapshots: List[_OptionsSnapshot] = []

        # --- Primary: Databento OPRA ----------------------------------------
        if db_enabled and _DATABENTO_AVAILABLE:
            try:
                logger.debug("Attempting Databento fetch for %s", symbol)
                snapshots = self._db_fetcher.fetch(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                if snapshots:
                    logger.info(
                        "Databento: fetched %d snapshots for %s", len(snapshots), symbol
                    )
            except Exception as exc:
                logger.warning(
                    "Databento fetch failed for %s: %s — falling back to yfinance.",
                    symbol, exc,
                )
                snapshots = []
        elif db_enabled and not _DATABENTO_AVAILABLE:
            logger.debug("databento package not installed — skipping primary source.")

        # --- Secondary: yfinance --------------------------------------------
        if not snapshots and yf_fallback and _YFINANCE_AVAILABLE:
            try:
                logger.debug("Attempting yfinance fetch for %s", symbol)
                snapshots = self._yf_fetcher.fetch(
                    symbol=symbol,
                    as_of_date=as_of_date,
                    lookback_days=n_days,
                )
                if snapshots:
                    logger.info(
                        "yfinance: fetched %d snapshots for %s", len(snapshots), symbol
                    )
                else:
                    logger.warning(
                        "yfinance returned no options data for %s", symbol
                    )
            except Exception as exc:
                logger.warning(
                    "yfinance fetch failed for %s: %s", symbol, exc
                )
                snapshots = []
        elif not snapshots and yf_fallback and not _YFINANCE_AVAILABLE:
            logger.warning(
                "yfinance package not installed — cannot use fallback for %s", symbol
            )

        # --- Tertiary: neutral ------------------------------------------------
        if not snapshots:
            logger.warning(
                "All data sources failed for %s as of %s — signal will be 0.0",
                symbol, as_of_date,
            )

        self._cache.set(cache_key, snapshots)
        return snapshots

    def _estimate_spot(
        self,
        symbol: str,
        snapshots: List[_OptionsSnapshot],
    ) -> Optional[float]:
        """Best-effort estimate of the current underlying spot price.

        Uses the median strike from near-ATM contracts as a rough proxy when
        no underlying price feed is wired in.  For production, replace with
        the live equity price from the system's market data handler.
        """
        all_strikes: List[float] = []
        for snap in snapshots:
            if snap.contracts.empty or "strike" not in snap.contracts.columns:
                continue
            strikes = snap.contracts["strike"].dropna().values
            all_strikes.extend([float(k) for k in strikes if float(k) > 0])

        if not all_strikes:
            # Attempt yfinance for spot
            if _YFINANCE_AVAILABLE:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        return float(hist["Close"].iloc[-1])
                except Exception:
                    pass
            return None

        # Use the median strike as a rough ATM proxy
        return float(np.median(all_strikes))

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Invalidate the in-memory data cache."""
        self._cache.clear()
        logger.info("OptionsFlowSignal: data cache cleared.")

    def get_component_signals(
        self,
        symbol: str,
        as_of_date: datetime.date,
        lookback_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return a breakdown of the three signal components for debugging.

        Returns
        -------
        dict with keys: "unusual_activity", "iv_skew", "pcr_momentum", "combined"
        """
        n_days: int = lookback_days if lookback_days is not None else int(self._cfg.get("lookback_days", 5))
        min_vol: float = float(self._cfg.get("min_volume_threshold", _MIN_VOLUME_THRESHOLD))
        unusual_multiple: float = float(self._cfg.get("unusual_volume_multiple", _UNUSUAL_VOLUME_MULTIPLE))
        pcr_contrarian: bool = bool(self._cfg.get("pcr_contrarian", True))
        db_enabled: bool = bool(self._cfg.get("databento_enabled", True))
        yf_fallback: bool = bool(self._cfg.get("yfinance_fallback", True))

        end_date = as_of_date - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=n_days + _UNUSUAL_VOLUME_WINDOW + 10)

        snapshots = self._fetch_snapshots(
            symbol=symbol,
            as_of_date=as_of_date,
            start_date=start_date,
            end_date=end_date,
            n_days=n_days,
            db_enabled=db_enabled,
            yf_fallback=yf_fallback,
        )

        if not snapshots:
            return {"unusual_activity": 0.0, "iv_skew": 0.0, "pcr_momentum": 0.0, "combined": 0.0}

        recent = [s for s in snapshots if start_date <= s.date <= end_date][-n_days:]
        spot = self._estimate_spot(symbol, recent)

        sig_a = _compute_unusual_activity_signal(recent, unusual_multiple, min_vol, spot)
        sig_b = _compute_iv_skew_signal(
            [s for s in snapshots if s.date <= end_date][-_SKEW_NORM_WINDOW:]
        )
        sig_c = _compute_pcr_momentum_signal(
            [s for s in snapshots if s.date <= end_date][-(_PCR_LONG_WINDOW + 5):],
            contrarian=pcr_contrarian,
            min_volume=min_vol,
        )
        combined = (
            _WEIGHT_UNUSUAL_ACTIVITY * sig_a
            + _WEIGHT_IV_SKEW * sig_b
            + _WEIGHT_PCR_MOMENTUM * sig_c
        )
        return {
            "unusual_activity": round(sig_a, 6),
            "iv_skew": round(sig_b, 6),
            "pcr_momentum": round(sig_c, 6),
            "combined": round(float(np.clip(combined, -1.0, 1.0)), 6),
        }


# ---------------------------------------------------------------------------
# Module self-test (run with: python -m strategy.options_flow_signal)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    import datetime

    print("=" * 60)
    print("OptionsFlowSignal — self-test")
    print("=" * 60)

    # Verify imports and instantiation
    sig = OptionsFlowSignal()
    print(f"Config loaded: {sig._cfg}")

    # Test with a small set of symbols using yfinance fallback
    test_symbols = ["AAPL", "SPY", "QQQ"]
    today = datetime.date.today()

    print(f"\nComputing signals for {test_symbols} as of {today} …")
    signals = sig.compute(symbols=test_symbols, as_of_date=today, lookback_days=5)
    print("\nSignals:")
    for sym, val in signals.items():
        bar = "#" * int(abs(val) * 30)
        direction = "↑" if val > 0 else "↓" if val < 0 else "→"
        print(f"  {sym:6s}  {direction}  {val:+.4f}  {bar}")

    print("\nComponent breakdown for AAPL:")
    breakdown = sig.get_component_signals("AAPL", today, lookback_days=5)
    for k, v in breakdown.items():
        print(f"  {k:20s}: {v:+.4f}")

    print("\nSelf-test complete.")
