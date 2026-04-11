"""
Momentum Factor IC Analysis
============================
Computes the Information Coefficient (IC = Spearman rank correlation)
between each scanner_v2 factor and next-day returns for 20 NASDAQ stocks.

OOS period: 2023-01-01 → 2026-04-01   (walk-forward, 4 windows)
Anti-overfitting: no IS fitting, pure OOS measurement.

Results are cached in .cache/computed/ — re-running with the same
price data and date range returns immediately from cache.

Usage:
    PYTHONPATH=. python diagnostics/momentum_factor_analysis.py

    # Force recompute (bust cache):
    PYTHONPATH=. python diagnostics/momentum_factor_analysis.py --force
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Allow imports from repo root
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from scipy import stats

from momentum_signals_exploration.data_store import load_result, save_result
from momentum_signals_exploration.price_cache import PriceCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AVGO",
    "JPM",
    "V",
    "MA",
    "UNH",
    "JNJ",
    "PG",
    "HD",
    "KO",
    "XOM",
    "CVX",
    "BAC",
    "GS",
]
OOS_START = "2023-01-01"
OOS_END = "2026-04-01"
WINDOWS = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-04-01"),
]
CACHE_KEY = f"momentum_factor_ic_{OOS_START}_{OOS_END}_{len(SYMS)}syms"
MAX_AGE = 14  # days before recomputing

FORCE = "--force" in sys.argv


# ---------------------------------------------------------------------------
# Factor computation (mirrors scanner_v2 SignalEngine factors)
# ---------------------------------------------------------------------------


def _safe_ic(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(a[mask], b[mask])
    return float(r), float(p)


def compute_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily cross-sectional factor scores for all symbols.
    Input: MultiIndex (symbol, date) with [open, high, low, close, volume]
    Output: flat DataFrame with columns [symbol, date, f1..f4, fwd_1d_ret]
    """
    rows = []
    syms = prices.index.get_level_values("symbol").unique().tolist()

    for sym in syms:
        try:
            df = prices.xs(sym, level="symbol").sort_index()
            close = df["close"].astype(float)
            volume = df["volume"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            # F1 — VWAP deviation (daily OHLC proxy — known to have IC -0.020)
            typical = (high + low + close) / 3
            vwap_day = (typical * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
            f1_vwap_dev = (close - vwap_day) / (vwap_day + 1e-9)

            # F2 — Relative strength vs equal-weight universe
            # Computed after loop using cross-sectional average

            # F3 — Volume surprise (log ratio vs 20-day avg)
            vol_avg = volume.rolling(20).mean()
            f3_vol = np.log(volume / (vol_avg + 1e-9))

            # F4 — Imbalance proxy (close vs prev close direction)
            f4_imbal = close.diff()

            # Forward 1-day return (what we're trying to predict)
            fwd_ret = close.pct_change(1).shift(-1)

            for dt in df.index:
                rows.append(
                    {
                        "symbol": sym,
                        "date": dt,
                        "f1_vwap_dev": float(f1_vwap_dev.get(dt, np.nan)),
                        "f3_vol_surp": float(f3_vol.get(dt, np.nan)),
                        "f4_imbal": float(f4_imbal.get(dt, np.nan)),
                        "raw_close": float(close.get(dt, np.nan)),
                        "fwd_1d_ret": float(fwd_ret.get(dt, np.nan)),
                    }
                )
        except Exception as e:
            logger.warning(f"  Factor error {sym}: {e}")

    panel = pd.DataFrame(rows).dropna(subset=["fwd_1d_ret"])

    # F2 — Relative strength: cross-sectional rank of 20-day return
    panel["ret_20d"] = panel.groupby("symbol")["raw_close"].transform(lambda s: s.pct_change(20))
    panel["f2_rel_strength"] = panel.groupby("date")["ret_20d"].rank(pct=True) - 0.5

    # Composite (equal weight, as original scanner_v2)
    for f in ["f1_vwap_dev", "f2_rel_strength", "f3_vol_surp", "f4_imbal"]:
        panel[f"{f}_z"] = panel.groupby("date")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9)
        )
    panel["composite"] = (
        panel["f1_vwap_dev_z"] * 0.30
        + panel["f2_rel_strength_z"] * 0.25
        + panel["f3_vol_surp_z"] * 0.20
        + panel["f4_imbal_z"] * 0.25
    )

    return panel


def run_ic_analysis(panel: pd.DataFrame) -> dict:
    """Compute overall and walk-forward ICs."""
    factors = {
        "F1 VWAP Dev (daily)": "f1_vwap_dev",
        "F2 Relative Strength": "f2_rel_strength",
        "F3 Volume Surprise": "f3_vol_surp",
        "F4 Imbalance Proxy": "f4_imbal",
        "Composite": "composite",
    }

    results = {}

    print("\n" + "=" * 65)
    print("  MOMENTUM FACTOR IC ANALYSIS")
    print(f"  OOS period: {OOS_START} → {OOS_END}  |  {len(SYMS)} symbols")
    print("=" * 65)
    print(f"\n  {'Factor':<30} {'IC':>8} {'p-value':>10}  Verdict")
    print("  " + "─" * 62)

    for label, col in factors.items():
        ic, p = _safe_ic(panel[col], panel["fwd_1d_ret"])
        sig = "✅" if p < 0.05 else ("⚠️" if p < 0.10 else "❌")
        print(f"  {label:<30} {ic:>+8.3f} {p:>10.3f}  {sig}")
        results[label] = {"ic": ic, "p": p, "col": col}

    # Walk-forward
    print(f"\n  {'Window':<10} ", end="")
    for label in factors:
        print(f"{label[:14]:>16}", end="")
    print()
    print("  " + "─" * 90)

    wf_rows = []
    for win_label, win_start, win_end in WINDOWS:
        mask = (panel["date"] >= win_start) & (panel["date"] <= win_end)
        sub = panel.loc[mask]
        row = {"window": win_label}
        print(f"  {win_label:<10} ", end="")
        for label, col in factors.items():
            ic, _ = _safe_ic(sub[col], sub["fwd_1d_ret"])
            row[label] = ic
            print(f"{ic:>+16.3f}", end="")
        print()
        wf_rows.append(row)

    print("=" * 65 + "\n")
    results["walk_forward"] = wf_rows
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Try cache first
    if not FORCE:
        cached = load_result(CACHE_KEY, max_age_days=MAX_AGE)
        if cached is not None:
            print(f"\n[DataStore] Loaded IC results from cache (key: {CACHE_KEY})")
            print("  Run with --force to recompute.\n")
            _print_cached(cached)
            return

    logger.info("Fetching daily price data via PriceCache...")
    cache = PriceCache()
    prices = cache.get_daily(SYMS, start=OOS_START, end=OOS_END)

    if prices is None or prices.empty:
        logger.error("No price data available — cannot compute IC")
        sys.exit(1)

    logger.info(
        f"Price data: {len(prices)} rows across {prices.index.get_level_values(0).nunique()} symbols"
    )

    logger.info("Computing factors...")
    panel = compute_factors(prices)

    logger.info("Running IC analysis...")
    results = run_ic_analysis(panel)

    # Cache results
    save_result(
        CACHE_KEY,
        results,
        script=__file__,
        max_age_days=MAX_AGE,
        artifact_type="ic_results",
    )
    logger.info(f"Results cached → key: {CACHE_KEY}")


def _print_cached(results: dict):
    print("\n  Cached IC Results:")
    for k, v in results.items():
        if k == "walk_forward":
            continue
        if isinstance(v, dict) and "ic" in v:
            print(f"    {k:<30} IC={v['ic']:+.3f}  p={v['p']:.3f}")


if __name__ == "__main__":
    main()
