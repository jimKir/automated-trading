"""
Technical Indicator IC Analysis
================================
Measures the Information Coefficient (IC = Spearman rank correlation)
of RSI, MACD, ADX, Stochastic, and PMO against next-day returns.

Also runs a walk-forward Sharpe analysis for ADX as a regime filter.

Results are cached — re-running with the same data returns immediately.

Usage:
    PYTHONPATH=. python diagnostics/technical_indicator_ic.py

    # Force recompute:
    PYTHONPATH=. python diagnostics/technical_indicator_ic.py --force

Validated findings (OOS 2023-2026):
  - PMO crossover: IC -0.032, p=0.005 ✅ ONLY SIGNIFICANT (contrarian)
  - ADX as FILTER: Sharpe +0.71 OOS, 4/4 positive walk-forward years ✅
  - RSI, MACD, Stochastic, TS/CS Momentum: all insignificant ❌
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from scipy import stats

from momentum_signals_exploration.price_cache import PriceCache
from momentum_signals_exploration.data_store import save_result, load_result

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO",
    "JPM","V","MA","UNH","JNJ","PG","HD","KO","XOM","CVX","BAC","GS",
]
OOS_START = "2023-01-01"
OOS_END   = "2026-04-01"
WINDOWS   = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-04-01"),
]
CACHE_KEY  = f"technical_indicator_ic_{OOS_START}_{OOS_END}_{len(SYMS)}syms"
MAX_AGE    = 14  # days

FORCE = "--force" in sys.argv


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series) -> pd.Series:
    fast  = close.ewm(span=12, min_periods=12).mean()
    slow  = close.ewm(span=26, min_periods=26).mean()
    macd  = fast - slow
    sig   = macd.ewm(span=9, min_periods=9).mean()
    return macd - sig


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Return ADX series (scalar strength, direction-agnostic)."""
    hi, lo, cl = high.values, low.values, close.values
    n = len(hi)
    if n < period + 5:
        return pd.Series(np.nan, index=close.index)

    tr_arr, dmp_arr, dmm_arr = [], [], []
    for i in range(1, n):
        hl  = hi[i]  - lo[i]
        hpc = abs(hi[i]  - cl[i-1])
        lpc = abs(lo[i]  - cl[i-1])
        tr_arr.append(max(hl, hpc, lpc))
        up   = hi[i]  - hi[i-1]
        down = lo[i-1] - lo[i]
        dmp_arr.append(up   if up   > down and up   > 0 else 0.0)
        dmm_arr.append(down if down > up   and down > 0 else 0.0)

    tr_s   = pd.Series(tr_arr)
    dmp_s  = pd.Series(dmp_arr)
    dmm_s  = pd.Series(dmm_arr)
    atr    = tr_s.ewm(span=period, min_periods=period).mean()
    di_p   = dmp_s.ewm(span=period, min_periods=period).mean() / (atr + 1e-9) * 100
    di_m   = dmm_s.ewm(span=period, min_periods=period).mean() / (atr + 1e-9) * 100
    dx     = (di_p - di_m).abs() / (di_p + di_m + 1e-9) * 100
    adx_s  = dx.ewm(span=period, min_periods=period).mean()
    adx_s.index = close.index[1:]
    return adx_s.reindex(close.index)


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k: int = 14, d: int = 3) -> pd.Series:
    lo_k = low.rolling(k).min()
    hi_k = high.rolling(k).max()
    k_s  = (close - lo_k) / (hi_k - lo_k + 1e-9) * 100
    return k_s.rolling(d).mean()


def _pmo_crossover(close: pd.Series) -> pd.Series:
    roc1  = (close / close.shift(1) - 1) * 100
    ema1  = roc1.ewm(span=35, min_periods=10).mean() * 20
    pmo   = ema1.ewm(span=20, min_periods=5).mean()
    sig   = pmo.ewm(span=10, min_periods=3).mean()
    return pmo - sig


def _ts_momentum(close: pd.Series) -> pd.Series:
    """Time-series momentum: 12-month minus 1-month return."""
    return close.pct_change(252) - close.pct_change(21)


def _cs_momentum(panel: pd.DataFrame) -> pd.Series:
    """Cross-sectional 20-day momentum rank (per date)."""
    ret20 = panel.groupby("symbol")["close"].transform(lambda s: s.pct_change(20))
    return panel.groupby("date")["close"].rank(pct=True) - 0.5


# ---------------------------------------------------------------------------
# Build panel
# ---------------------------------------------------------------------------

def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    syms = prices.index.get_level_values("symbol").unique()

    for sym in syms:
        try:
            df    = prices.xs(sym, level="symbol").sort_index()
            close = df["close"].astype(float)
            high  = df["high"].astype(float)
            low   = df["low"].astype(float)

            rsi_v   = _rsi(close)
            macd_v  = _macd_hist(close)
            adx_v   = _adx(high, low, close)
            stoch_v = _stochastic(high, low, close)
            pmo_v   = _pmo_crossover(close)
            tsm_v   = _ts_momentum(close)
            fwd_ret = close.pct_change(1).shift(-1)

            for dt in df.index:
                rows.append({
                    "symbol":   sym,
                    "date":     dt,
                    "close":    close.get(dt, np.nan),
                    "rsi":      rsi_v.get(dt, np.nan),
                    "macd":     macd_v.get(dt, np.nan),
                    "adx":      adx_v.get(dt, np.nan),
                    "stoch":    stoch_v.get(dt, np.nan),
                    "pmo":      pmo_v.get(dt, np.nan),
                    "ts_mom":   tsm_v.get(dt, np.nan),
                    "fwd_ret":  fwd_ret.get(dt, np.nan),
                })
        except Exception as e:
            logger.warning(f"  Panel error {sym}: {e}")

    panel = pd.DataFrame(rows).dropna(subset=["fwd_ret", "adx"])

    # Derived indicators
    panel["rsi_trend"]       = panel["rsi"] - 50          # >0 = bullish trend
    panel["rsi_contrarian"]  = -(panel["rsi"] - 50)       # >0 = oversold (contrarian)
    panel["macd_hist"]       = panel["macd"]
    panel["adx_dir"]         = panel["adx"]                # higher = more trend (direction-agnostic)
    panel["stoch_trend"]     = panel["stoch"] - 50
    panel["stoch_contrarian"]= -(panel["stoch"] - 50)
    panel["pmo_xover"]       = panel["pmo"]

    # CS momentum (requires full panel)
    panel["cs_mom"] = panel.groupby("date")["close"].rank(pct=True) - 0.5

    return panel


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def _safe_ic(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(a[mask], b[mask])
    return float(r), float(p)


INDICATORS = {
    "RSI(14) trend":         "rsi_trend",
    "RSI(14) contrarian":    "rsi_contrarian",
    "MACD histogram":        "macd_hist",
    "ADX directional":       "adx_dir",
    "Stochastic trend":      "stoch_trend",
    "Stochastic contrarian": "stoch_contrarian",
    "PMO crossover":         "pmo_xover",
    "TS Momentum 12-1mo":    "ts_mom",
    "CS Momentum":           "cs_mom",
}


def run_ic_analysis(panel: pd.DataFrame) -> dict:
    print("\n" + "=" * 65)
    print("  TECHNICAL INDICATOR IC ANALYSIS")
    print(f"  OOS period: {OOS_START} → {OOS_END}  |  {len(SYMS)} symbols")
    print("=" * 65)
    print(f"\n  {'Indicator':<28} {'IC':>8} {'p-value':>10}  Verdict")
    print("  " + "─" * 60)

    results = {}
    for label, col in INDICATORS.items():
        ic, p = _safe_ic(panel[col], panel["fwd_ret"])
        if p < 0.01:
            sig = "✅ SIGNIFICANT"
        elif p < 0.05:
            sig = "✅ p<0.05"
        elif p < 0.10:
            sig = "⚠️  marginal"
        else:
            sig = "❌ noise"
        print(f"  {label:<28} {ic:>+8.3f} {p:>10.3f}  {sig}")
        results[label] = {"ic": ic, "p": p, "col": col}

    # ADX as filter — walk-forward Sharpe
    print("\n  ADX as REGIME FILTER — walk-forward Sharpe (position halved when ADX < 20)")
    print(f"  {'Window':<10} {'Sharpe (ADX>20)':>18} {'Sharpe (no filter)':>20}")
    print("  " + "─" * 52)

    wf_adx = []
    for win_label, win_start, win_end in WINDOWS:
        mask = (panel["date"] >= win_start) & (panel["date"] <= win_end)
        sub  = panel.loc[mask].copy()

        # Strategy return: long top-3 symbols by rank, daily
        sub["rank"] = sub.groupby("date")["fwd_ret"].rank(ascending=False)
        sub["in_top"] = sub["rank"] <= 3

        # No filter
        base_ret = sub.loc[sub["in_top"], "fwd_ret"]
        base_sh  = base_ret.mean() / (base_ret.std() + 1e-9) * np.sqrt(252)

        # ADX filter: halve position when ADX < 20
        sub["size"] = np.where(sub["adx"] >= 20, 1.0, 0.5)
        filt_ret = sub.loc[sub["in_top"], "fwd_ret"] * sub.loc[sub["in_top"], "size"]
        filt_sh  = filt_ret.mean() / (filt_ret.std() + 1e-9) * np.sqrt(252)

        print(f"  {win_label:<10} {filt_sh:>+18.2f} {base_sh:>+20.2f}")
        wf_adx.append({"window": win_label, "sharpe_adx_filter": filt_sh, "sharpe_base": base_sh})

    # Combined
    all_top  = panel.loc[panel.groupby("date")["fwd_ret"].rank(ascending=False) <= 3]
    all_base = all_top["fwd_ret"]
    all_size = np.where(all_top["adx"] >= 20, 1.0, 0.5)
    all_filt = all_top["fwd_ret"] * all_size
    base_all = all_base.mean() / (all_base.std() + 1e-9) * np.sqrt(252)
    filt_all = all_filt.mean() / (all_filt.std() + 1e-9) * np.sqrt(252)
    print(f"  {'ALL':<10} {filt_all:>+18.2f} {base_all:>+20.2f}")
    print("=" * 65 + "\n")

    results["adx_walk_forward"] = wf_adx
    results["adx_all_sharpe"]   = {"filtered": float(filt_all), "base": float(base_all)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not FORCE:
        cached = load_result(CACHE_KEY, max_age_days=MAX_AGE)
        if cached is not None:
            print(f"\n[DataStore] Loaded IC results from cache (key: {CACHE_KEY})")
            print("  Run with --force to recompute.\n")
            _print_cached(cached)
            return

    logger.info("Fetching daily price data via PriceCache...")
    cache  = PriceCache()
    prices = cache.get_daily(SYMS, start=OOS_START, end=OOS_END)

    if prices is None or prices.empty:
        logger.error("No price data — cannot compute IC")
        sys.exit(1)

    logger.info(f"Price data: {len(prices)} rows  |  {prices.index.get_level_values(0).nunique()} symbols")

    logger.info("Computing indicators...")
    panel = build_panel(prices)

    logger.info("Running IC analysis...")
    results = run_ic_analysis(panel)

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
        if k in ("adx_walk_forward", "adx_all_sharpe"):
            continue
        if isinstance(v, dict) and "ic" in v:
            sig = "✅" if v["p"] < 0.05 else "❌"
            print(f"    {k:<28} IC={v['ic']:+.3f}  p={v['p']:.3f}  {sig}")
    if "adx_all_sharpe" in results:
        s = results["adx_all_sharpe"]
        print(f"\n    ADX filter Sharpe (all years): {s.get('filtered', '?'):+.2f} "
              f"vs base {s.get('base', '?'):+.2f}")


if __name__ == "__main__":
    main()
