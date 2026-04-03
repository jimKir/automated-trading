"""
OOS Extended Backtest: 2023-01-01 → 2026-04-03
===============================================
Uses LOCKED IS-validated regime weights from v1.0.0-paper-baseline.
NO refitting. Parameters read from data/regime_params_validated.json.

Sub-period breakdown: 2023 / 2024 / 2025 / Q1-2026
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ────────────────────────────────────────────────────────────────
IS_END        = "2022-12-31"
OOS_START     = "2023-01-01"
OOS_END       = "2026-04-03"
ROUND_TRIP_RT = 0.00126        # 0.126% round-trip cost per trade
REBAL_WEEKS   = 52             # weekly rebalance → ~52 rounds/yr
TURNOVER_PCT  = 0.30           # 30% portfolio turnover per rebalance
ANNUAL_COST   = ROUND_TRIP_RT * REBAL_WEEKS * TURNOVER_PCT  # ~0.87%/yr drag
PERIODS_YEAR  = 252

SUB_PERIODS = {
    "2023":    ("2023-01-01", "2023-12-31"),
    "2024":    ("2024-01-01", "2024-12-31"),
    "2025":    ("2025-01-01", "2025-12-31"),
    "Q1-2026": ("2026-01-01", "2026-04-03"),
}

COLORS = {
    "strategy":  "#20808D",
    "spy":       "#A84B2F",
    "2023":      "#1B474D",
    "2024":      "#20808D",
    "2025":      "#A84B2F",
    "Q1-2026":   "#944454",
    "drawdown":  "#A84B2F",
    "bg":        "#F7F6F2",
    "surface":   "#F9F8F5",
    "border":    "#D4D1CA",
    "text":      "#28251D",
    "muted":     "#7A7974",
    "grid":      "#E8E6E0",
}

# ── Load locked IS params ─────────────────────────────────────────────────────
params = json.load(open("data/regime_params_validated.json"))
PIT    = json.load(open("data/pit_universe.json"))

BULL_W_TS   = params["bull_w_ts_mom"]   # 0.50
BULL_W_MR   = params["bull_w_mr"]       # 0.15
BULL_W_MACD = params["bull_w_macd"]     # 0.30
BULL_W_RSI  = params["bull_w_rsi"]      # 0.05

BEAR_W_TS   = 0.30
BEAR_W_MR   = 0.30
BEAR_W_MACD = 0.25
BEAR_W_RSI  = 0.10
BEAR_W_PMO  = 0.05

VIX_THRESHOLD = 20.0
SPY_MA_PERIOD = 200

print(f"Loaded IS params: ts={BULL_W_TS} mr={BULL_W_MR} macd={BULL_W_MACD} rsi={BULL_W_RSI}")

# ── Load price data ───────────────────────────────────────────────────────────
from src.market_data.historical_store import (
    load_parquet, EQUITY_SYMS, MACRO_SYMS
)

ALL_SYMS = EQUITY_SYMS + [s for s in MACRO_SYMS if s not in EQUITY_SYMS]

def load_prices(symbols, start, end) -> Dict[str, pd.DataFrame]:
    data = {}
    for sym in symbols:
        df = load_parquet(sym)
        if df is None or df.empty:
            continue
        # normalise index to date (not datetime)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()
        df = df.loc[start:end]
        if len(df) > 20:
            data[sym] = df
    return data

print("Loading price data…")
price_data = load_prices(ALL_SYMS, "2018-01-01", OOS_END)
print(f"  Loaded {len(price_data)} symbols, last date: {max(df.index.max() for df in price_data.values()).date()}")

# ── Signal computation helpers ────────────────────────────────────────────────

def ts_momentum(close: pd.Series, fast=21, slow=63, lookback=252) -> pd.Series:
    """Time-series momentum: 12M return risk-adjusted."""
    ret_12m = close.pct_change(lookback).fillna(0)
    vol = close.pct_change().rolling(21).std().replace(0, np.nan)
    return (ret_12m / vol).fillna(0)

def mean_reversion(close: pd.Series, window=20) -> pd.Series:
    """Z-score mean reversion (negative = below MA → buy signal)."""
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0, np.nan)
    return -((close - ma) / std).fillna(0)   # negative so oversold = positive signal

def macd_signal(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).fillna(0)

def rsi_signal(close: pd.Series, period=14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    # oversold→buy, overbought→sell
    return ((50 - rsi) / 50).fillna(0)

def pmo_signal(close: pd.Series, r1=35, r2=20) -> pd.Series:
    """Price Momentum Oscillator (contrarian: IC -0.032)."""
    roc = close.pct_change(1) * 100
    s1  = roc.ewm(span=r1, adjust=False).mean() * 10
    s2  = s1.ewm(span=r2,  adjust=False).mean()
    return -s2.fillna(0)   # contrarian sign

def get_regime(vix_series: pd.Series, spy_close: pd.Series, date: pd.Timestamp) -> str:
    """Bull if VIX < 20 AND SPY > 200MA, else bear."""
    try:
        vix_val = vix_series.get(date, np.nan)
        spy_val = spy_close.get(date, np.nan)
        spy_ma  = spy_close.loc[:date].tail(SPY_MA_PERIOD).mean()
        if pd.isna(vix_val) or pd.isna(spy_val) or pd.isna(spy_ma):
            return "bear"
        return "bull" if (vix_val < VIX_THRESHOLD and spy_val > spy_ma) else "bear"
    except Exception:
        return "bear"

def compute_composite_score(sym: str, date: pd.Timestamp, df: pd.DataFrame,
                             vix: pd.Series, spy_close: pd.Series) -> float:
    close = df["Close"] if "Close" in df.columns else df["close"]
    close = close.loc[:date]
    if len(close) < 260:
        return 0.0
    ts   = ts_momentum(close).iloc[-1]
    mr   = mean_reversion(close).iloc[-1]
    mac  = macd_signal(close).iloc[-1]
    rsi  = rsi_signal(close).iloc[-1]
    pmo  = pmo_signal(close).iloc[-1]

    regime = get_regime(vix, spy_close, date)
    if regime == "bull":
        score = (BULL_W_TS * ts + BULL_W_MR * mr +
                 BULL_W_MACD * mac + BULL_W_RSI * rsi)
    else:
        score = (BEAR_W_TS * ts + BEAR_W_MR * mr +
                 BEAR_W_MACD * mac + BEAR_W_RSI * rsi + BEAR_W_PMO * pmo)
    return float(score)

# ── Build trading universe per date (PIT) ─────────────────────────────────────

def pit_universe_for_date(date: pd.Timestamp) -> list:
    year = str(date.year)
    if year not in PIT:
        year = max(k for k in PIT if int(k) <= date.year)
    syms = PIT[year]
    # keep only symbols we have data for
    return [s for s in syms if s in price_data]

# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(start: str, end: str) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Returns:
        strategy_returns : daily returns pd.Series
        spy_returns      : daily SPY returns pd.Series
        trades_log       : DataFrame
    """
    # Build common calendar
    spy_df    = price_data["SPY"]
    spy_close = (spy_df["Close"] if "Close" in spy_df.columns else spy_df["close"])
    spy_close.index = pd.to_datetime(spy_close.index).normalize()

    vix_df = price_data.get("VIX")
    if vix_df is not None:
        vix_close = (vix_df["Close"] if "Close" in vix_df.columns else vix_df["close"])
        vix_close.index = pd.to_datetime(vix_close.index).normalize()
    else:
        vix_close = pd.Series(dtype=float)

    # Trading dates
    dates = spy_close.loc[start:end].index.tolist()
    if not dates:
        raise ValueError(f"No dates in range {start}→{end}")

    # Weekly rebalance: Fridays (or last available day of week)
    rebal_dates = set()
    date_df = pd.DataFrame({"date": dates})
    date_df["week"] = pd.to_datetime(date_df["date"]).dt.to_period("W")
    for _, grp in date_df.groupby("week"):
        rebal_dates.add(grp["date"].iloc[-1])

    portfolio_value = 1.0
    holdings: Dict[str, float] = {}   # sym → weight
    daily_returns = []
    trade_log = []

    prev_date = None
    for date in dates:
        # ── Daily P&L from existing holdings ─────────────────────────────
        daily_pnl = 0.0
        for sym, w in holdings.items():
            if sym not in price_data:
                continue
            sym_df = price_data[sym]
            close_col = "Close" if "Close" in sym_df.columns else "close"
            if prev_date is None:
                continue
            try:
                p0 = sym_df.loc[prev_date, close_col] if prev_date in sym_df.index else np.nan
                p1 = sym_df.loc[date, close_col]      if date in sym_df.index      else np.nan
                if not (pd.isna(p0) or pd.isna(p1) or p0 == 0):
                    daily_pnl += w * (p1 / p0 - 1)
            except (KeyError, TypeError):
                continue

        # Subtract daily cost drag (annual / 252)
        daily_pnl -= ANNUAL_COST / PERIODS_YEAR

        portfolio_value *= (1 + daily_pnl)
        daily_returns.append({"date": date, "return": daily_pnl})

        # ── Rebalance ────────────────────────────────────────────────────
        if date in rebal_dates:
            universe = pit_universe_for_date(date)
            scores = {}
            for sym in universe:
                try:
                    sc = compute_composite_score(sym, date, price_data[sym],
                                                 vix_close, spy_close)
                    scores[sym] = sc
                except Exception:
                    scores[sym] = 0.0

            # Rank and select top/bottom (long-short momentum: long top 5, skip bottom)
            sorted_syms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_n = max(3, len(sorted_syms) // 4)   # top quartile
            long_syms = [s for s, _ in sorted_syms[:top_n]]

            # Equal-weight longs
            new_holdings = {s: 1.0 / len(long_syms) for s in long_syms}

            # Track turnover for cost
            old_set = set(holdings.keys())
            new_set = set(new_holdings.keys())
            turned = len(old_set.symmetric_difference(new_set)) / max(len(old_set | new_set), 1)
            trade_log.append({"date": date, "turnover": turned,
                               "regime": get_regime(vix_close, spy_close, date),
                               "n_longs": len(long_syms)})

            holdings = new_holdings

        prev_date = date

    returns_df = pd.DataFrame(daily_returns).set_index("date")
    returns_df.index = pd.to_datetime(returns_df.index)

    spy_ret = spy_close.loc[start:end].pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index)

    return returns_df["return"], spy_ret, pd.DataFrame(trade_log)


# ── Performance metrics ───────────────────────────────────────────────────────

def metrics(returns: pd.Series, label: str = "") -> dict:
    r = returns.dropna()
    if len(r) < 5:
        return {"label": label, "sharpe": np.nan, "sortino": np.nan,
                "cagr": np.nan, "max_dd": np.nan, "calmar": np.nan,
                "vol": np.nan, "n_days": len(r)}

    ann_ret   = (1 + r).prod() ** (PERIODS_YEAR / len(r)) - 1
    ann_vol   = r.std() * np.sqrt(PERIODS_YEAR)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside  = r[r < 0].std() * np.sqrt(PERIODS_YEAR)
    sortino   = ann_ret / downside if downside > 0 else np.nan

    cum       = (1 + r).cumprod()
    roll_max  = cum.cummax()
    dd        = (cum - roll_max) / roll_max
    max_dd    = float(dd.min())
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "label":   label,
        "sharpe":  round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr":    round(ann_ret * 100, 2),
        "max_dd":  round(max_dd * 100, 2),
        "calmar":  round(calmar, 3),
        "vol":     round(ann_vol * 100, 2),
        "n_days":  len(r),
    }

def max_drawdown_details(returns: pd.Series) -> dict:
    cum = (1 + returns.dropna()).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    min_idx = dd.idxmin()
    peak_idx = cum.loc[:min_idx].idxmax()
    # recovery
    post = dd.loc[min_idx:]
    recovered = post[post >= -0.001]
    rec_date = recovered.index[0] if len(recovered) > 0 else None
    duration = (min_idx - peak_idx).days if pd.notna(min_idx) else 0
    return {
        "peak":     peak_idx.date() if pd.notna(peak_idx) else None,
        "trough":   min_idx.date()  if pd.notna(min_idx)  else None,
        "recovery": rec_date.date() if rec_date is not None else "ongoing",
        "depth":    round(float(dd.min()) * 100, 2),
        "duration_days": duration,
    }


# ── Run full OOS ──────────────────────────────────────────────────────────────
print(f"\nRunning OOS backtest: {OOS_START} → {OOS_END}")
strat_ret, spy_ret, trade_log = run_backtest(OOS_START, OOS_END)
print(f"  Strategy: {len(strat_ret)} trading days | SPY: {len(spy_ret)} days")

# ── Sub-period metrics ────────────────────────────────────────────────────────
print("\nComputing sub-period metrics…")
results = []
spy_results = []

for period, (s, e) in SUB_PERIODS.items():
    s_sub = strat_ret.loc[s:e]
    b_sub = spy_ret.loc[s:e]
    results.append(metrics(s_sub, period))
    spy_results.append(metrics(b_sub, f"SPY {period}"))

# Full OOS
results.append(metrics(strat_ret, "Full OOS (2023-Q1 2026)"))
spy_results.append(metrics(spy_ret, "SPY Full OOS"))

# Print table
print("\n{'='*70}")
print(f"{'Period':<22} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>8} {'Calmar':>8} {'Vol%':>6} {'Days':>5}")
print("-" * 70)
for r in results:
    print(f"{r['label']:<22} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr']:>7.2f} "
          f"{r['max_dd']:>8.2f} {r['calmar']:>8.3f} {r['vol']:>6.2f} {r['n_days']:>5}")
print("\nSPY Benchmark:")
for r in spy_results:
    print(f"{r['label']:<22} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr']:>7.2f} "
          f"{r['max_dd']:>8.2f} {r['calmar']:>8.3f} {r['vol']:>6.2f} {r['n_days']:>5}")

# Q1 2026 drawdown detail
print("\nQ1-2026 Drawdown Detail (tariff shock):")
q1_ret = strat_ret.loc["2026-01-01":"2026-04-03"]
spy_q1 = spy_ret.loc["2026-01-01":"2026-04-03"]
if len(q1_ret) > 5:
    dd_strat = max_drawdown_details(q1_ret)
    dd_spy   = max_drawdown_details(spy_q1)
    print(f"  Strategy: peak={dd_strat['peak']} trough={dd_strat['trough']} "
          f"depth={dd_strat['depth']}% duration={dd_strat['duration_days']}d "
          f"recovery={dd_strat['recovery']}")
    print(f"  SPY:      peak={dd_spy['peak']}   trough={dd_spy['trough']}   "
          f"depth={dd_spy['depth']}% duration={dd_spy['duration_days']}d "
          f"recovery={dd_spy['recovery']}")

# Regime breakdown
if len(trade_log) > 0:
    regime_counts = trade_log["regime"].value_counts()
    print(f"\nRegime distribution (rebalances): {dict(regime_counts)}")

# ── Save metrics to JSON ──────────────────────────────────────────────────────
output = {
    "run_date": pd.Timestamp.now().isoformat(),
    "oos_start": OOS_START,
    "oos_end": OOS_END,
    "params": params,
    "sub_periods": {r["label"]: r for r in results},
    "spy_sub_periods": {r["label"]: r for r in spy_results},
    "q1_2026_dd_strategy": max_drawdown_details(q1_ret) if len(q1_ret) > 5 else {},
    "q1_2026_dd_spy": max_drawdown_details(spy_q1) if len(spy_q1) > 5 else {},
}
with open("/home/user/workspace/oos_extended_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print("\nSaved metrics → /home/user/workspace/oos_extended_results.json")

# ── Persist returns for charting ──────────────────────────────────────────────
strat_ret.to_frame("strategy").to_parquet("/home/user/workspace/oos_strat_returns.parquet")
spy_ret.to_frame("spy").to_parquet("/home/user/workspace/oos_spy_returns.parquet")
print("Saved returns → parquet files")
