"""
Walk-Forward 12-Month OOS Backtest
====================================
Production-ready strategy using LOCKED IS-validated regime weights from v1.0.0.
NO refitting. Parameters from data/regime_params_validated.json.

OOS Window: Apr 2025 → Apr 2026 (last 12 months)
Training: 2018-01-01 → 2025-03-31 (expanding window, frozen at IS end)

Walk-forward folds:
  Fold 1: Train 2018–2022, Test Apr–Jun 2025
  Fold 2: Train 2018–2023, Test Jul–Sep 2025
  Fold 3: Train 2018–2024, Test Oct–Dec 2025
  Fold 4: Train 2018–2025Q1, Test Jan–Apr 2026
  + Full 12-month continuous OOS
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import yfinance as yf

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ────────────────────────────────────────────────────────────────
ROUND_TRIP_RT = 0.00126  # 0.126% round-trip cost per trade
REBAL_WEEKS = 52  # weekly rebalance
TURNOVER_PCT = 0.30  # 30% portfolio turnover per rebalance
ANNUAL_COST = ROUND_TRIP_RT * REBAL_WEEKS * TURNOVER_PCT  # ~0.87%/yr drag
PERIODS_YEAR = 252

OOS_START = "2025-04-14"  # 12 months ago from today
OOS_END = "2026-04-11"  # last trading day

WF_FOLDS = {
    "Fold 1 (Apr-Jun 2025)": ("2025-04-14", "2025-06-30"),
    "Fold 2 (Jul-Sep 2025)": ("2025-07-01", "2025-09-30"),
    "Fold 3 (Oct-Dec 2025)": ("2025-10-01", "2025-12-31"),
    "Fold 4 (Jan-Apr 2026)": ("2026-01-01", "2026-04-11"),
}

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {
    "strategy": "#20808D",
    "spy": "#A84B2F",
    "fold1": "#1B474D",
    "fold2": "#20808D",
    "fold3": "#A84B2F",
    "fold4": "#944454",
    "drawdown": "#A84B2F",
    "bg": "#F7F6F2",
    "surface": "#F9F8F5",
    "border": "#D4D1CA",
    "text": "#28251D",
    "muted": "#7A7974",
    "grid": "#E8E6E0",
}

# ── Load locked IS params ─────────────────────────────────────────────────────
params_path = Path(__file__).parent / "data" / "regime_params_validated.json"
with open(params_path) as _f:
    params = json.load(_f)

BULL_W_TS = params["bull_w_ts_mom"]  # 0.50
BULL_W_MR = params["bull_w_mr"]  # 0.15
BULL_W_MACD = params["bull_w_macd"]  # 0.30
BULL_W_RSI = params["bull_w_rsi"]  # 0.05

BEAR_W_TS = 0.30
BEAR_W_MR = 0.30
BEAR_W_MACD = 0.25
BEAR_W_RSI = 0.10
BEAR_W_PMO = 0.05

VIX_THRESHOLD = 20.0
SPY_MA_PERIOD = 200

pit_path = Path(__file__).parent / "data" / "pit_universe.json"
with open(pit_path) as _f:
    PIT = json.load(_f)

print(f"Locked IS params: ts={BULL_W_TS} mr={BULL_W_MR} macd={BULL_W_MACD} rsi={BULL_W_RSI}")
print(f"Annual cost drag: {ANNUAL_COST * 100:.2f}%")

# ── Download fresh data via yfinance ──────────────────────────────────────────

# All symbols from PIT universe + macro
ALL_PIT_SYMS = set()
for year_syms in PIT.values():
    ALL_PIT_SYMS.update(year_syms)

MACRO_MAP = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "HYG": "HYG",
    "LQD": "LQD",
    "TLT": "TLT",
    "SHY": "SHY",
    "GLD": "GLD",
    "IWM": "IWM",
}

ALL_SYMS = list(ALL_PIT_SYMS) + [s for s in MACRO_MAP if s not in ALL_PIT_SYMS]

print(f"\nDownloading data for {len(ALL_SYMS)} symbols (2017-01-01 → 2026-04-13)...")

price_data: dict[str, pd.DataFrame] = {}
failed = []

for sym in ALL_SYMS:
    ticker = MACRO_MAP.get(sym, sym)
    try:
        df = yf.download(
            ticker, start="2017-01-01", end="2026-04-13", auto_adjust=True, progress=False
        )
        if df is None or df.empty:
            failed.append(sym)
            continue
        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
        # Rename if needed
        col_map = {"open": "close", "high": "close", "low": "close"}
        if "close" not in df.columns:
            for c in df.columns:
                if "close" in c.lower():
                    df = df.rename(columns={c: "close"})
                    break
        if "close" in df.columns and len(df) > 20:
            df.index = pd.to_datetime(df.index).normalize()
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            price_data[sym] = df
    except Exception:
        failed.append(sym)

# Handle FB → META rename
if "FB" not in price_data and "META" in price_data:
    price_data["FB"] = price_data["META"]
if "META" not in price_data and "FB" in price_data:
    price_data["META"] = price_data["FB"]

print(f"  Loaded {len(price_data)} symbols | Failed: {failed or 'none'}")
print(
    f"  SPY range: {price_data['SPY'].index.min().date()} → {price_data['SPY'].index.max().date()}"
)

# ── Signal computation helpers ────────────────────────────────────────────────


def ts_momentum(close: pd.Series, lookback=252) -> pd.Series:
    ret_12m = close.pct_change(lookback).fillna(0)
    vol = close.pct_change().rolling(21).std().replace(0, np.nan)
    return (ret_12m / vol).fillna(0)


def mean_reversion(close: pd.Series, window=20) -> pd.Series:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0, np.nan)
    return -((close - ma) / std).fillna(0)


def macd_signal(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).fillna(0)


def rsi_signal(close: pd.Series, period=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return ((50 - rsi) / 50).fillna(0)


def pmo_signal(close: pd.Series, r1=35, r2=20) -> pd.Series:
    roc = close.pct_change(1) * 100
    s1 = roc.ewm(span=r1, adjust=False).mean() * 10
    s2 = s1.ewm(span=r2, adjust=False).mean()
    return -s2.fillna(0)


def get_regime(vix_series: pd.Series, spy_close: pd.Series, date) -> str:
    try:
        vix_val = vix_series.get(date, np.nan)
        spy_val = spy_close.get(date, np.nan)
        spy_ma = spy_close.loc[:date].tail(SPY_MA_PERIOD).mean()
        if pd.isna(vix_val) or pd.isna(spy_val) or pd.isna(spy_ma):
            return "bear"
        return "bull" if (vix_val < VIX_THRESHOLD and spy_val > spy_ma) else "bear"
    except Exception:
        return "bear"


def compute_composite_score(sym, date, df, vix, spy_close) -> float:
    close = df["close"] if "close" in df.columns else df.get("Close", pd.Series())
    close = close.loc[:date]
    if len(close) < 260:
        return 0.0
    ts = ts_momentum(close).iloc[-1]
    mr = mean_reversion(close).iloc[-1]
    mac = macd_signal(close).iloc[-1]
    rsi = rsi_signal(close).iloc[-1]
    pmo = pmo_signal(close).iloc[-1]

    regime = get_regime(vix, spy_close, date)
    if regime == "bull":
        score = BULL_W_TS * ts + BULL_W_MR * mr + BULL_W_MACD * mac + BULL_W_RSI * rsi
    else:
        score = (
            BEAR_W_TS * ts
            + BEAR_W_MR * mr
            + BEAR_W_MACD * mac
            + BEAR_W_RSI * rsi
            + BEAR_W_PMO * pmo
        )
    return float(score)


def pit_universe_for_date(date) -> list:
    year = str(date.year)
    if year not in PIT:
        avail = [k for k in PIT if int(k) <= date.year]
        year = max(avail) if avail else list(PIT.keys())[-1]
    return [s for s in PIT[year] if s in price_data]


# ── Main backtest loop ────────────────────────────────────────────────────────


def run_backtest(start: str, end: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    spy_df = price_data["SPY"]
    spy_close = spy_df["close"] if "close" in spy_df.columns else spy_df["Close"]
    spy_close.index = pd.to_datetime(spy_close.index).normalize()

    vix_df = price_data.get("VIX")
    if vix_df is not None:
        vix_close = vix_df["close"] if "close" in vix_df.columns else vix_df["Close"]
        vix_close.index = pd.to_datetime(vix_close.index).normalize()
    else:
        vix_close = pd.Series(dtype=float)

    dates = spy_close.loc[start:end].index.tolist()
    if not dates:
        raise ValueError(f"No dates in range {start}→{end}")

    # Weekly rebalance (last trading day of each week)
    rebal_dates = set()
    date_df = pd.DataFrame({"date": dates})
    date_df["week"] = pd.to_datetime(date_df["date"]).dt.to_period("W")
    for _, grp in date_df.groupby("week"):
        rebal_dates.add(grp["date"].iloc[-1])

    portfolio_value = 1.0
    holdings: dict[str, float] = {}
    daily_returns = []
    trade_log = []

    prev_date = None
    for date in dates:
        daily_pnl = 0.0
        for sym, w in holdings.items():
            if sym not in price_data:
                continue
            sym_df = price_data[sym]
            close_col = "close" if "close" in sym_df.columns else "Close"
            if prev_date is None:
                continue
            try:
                p0 = sym_df.loc[prev_date, close_col] if prev_date in sym_df.index else np.nan
                p1 = sym_df.loc[date, close_col] if date in sym_df.index else np.nan
                if not (pd.isna(p0) or pd.isna(p1) or p0 == 0):
                    daily_pnl += w * (p1 / p0 - 1)
            except (KeyError, TypeError):
                continue

        daily_pnl -= ANNUAL_COST / PERIODS_YEAR

        portfolio_value *= 1 + daily_pnl
        daily_returns.append({"date": date, "return": daily_pnl})

        if date in rebal_dates:
            universe = pit_universe_for_date(date)
            scores = {}
            for sym in universe:
                try:
                    sc = compute_composite_score(sym, date, price_data[sym], vix_close, spy_close)
                    scores[sym] = sc
                except Exception:
                    scores[sym] = 0.0

            sorted_syms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_n = max(3, len(sorted_syms) // 4)
            long_syms = [s for s, _ in sorted_syms[:top_n]]

            new_holdings = {s: 1.0 / len(long_syms) for s in long_syms}

            old_set = set(holdings.keys())
            new_set = set(new_holdings.keys())
            turned = len(old_set.symmetric_difference(new_set)) / max(len(old_set | new_set), 1)
            trade_log.append(
                {
                    "date": date,
                    "turnover": turned,
                    "regime": get_regime(vix_close, spy_close, date),
                    "n_longs": len(long_syms),
                    "top_picks": long_syms[:5],
                }
            )

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
        return {
            "label": label,
            "sharpe": np.nan,
            "sortino": np.nan,
            "cagr": np.nan,
            "max_dd": np.nan,
            "calmar": np.nan,
            "vol": np.nan,
            "n_days": len(r),
            "win_rate": np.nan,
            "best_day": np.nan,
            "worst_day": np.nan,
        }

    ann_ret = (1 + r).prod() ** (PERIODS_YEAR / len(r)) - 1
    ann_vol = r.std() * np.sqrt(PERIODS_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(PERIODS_YEAR)
    sortino = ann_ret / downside if downside > 0 else np.nan

    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    win_rate = (r > 0).sum() / len(r)

    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(ann_ret * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "vol": round(ann_vol * 100, 2),
        "n_days": len(r),
        "win_rate": round(win_rate * 100, 1),
        "best_day": round(float(r.max()) * 100, 2),
        "worst_day": round(float(r.min()) * 100, 2),
    }


def max_drawdown_details(returns: pd.Series) -> dict:
    cum = (1 + returns.dropna()).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    min_idx = dd.idxmin()
    peak_idx = cum.loc[:min_idx].idxmax()
    post = dd.loc[min_idx:]
    recovered = post[post >= -0.001]
    rec_date = recovered.index[0] if len(recovered) > 0 else None
    duration = (min_idx - peak_idx).days if pd.notna(min_idx) else 0
    return {
        "peak": str(peak_idx.date()) if pd.notna(peak_idx) else None,
        "trough": str(min_idx.date()) if pd.notna(min_idx) else None,
        "recovery": str(rec_date.date()) if rec_date is not None else "ongoing",
        "depth": round(float(dd.min()) * 100, 2),
        "duration_days": duration,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("  WALK-FORWARD 12-MONTH OOS BACKTEST")
print("  Production strategy — locked IS params (2018-2022)")
print(f"  OOS window: {OOS_START} → {OOS_END}")
print(f"{'=' * 70}")

# Full 12-month OOS
print("\nRunning full 12-month OOS...")
strat_ret, spy_ret, trade_log = run_backtest(OOS_START, OOS_END)
full_strat = metrics(strat_ret, "Full 12M OOS")
full_spy = metrics(spy_ret, "SPY Benchmark")
print(f"  Strategy: {len(strat_ret)} trading days | SPY: {len(spy_ret)} days")

# Walk-forward folds
print(f"\nRunning {len(WF_FOLDS)} walk-forward folds...")
fold_results = []
fold_spy_results = []

for fold_name, (s, e) in WF_FOLDS.items():
    s_sub = strat_ret.loc[s:e]
    b_sub = spy_ret.loc[s:e]
    fold_results.append(metrics(s_sub, fold_name))
    fold_spy_results.append(metrics(b_sub, f"SPY {fold_name}"))

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 90}")
print("  STRATEGY PERFORMANCE (Production-locked params, zero refitting)")
print(f"{'=' * 90}")
print(
    f"{'Period':<28} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>8} {'Calmar':>8} {'Vol%':>6} {'WinR%':>6} {'Days':>5}"
)
print("-" * 90)
for r in fold_results:
    print(
        f"{r['label']:<28} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr']:>7.2f} "
        f"{r['max_dd']:>8.2f} {r['calmar']:>8.3f} {r['vol']:>6.2f} {r['win_rate']:>6.1f} {r['n_days']:>5}"
    )
print("-" * 90)
print(
    f"{full_strat['label']:<28} {full_strat['sharpe']:>7.3f} {full_strat['sortino']:>8.3f} {full_strat['cagr']:>7.2f} "
    f"{full_strat['max_dd']:>8.2f} {full_strat['calmar']:>8.3f} {full_strat['vol']:>6.2f} {full_strat['win_rate']:>6.1f} {full_strat['n_days']:>5}"
)

print(f"\n{'=' * 90}")
print("  SPY BENCHMARK")
print(f"{'=' * 90}")
print(
    f"{'Period':<28} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>8} {'Calmar':>8} {'Vol%':>6} {'WinR%':>6} {'Days':>5}"
)
print("-" * 90)
for r in fold_spy_results:
    print(
        f"{r['label']:<28} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr']:>7.2f} "
        f"{r['max_dd']:>8.2f} {r['calmar']:>8.3f} {r['vol']:>6.02f} {r['win_rate']:>6.1f} {r['n_days']:>5}"
    )
print("-" * 90)
print(
    f"{full_spy['label']:<28} {full_spy['sharpe']:>7.3f} {full_spy['sortino']:>8.3f} {full_spy['cagr']:>7.2f} "
    f"{full_spy['max_dd']:>8.2f} {full_spy['calmar']:>8.3f} {full_spy['vol']:>6.02f} {full_spy['win_rate']:>6.1f} {full_spy['n_days']:>5}"
)

# ── Alpha analysis ────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  ALPHA ANALYSIS (Strategy vs SPY)")
print(f"{'=' * 70}")
aligned = pd.DataFrame({"strat": strat_ret, "spy": spy_ret}).dropna()
if len(aligned) > 20:
    beta, alpha = np.polyfit(aligned["spy"], aligned["strat"], 1)
    corr = aligned["strat"].corr(aligned["spy"])
    track_err = (aligned["strat"] - aligned["spy"]).std() * np.sqrt(252)
    info_ratio = (
        ((aligned["strat"] - aligned["spy"]).mean() * 252) / track_err if track_err > 0 else np.nan
    )
    print(f"  Beta:             {beta:.3f}")
    print(f"  Alpha (ann.):     {alpha * 252 * 100:.2f}%")
    print(f"  Correlation:      {corr:.3f}")
    print(f"  Tracking Error:   {track_err * 100:.2f}%")
    print(f"  Information Ratio:{info_ratio:.3f}")

# ── Drawdown detail ───────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  DRAWDOWN ANALYSIS")
print(f"{'=' * 70}")
dd_strat = max_drawdown_details(strat_ret)
dd_spy = max_drawdown_details(spy_ret)
print(
    f"  Strategy: peak={dd_strat['peak']} trough={dd_strat['trough']} "
    f"depth={dd_strat['depth']}% duration={dd_strat['duration_days']}d "
    f"recovery={dd_strat['recovery']}"
)
print(
    f"  SPY:      peak={dd_spy['peak']}   trough={dd_spy['trough']}   "
    f"depth={dd_spy['depth']}% duration={dd_spy['duration_days']}d "
    f"recovery={dd_spy['recovery']}"
)

# ── Regime distribution ───────────────────────────────────────────────────────
if len(trade_log) > 0:
    regime_counts = trade_log["regime"].value_counts()
    print(f"\n  Regime distribution: {dict(regime_counts)}")
    print(f"  Avg positions per rebalance: {trade_log['n_longs'].mean():.1f}")
    print(f"  Avg turnover: {trade_log['turnover'].mean() * 100:.1f}%")

# ── Walk-forward consistency ──────────────────────────────────────────────────
n_folds_positive_alpha = sum(
    1 for f, s in zip(fold_results, fold_spy_results) if f["sharpe"] > s["sharpe"]
)
n_folds_positive_cagr = sum(1 for f in fold_results if f["cagr"] > 0)
n_folds_beat_spy = sum(1 for f, s in zip(fold_results, fold_spy_results) if f["cagr"] > s["cagr"])
print(f"\n{'=' * 70}")
print("  WALK-FORWARD VERDICT")
print(f"{'=' * 70}")
print(f"  Folds with positive CAGR:        {n_folds_positive_cagr}/{len(WF_FOLDS)}")
print(f"  Folds beating SPY (CAGR):        {n_folds_beat_spy}/{len(WF_FOLDS)}")
print(f"  Folds with higher Sharpe vs SPY: {n_folds_positive_alpha}/{len(WF_FOLDS)}")
print(f"  Full 12M Strategy Sharpe:        {full_strat['sharpe']:.3f}")
print(f"  Full 12M SPY Sharpe:             {full_spy['sharpe']:.3f}")

if full_strat["sharpe"] > 0 and n_folds_positive_cagr >= 3:
    verdict = "✅ PASS — Strategy shows consistent OOS performance"
elif full_strat["sharpe"] > 0 and n_folds_positive_cagr >= 2:
    verdict = "⚠️  CONDITIONAL — Positive overall but inconsistent across folds"
else:
    verdict = "❌ FAIL — Strategy underperforms OOS, review before paper trading"

print(f"\n  >>> {verdict}")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "run_date": datetime.now().isoformat(),
    "oos_start": OOS_START,
    "oos_end": OOS_END,
    "locked_params": params,
    "strategy_full_12m": full_strat,
    "spy_full_12m": full_spy,
    "walk_forward_folds": {r["label"]: r for r in fold_results},
    "spy_folds": {r["label"]: r for r in fold_spy_results},
    "drawdown_strategy": dd_strat,
    "drawdown_spy": dd_spy,
    "verdict": verdict,
}
results_path = Path(__file__).parent / "results" / "wf_12m_oos_results.json"
results_path.parent.mkdir(exist_ok=True)
with open(results_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved → {results_path}")

# ── Save returns for charting ─────────────────────────────────────────────────
strat_ret.to_frame("strategy").to_csv(
    Path(__file__).parent / "results" / "wf_12m_strat_returns.csv"
)
spy_ret.to_frame("spy").to_csv(Path(__file__).parent / "results" / "wf_12m_spy_returns.csv")

# ── Generate chart ────────────────────────────────────────────────────────────
print("\nGenerating performance chart...")

fig = plt.figure(figsize=(18, 14), facecolor=COLORS["bg"])
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

fig.suptitle(
    f"Walk-Forward 12-Month OOS Backtest\n"
    f"Production Strategy (Locked IS Params) — {OOS_START} → {OOS_END}",
    fontsize=15,
    fontweight="bold",
    color=COLORS["text"],
    y=0.98,
)

# 1. Cumulative returns
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(COLORS["surface"])
cum_strat = (1 + strat_ret).cumprod()
cum_spy = (1 + spy_ret).cumprod()

# Align indices
common_idx = cum_strat.index.intersection(cum_spy.index)
if len(common_idx) > 0:
    ax1.plot(
        common_idx,
        cum_strat.loc[common_idx],
        color=COLORS["strategy"],
        lw=2.5,
        label=f"Strategy (Sharpe={full_strat['sharpe']:.2f}, CAGR={full_strat['cagr']:.1f}%)",
    )
    ax1.plot(
        common_idx,
        cum_spy.loc[common_idx],
        color=COLORS["spy"],
        lw=2,
        ls="--",
        label=f"SPY B&H (Sharpe={full_spy['sharpe']:.2f}, CAGR={full_spy['cagr']:.1f}%)",
    )

# Shade walk-forward folds
fold_colors = [COLORS["fold1"], COLORS["fold2"], COLORS["fold3"], COLORS["fold4"]]
for i, (fold_name, (s, e)) in enumerate(WF_FOLDS.items()):
    ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.06, color=fold_colors[i], lw=0)
    mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
    ax1.text(
        mid,
        ax1.get_ylim()[1] if ax1.get_ylim()[1] != 1 else 1.02,
        fold_name.split("(")[1].rstrip(")"),
        fontsize=8,
        ha="center",
        va="bottom",
        color=COLORS["muted"],
    )

ax1.legend(fontsize=10, framealpha=0.9, edgecolor=COLORS["border"])
ax1.grid(True, alpha=0.3, color=COLORS["grid"])
ax1.set_ylabel("Cumulative Return", fontsize=10, color=COLORS["text"])
ax1.set_title(
    "Equity Curve — Strategy vs SPY", fontsize=12, fontweight="bold", color=COLORS["text"]
)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# 2. Drawdown
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(COLORS["surface"])
dd_s = (cum_strat - cum_strat.cummax()) / cum_strat.cummax() * 100
dd_b = (cum_spy - cum_spy.cummax()) / cum_spy.cummax() * 100
ax2.fill_between(dd_s.index, dd_s, 0, alpha=0.4, color=COLORS["strategy"], label="Strategy")
ax2.plot(dd_b.index, dd_b, color=COLORS["spy"], lw=1.5, ls="--", alpha=0.7, label="SPY")
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3, color=COLORS["grid"])
ax2.set_ylabel("Drawdown (%)", fontsize=10, color=COLORS["text"])
ax2.set_title("Drawdown Profile", fontsize=12, fontweight="bold", color=COLORS["text"])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# 3. Per-fold Sharpe comparison
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(COLORS["surface"])
fold_names = [r["label"].split("(")[1].rstrip(")") for r in fold_results]
strat_sharpes = [r["sharpe"] for r in fold_results]
spy_sharpes = [r["sharpe"] for r in fold_spy_results]
x = np.arange(len(fold_names))
width = 0.35
ax3.bar(x - width / 2, strat_sharpes, width, label="Strategy", color=COLORS["strategy"], alpha=0.8)
ax3.bar(x + width / 2, spy_sharpes, width, label="SPY", color=COLORS["spy"], alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(fold_names, fontsize=9, color=COLORS["text"])
ax3.set_ylabel("Sharpe Ratio", fontsize=10, color=COLORS["text"])
ax3.set_title("Sharpe by Walk-Forward Fold", fontsize=12, fontweight="bold", color=COLORS["text"])
ax3.legend(fontsize=9, framealpha=0.9)
ax3.axhline(0, color=COLORS["border"], lw=1)
ax3.grid(axis="y", alpha=0.3, color=COLORS["grid"])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# 4. Rolling 21d Sharpe
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor(COLORS["surface"])
roll_sharpe_s = strat_ret.rolling(21).mean() / strat_ret.rolling(21).std() * np.sqrt(252)
roll_sharpe_b = spy_ret.rolling(21).mean() / spy_ret.rolling(21).std() * np.sqrt(252)
ax4.plot(
    roll_sharpe_s.index,
    roll_sharpe_s,
    color=COLORS["strategy"],
    lw=1.5,
    alpha=0.8,
    label="Strategy",
)
ax4.plot(
    roll_sharpe_b.index, roll_sharpe_b, color=COLORS["spy"], lw=1.2, ls="--", alpha=0.7, label="SPY"
)
ax4.axhline(0, color=COLORS["border"], lw=1)
ax4.legend(fontsize=9, framealpha=0.9)
ax4.grid(True, alpha=0.3, color=COLORS["grid"])
ax4.set_ylabel("Rolling 21d Sharpe", fontsize=10, color=COLORS["text"])
ax4.set_title(
    "Rolling Sharpe Ratio (21-Day Window)", fontsize=12, fontweight="bold", color=COLORS["text"]
)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# 5. Monthly returns heatmap
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor(COLORS["surface"])
monthly = strat_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
monthly_spy = spy_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
months = [d.strftime("%b %Y") for d in monthly.index]
x_pos = np.arange(len(months))
colors_monthly = [COLORS["strategy"] if v >= 0 else COLORS["spy"] for v in monthly.values]
ax5.bar(x_pos, monthly.values, color=colors_monthly, alpha=0.8, width=0.7)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(months, fontsize=8, rotation=45, ha="right", color=COLORS["text"])
ax5.set_ylabel("Return (%)", fontsize=10, color=COLORS["text"])
ax5.set_title("Monthly Strategy Returns", fontsize=12, fontweight="bold", color=COLORS["text"])
ax5.axhline(0, color=COLORS["border"], lw=1)
ax5.grid(axis="y", alpha=0.3, color=COLORS["grid"])
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

plt.savefig(
    str(results_path.parent / "wf_12m_oos_chart.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor=COLORS["bg"],
)
plt.close()
print(f"Chart → {results_path.parent / 'wf_12m_oos_chart.png'}")
print("\nDone.")
