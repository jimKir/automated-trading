"""
Walk-Forward Paper Trading Emulation: 2020–2024
================================================
Runs the full paper trading emulator across five independent OOS periods
covering every major regime type:

  Period 1 — COVID crash + recovery        (Feb 2020 – Dec 2020)
  Period 2 — Post-COVID bull + rate fears  (Jan 2021 – Dec 2021)
  Period 3 — Rate hike bear market         (Jan 2022 – Dec 2022)
  Period 4 — Bear-to-bull recovery         (Jan 2023 – Dec 2023)
  Period 5 — Bull run + AI bubble          (Jan 2024 – Dec 2024)

Each period:
  - Uses a causal warmup window (2 years back) for signal initialisation
  - Runs the full stack: regime switching, ChoppyRegimeDetector, PositionAnomalyScorer
  - Records per-regime ChoppyRegimeDetector firing patterns (false positive / miss rates)

ChoppyRegimeDetector stress-test goals
---------------------------------------
  COVID crash     → should fire ORANGE/RED during Mar 2020 crash, quiet by Jun 2020
  2021 bull       → should mostly stay GREEN (low false positive rate)
  2022 bear       → should fire persistently across the year (true positive)
  2023 recovery   → should start YELLOW then fade to GREEN by Q2
  2024 bull       → should stay GREEN through most of the year (low false positive)

Key metrics tracked per period
-------------------------------
  Performance:  total return, Sharpe, max DD, vol, vs SPY
  ChoppyDetector: % days GREEN / YELLOW / ORANGE / RED
                  miss rate (ORANGE/RED during drawdown > 5%)
                  false positive rate (ORANGE/RED during SPY > +3%)
  PositionAnomalyScorer: avg scale per asset class, min scale episode

Output
------
  results/wf_emulation_<period>.json    — per-period results
  results/wf_emulation_chart.png        — consolidated 5-panel chart
  results/wf_emulation_summary.json     — cross-period summary + stress test scores
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from paper_trading_emulator import PaperTradingEmulator, load_sym
from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("WFEmulation")

# ── Regime periods ────────────────────────────────────────────────────────────
# Each entry: (label, warmup_start, paper_start, paper_end, expected_regime_type)
PERIODS = [
    (
        "COVID Crash & Recovery",
        "2018-01-01",
        "2020-02-01",
        "2020-12-31",
        "crash_recovery",
        "Feb 2020 crash (-35% SPY in 23 days) followed by V-shape recovery. "
        "ChoppyDetector should fire RED in Mar, GREEN by Jun.",
    ),
    (
        "Post-COVID Bull + Rate Fears",
        "2018-06-01",
        "2021-01-01",
        "2021-12-31",
        "bull_with_fears",
        "Strong tech bull. Delta wave Aug-Sep, taper tantrum Nov. "
        "Mostly GREEN with brief YELLOW spikes.",
    ),
    (
        "Rate Hike Bear Market",
        "2019-01-01",
        "2022-01-01",
        "2022-12-31",
        "sustained_bear",
        "Fed hiked 425bp. SPY -18%, NDX -33%, BTC -65%. "
        "ChoppyDetector should fire persistently YELLOW/ORANGE.",
    ),
    (
        "Bear-to-Bull Recovery",
        "2020-01-01",
        "2023-01-01",
        "2023-12-31",
        "recovery_bull",
        "SVB collapse Mar 2023, AI bull from May. Should start YELLOW, fade to GREEN by Q2.",
    ),
    (
        "AI Bull Run",
        "2021-01-01",
        "2024-01-01",
        "2024-12-31",
        "sustained_bull",
        "SPY +25%. Low vol, strong momentum. "
        "ChoppyDetector should stay mostly GREEN (low false positive test).",
    ),
]

# Chart colours per period
PERIOD_COLORS = [
    "#A84B2F",  # COVID — rust
    "#1B474D",  # Post-COVID bull — dark teal
    "#944454",  # Rate hike bear — mauve
    "#6E522B",  # Recovery — brown
    "#20808D",  # AI bull — teal
]

C = {
    "bg": "#F7F6F2",
    "surface": "#F9F8F5",
    "border": "#D4D1CA",
    "text": "#28251D",
    "muted": "#7A7974",
    "grid": "#E8E6E0",
    "spy": "#A84B2F",
    "green": "#437A22",
    "amber": "#964219",
    "red": "#A12C7B",
}

PERIODS_YEAR = 252
INITIAL_EQUITY = 100_000.0

# Universe (same as paper_trading_emulator)
UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "GLD",
    "TLT",
    "VGK",
    "EEM",
    "XLK",
    "XLE",
    "XLF",
    "VNQ",
    "AGG",
    "EWJ",
    "XLV",
    "BTC",
    "ETH",
    "ES",
    "NQ",
    "GC",
    "CL",
    "VIX",
    "HYG",
    "LQD",
    "DXY",
    "NFLX",
    "MMM",
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


# ── ChoppyDetector stress metrics ─────────────────────────────────────────────


def detector_stress_metrics(
    eq_log: pd.DataFrame,
    spy_close: pd.Series,
    period_start: str,
    period_end: str,
) -> dict:
    """
    Compute ChoppyRegimeDetector diagnostic metrics for one period.

    Returns:
      pct_green / yellow / orange / red
      false_positive_rate: % ORANGE+RED days when SPY 20d return > +3%
      miss_rate: % GREEN days when SPY 20d drawdown > -5%
      firing_lead_days: median days before drawdown peak that detector fired
    """
    ch = (
        eq_log["choppy_sc"]
        .reindex(spy_close.loc[period_start:period_end].index, method="ffill")
        .fillna(0.0)
    )
    spy_sub = spy_close.loc[period_start:period_end]

    pct_green = float((ch < 0.17).mean() * 100)
    pct_yellow = float(((ch >= 0.17) & (ch < 0.27)).mean() * 100)
    pct_orange = float(((ch >= 0.27) & (ch < 0.40)).mean() * 100)
    pct_red = float((ch >= 0.40).mean() * 100)

    # False positive: detector fired ORANGE+ but market was rising
    spy_20d_ret = spy_sub.pct_change(20)
    fp_mask = (ch >= 0.27) & (spy_20d_ret > 0.03)
    fp_rate = float(fp_mask.sum() / max((ch >= 0.27).sum(), 1) * 100)

    # Miss rate: detector quiet (GREEN) but market was falling hard
    spy_dd = (spy_sub - spy_sub.rolling(20).max()) / spy_sub.rolling(20).max()
    miss_mask = (ch < 0.17) & (spy_dd < -0.05)
    miss_rate = float(miss_mask.sum() / max((spy_dd < -0.05).sum(), 1) * 100)

    # Average scale applied
    avg_ews_scale = (
        float(eq_log.get("ews_scale", pd.Series(1.0)).mean())
        if "ews_scale" in eq_log.columns
        else None
    )

    return {
        "pct_green": round(pct_green, 1),
        "pct_yellow": round(pct_yellow, 1),
        "pct_orange": round(pct_orange, 1),
        "pct_red": round(pct_red, 1),
        "false_positive_rate_pct": round(fp_rate, 1),
        "miss_rate_pct": round(miss_rate, 1),
        "avg_ews_scale": avg_ews_scale,
    }


def performance_metrics(eq_log: pd.DataFrame, spy_close: pd.Series, start: str, end: str) -> dict:
    eq = eq_log["equity"]
    ret = eq.pct_change().dropna()
    if len(ret) < 5:
        return {}
    n = len(ret)
    ann = (1 + ret).prod() ** (PERIODS_YEAR / n) - 1
    vol = ret.std() * np.sqrt(PERIODS_YEAR)
    sh = float(ann / vol) if vol > 0 else 0.0
    cum = (1 + ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    down = ret[ret < 0].std() * np.sqrt(PERIODS_YEAR)
    so = float(ann / down) if down > 0 else 0.0

    spy_sub = spy_close.loc[start:end]
    spy_ret = spy_sub.pct_change().dropna()
    spy_ann = (1 + spy_ret).prod() ** (PERIODS_YEAR / len(spy_ret)) - 1
    spy_vol = spy_ret.std() * np.sqrt(PERIODS_YEAR)
    spy_sh = float(spy_ann / spy_vol) if spy_vol > 0 else 0.0
    spy_cum = (1 + spy_ret).cumprod()
    spy_dd = (spy_cum - spy_cum.cummax()) / spy_cum.cummax()

    total_ret = float((1 + ret).prod() - 1)
    spy_total = float((1 + spy_ret).prod() - 1)

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(ann * 100, 2),
        "sharpe": round(sh, 3),
        "sortino": round(so, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "ann_vol_pct": round(vol * 100, 2),
        "spy_total_return": round(spy_total * 100, 2),
        "spy_sharpe": round(spy_sh, 3),
        "spy_max_dd_pct": round(float(spy_dd.min()) * 100, 2),
        "alpha_pct": round((total_ret - spy_total) * 100, 2),
        "n_trading_days": n,
    }


# ── Single-period runner ──────────────────────────────────────────────────────


def run_period(
    label: str,
    warmup_start: str,
    paper_start: str,
    paper_end: str,
    all_data: dict[str, pd.DataFrame],
    config: dict,
) -> dict:
    """Run one paper-trading period and return full results."""
    log.info(f"\n{'=' * 65}")
    log.info(f"PERIOD: {label}")
    log.info(f"  Warmup: {warmup_start}  |  OOS: {paper_start} → {paper_end}")

    # Patch emulator constants for this period
    import paper_trading_emulator as pte

    pte.PAPER_START = paper_start
    pte.PAPER_END = paper_end
    pte.WARMUP_START = warmup_start
    pte.INITIAL_EQUITY = INITIAL_EQUITY

    emulator = PaperTradingEmulator(config)
    results = emulator.run(all_data)

    eq_log = pd.DataFrame(results["equity_log"]).set_index("date")
    eq_log.index = pd.to_datetime(eq_log.index)

    spy_close = all_data["SPY"]["Close"]
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)

    perf = performance_metrics(eq_log, spy_close, paper_start, paper_end)
    det_m = detector_stress_metrics(eq_log, spy_close, paper_start, paper_end)

    # Regime distribution
    if "regime" in eq_log.columns:
        rc = eq_log["regime"].value_counts(normalize=True) * 100
        regime_dist = {k: round(float(v), 1) for k, v in rc.items()}
    else:
        regime_dist = {}

    # Position anomaly: average scale per asset class at each rebalance
    pos_log = results.get("position_log", [])
    trade_log = pd.DataFrame(results.get("trade_log", []))
    n_trades = len(trade_log)
    total_cost = (
        float(trade_log["cost"].sum()) if "cost" in trade_log.columns and n_trades > 0 else 0.0
    )

    return {
        "label": label,
        "paper_start": paper_start,
        "paper_end": paper_end,
        "performance": perf,
        "detector": det_m,
        "regime_dist": regime_dist,
        "n_trades": n_trades,
        "n_rebalances": len(pos_log),
        "total_cost": round(total_cost, 2),
        "equity_log": results["equity_log"],
        "position_log": pos_log,
    }


# ── Chart generation ──────────────────────────────────────────────────────────


def generate_chart(period_results: list[dict], spy_close: pd.Series) -> str:
    """
    5-row chart: one row per period.
    Each row: equity curve (vs SPY) + choppy score colour bands.
    Bottom summary panel: cross-period bar chart.
    """
    n = len(period_results)
    fig = plt.figure(figsize=(18, 5 * n + 4), facecolor=C["bg"])
    outer = GridSpec(
        n + 1,
        1,
        figure=fig,
        hspace=0.55,
        top=0.94,
        bottom=0.04,
        left=0.06,
        right=0.97,
        height_ratios=[1.0] * n + [0.9],
    )

    for row_i, (res, (_, _, period_color)) in enumerate(
        zip(period_results, [(p[0], p[4], PERIOD_COLORS[i]) for i, p in enumerate(PERIODS)])
    ):
        inner = GridSpec(
            1, 3, figure=fig, subplot_spec=outer[row_i], wspace=0.28, width_ratios=[2.5, 0.7, 0.8]
        )

        ax_eq = fig.add_subplot(inner[0])  # equity curve
        ax_ch = fig.add_subplot(inner[1])  # choppy score timeline
        ax_met = fig.add_subplot(inner[2])  # text metrics

        for ax in [ax_eq, ax_ch]:
            ax.set_facecolor(C["surface"])
            ax.spines[["top", "right", "bottom", "left"]].set_color(C["border"])
            ax.tick_params(colors=C["muted"], labelsize=8)
            ax.grid(True, color=C["grid"], linewidth=0.4, alpha=0.7)
        ax_met.axis("off")
        ax_met.set_facecolor(C["surface"])
        ax_met.spines[["top", "right", "bottom", "left"]].set_color(C["border"])

        eq_log = pd.DataFrame(res["equity_log"]).set_index("date")
        eq_log.index = pd.to_datetime(eq_log.index)

        start = res["paper_start"]
        end = res["paper_end"]
        spy_sub = spy_close.loc[start:end]
        spy_cum = spy_sub / spy_sub.iloc[0]
        strat_cum = eq_log["equity"] / eq_log["equity"].iloc[0]

        # ── Equity curve ──────────────────────────────────────────────────
        ax_eq.plot(
            strat_cum.index,
            strat_cum.values,
            color=period_color,
            lw=2.0,
            label="Strategy",
            zorder=5,
        )
        ax_eq.plot(
            spy_cum.index,
            spy_cum.values,
            color=C["spy"],
            lw=1.3,
            linestyle="--",
            alpha=0.8,
            label="SPY",
            zorder=4,
        )
        ax_eq.axhline(1.0, color=C["muted"], lw=0.6, linestyle=":", alpha=0.5)
        ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}×"))
        ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_eq.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax_eq.get_xticklabels(), rotation=25, ha="right", fontsize=7.5)
        ax_eq.legend(
            loc="upper left",
            fontsize=7.5,
            framealpha=0.9,
            facecolor="white",
            edgecolor=C["border"],
            labelcolor=C["text"],
        )
        ax_eq.set_title(res["label"], fontsize=10, color=C["text"], fontweight="bold", pad=5)

        # Draw choppy score as background shading on equity chart
        ch_s = eq_log["choppy_sc"]
        for date in ch_s.index:
            sc = float(ch_s.get(date, 0.0))
            if sc >= 0.40:
                col, alpha = C["red"], 0.12
            elif sc >= 0.27:
                col, alpha = C["amber"], 0.10
            elif sc >= 0.17:
                col, alpha = "#BCE2E7", 0.08
            else:
                continue
            ax_eq.axvline(date, color=col, alpha=alpha, lw=1.0, zorder=1)

        # ── Choppy score timeline ─────────────────────────────────────────
        ch_dates = ch_s.index.to_pydatetime()
        ch_vals = ch_s.values
        ax_ch.fill_between(ch_dates, ch_vals, 0, alpha=0.6, color=period_color)
        ax_ch.axhline(0.17, color=C["muted"], lw=0.8, linestyle="--", alpha=0.7)
        ax_ch.axhline(0.27, color=C["amber"], lw=0.8, linestyle="--", alpha=0.7)
        ax_ch.axhline(0.40, color=C["red"], lw=0.8, linestyle="--", alpha=0.7)
        ax_ch.set_ylim(0, max(0.50, float(ch_s.max()) * 1.10))
        ax_ch.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax_ch.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_ch.get_xticklabels(), rotation=25, ha="right", fontsize=7)
        ax_ch.set_title("Choppy Score", fontsize=8.5, color=C["text"], fontweight="bold", pad=4)
        ax_ch.set_ylabel("Score", color=C["muted"], fontsize=8)

        # Threshold labels
        for y, label in [(0.17, "YLW"), (0.27, "ORG"), (0.40, "RED")]:
            if float(ch_s.max()) > y:
                ax_ch.text(
                    0.02,
                    y + 0.005,
                    label,
                    transform=ax_ch.get_yaxis_transform(),
                    fontsize=6.5,
                    color=C["muted"],
                    va="bottom",
                )

        # ── Metrics text ──────────────────────────────────────────────────
        p = res["performance"]
        d = res["detector"]
        metrics_lines = [
            ("STRATEGY", C["text"], 10.5, "bold"),
            (f"Return:  {p.get('total_return_pct', 0):+.1f}%", C["text"], 9, "normal"),
            (f"Sharpe:  {p.get('sharpe', 0):.2f}", C["text"], 9, "normal"),
            (f"MaxDD:   {p.get('max_dd_pct', 0):.1f}%", C["text"], 9, "normal"),
            ("", C["text"], 5, "normal"),
            ("SPY", C["spy"], 9, "bold"),
            (f"Return:  {p.get('spy_total_return', 0):+.1f}%", C["muted"], 9, "normal"),
            (f"MaxDD:   {p.get('spy_max_dd_pct', 0):.1f}%", C["muted"], 9, "normal"),
            ("", C["text"], 5, "normal"),
            ("DETECTOR", C["text"], 9, "bold"),
            (f"GREEN:   {d.get('pct_green', 0):.0f}%", C["green"], 8.5, "normal"),
            (f"YELLOW:  {d.get('pct_yellow', 0):.0f}%", C["amber"], 8.5, "normal"),
            (
                f"ORANGE+: {d.get('pct_orange', 0) + d.get('pct_red', 0):.0f}%",
                C["red"],
                8.5,
                "normal",
            ),
            (f"FP rate: {d.get('false_positive_rate_pct', 0):.0f}%", C["muted"], 8.5, "normal"),
            (f"Miss rt: {d.get('miss_rate_pct', 0):.0f}%", C["muted"], 8.5, "normal"),
        ]
        y_pos = 0.97
        for text, color, size, weight in metrics_lines:
            ax_met.text(
                0.06,
                y_pos,
                text,
                transform=ax_met.transAxes,
                fontsize=size,
                color=color,
                fontweight=weight,
                va="top",
                family="monospace",
            )
            y_pos -= (size + 1.5) / 100

    # ── Summary bar chart ─────────────────────────────────────────────────
    ax_sum = fig.add_subplot(outer[n])
    ax_sum.set_facecolor(C["surface"])
    ax_sum.spines[["top", "right", "bottom", "left"]].set_color(C["border"])
    ax_sum.tick_params(colors=C["muted"], labelsize=9)
    ax_sum.grid(True, color=C["grid"], linewidth=0.5, alpha=0.7, axis="y")

    [r["label"].replace(" + ", "\n+\n").replace(" - ", "\n−\n") for r in period_results]
    labels_short = [
        "COVID\nCrash",
        "Post-COVID\nBull",
        "Rate Hike\nBear",
        "Bear-to-Bull\nRecovery",
        "AI Bull\nRun",
    ]
    sharpes = [r["performance"].get("sharpe", 0) for r in period_results]
    spy_sh = [r["performance"].get("spy_sharpe", 0) for r in period_results]
    x = np.arange(len(period_results))
    w = 0.32

    bars1 = ax_sum.bar(
        x - w / 2, sharpes, w, label="Strategy Sharpe", color=PERIOD_COLORS, alpha=0.85, zorder=3
    )
    bars2 = ax_sum.bar(
        x + w / 2, spy_sh, w, label="SPY Sharpe", color=C["spy"], alpha=0.55, zorder=3
    )
    ax_sum.axhline(0, color=C["muted"], lw=0.8, alpha=0.6)
    ax_sum.axhline(
        0.5, color=PERIOD_COLORS[3], lw=1.0, linestyle=":", alpha=0.5, label="0.5 target"
    )

    for bar in bars1:
        v = bar.get_height()
        ax_sum.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.06 if v >= 0 else -0.12),
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
            color=C["text"],
            fontweight="bold",
        )
    for bar in bars2:
        v = bar.get_height()
        ax_sum.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.06 if v >= 0 else -0.12),
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
            color=C["muted"],
        )

    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(labels_short, fontsize=9, color=C["text"])
    ax_sum.set_ylabel("Sharpe Ratio", color=C["muted"], fontsize=10)
    ax_sum.legend(
        loc="upper right",
        framealpha=0.92,
        facecolor="white",
        edgecolor=C["border"],
        fontsize=9,
        labelcolor=C["text"],
    )
    ax_sum.set_title(
        "Sharpe Ratio per Regime — Strategy vs SPY Benchmark",
        fontsize=11,
        color=C["text"],
        fontweight="bold",
        pad=7,
    )

    # ── Master title ──────────────────────────────────────────────────────
    fig.text(
        0.06,
        0.975,
        "Walk-Forward Paper Emulation 2020–2024 — Five Regime Stress Test",
        fontsize=14,
        fontweight="bold",
        color=C["text"],
    )
    fig.text(
        0.06,
        0.960,
        "Full strategy stack  ·  IS-validated params  ·  "
        "WF-calibrated PositionAnomalyScorer  ·  ChoppyRegimeDetector v2  ·  "
        "$100k initial  ·  Realistic costs  ·  Strictly OOS",
        fontsize=9,
        color=C["muted"],
    )

    Path("results").mkdir(exist_ok=True)
    out = "results/wf_emulation_chart.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    log.info(f"Chart saved → {out}")
    return out


# ── Cross-period summary table ────────────────────────────────────────────────


def print_summary(period_results: list[dict]) -> dict:
    """Print a formatted summary table and return the summary dict."""
    print(f"\n{'=' * 110}")
    print("WALK-FORWARD EMULATION SUMMARY — 5 REGIME TYPES (2020–2024)")
    print(f"{'=' * 110}")
    print(
        f"{'Period':<28} {'Strat%':>7} {'Sh':>6} {'DD%':>7} | "
        f"{'SPY%':>6} {'SPYSh':>6} {'SPY DD':>7} | "
        f"{'Alpha':>7} | "
        f"{'GREEN%':>7} {'FP%':>5} {'Miss%':>6}"
    )
    print("-" * 110)

    for res in period_results:
        p = res["performance"]
        d = res["detector"]
        alpha = p.get("alpha_pct", 0)
        alpha_arrow = "▲" if alpha > 0 else "▼"
        "█" * int(d.get("pct_green", 0) / 10)
        print(
            f"{res['label']:<28} "
            f"{p.get('total_return_pct', 0):>+7.1f}% "
            f"{p.get('sharpe', 0):>6.2f} "
            f"{p.get('max_dd_pct', 0):>7.1f}% | "
            f"{p.get('spy_total_return', 0):>+6.1f}% "
            f"{p.get('spy_sharpe', 0):>6.2f} "
            f"{p.get('spy_max_dd_pct', 0):>7.1f}% | "
            f"{alpha_arrow}{abs(alpha):>6.1f}% | "
            f"{d.get('pct_green', 0):>7.1f}% "
            f"{d.get('false_positive_rate_pct', 0):>5.1f}% "
            f"{d.get('miss_rate_pct', 0):>6.1f}%"
        )

    # Aggregate stats
    sharpes = [r["performance"].get("sharpe", 0) for r in period_results]
    max_dds = [r["performance"].get("max_dd_pct", 0) for r in period_results]
    alphas = [r["performance"].get("alpha_pct", 0) for r in period_results]
    fp_rates = [r["detector"].get("false_positive_rate_pct", 0) for r in period_results]
    miss_r = [r["detector"].get("miss_rate_pct", 0) for r in period_results]
    green_p = [r["detector"].get("pct_green", 0) for r in period_results]

    print("-" * 110)
    print(
        f"{'MEAN':<28} "
        f"{'':>8} {np.mean(sharpes):>6.2f} {np.mean(max_dds):>7.1f}% | "
        f"{'':>7} {'':>6} {'':>8} | "
        f"{'':>3}{np.mean(alphas):>+6.1f}% | "
        f"{np.mean(green_p):>7.1f}% "
        f"{np.mean(fp_rates):>5.1f}% "
        f"{np.mean(miss_r):>6.1f}%"
    )

    # Detector stress test verdict
    print(f"\n{'=' * 110}")
    print("ChoppyRegimeDetector STRESS TEST VERDICT")
    print(f"{'=' * 110}")
    for res in period_results:
        p = res["performance"]
        d = res["detector"]
        orange_red = d.get("pct_orange", 0) + d.get("pct_red", 0)
        yellow_plus = d.get("pct_yellow", 0) + orange_red
        fp = d.get("false_positive_rate_pct", 0)
        d.get("miss_rate_pct", 0)

        if "COVID" in res["label"]:
            verdict = "✓ PASS" if orange_red > 25 else "✗ FAIL — detector didn't fire during crash"
            detail = f"ORANGE+RED={orange_red:.0f}% (expect >25%)"
        elif "Post-COVID" in res["label"]:
            verdict = "✓ PASS" if fp < 35 else "✗ FAIL — too many false positives in bull"
            detail = f"FP_rate={fp:.0f}% (expect <35%)"
        elif "Rate Hike" in res["label"]:
            verdict = "✓ PASS" if yellow_plus > 45 else "✗ FAIL — missed sustained bear"
            detail = f"YELLOW+={yellow_plus:.0f}% (expect >45%)"
        elif "Recovery" in res["label"]:
            verdict = "✓ PASS" if fp < 40 else "✗ FAIL — too noisy in recovery"
            detail = f"FP_rate={fp:.0f}% (expect <40%)"
        elif "AI Bull" in res["label"]:
            verdict = "✓ PASS" if fp < 30 else "✗ FAIL — too many false positives in calm bull"
            detail = f"FP_rate={fp:.0f}% (expect <30%)"
        else:
            verdict = "—"
            detail = ""

        print(f"  {res['label']:<28}  {verdict}  |  {detail}")

    return {
        "mean_sharpe": round(float(np.mean(sharpes)), 3),
        "mean_max_dd": round(float(np.mean(max_dds)), 2),
        "mean_alpha": round(float(np.mean(alphas)), 2),
        "periods_beat_spy": sum(1 for a in alphas if a > 0),
        "detector_mean_green_pct": round(float(np.mean(green_p)), 1),
        "detector_mean_fp_rate": round(float(np.mean(fp_rates)), 1),
        "detector_mean_miss_rate": round(float(np.mean(miss_r)), 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    config = load_config("config/settings.yaml")

    log.info("Loading all data (full history)...")
    all_data: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        df = load_sym(sym)
        if df is not None and len(df) > 200:
            all_data[sym] = df
    log.info(f"Loaded {len(all_data)} symbols")

    spy_close = all_data["SPY"]["Close"]
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)

    period_results = []
    for label, warmup, start, end, _rtype, desc in PERIODS:
        log.info(f"\nStarting period: {label} ({start} → {end})")
        log.info(f"  {desc}")
        try:
            res = run_period(label, warmup, start, end, all_data, config)
            period_results.append(res)
            p = res["performance"]
            d = res["detector"]
            log.info(
                f"  DONE: return={p.get('total_return_pct', 0):+.1f}%  "
                f"Sh={p.get('sharpe', 0):.2f}  DD={p.get('max_dd_pct', 0):.1f}%  "
                f"GREEN={d.get('pct_green', 0):.0f}%  FP={d.get('false_positive_rate_pct', 0):.0f}%"
            )
        except Exception as e:
            log.error(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    summary = print_summary(period_results)

    # Outputs
    Path("results").mkdir(exist_ok=True)
    chart_path = generate_chart(period_results, spy_close)

    for res in period_results:
        safe = res["label"].replace(" ", "_").replace("/", "-").replace("+", "plus")[:30]
        out = {k: v for k, v in res.items() if k not in ("equity_log", "position_log")}
        with open(f"results/wf_{safe}.json", "w") as f:
            json.dump(out, f, indent=2, default=str)

    full_summary = {
        "periods": [
            {k: v for k, v in r.items() if k not in ("equity_log", "position_log")}
            for r in period_results
        ],
        "aggregate": summary,
    }
    with open("results/wf_emulation_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2, default=str)

    print(f"\n  Chart: {chart_path}")
    print("  Summary: results/wf_emulation_summary.json")
    return period_results, summary


if __name__ == "__main__":
    main()
