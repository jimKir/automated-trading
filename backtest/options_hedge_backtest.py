"""
Protective put hedge backtest.
Compares unhedged vs hedged strategy across:
  - COVID crash: Feb 19 – May 31 2020
  - Rate hike 2022: Jan 1 – Dec 31 2022

Put pricing via Black-Scholes with VIX-derived IV.
ChoppyDetector score approximated from price/vol data.

Run: python backtest/options_hedge_backtest.py --save-results
"""
import os, sys, json, math, warnings, argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date, timedelta, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from execution.options_hedge import (
    ProtectivePutHedge, estimate_put_premium, HedgeState
)

PERIODS = [
    {"name": "COVID_Crash_2020",
     "start": "2020-01-15", "end": "2020-06-30",
     "label": "COVID Crash (Jan–Jun 2020)"},
    {"name": "RateHike_2022",
     "start": "2022-01-01", "end": "2022-12-31",
     "label": "Rate Hike Cycle (2022)"},
]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(start: str, end: str) -> pd.DataFrame:
    from data.data_store import DataStore
    store = DataStore()

    spy = store.load("SPY", start_date=start, end_date=end)
    tlt = store.load("TLT", start_date=start, end_date=end)
    vix = store.load("VIX", start_date=start, end_date=end)
    gld = store.load("GLD", start_date=start, end_date=end)

    df = pd.DataFrame(index=spy.index)
    df["spy"]  = spy["close"]
    df["tlt"]  = tlt["close"] if tlt is not None else np.nan
    df["vix"]  = vix["close"] if vix is not None else 20.0
    df["gld"]  = gld["close"] if gld is not None else np.nan

    df["vix"]  = df["vix"].ffill().fillna(20.0)
    df = df.ffill().dropna(subset=["spy"])
    return df

# ── ChoppyDetector proxy ──────────────────────────────────────────────────────

def compute_choppy_score(df: pd.DataFrame) -> pd.Series:
    """
    Approximate ChoppyDetector v4 score from price data.
    Uses vol spike + yield curve proxy + SPY momentum.
    """
    spy = df["spy"]
    vix = df["vix"]

    # Vol spike: 5d realised vol vs 60d baseline
    ret = spy.pct_change()
    vol5  = ret.rolling(5).std()  * np.sqrt(252)
    vol60 = ret.rolling(60).std() * np.sqrt(252)
    vol_spike = ((vol5 / vol60.replace(0, np.nan)) - 1).clip(0, 3) / 3

    # VIX level stress
    vix_stress = ((vix - 15) / 35).clip(0, 1)

    # VIX spike (5d change)
    vix_spike = vix.pct_change(5).clip(0, 2) / 2

    # SPY 21d momentum (negative = stress)
    mom = (-spy.pct_change(21) / 0.15).clip(0, 1)

    score = (0.30 * vol_spike +
             0.25 * vix_stress +
             0.25 * vix_spike +
             0.20 * mom).fillna(0).clip(0, 1)

    return score

# ── Portfolio simulation ──────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, use_hedge: bool = True,
             initial_equity: float = 100_000.0) -> pd.DataFrame:
    """
    Simulate multi-asset portfolio with optional put hedge.
    Portfolio: 40% SPY + 20% TLT + 15% GLD + 25% cash
    Rebalance weekly.
    """
    weights = {"spy": 0.40, "tlt": 0.20, "gld": 0.15}
    cash_w  = 0.25

    # Daily returns per asset
    rets = {}
    for a in weights:
        col = df[a] if a in df.columns else None
        if col is not None:
            rets[a] = col.pct_change().fillna(0)
        else:
            rets[a] = pd.Series(0.0, index=df.index)

    choppy = compute_choppy_score(df)
    hedge  = ProtectivePutHedge(broker=None, dry_run=True)

    equity      = initial_equity
    results     = []
    put_pnl_day = 0.0
    prev_score  = 0.0

    for i, (dt, row) in enumerate(df.iterrows()):
        spy_price = row["spy"]
        vix       = row["vix"]
        score     = choppy.iloc[i]
        today     = dt.date()

        # Portfolio return (weighted)
        port_ret = sum(weights[a] * rets[a].iloc[i] for a in weights)

        # Hedge logic
        hedge_cost_today = 0.0
        if use_hedge:
            # Check for regime transition
            prev_regime = hedge._classify(prev_score)
            new_regime  = hedge._classify(score)

            if prev_regime != new_regime:
                pos = hedge.on_regime_change(
                    new_score=score,
                    spy_price=spy_price,
                    portfolio_equity=equity,
                    vix_level=vix,
                    today=today,
                )
                # Deduct premium on open day
                if pos:
                    hedge_cost_today = pos.total_cost / equity

            # Daily mark-to-market of puts
            put_pnl_day = hedge.get_hedge_pnl(spy_price, today) / equity

        # Update equity
        equity *= (1 + port_ret - hedge_cost_today)
        prev_score = score

        results.append({
            "date":        dt,
            "equity":      equity,
            "port_ret":    port_ret,
            "spy_price":   spy_price,
            "choppy_score":score,
            "hedge_active":len(hedge.state.active_puts) > 0,
            "put_pnl":     put_pnl_day,
        })

    out = pd.DataFrame(results).set_index("date")
    out.attrs["hedge_summary"] = hedge.summary()
    return out

# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(df: pd.DataFrame, label: str) -> dict:
    ret  = df["equity"].pct_change().dropna()
    cum  = df["equity"] / df["equity"].iloc[0]
    dd   = (cum - cum.cummax()) / cum.cummax()
    ann_r = ret.mean() * 252
    ann_v = ret.std()  * np.sqrt(252)
    return {
        "label":        label,
        "total_return": round(float(cum.iloc[-1] - 1), 4),
        "sharpe":       round(ann_r / ann_v if ann_v > 0 else 0, 3),
        "max_dd":       round(float(dd.min()), 4),
        "calmar":       round(ann_r / abs(dd.min()) if dd.min() < 0 else 0, 3),
        "worst_day":    round(float(ret.min()), 4),
    }

# ── Charts ────────────────────────────────────────────────────────────────────

def make_chart(period: dict,
               unhedged: pd.DataFrame,
               hedged: pd.DataFrame,
               save_path: str):

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.suptitle(
        f"Protective Put Hedge — {period['label']}\n"
        f"ChoppyDetector ORANGE/RED trigger | 2% OTM SPY put | 21 DTE",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Cumulative returns ────────────────────────────────────────
    ax = axes[0]
    uh_cum = unhedged["equity"] / unhedged["equity"].iloc[0]
    hd_cum = hedged["equity"]   / hedged["equity"].iloc[0]

    ax.plot(uh_cum.index, uh_cum.values, color="gray",
            lw=2, ls="--", label=f"Unhedged  (MaxDD={metrics(unhedged,'')['max_dd']:.1%})")
    ax.plot(hd_cum.index, hd_cum.values, color="steelblue",
            lw=2.5, label=f"Hedged    (MaxDD={metrics(hedged,'')['max_dd']:.1%})")

    # Shade hedge-active periods
    hedge_on = hedged["hedge_active"]
    for i in range(len(hedge_on)-1):
        if hedge_on.iloc[i]:
            ax.axvspan(hedge_on.index[i], hedge_on.index[i+1],
                       alpha=0.12, color="green", lw=0)

    ax.set_ylabel("Cumulative Return (1.0 = start)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title("Cumulative Return — green shading = hedge active", fontsize=10)

    # ── Panel 2: ChoppyDetector score ─────────────────────────────────────
    ax2 = axes[1]
    score = hedged["choppy_score"]
    ax2.fill_between(score.index, score, alpha=0.4,
                     color="salmon", label="Choppy score")
    ax2.axhline(0.229, color="orange", ls="--", lw=1.2,
                label="ORANGE (0.229)")
    ax2.axhline(0.296, color="red",    ls="--", lw=1.2,
                label="RED (0.296)")
    ax2.axhline(0.192, color="green",  ls="--", lw=1.0,
                label="GREEN (0.192)")
    ax2.set_ylim(0, 0.8)
    ax2.set_ylabel("Choppy Score")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("ChoppyDetector Score (proxy)", fontsize=10)

    # ── Panel 3: Hedge P&L contribution ───────────────────────────────────
    ax3 = axes[2]
    hedge_pnl = hedged["put_pnl"]
    colors = ["green" if v >= 0 else "red" for v in hedge_pnl]
    ax3.bar(hedge_pnl.index, hedge_pnl * 100,
            color=colors, alpha=0.7, width=1)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylabel("Put P&L (%)")
    ax3.set_title("Daily Hedge P&L contribution", fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart → {save_path}")


def make_summary_chart(all_results: list,
                       save_path: str = "results/options_hedge_summary.png"):
    periods  = [r["period"]   for r in all_results]
    uh_dd    = [r["unhedged"]["max_dd"]    * 100 for r in all_results]
    hd_dd    = [r["hedged"]["max_dd"]      * 100 for r in all_results]
    uh_sh    = [r["unhedged"]["sharpe"]           for r in all_results]
    hd_sh    = [r["hedged"]["sharpe"]             for r in all_results]
    net_cost = [r["hedge_cost_pct"]               for r in all_results]

    x  = np.arange(len(periods))
    w  = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Protective Put Hedge — Impact Summary",
                 fontsize=13, fontweight="bold")

    # MaxDD comparison
    ax = axes[0]
    ax.bar(x - w/2, uh_dd, w, label="Unhedged", color="gray",      alpha=0.8)
    ax.bar(x + w/2, hd_dd, w, label="Hedged",   color="steelblue", alpha=0.8)
    for i, (u, h) in enumerate(zip(uh_dd, hd_dd)):
        delta = h - u
        ax.annotate(f"{delta:+.1f}%", xy=(i+w/2, h),
                    ha="center", va="bottom", fontsize=9,
                    color="green" if delta > 0 else "red")
    ax.set_xticks(x); ax.set_xticklabels(periods, rotation=10)
    ax.set_ylabel("Max Drawdown (%)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Max Drawdown")

    # Sharpe comparison
    ax = axes[1]
    ax.bar(x - w/2, uh_sh, w, label="Unhedged", color="gray",      alpha=0.8)
    ax.bar(x + w/2, hd_sh, w, label="Hedged",   color="steelblue", alpha=0.8)
    for i, (u, h) in enumerate(zip(uh_sh, hd_sh)):
        delta = h - u
        ax.annotate(f"{delta:+.2f}", xy=(i+w/2, max(h,0)+0.02),
                    ha="center", va="bottom", fontsize=9,
                    color="green" if delta > 0 else "red")
    ax.set_xticks(x); ax.set_xticklabels(periods, rotation=10)
    ax.set_ylabel("Sharpe Ratio"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Sharpe Ratio")

    # Hedge cost
    ax = axes[2]
    bars = ax.bar(periods, net_cost, color=["green" if v < 0 else "red"
                                             for v in net_cost], alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars, net_cost):
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.05 if v >= 0 else -0.15),
                f"{v:+.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Net hedge cost (% of portfolio)")
    ax.set_title("Net Hedge Cost\n(negative = put paid off)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Summary chart → {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run(save: bool = False):
    print("\n" + "="*65)
    print("  PROTECTIVE PUT HEDGE BACKTEST")
    print("  ChoppyDetector ORANGE/RED trigger | 2%/5% OTM SPY puts")
    print("="*65)

    all_results = []

    for period in PERIODS:
        print(f"\n{'─'*65}")
        print(f"  Period: {period['label']}")
        print(f"  Range:  {period['start']} → {period['end']}")

        try:
            df = load_data(period["start"], period["end"])
            print(f"  Loaded: {len(df)} trading days | "
                  f"SPY ${df['spy'].iloc[0]:.0f} → ${df['spy'].iloc[-1]:.0f}")
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            continue

        # Run both simulations
        unhedged = simulate(df, use_hedge=False)
        hedged   = simulate(df, use_hedge=True)

        um = metrics(unhedged, "Unhedged")
        hm = metrics(hedged,   "Hedged")
        hs = hedged.attrs.get("hedge_summary", {})

        total_paid  = hs.get("total_premium_paid", 0)
        total_payoff= hs.get("total_payoff_received", 0)
        net_cost_pct= (total_paid - total_payoff) / 100_000 * 100

        print(f"\n  {'Metric':<20} {'Unhedged':>12} {'Hedged':>12} {'Delta':>10}")
        print(f"  {'─'*56}")
        for key in ["total_return","sharpe","max_dd","worst_day","calmar"]:
            u = um[key]; h = hm[key]
            d = h - u
            flag = "✅" if (key in ("sharpe","total_return","calmar") and d > 0) or \
                           (key in ("max_dd","worst_day") and d > 0) else \
                   "⚠" if abs(d) < 0.005 else "❌"
            print(f"  {key:<20} {u:>12.4f} {h:>12.4f} {flag}{d:>+9.4f}")

        print(f"\n  Hedge stats:")
        print(f"    Puts opened:     {hs.get('n_opened',0)}")
        print(f"    Premium paid:    ${total_paid:,.0f}")
        print(f"    Payoff received: ${total_payoff:,.0f}")
        print(f"    Net cost:        ${total_paid-total_payoff:+,.0f} "
              f"({net_cost_pct:+.2f}% of portfolio)")

        result = {
            "period":        period["name"],
            "label":         period["label"],
            "unhedged":      um,
            "hedged":        hm,
            "delta_sharpe":  round(hm["sharpe"] - um["sharpe"], 3),
            "delta_max_dd":  round(hm["max_dd"]  - um["max_dd"],  4),
            "hedge_cost_pct":round(net_cost_pct, 3),
            "hedge_stats":   hs,
        }
        all_results.append(result)

        if save:
            make_chart(
                period, unhedged, hedged,
                f"results/options_hedge_{period['name'].lower()}.png"
            )

    if not all_results:
        print("No results — check data availability")
        return

    # Overall summary
    print(f"\n{'='*65}")
    print("  SUMMARY ACROSS PERIODS")
    print(f"{'='*65}")
    avg_dd_delta = np.mean([r["delta_max_dd"] for r in all_results])
    avg_sh_delta = np.mean([r["delta_sharpe"] for r in all_results])
    avg_cost     = np.mean([r["hedge_cost_pct"] for r in all_results])
    n_dd_better  = sum(1 for r in all_results if r["delta_max_dd"] > 0.005)

    print(f"  MaxDD improved:   {n_dd_better}/{len(all_results)} periods | "
          f"Mean ΔMaxDD: {avg_dd_delta:+.4f}")
    print(f"  Mean ΔSharpe:     {avg_sh_delta:+.3f}")
    print(f"  Mean net cost:    {avg_cost:+.2f}% of portfolio per period")

    verdict = ("ADOPT — hedge pays off significantly in stress periods" if n_dd_better == len(all_results)
               else "CONDITIONAL — hedge helps in some but not all periods" if n_dd_better > 0
               else "REJECT — hedge costs exceed drawdown protection")
    print(f"  Verdict: {verdict}")
    print(f"{'='*65}")

    if save:
        make_summary_chart(all_results)
        output = {
            "run_date":   datetime.now().isoformat(),
            "methodology":"ChoppyDetector proxy score + Black-Scholes put pricing",
            "periods":    all_results,
            "summary": {
                "mean_dd_delta":    round(float(avg_dd_delta), 4),
                "mean_sh_delta":    round(float(avg_sh_delta), 4),
                "mean_net_cost_pct":round(float(avg_cost), 3),
                "n_dd_improved":    n_dd_better,
                "verdict":          verdict,
            }
        }
        os.makedirs("results", exist_ok=True)
        with open("results/options_hedge_results.json","w") as f:
            import json
            json.dump(output, f, indent=2, default=str)
        print("  JSON → results/options_hedge_results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-results", action="store_true")
    args = p.parse_args()
    run(save=args.save_results)
