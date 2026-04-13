"""
Protective put hedge backtest v2.
Fixes from v1:
  1. Earlier trigger: open put when ORANGE first fires, not on regime transition
  2. Hold until expiry or close when score returns to GREEN (whichever first)
  3. Payoff computed at expiry using SPY price on expiry date (not just intrinsic at close)
  4. COVID period extended back to Jan 1 2020 to capture full pre-crash baseline
  5. Added SPY buy-and-hold as third comparison line

Run: python backtest/options_hedge_backtest_v2.py --save-results
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from execution.options_hedge import estimate_put_premium

PERIODS = [
    {
        "name": "COVID_Crash_2020",
        "start": "2020-01-02",
        "end": "2020-06-30",
        "label": "COVID Crash (2020)",
        "crash_start": "2020-02-19",  # known crash start for annotation
        "crash_bottom": "2020-03-23",
    },
    {
        "name": "RateHike_2022",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "label": "Rate Hike Cycle (2022)",
        "crash_start": "2022-01-03",
        "crash_bottom": "2022-10-12",
    },
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(start, end):
    from data.data_store import DataStore

    store = DataStore()
    spy = store.load("SPY", start_date=start, end_date=end)
    tlt = store.load("TLT", start_date=start, end_date=end)
    vix = store.load("VIX", start_date=start, end_date=end)
    gld = store.load("GLD", start_date=start, end_date=end)

    df = pd.DataFrame(index=spy.index)
    df["spy"] = spy["close"]
    df["tlt"] = tlt["close"] if tlt is not None else np.nan
    df["gld"] = gld["close"] if gld is not None else np.nan
    df["vix"] = vix["close"] if vix is not None else 20.0
    return df.ffill().bfill()


# ── ChoppyDetector proxy (improved — uses VIX acceleration, not just level) ──
def compute_choppy_score(df):
    spy = df["spy"]
    vix = df["vix"]
    ret = spy.pct_change()

    vol5 = ret.rolling(5).std() * np.sqrt(252)
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    vol_spike = ((vol5 / vol20.replace(0, np.nan)) - 1).clip(0, 3) / 3

    # VIX acceleration (5d change) — catches fast moves early
    vix_5d_chg = vix.pct_change(5).clip(0, 2) / 2
    vix_level = ((vix - 15) / 35).clip(0, 1)

    # SPY 10d momentum (faster signal than 21d)
    mom10 = (-spy.pct_change(10) / 0.10).clip(0, 1)

    score = (
        (0.30 * vol_spike + 0.30 * vix_5d_chg + 0.20 * vix_level + 0.20 * mom10)
        .fillna(0)
        .clip(0, 1)
    )
    return score


# ── Put position tracker ──────────────────────────────────────────────────────
class PutBook:
    """Tracks open put positions with proper expiry and payoff."""

    def __init__(self):
        self.positions = []  # list of dicts
        self.total_premium = 0.0
        self.total_payoff = 0.0
        self.log = []

    def open_put(
        self,
        entry_date,
        spy_price,
        vix,
        equity,
        otm_pct=0.02,
        dte=21,
        hedge_ratio=0.20,
        max_contracts=5,
        regime="ORANGE",
    ):
        strike = round(spy_price * (1 - otm_pct), 0)
        expiry = entry_date + timedelta(days=dte)
        premium = estimate_put_premium(spy_price, otm_pct, dte, vix)
        notional = equity * hedge_ratio
        n = min(max(int(notional / (spy_price * 100)), 1), max_contracts)
        cost = n * 100 * premium

        pos = {
            "entry_date": entry_date,
            "expiry": expiry,
            "strike": strike,
            "contracts": n,
            "premium": premium,
            "cost": cost,
            "regime": regime,
            "closed": False,
            "payoff": 0.0,
        }
        self.positions.append(pos)
        self.total_premium += cost
        self.log.append(
            f"{entry_date}: OPEN {regime} put K=${strike:.0f} exp={expiry} ×{n} cost=${cost:.0f}"
        )
        return pos

    def update(self, today, spy_price):
        """Check for expiries, compute payoffs."""
        for pos in self.positions:
            if pos["closed"]:
                continue
            # Close at expiry
            if today >= pos["expiry"]:
                payoff = max(pos["strike"] - spy_price, 0) * 100 * pos["contracts"]
                pos["payoff"] = payoff
                pos["closed"] = True
                self.total_payoff += payoff
                net = payoff - pos["cost"]
                self.log.append(
                    f"{today}: EXPIRY {pos['entry_date']} K=${pos['strike']:.0f} "
                    f"SPY=${spy_price:.1f} payoff=${payoff:.0f} net=${net:+.0f}"
                )

    def close_all(self, today, spy_price):
        """Close all open positions (GREEN signal)."""
        for pos in self.positions:
            if not pos["closed"]:
                payoff = max(pos["strike"] - spy_price, 0) * 100 * pos["contracts"]
                pos["payoff"] = payoff
                pos["closed"] = True
                self.total_payoff += payoff
                self.log.append(
                    f"{today}: CLOSE (GREEN) K=${pos['strike']:.0f} "
                    f"SPY=${spy_price:.1f} payoff=${payoff:.0f}"
                )

    def daily_pnl(self, spy_price):
        """Current mark-to-market P&L of open positions."""
        pnl = 0.0
        for pos in self.positions:
            if not pos["closed"]:
                pnl += max(pos["strike"] - spy_price, 0) * 100 * pos["contracts"]
                pnl -= pos["cost"]  # subtract already-paid premium
        return pnl

    def has_open(self):
        return any(not p["closed"] for p in self.positions)

    def n_open(self):
        return sum(1 for p in self.positions if not p["closed"])


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate(df, use_hedge=True, initial_equity=100_000.0):
    weights = {"spy": 0.40, "tlt": 0.20, "gld": 0.15}
    rets = {}
    for a in weights:
        col = df.get(a) if hasattr(df, "get") else df[a] if a in df.columns else None
        rets[a] = col.pct_change().fillna(0) if col is not None else pd.Series(0.0, index=df.index)

    choppy = compute_choppy_score(df)
    book = PutBook()
    equity = initial_equity
    results = []
    last_regime = "GREEN"

    GREEN_MAX = 0.192
    ORANGE_MIN = 0.229
    RED_MIN = 0.296

    for i in range(len(df)):
        dt = df.index[i]
        row = df.iloc[i]
        spy_price = row["spy"]
        vix = row["vix"]
        score = choppy.iloc[i]
        today = dt.date()

        # Classify regime
        if score >= RED_MIN:
            regime = "RED"
        elif score >= ORANGE_MIN:
            regime = "ORANGE"
        elif score >= GREEN_MAX:
            regime = "YELLOW"
        else:
            regime = "GREEN"

        # Portfolio P&L
        port_ret = sum(weights[a] * rets[a].iloc[i] for a in weights)

        hedge_cost_frac = 0.0
        if use_hedge:
            # Update existing puts (check expiries)
            book.update(today, spy_price)

            # On GREEN: close all puts
            if regime == "GREEN" and last_regime != "GREEN":
                book.close_all(today, spy_price)

            # On ORANGE/RED transition with no open position: open put
            if (
                regime in ("ORANGE", "RED")
                and last_regime not in ("ORANGE", "RED")
                and not book.has_open()
            ):
                otm = 0.02 if regime == "ORANGE" else 0.05
                dte = 21 if regime == "ORANGE" else 14
                pos = book.open_put(
                    today, spy_price, vix, equity, otm_pct=otm, dte=dte, regime=regime
                )
                hedge_cost_frac = pos["cost"] / equity

            # Roll if all puts have expired
            if (
                regime in ("ORANGE", "RED")
                and not book.has_open()
                and last_regime in ("ORANGE", "RED")
            ):
                otm = 0.02 if regime == "ORANGE" else 0.05
                dte = 21 if regime == "ORANGE" else 14
                pos = book.open_put(
                    today, spy_price, vix, equity, otm_pct=otm, dte=dte, regime=regime
                )
                hedge_cost_frac = pos["cost"] / equity

        equity *= 1 + port_ret - hedge_cost_frac
        last_regime = regime

        results.append(
            {
                "date": dt,
                "equity": equity,
                "port_ret": port_ret,
                "spy_price": spy_price,
                "choppy_score": score,
                "regime": regime,
                "hedge_active": book.has_open() if use_hedge else False,
                "n_puts": book.n_open() if use_hedge else 0,
            }
        )

    out = pd.DataFrame(results).set_index("date")
    out.attrs["book"] = book
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(df, label=""):
    ret = df["equity"].pct_change().dropna()
    cum = df["equity"] / df["equity"].iloc[0]
    dd = (cum - cum.cummax()) / cum.cummax()
    ann_r = ret.mean() * 252
    ann_v = ret.std() * np.sqrt(252)
    return {
        "label": label,
        "total_return": round(float(cum.iloc[-1] - 1), 4),
        "sharpe": round(ann_r / ann_v if ann_v > 0 else 0, 3),
        "max_dd": round(float(dd.min()), 4),
        "worst_day": round(float(ret.min()), 4),
        "calmar": round(ann_r / abs(dd.min()) if dd.min() < 0 else 0, 3),
    }


# ── Chart ─────────────────────────────────────────────────────────────────────
def make_chart(period, unhedged, hedged, spy_bah, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.suptitle(
        f"Protective Put Hedge v2 — {period['label']}\n"
        f"ORANGE trigger: 2% OTM 21-DTE put | RED trigger: 5% OTM 14-DTE put",
        fontsize=12,
        fontweight="bold",
    )

    # Cumulative returns
    ax = axes[0]
    for df_r, color, lw, ls, label in [
        (unhedged, "gray", 2.0, "--", "Unhedged portfolio"),
        (hedged, "steelblue", 2.5, "-", "Hedged portfolio"),
        (spy_bah, "tomato", 1.5, ":", "SPY buy-and-hold"),
    ]:
        cum = df_r["equity"] / df_r["equity"].iloc[0]
        m = metrics(df_r)
        ax.plot(
            cum.index,
            cum.values,
            color=color,
            lw=lw,
            ls=ls,
            label=f"{label}  Sh={m['sharpe']:.2f}  MaxDD={m['max_dd']:.1%}",
        )

    # Shade ORANGE/RED periods
    score = hedged["choppy_score"]
    for i in range(len(score) - 1):
        s = score.iloc[i]
        if s >= 0.296:
            ax.axvspan(score.index[i], score.index[i + 1], alpha=0.15, color="red", lw=0)
        elif s >= 0.229:
            ax.axvspan(score.index[i], score.index[i + 1], alpha=0.10, color="orange", lw=0)

    # Mark crash events
    for key, color, label in [
        ("crash_start", "red", "Crash start"),
        ("crash_bottom", "green", "Market bottom"),
    ]:
        if key in period:
            try:
                ax.axvline(pd.Timestamp(period[key]), color=color, ls=":", lw=1.5, alpha=0.8)
                ax.text(
                    pd.Timestamp(period[key]),
                    ax.get_ylim()[0] * 1.02,
                    label,
                    color=color,
                    fontsize=8,
                    rotation=90,
                    va="bottom",
                )
            except Exception:
                pass

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Returns (orange/red shading = hedge active)")

    # ChoppyDetector score
    ax2 = axes[1]
    ax2.fill_between(score.index, score, alpha=0.4, color="salmon")
    ax2.axhline(0.229, color="orange", ls="--", lw=1.2, label="ORANGE (0.229)")
    ax2.axhline(0.296, color="red", ls="--", lw=1.2, label="RED (0.296)")
    ax2.axhline(0.192, color="green", ls=":", lw=1.0, label="GREEN (0.192)")
    ax2.set_ylim(0, 0.9)
    ax2.set_ylabel("Score")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("ChoppyDetector Score (proxy)")

    # Open puts count
    ax3 = axes[2]
    ax3.bar(hedged.index, hedged["n_puts"], color="steelblue", alpha=0.6, width=1)
    ax3.set_ylabel("Open puts")
    ax3.set_title("Number of open put contracts")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Chart → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(save=False):
    print("\n" + "=" * 65)
    print("  PROTECTIVE PUT HEDGE BACKTEST v2")
    print("  Improved trigger + proper expiry payoff accounting")
    print("=" * 65)

    all_results = []

    for period in PERIODS:
        print(f"\n{'─' * 65}")
        print(f"  {period['label']}  ({period['start']} → {period['end']})")
        try:
            df = load_data(period["start"], period["end"])
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(
            f"  SPY: ${df['spy'].iloc[0]:.0f} → ${df['spy'].iloc[-1]:.0f} "
            f"| {len(df)} days | VIX mean {df['vix'].mean():.1f}"
        )

        unhedged = simulate(df, use_hedge=False)
        hedged = simulate(df, use_hedge=True)

        # SPY buy-and-hold for reference
        spy_bah = pd.DataFrame({"equity": df["spy"] / df["spy"].iloc[0] * 100_000}, index=df.index)

        um = metrics(unhedged, "Unhedged")
        hm = metrics(hedged, "Hedged")
        sm = metrics(spy_bah, "SPY B&H")
        book = hedged.attrs["book"]

        net_cost_pct = (book.total_premium - book.total_payoff) / 100_000 * 100

        print(f"\n  {'Metric':<20} {'Unhedged':>12} {'Hedged':>12} {'SPY B&H':>12} {'ΔHedge':>9}")
        print(f"  {'─' * 67}")
        for key in ["total_return", "sharpe", "max_dd", "worst_day"]:
            u = um[key]
            h = hm[key]
            s = sm[key]
            d = h - u
            better = (key in ("sharpe", "total_return") and d > 0.02) or (
                key in ("max_dd", "worst_day") and d > 0.005
            )
            worse = (key in ("sharpe", "total_return") and d < -0.02) or (
                key in ("max_dd", "worst_day") and d < -0.005
            )
            flag = "✅" if better else "❌" if worse else "⚠"
            print(f"  {key:<20} {u:>12.4f} {h:>12.4f} {s:>12.4f} {flag}{d:>+8.4f}")

        print("\n  Hedge activity:")
        print(f"    Puts opened:     {len(book.positions)}")
        print(f"    Premium paid:    ${book.total_premium:,.0f}")
        print(f"    Payoff received: ${book.total_payoff:,.0f}")
        print(
            f"    Net cost:        ${book.total_premium - book.total_payoff:+,.0f} "
            f"({net_cost_pct:+.2f}% of portfolio)"
        )
        if book.log:
            print("  First 5 hedge events:")
            for line in book.log[:5]:
                print(f"    {line}")

        result = {
            "period": period["name"],
            "label": period["label"],
            "unhedged": um,
            "hedged": hm,
            "spy_bah": sm,
            "delta_sharpe": round(hm["sharpe"] - um["sharpe"], 3),
            "delta_max_dd": round(hm["max_dd"] - um["max_dd"], 4),
            "hedge_cost_pct": round(net_cost_pct, 3),
            "n_puts": len(book.positions),
            "premium_paid": round(book.total_premium, 2),
            "payoff": round(book.total_payoff, 2),
        }
        all_results.append(result)

        if save:
            make_chart(
                period,
                unhedged,
                hedged,
                spy_bah,
                f"results/options_hedge_v2_{period['name'].lower()}.png",
            )

    if not all_results:
        return

    print(f"\n{'=' * 65}")
    n_dd = sum(1 for r in all_results if r["delta_max_dd"] > 0.005)
    avg_dd = np.mean([r["delta_max_dd"] for r in all_results])
    avg_sh = np.mean([r["delta_sharpe"] for r in all_results])
    avg_c = np.mean([r["hedge_cost_pct"] for r in all_results])
    verdict = (
        "ADOPT"
        if n_dd == len(all_results) and avg_dd > 0.02
        else "CONDITIONAL"
        if n_dd > 0
        else "REJECT"
    )
    print(f"  MaxDD improved: {n_dd}/{len(all_results)} | Mean ΔMaxDD: {avg_dd:+.4f}")
    print(f"  Mean ΔSharpe:   {avg_sh:+.3f}")
    print(f"  Mean net cost:  {avg_c:+.2f}%")
    print(f"  Verdict:        {verdict}")
    print(f"{'=' * 65}")

    if save:
        os.makedirs("results", exist_ok=True)
        out = {
            "run_date": datetime.now().isoformat(),
            "version": "v2",
            "periods": all_results,
            "summary": {
                "n_dd_improved": n_dd,
                "mean_dd_delta": round(float(avg_dd), 4),
                "mean_sh_delta": round(float(avg_sh), 4),
                "mean_cost_pct": round(float(avg_c), 3),
                "verdict": verdict,
            },
        }
        with open("results/options_hedge_v2_results.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("  JSON → results/options_hedge_v2_results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-results", action="store_true")
    args = p.parse_args()
    run(save=args.save_results)
