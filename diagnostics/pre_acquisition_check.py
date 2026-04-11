"""
Pre-Acquisition Check
======================
Run this BEFORE spending money on any new Databento dataset/schema.

Answers three questions for free:
  1. DATA EXISTS?      — does the schema actually have data for your date range?
  2. SIGNAL WORKS?     — does a cheap IC proxy show any predictive power?
  3. WORTH THE COST?   — estimated total spend vs expected IC value

Prints a GO / NO-GO verdict with reasoning.

Usage:
    # Check XNAS.ITCH imbalance before buying (what we should have done)
    PYTHONPATH=. python diagnostics/pre_acquisition_check.py \\
        --dataset XNAS.ITCH --schema imbalance \\
        --start 2023-01-01 --end 2026-04-01

    # Check a new dataset you're considering
    PYTHONPATH=. python diagnostics/pre_acquisition_check.py \\
        --dataset DBEQ.BASIC --schema trades \\
        --start 2024-01-01 --end 2026-04-01

Cost of running this script: $0.00
  - Metadata calls are free
  - Existence probes use limit=1 (Databento does not charge for 0-row responses)
  - IC proxy uses free yfinance price data only
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

KEY = os.environ.get("DATABENTO_KEY", "")

# Cost table (USD/GB) — update if Databento changes pricing
COST_PER_GB = {
    ("XNAS.ITCH", "imbalance"): 16.00,
    ("XNAS.ITCH", "statistics"): 16.00,
    ("XNAS.ITCH", "trades"): 32.00,
    ("XNAS.ITCH", "mbp-1"): 64.00,
    ("OPRA.PILLAR", "ohlcv-1d"): 150.00,
    ("OPRA.PILLAR", "trades"): 280.00,
    ("DBEQ.BASIC", "trades"): 11.00,
    ("DBEQ.BASIC", "ohlcv-1m"): 11.00,
    ("IEXG.TOPS", "trades"): 1.00,
}
GB_PER_SYM_DAY = {
    ("XNAS.ITCH", "imbalance"): 0.0016,
    ("XNAS.ITCH", "statistics"): 0.00002,
    ("XNAS.ITCH", "trades"): 0.008,
    ("OPRA.PILLAR", "ohlcv-1d"): 0.021,
    ("DBEQ.BASIC", "trades"): 0.0005,
}

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

_US_HOLIDAYS = np.array(
    [
        "2022-12-26",
        "2023-01-02",
        "2023-01-16",
        "2023-02-20",
        "2023-04-07",
        "2023-05-29",
        "2023-06-19",
        "2023-07-04",
        "2023-09-04",
        "2023-11-23",
        "2023-11-24",
        "2023-12-25",
        "2024-01-01",
        "2024-01-15",
        "2024-02-19",
        "2024-03-29",
        "2024-05-27",
        "2024-06-19",
        "2024-07-04",
        "2024-09-02",
        "2024-11-28",
        "2024-11-29",
        "2024-12-25",
        "2025-01-01",
        "2025-01-09",
        "2025-01-20",
        "2025-02-17",
        "2025-04-18",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-11-28",
        "2025-12-25",
        "2026-01-01",
        "2026-01-19",
        "2026-02-16",
        "2026-04-03",
    ],
    dtype="datetime64[D]",
)


def is_td(d: date) -> bool:
    return bool(np.is_busday(np.datetime64(d, "D"), holidays=_US_HOLIDAYS))


def trading_days(start: str, end: str) -> list[date]:
    out, cur = [], date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    while cur <= end_d:
        if is_td(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


# ── CHECK 1: Data existence ────────────────────────────────────────────────────


def check_data_exists(client, dataset: str, schema: str, start: str, end: str) -> dict:
    """
    Probe 8 evenly-spaced dates across the range.
    Returns: hit_rate, coverage_pct, first_good, last_good, gaps
    Cost: $0 (limit=1 probes, metadata only)
    """
    days = trading_days(start, end)
    if not days:
        return {"hit_rate": 0, "coverage_pct": 0, "error": "no trading days in range"}

    # Sample 8 evenly-spaced dates + first + last
    step = max(1, len(days) // 6)
    sample = sorted(set([days[0], days[-1]] + days[::step]))[:8]

    hits, misses, gaps = [], [], []
    for d in sample:
        try:
            start_dt = datetime(d.year, d.month, d.day, 0, 0, 0)
            end_dt = datetime(d.year, d.month, d.day, 23, 59, 59)
            store = client.timeseries.get_range(
                dataset=dataset,
                schema=schema,
                start=start_dt,
                end=end_dt,
                symbols=SYMS[:3],  # 3 symbols only
                limit=1,
            )
            df = store.to_df()
            if df.empty:
                misses.append(d)
                gaps.append(str(d))
            else:
                hits.append(d)
        except Exception:
            misses.append(d)
            gaps.append(f"{d}(err)")

    hit_rate = len(hits) / len(sample)
    first_good = min(hits) if hits else None
    last_good = max(hits) if hits else None

    # Estimate coverage: if last_good < end, flag potential cutoff
    cutoff_warning = None
    if last_good and last_good < date.fromisoformat(end) - timedelta(days=30):
        cutoff_warning = f"⚠️  Last data found: {last_good} — possible feed cutoff"

    return {
        "hit_rate": hit_rate,
        "hits": len(hits),
        "misses": len(misses),
        "sample_size": len(sample),
        "first_good": first_good,
        "last_good": last_good,
        "gaps": gaps,
        "cutoff_warning": cutoff_warning,
    }


# ── CHECK 2: IC proxy (free — yfinance only) ──────────────────────────────────


def check_ic_proxy(dataset: str, schema: str, start: str, end: str) -> dict:
    """
    Estimates potential IC without buying the data.
    Uses free price data (yfinance) to compute a PROXY signal
    based on what the schema would contain.

    Schema-specific proxy logic:
      imbalance  → closing auction proxy: last-30min volume imbalance from price
      trades     → trade-flow proxy: close vs open direction × volume
      statistics → opening cross proxy: gap from prev close × volume
      ohlcv-*    → simple momentum: close pct change
    """
    from scipy import stats

    try:
        import yfinance as yf
    except ImportError:
        return {"ic": None, "p": None, "error": "yfinance not installed"}

    # Download price data
    prices = {}
    for s in SYMS:
        try:
            df = yf.download(s, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                prices[s] = df
        except Exception:
            pass

    if len(prices) < 5:
        return {"ic": None, "p": None, "error": f"only {len(prices)} symbols downloaded"}

    # Build proxy signal based on schema type
    rows = []
    for sym, df in prices.items():
        close = df["Close"].squeeze().astype(float)
        volume = df["Volume"].squeeze().astype(float)
        high = df["High"].squeeze().astype(float)
        low = df["Low"].squeeze().astype(float)
        open_ = df["Open"].squeeze().astype(float)

        if "imbalance" in schema:
            # Proxy: signed volume in last part of day = (close - low) / (high - low)
            # High value = buying pressure → bullish imbalance
            proxy = (close - low) / (high - low + 1e-9) - 0.5
        elif "trades" in schema or "mbo" in schema or "mbp" in schema:
            # Proxy: intraday direction × volume surprise
            direction = np.sign(close - open_)
            vol_avg = volume.rolling(20).mean()
            proxy = direction * (volume / (vol_avg + 1e-9) - 1)
        elif "statistics" in schema:
            # Proxy: gap from prev close (opening cross direction)
            proxy = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)
        elif "ohlcv" in schema:
            # Proxy: 5-day momentum
            proxy = close.pct_change(5)
        else:
            proxy = close.pct_change(1)

        fwd = close.pct_change(1).shift(-1)

        for dt in proxy.index:
            if dt in fwd.index:
                pv = float(proxy.get(dt, np.nan))
                fv = float(fwd.get(dt, np.nan))
                if not (np.isnan(pv) or np.isnan(fv)):
                    rows.append({"sym": sym, "date": dt, "proxy": pv, "fwd": fv})

    if len(rows) < 50:
        return {"ic": None, "p": None, "error": "insufficient data"}

    panel = pd.DataFrame(rows)
    ic, p = stats.spearmanr(panel["proxy"], panel["fwd"])

    # Walk-forward by year
    panel["year"] = panel["date"].dt.year
    wf = {}
    for yr, grp in panel.groupby("year"):
        if len(grp) > 20:
            r, _ = stats.spearmanr(grp["proxy"], grp["fwd"])
            wf[yr] = float(r)

    positive_years = sum(1 for v in wf.values() if v > 0)

    return {
        "ic": float(ic),
        "p": float(p),
        "n_obs": len(rows),
        "walk_forward": wf,
        "positive_years": positive_years,
        "total_years": len(wf),
    }


# ── CHECK 3: Cost estimate ─────────────────────────────────────────────────────


def estimate_cost(dataset: str, schema: str, start: str, end: str, n_syms: int) -> dict:
    days = trading_days(start, end)
    n_days = len(days)

    cost_gb = COST_PER_GB.get((dataset, schema), 20.0)
    gb_per = GB_PER_SYM_DAY.get((dataset, schema), 0.001)
    est_gb = n_days * n_syms * gb_per
    est_cost = est_gb * cost_gb

    return {
        "trading_days": n_days,
        "n_symbols": n_syms,
        "cost_per_gb": cost_gb,
        "est_gb": est_gb,
        "est_cost_usd": est_cost,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Pre-acquisition check — run BEFORE spending on Databento data"
    )
    parser.add_argument("--dataset", default="XNAS.ITCH")
    parser.add_argument("--schema", default="imbalance")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument(
        "--symbols", type=int, default=20, help="Number of symbols you plan to fetch (default: 20)"
    )
    args = parser.parse_args()

    try:
        import databento as db

        client = db.Historical(key=KEY)
    except Exception as e:
        print(f"Databento client error: {e}")
        sys.exit(1)

    print()
    print("=" * 68)
    print("  PRE-ACQUISITION CHECK")
    print(f"  {args.dataset} / {args.schema}")
    print(f"  Range: {args.start} → {args.end}  |  {args.symbols} symbols")
    print("=" * 68)

    # ── Cost estimate (instant) ───────────────────────────────────────────────
    cost = estimate_cost(args.dataset, args.schema, args.start, args.end, args.symbols)
    print(f"\n{'─' * 68}")
    print("  COST ESTIMATE (before touching API)")
    print(f"{'─' * 68}")
    print(f"  Trading days in range:  {cost['trading_days']}")
    print(f"  Symbols:                {cost['n_symbols']}")
    print(f"  Rate:                   ${cost['cost_per_gb']:.2f}/GB")
    print(f"  Est. data:              {cost['est_gb'] * 1024:.0f} MB")
    print(f"  Est. total cost:        ${cost['est_cost_usd']:.2f} USD")

    if cost["est_cost_usd"] > 100:
        print("  ⚠️  HIGH COST — validate signal IC before committing")
    elif cost["est_cost_usd"] > 20:
        print("  ⚠️  MODERATE COST — run IC proxy check first")
    else:
        print("  ✅ LOW COST — reasonable to proceed")

    # ── IC proxy check (free) ─────────────────────────────────────────────────
    print(f"\n{'─' * 68}")
    print("  IC PROXY CHECK (free — yfinance price data)")
    print("  Measures predictive power of a cheap proxy for this schema")
    print(f"{'─' * 68}")
    ic_result = check_ic_proxy(args.dataset, args.schema, args.start, args.end)

    if ic_result.get("error"):
        print(f"  ⚠️  Could not compute proxy IC: {ic_result['error']}")
    else:
        ic = ic_result["ic"]
        p = ic_result["p"]
        wf = ic_result["walk_forward"]
        pos = ic_result["positive_years"]
        tot = ic_result["total_years"]

        sig = (
            "✅ SIGNIFICANT" if p < 0.05 else ("⚠️  marginal" if p < 0.10 else "❌ not significant")
        )
        print(f"  Proxy IC:    {ic:+.4f}  p={p:.4f}  {sig}")
        print(f"  Walk-forward ({pos}/{tot} positive years):")
        for yr, v in sorted(wf.items()):
            flag = "✅" if v > 0.02 else ("⚠️ " if v > 0 else "❌")
            print(f"    {yr}: {v:+.4f} {flag}")
        print(f"  Observations: {ic_result['n_obs']:,}")

        # Signal strength assessment
        if abs(ic) < 0.005 and p > 0.10:
            print()
            print("  ❌ WEAK PROXY SIGNAL — the underlying data is unlikely")
            print("     to have meaningful IC. Consider skipping this dataset.")
        elif abs(ic) >= 0.01 and p < 0.05:
            print()
            print("  ✅ STRONG PROXY SIGNAL — real data likely to have IC.")
            print("     Worth acquiring if cost is reasonable.")
        else:
            print()
            print("  ⚠️  MARGINAL PROXY SIGNAL — real data may or may not work.")
            print("     Consider a small-sample spot-fetch before full acquisition.")

    # ── Data existence check (free) ───────────────────────────────────────────
    print(f"\n{'─' * 68}")
    print("  DATA EXISTENCE CHECK (free — limit=1 probes, 8 dates sampled)")
    print(f"{'─' * 68}")
    exist = check_data_exists(client, args.dataset, args.schema, args.start, args.end)
    if exist.get("error"):
        print(f"  ❌ ERROR: {exist['error']}")
    else:
        hit_pct = exist["hit_rate"] * 100
        bar = "█" * int(hit_pct / 5) + "░" * (20 - int(hit_pct / 5))
        print(
            f"  Coverage:    {exist['hits']}/{exist['sample_size']} sampled dates "
            f"[{bar}] {hit_pct:.0f}%"
        )
        if exist["first_good"]:
            print(f"  First data:  {exist['first_good']}")
        if exist["last_good"]:
            print(f"  Last data:   {exist['last_good']}")
        if exist["cutoff_warning"]:
            print(f"  {exist['cutoff_warning']}")
        if exist["gaps"]:
            print(
                f"  Empty dates: {', '.join(exist['gaps'][:5])}"
                + (f" (+{len(exist['gaps']) - 5} more)" if len(exist["gaps"]) > 5 else "")
            )

        if exist["hit_rate"] < 0.5:
            print("\n  ❌ DATA SPARSE — less than 50% of sampled dates have data.")
            print("     Do NOT proceed. This dataset does not cover your range.")
        elif exist["cutoff_warning"]:
            print("\n  ⚠️  POSSIBLE FEED CUTOFF — data stops before your end date.")
            print("     Verify with Databento support before full acquisition.")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("  VERDICT")
    print(f"{'=' * 68}")

    # Score: 0-3
    score = 0
    reasons = []

    # Cost check
    if cost["est_cost_usd"] <= 20:
        score += 1
        reasons.append(f"✅ Cost is low (${cost['est_cost_usd']:.0f})")
    elif cost["est_cost_usd"] <= 100:
        score += 0.5
        reasons.append(f"⚠️  Cost is moderate (${cost['est_cost_usd']:.0f})")
    else:
        reasons.append(f"❌ Cost is high (${cost['est_cost_usd']:.0f}) — validate IC first")

    # IC proxy check
    if not ic_result.get("error"):
        if abs(ic_result.get("ic", 0)) >= 0.01 and ic_result.get("p", 1) < 0.05:
            score += 1
            reasons.append(
                f"✅ Proxy IC significant ({ic_result['ic']:+.4f}, p={ic_result['p']:.3f})"
            )
        elif abs(ic_result.get("ic", 0)) >= 0.005:
            score += 0.5
            reasons.append(f"⚠️  Proxy IC marginal ({ic_result['ic']:+.4f}, p={ic_result['p']:.3f})")
        else:
            reasons.append(f"❌ Proxy IC near zero ({ic_result.get('ic', 0):+.4f})")

    # Data existence check
    if not exist.get("error"):
        if exist["hit_rate"] >= 0.75 and not exist["cutoff_warning"]:
            score += 1
            reasons.append(
                f"✅ Data available across full range ({exist['hit_rate'] * 100:.0f}% coverage)"
            )
        elif exist["hit_rate"] >= 0.5:
            score += 0.5
            reasons.append(
                f"⚠️  Partial data coverage ({exist['hit_rate'] * 100:.0f}%) or possible cutoff"
            )
        else:
            reasons.append(f"❌ Data sparse or missing ({exist['hit_rate'] * 100:.0f}% coverage)")

    for r in reasons:
        print(f"  {r}")

    print()
    if score >= 2.5:
        print("  🟢 GO — data exists, proxy IC is promising, cost is justified")
    elif score >= 1.5:
        print("  🟡 CONDITIONAL GO — do a small spot-fetch first (~$2-5)")
        print("     to verify real IC before committing to full acquisition")
    else:
        print("  🔴 NO-GO — one or more critical checks failed")
        print("     Do not spend money on this dataset without resolving the issues above")

    print(f"{'=' * 68}")
    print()


if __name__ == "__main__":
    main()
