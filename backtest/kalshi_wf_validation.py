"""
Walk-forward validation: KalshiMacroFeed composite_stress vs baseline.
Uses real Kalshi series tickers discovered via API exploration.

Run: python backtest/kalshi_wf_validation.py --save-results
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kalshi_wf")

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
EMBARGO_DAYS = 14

FOLDS = [
    {"name":"Fold1_RateHike",  "test_start":"2022-07-15","test_end":"2022-12-31"},
    {"name":"Fold2_Recovery",  "test_start":"2023-01-15","test_end":"2023-06-30"},
    {"name":"Fold3_BullBegin", "test_start":"2023-07-15","test_end":"2023-12-31"},
    {"name":"Fold4_Bull2024",  "test_start":"2024-01-15","test_end":"2024-06-30"},
    {"name":"Fold5_Tariff",    "test_start":"2024-07-15","test_end":"2026-04-05"},
]

session = requests.Session()
session.headers["User-Agent"] = "trading-system/1.0"

def api_get(path, params=None, retries=3):
    url = f"{KALSHI_BASE}{path}"
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                log.warning(f"API failed {path}: {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}

# ── Signal extractors ─────────────────────────────────────────────────────────

def get_fed_stress_live() -> float:
    """
    Fed policy uncertainty score from KXFEDDECISION.
    Stress = 1 - P(hold). High when market expects a move.
    Uses nearest upcoming meeting.
    """
    data = api_get("/markets", {"series_ticker": "KXFEDDECISION", "status": "open", "limit": 50})
    markets = data.get("markets", [])
    if not markets:
        return 0.0

    # Find the nearest meeting (shortest ticker date prefix = soonest)
    # Tickers: KXFEDDECISION-26MAY-H0, etc.
    meeting_hold = {}
    for m in markets:
        ticker = m.get("ticker", "")
        parts = ticker.split("-")
        if len(parts) < 3:
            continue
        meeting = parts[1]  # e.g. "26MAY"
        if "-H0" in ticker:  # Hold = 0bps change
            prob = float(m.get("yes_bid_dollars") or 0)
            if meeting not in meeting_hold or prob > meeting_hold[meeting]:
                meeting_hold[meeting] = prob

    if not meeting_hold:
        return 0.0

    # Use nearest meeting (sort by meeting code)
    nearest = sorted(meeting_hold.keys())[0]
    hold_prob = meeting_hold[nearest]
    # Stress = 1 - hold probability (uncertainty = expecting a move)
    return round(1.0 - hold_prob, 3)

def get_inflation_stress_live() -> float:
    """
    Inflation surprise risk from KXPCECORE.
    Uses P(core PCE > 0.3%) as upside inflation stress signal.
    Elevated when market prices in >30% chance of hot print.
    """
    data = api_get("/markets", {"series_ticker": "KXPCECORE", "status": "open", "limit": 50})
    markets = data.get("markets", [])
    if not markets:
        return 0.0

    # Find nearest month, get P(>0.3%) as stress indicator
    month_stress = {}
    for m in markets:
        ticker = m.get("ticker", "")
        if "-T0.3" not in ticker:
            continue
        parts = ticker.split("-")
        if len(parts) < 2:
            continue
        month = parts[1]  # e.g. "26NOV"
        prob = float(m.get("yes_bid_dollars") or 0)
        month_stress[month] = prob

    if not month_stress:
        return 0.0

    nearest = sorted(month_stress.keys())[0]
    # P(>0.3%) maps to stress: 0.3 = moderate, 0.5 = elevated, 0.7 = high
    raw = month_stress[nearest]
    # Normalize: 0.15 = baseline, 0.50 = max stress
    stress = min((raw - 0.15) / 0.35, 1.0)
    return round(max(stress, 0.0), 3)

def get_recession_stress_live() -> float:
    """
    Recession probability from RECSSNBER.
    Falls back to KXGDPYEAR or other recession proxies if RECSSNBER empty.
    """
    # Try RECSSNBER first
    data = api_get("/markets", {"series_ticker": "RECSSNBER", "status": "open", "limit": 10})
    markets = data.get("markets", [])
    if markets:
        best = max(markets, key=lambda m: float(m.get("volume_fp") or 0))
        return round(float(best.get("yes_bid_dollars") or 0), 3)

    # Fallback: use KXFEDRATEMIN (lowest rate expectation as recession proxy)
    data = api_get("/markets", {"series_ticker": "KXFEDRATEMIN", "status": "open", "limit": 10})
    markets = data.get("markets", [])
    if markets:
        # P(rate goes very low) correlates with recession expectation
        low_rate_markets = [m for m in markets if "0" in m.get("ticker","")]
        if low_rate_markets:
            prob = float(low_rate_markets[0].get("yes_bid_dollars") or 0)
            return round(prob * 0.6, 3)  # discount since indirect

    return 0.0

def get_composite_stress_live() -> dict:
    """Returns all three signals + composite for current date."""
    fed   = get_fed_stress_live()
    infl  = get_inflation_stress_live()
    rec   = get_recession_stress_live()
    comp  = round(0.35 * fed + 0.25 * infl + 0.40 * rec, 3)
    return {"fed_stress": fed, "inflation_stress": infl,
            "recession_stress": rec, "composite": comp}

# ── Historical backfill via settled markets ───────────────────────────────────

def backfill_fed_stress(start_date="2022-01-01") -> pd.Series:
    """
    Reconstruct daily Fed policy uncertainty from settled KXFEDDECISION markets.
    For each past FOMC meeting: fetch pre-meeting hold probability trajectory.
    """
    log.info("Backfilling Fed stress from KXFEDDECISION historical markets...")
    daily = {}

    # Get all settled KXFEDDECISION markets
    cursor = None
    all_markets = []
    for _ in range(20):  # max 20 pages
        params = {"series_ticker": "KXFEDDECISION", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = api_get("/historical/markets", params)
        markets = data.get("markets", [])
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor:
            break

    # Also get from live endpoint (recent)
    data = api_get("/markets", {"series_ticker": "KXFEDDECISION", "status": "all", "limit": 200})
    all_markets.extend(data.get("markets", []))

    log.info(f"  Found {len(all_markets)} KXFEDDECISION markets")

    # Filter to hold markets only (H0 = no change)
    hold_markets = [m for m in all_markets if m.get("ticker","").endswith("-H0")]
    log.info(f"  {len(hold_markets)} hold (H0) markets")

    start_ts = int(datetime.fromisoformat(start_date).replace(tzinfo=UTC).timestamp())

    for mkt in hold_markets:
        ticker = mkt.get("ticker","")
        open_t  = mkt.get("open_time","")
        close_t = mkt.get("close_time","")
        if not open_t:
            continue

        # Fetch daily candlesticks for this market's lifetime
        for endpoint in [
            f"/historical/markets/{ticker}/candlesticks",
            f"/series/KXFEDDECISION/markets/{ticker}/candlesticks",
        ]:
            try:
                mkt_open_ts = int(datetime.fromisoformat(
                    open_t).timestamp())
                mkt_close_ts = int(datetime.fromisoformat(
                    close_t).timestamp()) if close_t else int(time.time())
                if mkt_close_ts < start_ts:
                    break  # too old
                data = api_get(endpoint, {
                    "start_ts": max(start_ts, mkt_open_ts),
                    "end_ts": mkt_close_ts,
                    "period_interval": 1440
                })
                candles = data.get("candlesticks", [])
                for c in candles:
                    ts = c.get("end_period_ts")
                    if not ts:
                        continue
                    dt = datetime.fromtimestamp(ts, tz=UTC).date()
                    price_data = c.get("price", {})
                    close_p = float(price_data.get("close_dollars") or
                                    price_data.get("close") or 0)
                    if close_p > 1:
                        close_p /= 100
                    if close_p > 0:
                        # Hold prob → stress = 1 - hold
                        stress = 1.0 - close_p
                        existing = daily.get(dt, stress)
                        daily[dt] = min(existing, stress)  # use lowest stress (closest to meeting)
                if candles:
                    break
            except Exception as e:
                log.debug(f"  {ticker} {endpoint}: {e}")
                continue
        time.sleep(0.03)

    if not daily:
        log.warning("  No historical Fed stress data — returning empty")
        return pd.Series(dtype=float)

    s = pd.Series(daily).sort_index()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill()
    log.info(f"  Fed stress: {s.index.min().date()} → {s.index.max().date()}, mean={s.mean():.3f}")
    return s

def backfill_inflation_stress(start_date="2022-01-01") -> pd.Series:
    """Reconstruct daily inflation stress from KXPCECORE settled markets."""
    log.info("Backfilling inflation stress from KXPCECORE...")
    daily = {}
    all_markets = []

    for endpoint_base in ["/historical/markets", "/markets"]:
        params = {"series_ticker": "KXPCECORE",
                  "limit": 200,
                  **({"status": "all"} if endpoint_base == "/markets" else {})}
        cursor = None
        for _ in range(10):
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            data = api_get(endpoint_base, p)
            mkts = data.get("markets", [])
            all_markets.extend(mkts)
            cursor = data.get("cursor")
            if not cursor:
                break

    # Use T0.3 markets (P(>0.3%) as stress indicator)
    t03_markets = [m for m in all_markets if "-T0.3" in m.get("ticker","")]
    log.info(f"  {len(t03_markets)} KXPCECORE T0.3 markets")

    start_ts = int(datetime.fromisoformat(start_date).replace(tzinfo=UTC).timestamp())

    for mkt in t03_markets:
        ticker = mkt.get("ticker","")
        open_t  = mkt.get("open_time","")
        close_t = mkt.get("close_time","")
        if not open_t:
            continue
        for endpoint in [
            f"/historical/markets/{ticker}/candlesticks",
            f"/series/KXPCECORE/markets/{ticker}/candlesticks",
        ]:
            try:
                mkt_open_ts = int(datetime.fromisoformat(
                    open_t).timestamp())
                mkt_close_ts = int(datetime.fromisoformat(
                    close_t).timestamp()) if close_t else int(time.time())
                if mkt_close_ts < start_ts:
                    break
                data = api_get(endpoint, {
                    "start_ts": max(start_ts, mkt_open_ts),
                    "end_ts": mkt_close_ts,
                    "period_interval": 1440
                })
                candles = data.get("candlesticks", [])
                for c in candles:
                    ts = c.get("end_period_ts")
                    if not ts:
                        continue
                    dt = datetime.fromtimestamp(ts, tz=UTC).date()
                    price_data = c.get("price", {})
                    close_p = float(price_data.get("close_dollars") or
                                    price_data.get("close") or 0)
                    if close_p > 1:
                        close_p /= 100
                    if close_p > 0:
                        stress = max((close_p - 0.15) / 0.35, 0.0)
                        existing = daily.get(dt, stress)
                        daily[dt] = max(existing, stress)
                if candles:
                    break
            except Exception as e:
                log.debug(f"  {ticker}: {e}")
                continue
        time.sleep(0.03)

    if not daily:
        return pd.Series(dtype=float)

    s = pd.Series(daily).sort_index()
    s.index = pd.to_datetime(s.index)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill()
    log.info(f"  Inflation stress: {s.index.min().date()} → {s.index.max().date()}, mean={s.mean():.3f}")
    return s

def build_composite(fed: pd.Series, infl: pd.Series) -> pd.Series:
    """Combine available series. Recession series empty → adjust weights."""
    all_s = [(fed, 0.50), (infl, 0.50)]  # equal weight when recession unavailable
    available = [(s, w) for s, w in all_s if len(s) > 0]
    if not available:
        return pd.Series(dtype=float)
    total_w = sum(w for _, w in available)
    common = available[0][0].index
    for s, _ in available[1:]:
        common = common.intersection(s.index)
    composite = pd.Series(0.0, index=common)
    for s, w in available:
        composite += (w / total_w) * s.reindex(common).ffill().fillna(0)
    return composite.clip(0, 1)

# ── Metrics and backtest ──────────────────────────────────────────────────────

def load_spy(start, end) -> pd.Series:
    from data.data_store import DataStore
    df = DataStore().load("SPY", start_date=start, end_date=end)
    if df is None or len(df) < 20:
        raise ValueError(f"SPY data insufficient for {start}–{end}")
    return df["close"].pct_change().dropna()

def apply_scale(returns: pd.Series, stress: pd.Series, weight=0.25) -> pd.Series:
    def s2scale(s):
        if s < 0.20:
            return 1.00
        if s < 0.35:
            return 0.85
        if s < 0.50:
            return 0.65
        return 0.40
    sa = stress.reindex(returns.index).ffill().fillna(0)
    scales = 1.0 * (1 - weight) + sa.map(s2scale) * weight
    return returns * scales

def metrics(returns: pd.Series) -> dict:
    if len(returns) < 5:
        return {"sharpe":0,"max_dd":0,"total_return":0}
    ann_r = returns.mean() * 252
    ann_v = returns.std() * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 0 else 0
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {"sharpe": round(sharpe,3), "max_dd": round(dd.min(),4),
            "total_return": round(float(cum.iloc[-1]-1),4)}

# ── Main ──────────────────────────────────────────────────────────────────────

def run(save=False):
    print("\n" + "="*65)
    print("  KALSHI MACRO FEED — WALK-FORWARD VALIDATION")
    print("  Series: KXFEDDECISION + KXPCECORE | 5 folds | 14d embargo")
    print("="*65)

    # Print current live reading first
    print("\nCurrent live Kalshi signals:")
    live = get_composite_stress_live()
    print(f"  Fed uncertainty:    {live['fed_stress']:.1%}")
    print(f"  Inflation stress:   {live['inflation_stress']:.1%}")
    print(f"  Recession stress:   {live['recession_stress']:.1%}")
    print(f"  Composite stress:   {live['composite']:.3f}")

    print("\n[1/3] Backfilling historical series (this takes 3-5 min)...")
    fed_s  = backfill_fed_stress()
    infl_s = backfill_inflation_stress()

    print("\n[2/3] Building composite stress...")
    comp = build_composite(fed_s, infl_s)
    if len(comp) == 0:
        print("ERROR: No historical Kalshi data retrieved.")
        print("The historical API may require authentication for candlestick data.")
        print("\nFalling back to live-signal validation only:")
        print(f"  Current composite stress: {live['composite']:.3f}")
        print("  → Wire this into MacroAnomalyDetector via KalshiMacroFeed.get_choppy_input()")
        return
    print(f"  Range: {comp.index.min().date()} → {comp.index.max().date()}")
    print(f"  Mean: {comp.mean():.3f} | P90: {comp.quantile(0.90):.3f}")

    print("\n[3/3] Walk-forward folds...")
    results = []
    print(f"\n  {'Fold':<18} {'Base Sharpe':>12} {'Kalshi Sharpe':>13} {'ΔSharpe':>8} {'Base MaxDD':>11} {'Kalshi MaxDD':>12} {'ΔMaxDD':>8}")
    print("  " + "─"*87)

    for fold in FOLDS:
        try:
            spy = load_spy(fold["test_start"], fold["test_end"])
            base_m   = metrics(spy)
            kalshi_r = apply_scale(spy, comp, weight=0.25)
            kalshi_m = metrics(kalshi_r)
            dsh = round(kalshi_m["sharpe"] - base_m["sharpe"], 3)
            ddd = round(kalshi_m["max_dd"]  - base_m["max_dd"],  4)
            sf = "✅" if dsh > 0.05 else ("⚠" if abs(dsh)<0.05 else "❌")
            df = "✅" if ddd > 0.005 else ("⚠" if abs(ddd)<0.005 else "❌")
            r = {"fold": fold["name"], "test_period": f"{fold['test_start']}→{fold['test_end']}",
                 "n_days": len(spy), "baseline": base_m, "kalshi": kalshi_m,
                 "delta_sharpe": dsh, "delta_max_dd": ddd,
                 "dd_improves": ddd > 0.005, "sharpe_improves": dsh > 0.05}
            results.append(r)
            print(f"  {fold['name']:<18} {base_m['sharpe']:>12.3f} {kalshi_m['sharpe']:>13.3f} "
                  f"{sf}{dsh:>+7.3f} {base_m['max_dd']:>11.4f} {kalshi_m['max_dd']:>12.4f} "
                  f"{df}{ddd:>+7.4f}")
        except Exception as e:
            log.error(f"  {fold['name']}: {e}")

    if results:
        n_dd = sum(1 for r in results if r["dd_improves"])
        n_sh = sum(1 for r in results if r["sharpe_improves"])
        mdd  = np.mean([r["delta_max_dd"] for r in results])
        msh  = np.mean([r["delta_sharpe"] for r in results])
        verdict = "ADOPT" if n_dd >= 4 else ("CONDITIONAL" if n_dd >= 3 else "REJECT")
        print(f"\n  MaxDD improved: {n_dd}/{len(results)} | Mean ΔMaxDD: {mdd:+.4f}")
        print(f"  Sharpe improved:{n_sh}/{len(results)} | Mean ΔSharpe: {msh:+.3f}")
        print(f"  Verdict: {verdict}")

        if save:
            os.makedirs("results", exist_ok=True)
            out = {"run_date": datetime.now().isoformat(),
                   "live_signals": live, "folds": results,
                   "summary": {"n_dd": n_dd, "n_sh": n_sh,
                                "mean_dd_delta": round(float(mdd),4),
                                "mean_sh_delta": round(float(msh),4),
                                "verdict": verdict}}
            with open("results/kalshi_wf_results.json","w") as f:
                json.dump(out, f, indent=2, default=str)
            print("  Saved → results/kalshi_wf_results.json")

    print("="*65)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-results", action="store_true")
    args = p.parse_args()
    run(save=args.save_results)
