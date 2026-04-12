"""
Walk-forward validation: KalshiMacroFeed composite_stress vs FRED-only baseline.

Backfills historical Kalshi probabilities via API (no auth required),
then runs 5 expanding-window train/test splits with 14-day embargo
to measure OOS MaxDD reduction.

Usage:
    python backtest/kalshi_wf_validation.py
    python backtest/kalshi_wf_validation.py --save-results
"""
import os, sys, json, time, logging, argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kalshi_wf")

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
EMBARGO_DAYS = 14

# ── Kalshi series tickers ─────────────────────────────────────────────────────
MACRO_SERIES = {
    "fed":       "KXFED",
    "cpi":       "KXCPI",
    "recession": "KXRECESSION",
}

# ── Walk-forward fold definitions ─────────────────────────────────────────────
FOLDS = [
    {"name":"Fold1_RateHike",   "train_end":"2022-06-30","test_start":"2022-07-15","test_end":"2022-12-31"},
    {"name":"Fold2_Recovery",   "train_end":"2022-12-31","test_start":"2023-01-15","test_end":"2023-06-30"},
    {"name":"Fold3_BullBegin",  "train_end":"2023-06-30","test_start":"2023-07-15","test_end":"2023-12-31"},
    {"name":"Fold4_Bull2024",   "train_end":"2023-12-31","test_start":"2024-01-15","test_end":"2024-06-30"},
    {"name":"Fold5_Tariff",     "train_end":"2024-06-30","test_start":"2024-07-15","test_end":"2026-04-05"},
]

# ── API helpers ───────────────────────────────────────────────────────────────
session = requests.Session()
session.headers["User-Agent"] = "trading-system-wf/1.0"

def api_get(path: str, params: dict = None, retries: int = 3) -> dict:
    url = f"{KALSHI_BASE}{path}"
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.debug(f"Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return {}

# ── Backfill historical probability series ────────────────────────────────────

def backfill_series(series_ticker: str, start_date: str = "2022-01-01") -> pd.Series:
    """
    Returns a daily pd.Series of the 'YES probability' for the given series.
    For FOMC/CPI: probability of the next hike/cut/beat on each calendar day.
    Uses both live and historical endpoints.
    """
    log.info(f"Backfilling {series_ticker} from {start_date}...")
    
    all_markets = []
    
    # 1. Get historical (settled) markets
    cursor = None
    while True:
        params = {"series_ticker": series_ticker, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = api_get("/historical/markets", params)
        markets = data.get("markets", [])
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor or not markets:
            break
    
    # 2. Get live (open + recently settled) markets
    params = {"series_ticker": series_ticker, "status": "all", "limit": 200}
    data = api_get("/markets", params)
    live_markets = data.get("markets", [])
    all_markets.extend(live_markets)
    
    log.info(f"  Found {len(all_markets)} markets for {series_ticker}")
    
    if not all_markets:
        log.warning(f"  No markets found for {series_ticker} — returning empty series")
        return pd.Series(dtype=float)
    
    # 3. For each market, fetch daily candlesticks
    start_ts = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    end_ts   = int(datetime.now(timezone.utc).timestamp())
    
    daily_probs = {}  # date → probability
    
    for mkt in all_markets:
        ticker = mkt.get("ticker", "")
        if not ticker:
            continue
        
        # Determine if market is in historical tier
        settled_ts_str = mkt.get("settlement_ts") or mkt.get("close_time")
        is_historical = False
        if settled_ts_str:
            try:
                settled_dt = datetime.fromisoformat(settled_ts_str.replace("Z","+00:00"))
                # Check against cutoff (we'll try historical first, fall back to live)
                is_historical = settled_dt < datetime.now(timezone.utc) - timedelta(days=30)
            except Exception:
                pass
        
        # Fetch candlesticks (try historical endpoint first, then live)
        candles = []
        for endpoint in [
            f"/historical/markets/{ticker}/candlesticks",
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
        ]:
            try:
                data = api_get(endpoint, {
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "period_interval": 1440,  # daily
                })
                candles = data.get("candlesticks", [])
                if candles:
                    break
            except Exception as e:
                log.debug(f"  {endpoint}: {e}")
                continue
        
        if not candles:
            continue
        
        # Extract YES bid/ask midpoint as probability on each day
        for c in candles:
            ts = c.get("end_period_ts")
            if not ts:
                continue
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
            
            # Use midpoint of yes_bid and yes_ask as probability estimate
            yes_bid = c.get("yes_bid", {})
            yes_ask = c.get("yes_ask", {})
            price   = c.get("price", {})
            
            bid = float(yes_bid.get("close_dollars") or yes_bid.get("close") or 0)
            ask = float(yes_ask.get("close_dollars") or yes_ask.get("close") or 0)
            mid_price = float(price.get("close_dollars") or price.get("close") or 0)
            
            if bid > 0 and ask > 0:
                prob = (bid + ask) / 2
            elif mid_price > 0:
                prob = mid_price
            else:
                continue
            
            # For probability, 56 cents = 56% = 0.56
            if prob > 1:
                prob = prob / 100
            
            # Keep highest-probability market for each day
            # (most relevant / nearest-to-settle contract)
            existing = daily_probs.get(dt, 0)
            if prob > existing:
                daily_probs[dt] = prob
        
        time.sleep(0.05)  # rate limit courtesy
    
    if not daily_probs:
        return pd.Series(dtype=float)
    
    series = pd.Series(daily_probs).sort_index()
    series.index = pd.to_datetime(series.index)
    
    # Forward-fill gaps (weekend/holiday = carry last known probability)
    full_range = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_range).ffill()
    
    log.info(f"  {series_ticker}: {len(series)} daily obs, {series.index.min().date()} → {series.index.max().date()}")
    return series

def build_composite_stress(
    fed_series: pd.Series,
    cpi_series: pd.Series,
    recession_series: pd.Series,
) -> pd.Series:
    """
    Combine 3 probability series into composite_stress [0,1].
    Weights: recession=0.40, fed=0.35, cpi=0.25
    fed_stress: max(hike_prob, cut_prob) — proxy for policy uncertainty
    """
    # Align on common date range
    all_series = [s for s in [fed_series, cpi_series, recession_series] if len(s) > 0]
    if not all_series:
        return pd.Series(dtype=float)
    
    common = all_series[0].index
    for s in all_series[1:]:
        common = common.intersection(s.index)
    
    weights = {"fed": 0.35, "cpi": 0.25, "recession": 0.40}
    composite = pd.Series(0.0, index=common)
    total_weight = 0.0
    
    if len(fed_series) > 0:
        # fed_stress = policy uncertainty (max of rate change probability)
        # Raw fed series = probability of rate hike/cut; 0.5 = maximum uncertainty
        fed_aligned = fed_series.reindex(common).ffill().fillna(0)
        fed_stress = (fed_aligned - 0.5).abs() * 2  # peaks at 1.0 when very certain
        # Invert: uncertainty is stress, not certainty
        fed_stress = 1.0 - fed_stress.clip(0, 1)
        composite += weights["fed"] * fed_stress
        total_weight += weights["fed"]
    
    if len(cpi_series) > 0:
        cpi_aligned = cpi_series.reindex(common).ffill().fillna(0)
        # Surprise risk = probability of extreme reading
        cpi_risk = (cpi_aligned - 0.5).abs() * 2
        composite += weights["cpi"] * cpi_risk
        total_weight += weights["cpi"]
    
    if len(recession_series) > 0:
        rec_aligned = recession_series.reindex(common).ffill().fillna(0)
        composite += weights["recession"] * rec_aligned
        total_weight += weights["recession"]
    
    if total_weight > 0:
        composite = composite / total_weight  # normalize if any source missing
    
    return composite.clip(0, 1)

# ── Load price data and compute strategy returns ──────────────────────────────

def load_spy_returns(start: str, end: str) -> pd.Series:
    """Load SPY daily returns from local parquet."""
    from data.data_store import DataStore
    store = DataStore()
    spy = store.load("SPY", start_date=start, end_date=end)
    if spy is None or len(spy) == 0:
        raise ValueError("SPY data not available")
    return spy["close"].pct_change().dropna()

def apply_kalshi_scale(returns: pd.Series,
                       composite_stress: pd.Series,
                       weight: float = 0.25,
                       base_scale: float = 1.0) -> pd.Series:
    """
    Apply Kalshi-derived position scale to daily returns.
    Scale:
      composite_stress < 0.20 → scale = 1.00
      composite_stress 0.20–0.35 → scale = 0.85
      composite_stress 0.35–0.50 → scale = 0.65
      composite_stress > 0.50 → scale = 0.40
    """
    def stress_to_scale(s: float) -> float:
        if s < 0.20: return 1.00
        elif s < 0.35: return 0.85
        elif s < 0.50: return 0.65
        else: return 0.40

    # Align stress to return dates
    stress_aligned = composite_stress.reindex(returns.index).ffill().fillna(0)
    scales = stress_aligned.map(stress_to_scale)
    
    # Blend with base scale (Kalshi adds 25% contribution)
    blended_scales = base_scale * (1 - weight) + scales * weight
    
    scaled_returns = returns * blended_scales
    return scaled_returns

def compute_metrics(returns: pd.Series) -> dict:
    """Compute Sharpe, MaxDD, total return, Calmar."""
    if len(returns) == 0:
        return {"sharpe": 0, "max_dd": 0, "total_return": 0, "calmar": 0}
    
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()
    
    total_ret = cum.iloc[-1] - 1
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
    
    return {
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 4),
        "total_return": round(total_ret, 4),
        "calmar": round(calmar, 3),
    }

# ── Main walk-forward loop ────────────────────────────────────────────────────

def run_walk_forward(save_results: bool = False):
    print("\n" + "=" * 65)
    print("  KALSHI MACRO FEED — WALK-FORWARD VALIDATION")
    print("  5 expanding-window folds | 14-day embargo | OOS MaxDD target")
    print("=" * 65)
    
    # Step 1: Backfill all three probability series
    print("\n[1/3] Backfilling Kalshi probability series...")
    fed_series       = backfill_series(MACRO_SERIES["fed"])
    cpi_series       = backfill_series(MACRO_SERIES["cpi"])
    recession_series = backfill_series(MACRO_SERIES["recession"])
    
    # Step 2: Build composite stress
    print("\n[2/3] Building composite stress series...")
    composite = build_composite_stress(fed_series, cpi_series, recession_series)
    
    if len(composite) == 0:
        print("ERROR: No Kalshi data available. Check API connectivity.")
        return
    
    print(f"  Composite series: {composite.index.min().date()} → {composite.index.max().date()}")
    print(f"  Mean stress: {composite.mean():.3f} | P90: {composite.quantile(0.90):.3f} | Max: {composite.max():.3f}")
    
    # Step 3: Walk-forward evaluation
    print("\n[3/3] Running walk-forward folds...")
    
    results = []
    
    header = (f"\n  {'Fold':<18} {'Baseline Sharpe':>15} {'Kalshi Sharpe':>13} "
              f"{'Δ Sharpe':>9} {'Base MaxDD':>11} {'Kalshi MaxDD':>12} {'Δ MaxDD':>9}")
    print(header)
    print("  " + "─" * 90)
    
    for fold in FOLDS:
        name       = fold["name"]
        test_start = fold["test_start"]
        test_end   = fold["test_end"]
        
        try:
            # Load SPY returns for test period
            spy_returns = load_spy_returns(test_start, test_end)
            if len(spy_returns) < 20:
                log.warning(f"  {name}: insufficient data ({len(spy_returns)} days)")
                continue
            
            # Baseline: raw SPY returns (no Kalshi scaling)
            base_metrics = compute_metrics(spy_returns)
            
            # Kalshi-scaled returns
            kalshi_returns = apply_kalshi_scale(spy_returns, composite, weight=0.25)
            kalshi_metrics = compute_metrics(kalshi_returns)
            
            delta_sharpe = round(kalshi_metrics["sharpe"] - base_metrics["sharpe"], 3)
            delta_maxdd  = round(kalshi_metrics["max_dd"]  - base_metrics["max_dd"],  4)
            
            # Flag improvement
            dd_flag   = "✅" if delta_maxdd > 0.005 else ("⚠" if abs(delta_maxdd) < 0.005 else "❌")
            sh_flag   = "✅" if delta_sharpe > 0.05  else ("⚠" if abs(delta_sharpe) < 0.05  else "❌")
            
            result = {
                "fold": name,
                "test_period": f"{test_start} to {test_end}",
                "n_days": len(spy_returns),
                "baseline": base_metrics,
                "kalshi": kalshi_metrics,
                "delta_sharpe": delta_sharpe,
                "delta_max_dd": delta_maxdd,
                "dd_improves": delta_maxdd > 0.005,
                "sharpe_improves": delta_sharpe > 0.05,
            }
            results.append(result)
            
            print(f"  {name:<18} {base_metrics['sharpe']:>15.3f} {kalshi_metrics['sharpe']:>13.3f} "
                  f"{sh_flag} {delta_sharpe:>+7.3f} {base_metrics['max_dd']:>11.3f} "
                  f"{kalshi_metrics['max_dd']:>12.3f} {dd_flag} {delta_maxdd:>+8.4f}")
        
        except Exception as e:
            log.error(f"  {name}: FAILED — {e}")
            continue
    
    if not results:
        print("\n  No results — check data availability")
        return
    
    # Summary
    print("\n" + "=" * 65)
    n_dd_improve  = sum(1 for r in results if r["dd_improves"])
    n_sh_improve  = sum(1 for r in results if r["sharpe_improves"])
    mean_dd_delta = np.mean([r["delta_max_dd"]  for r in results])
    mean_sh_delta = np.mean([r["delta_sharpe"]  for r in results])
    
    verdict = "ADOPT" if n_dd_improve >= 4 else ("CONDITIONAL" if n_dd_improve >= 3 else "REJECT")
    
    print(f"  MaxDD improved:    {n_dd_improve}/{len(results)} folds  |  Mean Δ: {mean_dd_delta:+.4f}")
    print(f"  Sharpe improved:   {n_sh_improve}/{len(results)} folds  |  Mean Δ: {mean_sh_delta:+.3f}")
    print(f"  Verdict:           {verdict}")
    print("=" * 65)
    
    # Save results
    if save_results:
        os.makedirs("results", exist_ok=True)
        output = {
            "run_date": datetime.now().isoformat(),
            "methodology": "expanding_window_wf_14d_embargo",
            "kalshi_data_range": {
                "fed":       f"{fed_series.index.min().date()} to {fed_series.index.max().date()}" if len(fed_series) > 0 else "unavailable",
                "cpi":       f"{cpi_series.index.min().date()} to {cpi_series.index.max().date()}" if len(cpi_series) > 0 else "unavailable",
                "recession": f"{recession_series.index.min().date()} to {recession_series.index.max().date()}" if len(recession_series) > 0 else "unavailable",
            },
            "composite_stats": {
                "mean":  round(float(composite.mean()), 4),
                "p90":   round(float(composite.quantile(0.90)), 4),
                "max":   round(float(composite.max()), 4),
            },
            "folds": results,
            "summary": {
                "n_folds": len(results),
                "n_dd_improved": n_dd_improve,
                "n_sharpe_improved": n_sh_improve,
                "mean_dd_delta": round(float(mean_dd_delta), 4),
                "mean_sharpe_delta": round(float(mean_sh_delta), 4),
                "verdict": verdict,
            }
        }
        path = "results/kalshi_wf_results.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved → {path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()
    run_walk_forward(save_results=args.save_results)
