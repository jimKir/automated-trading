#!/usr/bin/env python3
"""Build docs/data/snapshot.json for the live dashboard.

Pulls data from Alpaca paper-trading API and Yahoo Finance (via yfinance),
computes normalised equity curves and risk metrics, writes a single JSON file.
"""

import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ── Alpaca config ─────────────────────────────────────────────────────────
ALPACA_KEY = os.environ.get("APCA_API_KEY_ID", "")
ALPACA_SECRET = os.environ.get("APCA_API_SECRET_KEY", "")
BASE_URL = "https://paper-api.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "snapshot.json"


def alpaca_get(path: str, params: dict | None = None) -> dict | list:
    """GET from Alpaca API with error handling."""
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_account() -> dict:
    return alpaca_get("/v2/account")


def fetch_positions() -> list:
    return alpaca_get("/v2/positions")


def fetch_orders() -> list:
    return alpaca_get("/v2/orders", {"status": "all", "limit": "50", "direction": "desc"})


def fetch_portfolio_history() -> dict:
    return alpaca_get(
        "/v2/account/portfolio/history",
        {
            "period": "all",
            "timeframe": "1D",
            "extended_hours": "false",
        },
    )


def build_equity_df(history: dict) -> pd.DataFrame:
    """Convert portfolio history to a DataFrame with date + equity."""
    timestamps = history.get("timestamp", [])
    equities = history.get("equity", [])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s").normalize(),
            "equity": equities,
        }
    )
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def find_inception(equity_df: pd.DataFrame) -> str:
    """First date where equity > 0."""
    positive = equity_df[equity_df["equity"] > 0]
    if positive.empty:
        return equity_df["date"].iloc[0]
    return positive["date"].iloc[0]


def fetch_benchmarks(inception: str, end: str) -> pd.DataFrame:
    """Fetch SPY & QQQ daily closes via yfinance."""
    tickers = yf.download(
        "SPY QQQ",
        start=inception,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if tickers.empty:
        return pd.DataFrame(columns=["date", "SPY", "QQQ"])

    close = tickers["Close"]
    if isinstance(close, pd.Series):
        # single ticker edge case
        close = close.to_frame()
    close = close.reset_index()
    close.columns = [c[0] if isinstance(c, tuple) else c for c in close.columns]
    close.rename(columns={"Date": "date"}, inplace=True)
    close["date"] = pd.to_datetime(close["date"]).dt.strftime("%Y-%m-%d")
    return close[["date", "SPY", "QQQ"]]


def normalise(series: pd.Series) -> pd.Series:
    """Normalise to base 100 from first value."""
    first = series.iloc[0]
    if first == 0 or pd.isna(first):
        return series
    return (series / first) * 100.0


def compute_metrics(equity_series: pd.Series) -> dict:
    """Compute total return, max drawdown, Sharpe, volatility."""
    returns = equity_series.pct_change().dropna()
    total_return = (
        (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        if len(equity_series) > 1
        else 0.0
    )

    # Max drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min() * 100

    # Annualised volatility and Sharpe (rf=0)
    if len(returns) > 1:
        vol = returns.std() * math.sqrt(252) * 100
        mean_ret = returns.mean() * 252
        sharpe = mean_ret / (returns.std() * math.sqrt(252)) if returns.std() > 0 else 0.0
    else:
        vol = 0.0
        sharpe = 0.0

    return {
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "volatility_pct": round(vol, 2),
    }


def build_snapshot() -> dict:
    """Main: assemble the full snapshot dict."""
    # 1. Fetch Alpaca data
    account = fetch_account()
    positions = fetch_positions()
    orders = fetch_orders()
    history = fetch_portfolio_history()

    # 2. Build equity curve
    equity_df = build_equity_df(history)
    if equity_df.empty:
        print("ERROR: portfolio history returned no data", file=sys.stderr)
        sys.exit(1)

    inception = find_inception(equity_df)
    equity_df = equity_df[equity_df["date"] >= inception].reset_index(drop=True)

    # 3. Fetch benchmarks
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    benchmarks = fetch_benchmarks(inception, today)

    # 4. Merge on date
    merged = equity_df.merge(benchmarks, on="date", how="inner")
    if merged.empty:
        # Fallback: just use equity curve dates
        merged = equity_df.copy()
        merged["SPY"] = 100.0
        merged["QQQ"] = 100.0

    # 5. Normalise
    merged["bot"] = normalise(merged["equity"])
    merged["spy"] = normalise(merged["SPY"])
    merged["qqq"] = normalise(merged["QQQ"])

    equity_curve = merged[["date", "bot", "spy", "qqq"]].copy()
    equity_curve = equity_curve.round(2)

    # 6. Compute metrics from raw equity
    metrics = compute_metrics(merged["equity"])

    # Trades count
    filled_orders = [o for o in orders if o.get("status") == "filled"]
    metrics["trades_total"] = len(filled_orders)

    # Win rate (from filled sell orders with P&L info — approximate)
    metrics["win_rate_pct"] = None  # Not reliably computable from orders alone

    # 7. Account summary
    equity_val = float(account.get("equity", 0))
    last_equity = float(account.get("last_equity", equity_val))
    day_change = equity_val - last_equity
    day_change_pct = (day_change / last_equity * 100) if last_equity > 0 else 0.0
    initial_equity = merged["equity"].iloc[0] if not merged.empty else equity_val
    total_return_pct = ((equity_val / initial_equity) - 1) * 100 if initial_equity > 0 else 0.0

    # Benchmark returns
    spy_return = (
        ((merged["SPY"].iloc[-1] / merged["SPY"].iloc[0]) - 1) * 100 if len(merged) > 1 else 0.0
    )
    qqq_return = (
        ((merged["QQQ"].iloc[-1] / merged["QQQ"].iloc[0]) - 1) * 100 if len(merged) > 1 else 0.0
    )

    account_summary = {
        "equity": round(equity_val, 2),
        "cash": round(float(account.get("cash", 0)), 2),
        "long_market_value": round(float(account.get("long_market_value", 0)), 2),
        "day_change": round(day_change, 2),
        "day_change_pct": round(day_change_pct, 2),
        "total_return_pct": round(total_return_pct, 2),
    }

    benchmark_summary = {
        "spy_return_pct": round(spy_return, 2),
        "qqq_return_pct": round(qqq_return, 2),
        "vs_spy_pp": round(total_return_pct - spy_return, 2),
        "vs_qqq_pp": round(total_return_pct - qqq_return, 2),
    }

    # 8. Positions
    pos_list = []
    for p in positions:
        pos_list.append(
            {
                "symbol": p["symbol"],
                "qty": float(p["qty"]),
                "market_value": round(float(p["market_value"]), 2),
                "avg_entry_price": round(float(p["avg_entry_price"]), 2),
                "current_price": round(float(p["current_price"]), 2),
                "unrealized_pl": round(float(p["unrealized_pl"]), 2),
                "unrealized_pl_pct": round(float(p["unrealized_plpc"]) * 100, 2),
            }
        )
    pos_list.sort(key=lambda x: abs(x["unrealized_pl"]), reverse=True)

    # 9. Recent orders
    order_list = []
    for o in orders[:20]:
        order_list.append(
            {
                "submitted_at": o.get("submitted_at", ""),
                "symbol": o.get("symbol", ""),
                "side": o.get("side", ""),
                "qty": float(o.get("qty") or o.get("notional") or 0),
                "status": o.get("status", ""),
                "filled_avg_price": float(o["filled_avg_price"])
                if o.get("filled_avg_price")
                else None,
            }
        )

    # 10. Assemble
    snapshot = {
        "last_updated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inception_date": inception,
        "account": account_summary,
        "benchmarks": benchmark_summary,
        "metrics": metrics,
        "equity_curve": equity_curve.to_dict(orient="records"),
        "positions": pos_list,
        "recent_orders": order_list,
    }

    return snapshot


def main():
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("ERROR: APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set", file=sys.stderr)
        sys.exit(1)

    snapshot = build_snapshot()

    # Atomic write
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(snapshot, f, indent=2)
    tmp.replace(OUTPUT)

    # Summary
    n_points = len(snapshot["equity_curve"])
    n_pos = len(snapshot["positions"])
    eq = snapshot["account"]["equity"]
    inception = snapshot["inception_date"]
    print(f"Snapshot written to {OUTPUT}")
    print(f"  Inception: {inception}")
    print(f"  Equity: ${eq:,.2f}")
    print(f"  Positions: {n_pos}")
    print(f"  Equity curve points: {n_points}")
    print(f"  Last updated: {snapshot['last_updated']}")


if __name__ == "__main__":
    main()
