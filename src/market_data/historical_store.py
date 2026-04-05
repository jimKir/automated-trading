"""
Historical Data Store
======================
Systematic collection, storage, and incremental updating of all price data
needed by the trading system. Backed by Parquet files — fast, compressed,
and git-LFS compatible.

Structure:
    data/historical/
    ├── daily/
    │   ├── AAPL.parquet       ← OHLCV daily, full history from 2010
    │   ├── MSFT.parquet
    │   └── ...
    ├── macro/
    │   ├── VIX.parquet        ← ^VIX daily
    │   ├── HYG.parquet
    │   └── ...
    └── metadata.json          ← per-symbol: last_updated, row_count, source, gaps

Commands:
    python -m src.market_data.historical_store --collect    # first-time full download
    python -m src.market_data.historical_store --update     # append latest bars
    python -m src.market_data.historical_store --audit      # data quality report
    python -m src.market_data.historical_store --status     # quick coverage summary
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import traceback
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("HistoricalStore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent.parent
DATA_DIR     = REPO_ROOT / "data" / "historical"
DAILY_DIR    = DATA_DIR / "daily"
MACRO_DIR    = DATA_DIR / "macro"
META_FILE    = DATA_DIR / "metadata.json"
ALERT_EMAIL  = "kiritsis.di@gmail.com"

# ── Universe ───────────────────────────────────────────────────────────────────
EQUITY_SYMS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO",
    "JPM","V","MA","UNH","JNJ","PG","HD","KO","XOM","CVX","BAC","GS",
]
MACRO_SYMS = {
    "SPY":  "SPY",          # S&P 500 ETF
    "QQQ":  "QQQ",          # NASDAQ 100 ETF
    "VIX":  "^VIX",         # CBOE Volatility Index
    "HYG":  "HYG",          # High-yield bond ETF
    "LQD":  "LQD",          # Investment-grade bond ETF
    "TLT":  "TLT",          # 20Y Treasury ETF
    "SHY":  "SHY",          # 1-3Y Treasury ETF
    "GLD":  "GLD",          # Gold ETF
    "DXY":  "DX-Y.NYB",     # Dollar Index
    "IWM":  "IWM",          # Russell 2000 ETF
}

HISTORY_START = "2010-01-01"
REFRESH_TAIL  = 7   # always re-fetch last N trading days (bars get revised)


# ── Metadata ───────────────────────────────────────────────────────────────────

def load_meta() -> dict:
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_meta(meta: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, indent=2, default=str))


# ── Data fetcher ───────────────────────────────────────────────────────────────

def fetch_yfinance(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from yfinance. Returns None on failure.
    Cleans up the multi-level column index yfinance sometimes produces.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "date"
        df = df[df["close"].notna() & (df["close"] > 0)]
        return df.sort_index()
    except Exception as e:
        log.warning(f"yfinance fetch failed for {ticker}: {e}")
        return None


def fetch_alpaca(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Alpaca fallback for recent data not yet in yfinance."""
    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key:
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests   import StockBarsRequest
        from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit
        client = StockHistoricalDataClient(api_key=key, secret_key=secret)
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start, end=end, adjustment="all",
        )
        resp = client.get_stock_bars(req)
        if not resp or not resp.data or ticker not in resp.data:
            return None
        rows = [{"date": pd.Timestamp(b.timestamp).normalize(),
                 "open": b.open, "high": b.high, "low": b.low,
                 "close": b.close, "volume": b.volume}
                for b in resp.data[ticker]]
        df = pd.DataFrame(rows).set_index("date").sort_index()
        return df[df["close"] > 0]
    except Exception as e:
        log.debug(f"Alpaca fetch failed for {ticker}: {e}")
        return None


# ── Parquet I/O ────────────────────────────────────────────────────────────────

def parquet_path(sym: str, kind: str = "daily") -> Path:
    base = DAILY_DIR if kind == "daily" else MACRO_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{sym}.parquet"

def load_parquet(sym: str, kind: str = "daily") -> Optional[pd.DataFrame]:
    p = parquet_path(sym, kind)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.sort_index()
    except Exception as e:
        log.warning(f"Failed to read {p}: {e}")
        return None

def save_parquet(df: pd.DataFrame, sym: str, kind: str = "daily"):
    """Atomic write: temp → rename."""
    p = parquet_path(sym, kind)
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp, engine="pyarrow", compression="snappy")
    os.replace(tmp, p)


# ── Core operations ────────────────────────────────────────────────────────────

def collect_symbol(sym: str, ticker: str, kind: str = "daily",
                   start: str = HISTORY_START) -> dict:
    """
    First-time full download for one symbol.
    Returns status dict.
    """
    end = date.today().strftime("%Y-%m-%d")
    log.info(f"  Collecting {sym} ({ticker}) {start} → {end}")

    df = fetch_yfinance(ticker, start, end)
    if df is None or df.empty:
        df = fetch_alpaca(ticker, start, end)

    if df is None or df.empty:
        return {"sym": sym, "status": "FAILED", "rows": 0, "error": "no data"}

    save_parquet(df, sym, kind)
    return {"sym": sym, "status": "OK", "rows": len(df),
            "first": str(df.index[0].date()), "last": str(df.index[-1].date())}


def update_symbol(sym: str, ticker: str, kind: str = "daily") -> dict:
    """
    Incremental update: append only the missing tail.
    Always re-fetches the last REFRESH_TAIL trading days to capture bar revisions.
    Returns status dict.
    """
    existing = load_parquet(sym, kind)
    end = date.today().strftime("%Y-%m-%d")

    if existing is None or existing.empty:
        log.info(f"  {sym}: no existing data, doing full collect")
        return collect_symbol(sym, ticker, kind)

    last_date = existing.index[-1].date()
    refresh_from = (last_date - timedelta(days=REFRESH_TAIL * 2)).strftime("%Y-%m-%d")

    log.info(f"  {sym}: existing up to {last_date}, fetching from {refresh_from}")
    new_data = fetch_yfinance(ticker, refresh_from, end)
    if new_data is None or new_data.empty:
        new_data = fetch_alpaca(ticker, refresh_from, end)

    if new_data is None or new_data.empty:
        log.warning(f"  {sym}: update fetch returned nothing — keeping existing")
        return {"sym": sym, "status": "NO_NEW", "rows": len(existing)}

    # Remove overlap, append new
    cutoff = pd.Timestamp(refresh_from)
    old    = existing[existing.index < cutoff]
    merged = pd.concat([old, new_data]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    save_parquet(merged, sym, kind)
    new_rows = len(merged) - len(existing)
    return {"sym": sym, "status": "UPDATED", "rows": len(merged),
            "new_rows": new_rows, "last": str(merged.index[-1].date())}


# ── Quality validation ─────────────────────────────────────────────────────────

def validate_symbol(sym: str, kind: str = "daily") -> dict:
    """
    Run data quality checks on one symbol. Returns issues list.

    Checks:
      1. Completeness: are there unexpected gaps in the trading calendar?
      2. OHLC consistency: high >= close >= low >= open (for sane bars)
      3. Outliers: daily return > 5 std devs from mean (price spike / data error)
      4. Zero / negative prices
      5. Volume = 0 on non-holiday trading days
      6. Staleness: last bar is more than 3 trading days ago
    """
    df = load_parquet(sym, kind)
    issues = []

    if df is None or df.empty:
        return {"sym": sym, "status": "MISSING", "issues": ["No data file"],
                "rows": 0, "score": 0}

    rows = len(df)
    score = 100   # start at 100, deduct per issue

    # ── Check 1: Staleness ──────────────────────────────────────────────────
    last_date = df.index[-1].date()
    days_stale = (date.today() - last_date).days
    # Allow for weekends: if today is Monday, last bar should be Friday
    max_stale = 5  # 3 trading days + buffer for market closure / weekend
    if days_stale > max_stale:
        issues.append(f"STALE: last bar is {last_date} ({days_stale} days ago)")
        score -= 20

    # ── Check 2: Zero / negative prices ────────────────────────────────────
    neg = (df["close"] <= 0).sum()
    if neg > 0:
        issues.append(f"NEGATIVE_PRICE: {neg} bars with close <= 0")
        score -= 30

    # ── Check 3: OHLC consistency ───────────────────────────────────────────
    bad_ohlc = ((df["high"] < df["close"]) |
                (df["low"]  > df["close"]) |
                (df["high"] < df["low"])).sum()
    if bad_ohlc > 0:
        issues.append(f"BAD_OHLC: {bad_ohlc} bars where H<C or L>C or H<L")
        score -= min(bad_ohlc * 5, 25)

    # ── Check 4: Return outliers (>5 std dev) ──────────────────────────────
    ret = df["close"].pct_change().dropna()
    if len(ret) > 30:
        mean_r, std_r = ret.mean(), ret.std()
        outliers = ret[np.abs(ret - mean_r) > 5 * std_r]
        if len(outliers) > 0:
            worst = outliers.abs().nlargest(3)
            issues.append(
                f"OUTLIERS: {len(outliers)} returns > 5σ. "
                f"Worst: {', '.join(f'{d.date()}={v*100:.1f}%' for d,v in worst.items())}"
            )
            score -= min(len(outliers) * 3, 20)

    # ── Check 5: Calendar gaps ─────────────────────────────────────────────
    # Expected trading days (rough: Mon-Fri excluding obvious holidays)
    if len(df) > 10:
        # Use CustomBusinessDay with US holidays for accurate gap detection
        try:
            from pandas.tseries.holiday import USFederalHolidayCalendar
            us_bday = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
            date_range = pd.date_range(df.index[0], df.index[-1], freq=us_bday)
        except Exception:
            date_range = pd.bdate_range(df.index[0], df.index[-1])
        expected = len(date_range)
        actual   = len(df)
        missing  = expected - actual
        if missing > 15:
            gap_pct = missing / expected * 100
            issues.append(f"GAPS: {missing} missing trading days ({gap_pct:.1f}% of expected)")
            score -= min(int(gap_pct), 15)

    # ── Check 6: Zero volume ───────────────────────────────────────────────
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 5:
            issues.append(f"ZERO_VOLUME: {zero_vol} bars with volume=0")
            score -= min(zero_vol // 5, 10)

    score = max(0, score)
    status = "OK" if score >= 90 else ("WARN" if score >= 70 else "BAD")

    return {
        "sym":    sym,
        "status": status,
        "score":  score,
        "rows":   rows,
        "first":  str(df.index[0].date()),
        "last":   str(df.index[-1].date()),
        "issues": issues,
    }


# ── Email alerts ───────────────────────────────────────────────────────────────

def send_alert(subject: str, body: str, to: str = ALERT_EMAIL):
    """
    Send a plain-text email alert. Uses SMTP.
    Reads credentials from environment:
      ALERT_SMTP_HOST  (default: smtp.gmail.com)
      ALERT_SMTP_PORT  (default: 587)
      ALERT_SMTP_USER  (sender email)
      ALERT_SMTP_PASS  (app password or SMTP password)
    If credentials not set, logs the alert instead.
    """
    host = os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("ALERT_SMTP_PORT", "587"))
    user = os.environ.get("ALERT_SMTP_USER", "")
    pwd  = os.environ.get("ALERT_SMTP_PASS", "")

    if not user or not pwd:
        log.warning(f"[ALERT — email not configured] {subject}\n{body}")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[TradingSystem] {subject}"
        msg["From"]    = user
        msg["To"]      = to
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, pwd)
            server.sendmail(user, to, msg.as_string())

        log.info(f"Alert email sent to {to}: {subject}")
    except Exception as e:
        log.error(f"Failed to send alert email: {e}")


def alert_if_issues(results: list[dict]):
    """Check validation results and send email if any BAD or WARN symbols."""
    bad  = [r for r in results if r["status"] == "BAD"]
    warn = [r for r in results if r["status"] == "WARN"]

    if not bad and not warn:
        return

    lines = [
        f"Trading System — Data Quality Alert",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"{'BAD' if bad else 'WARNING'} quality data detected. DO NOT trade without resolving.",
        f"",
    ]

    if bad:
        lines.append(f"❌ BAD ({len(bad)} symbols):")
        for r in bad:
            lines.append(f"  {r['sym']:6} score={r['score']}/100  rows={r['rows']}")
            for issue in r["issues"]:
                lines.append(f"         → {issue}")
        lines.append("")

    if warn:
        lines.append(f"⚠️  WARN ({len(warn)} symbols):")
        for r in warn:
            lines.append(f"  {r['sym']:6} score={r['score']}/100  rows={r['rows']}")
            for issue in r["issues"]:
                lines.append(f"         → {issue}")
        lines.append("")

    lines += [
        f"Run  python -m src.market_data.historical_store --audit  for full details.",
        f"Run  python -m src.market_data.historical_store --update  to refresh data.",
    ]

    body = "\n".join(lines)
    subject = f"Data Quality Alert — {len(bad)} BAD, {len(warn)} WARN symbols"
    send_alert(subject, body)


# ── Public API (used by backtest/live engine) ──────────────────────────────────

def load_all_daily(start: str = "2021-01-01",
                   end: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """
    Load all equity and macro OHLCV data into memory.
    Returns {ticker: DataFrame} dict ready for the backtest engine.
    Falls back to yfinance download if local file is missing.
    """
    end = end or date.today().strftime("%Y-%m-%d")
    result = {}

    all_syms = [(s, s, "daily") for s in EQUITY_SYMS]
    all_syms += [(k, v, "macro") for k, v in MACRO_SYMS.items()]

    for sym, ticker, kind in all_syms:
        df = load_parquet(sym, kind)
        if df is None or df.empty:
            log.debug(f"  {sym}: no local file, fetching from yfinance")
            df = fetch_yfinance(ticker, start, end)
            if df is not None and not df.empty:
                save_parquet(df, sym, kind)
        if df is not None and not df.empty:
            df = df.loc[start:end]
            # Rename to title case for backtest engine compatibility
            df = df.rename(columns={"open":"Open","high":"High","low":"Low",
                                     "close":"Close","volume":"Volume"})
            result[ticker if kind=="macro" else sym] = df

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def cmd_collect(args):
    """First-time full download of all symbols."""
    print()
    print("=" * 60)
    print("  HISTORICAL DATA COLLECTION")
    print(f"  From: {HISTORY_START}  To: {date.today()}")
    print("=" * 60)

    meta = load_meta()
    all_syms = [(s, s, "daily") for s in EQUITY_SYMS]
    all_syms += [(k, v, "macro") for k, v in MACRO_SYMS.items()]

    ok, failed = [], []
    for sym, ticker, kind in all_syms:
        result = collect_symbol(sym, ticker, kind)
        meta[sym] = {**result, "last_updated": datetime.now().isoformat(),
                     "source": "yfinance", "kind": kind}
        if result["status"] == "OK":
            ok.append(sym)
            print(f"  ✅ {sym:<6} {result['rows']:>5} rows  {result['first']} → {result['last']}")
        else:
            failed.append(sym)
            print(f"  ❌ {sym:<6} FAILED — {result.get('error','')}")

    save_meta(meta)
    print()
    print(f"  Completed: {len(ok)} OK, {len(failed)} failed")
    if failed:
        print(f"  Failed: {failed}")
        send_alert("Data Collection Failures",
                   f"Failed to collect: {failed}\nRun --collect again.")
    print(f"  Data saved to: {DATA_DIR}")
    print("=" * 60)


def cmd_update(args):
    """Incremental update — append latest bars."""
    print()
    print("=" * 60)
    print("  INCREMENTAL DATA UPDATE")
    print(f"  Refreshing tail ({REFRESH_TAIL} trading days) + any missing bars")
    print("=" * 60)

    meta = load_meta()
    all_syms = [(s, s, "daily") for s in EQUITY_SYMS]
    all_syms += [(k, v, "macro") for k, v in MACRO_SYMS.items()]

    updated, no_change, failed = [], [], []
    for sym, ticker, kind in all_syms:
        result = update_symbol(sym, ticker, kind)
        meta[sym] = {**meta.get(sym, {}), **result,
                     "last_updated": datetime.now().isoformat()}
        if result["status"] == "UPDATED":
            updated.append(sym)
            print(f"  ✅ {sym:<6} +{result.get('new_rows',0)} rows → {result.get('last','?')}")
        elif result["status"] == "NO_NEW":
            no_change.append(sym)
            print(f"  ─  {sym:<6} already up to date")
        else:
            failed.append(sym)
            print(f"  ❌ {sym:<6} {result.get('status','FAILED')}")

    save_meta(meta)
    print()
    print(f"  Updated: {len(updated)}  No change: {len(no_change)}  Failed: {len(failed)}")
    if failed:
        send_alert("Data Update Failures",
                   f"Update failed for: {failed}\nRun --update again.")

    # Auto-validate after update
    print("\n  Running post-update quality check...")
    cmd_audit(args, quiet=True)
    print("=" * 60)


def cmd_audit(args, quiet: bool = False):
    """Full data quality audit."""
    if not quiet:
        print()
        print("=" * 65)
        print("  DATA QUALITY AUDIT")
        print("=" * 65)

    all_syms = [(s, "daily") for s in EQUITY_SYMS]
    all_syms += [(k, "macro") for k in MACRO_SYMS]

    results = []
    bad, warn, ok_syms = [], [], []

    for sym, kind in all_syms:
        r = validate_symbol(sym, kind)
        results.append(r)
        if r["status"] == "BAD":
            bad.append(r)
            if not quiet:
                print(f"  ❌ {sym:<6} score={r['score']:>3}/100  {r['rows']:>5} rows  "
                      f"{r.get('first','?')} → {r.get('last','?')}")
                for issue in r["issues"]:
                    print(f"       → {issue}")
        elif r["status"] == "WARN":
            warn.append(r)
            if not quiet:
                print(f"  ⚠️  {sym:<6} score={r['score']:>3}/100  {r['rows']:>5} rows  "
                      f"{r.get('first','?')} → {r.get('last','?')}")
                for issue in r["issues"]:
                    print(f"       → {issue}")
        else:
            ok_syms.append(sym)
            if not quiet:
                print(f"  ✅ {sym:<6} score={r['score']:>3}/100  {r['rows']:>5} rows  "
                      f"{r.get('first','?')} → {r.get('last','?')}")

    if not quiet:
        print()
        print(f"  Summary: {len(ok_syms)} OK, {len(warn)} WARN, {len(bad)} BAD")
        print("=" * 65)

    # Send alert email if issues found
    alert_if_issues(results)
    return results


def cmd_status(args):
    """Quick coverage summary."""
    print()
    print("=" * 55)
    print("  DATA COVERAGE STATUS")
    print("=" * 55)
    meta = load_meta()
    all_syms = EQUITY_SYMS + list(MACRO_SYMS.keys())
    missing = [s for s in all_syms if not parquet_path(s, "daily").exists()
               and not parquet_path(s, "macro").exists()]
    present = [s for s in all_syms if s not in missing]
    print(f"  Present: {len(present)}/{len(all_syms)} symbols")
    if missing:
        print(f"  Missing: {missing}")
        print(f"  Run: python -m src.market_data.historical_store --collect")
    else:
        lasts = []
        for sym in present:
            kind = "macro" if sym in MACRO_SYMS else "daily"
            df = load_parquet(sym, kind)
            if df is not None and not df.empty:
                lasts.append(df.index[-1].date())
        if lasts:
            print(f"  Latest bar:  {max(lasts)}")
            print(f"  Oldest last: {min(lasts)}")
            if (date.today() - max(lasts)).days > 5:
                print(f"  ⚠️  Data is stale — run: python -m src.market_data.historical_store --update")
            else:
                print(f"  ✅ Data is current")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical data store")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--collect", action="store_true",
                     help="First-time full download of all symbols")
    grp.add_argument("--update",  action="store_true",
                     help="Incremental update — append latest bars")
    grp.add_argument("--audit",   action="store_true",
                     help="Full data quality audit")
    grp.add_argument("--status",  action="store_true",
                     help="Quick coverage summary")
    args = parser.parse_args()

    if args.collect: cmd_collect(args)
    elif args.update: cmd_update(args)
    elif args.audit:  cmd_audit(args)
    elif args.status: cmd_status(args)
