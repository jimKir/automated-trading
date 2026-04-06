#!/usr/bin/env python3
"""
Daily Data Update — Incremental Sync + Quality Check + Staleness Alert
========================================================================
Run daily (via cron or scheduled task) to keep all cached data fresh.

Flow:
  1. Check which symbols have data older than today
  2. Fetch only the missing tail (delta) for each
  3. Run quality validation on updated data
  4. Check staleness of key symbols
  5. Report results (stdout + JSON + optional alerts)

Usage:
  python3 daily_data_update.py                # Run full daily update
  python3 daily_data_update.py --dry-run      # Check what would be updated
  python3 daily_data_update.py --key-only     # Update only key/liquid symbols
"""
import os, sys, json, logging, argparse
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "data_cache" / "update.log"),
    ],
)
logger = logging.getLogger("daily_update")

# Key symbols that MUST be up to date (alert if stale)
KEY_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",                      # Major ETFs
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", # Big tech
    "TSLA", "AMD", "NFLX", "JPM", "GS", "BA",        # Core holdings
]
KEY_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]

MAX_STALE_DAYS = 3  # Trading days — alert if data is older than this


def run_daily_update(dry_run=False, key_only=False):
    import pandas as pd
    from fetch_all import (
        load_cached, get_cached_end_date,
        _canonical_path, _atomic_write, _get_catalogue,
        _register, validate_dataframe, OHLCV_DIR, CRYPTO_DIR,
        CACHE_DIR, REFRESH_TAIL,
    )
    from fetch_with_retry import FetchWithRetry

    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    report = {
        "date": today,
        "mode": "dry_run" if dry_run else "update",
        "started": datetime.now().isoformat(),
        "updated": [],
        "skipped": [],
        "failed": [],
        "stale_alerts": [],
        "quality_warnings": [],
    }

    # Determine which symbols to update
    if key_only:
        symbols_eq = [(s, False) for s in KEY_SYMBOLS]
        symbols_cr = [(s, True) for s in KEY_CRYPTO]
        all_symbols = symbols_eq + symbols_cr
    else:
        all_symbols = []
        if OHLCV_DIR.exists():
            all_symbols += [(d.name, False) for d in OHLCV_DIR.iterdir() if d.is_dir()]
        if CRYPTO_DIR.exists():
            all_symbols += [
                (d.name.replace("-", "/", 1) if "/" not in d.name else d.name, True)
                for d in CRYPTO_DIR.iterdir() if d.is_dir()
            ]
        # Always include key crypto symbols even if they don't exist yet (ensures fresh daily fetch)
        for sym in KEY_CRYPTO:
            if (sym, True) not in all_symbols:
                all_symbols.append((sym, True))

    logger.info(f"Daily update: {len(all_symbols)} symbols to check ({today})")

    if dry_run:
        # Just report what would be updated
        needs_update = 0
        for sym, is_crypto in all_symbols:
            end = get_cached_end_date(sym, is_crypto)
            if end is None or end < yesterday:
                needs_update += 1
        logger.info(f"Dry run: {needs_update} symbols need updating")
        report["needs_update"] = needs_update
        return report

    # Initialize API with retry wrapper (max 3 retries per symbol)
    fetcher = FetchWithRetry(max_retries=3)
    cat = _get_catalogue()

    updated = 0
    failed = 0

    for i, (sym, is_crypto) in enumerate(all_symbols):
        cached_end = get_cached_end_date(sym, is_crypto)

        if cached_end and cached_end >= yesterday:
            report["skipped"].append(sym)
            continue

        # Fetch delta
        fetch_start = cached_end or (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        # Go back REFRESH_TAIL days from cached end to catch revisions
        if cached_end:
            fetch_start = (
                datetime.strptime(cached_end, "%Y-%m-%d") - timedelta(days=REFRESH_TAIL)
            ).strftime("%Y-%m-%d")

        # Use retry-aware fetch (max 3 retries with exponential backoff)
        df_new = fetcher.fetch_bars_with_retry(sym, fetch_start, today, is_crypto)

        if df_new is not None and not df_new.empty:
            # Merge with existing
            df_old = load_cached(sym, is_crypto)
            if df_old is not None:
                # Fix: ensure cutoff timestamp matches index timezone (handle both naive and aware)
                cutoff = pd.Timestamp(fetch_start, tz="UTC")
                if df_old.index.tz is None:
                    # If index is naive, use naive cutoff
                    cutoff = pd.Timestamp(fetch_start)
                else:
                    # If index is aware, ensure cutoff is in same timezone
                    cutoff = pd.Timestamp(fetch_start, tz="UTC").tz_convert(df_old.index.tz)
                df_old = df_old[df_old.index < cutoff]
                df_merged = pd.concat([df_old, df_new]).sort_index()
                df_merged = df_merged[~df_merged.index.duplicated(keep="last")]
            else:
                df_merged = df_new

            # Quality check
            qc = validate_dataframe(df_merged, sym)
            if qc["issues"]:
                report["quality_warnings"].append(qc)

            if _atomic_write(_canonical_path(sym, is_crypto), df_merged):
                report["updated"].append(sym)
                updated += 1
                _register(cat, sym, is_crypto, fetch_start, today,
                          len(df_merged), _canonical_path(sym, is_crypto))
            else:
                report["failed"].append(sym)
                failed += 1
        else:
            report["failed"].append(sym)
            failed += 1

        # Progress
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{len(all_symbols)} | updated:{updated} failed:{failed}")

    # ── Retry previously failed symbols ──
    if report["failed"]:
        logger.info(f"Retrying {len(report['failed'])} failed symbols...")
        retry_count = 0
        for sym in report["failed"][:50]:  # Retry up to 50 failed symbols
            is_crypto = (sym, True) in [(s, c) for s, c in all_symbols if c]
            cached_end = get_cached_end_date(sym, is_crypto)
            fetch_start = cached_end or (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
            if cached_end:
                fetch_start = (
                    datetime.strptime(cached_end, "%Y-%m-%d") - timedelta(days=REFRESH_TAIL)
                ).strftime("%Y-%m-%d")

            df_retry = fetcher.fetch_bars_with_retry(sym, fetch_start, today, is_crypto)
            if df_retry is not None and not df_retry.empty:
                df_old = load_cached(sym, is_crypto)
                if df_old is not None:
                    cutoff = pd.Timestamp(fetch_start, tz="UTC")
                    if df_old.index.tz is None:
                        cutoff = pd.Timestamp(fetch_start)
                    else:
                        cutoff = pd.Timestamp(fetch_start, tz="UTC").tz_convert(df_old.index.tz)
                    df_old = df_old[df_old.index < cutoff]
                    df_merged = pd.concat([df_old, df_retry]).sort_index()
                    df_merged = df_merged[~df_merged.index.duplicated(keep="last")]
                else:
                    df_merged = df_retry

                if _atomic_write(_canonical_path(sym, is_crypto), df_merged):
                    report["failed"].remove(sym)
                    retry_count += 1
                    logger.info(f"{sym}: ✓ Recovered on retry")

        logger.info(f"Retry result: {retry_count} symbols recovered")

    # ── Staleness check on key symbols ──
    logger.info("Checking staleness of key symbols...")
    cutoff_date = (datetime.now() - timedelta(days=MAX_STALE_DAYS + 2)).strftime("%Y-%m-%d")

    for sym in KEY_SYMBOLS:
        end = get_cached_end_date(sym, False)
        if end is None or end < cutoff_date:
            alert = {"symbol": sym, "last_date": end, "threshold": cutoff_date}
            report["stale_alerts"].append(alert)
            logger.warning(f"STALE: {sym} last bar {end or 'NONE'} (threshold: {cutoff_date})")

    for sym_safe in KEY_CRYPTO:
        sym = sym_safe.replace("-", "/", 1)
        end = get_cached_end_date(sym, True)
        if end is None:
            end = get_cached_end_date(sym_safe, True)
        if end is None or end < cutoff_date:
            report["stale_alerts"].append({"symbol": sym_safe, "last_date": end})
            logger.warning(f"STALE: {sym_safe} last bar {end or 'NONE'}")

    # ── Fire alerts if configured ──
    if report["stale_alerts"]:
        try:
            from src.market_data.monitoring.alerts import AlertManager, SlackChannel, Alert, AlertSeverity
            webhook_url = os.getenv("SLACK_WEBHOOK_URL")
            if webhook_url:
                mgr = AlertManager(channels=[SlackChannel(webhook_url)])
                stale_syms = [a["symbol"] for a in report["stale_alerts"]]
                mgr.alert_warning(
                    title="Data Staleness Alert",
                    message=f"{len(stale_syms)} key symbols have stale data: {', '.join(stale_syms[:10])}",
                    symbols=stale_syms,
                    threshold_days=MAX_STALE_DAYS,
                )
                logger.info("Slack alert sent for stale data")
        except Exception as e:
            logger.debug(f"Alert skipped (no Slack configured): {e}")

    # ── Save report ──
    report["finished"] = datetime.now().isoformat()
    report_path = CACHE_DIR / "daily_update_report.json"
    json.dump(report, open(report_path, "w"), indent=2, default=str)

    print(f"""
{'='*60}
  DAILY UPDATE COMPLETE — {today}
{'='*60}
  Updated:          {len(report['updated']):,}
  Skipped (fresh):  {len(report['skipped']):,}
  Failed:           {len(report['failed']):,}
  Quality warnings: {len(report['quality_warnings'])}
  Stale alerts:     {len(report['stale_alerts'])}
  Report:           {report_path}
{'='*60}
""")

    if report["stale_alerts"]:
        print("  STALE SYMBOLS:")
        for a in report["stale_alerts"]:
            print(f"    {a['symbol']:10s} — last bar: {a.get('last_date', 'NONE')}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Data Update")
    parser.add_argument("--dry-run", action="store_true", help="Check only, don't fetch")
    parser.add_argument("--key-only", action="store_true", help="Update only key/liquid symbols")
    args = parser.parse_args()

    run_daily_update(dry_run=args.dry_run, key_only=args.key_only)
