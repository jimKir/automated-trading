#!/usr/bin/env python3
"""
Verify all trading universe data files are present and up to date.
Usage: python scripts/verify_data.py
       python scripts/verify_data.py --download-missing
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

parser = argparse.ArgumentParser(description="Verify trading data completeness")
parser.add_argument("--download-missing", action="store_true",
                    help="Auto-download missing symbols via yfinance")
args = parser.parse_args()

import pandas as pd
import yaml

from data.data_store import DataStore

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "config", "settings.yaml")) as f:
    config = yaml.safe_load(f)

# Collect ALL symbols from every section of settings.yaml
_sections = [
    config.get("assets", {}).get("equities", {}).get("universe", []),
    config.get("assets", {}).get("crypto", {}).get("universe", []),
    config.get("assets", {}).get("futures", {}).get("universe", []),
    config.get("dynamic_universe", {}).get("candidates", {}).get("equities", []),
    config.get("dynamic_universe", {}).get("candidates", {}).get("futures", []),
    config.get("dynamic_universe", {}).get("candidates", {}).get("crypto", []),
    config.get("dynamic_universe", {}).get("predictive", {}).get("macro_symbols", []),
    ["VIX"],  # ChoppyDetector uses VIX directly
]
seen = set()
ALL_SYMBOLS = []
for section in _sections:
    for s in section:
        if s not in seen:
            seen.add(s)
            ALL_SYMBOLS.append(s)

store = DataStore()
missing = []
stale = []

today = pd.Timestamp.now().normalize()

print(f"\n{'Symbol':<12} {'Status':<12} {'Rows':>7} {'From':<12} "
      f"{'To':<12} {'Days Stale':>11}")
print("\u2500" * 68)

for symbol in ALL_SYMBOLS:
    df = store.load(symbol)
    if df is None or len(df) == 0:
        print(f"{symbol:<12} {'MISSING':<12}")
        missing.append(symbol)
    else:
        latest = df.index.max()
        days_old = (today - latest).days
        status = "OK" if days_old <= 5 else "STALE"
        if days_old > 5:
            stale.append(symbol)
        print(f"{symbol:<12} {status:<12} {len(df):>7} "
              f"{str(df.index.min().date()):<12} {str(latest.date()):<12} "
              f"{days_old:>11}d")

line_sep = '\u2500' * 68
print(f"\n{line_sep}")
print(f"Total: {len(ALL_SYMBOLS)} | OK: {len(ALL_SYMBOLS) - len(missing) - len(stale)} "
      f"| Missing: {len(missing)} | Stale: {len(stale)}")

if missing and args.download_missing:
    print(f"\nDownloading {len(missing)} missing symbols...")
    import yfinance as yf

    YF_MAP = {
        "BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "SOL/USD": "SOL-USD",
        "BNB/USD": "BNB-USD", "ADA/USD": "ADA-USD", "AVAX/USD": "AVAX-USD",
        "DOT/USD": "DOT-USD", "LINK/USD": "LINK-USD", "VIX": "^VIX",
        "^VIX": "^VIX",
    }
    data_dir = store.local_dir
    os.makedirs(data_dir, exist_ok=True)

    for sym in missing:
        ticker = YF_MAP.get(sym, sym)
        df = yf.download(ticker, start="2017-01-01", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            fname = (sym.replace("/", "_").replace("-", "_")
                     .replace("=", "_").replace("^", "") + ".parquet")
            df.to_parquet(os.path.join(data_dir, fname))
            print(f"  OK  {sym}: {len(df)} rows saved as {fname}")
        else:
            print(f"  FAIL {sym}: no data from yfinance")
elif missing:
    print(f"\nRun with --download-missing to auto-download: {missing}")

if stale:
    print(f"\nStale symbols (>5 days old): {stale}")
