# data/

Market data feed, dynamic universe scanning, and historical data storage.

## `feed.py` -- DataFeed

Multi-source data feed supporting yfinance (backtest + paper), CCXT (crypto live), and IBKR (equities/futures live).

- `fetch_yfinance(symbols, start, end)`: Parallel download using ThreadPoolExecutor (8 workers)
- `DataFeed` class: Unified interface for loading single/all symbols

## `dynamic_universe_scanner.py` -- DynamicUniverseScanner

Scans for additional trading candidates via Alpaca Screener API (called once per morning).

**8 hard filters:** Min $5M avg daily dollar volume, min $10 price, max $10k price, min 500k avg volume, not in core universe, not in excluded sectors, positive 20d momentum, realized vol < 60% annualized.

**Choppy regime gating:**
- GREEN (<0.17): up to 3 names added
- YELLOW (0.17-0.27): up to 2 names
- ORANGE (0.27-0.40): up to 1 name
- RED (>0.40): 0 names (no expansion in crisis)

All dynamic names capped at 8% max weight (vs 15% for core universe).

## `pit_universe.json` -- Point-in-Time Universe

Year-keyed JSON with 20 large-cap stocks per year for survivorship-bias-free backtesting. Tracks FB->META rename (2022) and TSLA/AVGO entry (2021).

## `historical/` Directory

- `daily/`: OHLCV parquet files per symbol (backfilled from 2010)
- `metadata.json`: Per-symbol metadata (last_updated, row_count, source, gaps)

## Other Data Files

- `oos_extended_results.json`: Extended OOS backtest results
- `oos_spy_returns.csv`, `oos_strat_returns.csv`: Daily return series for paper emulation comparison
- `regime_params_validated.json`: Calibrated regime parameters (choppy thresholds, v4 planned)
