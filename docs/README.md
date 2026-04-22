# Live Performance Dashboard

Static dashboard showing real-time performance of the automated trading bot
vs SPY and QQQ benchmarks.

## What it shows

- **KPI cards** — current equity, total return, benchmark comparison, max drawdown, open positions
- **Equity curve chart** — bot vs SPY vs QQQ, normalised to 100 at inception
- **Active positions** — symbol, quantity, market value, unrealised P&L
- **Recent orders** — last 20 orders with fill details

## How it works

Data is refreshed every 15 minutes during US market hours by the
`dashboard-refresh` GitHub Action. The action calls the Alpaca paper-trading
API and Yahoo Finance, computes metrics, and commits `docs/data/snapshot.json`
back to `main`. GitHub Pages serves the static site from `/docs`.

## Dashboard URL

<https://jimkir.github.io/automated-trading/>

## Local preview

Open `docs/index.html` in a browser. If `data/snapshot.json` exists it
will load automatically.
