# Diagnostics — Run Guide

Quick reference for running signal validation and backtests locally on Mac.

---

## One-Time Setup

```bash
# 1. Python 3.11 (required — 3.12+ breaks some packages)
brew install python@3.11

# 2. Java 17 (required for H2O AutoML)
brew install openjdk@17
echo 'export JAVA_HOME=$(brew --prefix openjdk@17)' >> ~/.zshrc
source ~/.zshrc

# 3. Clone repo and create virtual environment
cd ~/Desktop/trading/automated-trading
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip setuptools wheel
pip install numpy pandas scipy yfinance databento alpaca-py \
            requests vaderSentiment scikit-learn h2o \
            pyyaml boto3 pandas-datareader pytz tqdm

# 5. Set API keys
set -a && source .env && set +a
```

Your `.env` must contain:
```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
DATABENTO_KEY=...
```

---

## Every Time You Open a New Terminal

```bash
cd ~/Desktop/trading/automated-trading
source .venv/bin/activate
set -a && source .env && set +a
```

---

## Run the Databento Signal Validation

```bash
PYTHONPATH=. python diagnostics/validate_databento_signals.py
```

**Duration:** ~20–30 minutes (Databento API ~11s per call).  
**Output:** IC table printed to terminal + `/tmp/databento_validation.json`.

What it validates:
- Closing auction imbalance (XNAS.ITCH) — expected IC@5d ~0.06
- Opening cross volume anomaly (XNAS.ITCH statistics)
- OPRA real options flow (OPRA.PILLAR trades)

---

## Run the OOS Backtest (2023–2026)

```bash
PYTHONPATH=. python diagnostics/oos_3yr.py
```

**Duration:** ~5 minutes.  
Produces year-by-year performance table with full risk metrics.

---

## Run the Signal Quality Diagnostic

```bash
PYTHONPATH=. python diagnostics/signal_diagnostic_clean.py
```

Runs permutation test + walk-forward IC across all existing factors.
Includes regime breakdown (bull/bear/calm/crisis).

---

## Run the Calm Market Forensics

```bash
PYTHONPATH=. python diagnostics/calm_market_forensics.py
```

Diagnoses per-factor performance in low-vol vs high-vol regimes.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'strategy'` | Add `PYTHONPATH=.` before `python` |
| `No module named 'yfinance'` | Run `source .venv/bin/activate` first |
| `ModuleNotFoundError: pkg_resources` | You're on Python 3.14 — switch to 3.11 |
| `H2O init failed` | Run `java -version` — needs OpenJDK 17+ |
| `DATABENTO_KEY not set` | Run `set -a && source .env && set +a` |
| Databento `422 symbology` error | Already fixed in latest main — `git pull` |

---

## File Map

```
diagnostics/
├── validate_databento_signals.py   ← Databento IC validation (THIS FILE)
├── signal_diagnostic_clean.py      ← Raw signal Sharpe + permutation test
├── calm_market_forensics.py        ← Per-regime factor breakdown
├── oos_3yr.py                      ← Full OOS backtest 2023-2026
└── README.md                       ← This file
```
