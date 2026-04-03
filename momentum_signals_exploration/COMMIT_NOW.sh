#!/bin/bash
# Run this from inside your automated-trading repo:
#   cd ~/Documents/Claude/Projects
#   bash automated-trading/momentum_signals_exploration/COMMIT_NOW.sh

set -e
REPO="$HOME/Documents/Claude/Projects/automated-trading"
SRC="$HOME/Documents/Claude/Projects/trading/momentum_signals_exploration"
DEST="$REPO/momentum_signals_exploration"

echo "→ Syncing files into repo..."

# Copy only the source files (no .venv, no generated outputs, no old working versions)
rsync -av --exclude='.venv' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='scan_results.json' \
          --exclude='ranking_report.json' \
          --exclude='momentum_v2_report.json' \
          --exclude='*_dashboard.html' \
          --exclude='scanner_fixed.py' \
          --exclude='scanner_working.py' \
          --exclude='main_working.py' \
          --exclude='run_complete_analysis.py' \
          "$SRC/" "$DEST/"

cd "$REPO"

echo "→ Staging files..."
git add momentum_signals_exploration/

echo "→ Files staged:"
git diff --cached --name-only

echo ""
read -p "Commit with message below? (y/n): " yn
if [ "$yn" != "y" ]; then echo "Aborted."; exit 0; fi

git commit -m "feat(momentum): add production scanner v2 + analysis toolkit

Core improvements in scanner_v2.py:
- Batch data fetch: 1 API call for all symbols (was 50 sequential)
- VWAP deviation signal (40%): price dislocation from fair value
- Relative strength vs SPY (35%): pure alpha, strips market noise
- Volume surprise log-ratio (25%): confirms real participation
- Cross-sectional Z-score normalisation: comparable scores across universe
- Market regime detection: ADX + EMA20; skips scan in choppy markets
- Sector concentration limit: max 3 signals per sector

run_v2.py:
- Single command to run full pipeline
- Colour terminal output with regime banner
- Consensus signals: all 3 factors agree (highest confidence)
- Interactive HTML dashboard: score bar chart + rel-str vs VWAP scatter

Also includes:
- run_complete_analysis.py: working Alpaca v2 API + yfinance fallback
- analysis/: ranking comparison, DataBento integration, backtest tools
- Updated .gitignore: excludes .venv and generated outputs"

echo "→ Pushing..."
git push origin main

echo ""
echo "✅  Done — committed and pushed to github.com/jimKir/automated-trading"
