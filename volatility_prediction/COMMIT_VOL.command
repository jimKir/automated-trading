#!/bin/bash
# Double-click this file on macOS to commit volatility_prediction to GitHub.
# It copies the source files into the automated-trading repo, commits, and pushes.

set -e
cd "$(dirname "$0")"

REPO="$HOME/Documents/Claude/Projects/automated-trading"
SRC="$HOME/Documents/Claude/Projects/trading/volatility_prediction"
DEST="$REPO/volatility_prediction"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  COMMIT: volatility_prediction → GitHub"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Ensure repo exists
if [ ! -d "$REPO/.git" ]; then
    echo "✗ Repo not found at $REPO"
    echo "  Clone it first:  git clone https://github.com/jimKir/automated-trading.git $REPO"
    exit 1
fi

# Create destination
mkdir -p "$DEST"

echo "→ Syncing source files..."
rsync -av --delete \
    --exclude='.vol_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.h5' \
    --exclude='*.keras' \
    --exclude='*.pkl' \
    --exclude='*.html' \
    --exclude='*.json' \
    --exclude='*.csv' \
    --exclude='*.parquet' \
    --exclude='.DS_Store' \
    --exclude='checkpoints/' \
    --exclude='data/' \
    --exclude='cache/' \
    --exclude='.venv' \
    --include='requirements.txt' \
    "$SRC/" "$DEST/"

echo ""
echo "→ Staging changes..."
cd "$REPO"
git add volatility_prediction/

echo ""
echo "→ Changes to commit:"
git diff --cached --stat

echo ""
echo "→ Committing..."
git commit -m "feat(volatility): classification backtest + data caching + momentum indicators

- Add backtest_classification.py: 7-model comparison (XGBoost, LightGBM, RF,
  Extra Trees, Logistic Reg, LSTM, VotingEnsemble) on 3-class vol regime prediction
- Add hierarchical parquet data cache to DataPipeline (37x faster re-runs)
  Cache structure: .vol_cache/ohlcv/{SYMBOL}/*.parquet + vix/*.parquet
- Add --force-refresh flag to all scripts to bypass cache
- Expand features from 55 to 79: Stochastic Oscillator, ADX, PMO, RSI divergence,
  support/resistance proximity, combined confirmation signals
- Add MACD features (line, signal, histogram, crossover, acceleration, divergence)
- Update requirements.txt: add pyarrow, xgboost

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

echo ""
echo "→ Pulling latest from remote (rebase)..."
git pull --rebase origin main

echo ""
echo "→ Pushing to GitHub..."
git push origin main

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ DONE — pushed to https://github.com/jimKir/automated-trading"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Press any key to close..."
read -n 1
