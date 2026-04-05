#!/bin/bash
# Auto-commit volatility prediction to GitHub
set -e

REPO="$HOME/Documents/Claude/Projects/automated-trading"
SRC="$HOME/Documents/Claude/Projects/trading/volatility_prediction"
DEST="$REPO/volatility_prediction"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Volatility Prediction Engine — GitHub Commit"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check repo exists
if [ ! -d "$REPO/.git" ]; then
    echo "❌ Repo not found at: $REPO"
    echo "   Cloning from GitHub..."
    git clone https://github.com/jimKir/automated-trading.git "$REPO"
fi

echo "→ Syncing files into repo..."
mkdir -p "$DEST"

rsync -av --exclude='.venv' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='.vol_cache' \
          --exclude='vol_prediction_dashboard.html' \
          --exclude='vol_prediction_report.json' \
          --exclude='*.h5' \
          --exclude='*.keras' \
          --exclude='*.pkl' \
          --exclude='DO_COMMIT.command' \
          --exclude='.DS_Store' \
          "$SRC/" "$DEST/"

cd "$REPO"

echo ""
echo "→ Staging files..."
git add volatility_prediction/

echo ""
echo "→ Files to be committed:"
git diff --cached --name-only
echo ""

# Security check
if git diff --cached | grep -qi "PKYLHTDCWWAPTXZ6\|8eEbShK7\|api_key.*=.*'PK\|api_secret.*=.*'8e"; then
    echo "⚠️  ABORT — hardcoded credentials detected!"
    exit 1
fi
echo "✅ Security check passed"
echo ""

git commit -m "feat(volatility): add multi-model volatility prediction engine

Architecture:
- 5 OHLC volatility estimators (Yang-Zhang, Garman-Klass, Parkinson, Rogers-Satchell, close-to-close)
- 50+ features: HAR components, return distribution, volume, cross-asset (VIX), calendar, range
- Bidirectional LSTM with temporal attention mechanism
- Gradient boosting (LightGBM/XGBoost/sklearn fallback)
- HAR (Corsi 2009) baseline model
- Adaptive ensemble with QLIKE-weighted model combination

Evaluation:
- Walk-forward validation (no lookahead bias)
- QLIKE loss, MSE, MAE, R², Mincer-Zarnowitz regression

Output:
- Rich terminal output with sector summary, regime classification, alerts
- Interactive HTML dashboard (Chart.js) with 5 visualisations
- JSON report export

Same 94-symbol / 10-sector universe as momentum scanner V2."

echo ""
echo "→ Pushing to GitHub..."
git push origin main

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅  Done — committed and pushed!"
echo "  📎  github.com/jimKir/automated-trading"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Press any key to close..."
read -n 1
