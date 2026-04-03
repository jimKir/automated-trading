#!/bin/bash
# Auto-commit momentum scanner V2 to GitHub
# This script runs in Terminal when double-clicked from Finder

set -e

REPO="$HOME/Documents/Claude/Projects/automated-trading"
SRC="$HOME/Documents/Claude/Projects/trading/momentum_signals_exploration"
DEST="$REPO/momentum_signals_exploration"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Momentum Scanner V2 — GitHub Commit"
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
          --exclude='scan_results.json' \
          --exclude='ranking_report.json' \
          --exclude='momentum_v2_report.json' \
          --exclude='*_dashboard.html' \
          --exclude='scanner_fixed.py' \
          --exclude='scanner_working.py' \
          --exclude='main_working.py' \
          --exclude='run_complete_analysis.py' \
          --exclude='DO_COMMIT.command' \
          --exclude='COMMIT_NOW.sh' \
          "$SRC/" "$DEST/"

cd "$REPO"

echo ""
echo "→ Staging files..."
git add momentum_signals_exploration/

echo ""
echo "→ Files to be committed:"
git diff --cached --name-only
echo ""

# Security check
if git diff --cached | grep -qi "PKYLHTDCWWAPTXZ6JUSF66JGCS\|8eEbShK7MTfzLn1fLifrcfpunnfMSt5rvpq5uBNS21UY"; then
    echo "⚠️  ABORT — hardcoded credentials detected! Fix before committing."
    exit 1
fi
echo "✅ Security check passed — no credentials found"
echo ""

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
- analysis/: ranking comparison, DataBento integration, backtest tools
- Updated .gitignore: excludes .venv and generated outputs"

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
