#!/bin/bash
# ============================================================
# Automated Trading — Setup, Fetch & Maintain
# ============================================================
# Usage:
#   bash INSTALL_AND_RUN.sh              # First-time setup + full backfill
#   bash INSTALL_AND_RUN.sh update       # Daily delta update
#   bash INSTALL_AND_RUN.sh validate     # Run quality checks
#   bash INSTALL_AND_RUN.sh stats        # Cache statistics
#   bash INSTALL_AND_RUN.sh migrate      # Migrate legacy files
# ============================================================

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

# ── Find compatible Python ───────────────────────────────────
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        VERSION=$("$candidate" -c "import sys; print(sys.version_info[:2])")
        if [[ "$VERSION" == "(3, 12)" || "$VERSION" == "(3, 11)" || "$VERSION" == "(3, 10)" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ No compatible Python (3.10–3.12) found."
    echo "   brew install python@3.12"
    exit 1
fi

# ── Setup venv (reuse if exists) ─────────────────────────────
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment ($PYTHON)..."
    "$PYTHON" -m venv venv
fi
source venv/bin/activate

# Ensure core packages
pip install --upgrade pip --quiet 2>/dev/null

# Install all dependencies from requirements.txt, then register the package
pip install --quiet -r requirements.txt
pip install --quiet -e .
echo "✓ All dependencies installed"

# Fallback: if editable install fails, add repo root to Python path via .pth file
if ! python -c "from core.portfolio import Portfolio" 2>/dev/null; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "$(pwd)" > "$SITE_PACKAGES/trading_system.pth"
    echo "✓ Added repo root to Python path via .pth file"
fi

# ── Route command ────────────────────────────────────────────
CMD="${1:-backfill}"

case "$CMD" in
    update)
        echo "🔄 Running daily delta update..."
        python3 daily_data_update.py
        ;;
    update-key)
        echo "🔄 Updating key symbols only..."
        python3 daily_data_update.py --key-only
        ;;
    validate)
        echo "🔍 Running data quality checks..."
        python3 fetch_all.py --validate
        ;;
    stats)
        echo "📊 Cache statistics..."
        python3 fetch_all.py --stats
        ;;
    migrate)
        echo "🔧 Migrating legacy files..."
        python3 fetch_all.py --migrate
        ;;
    backfill|*)
        echo "🚀 Full backfill — fetching ALL instruments..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python3 fetch_all.py
        ;;
esac

echo ""
echo "Done. Venv active — run 'deactivate' when finished."
