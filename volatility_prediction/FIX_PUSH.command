#!/bin/bash
# Fix: pull remote changes then push
set -e
cd "$HOME/Documents/Claude/Projects/automated-trading"
echo "→ Pulling remote changes..."
git pull --rebase origin main
echo "→ Pushing..."
git push origin main
echo ""
echo "✅ Done — pushed to github.com/jimKir/automated-trading"
echo "Press any key to close..."
read -n 1
