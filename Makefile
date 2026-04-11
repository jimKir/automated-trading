# ============================================================
#  Development Commands
# ============================================================

.PHONY: lint format fix check test clean setup-hooks

# Run all checks (CI-style)
check: lint format-check
	@echo "✅ All checks passed"

# Lint with ruff
lint:
	python3 -m ruff check .

# Check formatting without modifying files
format-check:
	python3 -m ruff format --check .

# Auto-format code
format:
	python3 -m ruff format .

# Lint + auto-fix safe violations
fix:
	python3 -m ruff check --fix .
	python3 -m ruff format .

# Type check (src only for now)
typecheck:
	python3 -m mypy src/

# Run tests
test:
	python3 -m pytest tests/ -v

# Install dev dependencies and pre-commit hooks
setup-hooks:
	pip install ruff mypy pre-commit
	pre-commit install
	@echo "✅ Pre-commit hooks installed — linting runs automatically on each commit"

# Show remaining lint issues by category
lint-stats:
	python3 -m ruff check --statistics .

# Clean Python artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
