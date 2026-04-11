# Security Notice

## Credential Rotation Required

The following credentials were previously hardcoded in source files and appear
in git history. These **MUST** be rotated immediately regardless of whether the
repo is public or private:

1. **Alpaca API Key** (`PKYL...JUSF (redacted — rotate immediately)`) — rotate at https://app.alpaca.markets/paper/dashboard/overview
2. **Alpaca Secret Key** (`8eEb...21UY (redacted — rotate immediately)`) — rotate at same location as above
3. **Databento API Key** (`db-Sp...xk (redacted — rotate immediately)`) — rotate at https://app.databento.com/

### Commits containing secrets in git history

- `4b6e9e3` — Alpaca API key + secret in `tests/production_readiness_test.py`
- `17c29c1`, `a53b5ce`, `5a1b096`, `3ce3617`, `21c6788`, `61746c4`, `91c57e0`, `dfe589e`, `c40aa4d`, `13772e8` — Databento key in various files

> **Note:** Git history was NOT rewritten because it would break tagged release
> commits and force-push the entire history. Instead, rotate the credentials
> so the leaked values become invalid.

## What was fixed

- All hardcoded credentials removed from working tree
- Replaced with environment variable reads (`os.environ.get(...)`)
- `config/credentials.py` added as centralized credential helper
- `.env.example` updated with all required environment variables
- `.gitignore` updated to exclude `.env` files

## Going forward

- **Never** commit credentials to any file
- Use environment variables exclusively (see `.env.example`)
- For production deployment, use a secrets manager (AWS Secrets Manager, HashiCorp Vault, or GitHub Actions secrets)
- Run `grep -rE "api_key\s*=\s*['\"][A-Za-z0-9]{20,}['\"]" . --include="*.py"` before each commit to verify
