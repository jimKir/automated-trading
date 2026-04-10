# config/

System configuration and credential management.

## `settings.yaml` -- Master Configuration

Single source of truth for all trading system parameters.

| Section | Key Parameters |
|---|---|
| `system` | `mode: paper`, `log_level: INFO`, `healthcheck: true` |
| `capital` | `initial_equity: 25000`, `max_portfolio_heat: 0.75` |
| `risk` | `max_position_pct: 0.15`, `daily_loss_limit: 0.08`, `max_drawdown_halt: 0.15`, `kelly_fraction: 0.25` |
| `assets.equities` | 18 ETFs: SPY, QQQ, IWM, GLD, TLT, SHY, XLU, XLP, VGK, EEM, XLK, XLE, XLF, VNQ, AGG, EWJ, EMXC, XLV |
| `assets.crypto` | BTC-USD, ETH-USD, SOL-USD |
| `assets.futures` | ES=F, NQ=F, GC=F, CL=F |
| `strategy` | `lookback_fast: 20`, `lookback_slow: 60`, `zscore_entry: 2.0`, `regime_window: 126` |
| `dynamic_universe` | `top_n: 20`, `adaptive_caps: true`, `equity_cap: 60-90%` |
| `dynamic_candidates` | S&P 500 + NDX 100 screening, `min_avg_volume_usd: 5M`, `max_stocks: 500` |

## `credentials.py` -- Credential Loading

Loads all secrets from environment variables. Never hardcodes credentials.

- `get_alpaca_credentials()` -> (api_key, secret_key) from `ALPACA_API_KEY`/`APCA_API_KEY_ID`
- `get_databento_key()` -> key or None from `DATABENTO_API_KEY`
- `get_alert_email()` -> email or "" from `ALERT_EMAIL`

## Environment-Specific Overrides

- `config.yaml`: Advanced feature store/ingestion config (Databento, Alpaca data, storage paths)
- `config.dev.yaml`: Development overrides
- `config.prod.yaml`: Production overrides (e.g., live Alpaca URL, S3 storage)
