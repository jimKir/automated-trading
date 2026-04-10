# utils/

Shared utilities used across the trading system.

## `config_loader.py`

YAML config loader with environment variable interpolation. Resolves `${ENV_VAR}` placeholders against `os.environ`. Logs warning if variable not found.

```python
from utils.config_loader import load_config
config = load_config("config/settings.yaml")
```

## `indicators.py`

Shared technical indicators. Currently contains `compute_adx(highs, lows, closes, period=14)` -- Average Directional Index with Wilder smoothing. Returns 25.0 (neutral) for insufficient history.

## `logger.py`

Centralized logging with console (stdout) and date-stamped file handlers.

```python
from utils.logger import get_logger
log = get_logger("MyModule")
log.info("message")
```

Log format: `%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s`
Log files stored in `logs/` directory (created automatically).
