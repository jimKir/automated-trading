"""
YAML config loader with environment variable override support.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load YAML config, interpolating ${ENV_VAR} references."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")

    with open(cfg_path) as f:
        raw = f.read()

    # Simple env var injection  ${VAR_NAME}
    import re
    def replace_env(match):
        var = match.group(1)
        value = os.environ.get(var)
        if value is None:
            import logging
            logging.getLogger("ConfigLoader").warning(
                f"Environment variable ${{{var}}} not set; keeping placeholder"
            )
            return match.group(0)
        return value

    raw = re.sub(r"\$\{(\w+)\}", replace_env, raw)
    config = yaml.safe_load(raw)
    return config
