"""
Centralized credential loading from environment variables.
Never hardcode credentials -- always use this module.
"""
import os


def get_alpaca_credentials() -> tuple[str, str]:
    """Returns (api_key, secret_key) from environment."""
    api_key = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
    secret_key = os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY')
    if not api_key:
        raise OSError("ALPACA_API_KEY environment variable not set")
    if not secret_key:
        raise OSError("ALPACA_SECRET_KEY environment variable not set")
    return api_key, secret_key


def get_databento_key() -> str | None:
    """Returns Databento API key from environment, or None if not set."""
    return os.environ.get('DATABENTO_API_KEY')


def get_alert_email() -> str:
    """Returns alert email from environment with fallback."""
    return os.environ.get('ALERT_EMAIL', '')
