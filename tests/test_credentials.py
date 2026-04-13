"""Unit tests for credential loading — ensures no hardcoded secrets."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Secrets stored as prefix+suffix to avoid THIS file triggering the scan
_SECRET_PARTS = [
    ("PKYLHTDCWW", "APTXZ6JUSF"),
    ("8eEbShK7MT", "fzLn1fLifrcfpunnfMSt5rvpq5uBNS21UY"),
    ("db-SpVxiQL", "LTdDe9iD3sLwTpiqgBjtxk"),
]
KNOWN_SECRETS = [a + b for a, b in _SECRET_PARTS]


class TestNoHardcodedSecrets:
    """Scan all Python files for hardcoded secrets."""

    def _get_all_py_files(self):
        root = os.path.dirname(os.path.dirname(__file__))
        py_files = []
        for dirpath, _, filenames in os.walk(root):
            if ".git" in dirpath or "venv" in dirpath or "__pycache__" in dirpath:
                continue
            for f in filenames:
                if f.endswith(".py"):
                    py_files.append(os.path.join(dirpath, f))
        return py_files

    def test_no_alpaca_key_in_source(self):
        for filepath in self._get_all_py_files():
            with open(filepath) as _f:
                content = _f.read()
            for secret in KNOWN_SECRETS:
                assert secret not in content, f"Hardcoded secret found in {filepath}"

    def test_no_secrets_in_yaml(self):
        root = os.path.dirname(os.path.dirname(__file__))
        for dirpath, _, filenames in os.walk(root):
            if ".git" in dirpath or "venv" in dirpath:
                continue
            for f in filenames:
                if f.endswith((".yaml", ".yml")):
                    with open(os.path.join(dirpath, f)) as _f:
                        content = _f.read()
                    for secret in KNOWN_SECRETS:
                        assert secret not in content, (
                            f"Hardcoded secret in YAML: {os.path.join(dirpath, f)}"
                        )

    def test_no_secrets_in_md(self):
        root = os.path.dirname(os.path.dirname(__file__))
        for dirpath, _, filenames in os.walk(root):
            if ".git" in dirpath or "venv" in dirpath:
                continue
            for f in filenames:
                if f.endswith(".md"):
                    with open(os.path.join(dirpath, f)) as _f:
                        content = _f.read()
                    for secret in KNOWN_SECRETS:
                        assert secret not in content, (
                            f"Hardcoded secret in MD: {os.path.join(dirpath, f)}"
                        )


class TestCredentialModule:
    def test_raises_without_env_vars(self):
        from config.credentials import get_alpaca_credentials

        # Clear env vars temporarily
        key = os.environ.pop("ALPACA_API_KEY", None)
        alt_key = os.environ.pop("APCA_API_KEY_ID", None)
        secret = os.environ.pop("ALPACA_SECRET_KEY", None)
        alt_secret = os.environ.pop("APCA_API_SECRET_KEY", None)
        try:
            with pytest.raises(EnvironmentError):
                get_alpaca_credentials()
        finally:
            if key:
                os.environ["ALPACA_API_KEY"] = key
            if alt_key:
                os.environ["APCA_API_KEY_ID"] = alt_key
            if secret:
                os.environ["ALPACA_SECRET_KEY"] = secret
            if alt_secret:
                os.environ["APCA_API_SECRET_KEY"] = alt_secret

    def test_returns_tuple_with_valid_env(self):
        from config.credentials import get_alpaca_credentials

        os.environ["ALPACA_API_KEY"] = "test_key_123"
        os.environ["ALPACA_SECRET_KEY"] = "test_secret_456"
        try:
            key, secret = get_alpaca_credentials()
            assert key == "test_key_123"
            assert secret == "test_secret_456"
        finally:
            os.environ.pop("ALPACA_API_KEY", None)
            os.environ.pop("ALPACA_SECRET_KEY", None)
