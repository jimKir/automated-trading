"""
Build version resolution
=========================
Priority: BUILD_VERSION env var → git SHA → "dev-unknown"
"""

from __future__ import annotations

import os
import subprocess


def get_version() -> str:
    """Return the build version string."""
    # 1. Env var set by Docker / CI
    env_ver = os.environ.get("BUILD_VERSION", "").strip()
    if env_ver:
        return env_ver

    # 2. Git short SHA (local dev fallback)
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if sha:
            return f"dev-{sha}"
    except Exception:
        pass

    # 3. Final fallback
    return "dev-unknown"


__version__ = get_version()
