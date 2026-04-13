"""
data_store.py — Computed Results Cache for Momentum Exploration
================================================================
Avoids re-running expensive IC computations and factor scoring on every run.
Stores DataFrames as Parquet and dicts/scalars as JSON, keyed by a content
hash of the inputs so changing parameters automatically busts the cache.

Cache directory: .cache/computed/   (gitignored — see .gitignore)
Manifest:        .cache/manifest.json

Usage — decorator style (simplest):
    from momentum_signals_exploration.data_store import cached_result

    @cached_result("ic_scores_momentum", max_age_days=7)
    def compute_ic(prices_df):
        ...
        return result_df   # can be pd.DataFrame, dict, or scalar

Usage — explicit save/load:
    from momentum_signals_exploration.data_store import save_result, load_result

    df = compute_heavy_thing()
    save_result("factor_scores_2023_2026", df)

    # Later run:
    df = load_result("factor_scores_2023_2026")
    if df is None:
        df = compute_heavy_thing()

CLI — list all cached items:
    python -m momentum_signals_exploration.data_store --list
    python -m momentum_signals_exploration.data_store --clear-expired
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = _REPO_ROOT / ".cache" / "computed"
MANIFEST_FILE = _REPO_ROOT / ".cache" / "manifest.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except Exception:
            return {"entries": {}}
    return {"entries": {}}


def _save_manifest(manifest: dict):
    """Atomic write: temp file → rename."""
    _ensure_dirs()
    tmp = MANIFEST_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2, default=str))
        os.replace(tmp, MANIFEST_FILE)
    except Exception as e:
        logger.warning(f"[DataStore] Could not save manifest: {e}")
        with contextlib.suppress(Exception):
            tmp.unlink(missing_ok=True)


def _safe_key(key: str) -> str:
    """Sanitise key for use in filenames."""
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in key)


def _file_path(safe_key: str, content_hash: str, ext: str) -> Path:
    return CACHE_DIR / f"{safe_key}_{content_hash}.{ext}"


def _content_hash(key: str, params: dict | None = None) -> str:
    h = hashlib.md5(key.encode())
    if params:
        h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:8]


def _is_expired(entry: dict, max_age_days: int | None) -> bool:
    if max_age_days is None:
        return False
    created = datetime.fromisoformat(entry.get("created", "1970-01-01"))
    return datetime.now() - created > timedelta(days=max_age_days)


# ---------------------------------------------------------------------------
# Core save / load
# ---------------------------------------------------------------------------


def save_result(
    key: str,
    data: Any,
    params: dict | None = None,
    script: str = "",
    max_age_days: int | None = 30,
    artifact_type: str = "computed",
) -> Path:
    """
    Persist *data* under *key*.

    - pd.DataFrame  → Parquet
    - dict / list   → JSON
    - scalar        → JSON (wrapped in {"value": ...})

    Returns the Path of the saved file.
    """
    import pandas as pd

    _ensure_dirs()
    safe = _safe_key(key)
    chash = _content_hash(key, params)

    if isinstance(data, pd.DataFrame):
        ext = "parquet"
        path = _file_path(safe, chash, ext)
        # Atomic write
        tmp = path.with_suffix(".tmp")
        data.to_parquet(tmp, index=True, engine="pyarrow", compression="snappy")
        os.replace(tmp, path)
    else:
        ext = "json"
        path = _file_path(safe, chash, ext)
        payload = data if isinstance(data, (dict, list)) else {"value": data}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(tmp, path)

    # Compute expiry string for display
    expires_str = None
    if max_age_days is not None:
        expires_str = (datetime.now() + timedelta(days=max_age_days)).isoformat(timespec="seconds")

    # Update manifest
    manifest = _load_manifest()
    manifest.setdefault("entries", {})[key] = {
        "key": key,
        "file": str(path.relative_to(_REPO_ROOT)),
        "created": datetime.now().isoformat(timespec="seconds"),
        "expires": expires_str,
        "size_bytes": path.stat().st_size,
        "artifact_type": artifact_type,
        "script": script,
        "params_hash": chash,
    }
    _save_manifest(manifest)

    logger.info(f"[DataStore] Saved  '{key}' → {path.name}  ({path.stat().st_size:,} bytes)")
    return path


def load_result(
    key: str,
    params: dict | None = None,
    max_age_days: int | None = None,
) -> Any | None:
    """
    Load a previously saved result.
    Returns None if not found, expired, or corrupted (caller should recompute).
    """
    import pandas as pd

    manifest = _load_manifest()
    entry = manifest.get("entries", {}).get(key)

    if not entry:
        return None

    # Check expiry
    if max_age_days is not None and _is_expired(entry, max_age_days):
        logger.info(f"[DataStore] EXPIRED '{key}' (max_age_days={max_age_days})")
        return None

    # Verify params hash matches if params provided
    if params is not None:
        expected = _content_hash(key, params)
        if entry.get("params_hash") != expected:
            logger.info(f"[DataStore] STALE   '{key}' — params changed, recomputing")
            return None

    file_path = _REPO_ROOT / entry["file"]
    if not file_path.exists():
        logger.warning(f"[DataStore] Missing file for '{key}': {file_path}")
        return None

    try:
        if file_path.suffix == ".parquet":
            data = pd.read_parquet(file_path, engine="pyarrow")
        else:
            raw = json.loads(file_path.read_text())
            data = raw.get("value", raw) if isinstance(raw, dict) and "value" in raw else raw

        # Age info for log
        created = datetime.fromisoformat(entry["created"])
        age_days = (datetime.now() - created).days
        age_str = f"{age_days}d ago" if age_days > 0 else "today"
        logger.info(f"[DataStore] HIT     '{key}' (computed {age_str}) ← {file_path.name}")
        return data

    except Exception as e:
        logger.warning(f"[DataStore] Corrupted cache for '{key}': {e} — will recompute")
        return None


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def cached_result(
    key: str,
    max_age_days: int = 30,
    params_from_args: bool = False,
    artifact_type: str = "computed",
):
    """
    Decorator that caches the return value of a function.

    Usage:
        @cached_result("ic_scores", max_age_days=7)
        def compute_ic(df):
            ...
            return result_df

    If params_from_args=True, a hash of the function's arguments is
    included in the cache key so different inputs produce different entries.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            params = None
            if params_from_args:
                import hashlib
                import pickle

                try:
                    raw = pickle.dumps((args, kwargs))
                    params = {"args_hash": hashlib.md5(raw).hexdigest()[:12]}
                except Exception:
                    pass

            cached = load_result(key, params=params, max_age_days=max_age_days)
            if cached is not None:
                return cached

            logger.info(f"[DataStore] MISS    '{key}' — recomputing...")
            result = fn(*args, **kwargs)

            import inspect

            script = inspect.getfile(fn)
            save_result(
                key,
                result,
                params=params,
                script=script,
                max_age_days=max_age_days,
                artifact_type=artifact_type,
            )
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(entries: dict):
    if not entries:
        print("  [DataStore] Cache is empty.")
        return

    header = f"  {'Key':<45} {'Type':<18} {'Created':<20} {'Expires':<20} {'Size':>10}  {'File'}"
    print()
    print("=" * 130)
    print("  COMPUTED RESULTS CACHE")
    print("=" * 130)
    print(header)
    print("  " + "─" * 128)
    for e in sorted(entries.values(), key=lambda x: x.get("created", "")):
        key = e.get("key", "")[:44]
        atype = e.get("artifact_type", "")[:17]
        created = e.get("created", "?")[:19]
        expires = e.get("expires", "never")
        if expires:
            expires = expires[:19]
        size = e.get("size_bytes", 0)
        size_s = f"{size / 1024:.1f} KB" if size < 1_048_576 else f"{size / 1_048_576:.1f} MB"
        file_s = Path(e.get("file", "")).name
        # Mark expired
        is_exp = ""
        if expires and expires != "never":
            try:
                if datetime.fromisoformat(expires) < datetime.now():
                    is_exp = " [EXPIRED]"
            except Exception:
                pass
        print(f"  {key:<45} {atype:<18} {created:<20} {expires:<20} {size_s:>10}  {file_s}{is_exp}")
    print("=" * 130)
    print(f"  Cache dir: {CACHE_DIR}")
    print()


def _clear_expired():
    manifest = _load_manifest()
    entries = manifest.get("entries", {})
    removed = 0
    for key, e in list(entries.items()):
        expires = e.get("expires")
        if expires and expires != "never":
            try:
                if datetime.fromisoformat(expires) < datetime.now():
                    file_path = _REPO_ROOT / e["file"]
                    file_path.unlink(missing_ok=True)
                    del entries[key]
                    removed += 1
                    print(f"  Removed expired: {key}")
            except Exception:
                pass
    _save_manifest(manifest)
    print(f"\n  Cleared {removed} expired entries.")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if "--clear-expired" in args:
        _clear_expired()
    elif "--list" in args or not args:
        manifest = _load_manifest()
        _print_table(manifest.get("entries", {}))
    else:
        print("Usage: python -m momentum_signals_exploration.data_store [--list | --clear-expired]")
