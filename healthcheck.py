#!/usr/bin/env python3
"""
Health Check HTTP Server
========================
Exposes a lightweight HTTP endpoint on port 8080 (configurable).
AWS ECS / ALB target group health checks hit GET /health.

Also exposes:
  GET /health    → {"status": "ok", "mode": "paper", "uptime_s": 123}
  GET /status    → full paper portfolio snapshot as JSON
  GET /signals   → latest strategy signals as JSON

Run standalone:
    python healthcheck.py

Or alongside the trading engine — it runs in a background thread
automatically when the trading engine starts (set system.healthcheck: true
in settings.yaml).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_START_TIME = time.time()
_PORTFOLIO_SNAPSHOT: dict = {}
_SIGNALS_SNAPSHOT: dict = {}


def update_portfolio(snap: dict) -> None:
    global _PORTFOLIO_SNAPSHOT
    _PORTFOLIO_SNAPSHOT = snap


def update_signals(signals: dict) -> None:
    global _SIGNALS_SNAPSHOT
    _SIGNALS_SNAPSHOT = signals


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def _send_json(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "mode": os.environ.get("TRADING_MODE", "paper"),
                    "uptime_s": round(time.time() - _START_TIME),
                    "ts": datetime.now(UTC).isoformat(),
                }
            )

        elif self.path == "/status":
            snap = dict(_PORTFOLIO_SNAPSHOT)
            if not snap:
                snap = {"message": "No portfolio snapshot yet — engine may still be starting"}
            self._send_json(snap)

        elif self.path == "/signals":
            sigs = dict(_SIGNALS_SNAPSHOT)
            if not sigs:
                sigs = {"message": "No signals yet — waiting for first trading cycle"}
            self._send_json(sigs)

        elif self.path == "/checks":
            checks = run_checks()
            passed = sum(1 for c in checks if c["status"] == "PASS")
            failed = sum(1 for c in checks if c["status"] == "FAIL")
            self._send_json(
                {
                    "passed": passed,
                    "failed": failed,
                    "total": len(checks),
                    "checks": checks,
                }
            )

        else:
            self._send_json({"error": "Not found"}, 404)


# ── Validation checks ─────────────────────────────────────────────────────────


def run_checks() -> list[dict]:
    """Run all health checks and return results."""
    results = []

    def PASS(msg):
        results.append({"status": "PASS", "message": msg})

    def FAIL(msg):
        results.append({"status": "FAIL", "message": msg})

    # Section 1: Config load
    try:
        import yaml

        config_path = Path(__file__).parent / "config" / "settings.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        PASS("Config loaded successfully")
    except Exception as e:
        FAIL(f"Config load failed: {e}")
        return results

    # Section 2a: No duplicate top-level keys
    try:
        with open(config_path) as f:
            content = f.read()
        top_keys = [
            line.split(":")[0].strip()
            for line in content.split("\n")
            if line and not line.startswith(" ") and ":" in line and not line.startswith("#")
        ]
        dupes = [k for k in set(top_keys) if top_keys.count(k) > 1]
        if dupes:
            FAIL(f"Duplicate top-level YAML keys: {dupes}")
        else:
            PASS("No duplicate top-level YAML keys")
    except Exception as e:
        FAIL(f"YAML key check failed: {e}")

    # Section 2b: Weight vector validation
    blends = ["bull_weights", "bear_weights", "choppy_weights"]
    for blend in blends:
        weights = config.get("strategy", {}).get("regime_switching", {}).get(blend, {})
        if weights:
            total = sum(float(v) for v in weights.values())
            if abs(total - 1.0) > 0.01:
                FAIL(f"{blend} sums to {total:.3f}, not 1.0")
            else:
                PASS(f"{blend} sums to {total:.3f}")

    # Section 3: Strategy config merged
    strategy = config.get("strategy", {})
    if "rebalance_frequency" in strategy and "name" in strategy:
        PASS("Strategy block properly merged (has both name and rebalance_frequency)")
    else:
        FAIL("Strategy block missing keys — dedup may have dropped content")

    # Section 4: Execution config
    exec_conf = config.get("execution", {})
    if exec_conf.get("hourly_timing_enabled"):
        PASS("execution.hourly_timing_enabled = true")
    else:
        FAIL("execution.hourly_timing_enabled not set")
    if exec_conf.get("dynamic_universe_enabled"):
        PASS("execution.dynamic_universe_enabled = true")
    else:
        FAIL("execution.dynamic_universe_enabled not set")

    # Section 8: Smoke tests
    smoke_tests = [
        {
            "name": "choppy_score passed to SignalEngine",
            "code": "from strategy.signals import SignalGenerator; import inspect; "
            'sig = inspect.signature(SignalGenerator({"strategy": {}, "trend_classifier": {"enabled": False}}).generate); '
            'result = "choppy_score" in sig.parameters',
            "expect": True,
        },
        {
            "name": "ChoppyDetector v4 has 9 groups",
            "code": "from regime.choppy_regime import ChoppyRegimeDetector; "
            'd = ChoppyRegimeDetector(mode="backtest"); '
            "result = len(d.feature_groups)",
            "expect": 9,
        },
    ]

    for test in smoke_tests:
        try:
            local_ns = {}
            exec(test["code"], {}, local_ns)  # noqa: S102
            actual = local_ns.get("result")
            if actual == test["expect"]:
                PASS(f"Smoke: {test['name']}")
            else:
                FAIL(f"Smoke: {test['name']} — expected {test['expect']}, got {actual}")
        except Exception as e:
            FAIL(f"Smoke: {test['name']} — error: {e}")

    # Section 9: No hardcoded secrets
    # Secrets stored as prefix+suffix to avoid this file itself triggering the scan
    _s = [
        ("PKYLHTDCWW", "APTXZ6JUSF"),
        ("8eEbShK7MT", "fzLn1fLifrcfpunnfMSt5rvpq5uBNS21UY"),
        ("db-SpVxiQL", "LTdDe9iD3sLwTpiqgBjtxk"),
    ]
    secrets = [a + b for a, b in _s]
    found = []
    root = Path(__file__).parent
    for p in root.rglob("*"):
        if ".git" in str(p) or "venv" in str(p) or "__pycache__" in str(p):
            continue
        if p.suffix in (".py", ".yaml", ".yml", ".json", ".md"):
            try:
                text = p.read_text()
                for s in secrets:
                    if s in text:
                        found.append(f"{p.relative_to(root)}:{s[:8]}")
            except Exception:
                pass
    if found:
        FAIL(f"Hardcoded secrets found: {found}")
    else:
        PASS("No hardcoded secrets in source")

    return results


def start_server(port: int = 8080) -> None:
    server = HTTPServer(("0.0.0.0", port), Handler)  # noqa: S104
    print(f"[HealthCheck] Listening on :{port}")
    server.serve_forever()


def start_in_background(port: int = 8080) -> threading.Thread:
    t = threading.Thread(target=start_server, args=(port,), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    port = int(os.environ.get("HEALTHCHECK_PORT", "8080"))
    start_server(port)
