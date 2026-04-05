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
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_START_TIME = time.time()
_PORTFOLIO_SNAPSHOT: dict = {}
_SIGNALS_SNAPSHOT:   dict = {}


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
            self._send_json({
                "status":   "ok",
                "mode":     os.environ.get("TRADING_MODE", "paper"),
                "uptime_s": round(time.time() - _START_TIME),
                "ts":       datetime.utcnow().isoformat() + "Z",
            })

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

        else:
            self._send_json({"error": "Not found"}, 404)


def start_server(port: int = 8080) -> None:
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[HealthCheck] Listening on :{port}")
    server.serve_forever()


def start_in_background(port: int = 8080) -> threading.Thread:
    t = threading.Thread(target=start_server, args=(port,), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    port = int(os.environ.get("HEALTHCHECK_PORT", 8080))
    start_server(port)
