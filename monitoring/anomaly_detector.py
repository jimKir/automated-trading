"""
Runtime Anomaly Detector
========================
Monitors trading behaviour with 7 statistical health checks.
Runs each cycle (~60s) and flags anomalies that indicate bugs,
crash-loops, or risk-limit breaches.

Checks
------
1. Order Frequency Z-score      (z > 3.0)
2. Daily Portfolio Turnover     (> 1.0)
3. Round-Trip Detection         (> 2 per symbol per hour)
4. Signal Flip Rate             (> 3 flips per symbol per 24h)
5. Drawdown Velocity            (> 2% per hour)
6. Position Concentration HHI   (> 0.25)
7. Duplicate Order Detection    (same sym+side+qty within 5 min)
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from utils.logger import get_logger

log = get_logger("AnomalyDetector")

# Default thresholds (overridable via config)
_DEFAULTS = {
    "order_frequency_zscore": 3.0,
    "daily_turnover_ratio": 1.0,
    "max_roundtrips_per_symbol_1h": 2,
    "max_signal_flips_24h": 3,
    "max_hourly_drawdown_pct": 2.0,
    "portfolio_hhi": 0.25,
    "duplicate_order_window_sec": 300,
}

# Rolling windows
_ORDER_WINDOW_DAYS = 7
_TURNOVER_WINDOW_HOURS = 24
_ROUNDTRIP_WINDOW_HOURS = 1
_SIGNAL_WINDOW_HOURS = 24
_EQUITY_WINDOW_HOURS = 1
_DUPLICATE_WINDOW_SEC = 300


class AnomalyDetector:
    """Lightweight runtime anomaly detector that integrates with LiveEngine."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("monitoring", {})
        self.enabled = cfg.get("enabled", True)
        thresholds = cfg.get("thresholds", {})
        self.thresholds = {k: thresholds.get(k, v) for k, v in _DEFAULTS.items()}

        # Override duplicate window from config
        dup_window = self.thresholds.get("duplicate_order_window_sec", _DUPLICATE_WINDOW_SEC)
        self._dup_window_sec = int(dup_window)

        # Rolling state
        self._order_timestamps: list[datetime] = []
        self._order_log: list[dict] = []  # {symbol, side, qty, ts, dollar_value}
        self._signal_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self._equity_snapshots: list[tuple[datetime, float]] = []
        self._recent_orders: list[dict] = []  # for duplicate detection

    # ── Data ingestion (called by LiveEngine) ──────────────────────────

    def record_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        ts: datetime | None = None,
    ) -> None:
        """Record an executed order for anomaly tracking."""
        ts = ts or datetime.now(UTC)
        self._order_timestamps.append(ts)
        self._order_log.append(
            {
                "symbol": symbol,
                "side": side.lower(),
                "qty": qty,
                "dollar_value": qty * price,
                "ts": ts,
            }
        )
        self._recent_orders.append(
            {
                "symbol": symbol,
                "side": side.lower(),
                "qty": qty,
                "ts": ts,
            }
        )
        self._prune_old_data()

    def record_signal(self, symbol: str, signal_value: float, ts: datetime | None = None) -> None:
        """Record a signal for flip-rate tracking."""
        ts = ts or datetime.now(UTC)
        self._signal_history[symbol].append((ts, signal_value))
        self._prune_old_data()

    def record_equity(self, equity: float, ts: datetime | None = None) -> None:
        """Record an equity snapshot for drawdown tracking."""
        ts = ts or datetime.now(UTC)
        self._equity_snapshots.append((ts, equity))
        self._prune_old_data()

    # ── Core API ───────────────────────────────────────────────────────

    def run_checks(
        self,
        portfolio_weights: dict[str, float] | None = None,
        portfolio_value: float | None = None,
        capital_health: list[dict] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run all anomaly checks (7 behavioural + up to 3 capital health).

        Parameters
        ----------
        capital_health : list[dict], optional
            Output of ``CapitalManager.check_capital_health()``.
            Each entry: ``{"check": str, "value": float, "threshold": float, "status": str}``

        Returns dict of check_name -> {value, threshold, status} for ALL checks.
        """
        if not self.enabled:
            return {}

        results: dict[str, dict[str, Any]] = {}
        results["order_frequency_zscore"] = self._check_order_frequency()
        results["daily_turnover_ratio"] = self._check_turnover(portfolio_value)
        results["max_roundtrips_per_symbol_1h"] = self._check_roundtrips()
        results["max_signal_flips_24h"] = self._check_signal_flips()
        results["max_hourly_drawdown_pct"] = self._check_drawdown_velocity()
        results["portfolio_hhi"] = self._check_concentration(portfolio_weights)
        results["duplicate_orders_5min"] = self._check_duplicate_orders()

        # Capital health checks (injected by LiveEngine from CapitalManager)
        if capital_health:
            for ch in capital_health:
                results[ch["check"]] = {
                    "value": ch["value"],
                    "threshold": ch["threshold"],
                    "status": ch["status"],
                }

        return results

    def get_failed_checks(self, results: dict[str, dict[str, Any]]) -> list[str]:
        """Return names of checks that FAIL."""
        return [name for name, r in results.items() if r.get("status") == "FAIL"]

    def generate_report(
        self,
        results: dict[str, dict[str, Any]],
        account_snapshot: dict | None = None,
    ) -> dict[str, Any]:
        """Build a structured health report."""
        from version import __version__ as bot_version

        failed = self.get_failed_checks(results)
        status = "ANOMALY_DETECTED" if failed else "HEALTHY"
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "version": bot_version,
            "status": status,
            "checks": results,
            "failed_checks": failed,
            "account_snapshot": account_snapshot or {},
            "recent_orders": self._order_log[-10:],
        }

    # ── Check implementations ──────────────────────────────────────────

    def _check_order_frequency(self) -> dict[str, Any]:
        """Check 1: Order frequency z-score over rolling 7-day window."""
        threshold = self.thresholds["order_frequency_zscore"]
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=_ORDER_WINDOW_DAYS)
        recent = [t for t in self._order_timestamps if t >= cutoff]

        if len(recent) < 3:
            return {"value": 0.0, "threshold": threshold, "status": "PASS"}

        # Compute inter-order intervals in seconds
        sorted_ts = sorted(recent)
        intervals = [
            (sorted_ts[i + 1] - sorted_ts[i]).total_seconds()
            for i in range(len(sorted_ts) - 1)
        ]

        if not intervals:
            return {"value": 0.0, "threshold": threshold, "status": "PASS"}

        mean_interval = sum(intervals) / len(intervals)
        if len(intervals) < 2:
            return {"value": 0.0, "threshold": threshold, "status": "PASS"}

        variance = sum((x - mean_interval) ** 2 for x in intervals) / (len(intervals) - 1)
        std_interval = math.sqrt(variance) if variance > 0 else 0.0

        if std_interval == 0:
            return {"value": 0.0, "threshold": threshold, "status": "PASS"}

        # Z-score of the most recent interval (how abnormally fast?)
        latest_interval = intervals[-1]
        z_score = abs(mean_interval - latest_interval) / std_interval

        status = "FAIL" if z_score > threshold else "PASS"
        return {"value": round(z_score, 2), "threshold": threshold, "status": status}

    def _check_turnover(self, portfolio_value: float | None) -> dict[str, Any]:
        """Check 2: Daily portfolio turnover ratio (24h traded $ / portfolio value)."""
        threshold = self.thresholds["daily_turnover_ratio"]
        if not portfolio_value or portfolio_value <= 0:
            return {"value": 0.0, "threshold": threshold, "status": "PASS"}

        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=_TURNOVER_WINDOW_HOURS)
        recent = [o for o in self._order_log if o["ts"] >= cutoff]
        total_traded = sum(o["dollar_value"] for o in recent)
        ratio = total_traded / portfolio_value

        status = "FAIL" if ratio > threshold else "PASS"
        return {"value": round(ratio, 4), "threshold": threshold, "status": status}

    def _check_roundtrips(self) -> dict[str, Any]:
        """Check 3: Round-trips per symbol in 1h window."""
        threshold = self.thresholds["max_roundtrips_per_symbol_1h"]
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=_ROUNDTRIP_WINDOW_HOURS)
        recent = [o for o in self._order_log if o["ts"] >= cutoff]

        # A round-trip = a buy followed by a sell (or vice versa) for same symbol
        per_symbol: dict[str, list[str]] = defaultdict(list)
        for o in recent:
            per_symbol[o["symbol"]].append(o["side"])

        max_rt = 0
        for sym, sides in per_symbol.items():
            # Count direction changes as round-trips
            flips = 0
            for i in range(1, len(sides)):
                if sides[i] != sides[i - 1]:
                    flips += 1
            rt = flips // 1  # each flip is at least half a round-trip; 2 flips = 1 full RT
            max_rt = max(max_rt, rt)

        status = "FAIL" if max_rt > threshold else "PASS"
        return {"value": max_rt, "threshold": threshold, "status": status}

    def _check_signal_flips(self) -> dict[str, Any]:
        """Check 4: Signal flip rate per symbol in 24h."""
        threshold = self.thresholds["max_signal_flips_24h"]
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=_SIGNAL_WINDOW_HOURS)

        max_flips = 0
        for sym, history in self._signal_history.items():
            recent = [(ts, v) for ts, v in history if ts >= cutoff]
            if len(recent) < 2:
                continue
            # Count sign changes (ignoring zero signals)
            flips = 0
            prev_sign = None
            for _, val in recent:
                if val == 0:
                    continue
                sign = 1 if val > 0 else -1
                if prev_sign is not None and sign != prev_sign:
                    flips += 1
                prev_sign = sign
            max_flips = max(max_flips, flips)

        status = "FAIL" if max_flips > threshold else "PASS"
        return {"value": max_flips, "threshold": threshold, "status": status}

    def _check_drawdown_velocity(self) -> dict[str, Any]:
        """Check 5: Max hourly drawdown (equity drop % over any 1h window)."""
        threshold = self.thresholds["max_hourly_drawdown_pct"]
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=_EQUITY_WINDOW_HOURS)
        recent = [(ts, eq) for ts, eq in self._equity_snapshots if ts >= cutoff]

        if len(recent) < 2:
            return {"value": 0.0, "threshold": threshold, "status": "PASS"}

        peak = recent[0][1]
        max_dd_pct = 0.0
        for _, eq in recent:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd_pct = ((peak - eq) / peak) * 100.0
                max_dd_pct = max(max_dd_pct, dd_pct)

        status = "FAIL" if max_dd_pct > threshold else "PASS"
        return {"value": round(max_dd_pct, 2), "threshold": threshold, "status": status}

    def _check_concentration(self, weights: dict[str, float] | None) -> dict[str, Any]:
        """Check 6: Portfolio concentration via HHI (Herfindahl-Hirschman Index)."""
        threshold = self.thresholds["portfolio_hhi"]
        if not weights:
            return {"value": 0.0, "threshold": threshold, "status": "PASS"}

        total = sum(abs(w) for w in weights.values())
        if total == 0:
            return {"value": 0.0, "threshold": threshold, "status": "PASS"}

        # Normalise weights then compute HHI
        normed = [abs(w) / total for w in weights.values()]
        hhi = sum(w ** 2 for w in normed)

        status = "FAIL" if hhi > threshold else "PASS"
        return {"value": round(hhi, 4), "threshold": threshold, "status": status}

    def _check_duplicate_orders(self) -> dict[str, Any]:
        """Check 7: Duplicate orders (same symbol+side+qty) within window."""
        threshold = 0  # any duplicate is a fail
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self._dup_window_sec)
        recent = [o for o in self._recent_orders if o["ts"] >= cutoff]

        # Group by (symbol, side, qty) and count
        seen: dict[tuple, int] = defaultdict(int)
        duplicates = 0
        for o in recent:
            key = (o["symbol"], o["side"], round(o["qty"], 4))
            seen[key] += 1
            if seen[key] > 1:
                duplicates += 1

        status = "FAIL" if duplicates > threshold else "PASS"
        return {"value": duplicates, "threshold": threshold, "status": status}

    # ── Housekeeping ───────────────────────────────────────────────────

    def _prune_old_data(self) -> None:
        """Evict data older than the longest window (7 days)."""
        cutoff = datetime.now(UTC) - timedelta(days=_ORDER_WINDOW_DAYS + 1)

        self._order_timestamps = [t for t in self._order_timestamps if t >= cutoff]
        self._order_log = [o for o in self._order_log if o["ts"] >= cutoff]

        dup_cutoff = datetime.now(UTC) - timedelta(seconds=self._dup_window_sec * 2)
        self._recent_orders = [o for o in self._recent_orders if o["ts"] >= dup_cutoff]

        for sym in list(self._signal_history):
            self._signal_history[sym] = [
                (ts, v) for ts, v in self._signal_history[sym] if ts >= cutoff
            ]
            if not self._signal_history[sym]:
                del self._signal_history[sym]

        equity_cutoff = datetime.now(UTC) - timedelta(hours=2)
        self._equity_snapshots = [
            (ts, eq) for ts, eq in self._equity_snapshots if ts >= equity_cutoff
        ]
