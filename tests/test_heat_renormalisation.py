"""
Tests for optimizer Step 7b — restoring the heat budget consumed by the cap steps.

compute_weights normalises to `effective_heat` at Step 4, then the risk-parity
concentration cap, per-position cap and crypto cap can each only *reduce* weights.
Before this fix nothing redistributed the shortfall, so every cut leaked straight into
cash: live deployment sat at ~65% against a 75% cap, and raising max_portfolio_heat to
0.95 would have been absorbed by the same caps.

The invariants that matter:
  - gross exposure reaches the target when headroom exists,
  - no per-symbol or crypto-group cap is ever breached to get there,
  - when caps genuinely bind, the result saturates instead of overshooting.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from core.optimizer import PortfolioOptimizer

CONFIG = {
    "optimizer": {
        "method": "risk_parity",
        "long_only": True,
        "rp_concentration_cap": 3.5,
        "regime_scaling": False,
    }
}


def _prices(vols, n=300, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return {
        f"S{i}": pd.DataFrame(
            {"Close": 100 * np.exp(np.cumsum(rng.normal(0.0003, v, n)))}, index=idx
        )
        for i, v in enumerate(vols)
    }


def _gross(weights):
    return sum(abs(w) for w in weights.values())


@pytest.fixture
def optimizer():
    return PortfolioOptimizer(CONFIG)


class TestHeatRestoration:
    def test_reaches_target_heat_when_headroom_exists(self, optimizer):
        vols = [0.004 * (1 + i * 0.9) for i in range(15)]
        ph = _prices(vols)
        # Strongest signals on the highest-vol names, whose risk-parity budgets are
        # smallest — this is what makes the concentration cap bind in production.
        signals = {f"S{i}": 0.3 + 0.045 * i for i in range(15)}

        weights = optimizer.compute_weights(signals, ph, 0.15, 0.95)
        assert _gross(weights) == pytest.approx(0.95, abs=1e-4)

    def test_recovers_the_shortfall_the_caps_created(self):
        """Without Step 7b the same inputs deploy materially less than the target."""
        vols = [0.004 * (1 + i * 0.9) for i in range(15)]
        ph = _prices(vols)
        signals = {f"S{i}": 0.3 + 0.045 * i for i in range(15)}

        fixed = PortfolioOptimizer(CONFIG).compute_weights(signals, ph, 0.15, 0.95)

        unfixed_opt = PortfolioOptimizer(CONFIG)
        unfixed_opt._restore_target_heat = lambda w, *a, **k: w
        unfixed = unfixed_opt.compute_weights(signals, ph, 0.15, 0.95)

        assert _gross(unfixed) < 0.90, "expected the caps to create a shortfall"
        assert _gross(fixed) > _gross(unfixed)
        assert _gross(fixed) == pytest.approx(0.95, abs=1e-4)

    def test_never_breaches_per_position_cap(self, optimizer):
        vols = [0.004 * (1 + i * 0.9) for i in range(15)]
        ph = _prices(vols)
        signals = {f"S{i}": 0.3 + 0.045 * i for i in range(15)}

        for cap in (0.05, 0.08, 0.10, 0.15, 0.30):
            weights = PortfolioOptimizer(CONFIG).compute_weights(signals, ph, cap, 0.95)
            assert max(abs(w) for w in weights.values()) <= cap + 1e-9, f"cap {cap} breached"

    def test_saturates_rather_than_overshooting_when_caps_bind(self):
        """
        9 names capped at 8% each cannot exceed 72% gross. The result must stop there,
        not force the target by breaching a cap.
        """
        ph = _prices([0.004 * (1 + i * 0.6) for i in range(9)])
        signals = {f"S{i}": 0.9 - 0.05 * i for i in range(9)}

        weights = PortfolioOptimizer(CONFIG).compute_weights(signals, ph, 0.08, 0.95)
        gross = _gross(weights)

        assert gross <= 0.72 + 1e-9
        assert gross == pytest.approx(0.72, abs=1e-3)

    def test_never_exceeds_target_heat(self):
        """Restoration must not push gross exposure above the requested budget."""
        ph = _prices([0.004 * (1 + i * 0.5) for i in range(12)])
        signals = {f"S{i}": 0.8 - 0.03 * i for i in range(12)}

        for heat in (0.30, 0.50, 0.75, 0.95):
            weights = PortfolioOptimizer(CONFIG).compute_weights(signals, ph, 0.15, heat)
            assert _gross(weights) <= heat + 1e-6, f"overshot heat budget {heat}"

    def test_preserves_signal_ordering(self, optimizer):
        """Redistribution is weight-proportional, so the momentum tilt must survive."""
        ph = _prices([0.005] * 8)  # equal vol, so only signal strength differentiates
        signals = {f"S{i}": 0.9 - 0.08 * i for i in range(8)}

        weights = optimizer.compute_weights(signals, ph, 0.15, 0.95)
        ordered = [weights[f"S{i}"] for i in range(8)]
        assert ordered == sorted(ordered, reverse=True), "signal ordering was not preserved"

    def test_no_active_signals_returns_zeros(self, optimizer):
        ph = _prices([0.005] * 4)
        weights = optimizer.compute_weights(dict.fromkeys(ph, 0.0), ph, 0.15, 0.95)
        assert _gross(weights) == 0.0

    def test_single_position_capped_leaves_cash(self, optimizer):
        """One name capped at 15% cannot fill a 95% budget — and must not try to."""
        ph = _prices([0.005])
        weights = optimizer.compute_weights({"S0": 0.9}, ph, 0.15, 0.95)
        assert weights["S0"] == pytest.approx(0.15, abs=1e-9)
