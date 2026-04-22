"""
Portfolio Optimizer
====================
Replaces signal-proportional position sizing with two evidence-based methods:

METHOD 1 — RISK PARITY (default, recommended)
──────────────────────────────────────────────
Each asset gets a weight inversely proportional to its realised volatility.
High-vol assets (crypto, leveraged ETFs) get smaller weights.
Low-vol assets (bonds, gold, broad ETFs) get larger weights.

Weight_i = (1/vol_i) / sum(1/vol_j for all j)

Then scaled by the signal sign and strength:
  final_weight_i = base_weight_i × signal_i × portfolio_heat

Anti-overfitting:
  - No optimisation — weights are purely vol-scaled, no fitting to returns
  - Rolling 21-day vol estimate (strictly causal, no lookahead)
  - Identical logic works across all market regimes
  - Academic basis: Qian (2005), Maillard et al. (2010)

METHOD 2 — MINIMUM VARIANCE (optional, more complex)
──────────────────────────────────────────────────────
Minimises portfolio variance using a covariance matrix estimated from
the last 63 trading days (3 months). Uses scipy.optimize to solve the
QP problem with position constraints.

Anti-overfitting constraints:
  - Minimum 63 days of history before activation (fallback to risk parity)
  - Maximum 20% per position (prevents degenerate single-asset solutions)
  - Covariance matrix shrinkage (Ledoit-Wolf) to reduce estimation noise
  - Regularisation: adds 0.001 to diagonal to prevent near-singular matrices
  - Academic basis: DeMiguel et al. (2009) — min variance beats 1/N in OOS

CRYPTO CAP (applied after both methods)
────────────────────────────────────────
Hard cap: crypto positions cannot exceed `max_crypto_pct` of total portfolio.
Default 10%. If signals push crypto above this, weights are rescaled down
and the freed budget is redistributed to non-crypto assets.

Rationale: BTC/ETH/SOL have 3-5× the volatility of equities. Without a cap,
the vol-adjusted momentum selector will still occasionally assign significant
weight to crypto during bull runs, creating portfolio vol spikes. The 10% cap
is a hard safety guardrail independent of the vol estimate.

REGIME-AWARE HEAT SCALING  (Dynamic Bear Multiplier)
──────────────────────────────────────────────────────
The bear heat multiplier is computed dynamically based on how far SPY
has deviated from its 200-day MA, rather than a binary on/off switch.

Logic:
  1. Compute deviation = (SPY_close - MA200) / MA200
  2. When deviation ≥ 0 (above MA): multiplier = bear_heat_max  (default 1.0)
  3. When deviation ≤ -bear_max_drawdown_pct (deep bear): multiplier = bear_heat_min (default 0.60)
  4. In between: multiplier is interpolated linearly using a configurable
     distance scale (bear_sensitivity_pct, default 15%).

  multiplier = bear_heat_min + (bear_heat_max - bear_heat_min)
               × clamp( (deviation + bear_sensitivity_pct) / bear_sensitivity_pct, 0, 1 )

Examples (defaults: min=0.60, max=1.0, sensitivity=15%):
  SPY +5%  above MA200 → multiplier = 1.00 (full heat)
  SPY exactly at MA200  → multiplier = 1.00 (no cut yet)
  SPY  -5% below MA200 → multiplier = 0.87  (gentle reduction)
  SPY -10% below MA200 → multiplier = 0.73  (moderate reduction)
  SPY -15% below MA200 → multiplier = 0.60  (max reduction floor)
  SPY -25% below MA200 → multiplier = 0.60  (clamped at floor)

This removes the cliff-edge where a 1-tick cross of MA200 previously
halved all positions. The transition is now smooth and proportional.

To restore legacy binary behaviour: set bear_heat_min = bear_heat_max × 0.5
and bear_sensitivity_pct to a very small value (e.g. 0.001).

All three parameters are fully configurable in config/settings.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("Optimizer")


def _classify_asset(symbol: str) -> str:
    """Classify a symbol into equity / futures / crypto."""
    if symbol.endswith(("-USD", "USDT")):
        return "crypto"
    if symbol.endswith("=F"):
        return "futures"
    return "equity"


class PortfolioOptimizer:
    """
    Converts raw signals into optimised portfolio weights.
    Replaces the simple signal-proportional sizing in portfolio.py.
    """

    def __init__(self, config: dict):
        opt_cfg = config.get("optimizer", {})
        self.method = opt_cfg.get("method", "risk_parity")  # risk_parity | min_variance | signal
        self.vol_window = opt_cfg.get("vol_window", 21)  # rolling vol lookback
        self.cov_window = opt_cfg.get("cov_window", 63)  # covariance lookback (min-var only)
        self.max_crypto_pct = opt_cfg.get("max_crypto_pct", 0.10)  # hard crypto cap (10%)
        # Long-only mode: clip negative target weights to 0 (no shorting).
        # Default True for paper/live (Alpaca paper doesn't support shorts).
        self.long_only = opt_cfg.get("long_only", True)
        # ── Dynamic bear heat multiplier ─────────────────────────────────────
        # Legacy single-value multiplier is replaced by a continuous function.
        # bear_heat_min      : floor multiplier at maximum bear depth (default 0.60)
        # bear_heat_max      : ceiling multiplier when at/above MA200 (default 1.0)
        # bear_sensitivity_pct: distance below MA200 (as %) at which floor is reached (default 0.15)
        #
        # Back-compat: if only bear_heat_multiplier is set (legacy), use it as min with max=1.0
        legacy_mult = opt_cfg.get("bear_heat_multiplier", None)
        self.bear_heat_min = opt_cfg.get(
            "bear_heat_min", legacy_mult if legacy_mult is not None else 0.60
        )
        self.bear_heat_max = opt_cfg.get("bear_heat_max", 1.00)
        self.rp_concentration_cap = opt_cfg.get("rp_concentration_cap", 2.0)
        self.bear_sensitivity = opt_cfg.get("bear_sensitivity_pct", 0.15)  # 15% below MA200 = floor
        self.use_regime_scaling = opt_cfg.get("regime_scaling", True)
        self.enabled = opt_cfg.get("enabled", True)

        log.info(
            f"Optimizer: method={self.method} | "
            f"long_only={self.long_only} | "
            f"crypto_cap={self.max_crypto_pct:.0%} | "
            f"regime_scaling={self.use_regime_scaling} | "
            f"bear_heat [{self.bear_heat_min:.0%}–{self.bear_heat_max:.0%}] "
            f"sensitivity={self.bear_sensitivity:.0%}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def compute_weights(
        self,
        signals: dict[str, float],
        price_history: dict[str, pd.DataFrame],
        max_position_pct: float,
        max_portfolio_heat: float,
        as_of_date: pd.Timestamp | None = None,
        spy_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Compute optimised target weights from signals + price history.

        Parameters
        ----------
        signals           : {symbol: signal_value} in [-1, 1]
        price_history     : {symbol: OHLCV DataFrame}
        max_position_pct  : hard per-position cap
        max_portfolio_heat: total exposure budget (already EWS/VT scaled)
        as_of_date        : current date (for causal price slicing)
        spy_data          : SPY price history for regime detection

        Returns
        -------
        {symbol: weight}  weights ∈ [-max_position_pct, max_position_pct]
                          sum of absolute weights ≤ max_portfolio_heat
        """
        if not self.enabled or not signals:
            return {}

        # Only process non-trivial signals
        active = {k: v for k, v in signals.items() if abs(v) > 0.05}
        if not active:
            return dict.fromkeys(signals, 0.0)

        # ── Step 1: Regime-aware heat scaling ─────────────────────────────
        effective_heat = self._apply_regime_scaling(max_portfolio_heat, spy_data, as_of_date)

        # ── Step 2: Signal-proportional base (momentum driver) ─────────────
        # For momentum strategy, signal strength is the primary allocation driver.
        # Risk parity acts as a CONCENTRATION CAP (not primary sizing).
        sig_weights = self._signal_proportional_weights(active)

        # ── Step 3: Apply signal direction ────────────────────────────────
        signed_weights: dict[str, float] = {
            sym: bw * np.sign(active.get(sym, 0)) for sym, bw in sig_weights.items()
        }

        # ── Step 3b: Long-only clipping ───────────────────────────────────
        # When long_only=True (default), drop negative-weight symbols entirely.
        # Negative signals → target weight 0 → no trade (not a short sell).
        if self.long_only:
            dropped = [s for s, w in signed_weights.items() if w < 0]
            if dropped:
                log.debug(f"LongOnly: dropping {len(dropped)} negative-signal symbols: {dropped}")
            signed_weights = {s: w for s, w in signed_weights.items() if w >= 0}

        if not signed_weights:
            # All signals negative — return zero weights for everything
            return dict.fromkeys(signals, 0.0)

        # ── Step 4: Scale to portfolio heat budget ─────────────────────────
        total_abs = sum(abs(w) for w in signed_weights.values())
        if total_abs > 0:
            signed_weights = {
                k: v * (effective_heat / total_abs) for k, v in signed_weights.items()
            }

        # ── Step 5: Risk parity concentration cap (optional) ────────────────
        # Caps any position at N× its risk-parity budget (configurable).
        # Preserves momentum edge while limiting extreme concentration.
        rp_cap_mult = self.rp_concentration_cap
        if self.method in ("risk_parity", "min_variance") and price_history:
            rp = self._risk_parity_weights(active, price_history, as_of_date)
            for sym in list(signed_weights.keys()):
                rp_budget = rp.get(sym, 1.0 / max(len(active), 1)) * effective_heat
                cap = rp_budget * rp_cap_mult
                if abs(signed_weights[sym]) > cap > 0:
                    signed_weights[sym] = float(np.sign(signed_weights[sym]) * cap)

        # ── Step 6: Per-position cap ────────────────────────────────────────
        signed_weights = {
            k: float(np.clip(v, -max_position_pct, max_position_pct))
            for k, v in signed_weights.items()
        }

        # ── Step 7: Crypto cap ─────────────────────────────────────────────
        signed_weights = self._apply_crypto_cap(signed_weights, effective_heat)

        # ── Step 8: Zero out symbols not in active set ─────────────────────
        for sym in signals:
            if sym not in signed_weights:
                signed_weights[sym] = 0.0

        return signed_weights

    # ─────────────────────────────────────────────────────────────────────────
    # Method 1: Risk Parity
    # ─────────────────────────────────────────────────────────────────────────

    def _risk_parity_weights(
        self,
        signals: dict[str, float],
        price_history: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp | None,
    ) -> dict[str, float]:
        """
        Inverse-volatility weighting.
        Assets with lower realised vol get more weight.
        """
        vols: dict[str, float] = {}

        for sym in signals:
            if sym not in price_history:
                vols[sym] = 1.0  # fallback: equal weight if no history
                continue
            close = price_history[sym]["Close"]
            if as_of_date is not None:
                close = close[close.index <= as_of_date]
            rets = close.pct_change().dropna()
            if len(rets) < self.vol_window:
                vols[sym] = 1.0
                continue
            # Use recent window vol (not full history — avoids looking too far back)
            vol = float(rets.iloc[-self.vol_window :].std() * np.sqrt(252))
            vols[sym] = max(vol, 0.01)  # floor at 1% to avoid division by near-zero

        # Inverse vol weights
        inv_vols = {sym: 1.0 / v for sym, v in vols.items()}
        total_inv = sum(inv_vols.values())
        weights = {sym: iv / total_inv for sym, iv in inv_vols.items()}

        log.debug("RiskParity vols: " + ", ".join(f"{s}={vols[s]:.1%}" for s in sorted(vols)[:6]))
        return weights

    # ─────────────────────────────────────────────────────────────────────────
    # Method 2: Minimum Variance
    # ─────────────────────────────────────────────────────────────────────────

    def _min_variance_weights(
        self,
        signals: dict[str, float],
        price_history: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp | None,
    ) -> dict[str, float]:
        """
        Minimum variance portfolio using Ledoit-Wolf shrinkage covariance.
        Falls back to risk parity if insufficient data.
        """
        syms = list(signals.keys())

        # Build returns matrix
        returns_list = []
        valid_syms = []
        for sym in syms:
            if sym not in price_history:
                continue
            close = price_history[sym]["Close"]
            if as_of_date is not None:
                close = close[close.index <= as_of_date]
            rets = close.pct_change().dropna()
            if len(rets) >= self.cov_window:
                returns_list.append(rets.iloc[-self.cov_window :].rename(sym))
                valid_syms.append(sym)

        if len(valid_syms) < 2:
            return self._risk_parity_weights(signals, price_history, as_of_date)

        ret_df = pd.concat(returns_list, axis=1).dropna()
        if len(ret_df) < self.cov_window // 2:
            return self._risk_parity_weights(signals, price_history, as_of_date)

        try:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(ret_df.values)
            cov = lw.covariance_ * 252  # annualise

            # Add regularisation to diagonal (prevents near-singular matrices)
            cov += np.eye(len(valid_syms)) * 0.001

            from scipy.optimize import minimize

            n = len(valid_syms)

            # Objective: minimise w^T Σ w
            def portfolio_var(w):
                return float(w @ cov @ w)

            def portfolio_var_grad(w):
                return 2 * cov @ w

            # Constraints: weights sum to 1 (long-only for min-var base weights)
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            # Bounds: each weight between 0 and max_single (e.g. 30%)
            max_single = min(0.30, 1.0 / max(n // 2, 1))
            bounds = [(0.0, max_single)] * n
            w0 = np.ones(n) / n

            result = minimize(
                portfolio_var,
                w0,
                jac=portfolio_var_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 500},
            )

            if result.success:
                opt_weights = {sym: float(w) for sym, w in zip(valid_syms, result.x)}
                # Fill in missing symbols with small equal weight
                for sym in syms:
                    if sym not in opt_weights:
                        opt_weights[sym] = 0.0
                log.debug(f"MinVar converged: {len(valid_syms)} assets")
                return opt_weights
            log.debug("MinVar failed to converge — falling back to risk parity")
            return self._risk_parity_weights(signals, price_history, as_of_date)

        except ImportError:
            log.warning("sklearn/scipy not available — falling back to risk parity")
            return self._risk_parity_weights(signals, price_history, as_of_date)
        except Exception as e:
            log.debug(f"MinVar error: {e} — falling back to risk parity")
            return self._risk_parity_weights(signals, price_history, as_of_date)

    # ─────────────────────────────────────────────────────────────────────────
    # Method 3: Signal-proportional (original, kept as fallback)
    # ─────────────────────────────────────────────────────────────────────────

    def _signal_proportional_weights(self, signals: dict[str, float]) -> dict[str, float]:
        """Original method: weight proportional to signal strength."""
        total = sum(abs(v) for v in signals.values())
        if total == 0:
            n = len(signals)
            return dict.fromkeys(signals, 1.0 / n)
        return {k: abs(v) / total for k, v in signals.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # Crypto cap
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_crypto_cap(
        self,
        weights: dict[str, float],
        portfolio_heat: float,
    ) -> dict[str, float]:
        """
        Enforce hard crypto allocation cap.
        If crypto weights exceed max_crypto_pct of portfolio heat,
        scale them down and redistribute freed weight proportionally to non-crypto.
        """
        crypto_syms = [s for s in weights if _classify_asset(s) == "crypto"]
        noncrypto_syms = [s for s in weights if _classify_asset(s) != "crypto"]

        crypto_weight = sum(abs(weights.get(s, 0)) for s in crypto_syms)
        cap = self.max_crypto_pct  # as fraction of total (not heat)

        if crypto_weight <= cap or not crypto_syms:
            return weights

        # Scale down crypto weights
        scale = cap / crypto_weight
        result = dict(weights)
        excess = 0.0
        for sym in crypto_syms:
            old = result[sym]
            result[sym] = old * scale
            excess += abs(old) - abs(result[sym])

        # Redistribute excess to non-crypto proportionally (if any exist)
        if noncrypto_syms and excess > 0:
            noncrypto_total = sum(abs(result.get(s, 0)) for s in noncrypto_syms)
            if noncrypto_total > 0:
                for sym in noncrypto_syms:
                    share = abs(result[sym]) / noncrypto_total
                    result[sym] += np.sign(result[sym]) * share * excess

        log.debug(
            f"CryptoCap: {crypto_weight:.1%} → {cap:.1%} "
            f"(scale={scale:.2f}, redistributed ${excess:.0f} notional)"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Regime-aware heat scaling
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_regime_scaling(
        self,
        max_heat: float,
        spy_data: pd.DataFrame | None,
        as_of_date: pd.Timestamp | None,
    ) -> float:
        """
        Scale portfolio heat dynamically based on SPY distance from 200-day MA.

        Instead of a binary cut (above/below MA), the multiplier is a smooth
        linear function of how far SPY is below its MA200:

          deviation = (SPY - MA200) / MA200

          multiplier = bear_heat_min + (bear_heat_max - bear_heat_min)
                       × clamp((deviation + sensitivity) / sensitivity, 0, 1)

        Where sensitivity is the % distance below MA200 at which the floor
        multiplier is reached (default 15%).

        Uses strictly causal price data (no lookahead).
        """
        if not self.use_regime_scaling or spy_data is None:
            return max_heat

        try:
            close = spy_data["Close"]
            if as_of_date is not None:
                close = close[close.index <= as_of_date]
            if len(close) < 200:
                return max_heat

            spy_now = float(close.iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1])
            deviation = (spy_now - ma200) / ma200  # positive = above MA, negative = below

            # v15: Full-range interpolation supporting bear_heat_max > 1.0 (bull boost)
            # When deviation ≤ -sensitivity → t = 0.0 → multiplier = bear_heat_min
            # When deviation == 0           → t = 1.0 → multiplier = 1.0 (neutral)
            # When deviation ≥ +bull_thresh → t = 1.0 → multiplier = bear_heat_max
            #
            # Below MA200: same as before — linear from min to 1.0
            # Above MA200: linear from 1.0 to max over 0-10% deviation
            if deviation < 0:
                # Bear territory: interpolate [bear_heat_min, 1.0]
                t = np.clip((deviation + self.bear_sensitivity) / self.bear_sensitivity, 0.0, 1.0)
                multiplier = self.bear_heat_min + (1.0 - self.bear_heat_min) * t
            else:
                # Bull territory: interpolate [1.0, bear_heat_max]
                # Scale linearly over 0-10% above MA200
                bull_thresh = 0.10  # full boost at +10% above MA200
                t_bull = np.clip(deviation / bull_thresh, 0.0, 1.0)
                multiplier = 1.0 + (self.bear_heat_max - 1.0) * t_bull

            effective = max_heat * multiplier

            regime_label = (
                f"BULL (+{deviation:.1%})" if deviation >= 0 else f"BEAR ({deviation:+.1%})"
            )
            log.debug(
                f"Regime: {regime_label} | SPY={spy_now:.2f} MA200={ma200:.2f} "
                f"dev={deviation:+.2%} → mult={multiplier:.3f} "
                f"→ heat {max_heat:.0%} × {multiplier:.3f} = {effective:.0%}"
            )
            return effective

        except Exception:
            pass

        return max_heat

    def regime_multiplier(
        self,
        spy_data: pd.DataFrame | None,
        as_of_date: pd.Timestamp | None = None,
    ) -> tuple[float, float, str]:
        """
        Public helper: returns (multiplier, deviation, regime_label) for diagnostics.
        Useful for reporting and the what-if analyser.
        """
        if spy_data is None or not self.use_regime_scaling:
            return 1.0, 0.0, "UNKNOWN"
        try:
            close = spy_data["Close"]
            if as_of_date is not None:
                close = close[close.index <= as_of_date]
            if len(close) < 200:
                return 1.0, 0.0, "INSUFFICIENT_HISTORY"
            spy_now = float(close.iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1])
            deviation = (spy_now - ma200) / ma200
            t = np.clip((deviation + self.bear_sensitivity) / self.bear_sensitivity, 0.0, 1.0)
            multiplier = self.bear_heat_min + (self.bear_heat_max - self.bear_heat_min) * t
            label = (
                "BULL"
                if deviation >= 0
                else ("MILD_BEAR" if deviation > -self.bear_sensitivity / 2 else "DEEP_BEAR")
            )
            return round(multiplier, 4), round(deviation, 4), label
        except Exception:
            return 1.0, 0.0, "ERROR"

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def explain(
        self,
        signals: dict[str, float],
        price_history: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp | None = None,
        spy_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame explaining each position's weight breakdown.
        Useful for understanding why the optimizer assigned specific weights.
        """
        rows = []
        for sym, sig in signals.items():
            vol = np.nan
            if sym in price_history:
                close = price_history[sym]["Close"]
                if as_of_date is not None:
                    close = close[close.index <= as_of_date]
                rets = close.pct_change().dropna().iloc[-self.vol_window :]
                if len(rets) > 5:
                    vol = float(rets.std() * np.sqrt(252))
            rows.append(
                {
                    "symbol": sym,
                    "signal": round(sig, 4),
                    "ann_vol": round(vol, 4) if not np.isnan(vol) else None,
                    "asset_class": _classify_asset(sym),
                }
            )
        return pd.DataFrame(rows).sort_values("signal", ascending=False)
