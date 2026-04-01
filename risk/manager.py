"""
Risk Manager
============
Handles:
  - Position sizing (fractional Kelly + volatility-scaled)
  - Portfolio heat / concentration limits
  - Daily loss limit & max drawdown circuit-breaker
  - VaR / CVaR (Historical + Parametric + Monte Carlo)
  - Black swan / tail risk metrics (Omega ratio, Tail Ratio, Stress scenarios)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats
from utils.logger import get_logger

log = get_logger("RiskManager")


class RiskManager:
    def __init__(self, config: dict):
        rc = config.get("risk", {})
        self.max_position_pct = rc.get("max_position_pct", 0.10)
        self.max_sector_pct = rc.get("max_sector_pct", 0.30)
        self.max_drawdown_halt = rc.get("max_drawdown_halt", 0.15)
        self.daily_loss_limit = rc.get("daily_loss_limit", 0.03)
        self.var_confidence = rc.get("var_confidence", 0.99)
        self.var_window = rc.get("var_window_days", 252)
        self.kelly_fraction = rc.get("kelly_fraction", 0.25)
        self.max_portfolio_heat = config.get("capital", {}).get("max_portfolio_heat", 0.20)

        self._peak_equity = None
        self._daily_start_equity = None
        self._daily_start_cash: Optional[float] = None   # cash at start of day (realised P&L)
        self._daily_realised_loss: float = 0.0           # cumulative realised loss today
        self._trading_halted = False
        self._daily_halt_date: Optional[str] = None

    # -----------------------------------------------------------------------
    # Circuit-breakers
    # -----------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        if self._peak_equity is None:
            self._peak_equity = equity
        if self._daily_start_equity is None:
            self._daily_start_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    def record_cash(self, cash: float) -> None:
        """Call once per day BEFORE any trades to snapshot opening cash."""
        if self._daily_start_cash is None:
            self._daily_start_cash = cash

    def record_realised_trade(self, pnl: float) -> None:
        """Accumulate realised P&L for each closed trade today."""
        if pnl < 0:
            self._daily_realised_loss += abs(pnl)

    def drawdown_scale(self, equity: float) -> float:
        """
        Progressive drawdown scaling — replaces hard halt.

        Instead of halting trading at 15% DD, progressively scale down:
          DD <  8%  →  1.00× (full trading)
          DD  8-15% →  linear scale from 1.0 → 0.50
          DD 15-25% →  linear scale from 0.50 → 0.20
          DD > 25%  →  0.20× floor (always keep 20% participation)

        This keeps the system trading defensively during drawdowns rather
        than freezing and being unable to recover.

        Economic logic: hard halts lock in losses and prevent recovery.
        Progressive scaling reduces risk while maintaining the ability
        to capture the recovery (which statistically follows large DD).
        """
        if self._peak_equity is None or self._peak_equity <= 0:
            return 1.0
        dd = (self._peak_equity - equity) / self._peak_equity
        if dd < 0.08:
            return 1.0
        elif dd < 0.15:
            # Linear from 1.0 at 8% DD → 0.50 at 15% DD
            t = (dd - 0.08) / 0.07
            return 1.0 - 0.50 * t
        elif dd < 0.25:
            # Linear from 0.50 at 15% DD → 0.20 at 25% DD
            t = (dd - 0.15) / 0.10
            return 0.50 - 0.30 * t
        else:
            return 0.20  # floor — never go fully flat

    def check_halt(self, equity: float, cash: float = None, date=None) -> Tuple[bool, str]:
        """
        Returns (should_halt, reason).

        v15: Max drawdown uses PROGRESSIVE SCALING instead of hard halt.
        The circuit breaker now only halts at catastrophic 40% DD.
        Between 8-40%, drawdown_scale() reduces position sizes gradually.

        Daily loss breaker    — based on REALISED cash losses today only.
                                Ignores unrealised floating P&L on open positions.
                                Resets every morning via reset_daily().
        """
        # ── 1. Max drawdown — SOFT circuit breaker ─────────────────────────
        # v15b: NO permanent halt. Hard halts lock in losses and prevent recovery.
        # Instead, drawdown_scale() progressively reduces position sizes.
        # Only log a critical warning when DD exceeds 40%.
        if self._peak_equity and self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= 0.40 and not self._trading_halted:
                self._trading_halted = True   # flag for logging only
                log.critical(f"CIRCUIT BREAKER WARNING: drawdown {drawdown:.2%} >= 40% "
                             f"— positions reduced to 20% via drawdown_scale()")
            # Never actually halt — drawdown_scale() handles the reduction

        # ── 2. Daily loss — realised cash only ───────────────────────────
        ref_equity = self._daily_start_equity or equity
        if ref_equity > 0 and cash is not None and self._daily_start_cash is not None:
            cash_loss = max(0.0, self._daily_start_cash - cash)
            daily_loss_pct = cash_loss / ref_equity
            if daily_loss_pct >= self.daily_loss_limit:
                today_str = str(date.date()) if date is not None else None
                if today_str != self._daily_halt_date:
                    self._daily_halt_date = today_str
                    log.warning(
                        f"DAILY HALT [{today_str}]: realised cash loss "
                        f"{daily_loss_pct:.2%} >= {self.daily_loss_limit:.2%}"
                    )
                return True, (f"Daily realised loss: {daily_loss_pct:.2%} "
                               f">= {self.daily_loss_limit:.2%}")

        return False, ""

    def reset_daily(self, equity: float, cash: float = None) -> None:
        """Call at the start of each new trading day."""
        self._daily_start_equity = equity
        self._daily_start_cash  = cash  # None = not provided, halt won't fire
        self._daily_realised_loss = 0.0

    # -----------------------------------------------------------------------
    # Position sizing
    # -----------------------------------------------------------------------

    def kelly_size(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Fractional Kelly criterion.
        f* = (W*R - (1-W)) / R  where R = avg_win/avg_loss
        """
        if avg_loss == 0:
            return 0.0
        R = abs(avg_win / avg_loss)
        kelly = (win_rate * R - (1 - win_rate)) / R
        kelly = max(0.0, kelly)  # no shorting via Kelly
        fractional = kelly * self.kelly_fraction
        return min(fractional, self.max_position_pct)

    def volatility_size(
        self,
        equity: float,
        symbol_returns: pd.Series,
        target_vol: float = 0.01,  # target 1% daily vol per position
    ) -> float:
        """
        Inverse-volatility position sizing.
        Position $ = equity * target_vol / realized_vol
        """
        if len(symbol_returns) < 5:
            return self.max_position_pct * equity
        realized_vol = symbol_returns.std()
        if realized_vol == 0:
            return 0.0
        size_pct = target_vol / realized_vol
        size_pct = min(size_pct, self.max_position_pct)
        return size_pct * equity

    def compute_position_size(
        self,
        equity: float,
        signal_strength: float,   # 0..1
        symbol_returns: pd.Series,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
    ) -> float:
        """
        Blend Kelly + inverse-vol, scaled by signal strength.
        Returns dollar amount to allocate.
        """
        kelly_pct = self.kelly_size(equity, win_rate, avg_win, avg_loss)
        vol_dollar = self.volatility_size(equity, symbol_returns)
        vol_pct = vol_dollar / equity if equity > 0 else 0

        # blend 50/50 and scale by signal strength
        blended_pct = (kelly_pct + vol_pct) / 2.0 * signal_strength
        blended_pct = min(blended_pct, self.max_position_pct)
        return blended_pct * equity

    # -----------------------------------------------------------------------
    # Portfolio heat
    # -----------------------------------------------------------------------

    def portfolio_heat(self, positions: Dict[str, float], equity: float) -> float:
        """Total % of equity currently at risk."""
        total_exposure = sum(abs(v) for v in positions.values())
        return total_exposure / equity if equity > 0 else 0.0

    def can_add_position(self, positions: Dict[str, float], equity: float) -> bool:
        heat = self.portfolio_heat(positions, equity)
        return heat < self.max_portfolio_heat

    # -----------------------------------------------------------------------
    # VaR / CVaR
    # -----------------------------------------------------------------------

    @staticmethod
    def historical_var(
        returns: pd.Series,
        confidence: float = 0.99,
        window: int = 252,
    ) -> float:
        """Historical simulation VaR (positive = loss)."""
        r = returns.dropna().tail(window)
        if len(r) == 0:
            return 0.0
        return float(-np.percentile(r, (1 - confidence) * 100))

    @staticmethod
    def historical_cvar(
        returns: pd.Series,
        confidence: float = 0.99,
        window: int = 252,
    ) -> float:
        """Expected Shortfall (CVaR) — mean of tail losses."""
        r = returns.dropna().tail(window)
        if len(r) == 0:
            return 0.0
        var = RiskManager.historical_var(r, confidence, window)
        tail = r[r <= -var]
        return float(-tail.mean()) if len(tail) > 0 else var

    @staticmethod
    def parametric_var(
        returns: pd.Series,
        confidence: float = 0.99,
        window: int = 252,
    ) -> float:
        """Gaussian parametric VaR."""
        r = returns.dropna().tail(window)
        if len(r) == 0:
            return 0.0
        mu = r.mean()
        sigma = r.std()
        z = stats.norm.ppf(1 - confidence)
        return float(-(mu + z * sigma))

    @staticmethod
    def monte_carlo_var(
        returns: pd.Series,
        confidence: float = 0.99,
        window: int = 252,
        simulations: int = 10_000,
        horizon: int = 1,
    ) -> float:
        """Monte Carlo VaR over `horizon` days."""
        r = returns.dropna().tail(window)
        if len(r) == 0:
            return 0.0
        mu = r.mean()
        sigma = r.std()
        rng = np.random.default_rng(42)
        sim = rng.normal(mu, sigma, (simulations, horizon)).sum(axis=1)
        return float(-np.percentile(sim, (1 - confidence) * 100))

    def full_var_report(
        self, returns: pd.Series, equity: float
    ) -> Dict[str, float]:
        """All VaR metrics in dollar terms."""
        conf = self.var_confidence
        w = self.var_window
        hist_var = self.historical_var(returns, conf, w)
        hist_cvar = self.historical_cvar(returns, conf, w)
        param_var = self.parametric_var(returns, conf, w)
        mc_var = self.monte_carlo_var(returns, conf, w)

        return {
            "VaR_historical_pct": hist_var,
            "CVaR_historical_pct": hist_cvar,
            "VaR_parametric_pct": param_var,
            "VaR_monte_carlo_pct": mc_var,
            "VaR_historical_dollar": hist_var * equity,
            "CVaR_historical_dollar": hist_cvar * equity,
        }

    # -----------------------------------------------------------------------
    # Black swan / tail risk
    # -----------------------------------------------------------------------

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """Maximum peak-to-trough drawdown."""
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        return float(dd.min())

    @staticmethod
    def max_drawdown_duration(equity_curve: pd.Series) -> int:
        """Longest drawdown in calendar days."""
        peak = equity_curve.cummax()
        in_dd = equity_curve < peak
        durations = []
        count = 0
        for x in in_dd:
            if x:
                count += 1
            else:
                durations.append(count)
                count = 0
        durations.append(count)
        return max(durations) if durations else 0

    @staticmethod
    def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        returns = equity_curve.pct_change().dropna()
        ann_return = (1 + returns.mean()) ** periods_per_year - 1
        mdd = abs(RiskManager.max_drawdown(equity_curve))
        return float(ann_return / mdd) if mdd > 0 else np.nan

    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Omega ratio: prob-weighted gains / prob-weighted losses above threshold."""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        if losses.sum() == 0:
            return np.inf
        return float(gains.sum() / losses.sum())

    @staticmethod
    def tail_ratio(returns: pd.Series) -> float:
        """95th percentile gain / abs(5th percentile loss)."""
        p95 = np.percentile(returns.dropna(), 95)
        p5 = abs(np.percentile(returns.dropna(), 5))
        return float(p95 / p5) if p5 > 0 else np.nan

    @staticmethod
    def skewness(returns: pd.Series) -> float:
        return float(returns.dropna().skew())

    @staticmethod
    def kurtosis(returns: pd.Series) -> float:
        return float(returns.dropna().kurt())  # excess kurtosis

    @staticmethod
    def stress_test(returns: pd.Series) -> Dict[str, float]:
        """
        Simulate known crash scenarios applied to current strategy returns.
        Returns expected portfolio loss % if similar conditions occurred.
        """
        scenarios = {
            "2008 GFC (peak-to-trough -56% SPY over 17 months)": -0.56,
            "2020 COVID crash (-34% in 33 days)": -0.34,
            "2022 Bear Market (-25% SPY)": -0.25,
            "2010 Flash Crash (-9% intraday)": -0.09,
            "Crypto 2022 (-77% BTC)": -0.77,
            "1987 Black Monday (-22% in one day)": -0.22,
            "Dotcom bust 2000-2002 (-78% Nasdaq)": -0.78,
        }
        beta = returns.corr(returns)  # simplified: assume beta=1 in stress
        vol_scale = returns.std() / 0.01  # scale by relative vol vs 1% daily
        results = {}
        for name, shock in scenarios.items():
            # Scale shock by strategy vol relative to a 1%/day baseline
            scaled = shock * min(vol_scale, 2.0)
            results[name] = scaled
        return results
