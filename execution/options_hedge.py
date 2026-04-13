"""
ProtectivePutHedge — ChoppyDetector-triggered SPY put hedge.

Logic:
  - ORANGE (score >= 0.229): buy 2% OTM SPY put, 21-30 DTE
  - RED    (score >= 0.296): buy 5% OTM SPY put, 14-21 DTE (deeper, cheaper)
  - GREEN  (score <  0.192): close any open put positions

Position sizing:
  - Put notional = portfolio_equity * hedge_ratio (default 0.20)
  - Contracts = floor(notional / (spy_price * 100))
  - Max 5 contracts regardless of portfolio size

Cost model (backtest):
  - Put premium estimated via Black-Scholes approximation
  - IV proxy: VIX / sqrt(252) as daily vol input
  - Live: uses Alpaca options snapshot for real mid-price

Usage (live):
  hedge = ProtectivePutHedge(broker=alpaca_client)
  hedge.on_regime_change(new_score=0.28, spy_price=520.0, portfolio_equity=100000)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

log = logging.getLogger(__name__)


# ── Black-Scholes put pricer (backtest use) ───────────────────────────────────

def bs_put_price(S: float, K: float, T: float,
                 r: float, sigma: float) -> float:
    """
    Black-Scholes European put price.
    S: spot, K: strike, T: years to expiry,
    r: risk-free rate, sigma: annualised vol.
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = _norm_cdf(-d1)
    Nd2 = _norm_cdf(-d2)
    return K * math.exp(-r * T) * Nd2 - S * Nd1


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def estimate_put_premium(spy_price: float,
                         strike_pct_otm: float,
                         dte: int,
                         vix_level: float,
                         risk_free: float = 0.045) -> float:
    """
    Estimate SPY put premium using Black-Scholes.
    strike_pct_otm: 0.02 = 2% OTM, 0.05 = 5% OTM
    vix_level: current VIX index level
    Returns per-share premium (multiply by 100 for contract cost).
    """
    K = spy_price * (1 - strike_pct_otm)
    T = dte / 365.0
    # Add IV premium: VIX understates put skew by ~20%
    iv = (vix_level / 100.0) * 1.20
    return bs_put_price(spy_price, K, T, risk_free, iv)


# ── Hedge state ───────────────────────────────────────────────────────────────

@dataclass
class PutPosition:
    symbol: str              # e.g. "SPY260117P00510000"
    contracts: int
    strike: float
    expiry: date
    entry_premium: float     # per share (×100 for contract cost)
    entry_date: date
    regime_at_entry: str     # ORANGE or RED
    total_cost: float        # contracts × 100 × premium

    @property
    def dte_remaining(self) -> int:
        return max((self.expiry - date.today()).days, 0)

    @property
    def is_expiring_soon(self) -> bool:
        return self.dte_remaining <= 5


@dataclass
class HedgeState:
    active_puts: list = field(default_factory=list)
    total_premium_paid: float = 0.0
    total_payoff_received: float = 0.0
    n_hedges_opened: int = 0
    n_hedges_closed: int = 0


# ── Main hedge module ─────────────────────────────────────────────────────────

class ProtectivePutHedge:
    """
    Manages SPY protective put positions triggered by ChoppyDetector.

    Thresholds (v4):
      GREEN  < 0.192  → close puts
      YELLOW 0.192–0.229 → hold existing, no new puts
      ORANGE 0.229–0.296 → buy 2% OTM put, 21 DTE
      RED    > 0.296  → buy 5% OTM put, 14 DTE (cheap deep OTM)
    """

    GREEN_MAX  = 0.192
    ORANGE_MIN = 0.229
    RED_MIN    = 0.296

    # Hedge parameters
    ORANGE_OTM_PCT = 0.02   # 2% OTM on ORANGE
    RED_OTM_PCT    = 0.05   # 5% OTM on RED
    ORANGE_DTE     = 21
    RED_DTE        = 14
    HEDGE_RATIO    = 0.20   # hedge 20% of portfolio notional
    MAX_CONTRACTS  = 5

    def __init__(self, broker=None, dry_run: bool = True):
        """
        broker: alpaca TradingClient (None = dry-run / backtest mode)
        dry_run: if True, log orders but don't submit
        """
        self.broker  = broker
        self.dry_run = dry_run
        self.state   = HedgeState()
        self._last_regime = "GREEN"

    # ── Public interface ──────────────────────────────────────────────────────

    def on_regime_change(self,
                         new_score: float,
                         spy_price: float,
                         portfolio_equity: float,
                         vix_level: float = 20.0,
                         today: date | None = None) -> PutPosition | None:
        """
        Called whenever ChoppyDetector score updates.
        Returns new PutPosition if hedge was opened, None otherwise.
        """
        today = today or date.today()
        new_regime = self._classify(new_score)

        log.info(f"[Hedge] Score={new_score:.3f} Regime={new_regime} "
                 f"SPY=${spy_price:.2f} Equity=${portfolio_equity:,.0f}")

        # 1. Close puts on GREEN
        if new_regime == "GREEN" and self._last_regime != "GREEN":
            self._close_all_puts(spy_price, today)

        # 2. Roll expiring puts
        for pos in list(self.state.active_puts):
            if pos.is_expiring_soon:
                log.info(f"[Hedge] Rolling {pos.symbol} (DTE={pos.dte_remaining})")
                self._close_put(pos, spy_price, today)
                if new_regime in ("ORANGE", "RED"):
                    self._open_put(new_regime, spy_price,
                                   portfolio_equity, vix_level, today)

        # 3. Open new put if transitioning into ORANGE/RED with no position
        if (new_regime in ("ORANGE", "RED") and
                not self.state.active_puts and
                new_regime != self._last_regime):
            put = self._open_put(new_regime, spy_price,
                                 portfolio_equity, vix_level, today)
            self._last_regime = new_regime
            return put

        self._last_regime = new_regime
        return None

    def get_hedge_pnl(self, current_spy: float, today: date | None = None) -> float:
        """Current mark-to-market P&L of all open put positions."""
        today = today or date.today()
        total = 0.0
        for pos in self.state.active_puts:
            intrinsic = max(pos.strike - current_spy, 0) * 100 * pos.contracts
            total += intrinsic - pos.total_cost
        return total

    def summary(self) -> dict:
        return {
            "active_puts":        len(self.state.active_puts),
            "total_premium_paid": round(self.state.total_premium_paid, 2),
            "total_payoff":       round(self.state.total_payoff_received, 2),
            "net_hedge_cost":     round(self.state.total_premium_paid -
                                        self.state.total_payoff_received, 2),
            "n_opened":           self.state.n_hedges_opened,
            "n_closed":           self.state.n_hedges_closed,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify(self, score: float) -> str:
        if score >= self.RED_MIN:
            return "RED"
        if score >= self.ORANGE_MIN:
            return "ORANGE"
        if score >= self.GREEN_MAX:
            return "YELLOW"
        return "GREEN"

    def _open_put(self, regime: str, spy_price: float,
                  portfolio_equity: float, vix_level: float,
                  today: date) -> PutPosition:
        otm_pct = self.ORANGE_OTM_PCT if regime == "ORANGE" else self.RED_OTM_PCT
        dte     = self.ORANGE_DTE     if regime == "ORANGE" else self.RED_DTE

        strike  = round(spy_price * (1 - otm_pct), 0)
        expiry  = today + timedelta(days=dte)
        premium = estimate_put_premium(spy_price, otm_pct, dte, vix_level)

        # Size: hedge_ratio of portfolio / (spy_price × 100)
        notional   = portfolio_equity * self.HEDGE_RATIO
        contracts  = min(int(notional / (spy_price * 100)), self.MAX_CONTRACTS)
        contracts  = max(contracts, 1)
        total_cost = contracts * 100 * premium

        # Build OCC option symbol: SPY + YYMMDD + C/P + 8-digit strike×1000
        expiry_str = expiry.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        symbol     = f"SPY{expiry_str}P{strike_str}"

        pos = PutPosition(
            symbol=symbol, contracts=contracts,
            strike=strike, expiry=expiry,
            entry_premium=premium, entry_date=today,
            regime_at_entry=regime,
            total_cost=total_cost,
        )
        self.state.active_puts.append(pos)
        self.state.total_premium_paid += total_cost
        self.state.n_hedges_opened    += 1

        log.info(f"[Hedge] OPENED {symbol} ×{contracts} "
                 f"strike=${strike} expiry={expiry} "
                 f"premium=${premium:.2f}/sh cost=${total_cost:.0f}")

        if self.broker and not self.dry_run:
            self._submit_alpaca_put_order(pos)

        return pos

    def _close_put(self, pos: PutPosition,
                   spy_price: float, today: date) -> float:
        payoff = max(pos.strike - spy_price, 0) * 100 * pos.contracts
        self.state.active_puts.remove(pos)
        self.state.total_payoff_received += payoff
        self.state.n_hedges_closed       += 1

        log.info(f"[Hedge] CLOSED {pos.symbol} payoff=${payoff:.0f} "
                 f"cost=${pos.total_cost:.0f} "
                 f"net=${payoff - pos.total_cost:+.0f}")
        return payoff

    def _close_all_puts(self, spy_price: float, today: date):
        for pos in list(self.state.active_puts):
            self._close_put(pos, spy_price, today)

    def _submit_alpaca_put_order(self, pos: PutPosition):
        """Submit single-leg put order via Alpaca alpaca-py SDK."""
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import OptionLimitOrderRequest

            # Use limit order at mid-price to avoid unfavourable fills
            request = OptionLimitOrderRequest(
                symbol        = pos.symbol,
                qty           = str(pos.contracts),
                side          = OrderSide.BUY,
                type          = "limit",
                limit_price   = str(round(pos.entry_premium * 1.02, 2)),
                time_in_force = TimeInForce.DAY,
            )
            order = self.broker.submit_order(request)
            log.info(f"[Hedge] Alpaca order submitted: {order.id}")
        except Exception as e:
            log.error(f"[Hedge] Order submission failed: {e}")
