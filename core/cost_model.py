"""
Realistic Cost Model
=====================
Models ALL material transaction and carrying costs:

1. Commission         — per-trade, tiered by asset class (IBKR-realistic)
2. Bid-ask spread     — half-spread paid on entry AND exit, asset-class-specific
3. Market impact      — square-root model: impact scales with sqrt(order_size / ADV)
4. Overnight financing— short reborrow cost + long margin interest (for leveraged longs)
5. Futures roll cost  — basis paid when rolling front-month contracts (~4x/year)
6. Crypto funding     — perpetual funding rate (8h intervals, mean-reverting)

All costs are deducted from cash in real-time so the equity curve
reflects true after-cost performance.

Typical all-in round-trip cost estimates (vs the old flat 0.15%):
  ETF (SPY, QQQ)     : ~0.04%  (ultra-liquid, tight spread)
  Small-cap ETF      : ~0.12%  (wider spread, more impact)
  Futures (ES, NQ)   : ~0.02%  (tick spread + low commission)
  Crypto (BTC, ETH)  : ~0.10%  (exchange fee + spread)
  Small crypto (SOL) : ~0.15%
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.logger import get_logger

log = get_logger("CostModel")


# ---------------------------------------------------------------------------
# Asset classification helpers
# ---------------------------------------------------------------------------

# Mega/large-cap equities with ETF-like liquidity (SPY-comparable)
_MEGA_CAP_EQUITIES = frozenset(
    {
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "TSLA",
        "AVGO",
        "JPM",
        "V",
        "MA",
        "UNH",
        "JNJ",
        "PG",
        "HD",
        "KO",
        "XOM",
        "CVX",
        "BAC",
        "GS",
        "BRK-B",
        "LLY",
        "WMT",
        "COST",
        "NFLX",
        "ORCL",
        "AMD",
        "INTC",
        "QCOM",
        "TMO",
        "ABT",
        "DHR",
        "MS",
        "BLK",
        "SCHW",
        "C",
        "AXP",
        "RTX",
        "LMT",
    }
)


def _classify(symbol: str) -> str:
    """
    Returns asset class string:
      'equity_mega' : AAPL, MSFT, NVDA — mega-cap stocks with ETF-like liquidity
      'equity_large': typical S&P 500 names — slightly wider spread
      'etf_large'   : SPY, QQQ, IWM, GLD, TLT
      'etf_sector'  : XLK, XLE, XLF
      'future'      : ES=F, NQ=F, GC=F, CL=F
      'crypto_major': BTC-USD, ETH-USD
      'crypto_minor': SOL-USD and other alts
    """
    if symbol in ("ES=F", "NQ=F", "GC=F", "CL=F"):
        return "future"
    if symbol.endswith(("-USD", "USDT")):
        if symbol in ("BTC-USD", "ETH-USD", "BTCUSDT", "ETHUSDT"):
            return "crypto_major"
        return "crypto_minor"
    if symbol in ("XLK", "XLE", "XLF", "XLV", "XLP", "XLI", "XLB", "XLU", "XLRE"):
        return "etf_sector"
    if symbol in ("SPY", "QQQ", "IWM", "GLD", "TLT", "SHY", "HYG", "LQD", "VGK", "EEM", "VTI"):
        return "etf_large"
    if symbol in _MEGA_CAP_EQUITIES:
        return "equity_mega"  # SPY-like liquidity, slightly higher spread than ETF
    # Default: treat as liquid large-cap equity
    return "equity_large"


# ---------------------------------------------------------------------------
# Per-asset-class cost parameters (based on IBKR tiered + market reality 2024)
# ---------------------------------------------------------------------------


@dataclass
class AssetCostParams:
    # Commission
    commission_pct: float  # % of notional (one way)

    # Bid-ask spread
    half_spread_pct: float  # half of bid-ask spread, % (one way)

    # Market impact — square-root model coefficient
    # impact_pct = impact_coeff * sqrt(order_notional / adv_estimate)
    impact_coeff: float  # dimensionless
    adv_estimate: float  # average daily volume in $ (fallback if no data)

    # Overnight financing (annualised %)
    long_financing_rate: float  # margin interest for leveraged long (ann. %)
    short_borrow_rate: float  # stock borrow rate for short (ann. %)

    # Futures-specific
    roll_cost_pct: float  # cost per roll as % of notional (0 for non-futures)
    rolls_per_year: int  # typically 4 for quarterly contracts

    # Crypto-specific
    funding_rate_daily: float  # daily equivalent of perpetual funding rate


# Realistic parameters by asset class
COST_PARAMS: dict[str, AssetCostParams] = {
    "etf_large": AssetCostParams(
        commission_pct=0.0002,  # IBKR: ~$0.005/share ≈ 0.02% on $25 ETF; capped here
        half_spread_pct=0.0001,  # SPY spread ≈ $0.01 on $550 ≈ 0.002%; use 0.01%
        impact_coeff=0.05,
        adv_estimate=30_000_000_000,  # SPY: ~$30B ADV
        long_financing_rate=0.00,  # fully-paid (no leverage assumed)
        short_borrow_rate=0.003,  # ~0.3% ann. for easy-to-borrow ETFs
        roll_cost_pct=0.0,
        rolls_per_year=0,
        funding_rate_daily=0.0,
    ),
    "etf_sector": AssetCostParams(
        commission_pct=0.0002,
        half_spread_pct=0.0003,  # slightly wider than SPY/QQQ
        impact_coeff=0.08,
        adv_estimate=2_000_000_000,
        long_financing_rate=0.00,
        short_borrow_rate=0.005,  # 0.5% ann.
        roll_cost_pct=0.0,
        rolls_per_year=0,
        funding_rate_daily=0.0,
    ),
    "future": AssetCostParams(
        commission_pct=0.00005,  # ~$2 per contract on $200k ES notional ≈ 0.001%
        half_spread_pct=0.00005,  # 1 tick on ES = $12.50 / $200k ≈ 0.006%; use 0.005%
        impact_coeff=0.03,
        adv_estimate=50_000_000_000,  # ES has ~$200B+ ADV in notional
        long_financing_rate=0.00,  # futures are already leveraged via margin, not modelled
        short_borrow_rate=0.00,  # no borrow on futures
        roll_cost_pct=0.0003,  # ~0.03% per roll (basis cost)
        rolls_per_year=4,
        funding_rate_daily=0.0,
    ),
    "crypto_major": AssetCostParams(
        commission_pct=0.0004,  # Binance maker/taker ~0.04% (post-discount)
        half_spread_pct=0.0002,  # BTC spread ≈ $10 on $85k ≈ 0.01%; use 0.02%
        impact_coeff=0.10,
        adv_estimate=10_000_000_000,  # BTC spot ADV ~$10B
        long_financing_rate=0.00,
        short_borrow_rate=0.00,
        roll_cost_pct=0.0,
        rolls_per_year=0,
        funding_rate_daily=0.0001,  # 0.01%/day = 3.65%/year (typical bullish funding)
    ),
    "crypto_minor": AssetCostParams(
        commission_pct=0.0006,
        half_spread_pct=0.0005,
        impact_coeff=0.15,
        adv_estimate=500_000_000,
        long_financing_rate=0.00,
        short_borrow_rate=0.00,
        roll_cost_pct=0.0,
        rolls_per_year=0,
        funding_rate_daily=0.0002,  # higher funding on smaller cryptos
    ),
    # ── Equities ───────────────────────────────────────────────────────────────
    # Mega-cap stocks (AAPL, MSFT, NVDA, JPM…) — SPY-like liquidity
    # Spread ~$0.01 on $150-$800 stock = 0.001-0.007%; use 0.005% (conservative)
    # Commission: IBKR tiered ~$0.005/share ≈ 0.005% on $100 stock
    # Short borrow: 0.3% ann for S&P 500 members (easy to borrow)
    "equity_mega": AssetCostParams(
        commission_pct=0.0001,  # IBKR tiered: ~$0.005/share on $100 stock
        half_spread_pct=0.0005,  # $0.01 spread on $200 stock = 0.005%
        impact_coeff=0.05,  # similar to SPY (very liquid)
        adv_estimate=5_000_000_000,  # AAPL: ~$10B ADV; NVDA: ~$15B; use conservative $5B
        long_financing_rate=0.00,  # no leverage assumed
        short_borrow_rate=0.003,  # 0.3% ann. (S&P 500 easy-to-borrow)
        roll_cost_pct=0.0,
        rolls_per_year=0,
        funding_rate_daily=0.0,
    ),
    # Standard large-cap equity (S&P 500, not mega-cap)
    "equity_large": AssetCostParams(
        commission_pct=0.0002,  # slightly higher for lower-priced shares
        half_spread_pct=0.0010,  # $0.01-0.02 spread on $100-200 stock
        impact_coeff=0.08,
        adv_estimate=500_000_000,  # typical S&P 500: $500M-$2B ADV
        long_financing_rate=0.00,
        short_borrow_rate=0.005,  # 0.5% ann.
        roll_cost_pct=0.0,
        rolls_per_year=0,
        funding_rate_daily=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Core CostModel
# ---------------------------------------------------------------------------


@dataclass
class CostBreakdown:
    commission: float = 0.0
    half_spread: float = 0.0
    market_impact: float = 0.0
    total_transaction: float = 0.0  # sum of above (one way)

    overnight_financing: float = 0.0
    futures_roll: float = 0.0
    crypto_funding: float = 0.0
    total_carrying: float = 0.0  # sum of carrying costs for one day

    @property
    def total(self) -> float:
        return self.total_transaction + self.total_carrying


class CostModel:
    """
    Computes realistic costs for every trade and every holding day.

    Usage pattern:
        cost_model = CostModel(config)

        # On each trade:
        breakdown = cost_model.transaction_cost(symbol, notional, is_buy=True)
        cash -= breakdown.total_transaction

        # Each day for each open position:
        daily_cost = cost_model.daily_carrying_cost(symbol, position_value, is_long=True)
        cash -= daily_cost.total_carrying

        # On futures roll dates:
        roll_cost = cost_model.futures_roll_cost(symbol, position_value)
        cash -= roll_cost
    """

    def __init__(self, config: dict):
        cost_cfg = config.get("costs", {})

        # Allow config overrides
        self._commission_override = cost_cfg.get("commission_pct")
        self._spread_override = cost_cfg.get("half_spread_pct")
        self._impact_scale = cost_cfg.get("impact_scale", 1.0)
        self._financing_rate_override = cost_cfg.get("overnight_financing_rate")
        self._short_borrow_override = cost_cfg.get("short_borrow_rate")

        # Tax on realised gains (Greece: 15% on stock gains; 0% on ETFs held >1yr — simplified)
        self._tax_rate = cost_cfg.get("capital_gains_tax_rate", 0.0)  # opt-in only

        # Cumulative cost tracking
        self.total_commission: float = 0.0
        self.total_spread: float = 0.0
        self.total_impact: float = 0.0
        self.total_overnight: float = 0.0
        self.total_roll: float = 0.0
        self.total_funding: float = 0.0

    # ------------------------------------------------------------------
    # Transaction cost (paid once per trade)
    # ------------------------------------------------------------------

    def transaction_cost(
        self,
        symbol: str,
        notional: float,  # abs(quantity * price)
        is_buy: bool = True,
        adv_override: float | None = None,
    ) -> CostBreakdown:
        """
        Returns the full one-way transaction cost breakdown.
        Call this for BOTH entry and exit — each leg pays half-spread + commission + impact.
        """
        p = COST_PARAMS[_classify(symbol)]

        commission = notional * (self._commission_override or p.commission_pct)

        half_spread = notional * (self._spread_override or p.half_spread_pct)

        # Square-root market impact model: impact = coeff * sqrt(order / ADV)
        adv = adv_override or p.adv_estimate
        if adv > 0:
            impact = notional * p.impact_coeff * self._impact_scale * np.sqrt(notional / adv)
        else:
            impact = 0.0

        # Impact cap: never exceed 0.5% of notional (prevents extreme distortion on small ADV)
        impact = min(impact, notional * 0.005)

        total_tx = commission + half_spread + impact

        breakdown = CostBreakdown(
            commission=commission,
            half_spread=half_spread,
            market_impact=impact,
            total_transaction=total_tx,
        )

        # Accumulate
        self.total_commission += commission
        self.total_spread += half_spread
        self.total_impact += impact

        log.debug(
            f"[{symbol}] TX cost: notional=${notional:,.0f} | "
            f"comm=${commission:.2f} spread=${half_spread:.2f} impact=${impact:.2f} | "
            f"total=${total_tx:.2f} ({total_tx / notional * 100:.4f}%)"
        )
        return breakdown

    # ------------------------------------------------------------------
    # Daily carrying costs (charged each calendar / trading day)
    # ------------------------------------------------------------------

    def daily_carrying_cost(
        self,
        symbol: str,
        position_value: float,  # abs market value of position
        is_long: bool = True,
        is_leveraged: bool = False,
    ) -> CostBreakdown:
        """
        Daily cost of holding a position overnight.
        Charged every trading day to the cash balance.
        """
        p = COST_PARAMS[_classify(symbol)]
        asset_class = _classify(symbol)

        overnight = 0.0
        funding = 0.0

        if is_long:
            # Only charge financing if position is leveraged (margin)
            if is_leveraged:
                rate = (self._financing_rate_override or p.long_financing_rate) / 252
                overnight = position_value * rate
        else:
            # Short position: pay stock borrow rate daily
            rate = (self._short_borrow_override or p.short_borrow_rate) / 252
            overnight = position_value * rate

        # Crypto funding rate (longs pay shorts in bull market)
        if asset_class in ("crypto_major", "crypto_minor"):
            funding = position_value * p.funding_rate_daily
            if not is_long:
                funding = -funding  # shorts RECEIVE funding in bull market

        total_carrying = overnight + max(0, funding)  # only debit, never credit to avoid distortion

        breakdown = CostBreakdown(
            overnight_financing=overnight,
            crypto_funding=max(0, funding),
            total_carrying=total_carrying,
        )

        self.total_overnight += overnight
        self.total_funding += max(0, funding)

        return breakdown

    # ------------------------------------------------------------------
    # Futures roll cost (charged on roll dates ~4x/year)
    # ------------------------------------------------------------------

    def futures_roll_cost(self, symbol: str, position_value: float) -> float:
        """
        Cost of rolling a futures contract.
        Returns the $ cost to deduct from cash.
        """
        p = COST_PARAMS[_classify(symbol)]
        if p.roll_cost_pct == 0:
            return 0.0
        cost = abs(position_value) * p.roll_cost_pct
        self.total_roll += cost
        log.debug(f"[{symbol}] Futures roll cost: ${cost:.2f}")
        return cost

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, float]:
        total = (
            self.total_commission
            + self.total_spread
            + self.total_impact
            + self.total_overnight
            + self.total_roll
            + self.total_funding
        )
        return {
            "cost_commission_total": self.total_commission,
            "cost_spread_total": self.total_spread,
            "cost_market_impact_total": self.total_impact,
            "cost_overnight_financing_total": self.total_overnight,
            "cost_futures_roll_total": self.total_roll,
            "cost_crypto_funding_total": self.total_funding,
            "cost_total": total,
        }

    def reset(self) -> None:
        self.total_commission = 0.0
        self.total_spread = 0.0
        self.total_impact = 0.0
        self.total_overnight = 0.0
        self.total_roll = 0.0
        self.total_funding = 0.0
