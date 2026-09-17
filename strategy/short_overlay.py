"""
Bear-Regime Short Overlay (paper trading)
=========================================
Gap B: previously a bear regime could only express negative information as
"don't hold" (go to cash). This overlay converts negative ranked momentum on
liquid, easy-to-borrow US equity ETFs into capped SHORT targets so the
portfolio earns bear-market alpha instead of only defending.

Design invariants
-----------------
* equity/ETF shorts only — NO crypto shorts, NO futures (Alpaca constraint)
* only a fixed liquid-ETF universe (deep borrow, tight spreads)
* only when the blended momentum signal is negative (< min_signal_strength)
  AND (bear regime OR the name trades below its own 200d MA)
* per-name cap ``max_single_short_pct``, aggregate cap
  ``max_short_notional_pct`` of equity
* shorts count toward portfolio heat: |longs| + |shorts| <= max_portfolio_heat
* cover on regime improvement (bear -> yellow+) at the next rebalance
* hard stop: cover any short that moves +hard_stop_pct against us
  (intraday loop in live, daily High in backtest)

Weights are returned as NEGATIVE portfolio weights (fraction of equity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from utils.logger import get_logger

log = get_logger("ShortOverlay")

# Liquid, easy-to-borrow US equity ETFs (config key: universe: liquid_etfs_only)
LIQUID_ETF_UNIVERSE: list[str] = [
    "SPY", "QQQ", "IWM", "DIA", "MDY",       # broad equity
    "EEM", "VGK", "EWJ",                     # international
    "XLE", "XLF", "XLV", "XLU", "XLP", "XLY", "XLK",  # sectors
    "VNQ",                                   # real estate
]

_UNIVERSES = {"liquid_etfs_only": LIQUID_ETF_UNIVERSE}

MA_WINDOW = 200


@dataclass
class ShortingConfig:
    """Parsed ``shorting:`` config block."""

    enabled: bool = False
    asset_classes: list[str] = field(default_factory=lambda: ["equity_etf"])
    max_short_notional_pct: float = 0.30
    max_single_short_pct: float = 0.08
    min_signal_strength: float = 0.0
    universe: str = "liquid_etfs_only"
    cover_on_regime_improvement: bool = True
    hard_stop_pct: float = 0.08

    @classmethod
    def from_config(cls, config: dict) -> "ShortingConfig":
        raw = config.get("shorting", {}) if isinstance(config, dict) else {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            asset_classes=list(raw.get("asset_classes", ["equity_etf"])),
            max_short_notional_pct=float(raw.get("max_short_notional_pct", 0.30)),
            max_single_short_pct=float(raw.get("max_single_short_pct", 0.08)),
            min_signal_strength=float(raw.get("min_signal_strength", 0.0)),
            universe=str(raw.get("universe", "liquid_etfs_only")),
            cover_on_regime_improvement=bool(raw.get("cover_on_regime_improvement", True)),
            hard_stop_pct=float(raw.get("hard_stop_pct", 0.08)),
        )

    @property
    def universe_symbols(self) -> list[str]:
        return list(_UNIVERSES.get(self.universe, LIQUID_ETF_UNIVERSE))


def is_bear_regime(spy_close: pd.Series | None) -> bool:
    """Bear regime = SPY below its 200d moving average.

    Returns False when there is insufficient history (fail-safe: no shorts).
    """
    if spy_close is None or len(spy_close) < MA_WINDOW:
        return False
    ma = float(spy_close.iloc[-MA_WINDOW:].mean())
    return float(spy_close.iloc[-1]) < ma


def below_ma(close: pd.Series, window: int = MA_WINDOW) -> bool:
    """True if the last close is below the trailing `window`-day MA."""
    if close is None or len(close) < window:
        return False
    return float(close.iloc[-1]) < float(close.iloc[-window:].mean())


def compute_short_targets(
    signals: dict[str, float],
    price_history: dict[str, pd.DataFrame],
    *,
    cfg: ShortingConfig,
    bear_regime: bool,
    long_gross: float,
    max_portfolio_heat: float,
    guard=None,
) -> dict[str, float]:
    """Compute capped short targets (negative weights, fractions of equity).

    Parameters
    ----------
    signals : blended ranked momentum scores per symbol.
    price_history : {symbol: DataFrame with Close} used for the MA200 check.
    cfg : ShortingConfig.
    bear_regime : SPY below 200d MA (relaxes the per-name MA200 requirement).
    long_gross : sum of positive target weights already allocated.
    max_portfolio_heat : total gross exposure cap (|long| + |short|).
    guard : optional TradeableUniverse for runtime shortable/ETB checks.

    Returns
    -------
    dict {symbol: negative_weight}. Empty when shorting is disabled or no
    candidate qualifies.
    """
    if not cfg.enabled:
        return {}
    if "equity_etf" not in cfg.asset_classes:
        # Only equity/ETF shorts exist — crypto/futures shorts are unsupported
        return {}

    budget = min(cfg.max_short_notional_pct, max(0.0, max_portfolio_heat - long_gross))
    if budget <= 0:
        return {}

    eligible: dict[str, float] = {}
    for sym in cfg.universe_symbols:
        sig = signals.get(sym)
        if sig is None:
            continue
        if sig >= cfg.min_signal_strength:
            continue  # only short negative-ranked momentum
        if not bear_regime:
            df = price_history.get(sym)
            close = df["Close"] if df is not None and "Close" in df.columns else None
            if close is None or not below_ma(close):
                continue
        if guard is not None and not guard.is_shortable(sym):
            log.info(f"[SHORT] Skipping {sym} — not shortable/easy-to-borrow on Alpaca")
            continue
        eligible[sym] = abs(sig)

    if not eligible:
        return {}

    total_mag = sum(eligible.values())
    targets: dict[str, float] = {}
    for sym, mag in eligible.items():
        w = budget * (mag / total_mag)
        w = min(w, cfg.max_single_short_pct)
        if w > 0:
            targets[sym] = -w
    return targets


def merge_short_targets(
    target_weights: dict[str, float],
    short_targets: dict[str, float],
) -> dict[str, float]:
    """Overlay short targets on top of the long-only target weights.

    A short target replaces any long target for the same symbol (the overlay
    only fires on names whose signal is negative, so a long target for the
    same symbol should not exist — but never stack both).
    """
    out = dict(target_weights)
    for sym, w in short_targets.items():
        out[sym] = w
    return out


def hard_stop_covers(
    positions: dict[str, dict],
    prices: dict[str, float],
    hard_stop_pct: float,
) -> list[str]:
    """Shorts that have moved up >= hard_stop_pct against entry.

    ``positions`` maps symbol → dict with ``quantity`` and ``avg_price``
    (``avg_entry_price`` also accepted; Alpaca position dicts).
    Returns symbols that must be covered immediately.
    """
    covers: list[str] = []
    for sym, pos in positions.items():
        qty = float(pos.get("quantity", 0) or 0)
        if qty >= 0:
            continue  # longs or flat — not our concern here
        entry = float(
            pos.get("avg_price", pos.get("avg_entry_price", 0)) or 0
        )
        price = prices.get(sym)
        if entry <= 0 or price is None or price <= 0:
            continue
        adverse_move = (price - entry) / entry
        if adverse_move >= hard_stop_pct:
            log.warning(
                f"[SHORT-STOP] {sym}: short from ${entry:.2f} now ${price:.2f} "
                f"(+{adverse_move:.1%} adverse >= {hard_stop_pct:.0%} hard stop) — covering"
            )
            covers.append(sym)
    return covers
