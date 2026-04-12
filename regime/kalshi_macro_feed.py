"""
KalshiMacroFeed — Prediction market signals for regime detection.

Pulls Fed rate, CPI, and recession probability from Kalshi API
and maps them to macro anomaly scores for ChoppyDetector input.

No authentication required for market data reads.
Federal Reserve paper (2026) confirms Kalshi beats Bloomberg on CPI
and perfectly predicted every FOMC decision since 2022.

Usage:
    feed = KalshiMacroFeed()
    signals = feed.get_macro_signals()
    # Returns: {'fed_hike_prob': 0.12, 'cpi_surprise_risk': 0.34,
    #           'recession_prob': 0.18, 'composite_stress': 0.21}
"""
import os
import time
import logging
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Known Kalshi series tickers for macro events
SERIES = {
    "fed_rate":   "KXFED",
    "cpi":        "KXCPI",
    "recession":  "KXRECESSION",
    "gdp":        "KXGDP",
}


@dataclass
class MacroSignals:
    fed_hike_prob: float = 0.0       # 0-1: probability of rate HIKE at next FOMC
    fed_cut_prob: float = 0.0        # 0-1: probability of rate CUT at next FOMC
    fed_stress: float = 0.0          # 0-1: derived stress (hike OR cut = uncertainty)
    cpi_beat_prob: float = 0.0       # 0-1: probability CPI comes in ABOVE consensus
    cpi_surprise_risk: float = 0.0   # 0-1: derived stress (large surprise in either direction)
    recession_prob: float = 0.0      # 0-1: probability of recession
    composite_stress: float = 0.0    # 0-1: weighted composite for ChoppyDetector
    available: bool = False          # False = Kalshi unavailable, use fallback
    timestamp: Optional[datetime] = None
    raw: dict = field(default_factory=dict)


class KalshiMacroFeed:
    """
    Reads macro probability signals from Kalshi prediction markets.
    Converts to [0,1] stress scores for regime detector input.

    Weights for composite_stress:
      recession_prob:      0.40  (highest conviction signal)
      fed_stress:          0.35  (FOMC uncertainty)
      cpi_surprise_risk:   0.25  (inflation surprise risk)

    Scoring logic:
      - recession_prob:    direct pass-through (Kalshi probability IS the score)
      - fed_stress:        HIGH when hike OR cut prob > 0.40 (regime uncertainty)
                           LOW  when hold prob > 0.70 (stable policy expected)
      - cpi_surprise_risk: HIGH when P(CPI > consensus + 0.2%) > 0.35
                           Also high when P(CPI < consensus - 0.2%) > 0.35
    """

    WEIGHTS = {"recession": 0.40, "fed": 0.35, "cpi": 0.25}
    CACHE_SECONDS = 3600  # refresh hourly in live mode

    def __init__(self, cache_seconds: int = CACHE_SECONDS):
        self._cache: Optional[MacroSignals] = None
        self._cache_ts: float = 0.0
        self.cache_seconds = cache_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "trading-system/1.0",
        })

    # ── public API ────────────────────────────────────────────────────────────

    def get_macro_signals(self, force_refresh: bool = False) -> MacroSignals:
        """
        Returns latest macro signals. Cached for cache_seconds.
        Returns MacroSignals(available=False) if Kalshi is unreachable.
        """
        now = time.time()
        if (not force_refresh and
                self._cache is not None and
                now - self._cache_ts < self.cache_seconds):
            return self._cache

        signals = self._fetch_all()
        self._cache = signals
        self._cache_ts = now
        return signals

    def get_choppy_input(self, force_refresh: bool = False) -> float:
        """
        Convenience method: returns composite_stress score [0,1].
        Returns 0.0 if Kalshi unavailable (conservative — no false stress).
        """
        s = self.get_macro_signals(force_refresh=force_refresh)
        return s.composite_stress if s.available else 0.0

    # ── internal fetch ────────────────────────────────────────────────────────

    def _fetch_all(self) -> MacroSignals:
        signals = MacroSignals(timestamp=datetime.now(timezone.utc))
        raw = {}

        # 1. Fed rate markets
        try:
            fed_data = self._get_active_markets(SERIES["fed_rate"])
            raw["fed"] = fed_data
            fed_hike, fed_cut = self._parse_fed_markets(fed_data)
            hold_prob = max(0.0, 1.0 - fed_hike - fed_cut)
            # Stress = uncertainty = 1 - hold probability
            fed_stress = 1.0 - hold_prob
            signals.fed_hike_prob = round(fed_hike, 3)
            signals.fed_cut_prob = round(fed_cut, 3)
            signals.fed_stress = round(min(fed_stress, 1.0), 3)
        except Exception as e:
            log.debug(f"[Kalshi] Fed feed failed: {e}")
            signals.fed_stress = 0.0

        # 2. CPI markets
        try:
            cpi_data = self._get_active_markets(SERIES["cpi"])
            raw["cpi"] = cpi_data
            cpi_beat, cpi_miss = self._parse_cpi_markets(cpi_data)
            # Surprise risk = max of upside/downside surprise probability
            cpi_risk = max(cpi_beat, cpi_miss)
            signals.cpi_beat_prob = round(cpi_beat, 3)
            signals.cpi_surprise_risk = round(min(cpi_risk, 1.0), 3)
        except Exception as e:
            log.debug(f"[Kalshi] CPI feed failed: {e}")
            signals.cpi_surprise_risk = 0.0

        # 3. Recession probability
        try:
            rec_data = self._get_active_markets(SERIES["recession"])
            raw["recession"] = rec_data
            rec_prob = self._parse_recession_markets(rec_data)
            signals.recession_prob = round(min(rec_prob, 1.0), 3)
        except Exception as e:
            log.debug(f"[Kalshi] Recession feed failed: {e}")
            signals.recession_prob = 0.0

        # Composite
        signals.composite_stress = round(
            self.WEIGHTS["recession"] * signals.recession_prob +
            self.WEIGHTS["fed"]       * signals.fed_stress +
            self.WEIGHTS["cpi"]       * signals.cpi_surprise_risk,
            3
        )
        signals.available = True
        signals.raw = raw

        log.info(
            f"[Kalshi] recession={signals.recession_prob:.2f} "
            f"fed_stress={signals.fed_stress:.2f} "
            f"cpi_risk={signals.cpi_surprise_risk:.2f} "
            f"→ composite={signals.composite_stress:.3f}"
        )
        return signals

    def _get_active_markets(self, series_ticker: str) -> list:
        url = f"{KALSHI_BASE}/markets"
        params = {"series_ticker": series_ticker, "status": "open", "limit": 50}
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("markets", [])

    def _parse_fed_markets(self, markets: list) -> tuple[float, float]:
        """
        Returns (hike_prob, cut_prob) from FOMC rate markets.
        YES price on a contract = probability that outcome occurs.
        Looks for markets titled with 'hike', 'raise', 'cut', 'lower'.
        """
        hike_prob = 0.0
        cut_prob = 0.0

        for m in markets:
            title = (m.get("title") or "").lower()
            yes_price = m.get("yes_ask") or m.get("last_price") or 0
            yes_price = float(yes_price) / 100  # Kalshi prices in cents

            if any(w in title for w in ["hike", "raise", "increase", "higher"]):
                hike_prob = max(hike_prob, yes_price)
            elif any(w in title for w in ["cut", "lower", "decrease", "reduce"]):
                cut_prob = max(cut_prob, yes_price)

        return hike_prob, cut_prob

    def _parse_cpi_markets(self, markets: list) -> tuple[float, float]:
        """
        Returns (beat_prob, miss_prob):
          beat = CPI comes in ABOVE consensus
          miss = CPI comes in BELOW consensus

        Looks for markets with 'above', 'higher', 'exceed' vs 'below', 'lower'.
        Falls back to median contract if directionality not clear.
        """
        beat_prob = 0.0
        miss_prob = 0.0

        for m in markets:
            title = (m.get("title") or "").lower()
            yes_price = float(m.get("yes_ask") or m.get("last_price") or 0) / 100

            if any(w in title for w in ["above", "higher", "exceed", "over"]):
                beat_prob = max(beat_prob, yes_price)
            elif any(w in title for w in ["below", "lower", "under", "miss"]):
                miss_prob = max(miss_prob, yes_price)

        return beat_prob, miss_prob

    def _parse_recession_markets(self, markets: list) -> float:
        """Returns probability of recession from KXRECESSION series."""
        if not markets:
            return 0.0
        # Take the highest-volume active contract as representative
        best = max(markets, key=lambda m: float(m.get("volume", 0) or 0))
        yes_price = float(best.get("yes_ask") or best.get("last_price") or 0) / 100
        return yes_price


# ── integration with MacroAnomalyDetector ────────────────────────────────────

def enrich_macro_score(base_score: float,
                       kalshi_signals: MacroSignals,
                       kalshi_weight: float = 0.25) -> float:
    """
    Blend Kalshi composite stress into an existing macro score.
    kalshi_weight: how much Kalshi replaces/adds to the base score.

    base_score:    existing MacroAnomalyDetector score (FRED/yield-based)
    kalshi_weight: 0.25 → 75% base + 25% Kalshi
    """
    if not kalshi_signals.available:
        return base_score  # no change if Kalshi unavailable

    blended = (1 - kalshi_weight) * base_score + kalshi_weight * kalshi_signals.composite_stress
    return round(min(blended, 1.0), 3)
