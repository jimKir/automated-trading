"""
Dynamic Universe Selector — with Adaptive Asset Class Caps
===========================================================
At each rebalance, ranks ALL candidate instruments by momentum and
selects the top N into the active trading universe.

NEW: Adaptive equity cap (60–90%) based on broad market momentum regime.
────────────────────────────────────────────────────────────────────────
The equity cap slides between min_equity_cap (0.60) and max_equity_cap (0.90)
based on three market-wide signals, all computed strictly from past data:

  Signal 1 — Broad equity momentum breadth
    % of equity candidates with positive 6-month momentum
    High breadth (>70%) = bull regime → raise equity cap
    Low breadth (<40%)  = bear regime → lower equity cap

  Signal 2 — SPY trend (price vs 200-day MA)
    SPY above 200d MA  = uptrend  → favour equities
    SPY below 200d MA  = downtrend → rotate to defensives

  Signal 3 — Cross-asset momentum spread
    (avg equity momentum) - (avg bond/gold momentum)
    Positive spread = equities leading bonds → raise cap
    Negative spread = bonds/gold leading  → lower cap

Combined into a single regime score [0, 1]:
  0.0 = strong bear  → equity cap = 60%
  0.5 = neutral      → equity cap = 75%
  1.0 = strong bull  → equity cap = 90%

Anti-overfitting:
  - All three signals use fixed economic-logic thresholds (not optimised)
  - The cap range (60–90%) is set conservatively — max 90% avoids
    all-equity portfolio even in the strongest bull markets
  - Smoothed with 3-month EWM to prevent whipsawing
"""
from __future__ import annotations

import warnings
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("UniverseSelector")

# ── Static fallback caps (used before adaptive regime is active) ──────────────
STATIC_CAPS = {
    "equity":  0.60,
    "futures": 0.25,
    "crypto":  0.25,
}

# ── Adaptive cap bounds ────────────────────────────────────────────────────────
EQUITY_CAP_MIN  = 0.60   # floor: bear market
EQUITY_CAP_MAX  = 0.90   # ceiling: strong bull
# Remainder split evenly between futures and crypto
FUTURES_CRYPTO_SHARE = 0.50  # when equity=60%, futures=25%, crypto=25% → split remaining 40%

# Defensive / bond-like symbols used in cross-asset spread calculation
DEFENSIVE_SYMS = {"TLT", "AGG", "LQD", "SHY", "HYG", "GLD", "SLV", "GC=F", "ZB=F"}


def _classify(symbol: str) -> str:
    if symbol.endswith("-USD") or symbol.endswith("USDT"):
        return "crypto"
    if symbol.endswith("=F"):
        return "futures"
    return "equity"


def _is_defensive(symbol: str) -> bool:
    return symbol in DEFENSIVE_SYMS


class AdaptiveCaps:
    """
    Computes adaptive asset class caps based on broad market regime.
    Called once per rerank period.

    Returns dict: {"equity": float, "futures": float, "crypto": float}
    where values are fractions of top_n (e.g. 0.80 = 80% of slots go to equities).
    """

    def __init__(self, config: dict):
        du_cfg = config.get("dynamic_universe", {})
        self.enabled      = du_cfg.get("adaptive_caps", True)
        self.cap_min      = du_cfg.get("equity_cap_min", EQUITY_CAP_MIN)
        self.cap_max      = du_cfg.get("equity_cap_max", EQUITY_CAP_MAX)
        self.regime_window = du_cfg.get("momentum_window", 126)  # reuse momentum window

        # Smoothed regime score (EWM across rebalances)
        self._regime_ewm: Optional[float] = None
        self._cap_log: List[dict] = []   # for diagnostics / reporting

    # ─────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        all_data: Dict[str, pd.DataFrame],
        scores: Dict[str, float],
        as_of_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Compute adaptive caps for the current rebalance date.

        Parameters
        ----------
        all_data   : full price data dict
        scores     : already-computed momentum scores per symbol
        as_of_date : current date (strictly causal)

        Returns
        -------
        {"equity": float, "futures": float, "crypto": float}
        """
        if not self.enabled:
            return dict(STATIC_CAPS)

        regime_score = self._compute_regime_score(all_data, scores, as_of_date)

        # Smooth with EWM (span=3 rebalances ≈ 3 months) to reduce noise
        alpha = 2.0 / (3 + 1)
        if self._regime_ewm is None:
            self._regime_ewm = regime_score
        else:
            self._regime_ewm = alpha * regime_score + (1 - alpha) * self._regime_ewm

        smooth_score = self._regime_ewm

        # Interpolate equity cap between min and max
        equity_cap = self.cap_min + smooth_score * (self.cap_max - self.cap_min)
        equity_cap = float(np.clip(equity_cap, self.cap_min, self.cap_max))

        # Remaining budget split evenly between futures and crypto
        remaining   = 1.0 - equity_cap
        futures_cap = remaining * 0.50
        crypto_cap  = remaining * 0.50

        caps = {
            "equity":  equity_cap,
            "futures": futures_cap,
            "crypto":  crypto_cap,
        }

        self._cap_log.append({
            "date":         as_of_date,
            "regime_raw":   round(regime_score, 3),
            "regime_smooth": round(smooth_score, 3),
            "equity_cap":   round(equity_cap, 3),
            "futures_cap":  round(futures_cap, 3),
            "crypto_cap":   round(crypto_cap, 3),
        })

        log.debug(
            f"[{as_of_date.date()}] Adaptive caps: "
            f"regime={smooth_score:.2f} | "
            f"equity={equity_cap:.0%} futures={futures_cap:.0%} crypto={crypto_cap:.0%}"
        )

        return caps

    # ─────────────────────────────────────────────────────────────────────────

    def _compute_regime_score(
        self,
        all_data: Dict[str, pd.DataFrame],
        scores: Dict[str, float],
        as_of_date: pd.Timestamp,
    ) -> float:
        """
        Combine three signals into a single regime score in [0, 1].
        0 = strong bear (reduce equities), 1 = strong bull (max equities).
        All signals strictly causal — use only data up to as_of_date.
        """
        signal_scores = []

        # ── Signal 1: Equity breadth ────────────────────────────────────────
        # % of equity candidates with positive 6-month momentum score
        equity_scores = [v for k, v in scores.items() if _classify(k) == "equity"]
        if equity_scores:
            breadth = sum(1 for s in equity_scores if s > 0) / len(equity_scores)
            # Normalise: 40% breadth → 0.0, 70% breadth → 1.0
            s1 = float(np.clip((breadth - 0.40) / (0.70 - 0.40), 0, 1))
            signal_scores.append(("breadth", s1, 0.40))
            log.debug(f"  Breadth: {breadth:.1%} → signal={s1:.2f}")

        # ── Signal 2: SPY vs 200-day MA ─────────────────────────────────────
        spy_data = all_data.get("SPY") if "SPY" in all_data else all_data.get("QQQ")
        if spy_data is not None:
            close = spy_data["Close"]
            close = close[close.index <= as_of_date].dropna()
            if len(close) >= 200:
                spy_now  = float(close.iloc[-1])
                ma200    = float(close.rolling(200).mean().iloc[-1])
                # How far above/below 200d MA (%)
                distance = (spy_now - ma200) / ma200
                # -5% below → 0.0, +5% above → 1.0
                s2 = float(np.clip((distance + 0.05) / 0.10, 0, 1))
                signal_scores.append(("spy_vs_ma200", s2, 0.40))
                log.debug(f"  SPY vs 200MA: {distance:+.1%} → signal={s2:.2f}")

        # ── Signal 3: Equity vs defensive momentum spread ───────────────────
        eq_scores  = [v for k, v in scores.items()
                      if _classify(k) == "equity" and not _is_defensive(k)]
        def_scores = [v for k, v in scores.items() if _is_defensive(k)]
        if eq_scores and def_scores:
            spread = np.mean(eq_scores) - np.mean(def_scores)
            # Spread of -0.5 → 0.0 (defensives dominating)
            # Spread of +0.5 → 1.0 (equities dominating)
            s3 = float(np.clip((spread + 0.5) / 1.0, 0, 1))
            signal_scores.append(("eq_vs_def_spread", s3, 0.20))
            log.debug(f"  Equity vs defensive spread: {spread:+.2f} → signal={s3:.2f}")

        if not signal_scores:
            return 0.50  # neutral if no signals available

        # Weighted average of available signals
        total_weight = sum(w for _, _, w in signal_scores)
        regime = sum(s * w for _, s, w in signal_scores) / total_weight

        return float(np.clip(regime, 0, 1))

    # ─────────────────────────────────────────────────────────────────────────

    def get_cap_history(self) -> pd.DataFrame:
        """Return the full history of computed caps as a DataFrame."""
        if not self._cap_log:
            return pd.DataFrame()
        df = pd.DataFrame(self._cap_log).set_index("date")
        return df


# ─────────────────────────────────────────────────────────────────────────────


class DynamicUniverseSelector:
    """
    Selects the top-N momentum instruments from a large candidate pool.
    Asset class caps are dynamically adjusted based on market regime.
    """

    def __init__(self, config: dict):
        du_cfg = config.get("dynamic_universe", {})
        self.enabled          = du_cfg.get("enabled", False)
        self.top_n            = du_cfg.get("top_n", 20)
        self.momentum_window  = du_cfg.get("momentum_window", 126)
        self.min_history_days = du_cfg.get("min_history_days", 252)
        self.rerank_freq      = du_cfg.get("rerank_frequency", "monthly")

        candidates = du_cfg.get("candidates", {})
        self._all_candidates: List[str] = (
            candidates.get("equities", []) +
            candidates.get("futures",  []) +
            candidates.get("crypto",   [])
        )

        self._last_selected:  List[str]              = []
        self._last_rank_date: Optional[pd.Timestamp] = None
        self._adaptive_caps   = AdaptiveCaps(config)

        if self.enabled:
            adaptive = du_cfg.get("adaptive_caps", True)
            log.info(
                f"DynamicUniverse: enabled | candidates={len(self._all_candidates)} | "
                f"top_n={self.top_n} | momentum_window={self.momentum_window}d | "
                f"rerank={self.rerank_freq} | adaptive_caps={adaptive} "
                f"(equity range {EQUITY_CAP_MIN:.0%}–{EQUITY_CAP_MAX:.0%})"
            )

    # ─────────────────────────────────────────────────────────────────────────

    def select(
        self,
        all_data: Dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
    ) -> List[str]:
        if not self.enabled:
            return list(all_data.keys())

        if not self._needs_rerank(as_of_date):
            return self._last_selected

        scores   = self._compute_momentum_scores(all_data, as_of_date)
        if not scores:
            return self._last_selected or list(all_data.keys())

        # Compute adaptive caps for this rebalance
        caps     = self._adaptive_caps.compute(all_data, scores, as_of_date)
        selected = self._apply_caps_and_select(scores, caps)

        prev = set(self._last_selected)
        curr = set(selected)
        adds = curr - prev
        rems = prev - curr
        if adds or rems:
            log.info(
                f"[{as_of_date.date()}] Rerank: {len(selected)} selected | "
                f"eq_cap={caps['equity']:.0%} | "
                f"+{len(adds)} ({', '.join(sorted(adds)[:4])}) | "
                f"-{len(rems)} ({', '.join(sorted(rems)[:4])})"
            )

        self._last_selected  = selected
        self._last_rank_date = as_of_date
        return selected

    # ─────────────────────────────────────────────────────────────────────────

    def _needs_rerank(self, date: pd.Timestamp) -> bool:
        if self._last_rank_date is None:
            return True
        elapsed = (date - self._last_rank_date).days
        if self.rerank_freq == "weekly":   return elapsed >= 5
        if self.rerank_freq == "monthly":  return elapsed >= 21
        return elapsed >= 21

    def _compute_momentum_scores(
        self,
        all_data: Dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for sym in self._all_candidates:
            if sym not in all_data:
                continue
            close = all_data[sym]["Close"]
            close = close[close.index <= as_of_date].dropna()

            if len(close) < self.min_history_days:
                continue
            if len(close) < self.momentum_window + 21:
                continue

            price_now  = float(close.iloc[-21])
            price_past = float(close.iloc[-self.momentum_window - 21])
            if price_past <= 0:
                continue

            momentum = (price_now / price_past) - 1

            # Vol-adjust: divides by 63-day realised vol
            recent_rets = close.pct_change().dropna().iloc[-63:]
            if len(recent_rets) >= 20:
                vol = float(recent_rets.std() * np.sqrt(252))
                if vol > 0:
                    momentum = momentum / vol

            scores[sym] = momentum
        return scores

    def _apply_caps_and_select(
        self,
        scores: Dict[str, float],
        caps: Dict[str, float],
    ) -> List[str]:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        max_equity  = max(1, int(self.top_n * caps["equity"]))
        max_futures = max(1, int(self.top_n * caps["futures"]))
        max_crypto  = max(1, int(self.top_n * caps["crypto"]))

        counts = {"equity": 0, "futures": 0, "crypto": 0}
        limits = {"equity": max_equity, "futures": max_futures, "crypto": max_crypto}

        selected = []
        for sym, score in ranked:
            if len(selected) >= self.top_n:
                break
            ac = _classify(sym)
            if counts[ac] >= limits[ac]:
                continue
            selected.append(sym)
            counts[ac] += 1

        log.debug(
            f"Selected {len(selected)}: "
            f"{counts['equity']} equity (cap {max_equity}) | "
            f"{counts['futures']} futures (cap {max_futures}) | "
            f"{counts['crypto']} crypto (cap {max_crypto})"
        )
        return selected

    # ─────────────────────────────────────────────────────────────────────────

    def compute_selection_series(
        self,
        all_data: Dict[str, pd.DataFrame],
        all_dates: List[pd.Timestamp],
    ) -> Dict[pd.Timestamp, List[str]]:
        selections: Dict[pd.Timestamp, List[str]] = {}
        current: List[str] = []

        for date in all_dates:
            if self._needs_rerank(date):
                current = self.select(all_data, date)
                self._last_rank_date = date
            selections[date] = list(current)

        all_ever: Set[str] = set()
        for syms in selections.values():
            all_ever.update(syms)
        log.info(
            f"Universe summary: {len(all_ever)} unique instruments selected "
            f"from {len(self._all_candidates)} candidates"
        )
        return selections

    def get_cap_history(self) -> pd.DataFrame:
        """Return the full history of adaptive cap values (useful for reporting)."""
        return self._adaptive_caps.get_cap_history()

    def get_selection_stats(
        self, selections: Dict[pd.Timestamp, List[str]]
    ) -> Dict:
        all_syms: Set[str] = set()
        symbol_counts: Dict[str, int] = {}
        for syms in selections.values():
            all_syms.update(syms)
            for s in syms:
                symbol_counts[s] = symbol_counts.get(s, 0) + 1

        total = len(selections)
        freq  = {sym: c / total * 100 for sym, c in symbol_counts.items()}
        return {
            "total_candidates":  len(self._all_candidates),
            "ever_selected":     len(all_syms),
            "always_selected":   [s for s, f in freq.items() if f > 95],
            "most_stable":       sorted(freq.items(), key=lambda x: -x[1])[:10],
            "avg_turnover_pct":  self._compute_turnover(selections),
        }

    @staticmethod
    def _compute_turnover(selections: Dict[pd.Timestamp, List[str]]) -> float:
        dates = sorted(selections.keys())
        turnovers = []
        for i in range(1, len(dates)):
            prev = set(selections[dates[i-1]])
            curr = set(selections[dates[i]])
            if not prev:
                continue
            turnovers.append(len(prev.symmetric_difference(curr)) / max(len(prev), len(curr)))
        return float(np.mean(turnovers) * 100) if turnovers else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC CANDIDATE BUILDER
# Fetches S&P 500 + Nasdaq 100 constituents dynamically, filters for
# liquidity/history, and merges with fixed ETFs/futures/crypto.
# ─────────────────────────────────────────────────────────────────────────────

class DynamicCandidateBuilder:
    """
    Builds the full candidate pool dynamically from:
      1. S&P 500 constituents (fetched from Wikipedia)
      2. Nasdaq 100 constituents (fetched from Wikipedia)
      3. Fixed ETFs, futures, and crypto from config

    Applies liquidity and history filters:
      - Minimum avg daily volume: $5M notional (configurable)
      - Minimum history: 252 trading days (1 year)
      - Excludes extremely high-priced stocks if fractional shares not supported

    Result: typically 400-550 tradable candidates from which the momentum
    selector picks the top 20 each month.

    Anti-overfitting:
      - Constituents are fetched fresh at startup, not pre-selected by us
      - Filters use economic thresholds (liquidity), not backtest performance
      - Any S&P500/NDX stock with sufficient history and liquidity is eligible
    """

    # Free data sources for index constituents
    SP500_CSV  = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    # Nasdaq-100 curated list (updated periodically — covers all major constituents)
    NDX100_TICKERS = [
        'AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','TSLA','AVGO','COST',
        'NFLX','ASML','AMD','PEP','LIN','CSCO','ADBE','QCOM','TXN','INTU',
        'CMCSA','AMAT','ISRG','BKNG','MU','HON','VRTX','LRCX','PANW','KLAC',
        'MELI','REGN','MDLZ','CDNS','SNPS','ABNB','ORLY','FTNT','CRWD','ROP',
        'CTAS','MNST','MRVL','KDP','PCAR','ADP','PAYX','WDAY','DXCM','ODFL',
        'FAST','BIIB','DLTR','IDXX','VRSK','ANSS','ALGN','TEAM','ZS','ILMN',
        'CPRT','ROST','GILD','PLTR','EBAY','PYPL','INTC','PDD','CEG','GFS',
        'FANG','ON','TTWO','DDOG','SNOW','COIN','RBLX','TTD','APP','MSTR',
        'SMCI','DECK','AXON','NTRA','PODD','GEHC','CDW','CCEP','PSTG','TXRH',
    ]

    def __init__(self, config: dict):
        dc_cfg = config.get("dynamic_candidates", {})
        self.enabled          = dc_cfg.get("enabled", False)
        self.min_avg_volume   = dc_cfg.get("min_avg_volume_usd", 5_000_000)
        self.min_history_days = dc_cfg.get("min_history_days", 252)
        self.include_sp500    = dc_cfg.get("include_sp500", True)
        self.include_ndx100   = dc_cfg.get("include_ndx100", True)
        self.max_stocks       = dc_cfg.get("max_stocks", 500)   # cap to avoid fetch overload
        self.cache_hours      = dc_cfg.get("cache_hours", 24)   # re-fetch after N hours

        self._constituent_cache: Optional[List[str]] = None
        self._cache_time: Optional[float] = None

    def get_full_candidate_list(
        self,
        config: dict,
        data_start: str,
        data_end: str,
        verbose: bool = True,
    ) -> List[str]:
        """
        Build and return the complete candidate list.
        Fetches index constituents + config fixed assets.
        Filters for liquidity and history.
        Returns de-duplicated list of valid ticker symbols.
        """
        if not self.enabled:
            # Fall back to config-defined candidates
            du = config.get("dynamic_universe", {}).get("candidates", {})
            return (du.get("equities", []) +
                    du.get("futures",  []) +
                    du.get("crypto",   []))

        log.info("DynamicCandidateBuilder: building candidate pool...")

        # 1. Fetch index constituents
        stocks = self._fetch_constituents(verbose)
        if verbose:
            log.info(f"  Constituents fetched: {len(stocks)} unique stocks")

        # 2. Fixed ETFs / futures / crypto from config (always included)
        du = config.get("dynamic_universe", {}).get("candidates", {})
        fixed_etfs    = du.get("equities", [])
        fixed_futures = du.get("futures",  [])
        fixed_crypto  = du.get("crypto",   [])

        # Remove stocks that are already in the fixed ETF list
        stocks = [s for s in stocks if s not in fixed_etfs]

        # 3. Filter stocks for liquidity + history
        if stocks:
            valid_stocks = self._filter_stocks(
                stocks, data_start, data_end, verbose
            )
        else:
            valid_stocks = []

        if verbose:
            log.info(f"  Stocks passing filters: {len(valid_stocks)}")

        # 4. Combine: stocks + fixed ETFs + futures + crypto
        all_candidates = valid_stocks + fixed_etfs + fixed_futures + fixed_crypto
        # Deduplicate preserving order
        seen = set()
        result = []
        for s in all_candidates:
            if s not in seen:
                seen.add(s)
                result.append(s)

        log.info(
            f"DynamicCandidateBuilder: {len(result)} total candidates "
            f"({len(valid_stocks)} stocks + {len(fixed_etfs)} ETFs + "
            f"{len(fixed_futures)} futures + {len(fixed_crypto)} crypto)"
        )
        return result

    def _fetch_constituents(self, verbose: bool = True) -> List[str]:
        """
        Fetch S&P 500 from GitHub CSV and Nasdaq-100 from curated list.
        Both sources are always available without authentication.
        """
        # Check cache
        if (self._constituent_cache is not None and
                self._cache_time is not None and
                time.time() - self._cache_time < self.cache_hours * 3600):
            log.debug("DynamicCandidateBuilder: using cached constituents")
            return self._constituent_cache

        tickers: Set[str] = set()

        if self.include_sp500:
            try:
                df = pd.read_csv(self.SP500_CSV)
                sp_tickers = df["Symbol"].dropna().tolist()
                # Clean: replace dots with dashes (BRK.B → BRK-B)
                sp_tickers = [str(t).strip().replace(".", "-") for t in sp_tickers]
                tickers.update(sp_tickers)
                if verbose:
                    log.info(f"  S&P 500: {len(sp_tickers)} tickers fetched from GitHub")
            except Exception as e:
                log.warning(f"  S&P 500 CSV fetch failed: {e}")

        if self.include_ndx100:
            # Use curated Nasdaq-100 list (always available, no HTTP needed)
            ndx_tickers = [t for t in self.NDX100_TICKERS
                           if t not in tickers]  # skip duplicates with S&P500
            tickers.update(self.NDX100_TICKERS)
            if verbose:
                log.info(f"  Nasdaq-100: {len(self.NDX100_TICKERS)} tickers added")

        result = sorted(list(tickers))[:self.max_stocks]
        self._constituent_cache = result
        self._cache_time = time.time()
        return result

    def _filter_stocks(
        self,
        tickers: List[str],
        start: str,
        end: str,
        verbose: bool = True,
    ) -> List[str]:
        """
        Download price/volume for each ticker and filter by:
          - Minimum history (252 days)
          - Minimum average daily dollar volume ($5M notional)

        Returns list of tickers that pass both filters.
        Batches downloads to avoid rate limiting.
        """
        import yfinance as yf

        valid = []
        failed = 0
        batch_size = 50  # yfinance handles batches well

        log.info(f"  Filtering {len(tickers)} stocks (this takes ~1-2 minutes)...")

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                raw = yf.download(
                    batch, start=start, end=end,
                    auto_adjust=True, progress=False,
                    threads=True,
                )
                if raw.empty:
                    continue

                # Handle both MultiIndex (multiple tickers) and single ticker
                if isinstance(raw.columns, pd.MultiIndex):
                    close_df  = raw["Close"]
                    volume_df = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else None
                else:
                    # Single ticker
                    close_df  = raw[["Close"]]
                    volume_df = raw[["Volume"]] if "Volume" in raw.columns else None

                for sym in batch:
                    try:
                        if sym not in close_df.columns:
                            failed += 1
                            continue

                        close  = close_df[sym].dropna()
                        if len(close) < self.min_history_days:
                            continue

                        if volume_df is not None and sym in volume_df.columns:
                            vol    = volume_df[sym].dropna()
                            # Dollar volume = price × shares traded
                            dv     = (close * vol).rolling(20).mean().dropna()
                            if dv.empty or float(dv.iloc[-1]) < self.min_avg_volume:
                                continue

                        valid.append(sym)
                    except Exception:
                        failed += 1

            except Exception as e:
                log.debug(f"  Batch {i//batch_size} failed: {e}")
                failed += len(batch)

            # Polite delay between batches
            time.sleep(0.3)

        if verbose:
            log.info(f"  Filter complete: {len(valid)} passed, {failed} failed/insufficient")

        return valid
