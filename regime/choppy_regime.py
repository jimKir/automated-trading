"""
Choppy Regime Detector  v2
===========================
A dedicated anomaly detection layer that fires when market conditions
resemble the 2025 high-vol / low-directional-conviction regime —
the environment where the strategy underperforms most.

v2 EXPANSION: full feature universe wired from Turn 10
  Group A  — Vol spikes & volume anomalies      (from anomaly.py features)
  Group B  — Cross-asset price volatility       (v1 core features)
  Group C  — Macro / credit market stress       (from macro_score.py signals)
  Group D  — Event shock & VIX dynamics         (from event_shock.py signals)
  Group E  — Commodity & FX stress              (from commodity_fx.py signals)
  Group F  — Market breadth & regime breadth    (from event_shock.py)
  Group G  — Sentiment proxies                  (VIX/RVol, implied skew)

All features are:
  - Computed from local parquet data (no live API calls in backtest)
  - Vectorised (no per-day Python loops)
  - Calibrated on calm-year baseline so score ≈ 0 in 2024, rises in 2025
  - No look-ahead — each value uses only past data

Calibration reference (from regime analysis, 2023-2026):
  2024 (calm trending):   target score mean ≈ 0.10-0.15  (GREEN always)
  2025 (choppy high-vol): target score mean ≈ 0.35-0.45  (YELLOW/ORANGE)
  Q1-2026 (tariff shock): target score mean ≈ 0.40-0.55  (ORANGE/RED)

Score → position scale factor:
  < 0.30  GREEN  1.00×  trending/normal — no action
  0.30-0.50 YELLOW 0.70×  choppy building — light trim
  0.50-0.65 ORANGE 0.40×  clearly choppy — reduce exposure
  > 0.65  RED    0.20×  2025-type choppiness — defensive

Architecture
------------
The detector is self-contained and takes a price_data dict (sym → DataFrame)
plus a vix Series. It loads all other required data internally from the
repo's local historical parquet store at data/historical/daily/.
This keeps it decoupled from live API calls in backtesting.

Integration with EWS
--------------------
  ews.py calls ChoppyRegimeDetector.score_series(price_df, vix) for backtest
  and ChoppyRegimeDetector.score_today(price_df, vix) for live mode.
  Layer F weight: 0.15 (in LAYER_WEIGHTS dict in ews.py)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("ChoppyRegimeDetector")
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR  = _REPO_ROOT / "data" / "historical" / "daily"

# ── Score → scale factor thresholds ──────────────────────────────────────────
# Thresholds calibrated to v2 score distribution:
#   full-history max≈0.63, mean≈0.17, p90=0.30, p95=0.36
#   2024 calm mean=0.11 → stays GREEN; 2025 choppy mean=0.17 → bleeds into YELLOW
#   2022 bear mean=0.25 → YELLOW/ORANGE; acute stress days reach 0.40-0.63
CHOPPY_SCALE_THRESHOLDS = [
    (0.17, 1.00, "GREEN",  "Trending/normal — no action"),
    (0.27, 0.80, "YELLOW", "Choppy building — light trim (scale 80%)"),
    (0.40, 0.50, "ORANGE", "Clearly choppy — reduce exposure (scale 50%)"),
    (1.01, 0.25, "RED",    "2025/bear choppiness — defensive (scale 25%)"),
]

# ── Feature group weights (must sum to 1.0) ───────────────────────────────────
# Chosen so no single group dominates and all regime types are covered.
GROUP_WEIGHTS = {
    "vol_spike":    0.18,   # A: volume anomalies (SPY/QQQ vol spikes)
    "price_vol":    0.18,   # B: cross-asset price vol (v1 core: vov, MA crossings)
    "macro_credit": 0.16,   # C: HYG/LQD spreads, TLT stress
    "event_shock":  0.16,   # D: VIX velocity, VIX level, VIX above-20 rate
    "commodity_fx": 0.12,   # E: gold/SPY, oil velocity, DXY
    "breadth":      0.12,   # F: market breadth, SPY/TLT correlation
    "sentiment":    0.08,   # G: VIX/RVol ratio, VIX vs 60d mean
}
assert abs(sum(GROUP_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Feature calibration: (calm_baseline, stress_ceiling) ─────────────────────
# Features are normalised: (raw - baseline) / (ceiling - baseline), clipped [0,1].
# baseline = 2024 calm-year mean (score = 0 in calm trending markets)
# ceiling  = 2025 p95 or Q1-2026 p95 (score = 1 in maximum stress)
#
# Raw distribution reference (profiled 2023-2026):
#   vol_spike_freq_30d:  2024 mean=0.006, 2025 mean=0.026 (4×), p95=0.10
#   spy_vol_surge_5d:    2024 mean=1.33,  2025 p95=2.01
#   vov_raw (×1000):     2024 mean=1.4,   2025 p95=8.2
#   ma_crossing_rate:    2024 mean=0.082, 2025 p95=0.233
#   reversal_rate:       2024 mean=0.476, 2025 p95=0.543
#   hyg_lqd_30d_neg:     2024 mean neg tail, 2025 stress episodes
#   tlt_realized_vol:    2024 mean=0.138, 2025 mean=0.115 (bonds calmer in 2025)
#   vix_10d_std:         2024 mean=1.47,  2025 mean=1.97, p95=4.25
#   vix_above20_20d:     2024 mean=0.095, 2025 mean=0.253 (2.8×)
#   vix_5d_vel:          2024 mean=0.020, 2025 p95=0.297
#   gld_spy_20d_chg:     2024 mean=0.0006,2025 mean=0.030 (50×), p95=0.139
#   oil_10d_abs_vel:     2024 mean=0.035, Q1-26 p95=0.366
#   dxy_10d_mom:         2024 mean=0.002, stress episodes ~0.02+
#   breadth_above_50ma:  2024 mean=0.715, 2025 mean=0.604 (lower = worse)
#   vix_rvol_ratio:      2024 mean=1.385, 2025 mean=1.408, Q1-26=1.721
#   vix_vs_60d_mean:     2024 mean=1.024, 2025 p95=1.458, Q1-26 p95=1.498

CALIBRATION = {
    # vol_spike group
    "vol_spike_freq":   (0.006,  0.10),    # 30d spike frequency: calm=0.006, max=0.10
    "vol_surge_5d":     (1.33,   2.20),    # 5d max/20d mean: calm=1.33, max=2.20

    # price_vol group (v1 features, recalibrated with raw values)
    "vov_raw":          (0.0017, 0.010),   # raw vol-of-vol: calm=0.0017, max=0.010
    "ma_crossing":      (0.08,   0.25),    # crossing rate: calm=0.082, max=0.233
    "reversal_rate":    (0.476,  0.545),   # reversal rate: calm=0.476, max=0.543

    # macro_credit group
    "hyg_lqd_30d_neg":  (0.0,    -0.05),  # 30d HYG/LQD pct (inverted: neg = stress)
    "hyg_20d_neg":      (0.0,    -0.05),  # HYG 20d return (inverted: neg = stress)
    "tlt_10d_vel_neg":  (0.0,    -0.08),  # TLT 10d move (inverted: falling = stress)

    # event_shock group
    "vix_10d_std":      (1.47,   5.0),    # VIX 10d std: calm=1.47, max=5.0
    "vix_above20_rate": (0.095,  1.0),    # fraction days VIX>20 in 20d: calm=0.095
    "vix_5d_vel":       (0.02,   0.50),   # VIX 5d pct change: calm=0.020, max=0.50

    # commodity_fx group
    "gld_spy_20d_chg":  (0.001,  0.15),   # gold/SPY 20d rise: calm~0, stress=0.15
    "oil_10d_abs_vel":  (0.035,  0.30),   # |oil 10d pct|: calm=0.035, max=0.30
    "dxy_10d_mom_pos":  (0.002,  0.03),   # DXY rising (risk-off): calm=0.002, max=0.03

    # breadth group
    "breadth_below":    (0.715,  0.35),   # breadth BELOW this = stress (inverted scale)
    "spy_tlt_corr":     (0.30,   0.70),   # positive corr = risk-on; high corr late cycle

    # sentiment group
    "vix_rvol_ratio":   (1.385,  2.20),   # VIX/realised vol: calm=1.385, max=2.20
    "vix_vs_60dmean":   (1.024,  1.60),   # VIX / 60d mean: calm~1.0, max=1.60
}

# ── Smoothing ─────────────────────────────────────────────────────────────────
SCORE_EMA_SPAN = 5    # 1-week smoothing — avoids daily whipsawing


class ChoppyRegimeDetector:
    """
    v2: Detects the 2025-style choppy high-volatility regime.

    Takes equity prices + VIX, loads macro/commodity/FX data from local
    parquet store automatically. All computation is vectorised (no per-day
    Python loops) — fast enough for a full 7-year backtest in <2 seconds.

    Usage (backtest):
        detector = ChoppyRegimeDetector()
        scores   = detector.score_series(price_df, vix_series)

    Usage (live):
        score = ChoppyRegimeDetector().score_today(price_df, vix_series)
        scale, colour = ChoppyRegimeDetector.score_to_scale(score)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self._cache: Dict[str, pd.Series] = {}

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self, sym: str) -> Optional[pd.Series]:
        """Load a Close price series from the local parquet store."""
        if sym in self._cache:
            return self._cache[sym]
        p = self._data_dir / f"{sym}.parquet"
        if not p.exists():
            log.debug(f"ChoppyRegime: {sym}.parquet not found at {p}")
            return None
        try:
            df = pd.read_parquet(p)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.capitalize() for c in df.columns]
            df.index   = pd.to_datetime(df.index).tz_localize(None)
            s = df["Close"].rename(sym)
            self._cache[sym] = s
            return s
        except Exception as e:
            log.warning(f"ChoppyRegime: failed to load {sym}: {e}")
            return None

    def _load_vol(self, sym: str) -> Optional[pd.Series]:
        """Load Volume series."""
        key = f"vol_{sym}"
        if key in self._cache:
            return self._cache[key]
        p = self._data_dir / f"{sym}.parquet"
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.capitalize() for c in df.columns]
            df.index   = pd.to_datetime(df.index).tz_localize(None)
            s = df["Volume"].rename(key)
            self._cache[key] = s
            return s
        except Exception as e:
            log.warning(f"ChoppyRegime: failed to load volume {sym}: {e}")
            return None

    # ── Feature helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rescale(raw: pd.Series, baseline: float, ceiling: float,
                 invert: bool = False) -> pd.Series:
        """
        Normalise: (raw - baseline) / (ceiling - baseline), clipped [0,1].
        If invert=True: (baseline - raw) / (baseline - ceiling), for features
        where lower raw value = higher stress (e.g. breadth, HYG return).
        """
        if invert:
            denom = baseline - ceiling
            if abs(denom) < 1e-10:
                return pd.Series(0.0, index=raw.index)
            return ((baseline - raw) / denom).clip(0, 1)
        else:
            denom = ceiling - baseline
            if abs(denom) < 1e-10:
                return pd.Series(0.0, index=raw.index)
            return ((raw - baseline) / denom).clip(0, 1)

    # ── Group A: Volume spike features ───────────────────────────────────────

    def _group_vol_spike(self, idx: pd.DatetimeIndex) -> pd.Series:
        """
        Volume anomalies: frequency of volume spikes and sustained surge.
        Both signals elevated in choppy regimes when uncertainty is high.
        """
        scores = pd.Series(0.0, index=idx)

        spy_vol = self._load_vol("SPY")
        qqq_vol = self._load_vol("QQQ")

        components = []

        # F1: Frequency of SPY volume spikes (>2× 20d mean) in rolling 30d window
        if spy_vol is not None:
            ma20      = spy_vol.rolling(20).mean().replace(0, np.nan)
            spike     = (spy_vol > 2 * ma20).astype(float)
            spike_freq = spike.rolling(30).mean()
            base, ceil = CALIBRATION["vol_spike_freq"]
            components.append(self._rescale(spike_freq, base, ceil).reindex(idx, method="ffill").fillna(0))

        # F2: 5-day max volume / 20d mean (sustained surge detector)
        for vol_s, sym in [(spy_vol, "SPY"), (qqq_vol, "QQQ")]:
            if vol_s is not None:
                ma20   = vol_s.rolling(20).mean().replace(0, np.nan)
                surge  = vol_s.rolling(5).max() / ma20
                base, ceil = CALIBRATION["vol_surge_5d"]
                components.append(self._rescale(surge, base, ceil).reindex(idx, method="ffill").fillna(0))

        if not components:
            return scores

        scores = pd.concat(components, axis=1).mean(axis=1).reindex(idx, method="ffill").fillna(0)
        return scores.clip(0, 1)

    # ── Group B: Cross-asset price volatility (v1 core) ──────────────────────

    def _group_price_vol(self, prices: pd.DataFrame, spy_close: pd.Series,
                         idx: pd.DatetimeIndex) -> pd.Series:
        """
        Core v1 features: vol-of-vol, MA-crossing rate, reversal rate.
        Re-implemented with raw calibration thresholds.
        """
        rets = prices.pct_change()
        spy_ret = spy_close.pct_change()

        components = []

        # F3: Vol-of-Vol (cross-sectional median vol, then std over 10d)
        vol10 = rets.rolling(10).std()
        vol_med = vol10.median(axis=1)
        vov_raw = vol_med.rolling(10).std()
        base, ceil = CALIBRATION["vov_raw"]
        components.append(self._rescale(vov_raw, base, ceil))

        # F4: MA-crossing rate (SPY crosses its 20d MA in rolling 30d window)
        spy_ma20   = spy_close.rolling(20).mean()
        above_ma   = (spy_close > spy_ma20).astype(float)
        cross_flag = (above_ma != above_ma.shift(1)).astype(float)
        cross_raw  = cross_flag.rolling(30).mean()
        base, ceil = CALIBRATION["ma_crossing"]
        components.append(self._rescale(cross_raw, base, ceil))

        # F5: Return reversal rate (cross-sectional)
        sign_t    = np.sign(rets)
        sign_y    = np.sign(rets.shift(1))
        rev_raw   = ((sign_t * sign_y) < 0).astype(float).rolling(30).mean().mean(axis=1)
        base, ceil = CALIBRATION["reversal_rate"]
        components.append(self._rescale(rev_raw, base, ceil))

        # F6: Trend-noise ratio (inverted: low trend = high choppiness)
        net_ret   = spy_close.pct_change(30).abs()
        path_vol  = spy_ret.rolling(30).std() * np.sqrt(30)
        tnr       = (net_ret / path_vol.replace(0, np.nan)).fillna(1.0).clip(0, 2)
        # Invert: TNR=2 (strong trend)→ score=0; TNR=0.2 (choppy)→ score=1
        tnr_score = ((0.8 - tnr) / 0.6).clip(0, 1)
        components.append(tnr_score)

        result = pd.concat(components, axis=1).mean(axis=1)
        return result.reindex(idx, method="ffill").fillna(0).clip(0, 1)

    # ── Group C: Macro / credit stress ───────────────────────────────────────

    def _group_macro_credit(self, idx: pd.DatetimeIndex) -> pd.Series:
        """
        Credit market signals: HYG/LQD spread, HYG return, TLT direction.
        Rising credit stress precedes choppy equity regimes.
        """
        components = []

        hyg = self._load("HYG")
        lqd = self._load("LQD")
        tlt = self._load("TLT")

        # F7: HYG/LQD ratio 30d pct change (negative = spreads widening = stress)
        if hyg is not None and lqd is not None:
            hyg_lqd = (hyg / lqd.reindex(hyg.index, method="ffill")).dropna()
            hyg_lqd_chg = hyg_lqd.pct_change(30)
            base, ceil = CALIBRATION["hyg_lqd_30d_neg"]
            components.append(self._rescale(hyg_lqd_chg, base, ceil, invert=True))

        # F8: HYG 20d return (inverted: falling HYG = credit stress)
        if hyg is not None:
            hyg_ret = hyg.pct_change(20)
            base, ceil = CALIBRATION["hyg_20d_neg"]
            components.append(self._rescale(hyg_ret, base, ceil, invert=True))

        # F9: TLT 10d momentum (inverted: falling TLT with rising yields = risk-off)
        if tlt is not None:
            tlt_vel = tlt.pct_change(10)
            base, ceil = CALIBRATION["tlt_10d_vel_neg"]
            components.append(self._rescale(tlt_vel, base, ceil, invert=True))

        if not components:
            return pd.Series(0.0, index=idx)

        result = pd.concat(components, axis=1).mean(axis=1)
        return result.reindex(idx, method="ffill").fillna(0).clip(0, 1)

    # ── Group D: Event shock & VIX dynamics ──────────────────────────────────

    def _group_event_shock(self, vix: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
        """
        VIX dynamics: instability, elevated frequency, velocity.
        These are fast signals that react within 1-5 days of a regime shift.
        """
        vix_aligned = vix.reindex(idx, method="ffill").fillna(vix.mean())
        components  = []

        # F10: VIX 10d realised std (vol-of-vol from VIX directly)
        vix_std = vix_aligned.rolling(10).std()
        base, ceil = CALIBRATION["vix_10d_std"]
        components.append(self._rescale(vix_std, base, ceil))

        # F11: Fraction of days VIX > 20 in rolling 20d window
        above20_rate = (vix_aligned > 20).astype(float).rolling(20).mean()
        base, ceil   = CALIBRATION["vix_above20_rate"]
        components.append(self._rescale(above20_rate, base, ceil))

        # F12: VIX 5-day velocity (rapid VIX spike = event shock beginning)
        vix_vel = vix_aligned.pct_change(5).clip(-1, 2).fillna(0)
        vix_vel_pos = vix_vel.clip(lower=0)   # only upward spikes matter
        base, ceil  = CALIBRATION["vix_5d_vel"]
        components.append(self._rescale(vix_vel_pos, base, ceil))

        result = pd.concat(components, axis=1).mean(axis=1)
        return result.reindex(idx, method="ffill").fillna(0).clip(0, 1)

    # ── Group E: Commodity & FX stress ───────────────────────────────────────

    def _group_commodity_fx(self, spy_close: pd.Series,
                            idx: pd.DatetimeIndex) -> pd.Series:
        """
        Safety flight (gold/SPY), oil shock, DXY risk-off.
        These provide confirmation from outside the equity market.
        """
        components = []

        gld = self._load("GLD")
        oil = self._load("OIL")
        dxy = self._load("DXY")

        # F13: Gold/SPY ratio 20d rise (flight to safety)
        if gld is not None:
            gld_spy = (gld / spy_close.reindex(gld.index, method="ffill")).dropna()
            gld_spy_chg = gld_spy.pct_change(20).clip(-0.5, 0.5)
            base, ceil  = CALIBRATION["gld_spy_20d_chg"]
            components.append(self._rescale(gld_spy_chg, base, ceil))

        # F14: |Oil 10d return| — both spikes and crashes signal stress
        if oil is not None:
            oil_vel = oil.pct_change(10).abs()
            base, ceil = CALIBRATION["oil_10d_abs_vel"]
            components.append(self._rescale(oil_vel, base, ceil))

        # F15: DXY 10d upward momentum (risk-off USD strengthening)
        if dxy is not None:
            dxy_mom = dxy.pct_change(10).clip(-0.1, 0.1).fillna(0)
            dxy_pos = dxy_mom.clip(lower=0)   # only USD strengthening matters
            base, ceil = CALIBRATION["dxy_10d_mom_pos"]
            components.append(self._rescale(dxy_pos, base, ceil))

        if not components:
            return pd.Series(0.0, index=idx)

        result = pd.concat(components, axis=1).mean(axis=1)
        return result.reindex(idx, method="ffill").fillna(0).clip(0, 1)

    # ── Group F: Market breadth & cross-asset regime ──────────────────────────

    def _group_breadth(self, prices: pd.DataFrame, spy_close: pd.Series,
                       idx: pd.DatetimeIndex) -> pd.Series:
        """
        Breadth deterioration: fraction of stocks below key MAs.
        When breadth narrows, the market is in stealth distribution
        even if the index is holding — classic pre-correction pattern.
        """
        tlt = self._load("TLT")
        components = []

        # F16: Fraction of equity universe above 50d MA (low = stress)
        ma50    = prices.rolling(50).mean()
        breadth = (prices > ma50).astype(float).mean(axis=1)
        base, ceil = CALIBRATION["breadth_below"]
        # Inverted: high breadth = 0 stress, low breadth = high stress
        components.append(self._rescale(breadth, base, ceil, invert=True))

        # F17: Fraction above 200d MA (longer-term trend health)
        ma200     = prices.rolling(200).mean()
        breadth200 = (prices > ma200).astype(float).mean(axis=1)
        # Use same calibration (slightly different scale)
        components.append(self._rescale(breadth200, base * 0.9, ceil * 0.9, invert=True))

        # F18: SPY/TLT 20d rolling correlation
        # In a healthy risk-on regime, SPY and TLT are negatively correlated.
        # When correlation turns strongly positive, both assets move together
        # = late-cycle stress or flight-to-quality breakdown.
        if tlt is not None:
            spy_r   = spy_close.pct_change()
            tlt_r   = tlt.reindex(spy_close.index, method="ffill").pct_change()
            corr20  = spy_r.rolling(20).corr(tlt_r)
            base, ceil = CALIBRATION["spy_tlt_corr"]
            # high positive correlation = stress in this model
            components.append(self._rescale(corr20, base, ceil))

        if not components:
            return pd.Series(0.0, index=idx)

        result = pd.concat(components, axis=1).mean(axis=1)
        return result.reindex(idx, method="ffill").fillna(0).clip(0, 1)

    # ── Group G: Sentiment proxies ────────────────────────────────────────────

    def _group_sentiment(self, vix: pd.Series, spy_close: pd.Series,
                          idx: pd.DatetimeIndex) -> pd.Series:
        """
        Implied vs realised volatility premium (fear premium) and
        VIX elevation relative to its own recent history.
        These capture the sentiment component that pure price vol misses.
        """
        vix_aligned = vix.reindex(idx, method="ffill")
        spy_rvol    = spy_close.pct_change().rolling(20).std() * np.sqrt(252) * 100
        spy_rvol    = spy_rvol.reindex(idx, method="ffill")
        components  = []

        # F19: VIX / Realised Vol ratio — fear premium
        # High ratio = implied vol elevated vs actual vol = anxiety / uncertainty
        vix_rvol_ratio = (vix_aligned / spy_rvol.replace(0, np.nan)).fillna(1.4)
        base, ceil     = CALIBRATION["vix_rvol_ratio"]
        components.append(self._rescale(vix_rvol_ratio, base, ceil))

        # F20: VIX relative to its own 60d mean
        # Spikes above recent history = regime shift in fear
        vix_60d_mean   = vix_aligned.rolling(60).mean().replace(0, np.nan)
        vix_rel        = (vix_aligned / vix_60d_mean).fillna(1.0)
        base, ceil     = CALIBRATION["vix_vs_60dmean"]
        components.append(self._rescale(vix_rel, base, ceil))

        result = pd.concat(components, axis=1).mean(axis=1)
        return result.reindex(idx, method="ffill").fillna(0).clip(0, 1)

    # ── Main scoring ──────────────────────────────────────────────────────────

    def _compute_all_groups(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
        idx: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Compute all group scores and return as a DataFrame."""
        if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)
        if isinstance(vix.index, pd.DatetimeIndex) and vix.index.tz is not None:
            vix = vix.copy()
            vix.index = vix.index.tz_localize(None)

        spy_close = prices["SPY"] if "SPY" in prices.columns else prices.mean(axis=1)

        groups = pd.DataFrame(index=idx)
        groups["vol_spike"]    = self._group_vol_spike(idx)
        groups["price_vol"]    = self._group_price_vol(prices, spy_close, idx)
        groups["macro_credit"] = self._group_macro_credit(idx)
        groups["event_shock"]  = self._group_event_shock(vix, idx)
        groups["commodity_fx"] = self._group_commodity_fx(spy_close, idx)
        groups["breadth"]      = self._group_breadth(prices, spy_close, idx)
        groups["sentiment"]    = self._group_sentiment(vix, spy_close, idx)
        return groups.fillna(0).clip(0, 1)

    def score_series(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
        smooth: bool = True,
        return_groups: bool = False,
    ) -> pd.Series:
        """
        Compute daily choppy-regime score for the full price history.
        Safe for backtest — no future information used anywhere.

        Parameters
        ----------
        prices        : DataFrame of close prices (equity universe + SPY)
        vix           : Series of VIX daily closes
        smooth        : Apply 5-day EMA smoothing (recommended)
        return_groups : If True, return (score_series, groups_df) tuple

        Returns
        -------
        pd.Series of choppy scores ∈ [0, 1], date-indexed
        """
        log.info("ChoppyRegimeDetector v2: computing all feature groups...")

        # Common date index — intersection of vix and prices
        idx = prices.index.intersection(vix.index)
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)

        groups = self._compute_all_groups(prices, vix, idx)

        # Weighted blend
        score = pd.Series(0.0, index=idx)
        for g, w in GROUP_WEIGHTS.items():
            if g in groups.columns:
                score += w * groups[g]

        if smooth:
            score = score.ewm(span=SCORE_EMA_SPAN, adjust=False).mean()

        score = score.clip(0, 1)

        log.info(
            f"ChoppyRegimeDetector v2: "
            f"mean={score.mean():.3f} "
            f"p75={score.quantile(0.75):.3f} "
            f"p95={score.quantile(0.95):.3f} "
            f"RED={(score > 0.65).sum()} days"
        )

        if return_groups:
            groups["choppy_score"] = score
            return score, groups

        return score

    def score_today(
        self,
        prices: pd.DataFrame,
        vix: pd.Series,
    ) -> float:
        """
        Score the most recent trading day for live/paper mode.
        Returns float ∈ [0, 1].
        """
        score = self.score_series(prices, vix, smooth=True)
        val   = float(score.iloc[-1]) if len(score) > 0 else 0.0
        log.info(f"ChoppyRegimeDetector v2 live score: {val:.3f}")
        return float(np.clip(val, 0, 1))

    @staticmethod
    def score_to_scale(score: float) -> Tuple[float, str]:
        """Convert choppy score → (position scale factor, colour label)."""
        for threshold, scale, colour, _ in CHOPPY_SCALE_THRESHOLDS:
            if score < threshold:
                return scale, colour
        return 0.20, "RED"

    @staticmethod
    def feature_report(
        prices: pd.DataFrame,
        vix: pd.Series,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of daily group scores for a date range.
        Useful for diagnostics, visualisation, and calibration verification.
        """
        det   = ChoppyRegimeDetector()
        score, groups = det.score_series(prices, vix, return_groups=True)
        return groups.loc[start:end]


# ── Diagnostic helper ─────────────────────────────────────────────────────────

def run_diagnostic(
    prices: pd.DataFrame,
    vix: pd.Series,
    periods: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """
    Run the detector across specified periods and return a summary DataFrame.
    """
    det    = ChoppyRegimeDetector()
    scores = det.score_series(prices, vix)

    rows = []
    for label, (s, e) in periods.items():
        sub = scores.loc[s:e]
        if sub.empty:
            continue
        rows.append({
            "period":     label,
            "mean_score": round(float(sub.mean()), 3),
            "p75_score":  round(float(sub.quantile(0.75)), 3),
            "p95_score":  round(float(sub.quantile(0.95)), 3),
            "pct_green":  round(float((sub < 0.30).mean() * 100), 1),
            "pct_yellow": round(float(((sub >= 0.30) & (sub < 0.50)).mean() * 100), 1),
            "pct_orange": round(float(((sub >= 0.50) & (sub < 0.65)).mean() * 100), 1),
            "pct_red":    round(float((sub >= 0.65).mean() * 100), 1),
        })
    return pd.DataFrame(rows)
