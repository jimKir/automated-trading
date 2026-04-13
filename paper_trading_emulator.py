"""
Paper Trading Emulator
======================
Walks through Dec 2025 – Apr 2026 completely out-of-sample, simulating
exactly what the live engine would have done: weekly rebalance, full
strategy stack (regime switching, vol-engine, position anomaly scorer,
EWS choppy detector), realistic costs.

Key design decisions
--------------------
- STRICT causal boundary: each weekly rebalance uses ONLY data up to
  and including the Friday close. No Saturday/Sunday peeking.
- Prices fed to SignalGenerator are sliced to [warmup_start : rebalance_date].
- All protective layers are computed from the same causal slice.
- Regime switching uses IS-validated params (v1.0.0-paper-baseline).
- Costs: 0.126% round-trip (equity/ETF), 0.20% (crypto), 0.10% (futures).

Universe
--------
  Equities/ETFs: SPY QQQ IWM GLD TLT VGK EEM XLK XLE XLF VNQ AGG EWJ XLV
  Crypto:        BTC ETH SOL
  Futures:       ES NQ GC CL

Output files
------------
  results/paper_trading_equity_curve.png
  results/paper_trading_positions.csv
  results/paper_trading_weekly_pnl.csv
  results/paper_trading_summary.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from regime.choppy_regime import ChoppyRegimeDetector
from risk.position_anomaly import AssetClass, PositionAnomalyScorer, apply_position_scales, classify
from strategy.signals import SignalGenerator
from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("PaperTradingEmulator")

# ── Constants ─────────────────────────────────────────────────────────────────
PAPER_START = "2025-12-01"
PAPER_END = "2026-04-02"
WARMUP_START = "2017-01-01"  # data loaded from here for signal warmup
INITIAL_EQUITY = 25_000.0
PERIODS_YEAR = 252

# Round-trip costs by asset class
COSTS = {
    AssetClass.CRYPTO: 0.0020,  # 0.20% (spread + exchange fee)
    AssetClass.EQUITY: 0.00126,  # 0.126%
    AssetClass.ETF_EQUITY: 0.00126,
    AssetClass.ETF_HEDGE: 0.00126,
    AssetClass.COMMODITY: 0.00100,  # futures slightly cheaper
    AssetClass.FX: 0.00080,
    AssetClass.UNKNOWN: 0.00126,
}

# Symbol → local parquet file name mapping
SYM_MAP = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "ES=F": "ES",
    "NQ=F": "NQ",
    "GC=F": "GC",
    "CL=F": "CL",
    "^VIX": "VIX",
}

DATA_DIR = Path("data/historical/daily")

C = {
    "strat": "#20808D",
    "spy": "#A84B2F",
    "btc": "#944454",
    "bg": "#F7F6F2",
    "surface": "#F9F8F5",
    "border": "#D4D1CA",
    "text": "#28251D",
    "muted": "#7A7974",
    "grid": "#E8E6E0",
    "green": "#437A22",
    "red": "#A12C7B",
    "amber": "#964219",
}


# ── Data loading ──────────────────────────────────────────────────────────────


def load_sym(sym: str) -> pd.DataFrame | None:
    """Load OHLCV DataFrame from local parquet store."""
    name = SYM_MAP.get(
        sym,
        sym.replace("=F", "").replace("^", "").replace("-", "")[:6]
        if "=" in sym or "^" in sym
        else sym.replace("-USD", ""),
    )
    for candidate in [name, sym]:
        p = DATA_DIR / f"{candidate}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.capitalize() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if "Close" not in df.columns and "close" in df.columns:
                df = df.rename(
                    columns={
                        "close": "Close",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "volume": "Volume",
                    }
                )
            return df
    return None


# ── Main emulator ─────────────────────────────────────────────────────────────


class PaperTradingEmulator:
    def __init__(self, config: dict):
        self.config = config
        self.signal_gen = SignalGenerator(config)
        self.pos_scorer = PositionAnomalyScorer()
        self.choppy_det = ChoppyRegimeDetector()

        # Portfolio state
        self.equity = INITIAL_EQUITY
        self.cash = INITIAL_EQUITY
        self.positions: dict[str, float] = {}  # sym → dollar value
        self.weights: dict[str, float] = {}  # sym → target weight

        # Logs
        self.equity_log: list[dict] = []
        self.trade_log: list[dict] = []
        self.position_log: list[dict] = []
        self.signal_log: list[dict] = []

    def _rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """Weekly: last trading day of each ISO week within the paper window."""
        df = pd.DataFrame({"d": dates})
        df["wk"] = df["d"].dt.to_period("W")
        return [grp["d"].iloc[-1] for _, grp in df.groupby("wk")]

    def _get_regime(self, vix_series: pd.Series, spy_close: pd.Series, date: pd.Timestamp) -> str:
        rs = self.config.get("strategy", {}).get("regime_switching", {})
        vix_thresh = rs.get("bull_vix_threshold", 20.0)
        ma_period = rs.get("bull_spy_ma_period", 200)
        try:
            vix_val = float(vix_series.asof(date))
            spy_val = float(spy_close.asof(date))
            spy_ma = float(spy_close.loc[:date].tail(ma_period).mean())
            return "bull" if (vix_val < vix_thresh and spy_val > spy_ma) else "bear"
        except Exception:
            return "bear"

    def _round_trip_cost(self, sym: str, trade_value: float) -> float:
        ac = classify(sym)
        rate = COSTS.get(ac, COSTS[AssetClass.UNKNOWN])
        return abs(trade_value) * rate

    def run(self, all_data: dict[str, pd.DataFrame]) -> dict:
        """
        Main emulation loop.

        Parameters
        ----------
        all_data : {symbol: OHLCV DataFrame} — full history from warmup_start
        """
        log.info(f"Starting paper trading emulation: {PAPER_START} → {PAPER_END}")

        # Build common index for paper window
        spy_df = all_data.get("SPY")
        spy_close = spy_df["Close"]
        spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)

        vix_s = None
        for vname in ["VIX", "^VIX"]:
            if vname in all_data:
                vix_s = all_data[vname]["Close"]
                vix_s.index = pd.to_datetime(vix_s.index).tz_localize(None)
                break
        if vix_s is None:
            # Synthetic VIX from SPY vol
            vix_s = spy_close.pct_change().rolling(20).std() * np.sqrt(252) * 100

        # Pre-compute ChoppyRegimeDetector over full history (causal)
        price_df_full = pd.DataFrame(
            {sym: df["Close"] for sym, df in all_data.items() if "Close" in df.columns}
        )
        price_df_full.index = (
            pd.to_datetime(price_df_full.index).tz_localize(None)
            if price_df_full.index.tz is not None
            else price_df_full.index
        )
        choppy_scores = self.choppy_det.score_series(price_df_full, vix_s)
        log.info("ChoppyRegimeDetector pre-computed")

        # Paper trading dates
        paper_dates = spy_close.loc[PAPER_START:PAPER_END].index.tolist()
        rebal_dates = set(self._rebalance_dates(pd.DatetimeIndex(paper_dates)))
        log.info(f"Paper dates: {len(paper_dates)} | rebalance dates: {len(rebal_dates)}")

        for date in paper_dates:
            # ── Daily mark-to-market ─────────────────────────────────────────
            portfolio_value = self.cash
            pos_values = {}
            for sym, pos_val in self.positions.items():
                if sym not in all_data:
                    continue
                df = all_data[sym]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                try:
                    price_today = float(df["Close"].asof(date))
                    price_prev = float(
                        df["Close"].asof(
                            paper_dates[paper_dates.index(date) - 1]
                            if paper_dates.index(date) > 0
                            else date
                        )
                    )
                    new_val = pos_val * (price_today / price_prev) if price_prev > 0 else pos_val
                    pos_values[sym] = new_val
                    portfolio_value += new_val
                except Exception:
                    pos_values[sym] = pos_val
                    portfolio_value += pos_val

            self.positions = pos_values
            self.equity = portfolio_value

            # Regime
            regime = self._get_regime(vix_s, spy_close, date)
            vix_now = float(vix_s.asof(date)) if not vix_s.empty else 20.0

            # Choppy score
            try:
                choppy_sc = float(choppy_scores.asof(date))
            except Exception:
                choppy_sc = 0.0

            self.equity_log.append(
                {
                    "date": date,
                    "equity": round(self.equity, 2),
                    "cash": round(self.cash, 2),
                    "n_positions": len(self.positions),
                    "regime": regime,
                    "vix": round(vix_now, 2),
                    "choppy_sc": round(choppy_sc, 3),
                }
            )

            # ── Rebalance ───────────────────────────────────────────────────
            if date not in rebal_dates:
                continue

            log.info(f"\n{'─' * 55}")
            log.info(
                f"REBALANCE {date.date()}  regime={regime}  "
                f"VIX={vix_now:.1f}  choppy={choppy_sc:.3f}  "
                f"equity=${self.equity:,.0f}"
            )

            # Slice historical data up to this date (causal)
            hist = {}
            for sym, df in all_data.items():
                df_c = df.copy()
                df_c.index = pd.to_datetime(df_c.index).tz_localize(None)
                sliced = df_c.loc[:date]
                if len(sliced) >= 60:
                    hist[sym] = sliced

            # Generate signals
            try:
                raw_signals = self.signal_gen.generate_latest(hist)
            except Exception as e:
                log.warning(f"Signal generation failed: {e} — holding positions")
                continue

            # Position anomaly scales (per-symbol asymmetric guard)
            try:
                port_score = choppy_sc
                price_close_df = pd.DataFrame(
                    {sym: df["Close"] for sym, df in hist.items() if "Close" in df.columns}
                )
                pos_scales = self.pos_scorer.score_today(price_close_df, portfolio_score=port_score)
            except Exception as e:
                log.debug(f"PositionAnomalyScorer failed: {e}")
                pos_scales = {}

            # Apply position scales to signals
            scaled_signals = apply_position_scales(raw_signals, pos_scales)

            # EWS choppy portfolio scale
            from regime.choppy_regime import CHOPPY_SCALE_THRESHOLDS

            ews_scale, ews_colour = 1.0, "GREEN"
            for thresh, scale_val, colour, _ in CHOPPY_SCALE_THRESHOLDS:
                if choppy_sc < thresh:
                    ews_scale, ews_colour = scale_val, colour
                    break

            # Target weights from signals (signal-proportional, top-N by strength)
            active = {s: v for s, v in scaled_signals.items() if abs(v) > 0.05}
            if not active:
                log.info("  No active signals — staying in cash")
                # Liquidate everything
                for sym in list(self.positions.keys()):
                    val = self.positions.pop(sym)
                    cost = self._round_trip_cost(sym, val)
                    self.cash += val - cost
                    self.trade_log.append(
                        {"date": date, "sym": sym, "action": "sell", "value": val, "cost": cost}
                    )
                continue

            # Long-only: top signals
            longs = {s: v for s, v in active.items() if v > 0}
            if not longs:
                longs = {s: abs(v) for s, v in active.items()}

            # Weight proportional to signal strength, capped at 15%
            max_pos = 0.15 * ews_scale
            max_heat = 0.75 * ews_scale
            total_sig = sum(longs.values())
            target_weights: dict[str, float] = {}
            heat = 0.0
            for sym, sig in sorted(longs.items(), key=lambda x: -x[1]):
                w = (sig / total_sig) * min(max_heat, 0.95)
                w = min(w, max_pos)
                if heat + w > max_heat:
                    w = max(0, max_heat - heat)
                if w > 0.01:
                    target_weights[sym] = w
                    heat += w

            target_values = {sym: w * self.equity for sym, w in target_weights.items()}

            # Log signal details
            log.info(
                f"  Signals: {len(longs)} longs  "
                f"EWS={ews_colour}({ews_scale:.0%})  "
                f"target_heat={heat:.0%}"
            )
            crypto_scales_log = {
                s: round(pos_scales.get(s, 1.0), 2)
                for s in target_weights
                if classify(s) == AssetClass.CRYPTO
            }
            if crypto_scales_log:
                log.info(f"  Crypto pos_scales: {crypto_scales_log}")

            # Execute trades: close positions not in new targets, open/resize others
            total_cost = 0.0
            # Close exits
            for sym in list(self.positions.keys()):
                if sym not in target_values:
                    val = self.positions.pop(sym)
                    cost = self._round_trip_cost(sym, val)
                    self.cash += val - cost
                    total_cost += cost
                    self.trade_log.append(
                        {
                            "date": date,
                            "sym": sym,
                            "action": "close",
                            "value": round(val, 2),
                            "cost": round(cost, 4),
                            "regime": regime,
                        }
                    )

            # Open / resize
            for sym, target_val in target_values.items():
                current_val = self.positions.get(sym, 0.0)
                delta = target_val - current_val
                if abs(delta) < 10.0:  # skip micro-adjustments
                    continue
                cost = self._round_trip_cost(sym, delta)
                if delta > 0:
                    if self.cash >= delta + cost:
                        self.cash -= delta + cost
                        self.positions[sym] = current_val + delta
                        total_cost += cost
                        self.trade_log.append(
                            {
                                "date": date,
                                "sym": sym,
                                "action": "buy",
                                "value": round(delta, 2),
                                "cost": round(cost, 4),
                                "regime": regime,
                            }
                        )
                else:
                    reduce = abs(delta)
                    self.positions[sym] = current_val - reduce
                    self.cash += reduce - cost
                    total_cost += cost
                    self.trade_log.append(
                        {
                            "date": date,
                            "sym": sym,
                            "action": "trim",
                            "value": round(delta, 2),
                            "cost": round(cost, 4),
                            "regime": regime,
                        }
                    )

            # Snapshot positions
            pos_snapshot = {
                "date": date,
                "equity": round(self.equity, 2),
                "cash": round(self.cash, 2),
                "cash_pct": round(self.cash / self.equity * 100, 1),
                "n_positions": len(self.positions),
                "total_costs": round(total_cost, 2),
                "regime": regime,
                "ews_colour": ews_colour,
                "ews_scale": ews_scale,
                "choppy_score": round(choppy_sc, 3),
                "positions": {s: round(v, 2) for s, v in self.positions.items()},
            }
            self.position_log.append(pos_snapshot)

            top3 = sorted(self.positions.items(), key=lambda x: -abs(x[1]))[:3]
            top3_str = "  ".join(f"{s}=${v:,.0f}" for s, v in top3)
            log.info(
                f"  Portfolio: ${self.equity:,.0f}  cash={self.cash / self.equity:.0%}"
                f"  n_pos={len(self.positions)}  costs=${total_cost:.2f}"
            )
            log.info(f"  Top3: {top3_str}")

        log.info(f"\n{'=' * 55}")
        log.info(f"Emulation complete. Final equity: ${self.equity:,.2f}")
        return self._compile_results()

    def _compile_results(self) -> dict:
        eq = pd.DataFrame(self.equity_log).set_index("date")
        eq.index = pd.to_datetime(eq.index)
        ret = eq["equity"].pct_change().dropna()
        ann = (1 + ret).prod() ** (PERIODS_YEAR / len(ret)) - 1
        vol = ret.std() * np.sqrt(PERIODS_YEAR)
        sh = ann / vol if vol > 0 else np.nan
        cum = eq["equity"] / eq["equity"].iloc[0]
        dd = (cum - cum.cummax()) / cum.cummax()
        mdd = float(dd.min()) * 100
        # SPY comparison
        spy_df = load_sym("SPY")
        spy_ret = spy_df["Close"].loc[PAPER_START:PAPER_END].pct_change().dropna()
        spy_ann = (1 + spy_ret).prod() ** (PERIODS_YEAR / len(spy_ret)) - 1
        spy_vol = spy_ret.std() * np.sqrt(PERIODS_YEAR)
        spy_sh = spy_ann / spy_vol if spy_vol > 0 else np.nan
        spy_cum = spy_df["Close"].loc[PAPER_START:PAPER_END]
        spy_cum = spy_cum / spy_cum.iloc[0]
        spy_dd = (spy_cum - spy_cum.cummax()) / spy_cum.cummax()
        return {
            "period": f"{PAPER_START} → {PAPER_END}",
            "initial_equity": INITIAL_EQUITY,
            "final_equity": round(self.equity, 2),
            "total_return_pct": round((self.equity / INITIAL_EQUITY - 1) * 100, 2),
            "cagr_pct": round(ann * 100, 2),
            "sharpe": round(float(sh), 3) if not np.isnan(sh) else None,
            "max_dd_pct": round(mdd, 2),
            "ann_vol_pct": round(vol * 100, 2),
            "spy_total_return": round((spy_ret.apply(lambda x: 1 + x).prod() - 1) * 100, 2),
            "spy_sharpe": round(float(spy_sh), 3),
            "spy_max_dd": round(float(spy_dd.min()) * 100, 2),
            "n_trades": len(self.trade_log),
            "n_rebalances": len(self.position_log),
            "equity_log": self.equity_log,
            "position_log": self.position_log,
            "trade_log": self.trade_log,
        }


# ── Chart generation ──────────────────────────────────────────────────────────


def generate_chart(results: dict) -> str:
    eq_log = pd.DataFrame(results["equity_log"]).set_index("date")
    eq_log.index = pd.to_datetime(eq_log.index)

    # SPY benchmark
    spy_df = load_sym("SPY")
    spy_c = spy_df["Close"].loc[PAPER_START:PAPER_END]
    spy_cum = spy_c / spy_c.iloc[0]

    strat_cum = eq_log["equity"] / eq_log["equity"].iloc[0]
    dd = (strat_cum - strat_cum.cummax()) / strat_cum.cummax()
    spy_dd = (spy_cum - spy_cum.cummax()) / spy_cum.cummax()

    fig = plt.figure(figsize=(16, 14), facecolor=C["bg"])
    gs = GridSpec(
        4,
        2,
        figure=fig,
        hspace=0.45,
        wspace=0.30,
        top=0.91,
        bottom=0.06,
        left=0.07,
        right=0.97,
        height_ratios=[2.0, 0.8, 0.7, 0.9],
    )

    ax_eq = fig.add_subplot(gs[0, :])  # equity curve (full width)
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_eq)  # drawdown
    ax_reg = fig.add_subplot(gs[2, 0])  # regime/VIX
    ax_cs = fig.add_subplot(gs[2, 1])  # choppy score
    ax_pos = fig.add_subplot(gs[3, :])  # position heat

    for ax in [ax_eq, ax_dd, ax_reg, ax_cs, ax_pos]:
        ax.set_facecolor(C["surface"])
        ax.spines[["top", "right", "bottom", "left"]].set_color(C["border"])
        ax.tick_params(colors=C["muted"], labelsize=9)
        ax.grid(True, color=C["grid"], linewidth=0.5, alpha=0.7)

    # ── Equity curve ─────────────────────────────────────────────────────────
    ax_eq.plot(
        strat_cum.index, strat_cum.values, color=C["strat"], lw=2.2, label="Strategy", zorder=5
    )
    ax_eq.plot(
        spy_cum.index,
        spy_cum.values,
        color=C["spy"],
        lw=1.5,
        linestyle="--",
        alpha=0.85,
        label="SPY",
        zorder=4,
    )
    ax_eq.axhline(1.0, color=C["muted"], lw=0.7, linestyle=":", alpha=0.5)
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}×"))
    ax_eq.legend(
        loc="upper left",
        framealpha=0.92,
        facecolor="white",
        edgecolor=C["border"],
        fontsize=10,
        labelcolor=C["text"],
    )
    tr = results["total_return_pct"]
    sh = results["sharpe"] or 0.0
    mdd = results["max_dd_pct"]
    spy_tr = results["spy_total_return"]
    spy_sh = results["spy_sharpe"]
    ax_eq.text(
        0.98,
        0.97,
        f"Strategy: {tr:+.1f}%  Sharpe {sh:.2f}  MaxDD {mdd:.1f}%\n"
        f"SPY:      {spy_tr:+.1f}%  Sharpe {spy_sh:.2f}  MaxDD {results['spy_max_dd']:.1f}%",
        transform=ax_eq.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=C["text"],
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": C["border"],
            "alpha": 0.9,
            "linewidth": 0.8,
        },
    )
    ax_eq.set_title(
        "Paper Trading Emulation — Dec 2025 → Apr 2026 (Fully OOS)",
        fontsize=12,
        color=C["text"],
        fontweight="bold",
        pad=7,
    )

    # ── Drawdown ─────────────────────────────────────────────────────────────
    ax_dd.fill_between(
        dd.index, dd.values * 100, 0, alpha=0.5, color=C["strat"], label="Strategy DD", zorder=5
    )
    ax_dd.plot(
        spy_dd.index,
        spy_dd.values * 100,
        color=C["spy"],
        lw=1.2,
        linestyle="--",
        alpha=0.8,
        label="SPY DD",
        zorder=4,
    )
    ax_dd.axhline(0, color=C["muted"], lw=0.6, alpha=0.5)
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_dd.set_ylabel("Drawdown", color=C["muted"], fontsize=9)
    ax_dd.legend(
        loc="lower left",
        framealpha=0.9,
        facecolor="white",
        edgecolor=C["border"],
        fontsize=8.5,
        labelcolor=C["text"],
    )
    plt.setp(ax_dd.get_xticklabels(), visible=False)

    # ── VIX + regime ─────────────────────────────────────────────────────────
    vix_s = eq_log["vix"]
    regime_s = eq_log["regime"]
    ax_reg.plot(vix_s.index, vix_s.values, color=C["amber"], lw=1.5, label="VIX")
    ax_reg.axhline(20, color=C["muted"], lw=0.8, linestyle="--", alpha=0.6)
    ax_reg.text(vix_s.index[-1], 20.5, "20", fontsize=7.5, color=C["muted"])
    # Shade bear regime
    bull = regime_s == "bull"
    for i in range(len(bull) - 1):
        if not bull.iloc[i]:
            ax_reg.axvspan(bull.index[i], bull.index[i + 1], alpha=0.12, color=C["red"], zorder=0)
    ax_reg.set_ylabel("VIX", color=C["muted"], fontsize=9)
    ax_reg.set_title(
        "VIX + Regime (pink = bear)", fontsize=9, color=C["text"], fontweight="bold", pad=4
    )
    ax_reg.legend(
        fontsize=8, framealpha=0.9, facecolor="white", edgecolor=C["border"], labelcolor=C["text"]
    )

    # ── Choppy score ─────────────────────────────────────────────────────────
    ch_s = eq_log["choppy_sc"]
    ax_cs.fill_between(
        ch_s.index, ch_s.values, 0, alpha=0.6, color=C["strat"], label="Choppy score"
    )
    ax_cs.axhline(0.17, color=C["muted"], lw=0.8, linestyle="--", alpha=0.6)
    ax_cs.axhline(0.27, color=C["amber"], lw=0.8, linestyle="--", alpha=0.6)
    ax_cs.text(ch_s.index[-1], 0.18, "YELLOW", fontsize=6.5, color=C["amber"])
    ax_cs.text(ch_s.index[-1], 0.28, "ORANGE", fontsize=6.5, color=C["red"])
    ax_cs.set_ylim(0, max(0.5, float(ch_s.max()) * 1.1))
    ax_cs.set_ylabel("Choppy Score", color=C["muted"], fontsize=9)
    ax_cs.set_title(
        "ChoppyRegimeDetector (EWS Layer F)", fontsize=9, color=C["text"], fontweight="bold", pad=4
    )

    # ── Position heatmap (# positions + cash %) over time) ───────────────────
    pos_df = pd.DataFrame(results["position_log"])
    pos_df["date"] = pd.to_datetime(pos_df["date"])
    pos_df = pos_df.set_index("date")
    ax_pos.bar(
        pos_df.index, pos_df["n_positions"], color=C["strat"], alpha=0.7, label="# Positions"
    )
    ax_pos2 = ax_pos.twinx()
    ax_pos2.plot(
        pos_df.index, pos_df["cash_pct"], color=C["amber"], lw=1.5, linestyle="--", label="Cash %"
    )
    ax_pos2.set_ylabel("Cash %", color=C["amber"], fontsize=9)
    ax_pos2.tick_params(colors=C["amber"], labelsize=8)
    ax_pos.set_ylabel("# Positions", color=C["muted"], fontsize=9)
    ax_pos.set_title(
        "Position Count & Cash Allocation (rebalance dates)",
        fontsize=9,
        color=C["text"],
        fontweight="bold",
        pad=4,
    )
    lines1, labels1 = ax_pos.get_legend_handles_labels()
    lines2, labels2 = ax_pos2.get_legend_handles_labels()
    ax_pos.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=8.5,
        framealpha=0.9,
        facecolor="white",
        edgecolor=C["border"],
        labelcolor=C["text"],
    )

    # ── Main title ────────────────────────────────────────────────────────────
    fig.text(
        0.07,
        0.955,
        "Paper Trading Emulation — Full Strategy Stack, Fully OOS",
        fontsize=14,
        fontweight="bold",
        color=C["text"],
    )
    fig.text(
        0.07,
        0.935,
        "IS-validated regime params  ·  WF-calibrated PositionAnomalyScorer  ·  "
        "ChoppyRegimeDetector  ·  $25k initial  ·  Realistic costs",
        fontsize=9,
        color=C["muted"],
    )

    out = "results/paper_trading_equity_curve.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(out, dpi=155, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    log.info(f"Chart saved → {out}")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    config = load_config("config/settings.yaml")

    # Symbols to load
    UNIVERSE = [
        "SPY",
        "QQQ",
        "IWM",
        "GLD",
        "TLT",
        "VGK",
        "EEM",
        "XLK",
        "XLE",
        "XLF",
        "VNQ",
        "AGG",
        "EWJ",
        "XLV",
        "EMXC",
        "BTC",
        "ETH",
        "SOL",  # crypto (local names)
        "ES",
        "NQ",
        "GC",
        "CL",  # futures (local names)
        "VIX",
        "HYG",
        "LQD",
        "DXY",
        "NFLX",
        "MMM",
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
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
    ]

    log.info("Loading all data...")
    all_data: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        df = load_sym(sym)
        if df is not None and len(df) > 100:
            all_data[sym] = df
        else:
            log.debug(f"  skip {sym} (not found)")
    log.info(f"Loaded {len(all_data)} symbols")

    # Run emulation
    emulator = PaperTradingEmulator(config)
    results = emulator.run(all_data)

    # Save outputs
    Path("results").mkdir(exist_ok=True)

    pd.DataFrame(results["trade_log"]).to_csv("results/paper_trading_trades.csv", index=False)
    pd.DataFrame(results["equity_log"]).to_csv("results/paper_trading_equity_log.csv", index=False)

    pos_rows = []
    for snap in results["position_log"]:
        row = {k: v for k, v in snap.items() if k != "positions"}
        row["top_positions"] = str(sorted(snap["positions"].items(), key=lambda x: -abs(x[1]))[:5])
        pos_rows.append(row)
    pd.DataFrame(pos_rows).to_csv("results/paper_trading_positions.csv", index=False)

    summary = {
        k: v for k, v in results.items() if k not in ("equity_log", "position_log", "trade_log")
    }
    with open("results/paper_trading_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Generate chart
    chart_path = generate_chart(results)

    # Print summary
    print(f"\n{'=' * 60}")
    print("PAPER TRADING EMULATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Period:       {results['period']}")
    print(f"  Initial:      ${results['initial_equity']:,.0f}")
    print(f"  Final:        ${results['final_equity']:,.2f}")
    print(f"  Total return: {results['total_return_pct']:+.2f}%")
    print(f"  CAGR:         {results['cagr_pct']:+.2f}%")
    print(f"  Sharpe:       {results['sharpe']}")
    print(f"  Max DD:       {results['max_dd_pct']:.2f}%")
    print(f"  Volatility:   {results['ann_vol_pct']:.2f}% ann.")
    print(f"  Trades:       {results['n_trades']}")
    print(f"  Rebalances:   {results['n_rebalances']}")
    print(
        f"\n  SPY baseline: {results['spy_total_return']:+.2f}% | "
        f"Sharpe {results['spy_sharpe']} | MaxDD {results['spy_max_dd']:.2f}%"
    )
    print(f"\n  Chart: {chart_path}")
    print("  Summary: results/paper_trading_summary.json")


if __name__ == "__main__":
    main()
