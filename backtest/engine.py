"""
Backtesting Engine
==================
Event-driven daily backtester.
Produces full performance report with:
  - Returns, Sharpe, Sortino, Calmar
  - Max Drawdown, Duration
  - VaR/CVaR (historical, parametric, MC)
  - Black swan metrics: Omega, Tail Ratio, Skew, Excess Kurtosis
  - Stress test scenarios
  - Trade statistics
  - Benchmark comparison (alpha, beta, information ratio)
  - Early Warning System (EWS) regime gating (optional)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

from core.portfolio import Portfolio
from risk.manager import RiskManager
from strategy.signals import SignalGenerator
from utils.logger import get_logger
from core.intraday_shock import VOL_LOOKBACK  # volume baseline window

log = get_logger("Backtest")

PERIODS_PER_YEAR = 252


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        self.bt_cfg = config.get("backtest", {})
        self.start = self.bt_cfg.get("start_date", "2018-01-01")
        self.end = self.bt_cfg.get("end_date", "2025-12-31")
        self.commission = self.bt_cfg.get("commission_pct", 0.001)
        self.slippage = self.bt_cfg.get("slippage_pct", 0.0005)
        self.benchmark_sym = self.bt_cfg.get("benchmark", "SPY")
        self.rebalance_freq = config.get("strategy", {}).get("rebalance_frequency", "weekly")
        self.use_ews            = config.get("ews",            {}).get("enabled", False)
        self.use_vol_targeting  = config.get("vol_targeting",  {}).get("enabled", False)
        self.use_intraday_shock = config.get("intraday_shock", {}).get("enabled", False)

    # -----------------------------------------------------------------------
    # Core loop
    # -----------------------------------------------------------------------

    def run(
        self,
        all_data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
        run_label: str = "",
    ) -> Dict[str, Any]:
        """
        Run the backtest.

        Parameters
        ----------
        all_data       : {symbol: OHLCV DataFrame}
        benchmark_data : OHLCV DataFrame for the benchmark symbol
        run_label      : optional label for comparison runs (e.g. "with_ews")

        Returns
        -------
        dict with performance metrics and result DataFrames
        """
        log.info(f"Starting backtest: {self.start} → {self.end}"
                 + (f" [{run_label}]" if run_label else ""))

        portfolio  = Portfolio(self.config)
        risk_mgr   = RiskManager(self.config)
        signal_gen = SignalGenerator(self.config)

        # ── Dynamic universe selector ─────────────────────────────────────────
        universe_selector = None
        universe_selections: dict = {}
        if self.config.get("dynamic_universe", {}).get("enabled", False):
            try:
                from strategy.universe import DynamicUniverseSelector
                universe_selector = DynamicUniverseSelector(self.config)
                log.info("Dynamic universe selector enabled — pre-computing selections...")
                universe_selections = universe_selector.compute_selection_series(
                    all_data, []
                )
            except Exception as e:
                log.warning(f"Dynamic universe selector failed: {e}")

        # ── Load macro data for credit regime signal ─────────────────────────
        pred_cfg = self.config.get("strategy", {}).get("predictive", {})
        if pred_cfg.get("credit_regime_enabled", False):
            macro_syms = pred_cfg.get("macro_symbols", ["HYG", "LQD", "^VIX", "SHY"])
            try:
                from data.feed import fetch_yfinance
                macro_data = fetch_yfinance(macro_syms, self.start, self.end)
                if "TLT" in all_data and "TLT" not in macro_data:
                    macro_data["TLT"] = all_data["TLT"]
                signal_gen.set_macro_data(macro_data)
                log.info(f"Credit regime: loaded {len(macro_data)} macro symbols")
            except Exception as e:
                log.warning(f"Credit regime: failed to load macro data: {e}")

        # ── UNION date alignment — assets with different calendars all participate
        # Crypto trades weekends, equities/futures weekdays — each on its own schedule
        all_dates_set = set()
        for df in all_data.values():
            all_dates_set.update(df.index)
        all_dates = sorted(all_dates_set)
        all_dates = [d for d in all_dates
                     if pd.Timestamp(self.start, tz="UTC") <= d <= pd.Timestamp(self.end, tz="UTC")]

        if not all_dates:
            raise ValueError("No overlapping dates found across assets in the specified range.")

        log.info(f"Trading days: {len(all_dates)}")

        # ── Pre-compute dynamic universe selections if enabled ────────────────
        if universe_selector is not None:
            # Reset selector state and compute for real this time
            universe_selector._last_selected   = []
            universe_selector._last_rank_date  = None
            universe_selections = universe_selector.compute_selection_series(
                all_data, all_dates
            )
            # Log stats
            stats = universe_selector.get_selection_stats(universe_selections)
            log.info(
                f"Dynamic universe: {stats['total_candidates']} candidates → "
                f"{stats['ever_selected']} ever selected | "
                f"avg turnover {stats['avg_turnover_pct']:.1f}%/rebalance"
            )
        else:
            log.info(f"Trading universe: {list(all_data.keys())}")

        # ── Build combined price DataFrame for EWS ──────────────────────────
        # (Close prices, date-aligned)
        price_df = pd.DataFrame({
            sym: df["Close"]
            for sym, df in all_data.items()
        }).sort_index()

        # ── Pre-compute vol targeting scale series ───────────────────────────
        # We can't compute the final scale series yet (no equity returns exist).
        # Instead, VolatilityTargeter is initialised here and called live each
        # day using the equity curve built so far. This is strictly causal.
        vt_obj = None
        if self.use_vol_targeting:
            try:
                from core.vol_targeting import VolatilityTargeter
                vt_obj = VolatilityTargeter(self.config)
                log.info("Vol targeting enabled.")
            except Exception as e:
                log.warning(f"Vol targeting failed to init: {e}")

        # Pre-fetch VIX for H2O vol forecaster (if not already loaded by ISD)
        _vix_for_h2o: pd.Series = pd.Series(dtype=float)
        if self.use_vol_targeting and self.config.get("vol_targeting",{}).get("use_h2o_vol", False):
            try:
                import yfinance as yf
                _vix_raw = yf.download("^VIX", start=self.start, end=self.end,
                                       auto_adjust=True, progress=False)
                if not _vix_raw.empty:
                    _vix_for_h2o = _vix_raw["Close"].squeeze().dropna()
                    if _vix_for_h2o.index.tz is not None:
                        _vix_for_h2o.index = _vix_for_h2o.index.tz_localize(None)
                log.info(f"VIX loaded for H2O vol forecaster: {len(_vix_for_h2o)} bars")
            except Exception as e:
                log.warning(f"VIX fetch for H2O failed: {e}")

        # ── Fetch VIX for intraday shock backtest simulation ────────────────────
        isd_obj        = None
        isd_bt_scales: Optional[pd.Series] = None
        if self.use_intraday_shock:
            try:
                from core.intraday_shock import IntradayShockDetector
                isd_obj = IntradayShockDetector(self.config)
                log.info("Intraday shock detector: fetching VIX history...")
                import yfinance as yf
                vix_raw = yf.download("^VIX", start=self.start, end=self.end,
                                      auto_adjust=True, progress=False)
                if not vix_raw.empty:
                    vix_series = vix_raw["Close"].squeeze().dropna()
                    if vix_series.index.tz is not None:
                        vix_series.index = vix_series.index.tz_localize(None)
                else:
                    vix_series = pd.Series(dtype=float)
                log.info(f"VIX data: {len(vix_series)} days loaded")

                # Also fetch SPY volume for volume shock detection
                spy_vol_raw = yf.download("SPY", start=self.start, end=self.end,
                                          auto_adjust=True, progress=False)
                if not spy_vol_raw.empty and "Volume" in spy_vol_raw.columns:
                    spy_vol_series = spy_vol_raw["Volume"].squeeze().dropna()
                    if spy_vol_series.index.tz is not None:
                        spy_vol_series.index = spy_vol_series.index.tz_localize(None)
                else:
                    spy_vol_series = pd.Series(dtype=float)
                log.info(f"SPY volume: {len(spy_vol_series)} days loaded for ISD")
            except Exception as e:
                log.warning(f"Intraday shock init failed: {e}")
                isd_obj = None

        # ── Pre-compute EWS scores ───────────────────────────────────────────
        ews_scores: Optional[pd.Series] = None
        ews_obj = None
        if self.use_ews:
            try:
                from regime.ews import EarlyWarningSystem
                ews_obj = EarlyWarningSystem(self.config)
                log.info("EWS enabled — pre-computing stress scores (this takes ~2-3 min)...")
                ews_scores = ews_obj.compute_backtest_scores(price_df, self.start, self.end)
                log.info("EWS pre-computation complete.")
            except Exception as e:
                log.warning(f"EWS failed to initialise: {e} — running without EWS")
                ews_scores = None

        # Determine rebalance dates
        rebalance_dates = self._rebalance_schedule(all_dates)
        log.info(f"Rebalance days: {len(rebalance_dates)} ({self.rebalance_freq})")

        risk_mgr.update_equity(portfolio.equity)
        risk_mgr.reset_daily(portfolio.equity, cash=portfolio.cash)

        ews_scale_log = []   # track EWS scale factor over time
        vt_scale_log  = []   # track vol targeting scale factor over time
        isd_scale_log = []   # track intraday shock scale factor over time
        _equity_buffer: list = []   # rolling equity values for vol estimation


        # Pre-compute intraday shock scales for backtest (needs equity curve first
        # — done incrementally below using equity buffer)

        for i, date in enumerate(all_dates):
            # ── Daily reset at TOP of loop (before halt check) ────────────────
            # If reset is at the bottom, a halted day skips it → permanent halt.
            if i > 0 and date.date() != all_dates[i - 1].date():
                risk_mgr.reset_daily(portfolio.equity, cash=portfolio.cash)
                risk_mgr.update_equity(portfolio.equity)

            # Update prices — only symbols that have data on this date (UNION calendar)
            prices = {}
            for sym, df in all_data.items():
                if date in df.index:
                    val = df.loc[date, "Close"]
                    if not np.isnan(float(val)):
                        prices[sym] = float(val)

            if not prices:
                portfolio.record_equity(date)
                continue

            portfolio.update_prices(prices)

            # Check risk circuit breakers (pass cash for realised-only daily check)
            halt, reason = risk_mgr.check_halt(
                portfolio.equity, cash=portfolio.cash, date=date
            )
            if halt:
                portfolio.record_equity(date)
                continue

            # ── EWS scale factor for today ────────────────────────────────────
            ews_scale  = 1.0
            ews_colour = "GREEN"
            if ews_scores is not None and ews_obj is not None:
                ews_scale, ews_colour = ews_obj.get_scale_factor(date, ews_scores)
                ews_scale_log.append({"date": date, "scale": ews_scale, "regime": ews_colour})

            # ── Vol targeting scale factor for today ──────────────────────────
            # Compute daily portfolio return from equity buffer (strictly causal)
            vt_scale = 1.0
            if vt_obj is not None:
                _equity_buffer.append(portfolio.equity)
                if len(_equity_buffer) >= 2:
                    e_prev = _equity_buffer[-2]
                    e_curr = _equity_buffer[-1]
                    daily_ret = (e_curr - e_prev) / e_prev if e_prev > 0 else 0.0
                    vt_scale = vt_obj.update_and_get_scale(daily_ret)
                vt_scale_log.append({"date": date, "vt_scale": vt_scale})

            # ── Intraday shock scale ───────────────────────────────────────
            # Uses day-over-day VIX and equity changes as proxy for intraday moves.
            # Morning snapshot = yesterday's closing equity and VIX.
            # 'Intraday' move = today's close vs yesterday's close.
            isd_scale = 1.0
            if isd_obj is not None and i > 0 and not vix_series.empty:
                date_naive      = date.replace(tzinfo=None)
                prev_date_naive = all_dates[i-1].replace(tzinfo=None)

                vix_today = float(vix_series.asof(date_naive))
                vix_prev  = float(vix_series.asof(prev_date_naive))

                eq_prev = float(_equity_buffer[-1]) if _equity_buffer else portfolio.equity
                eq_curr = portfolio.equity

                # SPY volume for volume shock detection
                spy_vol_today = None
                if 'spy_vol_series' in locals() and not spy_vol_series.empty:
                    _sv = spy_vol_series.asof(date_naive)
                    if _sv is not None and not pd.isna(_sv):
                        spy_vol_today = float(_sv)
                    # Feed yesterday's volume into the rolling history
                    _sv_prev = spy_vol_series.asof(prev_date_naive)
                    if _sv_prev is not None and not pd.isna(_sv_prev):
                        isd_obj._volume_history.append(float(_sv_prev))
                        if len(isd_obj._volume_history) > VOL_LOOKBACK:
                            isd_obj._volume_history = isd_obj._volume_history[-VOL_LOOKBACK:]

                if vix_prev > 0 and eq_prev > 0:
                    from core.intraday_shock import ShockState as _SS, SCALE_RECOVERY as _SR
                    if isd_obj.current_state == _SS.RECOVERY:
                        isd_obj._recovery_day += 1
                        if isd_obj._recovery_day >= len(_SR):
                            isd_obj._state        = _SS.CLEAR
                            isd_obj._recovery_day = 0

                    isd_obj._morning_vix    = vix_prev
                    isd_obj._morning_equity = eq_prev
                    isd_obj._today          = date.date()

                    # price_chg proxy: equity return
                    price_chg_proxy = (eq_curr - eq_prev) / eq_prev if eq_prev > 0 else 0.0

                    isd_scale, isd_state, isd_reason = isd_obj.check(
                        vix_today, eq_curr,
                        current_volume=spy_vol_today,
                        prev_close=eq_prev,
                        current_close=eq_curr,
                    )
                    isd_scale_log.append({"date": date, "scale": isd_scale,
                                          "state": isd_state.value})
                    if isd_scale < 1.0:
                        log.debug(
                            f"[{date.date()}] ISD {isd_state.value}: "
                            f"scale={isd_scale:.0%} | {isd_reason}"
                        )

            # v15b: Combined scale = min of independent layers (not multiplicative!)
            # Vol targeting DISABLED — creates whipsaw that kills returns.
            # Drawdown scaling DISABLED — locks system out of recovery rallies.
            # Risk is managed via: lower max_heat, ISD, trend classifier,
            # regime scaling in optimizer, and risk parity position sizing.
            combined_scale = max(min(ews_scale, isd_scale), 0.50)

            # Rebalance on schedule
            if date in rebalance_dates:
                # ── Dynamic universe: get active instruments for today ────────
                if universe_selector is not None and date in universe_selections:
                    active_syms = set(universe_selections[date])
                    # Close out positions in instruments no longer selected
                    for sym in list(portfolio.positions.keys()):
                        if sym not in active_syms and sym in prices:
                            pos = portfolio.positions[sym]
                            if abs(pos.quantity) > 1e-8:
                                portfolio.execute_order(
                                    sym, -pos.quantity, prices[sym], date,
                                    self.commission, self.slippage
                                )
                    active_data = {
                        sym: df for sym, df in all_data.items()
                        if sym in active_syms
                    }
                else:
                    active_data = all_data

                historical_data = {
                    sym: df[df.index <= date] for sym, df in active_data.items()
                }
                signals = signal_gen.generate_latest(historical_data)

                max_pos  = self.config.get("risk", {}).get("max_position_pct", 0.15)
                max_heat = self.config.get("capital", {}).get("max_portfolio_heat", 0.40)

                # ── H2O vol targeting override at rebalance (batched) ────────
                # At each weekly rebalance, batch-predict vol for active symbols
                # and use the portfolio-mean vol to override the reactive EWMA scale.
                if vt_obj is not None and hasattr(vt_obj, '_h2o_fc') and len(_vix_for_h2o) > 0:
                    try:
                        vt_obj._load_h2o()  # ensure loaded
                        if vt_obj._h2o_fc is not None:
                            sym_returns = {
                                sym: np.log(df['Close']/df['Close'].shift(1)).dropna()
                                for sym, df in historical_data.items()
                                if 'Close' in df.columns and len(df) >= 63
                            }
                            # Single batched H2O call for all active symbols
                            h2o_vols = vt_obj._h2o_fc.predict_batch(
                                sym_returns, _vix_for_h2o, date
                            )
                            if h2o_vols:
                                port_vol_h2o = float(np.mean([
                                    v for v in h2o_vols.values() if 0.01 < v < 5.0
                                ]))
                                h2o_vt_scale = vt_obj.scale_from_vol(port_vol_h2o)
                                log.debug(f"[{date.date()}] H2O port_vol={port_vol_h2o:.1%} "
                                          f"→ scale={h2o_vt_scale:.3f} (ewma={vt_scale:.3f})")
                                vt_scale = h2o_vt_scale
                                # VT override logged for diagnostics but NOT used in combined_scale
                                # Vol targeting is disabled — see note above
                    except Exception as _e:
                        log.debug(f"H2O rebalance vol override failed: {_e}")

                # Apply combined scale (EWS × vol targeting) to position limits
                effective_max_pos  = max_pos  * combined_scale
                effective_max_heat = max_heat * combined_scale

                if ews_scale < 1.0:
                    log.debug(f"[{date.date()}] EWS {ews_colour}: "
                              f"scaling positions to {ews_scale:.0%}")

                # Pass price history + SPY for optimizer (risk parity / min-var)
                spy_hist = all_data.get("SPY")
                target_weights = portfolio.compute_target_weights(
                    signals,
                    max_position_pct   = effective_max_pos,
                    max_portfolio_heat = effective_max_heat,
                    price_history      = historical_data,
                    as_of_date         = date,
                    spy_data           = spy_hist,
                )
                orders = portfolio.compute_orders(target_weights, prices)

                for sym, qty in orders.items():
                    if sym not in prices:
                        continue
                    portfolio.execute_order(
                        symbol=sym,
                        quantity=qty,
                        price=prices[sym],
                        date=date,
                        commission_pct=self.commission,
                        slippage_pct=self.slippage,
                    )

            # Check stop-losses
            self._check_stops(portfolio, prices, date)

            # Apply daily carrying costs
            portfolio.apply_daily_costs(date)

            # Record equity
            portfolio.record_equity(date)

            # (daily reset is now at the TOP of the loop — see above)

        # -----------------------------------------------------------------------
        # Compute metrics
        # -----------------------------------------------------------------------
        equity_series = portfolio.get_equity_series()
        returns = equity_series.pct_change().dropna()

        bench_returns = None
        if benchmark_data is not None and not benchmark_data.empty:
            bench_close = benchmark_data["Close"].reindex(equity_series.index).ffill()
            bench_returns = bench_close.pct_change().dropna()

        metrics = self._compute_metrics(equity_series, returns, bench_returns, risk_mgr)
        metrics["equity_curve"] = equity_series
        metrics["returns"] = returns
        metrics["trades"] = portfolio.get_trade_df()
        metrics["positions_final"] = {
            sym: {"qty": p.quantity, "value": p.market_value, "pnl": p.unrealised_pnl}
            for sym, p in portfolio.positions.items()
        }
        metrics["final_equity"] = portfolio.equity
        metrics["cash"] = portfolio.cash
        metrics["run_label"] = run_label

        # Cost breakdown
        cost_summary = portfolio.cost_model.get_summary()
        metrics.update(cost_summary)
        initial_equity = self.config.get("capital", {}).get("initial_equity", 25000)
        metrics["cost_total_pct_of_initial"] = (
            cost_summary["cost_total"] / initial_equity * 100
            if initial_equity > 0 else 0
        )

        # Intraday shock summary
        if isd_scale_log:
            isd_df = pd.DataFrame(isd_scale_log).set_index("date")
            state_counts = isd_df["state"].value_counts().to_dict() if "state" in isd_df.columns else {}
            metrics["isd_days_caution"]  = int(state_counts.get("CAUTION",  0))
            metrics["isd_days_shock"]    = int(state_counts.get("SHOCK",    0))
            metrics["isd_days_recovery"] = int(state_counts.get("RECOVERY", 0))
            metrics["isd_avg_scale"]     = float(isd_df["scale"].mean())
            log.info(
                f"Intraday shock: CAUTION={metrics['isd_days_caution']}d | "
                f"SHOCK={metrics['isd_days_shock']}d | "
                f"RECOVERY={metrics['isd_days_recovery']}d | "
                f"avg_scale={metrics['isd_avg_scale']:.3f}"
            )

        # Vol targeting summary
        if vt_scale_log:
            vt_df = pd.DataFrame(vt_scale_log).set_index("date")
            metrics["vt_avg_scale"]      = float(vt_df["vt_scale"].mean())
            metrics["vt_min_scale"]      = float(vt_df["vt_scale"].min())
            metrics["vt_max_scale"]      = float(vt_df["vt_scale"].max())
            metrics["vt_days_scaled_up"] = int((vt_df["vt_scale"] > 1.05).sum())
            metrics["vt_days_scaled_dn"] = int((vt_df["vt_scale"] < 0.95).sum())
            log.info(
                f"Vol targeting: avg_scale={metrics['vt_avg_scale']:.2f}  "
                f"range=[{metrics['vt_min_scale']:.2f}, {metrics['vt_max_scale']:.2f}]  "
                f"scaled_up={metrics['vt_days_scaled_up']}d  "
                f"scaled_down={metrics['vt_days_scaled_dn']}d"
            )

        # EWS summary
        if ews_scale_log:
            ews_df = pd.DataFrame(ews_scale_log).set_index("date")
            regime_counts = ews_df["regime"].value_counts().to_dict()
            metrics["ews_regime_counts"]  = regime_counts
            metrics["ews_avg_scale"]      = float(ews_df["scale"].mean())
            metrics["ews_days_green"]     = int((ews_df["regime"] == "GREEN").sum())
            metrics["ews_days_yellow"]    = int((ews_df["regime"] == "YELLOW").sum())
            metrics["ews_days_orange"]    = int((ews_df["regime"] == "ORANGE").sum())
            metrics["ews_days_red"]       = int(((ews_df["regime"] == "RED") |
                                                 (ews_df["regime"] == "CRITICAL")).sum())
            metrics["ews_scores"]         = ews_scores
            log.info(f"EWS regime distribution: {regime_counts}")
            log.info(f"EWS avg scale factor: {metrics['ews_avg_scale']:.2%}")

        log.info(f"Total costs: ${cost_summary['cost_total']:,.2f} "
                 f"({metrics['cost_total_pct_of_initial']:.2f}% of initial equity)")

        self._log_summary(metrics)
        return metrics

    # -----------------------------------------------------------------------
    # Comparison run: baseline vs EWS
    # -----------------------------------------------------------------------

    def run_comparison(
        self,
        all_data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict]:
        """
        Run two backtests back-to-back:
          1. Baseline (EWS disabled)
          2. With EWS enabled
        Returns dict with keys "baseline" and "with_ews".
        """
        import copy

        # Baseline — EWS off
        cfg_base = copy.deepcopy(self.config)
        cfg_base.setdefault("ews", {})["enabled"] = False
        engine_base = BacktestEngine(cfg_base)
        log.info("\n" + "="*60)
        log.info("COMPARISON RUN 1/2: Baseline (no EWS)")
        log.info("="*60)
        metrics_base = engine_base.run(all_data, benchmark_data, run_label="Baseline")

        # With EWS
        cfg_ews = copy.deepcopy(self.config)
        cfg_ews.setdefault("ews", {})["enabled"] = True
        engine_ews = BacktestEngine(cfg_ews)
        log.info("\n" + "="*60)
        log.info("COMPARISON RUN 2/2: With EWS")
        log.info("="*60)
        metrics_ews = engine_ews.run(all_data, benchmark_data, run_label="With EWS")

        self._log_comparison(metrics_base, metrics_ews)

        return {"baseline": metrics_base, "with_ews": metrics_ews}

    def _log_comparison(self, base: dict, ews: dict) -> None:
        log.info("\n" + "="*65)
        log.info("  COMPARISON: Baseline vs With EWS")
        log.info("="*65)
        log.info(f"  {'Metric':<35} {'Baseline':>12} {'With EWS':>12} {'Delta':>10}")
        log.info("-"*65)
        metrics_to_compare = [
            ("Total Return (%)",       "total_return_pct",          True),
            ("Ann. Return (%)",        "ann_return_pct",            True),
            ("Ann. Volatility (%)",    "ann_volatility_pct",        False),
            ("Sharpe Ratio",           "sharpe_ratio",              True),
            ("Sortino Ratio",          "sortino_ratio",             True),
            ("Calmar Ratio",           "calmar_ratio",              True),
            ("Max Drawdown (%)",       "max_drawdown_pct",          False),
            ("MDD Duration (days)",    "max_drawdown_duration_days",False),
            ("Win Rate (%)",           "win_rate_pct",              True),
            ("VaR 99% (%)",            "var_hist_99_pct",           False),
            ("CVaR 99% (%)",           "cvar_hist_99_pct",          False),
            ("Omega Ratio",            "omega_ratio",               True),
            ("Tail Ratio",             "tail_ratio",                True),
        ]
        for label, key, higher_better in metrics_to_compare:
            bv = base.get(key)
            ev = ews.get(key)
            if bv is None or ev is None:
                continue
            try:
                delta = ev - bv
                sign  = "+" if delta > 0 else ""
                better = (delta > 0) == higher_better
                flag   = "✓" if better else "✗"
                log.info(f"  {label:<35} {bv:>12.4f} {ev:>12.4f} {sign}{delta:>9.4f} {flag}")
            except Exception:
                pass
        log.info("="*65)

        # EWS regime summary
        if "ews_regime_counts" in ews:
            log.info(f"\n  EWS Regime Distribution (With EWS run):")
            log.info(f"    GREEN    (full):      {ews.get('ews_days_green',0):>5} days")
            log.info(f"    YELLOW   (70%):       {ews.get('ews_days_yellow',0):>5} days")
            log.info(f"    ORANGE   (40%):       {ews.get('ews_days_orange',0):>5} days")
            log.info(f"    RED/CRIT (≤20%):      {ews.get('ews_days_red',0):>5} days")
            log.info(f"    Avg scale factor:     {ews.get('ews_avg_scale',1.0):.1%}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _rebalance_schedule(self, dates: list) -> set:
        s = pd.Series(dates, index=dates)
        if self.rebalance_freq == "daily":
            return set(dates)
        elif self.rebalance_freq == "weekly":
            return set(s.resample("W-FRI").last().dropna())
        elif self.rebalance_freq == "monthly":
            return set(s.resample("BME").last().dropna())
        return set(dates)

    def _check_stops(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> None:
        to_close = []
        for sym, pos in portfolio.positions.items():
            if pos.stop_loss <= 0 or sym not in prices:
                continue
            price = prices[sym]
            if pos.quantity > 0 and price <= pos.stop_loss:
                to_close.append((sym, -pos.quantity, price))
            elif pos.quantity < 0 and price >= pos.stop_loss:
                to_close.append((sym, -pos.quantity, price))

        for sym, qty, price in to_close:
            log.info(f"[{date.date()}] STOP LOSS: {sym} qty={qty:.2f} @ {price:.4f}")
            portfolio.execute_order(sym, qty, price, date, self.commission, self.slippage)

    def _compute_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        bench_returns: Optional[pd.Series],
        risk_mgr: RiskManager,
    ) -> dict:
        n = len(returns)
        if n == 0:
            return {}

        ann_factor = PERIODS_PER_YEAR

        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        ann_return = (1 + returns).prod() ** (ann_factor / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor)

        risk_free = 0.04 / ann_factor  # 4% annualised risk-free rate assumption
        sharpe = ((returns.mean() - risk_free) / returns.std() * np.sqrt(ann_factor)
                  if returns.std() > 0 else np.nan)

        downside = returns[returns < 0].std() * np.sqrt(ann_factor)
        sortino = (ann_return - 0.04) / downside if downside > 0 else np.nan

        mdd = RiskManager.max_drawdown(equity)
        mdd_duration = RiskManager.max_drawdown_duration(equity)
        calmar = RiskManager.calmar_ratio(equity, ann_factor)

        daily_wins = (returns > 0).sum()
        win_rate = daily_wins / n

        conf = risk_mgr.var_confidence
        hist_var   = RiskManager.historical_var(returns, conf)
        hist_cvar  = RiskManager.historical_cvar(returns, conf)
        param_var  = RiskManager.parametric_var(returns, conf)
        mc_var     = RiskManager.monte_carlo_var(returns, conf)

        skew       = RiskManager.skewness(returns)
        kurt       = RiskManager.kurtosis(returns)
        omega      = RiskManager.omega_ratio(returns)
        tail_ratio = RiskManager.tail_ratio(returns)

        stress = RiskManager.stress_test(returns)

        alpha = beta = info_ratio = tracking_error = None
        if bench_returns is not None and len(bench_returns) > 10:
            aligned = returns.align(bench_returns, join="inner")
            r_strat, r_bench = aligned[0].dropna(), aligned[1].dropna()
            if len(r_strat) > 10:
                cov = np.cov(r_strat, r_bench)
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
                bench_ann = (1 + r_bench.mean()) ** ann_factor - 1
                alpha = ann_return - (0.04 + beta * (bench_ann - 0.04))
                tracking_error = (r_strat - r_bench).std() * np.sqrt(ann_factor)
                info_ratio = alpha / tracking_error if tracking_error > 0 else np.nan

        return {
            "total_return_pct":          total_return * 100,
            "ann_return_pct":            ann_return * 100,
            "ann_volatility_pct":        ann_vol * 100,
            "trading_days":              n,
            "sharpe_ratio":              sharpe,
            "sortino_ratio":             sortino,
            "calmar_ratio":              calmar,
            "max_drawdown_pct":          mdd * 100,
            "max_drawdown_duration_days": mdd_duration,
            "win_rate_pct":              win_rate * 100,
            "var_hist_99_pct":           hist_var * 100,
            "cvar_hist_99_pct":          hist_cvar * 100,
            "var_parametric_99_pct":     param_var * 100,
            "var_monte_carlo_99_pct":    mc_var * 100,
            "skewness":                  skew,
            "excess_kurtosis":           kurt,
            "omega_ratio":               omega,
            "tail_ratio":                tail_ratio,
            "stress_scenarios":          stress,
            "alpha_ann_pct":             alpha * 100 if alpha is not None else None,
            "beta":                      beta,
            "information_ratio":         info_ratio,
            "tracking_error_pct":        tracking_error * 100 if tracking_error is not None else None,
        }

    def _log_summary(self, metrics: dict) -> None:
        label = metrics.get("run_label", "")
        log.info("=" * 60)
        log.info(f"BACKTEST RESULTS {('— ' + label) if label else ''}")
        log.info("=" * 60)
        for k in ["total_return_pct", "ann_return_pct", "ann_volatility_pct",
                  "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                  "max_drawdown_pct", "max_drawdown_duration_days",
                  "win_rate_pct", "var_hist_99_pct", "cvar_hist_99_pct",
                  "skewness", "excess_kurtosis", "omega_ratio", "tail_ratio",
                  "alpha_ann_pct", "beta", "information_ratio"]:
            v = metrics.get(k)
            if v is not None:
                log.info(f"  {k:<35} {v:>12.4f}")
        log.info("=" * 60)
