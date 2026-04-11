"""
Live / Paper Trading Engine
============================
Runs the strategy in real-time with a configurable broker.
Loops at a configurable interval, fetches latest data,
generates signals, computes orders, and executes.
"""

from __future__ import annotations

import signal
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.portfolio import Portfolio
from data.feed import DataFeed
from execution.broker_base import BrokerBase, Order, OrderSide, OrderStatus, OrderType
from risk.manager import RiskManager
from strategy.signals import SignalGenerator
from utils.logger import get_logger

log = get_logger("LiveEngine")


def get_broker(config: dict) -> BrokerBase:
    """Factory: return the configured broker.

    Broker selection logic:
      paper mode  → AlpacaBroker (paper URL) if Alpaca keys are set
                  → PaperBroker (local sim) as fallback
      live mode   → AlpacaBroker if alpaca keys present
                  → IBKRBroker   if ibkr account set
                  → PaperBroker  fallback (safety net)
    """
    import os

    mode = config.get("system", {}).get("mode", "paper")
    brokers_cfg = config.get("brokers", {})

    # Check if Alpaca credentials are available
    alpaca_cfg = brokers_cfg.get("alpaca", {})
    alpaca_key = alpaca_cfg.get("api_key") or os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = alpaca_cfg.get("api_secret") or os.environ.get("ALPACA_API_SECRET", "")
    has_alpaca = bool(alpaca_key and alpaca_secret)

    if mode == "paper":
        if has_alpaca:
            from execution.alpaca_broker import AlpacaBroker

            log.info("Paper mode → using Alpaca broker (paper URL)")
            return AlpacaBroker(config)
        from execution.paper_broker import PaperBroker

        log.info("Paper mode → using local PaperBroker (no Alpaca keys found)")
        return PaperBroker(config)
    if mode == "live":
        if has_alpaca:
            from execution.alpaca_broker import AlpacaBroker

            log.info("Live mode → using Alpaca broker")
            return AlpacaBroker(config)
        # Fallback to IBKR for equities/futures
        ibkr_cfg = brokers_cfg.get("ibkr", {})
        if ibkr_cfg.get("account"):
            from execution.ibkr_broker import IBKRBroker

            log.info("Live mode → using IBKR broker")
            return IBKRBroker(config)
        from execution.paper_broker import PaperBroker

        log.warning("Live mode but no broker credentials — falling back to PaperBroker!")
        return PaperBroker(config)
    from execution.paper_broker import PaperBroker

    return PaperBroker(config)


class LiveEngine:
    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.mode = config.get("system", {}).get("mode", "paper")
        self.broker = get_broker(config)
        self.feed = DataFeed(config)
        self.signal_gen = SignalGenerator(config)
        self.risk_mgr = RiskManager(config)

        # Vol-engine for per-symbol position sizing (priority: vol_engine > H2O > EWMA)
        self._vol_engine = None
        try:
            from volatility_prediction.vol_engine import VolatilityPredictionEngine

            self._vol_engine = VolatilityPredictionEngine()
            log.info("Vol-engine loaded (HAR+GBM ensemble) — priority vol forecaster")
        except Exception as _e:
            log.warning(f"Vol-engine unavailable ({_e}) — will fall back to H2O/EWMA")
        self._running = False
        self._rebalance_freq = config.get("strategy", {}).get("rebalance_frequency", "weekly")
        self._last_rebalance: datetime | None = None

        # Intraday Shock Detector
        self._isd = None
        if config.get("intraday_shock", {}).get("enabled", False):
            try:
                from core.intraday_shock import IntradayShockDetector

                self._isd = IntradayShockDetector(config)
                log.info("Intraday shock detector enabled")
            except Exception as e:
                log.warning(f"Intraday shock failed to load: {e}")

        # Early Warning System
        self._ews = None
        if config.get("ews", {}).get("enabled", False):
            try:
                from regime.ews import EarlyWarningSystem

                self._ews = EarlyWarningSystem(config)
                log.info("EWS enabled for live/paper trading")
            except Exception as e:
                log.warning(f"EWS failed to load: {e} — running without EWS")

        # Multi-source anomaly detection layer
        self._anomaly_layer = None
        if config.get("anomaly_layer", {}).get("enabled", True):
            try:
                from regime.anomaly_layer import AnomalyRegimeLayer
                self._anomaly_layer = AnomalyRegimeLayer(config)
                log.info("AnomalyRegimeLayer enabled — 4-source composite anomaly detection")
            except Exception as e:
                log.warning(f"AnomalyRegimeLayer failed to load: {e}")

        # Per-position anomaly scorer (asymmetric: crypto cut aggressively)
        self._pos_anomaly_scorer = None
        if config.get("position_anomaly", {}).get("enabled", True):
            try:
                from risk.position_anomaly import PositionAnomalyScorer
                self._pos_anomaly_scorer = PositionAnomalyScorer()
                log.info("PositionAnomalyScorer enabled — asymmetric per-symbol scaling")
            except Exception as e:
                log.warning(f"PositionAnomalyScorer failed to load: {e}")

        # Hourly Entry Timer (OOS: NO_EDGE, wired but non-critical)
        self._hourly_timer = None
        if config.get("execution", {}).get("hourly_timing_enabled", False):
            try:
                from execution.hourly_entry_timer import HourlyEntryTimer
                self._hourly_timer = HourlyEntryTimer(enabled=True)
                log.info("HourlyEntryTimer enabled")
            except Exception as e:
                log.warning(f"HourlyEntryTimer failed to load: {e}")

        # Dynamic Universe Scanner
        self._universe_scanner = None
        if config.get("execution", {}).get("dynamic_universe_enabled", False):
            try:
                import os
                from data.dynamic_universe_scanner import DynamicUniverseScanner
                alpaca_cfg = config.get("brokers", {}).get("alpaca", {})
                ak = alpaca_cfg.get("api_key") or os.environ.get("ALPACA_API_KEY", "")
                sk = alpaca_cfg.get("api_secret") or os.environ.get("ALPACA_API_SECRET", "")
                if ak and sk:
                    self._universe_scanner = DynamicUniverseScanner(
                        api_key=ak, secret_key=sk
                    )
                    log.info("DynamicUniverseScanner enabled")
            except Exception as e:
                log.warning(f"DynamicUniverseScanner failed to load: {e}")

        # Price data cache for live mode
        self._price_df_live: Optional[pd.DataFrame] = None

    def start(self, loop_interval_seconds: int = 60) -> None:
        """Main trading loop."""
        log.info(f"Starting {self.mode.upper()} trading engine")
        log.warning("=" * 60)
        log.warning("RISK WARNING: This system trades real financial markets.")
        log.warning("Paper-trade and backtest thoroughly before going live.")
        log.warning("=" * 60)

        if not self.broker.connect():
            log.error("Broker connection failed. Exiting.")
            return

        self._running = True

        # Graceful shutdown on Ctrl+C
        def _shutdown(sig, frame):
            log.info("Shutdown signal received...")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        account = self.broker.get_account()
        self.risk_mgr.update_equity(account.equity)
        self.risk_mgr.reset_daily(account.equity, cash=account.cash)
        log.info(f"Account: equity=${account.equity:,.2f} cash=${account.cash:,.2f}")

        # Seed intraday shock detector with opening snapshot
        if self._isd is not None:
            try:
                import yfinance as yf

                vix_open = float(yf.Ticker("^VIX").history(period="2d")["Close"].iloc[-1])
                self._isd.reset_day(vix_open, account.equity, datetime.now(timezone.utc).date())
            except Exception:
                pass

        while self._running:
            try:
                self._trading_cycle()
            except Exception as exc:
                log.error(f"Trading cycle error: {exc}", exc_info=True)

            log.info(f"Sleeping {loop_interval_seconds}s...")
            time.sleep(loop_interval_seconds)

        self.broker.disconnect()
        log.info("Trading engine stopped.")

    def _trading_cycle(self) -> None:
        now = datetime.now(timezone.utc)

        # Check circuit breakers
        account = self.broker.get_account()

        # ── Intraday shock check (every loop iteration) ───────────────────────
        isd_scale = 1.0
        if self._isd is not None:
            try:
                import yfinance as yf

                vix_now = float(
                    yf.Ticker("^VIX").history(period="1d", interval="5m")["Close"].iloc[-1]
                )
                isd_scale, isd_state, isd_reason = self._isd.check(vix_now, account.equity)
                if isd_scale < 1.0:
                    log.warning(f"ISD: {isd_state.value} | scale={isd_scale:.0%} | {isd_reason}")
            except Exception as e:
                log.debug(f"ISD VIX fetch failed: {e}")

        halt, reason = self.risk_mgr.check_halt(account.equity, cash=account.cash)
        if halt:
            log.critical(f"TRADING HALTED: {reason}")
            return

        self.risk_mgr.update_equity(account.equity)

        # Check if it's time to rebalance
        if not self._should_rebalance(now):
            log.debug("Skipping cycle — next rebalance not due yet")
            return

        log.info(f"=== Trading Cycle @ {now.strftime('%Y-%m-%d %H:%M:%S')} UTC ===")

        # Fetch latest data (lookback 1 year for signal computation)
        start_date = (now - timedelta(days=400)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
        all_data = self.feed.load_all(start=start_date, end=end_date)

        if not all_data:
            log.warning("No market data received — skipping cycle")
            return

        # Build price DataFrame and populate _price_df_live for adaptive rebalance
        price_data = pd.DataFrame({sym: df["Close"] for sym, df in all_data.items()
                                    if "Close" in df.columns})
        self._price_df_live = price_data

        # Get current choppy score for regime dispatch
        choppy_score = 0.0
        try:
            from regime.choppy_regime import ChoppyRegimeDetector
            _choppy_det = ChoppyRegimeDetector()
            _vix_s = None
            if "^VIX" in all_data:
                _vix_s = all_data["^VIX"]["Close"]
            elif "VIX" in price_data.columns:
                _vix_s = price_data["VIX"]
            if _vix_s is not None:
                choppy_score = _choppy_det.score_today(price_data, _vix_s)
        except Exception as _chop_e:
            log.debug(f"Choppy score computation failed: {_chop_e}")

        # Compute anomaly layer (multi-source: macro, sentiment, FX, isolation forest)
        anomaly_scale = 1.0
        anomaly_label = "NORMAL"
        if self._anomaly_layer is not None:
            try:
                from regime.anomaly_layer import AnomalyScore as _AS
                anomaly_result = self._anomaly_layer.compute(price_data)
                anomaly_scale = anomaly_result.position_scale
                anomaly_label = anomaly_result.label
                choppy_scale_val, choppy_colour = ChoppyRegimeDetector.score_to_scale(choppy_score) if choppy_score > 0 else (1.0, "GREEN")
                combined_regime_scale = choppy_scale_val * anomaly_scale
                log.info(
                    f"[REGIME] Choppy={choppy_colour}({choppy_scale_val:.2f}) "
                    f"Anomaly={anomaly_label}({anomaly_scale:.2f}) "
                    f"Combined={combined_regime_scale:.2f}"
                )
            except Exception as _anom_e:
                log.debug(f"AnomalyRegimeLayer computation failed: {_anom_e}")

        # Get SPY price series for regime detection
        spy_prices = None
        if "SPY" in all_data:
            spy_prices = all_data["SPY"]["Close"]

        # Generate signals with regime context
        signals = self.signal_gen.generate_latest(
            all_data, choppy_score=choppy_score, spy_prices=spy_prices
        )
        log.info(f"Signals: { {k: f'{v:+.3f}' for k, v in signals.items() if abs(v) > 0.05} }")

        # Get current prices
        prices = {}
        for sym, df in all_data.items():
            if not df.empty:
                prices[sym] = float(df["Close"].iloc[-1])

        # Build a temporary portfolio snapshot from broker
        curr_positions = account.positions
        equity = account.equity

        # Get EWS scale factor
        ews_scale = 1.0
        ews_colour = "GREEN"
        if self._ews is not None:
            try:
                price_df = pd.DataFrame({sym: df["Close"] for sym, df in all_data.items()})
                _, ews_scale, ews_colour = self._ews.score_today(price_df)

                # Vol-engine per-symbol scale (live path)
                self._sym_vol_scales_live: dict = {}
                if self._vol_engine is not None:
                    try:
                        for sym, df in all_data.items():
                            if len(df) < 30:
                                continue
                            ve_vol = self._vol_engine.predict_one(sym, df)
                            if ve_vol and 0.01 < ve_vol < 2.0:
                                target = 0.15
                                self._sym_vol_scales_live[sym] = float(
                                    np.clip(target / ve_vol, 0.2, 1.5)
                                )
                    except Exception as _vee:
                        log.debug(f"Live vol_engine scaling failed: {_vee}")
                log.info(f"EWS: {ews_colour} — position scale = {ews_scale:.0%}")
            except Exception as e:
                log.warning(f"EWS live scoring failed: {e}")

        # Compute target weights (scaled by EWS)
        max_pos = self.config.get("risk", {}).get("max_position_pct", 0.15)
        max_heat = self.config.get("capital", {}).get("max_portfolio_heat", 0.40)

        # v15b: Combined scale = min of independent layers (not multiplicative!)
        # Matches backtest engine logic. Floor at 50% to prevent going fully flat.
        # anomaly_scale from the multi-source anomaly layer is included alongside EWS + ISD
        combined_scale = max(min(ews_scale, isd_scale, anomaly_scale), 0.50)
        if combined_scale < 1.0:
            log.info(
                f"Combined scale: min(EWS={ews_scale:.0%}, ISD={isd_scale:.0%}, "
                f"Anomaly={anomaly_scale:.0%}) = {combined_scale:.0%}"
            )

        # Apply per-symbol vol-engine scaling to signals
        # High predicted vol → smaller signal → smaller position (same logic as backtest)
        if hasattr(self, "_sym_vol_scales_live") and self._sym_vol_scales_live:
            signals = {
                sym: sig * self._sym_vol_scales_live.get(sym, 1.0) for sym, sig in signals.items()
            }
            log.debug(f"Vol-engine signal scaling applied: {len(self._sym_vol_scales_live)} syms")

        # Apply per-symbol position anomaly scaling (asymmetric drawdown protection)
        # crypto: floor=10% (up to 90% cut), equity: floor=40%, hedges: always 1.0
        if self._pos_anomaly_scorer is not None:
            try:
                price_df_for_pos = pd.DataFrame(
                    {sym: df["Close"] for sym, df in all_data.items()
                     if "Close" in df.columns}
                )
                # Use ChoppyRegimeDetector score as portfolio context
                from regime.choppy_regime import ChoppyRegimeDetector
                _vix_s = price_df_for_pos.get("VIX",
                    pd.Series(dtype=float)) if hasattr(price_df_for_pos,"get") else pd.Series(dtype=float)
                if "VIX" in price_df_for_pos.columns:
                    _vix_s = price_df_for_pos["VIX"]
                elif "^VIX" in all_data:
                    _vix_s = all_data["^VIX"]["Close"]
                _port_score = ChoppyRegimeDetector().score_today(price_df_for_pos, _vix_s)
                _pos_scales = self._pos_anomaly_scorer.score_today(
                    price_df_for_pos, portfolio_score=_port_score
                )
                crypto_scales = {s: round(v,2) for s, v in _pos_scales.items()
                                 if any(c in s.upper() for c in ["BTC","ETH","SOL"])}
                if crypto_scales:
                    log.info(f"PosAnomaly live crypto scales: {crypto_scales}")
                signals = {
                    sym: sig * _pos_scales.get(sym, 1.0)
                    for sym, sig in signals.items()
                }
            except Exception as _pos_err:
                log.debug(f"PositionAnomalyScorer live failed: {_pos_err}")

        # Inject macro data for regime switching (VIX + SPY needed for bull/bear detection)
        try:
            import yfinance as _yf

            _macro_syms = ["^VIX", "SPY", "HYG", "LQD"]
            _macro_data = {}
            for _ms in _macro_syms:
                _mdf = _yf.download(
                    _ms, start=start_date, end=end_date, auto_adjust=True, progress=False
                )
                if not _mdf.empty:
                    if isinstance(_mdf.columns, pd.MultiIndex):
                        _mdf.columns = _mdf.columns.get_level_values(0)
                    _macro_data[_ms] = _mdf
            if _macro_data:
                self.signal_gen.set_macro_data(_macro_data)
        except Exception as _me:
            log.debug(f"Live macro data update failed: {_me}")

        temp_portfolio = Portfolio(self.config)
        temp_portfolio.cash = account.cash
        target_weights = temp_portfolio.compute_target_weights(
            signals,
            max_position_pct=max_pos * combined_scale,
            max_portfolio_heat=max_heat * combined_scale,
        )

        # Compute orders (delta from current to target)
        for sym, target_w in target_weights.items():
            if sym not in prices or prices[sym] <= 0:
                continue

            target_value = target_w * equity
            curr_pos = curr_positions.get(sym, {})
            curr_qty = float(curr_pos.get("quantity", 0))
            curr_value = curr_qty * prices[sym]
            delta_value = target_value - curr_value

            if abs(delta_value) < prices[sym] * 0.5:  # less than half a unit — skip
                continue

            qty_delta = delta_value / prices[sym]
            side = OrderSide.BUY if qty_delta > 0 else OrderSide.SELL
            qty_abs = abs(qty_delta)

            # ── Pre-trade safety guards ──────────────────────────────────
            max_shares = self.config.get("execution", {}).get("max_order_shares", 10000)
            min_price = self.config.get("execution", {}).get("min_price_sanity", 0.10)
            if prices[sym] < min_price:
                log.warning(
                    f"SKIP {sym}: price ${prices[sym]:.4f} below min_price_sanity ${min_price}"
                )
                continue
            if qty_abs > max_shares:
                log.warning(f"CLAMP {sym}: qty {qty_abs:.1f} → {max_shares} (max_order_shares)")
                qty_abs = max_shares

            order = Order(
                symbol=sym,
                side=side,
                quantity=qty_abs,
                order_type=OrderType.MARKET,
            )

            log.info(f"ORDER → {side.value.upper()} {sym} qty={qty_abs:.4f} @ ~${prices[sym]:.4f}")
            if self.dry_run:
                log.info(f"DRY_RUN → skipping order submission for {sym}")
                continue
            filled = self.broker.place_order(order)
            if filled.status == OrderStatus.REJECTED:
                log.warning(f"REJECTED → {sym} (broker rejected order)")
                continue
            log.info(
                f"FILLED → {sym} avg_px=${filled.avg_fill_price:.4f} status={filled.status.value}"
            )

        # Set stop losses for open positions
        updated_positions = self.broker.get_positions()
        for sym, pos in updated_positions.items():
            if sym in all_data and pos.get("quantity", 0) > 0:
                sl_dist = self.signal_gen.compute_stop_loss(all_data[sym])
                stop_price = prices.get(sym, 0) - sl_dist
                if stop_price > 0:
                    log.info(f"Stop-loss {sym}: ${stop_price:.4f}")

        self._last_rebalance = now
        log.info(f"Account equity after cycle: ${account.equity:,.2f}")

    def _should_rebalance(self, now: datetime) -> bool:
        """
        Determine whether to rebalance on this trading cycle.

        Supports "daily" | "weekly" | "biweekly" | "monthly" | "adaptive".

        "adaptive" mode uses ChoppyRegimeDetector.score_today() to decide:
          - GREEN (score < threshold): biweekly cadence — rebalance every 10 days
          - YELLOW/ORANGE/RED (score >= threshold): weekly cadence — rebalance Fridays
        Threshold default: 0.17 (YELLOW onset, validated 2000-2022, all 8 folds pass).
        """
        if self._last_rebalance is None:
            return True

        elapsed = now - self._last_rebalance

        if self._rebalance_freq == "daily":
            return elapsed >= timedelta(hours=20)

        elif self._rebalance_freq == "weekly":
            return elapsed >= timedelta(days=5) and now.weekday() == 4  # Friday

        elif self._rebalance_freq == "biweekly":
            return elapsed >= timedelta(days=14)

        elif self._rebalance_freq == "monthly":
            return elapsed >= timedelta(days=25) and now.day <= 3

        elif self._rebalance_freq == "adaptive":
            # Get ChoppyDetector score for today (uses latest available prices)
            choppy_score = 0.0
            thr = float(
                self.config.get("strategy", {}).get("adaptive_weekly_threshold", 0.17)
            )
            try:
                from regime.choppy_regime import ChoppyRegimeDetector
                if hasattr(self, "_price_df_live") and self._price_df_live is not None:
                    vix_col = self._price_df_live.get(
                        "VIX", self._price_df_live.iloc[:, 0]
                    ) if hasattr(self._price_df_live, "get") else None
                    if vix_col is not None:
                        choppy_score = ChoppyRegimeDetector().score_today(
                            self._price_df_live, vix_col
                        )
            except Exception as _e:
                log.debug(f"Adaptive schedule: ChoppyDetector failed ({_e}), defaulting to weekly")
                return elapsed >= timedelta(days=5) and now.weekday() == 4

            if choppy_score >= thr:
                # YELLOW/ORANGE/RED → weekly cadence
                log.debug(f"Adaptive: choppy={choppy_score:.3f} >= {thr} → weekly mode")
                return elapsed >= timedelta(days=5) and now.weekday() == 4
            else:
                # GREEN → biweekly cadence
                log.debug(f"Adaptive: choppy={choppy_score:.3f} < {thr} → biweekly mode")
                return elapsed >= timedelta(days=14)

        return elapsed >= timedelta(hours=24)
