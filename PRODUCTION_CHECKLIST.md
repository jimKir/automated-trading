# Production Readiness Checklist

**Date:** 2026-04-10
**Strategy:** Multi-Asset Momentum + Mean Reversion
**ChoppyDetector:** v2 (7 groups, 20 features) + v4 order flow layer wired
**Universe:** 10 instruments (SPY, QQQ, IWM, GLD, TLT, SHY, XLU, XLP, BTC-USD, ETH-USD)

## Modules
- [x] SignalEngine — IS-validated, pre-VWAP bull blend (VWAP-SMA removed: NOISE OOS verdict)
- [x] ChoppyDetector v2 — credit + commodity/FX + breadth + sentiment groups, IS-calibrated thresholds
- [x] OrderFlowAnomalyDetector — Group 9 wired, daily OHLCV proxy for order flow
- [x] PositionAnomalyScorer — WF-calibrated, crypto floor 0.10, equity floor 0.40
- [x] HourlyEntryTimer — 12:00 ET rule for SPY/QQQ/IWM, session windows for BTC/ETH
- [x] DynamicUniverseScanner — Alpaca Screener API, 8 hard filters, max 3 names
- [x] LiveEngine — all modules wired, adaptive rebalance, paper mode tested, dry_run flag

## Performance (OOS baselines)
- ChoppyDetector v2 full-history: mean=0.209, p95=0.358, score range [0.09, 0.60]
- Current regime (2026-04-02): YELLOW (score=0.246, scale=80%)
- SignalEngine: 53 instruments processed, 0 NaN signals
- PositionAnomalyScorer: crypto scales 0.88-0.98, equity 0.87-0.89, hedges 1.00

## Go/No-Go Threshold
- Paper trading: START NOW (all critical systems wired and tested)
- Live capital: WAIT for 12 months paper trading with Sharpe > 0.50 and drawdown episode survived

## Key Config
- Rebalance: adaptive (biweekly GREEN, weekly YELLOW+)
- Max single-stock weight: 8% (dynamic names)
- Crypto floor exposure: 10% minimum when signals positive
- ChoppyDetector gate: ORANGE → 50% scale, RED → 25% scale
- Bull weights: ts_mom=0.50, mr=0.15, macd=0.30, rsi=0.05 (sum=1.00)

## Known Limitations
- VWAP Factor 15: rejected OOS (NOISE verdict), not active in bull blend
- Hourly entry timing: NO_EDGE OOS, wired but adds minimal value on daily strategy
- HYG/LQD/USO/XLE: rejected as portfolio holdings, HYG used as detector input only
- XLU/XLP/USO parquet data not in local store (symbols in config universe)
- Order flow layer value only measurable in live mode (daily proxy insufficient)
- Alpaca API credentials not validated in CI (graceful degradation confirmed)

## Issues Fixed During Dress Rehearsal
1. **LiveEngine `self._config` bug** — changed to `self.config` in `_should_rebalance()`
2. **LiveEngine missing `dry_run` flag** — added constructor parameter and order submission guard
3. **LiveEngine missing subsystems** — wired HourlyEntryTimer and DynamicUniverseScanner
4. **Missing module: `regime/order_flow_anomaly.py`** — created Group 9 order flow detector
5. **Missing module: `regime/calibrate_choppy.py`** — created calibration utility
6. **Missing module: `execution/hourly_entry_timer.py`** — created intraday entry timing
7. **Missing module: `data/dynamic_universe_scanner.py`** — created Alpaca-based scanner
8. **VWAP-SMA in bull blend** — removed per NOISE OOS verdict, restored original weights
9. **regime_params_validated.json** — added `choppy_thresholds_v4` key
10. **LiveEngine `_price_df_live` missing** — added initialization in constructor

## Tags
- `v1.0.0-paper-baseline` — IS-validated weights
- `choppy-v3` — credit features
- `choppy-v4` — order flow layer
- `dynamic-universe-v1` — Alpaca Screener integration
- `universe-v2` — commodity/credit extension (rejected)
- `prod-ready-v1` — this dress rehearsal
