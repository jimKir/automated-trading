# strategy/

Signal generation and universe management for the trading system.

## Files

### `signals.py` -- SignalGenerator

16-factor signal engine with regime-conditional blending.

**Reactive factors (per-asset):** F1 TS Momentum, F2 Mean Reversion (z-score entry 2.0/exit 0.5), F3 MACD, F4 RSI, F5 Vol Regime Filter, F6 Cross-Sectional Momentum, F13 PMO Crossover, F14 VWAP Daily Proxy, F15 VWAP-SMA Deviation, F16 Stochastic Contrarian.

**Predictive factor (cross-asset):** F7 Credit Regime (HYG/LQD spread + VIX momentum + yield curve), 30% weight.

**Volume confirmation (post-blend multiplier 0.5-1.3x):** F8 OBV Trend, F9 Volume Trend Ratio, F10 Chaikin Money Flow, F11 H2O Trend Classifier, F12 Price-Volume Segments.

**Regime dispatch:** Bull regime (VIX<20 AND SPY>200MA) uses momentum-heavy weights (TS_MOM=0.60). Bear/neutral enables full factor set including PMO and Stochastic. Gated T3 choppy override shifts to mean-reversion dominant (MR=0.42) when choppy_score >= 0.17.

Key class: `SignalGenerator` with methods `generate()`, `generate_latest()`, `compute_stop_loss()`.

### `universe.py` -- DynamicUniverseSelector + AdaptiveCaps

Selects top 20 instruments monthly from 55 candidates by vol-adjusted 6-month momentum.

- `AdaptiveCaps`: Adjusts equity allocation (60% bear - 90% bull) based on breadth, SPY vs 200d MA, equity vs defensive spread. EWM smoothed to prevent whipsawing.
- `DynamicUniverseSelector`: Ranks candidates, applies asset class caps, returns active universe.
- `DynamicCandidateBuilder`: Fetches S&P 500 + NDX 100 constituents at startup.

### `alpaca_microstructure.py` -- AlpacaDataFetcher

Four microstructure signals from Alpaca 1-min bars: VWAP distance trend, opening gap fill rate, trade intensity, options flow. Requires Alpaca API credentials.

### `databento_imbalance.py` -- NASDAQ Closing Auction

Order imbalance signal from Databento XNAS.ITCH feed (3:50-4:00 PM ET). Weight 0.35 in Databento composite. Anti-lookahead: 1-day shift applied.

### `databento_opening_cross.py` -- NASDAQ Opening Cross

Volume anomaly signal from opening auction. Primary (80%): opening volume anomaly + gap direction. Secondary (20%): closing cross contrarian.

### `databento_options_flow.py` -- OPRA Options Flow

Call/put flow imbalance from Databento OPRA.PILLAR. OTM contracts weighted 1.5x. Weight 0.40 in Databento composite. Graceful degradation if data unavailable.

## Known Limitations

- ADX gate is disabled (hurts Sharpe in all OOS windows)
- Hourly entry timing provides minimal OOS edge (+17 bps)
- Databento signals require paid API access
