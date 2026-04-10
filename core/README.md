# core/

Portfolio management, cost modeling, volatility targeting, and advanced analytical modules.

## `portfolio.py` -- Portfolio

Position management, target weight computation, trade logging, and equity curve tracking. Integrates with `PortfolioOptimizer` and `CostModel`. Tracks futures roll events.

## `cost_model.py` -- CostModel

6-layer realistic transaction cost model: commission (tiered by asset class), bid-ask spread (half-spread), market impact (square-root model), overnight financing, futures roll costs, crypto funding rates. Typical round-trip: ~0.04% SPY, ~0.12% small-cap ETFs, ~0.02% futures, ~0.10% major crypto.

## `optimizer.py` -- PortfolioOptimizer

Two methods: **Risk Parity** (default, weights inverse to volatility) and **Minimum Variance** (63d covariance with Ledoit-Wolf shrinkage). Crypto hard-capped at 10%. Regime-aware heat scaling with smooth interpolation around SPY 200d MA.

## `vol_targeting.py` -- VolatilityTargeting

Scales portfolio exposure to maintain fixed annual target volatility (~15%). Uses 21-day rolling EWMA (lambda=0.94) or H2O forecaster prediction. Leverage bounds: 0.1x floor, 1.5x cap.

## `crisis_alpha_amplifier.py` -- CrisisAlphaAmplifier

VIX-regime position scaling: CRISIS (VIX>30, rising) = 1.60x, ELEVATED (20-30) = 1.25x, NORMAL (15-20) = 1.00x, SUPPRESSED (<15) = 0.80x. Min 3 days in regime before switching (anti-whipsaw).

## `kelly_sizer.py` -- KellySizer

Fractional Kelly sizing (0.25 fraction) based on rolling 63-day information coefficient (IC). Falls back to raw signal magnitude for symbols with < 63 days history.

## `intraday_shock.py` -- IntradayShockDetector

Runs every 5 minutes (daily proxy in backtest). Four detection mechanisms: VIX spike (>15% = SHOCK, >10% = CAUTION), portfolio equity drop (>3% = SHOCK), volume shock (panic/climactic), plus 5-day gradual recovery ramp.

## `h2o_trend_classifier.py` -- H2OTrendClassifier

H2O AutoML classifier outputting 5 trend states with multipliers: STRONG_UP (1.40x), MILD_UP (1.15x), NEUTRAL (1.00x), MILD_DOWN (0.85x), STRONG_DOWN (0.60x). 28 causal features, monthly walk-forward retraining.

## `h2o_vol_forecaster.py` -- H2OVolForecaster

Predicts next-week realized volatility. Trained on 11,625 pooled weekly observations (18 symbols, 2010-2022). OOS validated 2023-2026: beats EWMA on 6/6 metrics (MAE -6.3%, Bias -90.9%). Quarterly retraining via `python core/h2o_vol_forecaster.py --retrain`. Requires Java 17.

## `price_volume_segments.py` -- PriceVolumeSegments

Wyckoff-inspired momentum quality scorer ([-1, +1]). 12 segment features (6 price, 6 volume). STRONG (+0.8 to +1.0): price rising + volume expanding. REVERSAL (-0.3 to -1.0): volume expanding but price stalling.

## Known Limitations

- H2O modules require Java 17 runtime
- CrisisAlphaAmplifier and KellySizer are standalone modules (wired into LiveEngine but can be disabled)
- Trained H2O model stored in `models/h2o_vol_model/`
