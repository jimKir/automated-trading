# risk/

Risk management: per-instrument anomaly scaling and portfolio-level risk controls.

## `position_anomaly.py` -- PositionAnomalyScorer

Per-instrument anomaly scale factors (0 to 1) that asymmetrically cut exposure based on asset class.

**Asset class sensitivity:**

| Class | Symbols | Sensitivity | Floor | Max Cut |
|---|---|---|---|---|
| CRYPTO | BTC-USD, ETH-USD, SOL-USD | 1.40 | 0.10 | 90% |
| EQUITY | Individual stocks | 0.50 | 0.40 | 60% |
| ETF_EQUITY | SPY, QQQ, IWM, etc. | 0.50 | 0.55 | 45% |
| ETF_HEDGE | TLT, GLD, SHY, AGG | 0.00 | 1.00 | Never cut |
| COMMODITY | CL=F, GC=F | 0.50 | 0.35 | 65% |

**4 features per symbol:** G1 Vol Spike (20d/60d ratio), G2 Momentum Churn (regime-relative TNR z-score), G3 Drawdown from 20d Peak, G4 Portfolio Stress (ChoppyDetector score).

Scores are EMA-smoothed over 3 days with per-class baselines subtracted to ignore chronic noise (crypto baseline=0.16).

Key methods: `score_today()` (live), `score_day()` (backtest), `apply_position_scales()`.

## `manager.py` -- RiskManager

Portfolio-level risk controls, position sizing, and circuit breakers.

**Progressive drawdown scaling** (replaces hard halts):
- DD < 8%: 1.0x (full)
- DD 8-15%: linear to 0.5x
- DD 15-25%: linear to 0.2x
- DD > 25%: 0.2x floor

**Circuit breakers:** Daily realized loss > 8% of equity triggers trading halt. Max DD logged but uses progressive scaling instead of hard halt.

**Position sizing:** Fractional Kelly (0.25 fraction) and volatility-based sizing. Portfolio heat cap (max 75%).

**Risk metrics:** Historical/Parametric/Monte Carlo VaR, CVaR, Omega ratio, Tail ratio, Skewness, Kurtosis, Calmar ratio, stress test scenarios.

## Integration in LiveEngine

```
signals -> PositionAnomalyScorer.score_today() -> per-symbol scale
signals = {sym: sig * scale[sym] for sym, sig in signals.items()}
target_weights = Portfolio.compute_target_weights(scaled_signals)
target_weights *= min(ews_scale, isd_scale) * dd_scale
```
