# backtest/

Event-driven backtesting engine with reporting and overfitting validation.

## `engine.py` -- BacktestEngine

Daily-frequency event-driven simulator with full module integration (EWS, vol targeting, intraday shock detection, dynamic universe selection).

**Performance metrics computed:** Returns, Sharpe, Sortino, Calmar, Max Drawdown, Historical/Parametric/Monte Carlo VaR, CVaR, Omega Ratio, Tail Ratio, Skew, Kurtosis, stress test scenarios, benchmark comparison (alpha, beta, information ratio).

Key methods: `run()` for single backtest, `run_comparison()` for strategy comparison (e.g., base vs EWS-enhanced).

Default parameters: commission 0.1%, slippage 0.05%, benchmark SPY, rebalance weekly.

## `engine_rebalance_patch.py` -- Adaptive Rebalance Scheduler

Drop-in replacement for `BacktestEngine._rebalance_schedule` with intelligent overrides:

- **Biweekly option:** Signal IC peaks at 10 trading days (not 5)
- **Signal-change skip:** Skip rebalance if signal shift < 0.15 (reduces turnover)
- **VIX-spike forced rebalance:** Force rebalance on +20% VIX day (crisis alpha Sharpe 2.03)
- **Adaptive scheduler:** Biweekly in GREEN, weekly in YELLOW+ (choppy score driven)

## `wf_validator.py` -- Walk-Forward Overfitting Validator

Three independent validation methods -- all must pass for an improvement to be considered genuine:

1. **Expanding walk-forward:** Train on progressively longer periods, test OOS
2. **Sensitivity analysis:** Test across target_vol in {0.10, 0.12, 0.15, 0.18, 0.20} -- genuine improvements are stable
3. **Permutation test:** Shuffle returns 500 times -- improvement must beat random >5%

## `reporter.py` -- Performance Reports

Generates matplotlib performance plots (equity curve, drawdown, distributions) and JSON metrics output. 4x2 plot grid (20x24 figure).
