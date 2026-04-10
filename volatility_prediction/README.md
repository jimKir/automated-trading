# volatility_prediction/

Standalone volatility prediction engine with multi-model ensemble approach.

## `vol_engine.py` -- VolatilityPredictionEngine

Production-grade volatility forecasting using an adaptive ensemble of five estimators:

1. **Yang-Zhang** -- Range-based vol estimator (most efficient for OHLC data)
2. **Garman-Klass** -- Range-based, assumes no drift
3. **Parkinson** -- Range-based, uses high-low spread
4. **HAR** -- Heterogeneous AutoRegressive baseline model
5. **Bi-LSTM with Temporal Attention** -- Captures nonlinear dynamics

Ensemble weights adapted by recent walk-forward performance. Evaluated using QLIKE loss.

## `backtest_full.py`

Full walk-forward backtest of the vol prediction engine. Tests forecasting accuracy across all symbols and time periods.

## `backtest_classification.py`

Classification variant -- predicts high/low vol regime labels rather than continuous vol levels.

## `run_vol.py`

CLI runner for training and evaluating the vol prediction engine.

## Requirements

Separate `requirements.txt` in this directory for standalone use. The main `requirements.txt` includes all necessary dependencies.

## Integration

The H2O vol forecaster (`core/h2o_vol_forecaster.py`) is the production module used by `LiveEngine` and `BacktestEngine`. This directory contains the research/exploration code and alternative models.
