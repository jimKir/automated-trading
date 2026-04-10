# regime/

Regime detection and Early Warning System (EWS) for defensive position scaling.

## ChoppyRegimeDetector v2 (`choppy_regime.py`)

Dedicated anomaly detector for 2025-style high-vol/low-conviction markets. Produces a single score (0-1) mapped to position scale factors.

**7 Feature Groups** (20 features total, weights sum to 1.0):

| Group | Weight | Features |
|---|---|---|
| A: vol_spike | 0.18 | SPY/QQQ volume spike frequency, 5d surge ratio |
| B: price_vol | 0.18 | Vol-of-vol, MA crossing rate, return reversal rate |
| C: macro_credit | 0.16 | HYG/LQD 30d change, HYG 20d return, TLT 10d momentum |
| D: event_shock | 0.16 | VIX 10d std, VIX above-20 rate, VIX 5d velocity |
| E: commodity_fx | 0.12 | Gold/SPY ratio, oil velocity, DXY momentum |
| F: breadth | 0.12 | % above 50d/200d MA, SPY/TLT correlation |
| G: sentiment | 0.08 | VIX/realized vol ratio, VIX vs 60d mean |

**Thresholds:** GREEN <0.17 (1.0x), YELLOW 0.17-0.27 (0.8x), ORANGE 0.27-0.40 (0.5x), RED >0.40 (0.25x).

Features calibrated on 2024 calm (baseline) vs 2025 stress (ceiling), 5-day EMA smoothed.

## Early Warning System (`ews.py`)

Orchestrates 6 independent layers into a composite risk score:

| Layer | Weight | Module | Method |
|---|---|---|---|
| A: Anomaly | 0.30 | `anomaly.py` | Isolation Forest on position-level features |
| B: Macro | 0.25 | `macro_score.py` | FRED yield curve, credit spreads, VIX, DXY |
| C: Event Shock | 0.15 | `event_shock.py` | VIX velocity, term structure, breadth collapse |
| D: Commodity/FX | 0.10 | `commodity_fx.py` | Oil, gold/SPY, copper, DXY, JPY, EUR |
| E: Intraday | 0.05 | `intraday_regime.py` | SPY ADX + EMA + VIX level |
| F: Choppy | 0.15 | `choppy_regime.py` | ChoppyRegimeDetector v2 |

EWS thresholds: GREEN <0.25 (1.0x), YELLOW 0.25-0.40 (0.7x), ORANGE 0.40-0.55 (0.4x), RED 0.55-0.70 (0.2x), CRITICAL >0.70 (0.05x).

## Other Files

- `order_flow_anomaly.py`: Standalone volume/close-position anomaly detector (3 features). Built for planned ChoppyDetector v4 as Group 9.
- `calibrate_choppy.py`: Calibration utility. Profiles score distributions across reference periods (2020 COVID, 2022 bear, 2024 calm, 2025 choppy, 2026 Q1 tariff). Generates v4 thresholds to `data/regime_params_validated.json`.
- `anomaly.py`: Isolation Forest (retrained every 21 days on 756-day window, 5% contamination).
- `macro_score.py`: FRED data with cascade fallback (FRED API -> pandas_datareader -> yfinance proxy -> zeros).
- `event_shock.py`: VIX velocity, term structure, put-call ratio, breadth, cross-asset shock.
- `commodity_fx.py`: Oil (CL=F), gold (GC=F), copper (HG=F), DXY, JPY, EUR, EEM.
- `intraday_regime.py`: 6 regime labels (TRENDING_UP through HIGH_FEAR) from SPY hourly bars.

## Design Principles

- No single layer causes full exposure reduction (confirmation required)
- All features strictly causal (no look-ahead)
- Scale factor never reaches zero (always maintains some position)
