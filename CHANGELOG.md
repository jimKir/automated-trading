# Changelog

All notable changes to the automated-trading system are documented here.

## [2026-04-14] — SignalGenerator shape mismatch fix

### Bug: `ValueError: operands could not be broadcast together with shapes (400,) (399,) (399,)`

**Affected path:** Live engine → `SignalGenerator._compute_symbol_signal()` →
PMO/stochastic/regime blending in `strategy/signals.py:862`.

**Root cause:** The PMO factor (`_pmo_crossover`) and stochastic factor
(`_stochastic_contrarian`) could produce pandas Series with N-1 elements
when the High/Low columns fetched from Alpaca's live API had slightly
different index alignment from Close. These shorter Series propagated into
`bear_blend` and `choppy_blend` (which include PMO weight 0.12 + stochastic
weight 0.10), while `bull_blend` (which uses only ts_momentum, mean_reversion,
MACD, and RSI — no PMO/stochastic) kept the full N-length index. The
`np.where(bull_regime, bull_blend, np.where(t3_gate, choppy_blend, bear_blend))`
call then failed because numpy received arrays of shapes (400,), (399,), (399,).

**Impact:**
- **Live engine:** Every trading cycle crashed at signal generation. The engine
  caught the error, logged it, and slept 60s before retrying — but no trades
  could execute.
- **OOS backtest (`run_wf_12m_oos.py`):** NOT affected. The OOS runner uses
  standalone signal functions (`ts_momentum`, `mean_reversion`, `macd_signal`),
  not the `SignalGenerator` class. A full re-run on 2026-04-14 confirmed
  identical results (Sharpe 3.005, CAGR 58.07%, all 4 folds unchanged).

**Fix (3 layers):**

1. **Reindex at source** (commit `3a136f3`): `pmo_sig` and `stoch_sig` are now
   `.reindex(close.index).fillna(0)` immediately after computation, ensuring
   they always match the canonical index length.

2. **Belt-and-suspenders reindex** (commit `3a136f3`): Before the `np.where`
   call, all 5 arrays (`bull_regime`, `t3_gate`, `bull_blend`, `bear_blend`,
   `choppy_blend`) are explicitly reindexed to `close.index` and extracted as
   `.values` numpy arrays.

3. **Shape assertion** (commit `2615015`): A post-reindex check validates all
   arrays match `len(close)`. If any don't, it raises a descriptive
   `ValueError` naming only the mismatched arrays and the symbol, e.g.:
   `"Shape mismatch in _compute_symbol_signal for AAPL: choppy_blend_arr=(399,) expected=400"`

**Tests:**
- `tests/test_leakage_audit.py::TestSignalBlendShapeAlignment` — 2 tests
  verifying reindex alignment and assertion message format (commit `2615015`).
- `tests/test_signal_shape_regression.py` — 4 regression tests calling
  `_compute_symbol_signal` directly with mismatched inputs, consistent inputs,
  PMO/stochastic-specific mismatch, and assertion-fires scenarios.

### Also fixed in this session

- **`ModuleNotFoundError: No module named 'core'`** — Terraform container
  command changed from `python execution/live_engine.py` to `python main.py`
  which has `sys.path` set correctly. Safety-net `sys.path` fix also added to
  `live_engine.py`. (commit `a5b8944`)

- **ECR image tag mismatch** — Paper deploy pushed `paper-latest` but Terraform
  task definition was hardcoded to `:latest` (production). Added `image_tag`
  variable. (commit `b48a2e4`)

- **Alpaca fractional short sells** — `AlpacaBroker.place_order()` now floors
  fractional quantities to whole shares for short sells. (commit `3432313`)

- **Alpaca wash trade rejections** — Engine now cancels conflicting open orders
  before placing opposite-side orders. (commit `3432313`)
