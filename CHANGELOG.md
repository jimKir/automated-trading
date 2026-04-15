# Changelog

All notable changes to the automated-trading system are documented here.

## [2026-04-15] — Runtime anomaly detector with 7 statistical checks + email alerting

### New: `monitoring/anomaly_detector.py`

Runtime anomaly detector that monitors trading behaviour with 7 statistical health
checks, running each cycle (~60s). Designed to catch crash-loop bugs, excessive
churn, and risk-limit breaches before they cause real damage.

**7 anomaly checks:**
1. **Order Frequency Z-score** (z > 3.0) — detects burst ordering from crash-loops
2. **Daily Portfolio Turnover** (> 1.0) — catches churn for a swing/position trader
3. **Round-Trip Detection** (> 2 per symbol/hour) — catches buy-sell flipping
4. **Signal Flip Rate** (> 3 flips per symbol/24h) — catches unstable signals
5. **Drawdown Velocity** (> 2% per hour) — catches cascading losses
6. **Position Concentration HHI** (> 0.25) — catches allocation bugs
7. **Duplicate Order Detection** (same sym+side+qty within 5 min) — catches ECS overlap

All thresholds are configurable via the `monitoring` section in config.yaml.

### New: `monitoring/alerting.py`

Email alerting via stdlib `smtplib` + `email.mime` (no external deps):
- HTML formatted alert emails with failed checks, account snapshot, version, recent orders
- SMTP config via env vars: `ALERT_SMTP_HOST`, `ALERT_SMTP_PORT`, `ALERT_SMTP_USER`,
  `ALERT_SMTP_PASS`, `ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO`
- Alert throttling: max 1 email per anomaly type per hour (configurable cooldown)
- Recovery emails when checks pass after failing
- Graceful no-op when SMTP is not configured

### Modified: `execution/live_engine.py`

- Init `AnomalyDetector` and `AlertManager` in `__init__` (skip gracefully if disabled)
- Record equity, signals, and orders during `_trading_cycle` for anomaly tracking
- Run all 7 checks after each trading cycle
- On anomaly: generate structured health report and send alert email
- On recovery: send RESOLVED email

### Tests (30+ in `tests/test_anomaly_detector.py`)

- Each of the 7 checks tested individually (pass + fail cases)
- Alert throttling: second alert within cooldown is suppressed
- Recovery email: sent when checks pass after failing
- Graceful behaviour: no crash when SMTP not configured
- Config loading: default thresholds, custom overrides, env var precedence

---

## [2026-04-15] — Startup instance guard + version stamping

### Feature 1: Startup Instance Guard

On process start, before any trading logic, the engine now cancels ALL open/pending
Alpaca orders to clear stale orders from crashed instances. This is broker-level
cleanup — works regardless of how the process starts or restarts.

- New method: `AlpacaBroker.cancel_all_open_orders()` — calls `trading_client.cancel_orders()`
  to cancel everything in one API call, returns count of cancelled orders
- New method: `LiveEngine._cleanup_stale_orders()` — called in `__init__` before any
  trading cycle, logs a warning if stale orders were found

### Feature 2: Version Stamping

Build metadata is now embedded in the running container and logged on startup.

- New file: `version.py` — resolves version from: (a) `BUILD_VERSION` env var,
  (b) git SHA via subprocess, (c) `"dev-unknown"` fallback
- `main.py` logs the version at startup
- `LiveEngine.__init__` prints a startup banner showing: version, build timestamp,
  mode (paper/live), rebalance cadence, and last rebalance time from Alpaca seed
- `Dockerfile` accepts `BUILD_SHA`, `BUILD_TIMESTAMP`, `BUILD_VERSION` as ARGs, sets as ENV vars
- `.github/workflows/ci.yml` passes these build args during `docker build` steps
  in CI, production deploy, and paper deploy jobs

### Tests (11 new in test_instance_guard.py)

- `TestCancelAllOpenOrders` — 4 tests: cancels and returns count, zero on empty,
  zero on None, zero on exception
- `TestCleanupStaleOrdersOnInit` — 3 tests: called during init, tolerates broker
  without method, exception does not crash init
- `TestVersionResolution` — 4 tests: env var priority, git SHA fallback,
  dev-unknown fallback, empty git output falls through

---

## [2026-04-15] — Fix 486 day trades: rebalance cadence + duplicate order guard

### Investigation: 486 day trades in ~2 days triggered PDT flag

The paper trading bot executed 486 day trades over ~2 days (Apr 13-14), causing
PDT flagging on the Alpaca paper account. Order analysis revealed 1503 orders
with massive churn: GLD 163 orders, XLE 168, EEM 157, IWM 150, VGK 132.

**Two root causes identified and fixed:**

### Bug 1: `_should_rebalance()` returned True every cycle (~60s) instead of ~10 days

**Root cause:** `_last_rebalance` was initialised to `None` in `LiveEngine.__init__`
and only set in-memory at end of `_trading_cycle()`. Every ECS container restart
reset it to `None`, causing line 556 (`if self._last_rebalance is None: return True`)
to fire on every cycle. With crash-loop restarts, the engine traded every ~60 seconds
instead of respecting the adaptive cadence (~10 days in GREEN regime).

**Fix:** On startup, `LiveEngine.__init__` now queries `AlpacaBroker.get_last_filled_order_time()`
to seed `_last_rebalance` from the most recent filled order. This survives container
restarts without external storage — Alpaca's order history is the source of truth.

- New method: `AlpacaBroker.get_last_filled_order_time()` — fetches most recent
  closed order's `filled_at` timestamp via the Alpaca API
- `LiveEngine.__init__` now calls this on startup to seed `_last_rebalance`

### Bug 2: 57 exact duplicate orders from overlapping ECS task instances

**Root cause:** During crash-loop restarts on Apr 14, multiple ECS tasks ran
simultaneously between 20:01-20:44 UTC. Each instance submitted the same orders
independently, resulting in 57 exact duplicates (same symbol + same side + same
quantity in the same minute).

**Fix:** Added an explicit duplicate order guard in `_trading_cycle()` before
`place_order()`. For each symbol, the engine now calls `broker.get_open_orders(sym)`
and skips if any open/pending orders exist for the same symbol and side. This is
defense-in-depth on top of the existing `cancel_conflicting_orders()` wash-trade
prevention.

### Tests (13 new, 25 total in test_order_guards.py)

- `TestDuplicateOrderGuard` — 4 tests: skip on same-side, allow on empty, allow
  opposite-side, skip on multiple same-side
- `TestRebalanceCadence` — 6 tests: first cycle allows, seeded blocks within
  cadence, seeded allows after cadence, weekly/daily enforcement
- `TestGetLastFilledOrderTime` — 3 tests: returns filled_at, returns None on
  empty, returns None on exception

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
