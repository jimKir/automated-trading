# Fix-on-probation — July 2026

Follow-up to the strategy review (PR #3) that root-caused the -11.9pp paper-trading gap
against SPY. The review's verdict was *fix, don't kill* — but on a deadline. This document
records what was changed, what the validation backtest says about it, and the terms the
strategy is now judged against.

## What was fixed

| # | Change | Rationale |
|---|--------|-----------|
| 1 | Universe selection factor 126d → 63d momentum (`dynamic_universe.momentum_window`) | The 126d factor turned anti-predictive in 2026 — rank IC **-0.088 (t=-3.68)**, i.e. it was actively selecting losers. 63d still carries signal. |
| 2 | `dynamic_universe.top_n` 20 → 9 | 20 names diluted every edge to noise and drove churn; 9 concentrates without becoming a single-name bet. |
| 3 | Anomaly layer recalibrated (`regime/sentiment_anomaly.py`, `regime/anomaly.py`) | Three independently miscalibrated inputs pinned the composite at ELEVATED on ordinary days — see below. |
| 4 | Optimizer restores heat consumed by cap steps (`core/optimizer.py`) | Steps 5–7 could only cut weight; every cut leaked to cash and was never redistributed, so raising `max_portfolio_heat` to 0.95 had almost no effect. |
| 5 | Paper-trading inception re-baselined 2026-03-24 → 2026-04-22 (`scripts/build_dashboard_snapshot.py`) | The account was funded 2026-03-24 but placed no orders until 2026-04-22. Charging the strategy a month of flat-line it never traded diluted every rate metric. |
| 6 | 60/40 SPY/AGG benchmark + probation terms on the dashboard | The bot is deliberately not fully deployed; a long-only equity benchmark flatters it. 60/40 is the honest comparison and the replacement candidate. |
| 7 | **Entry-day mark-to-market bug** (`core/portfolio.py`) | Found while building the validation backtest, not planned. Not a strategy change — a measurement fix that invalidates prior backtest numbers. See below. |

### Anomaly layer — root cause (item 3)

Not a stale cache and not a stale IsolationForest fit. Three separate normalisation ranges
were calibrated against pre-2022 data and never revisited:

- **SPY/TLT correlation (largest contributor).** Scored any positive stock/bond correlation
  as stress via `(corr - 0)/0.5`. True pre-2022 (mean -0.39); the structural norm post-2022
  (mean +0.12, **+0.34 in 2026**). Calm days scored ~0.56. Now measured as excess over its
  own trailing 252-day median, so it self-recalibrates through regime shifts.
- **Vol-of-VIX.** Calm floor of 0.030 against a measured median of 0.062 (0.079 in 2026), so
  a typical day scored 0.27–0.41. Recalibrated to the observed distribution.
- **IsolationForest score map.** `(-raw + 0.1)/0.3` returned a nonzero anomaly score for any
  `raw < 0.1`, which on purely normal data is ~84% of days. `decision_function` is centred so
  the contamination quantile sits at 0; the map now starts there.

Also fixed: `score_at` (the live path) pulled only 90 calendar days, too short to warm the
252-bar rolling windows, so live scores silently diverged from the backtest series.

### Entry-day mark-to-market bug (item 7)

A fresh `Position` starts at `current_price = 0` and `update_prices` only runs at the top of
the next day, so a position opened today contributed no market value for the rest of the
session. Recorded equity dropped by the entire notional on every entry day and recovered the
next one.

This affected **every backtest in the repo**, always overstating both return and risk. On the
2026 YTD window the correction moves the new config from +50.04% / Sharpe 1.06 / -43.10% DD to
**+2.99% / Sharpe 0.03 / -14.39% DD**. Any historical backtest result in this repository that
predates this commit should be treated as unreliable.

## Validation backtest

Four-way comparison over 2026-01-02 → 2026-07-23 (203 trading days, 83 symbols).
Source: `scripts/run_fix_validation_2026_07.py` → `results/fix_validation_2026_07.json`,
chart `results/fix_validation_2026_07.png`.

| Arm | Return | Sharpe | Max DD | Ann vol | Trades | Avg heat |
|-----|--------:|-------:|-------:|--------:|-------:|---------:|
| SPY buy & hold | +8.63% | 0.61 | -8.88% | 11.45% | — | 1.00 |
| **60/40 SPY/AGG** (probation benchmark) | +5.03% | **0.32** | -5.84% | 7.59% | — | 1.00 |
| New config (63d, top_n 9, heat 0.95) | +2.99% | **0.03** | -14.39% | 11.57% | 246 | 0.46 |
| Old config (126d, top_n 20, heat 0.75) | +0.66% | -0.27 | -9.92% | 10.00% | 642 | 0.38 |

**Read this honestly.** The fixes are a real improvement on the deployed configuration —
+2.33pp of return, Sharpe from -0.27 to +0.03, and 62% fewer trades. They are *not* enough to
clear the bar. On this window the new config still loses to 60/40 on return (-2.04pp), Sharpe
(0.03 vs 0.32) and drawdown (-14.39% vs -5.84%). It would fail probation if the window ended
today.

Two caveats in both directions: 203 days is too short for a Sharpe estimate to be worth much,
and the window is a bull tape (SPY +8.63%) in which a partially deployed, partly hedged book
is structurally disadvantaged.

### Remaining binding constraint: breadth, not heat

Realised heat is 0.46 against a 0.95 target even *after* the renormalisation fix. Gross
exposure is bounded by `funded_names × max_position_pct`, and with `max_position_pct` at 0.15
the 0.95 target needs 7+ names carrying positive signals simultaneously. The 2026 window funds
**3.9 on average**. The optimizer fix removed the leak; it cannot manufacture candidates.

So the cash drag is now a *signal breadth* problem. Raising `max_position_pct` would close the
gap arithmetically while concentrating risk into whatever few names pass the filters — that is
a deliberate risk decision, not a bug fix, and is explicitly left out of scope here. It is the
first thing to evaluate if the strategy is still short of target heat at the next checkpoint.

## New probation terms

| Term | Value |
|------|-------|
| Benchmark | **60/40 SPY/AGG**, monthly rebalanced (no longer SPY alone) |
| Deadline | **2026-10-31** |
| Criterion | Bot Sharpe must **exceed** 60/40 Sharpe over the probation window |
| If it fails | Replace with 60/40 + 200d-MA overlay |

Carried in `docs/data/snapshot.json` as `probation_deadline` / `probation_benchmark` and shown
on the dashboard, so the bar is visible on every refresh rather than living only in this file.

Current standing at re-baselined inception 2026-04-22: bot **-0.6pp** vs 60/40, -2.74pp vs SPY.

### What would change the verdict

- Sharpe above 60/40's over the probation window, with the gap not attributable to a single
  position.
- Realised heat converging toward target, or an explicit decision on `max_position_pct` /
  breadth rather than a silent shortfall.
- Anomaly layer producing a distribution of states over the next quarter. If it still reads
  ELEVATED nearly every day, the recalibration missed something and the layer should be
  switched off rather than left to tax returns invisibly.

## Deployment status

Code changes are merged but **not yet live**. The container must be rebuilt and the ECS
service redeployed before any of this affects paper trading; until then the dashboard reflects
the old configuration. Commands in the PR description.
