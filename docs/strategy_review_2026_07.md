# Strategy Review — July 2026

**Scope:** 4 months of Alpaca paper trading (inception 2026-03-24, data through 2026-07-23).
**Headline:** Bot +1.41% vs SPY +13.30% vs QQQ +18.62% → −11.9pp vs SPY.
**Verdict:** **FIX** — but the strategy has not actually been fairly tested yet, and it is on probation against a 60/40 benchmark with a pre-committed kill date.

---

## 1. Deployed state — what is actually running

| Item | State |
|---|---|
| HEAD | `0acdf75` on `main` (dashboard auto-commit) |
| Last real code commit | `5e673b1` (2026-05-16) — CI lint fix only |
| Last *strategy* change | 2026-04-22 (`9853332`, long-only clipping) |
| PR #2 "P0-P2 strategy guards" | **OPEN since 2026-04-26 — NEVER MERGED** |
| Trading runtime | AWS ECS + EventBridge (not GitHub Actions) |
| GitHub Actions | monitoring only: dashboard refresh (15 min), daily summary, weekly scorecard, health check, CI |

### Live parameters (`config/settings.yaml`)

```
dynamic_universe.top_n:            20        # <-- holds 20 names, not 3
dynamic_universe.momentum_window:  126       # 6-month selection factor
capital.max_portfolio_heat:        0.75      # <-- hard 25% cash floor
risk.max_position_pct:             0.15
optimizer.method:                  risk_parity
optimizer.rp_concentration_cap:    3.5
vol_targeting.enabled:             false
rebalance_frequency:               adaptive
```

### What PR #2 would have changed (and still hasn't)

- `max_portfolio_heat` 0.75 → **0.95** (the cash-drag fix)
- `initial_equity` 25,000 → 100,000 (aligns with the real paper account)
- `rebalance_guards`: 24h minimum interval, min order delta 2% of position
- `risk_limits.max_daily_turnover_x: 2.0` (churn cap, persisted across restarts)
- `signals.cache_per_session: true` (stop intraday signal flapping)
- `reentry_ramp`: 50% / 75% / 100% gradual redeployment

**This PR is the single highest-value un-merged artifact in the repo.** It directly addresses root cause #2 below.

### Live portfolio (2026-07-24 snapshot)

Equity $101,407 · Cash $35,756 · **Invested 64.74%**

VNQ 8.74% · IWM 8.45% · XLE 6.39% · EMXC 5.92% · XLK 5.31% · XLV 5.14% · EEM 4.84% · AGG 3.14% · VGK 2.90% · XLP 2.66% · SPY 2.65% · SHY 2.64% · XLF 2.25% · QQQ 2.25% · XLU 1.46%

The 64.74% deployment is not a mystery — it is exactly `max_portfolio_heat (0.75) × combined_scale (~0.86)`.

---

## 2. Root causes of the −11.9pp gap

### Gap decomposition

Splitting the record at 2026-04-22 (the first date the bot was reliably live):

| Period | Bot | SPY | Gap |
|---|---:|---:|---:|
| Phase 1 — Mar 24 → Apr 21 | **+0.09%** | +7.79% | **−7.70pp** |
| Phase 2 — Apr 22 → Jul 23 | +1.71% | +4.06% | −2.35pp |
| **Total** | **+1.80%** | **+13.30%** | **−11.50pp** |

**67% of the entire underperformance happened before the bot was meaningfully trading.**

---

### Root cause #1 — The track record is mis-dated. (−7.70pp, 67% of the gap)

The dashboard's inception is 2026-03-24, but the bot did not trade then:

- Bot equity is **exactly 100.00 for 13 consecutive trading days**, with literally zero response to SPY days of −1.79%, +2.56%, +2.56%, +2.21%.
- `CHANGELOG.md:174` — the first real order flow was **Apr 13–14**, and it was the *486-day-trades incident* (1,503 orders, PDT flag on the account).
- Commit `1f283ec` (2026-04-20): *"infra: enable paper trading schedules and set desired_count=1"* — the ECS schedule was only switched on Apr 20.

So "4 months of paper trading" is really **~3 months**, whose first week was a churn incident. The headline −11.9pp is mostly an accounting artifact of a backdated inception, not evidence about the strategy.

**This does not make the strategy good. It makes the number you are deciding on wrong.**

### Root cause #2 — Structural cash drag. (−1.43pp of the −2.35pp live gap)

`max_portfolio_heat: 0.75` is a hard ceiling: even in a perfect GREEN regime the bot cannot deploy more than 75%. Observed deployment is 64.74%, i.e. an additional ~0.86 scaler is being applied.

The chain (`execution/live_engine.py:481`) is a `min()`, not a product — this part is correctly designed:
```python
combined_scale = max(min(ews_scale, isd_scale, anomaly_scale), 0.50)
effective_heat = max_heat * combined_scale          # line 550
```
A 0.86 combined scale is consistent with the **multi-source anomaly layer sitting in ELEVATED (0.85×)**, not with the ChoppyDetector being stuck.

Then `core/optimizer.py:202-226` normalizes weights to exactly `effective_heat` and every later step (risk-parity cap, per-position cap, crypto cap) can only **cut** — nothing renormalizes back up. Caps leak straight into cash.

Quantified: over Phase 2, holding **65% SPY / 35% cash returned +2.63% vs SPY's +4.06%** — that is 1.43pp of pure drag for nothing.

### Root cause #3 — Rotation churn, not over-diversification. (−1.69pp)

This is where the brief's premise is wrong, and it matters:

| Phase 2, Apr 22 → Jul 23 | Return |
|---|---:|
| SPY | +4.06% |
| 65% SPY / 35% cash (isolates cash drag) | +2.63% |
| **The bot's own 15 holdings, equal-weight, 65% heat, buy & hold** | **+3.40%** |
| **What the bot actually did** | **+1.71%** |

**The bot's stock selection was fine — it beat SPY-at-the-same-deployment by +0.77pp.** It then gave back 1.69pp by *trading* that basket instead of holding it. The 20 orders on 2026-07-24 fired within 15 seconds and included both a buy and a sell of VNQ on the same day.

Over-diversification (15 × ~4%) cost approximately nothing in this period. It is a real design smell, but it is **not** a top-3 root cause, and "concentrate the book" is not where the money is.

### Supporting finding — the selection factor is broken, the signal factors are not

Cross-sectional rank IC across 30 ETFs:

| Factor | 2025H2+2026, fwd 21d | **2026 only, fwd 21d** |
|---|---:|---:|
| `ts_mom_126` ← **used by the universe selector** | +0.038 (t=+1.85) | **−0.088 (t=−3.68)** |
| `ts_mom_63` | +0.114 (t=+5.71) | +0.005 (t=+0.19) |
| `ts_mom_20` | +0.035 (t=+1.66) | +0.060 (t=+1.90) |
| `macd_norm` | +0.085 (t=+4.17) | +0.060 (t=+1.86) |

`dynamic_universe.momentum_window: 126` — the factor that picks *what the bot owns* — has been **significantly anti-predictive in 2026** (t = −3.68). The factors in the *signal blend* (MACD, short momentum) are still working. The bot is choosing its universe with a broken ruler and then scoring it with a working one.

---

## 3. Alternatives, 2026 data

Backtested on `data/historical/daily/` (extended to 2026-07-23 via yfinance, since the local parquet set stops 2026-04-10). 30-ETF universe, monthly rerank, no costs.

### 2026 YTD (Jan 1 → Jul 23)

| Strategy | Return | MaxDD | Vol | Sharpe |
|---|---:|---:|---:|---:|
| (a) Buy & hold SPY | **+8.63%** | −8.88% | 13.9% | **1.16** |
| (b) 200-day MA timing on SPY | +4.67% | −6.59% | 12.2% | 0.74 |
| (c) **Top-3 momentum rotation** | **−0.37%** | **−24.48%** | 43.2% | 0.20 |
| (d) 60/40 SPY/AGG monthly | +4.96% | −5.86% | 9.2% | 1.01 |
| (e) Deployed design simulated (top-20 @ 65% heat) | +6.32% | −5.35% | 10.8% | 1.09 |

### Paper-trading window (Mar 24 → Jul 23) — like-for-like with the bot

| Strategy | Return | MaxDD | Vol | Sharpe |
|---|---:|---:|---:|---:|
| (a) Buy & hold SPY | **+13.30%** | −4.49% | 14.6% | **2.68** |
| (b) 200-day MA timing on SPY | +9.95% | −4.49% | 12.1% | 2.44 |
| (c) **Top-3 momentum rotation** | **−1.87%** | **−18.64%** | 39.1% | 0.05 |
| (d) 60/40 SPY/AGG monthly | +7.81% | −2.87% | 9.8% | 2.39 |
| (e) Deployed design simulated | +5.91% | −3.28% | 10.6% | 1.70 |
| **(f) Bot, actual** | **+1.41%** | **−2.34%** | 5.9% | **1.18** |

### Two uncomfortable conclusions

**1. "Restore the validated top-3 design" would have been a disaster.** Top-3 momentum returned **−1.87% with a −18.6% drawdown** in the paper window — worse than the bot on every axis. The 126-day momentum leaderboard loaded into precious metals right before they broke:

```
2026-02-27  SLV +142.9% | GLD +54.7% | SOXX +40.7%
   ...then Apr 22 -> Jul 23:  SLV -26.0%,  GLD -14.6%
```

**2. There was never a top-3 design in this repo.** `top_n: 20` has been the value since the *first commit* (`e264b4e`, 2026-04-02); git history shows no prior value. The only "top-3" in the codebase is `diagnostics/technical_indicator_ic.py:279`, an IC diagnostic that constructs a top-3 long portfolio to measure factor quality. That diagnostic appears to have been mistaken for the production design. **The bot never drifted from top-3 — it was never top-3.**

**3. The bot currently loses to 60/40 on every metric.** 60/40 delivered +7.81% with a smaller drawdown (−2.87%) and a higher Sharpe (2.39 vs 1.18). A multi-asset momentum system carrying this much machinery must clear that bar.

Also worth noting: realized beta to SPY in Phase 2 is **−0.05** (correlation −0.10) on a book that is 65% long equity ETFs. On SPY's four biggest up days the bot was *down*. That is not a diversified equity portfolio; it is an accidental market-neutral book.

---

## 4. Verdict and action plan

### **FIX** — with probation and a pre-committed kill criterion.

Restructuring or replacing now would be a decision made on a number (−11.9pp) that is 67% measurement error. But "fix" must not be read as a vote of confidence: on the ~3 months of real data available, the bot is beaten by a 60/40 portfolio that requires no code at all.

### Immediate (this week)

1. **Merge PR #2.** It is the cash-drag and churn fix, it has tests (`tests/test_strategy_guards.py`, 445 lines), and it has been sitting open for 3 months. Heat 0.75 → 0.95 alone recovers roughly 1.4pp per quarter of pure drag.
2. **Re-baseline the track record to 2026-04-22** and stop quoting −11.9pp. Add a note to the dashboard that inception predates first fill. The current number is misleading whoever reads it, including you.
3. **Fix the selection factor.** `dynamic_universe.momentum_window: 126` → `63`, or blend 63/126. The 126d factor is at t = −3.68 in 2026; 63d and MACD still carry signal.

### Short term (next 4 weeks)

4. **Investigate why the anomaly layer sits at ELEVATED (0.85×).** That scaler, not the ChoppyDetector, is what is trimming deployment below the heat cap. The ChoppyDetector was checked and degrades safely to 0.0 on missing data — it is not the culprit.
5. **Reduce `top_n` 20 → 8–10.** Not to 3. This is a modest concentration improvement, not the main lever — size the expectation accordingly.
6. **Hold the basket between reranks.** The 1.69pp rotation drag says the bot is trading inside its own signal. PR #2's `min_order_delta_pct_of_position: 0.02` plus session signal caching should largely handle this; verify with a turnover report after one month.

### Probation — decide by **2026-10-31**

Benchmark is **60/40 SPY/AGG**, not SPY. Measured from 2026-04-22.

- **Beats 60/40 on Sharpe** → keep, continue tuning.
- **Fails** → **REPLACE**. Default replacement is 60/40 rebalanced monthly, or 200-day MA timing on SPY if drawdown control is the priority. Both are ~20 lines of code and both beat the bot over the window measured here.

Write the kill criterion down now, before the next tuning cycle makes it negotiable.

### Do not do

- **Do not switch to top-3.** −1.87% return, −18.6% drawdown in this window.
- **Do not add another safety layer.** There are already six (EWS, ISD, anomaly layer, position anomaly, trend classifier, regime heat scaling). They are why the bot has a 5.9% vol and −0.05 beta in a bull market.

---

## Appendix — reproduction

- Live state: `docs/data/snapshot.json` (auto-refreshed by `.github/workflows/dashboard-refresh.yml`)
- Alternatives backtest: 30-ETF universe, 126d momentum, monthly rerank; prices from `data/historical/daily/` extended via yfinance through 2026-07-23. Local parquet coverage ends 2026-04-10, so the paper window cannot be reproduced from the repo data alone.
- Costs are excluded from the alternatives table; including them widens the gap against the high-turnover options (c) and (e).
