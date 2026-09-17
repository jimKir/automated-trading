# Bear-Regime Short Selling — Design Notes

Status: **implemented, tested, DISABLED by default** (`shorting.enabled: false`).
Config: `shorting:` block in `config/settings.yaml`.
Code: `strategy/short_overlay.py`, wiring in `execution/live_engine.py` and
`backtest/engine.py`. Related: `execution/tradeable_universe.py` (execution guards).

## Validation verdict (2026-09-17)

`backtest/short_selling_backtest.py` compared long-only vs +short overlay
(`results/short_selling_backtest.json`, `.png`):

| Window | Arm | Return | Sharpe | MaxDD | Turnover/yr | % days shorted | Short P&L |
|---|---|---|---|---|---|---|---|
| 2022 bear (365d) | long-only | -26.2% | -1.55 | -27.9% | 25.5x | 0% | — |
| 2022 bear (365d) | + overlay | -30.2% | **-1.97** | **-32.1%** | 95.8x | 98.1% | +$1,819 |
| 2026 YTD (149d) | long-only | -4.4% | -0.62 | -15.5% | 23.1x | 0% | — |
| 2026 YTD (149d) | + overlay | -5.0% | **-0.69** | **-15.7%** | 46.8x | 61.7% | -$165 |

The overlay **materially worsened Sharpe and MaxDD on both windows**, so the
config default is OFF per the implementation plan. The mechanism worked as
designed — 2022 gross short P&L was positive (+$1.8k on $25k) — but daily
re-targeting churn quadrupled turnover and the transaction-cost drag
overwhelmed it. Re-enabling requires solving the turnover problem first
(weekly-gated short re-targeting, minimum short-holding period, or
band-based target tolerance), then re-running this validation.

## Rationale

Previously a bear regime could only express itself defensively: reduce longs,
hold cash. The 63d ranked momentum selector produces strong negative signals
during drawdowns (2022, early-2026) but the old system discarded them — the
worst-ranked name simply wasn't held. This overlay converts that information
into capped short exposure on liquid US equity ETFs, so the portfolio can earn
bear-market alpha instead of only hiding from it.

Shorting is deliberately narrow:

- **Asset classes: equity ETFs only.** No crypto shorts (Alpaca crypto is
  long-only; no borrow market), no futures (Alpaca has none).
- **Fixed liquid universe** — 16 broad/sector ETFs (SPY, QQQ, IWM, DIA, MDY,
  EEM, VGK, EWJ, XLE, XLF, XLV, XLU, XLP, XLY, XLK, VNQ). Deep, persistent
  borrow; tight spreads; no single-name squeeze exposure.
- **Signal gate**: only names whose blended ranked momentum score is negative
  (< `min_signal_strength`, default 0.0).
- **Regime gate**: in a bear regime (SPY < 200d MA) any negative-signal
  universe name qualifies. Outside a bear regime a name must additionally
  trade below its own 200d MA — idiosyncratic downtrends only, no
  counter-trend shorts in bull markets.

## Position sizing and risk controls

| Control | Config key | Value | Notes |
|---|---|---|---|
| Aggregate short notional | `max_short_notional_pct` | 30% of equity | hard budget |
| Per-name cap | `max_single_short_pct` | 8% of equity | concentration limit |
| Portfolio heat | `capital.max_portfolio_heat` | 0.95 | shorts consume remaining heat: budget = min(30%, heat − |longs|) |
| Hard stop | `hard_stop_pct` | +8% adverse | intraday cover, every engine cycle |
| Regime exit | `cover_on_regime_improvement` | true | cover all shorts at the rebalance after bear→yellow+ |
| Borrow check | `execution_guards.verify_with_alpaca` | true | `shortable` AND `easy_to_borrow` on the Alpaca asset, cached per session |

Sizing is proportional to |signal| across qualifying names, then clipped to
the per-name cap; leftovers are not redistributed (keeps the cap absolute).

Weights are negative fractions of equity and merge into the same target-weight
pipeline as longs, so every existing guard applies unchanged:

- **Daily turnover cap** (`risk_limits.max_daily_turnover_x`): short entries,
  extensions and covers all count as turnover. Exception: hard-stop covers
  deliberately bypass the cap — a risk exit must never be suppressed by a
  cost guard.
- **Re-entry ramp** (PR #2/P1-2): shorts are sized from the *effective* heat
  budget, so a ramped-down book also ramps down short capacity.
- **Daily loss limit / drawdown halt** (`risk/manager.py`): these read equity
  and cash, both of which already include short P&L with the correct sign
  (short proceeds raise cash; the short's negative market value nets against
  it; `equity = cash + Σ market_value` stays correct).
- **Execution guards**: the overlay runs after the tradeable-universe filter,
  so a phantom instrument can never be shorted even if its signal qualifies.
- **Order-layer sell cap**: the legacy "never sell more than you hold" guard
  is relaxed *only* for names the overlay targeted this cycle and that passed
  the shortable/ETB check.

Backtest intraday realism: the daily-bar engine simulates the +8% hard stop
from the day's High (if High ≥ entry×1.08 the short is covered at the stop
price), and the intraday circuit-breaker excursion uses High for shorts
(previously longs-only logic would have read a squeeze as a gain).

## Pattern Day Trader

The Alpaca account's `pattern_day_trader` flag is checked at engine start and
a warning is logged if set. Short entries/covers are round-trips; if the
account is flagged PDT and equity falls below $25k, day-trading restrictions
can block same-day covers — dangerous for a strategy with a hard stop. The
paper account used for probation is above that threshold, but check it after
any capital change.

## Not modelled (paper parity limits)

- **Borrow fees**: not charged. The universe is restricted to liquid ETFs
  where borrow is typically ≤0.3%/yr — immaterial at these holding periods.
- **Locate risk / forced buy-in**: Alpaca paper never force-liquidates a
  short. On live capital, a borrow recall could force a cover at the worst
  time; the liquid-ETF universe minimises this.
- **Dividend payments**: shorts pay dividends. Ex-dates during a short cost
  ~0.1–0.5% quarterly per name; not simulated.
- **Margin interest on proceeds**: not credited/charged in paper.

## What would make us keep it disabled

It is already disabled. Conditions that would ALSO block any re-enable
(check before setting `shorting.enabled: true`, one config line):

1. A re-run of the validation backtest
   (`backtest/short_selling_backtest.py`) after turnover fixes still shows
   the overlay materially worsening Sharpe/MaxDD on both windows, or
2. Paper deployment shows repeated hard-stop covers (>2/month) — the entry
   timing is fighting momentum rather than riding it, or
3. Realised short borrow/locate behaviour diverges from paper (recalls,
   rejections), or
4. Before any live-capital review — shorting is off by default for live
   capital until explicitly approved.

Disabling is fully backward compatible: when `enabled: false` the whole
overlay block (including its cover path) is skipped, the sell-cap guard
returns to its strict long-only form, and no new shorts can be generated.
**Important:** any already-open shorts are no longer managed — cover them
before flipping the switch (manual `buy_to_cover` order, or one last
rebalance with shorting still enabled while the regime gate blocks new
entries).

## Verification quick list

1. `python healthcheck.py` — shorting config checks + guard checks.
2. `python backtest/short_selling_backtest.py` — regenerates
   `results/short_selling_backtest.{json,png}`.
3. `python scripts/status.py` — shorts shown with negative qty, `Side=SHORT`,
   correct P&L sign, plus a gross short-exposure summary line.
4. Logs to grep in paper: `[SHORT] Overlay targets`, `[SHORT] Opening`,
   `[SHORT-STOP]`, `[SHORT] Regime improved — covering`, `[SHORT] Account is
   flagged pattern_day_trader`.
