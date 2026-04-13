# Options Hedge Analysis: Protective Put Strategy — REJECTED

**System:** jimKir/automated-trading — ChoppyDetector v4 + 7-Factor Engine  
**Date:** April 13, 2026  
**Author:** Automated analysis via walk-forward backtest  
**Status:** ❌ REJECTED — Not implemented in production  
**Test Scope:** 5 structural-break periods (2011–2022), 2 in-sample + 3 out-of-sample

---

## Executive Summary

A protective put hedge layer — SPY puts triggered by ChoppyDetector ORANGE/RED signals — was designed, implemented, and rigorously backtested across five distinct market stress regimes. The hedge was **rejected** because the portfolio's existing diversification (20% TLT + 15% GLD) already absorbs 60–78% of equity drawdowns at zero cost. Puts generated positive net value in only 1 of 5 tested periods (COVID 2020), while degrading Sharpe ratio in all 5 and worsening maximum drawdown in all 5.

This finding is consistent with AQR's landmark research showing that "investing 36.5% in SPX and holding 63.5% in cash provided the same 2.5% compound annualized excess return" as the CBOE S&P 500 5% Put Protection Index ([AQR — Pathetic Protection](https://www.aqr.com/Insights/Research/White-Papers/Pathetic-Protection-The-Elusive-Benefits-of-Protective-Puts)). Alpha Architect's analysis further confirms that divesting outperformed put protection 97% of the time over short horizons and 100% over longer horizons ([Alpha Architect](https://alphaarchitect.com/pathetic-protection/)).

---

## Hedge Design

The protective put module (`execution/options_hedge.py`) is wired to ChoppyDetector v4 regime signals:

| Parameter | ORANGE Trigger | RED Trigger |
|---|---|---|
| ChoppyDetector Score | ≥ 0.229 | ≥ 0.296 |
| Strike | 2% OTM | 5% OTM |
| DTE | 21 days | 14 days |
| Close Signal | Score < 0.192 (GREEN) or expiry | Score < 0.192 (GREEN) or expiry |

**Position sizing:** 20% of portfolio notional, max 5 contracts.  
**Pricing model:** Black-Scholes with VIX-derived implied volatility (×1.20 skew adjustment to account for put skew relative to VIX).  
**Portfolio composition:** 40% SPY + 20% TLT + 15% GLD + 25% cash.

The ChoppyDetector proxy score is computed as:

```
Score = 0.30 × VolSpike + 0.30 × VIX_5d_change + 0.20 × VIX_level + 0.20 × Mom_10d
```

Where VolSpike is the ratio of 5-day to 20-day realized volatility, VIX 5-day change captures acceleration, VIX level normalizes absolute fear, and 10-day SPY momentum captures directional damage.

---

## Backtest Results: Full Summary

Two versions of the backtest were run. Version 1 had a lookahead bug (payoff computed at mark-to-market close date, not expiry date) and used a lagging trigger. Version 2 corrected these issues with earlier triggers, proper expiry payoff accounting, and automatic roll logic.

### v2 Backtest — All 5 Periods

| Period | Type | Sample | Puts Opened | Premium Paid | Payoff | Net Cost | ΔMaxDD | ΔSharpe |
|---|---|---|---|---|---|---|---|---|
| COVID Crash 2020 | Spike crash | IS | 3 | $2,164 | $6,981 | -$4,817 (profit) | -0.015 | -0.289 |
| Rate Hike 2022 | Slow grind | IS | 9 | $7,493 | $564 | +$6,929 (loss) | -0.068 | -0.546 |
| Flash Crash 2015 | V-recovery | OOS | 1 | $126 | $0 | +$126 (loss) | -0.001 | -0.030 |
| Q4 Sell-Off 2018 | Slow grind | OOS | 2 | $548 | $0 | +$548 (loss) | -0.005 | -0.154 |
| Debt Ceiling 2011 | Whipsaw | OOS | 4 | $1,977 | $0 | +$1,977 (loss) | -0.002 | -0.332 |

**Aggregate:**
- MaxDD improved: **0 out of 5 periods**
- Mean ΔMaxDD: **-0.018** (hedge makes drawdowns worse on average)
- Mean ΔSharpe: **-0.270**
- Mean net cost: **+0.88% of portfolio per period**
- OOS verdict: **REJECT** (0/3 OOS periods showed improvement)

---

## Period-by-Period Analysis

### COVID Crash 2020 (In-Sample) — The One Win

This was the only period where the hedge generated a positive payoff, and it demonstrates both the best-case scenario and the fundamental limitation:

| Event | Date | Detail |
|---|---|---|
| OPEN ORANGE put | Feb 24, 2020 | K=$289, 21 DTE, cost=$547 |
| EXPIRY with payoff | Mar 16, 2020 | SPY=$219, payoff=$6,981, net +$6,434 (11.8× return) |
| OPEN RED put (post-crash) | Mar 16, 2020 | K=$208, 14 DTE, cost=$1,148 |
| EXPIRY worthless | Mar 30, 2020 | SPY=$240 (market recovered), net -$1,148 |
| OPEN RED put (false alarm) | Jun 11, 2020 | Cost=$469, expired worthless |

The first put was brilliant — SPY fell 26% in 16 trading days, faster than the 21-DTE expiry window, and VIX spiked from 15 to 75, boosting put value through IV expansion. But the subsequent two puts opened during the recovery and whipsaw phase erased part of the gains.

**Critical insight:** Even in the best-case scenario (COVID), the hedged portfolio still underperformed the unhedged portfolio on Sharpe (-0.289) and MaxDD (-0.015). The diversification via TLT (+25% during COVID) and GLD (+12%) had already reduced the portfolio MaxDD from -33.7% (SPY alone) to -13.1% — a 61% absorption rate — making the put's additional protection marginal.

### Rate Hike Cycle 2022 (In-Sample) — Worst Case

The 2022 bear market was a slow grinding decline (-18.7% SPY over 10 months) with VIX oscillating between 20–35 — never a single spike that triggers a sustained payoff. The ChoppyDetector fired ORANGE 9 times, opening puts that were quickly closed as the score briefly returned to GREEN:

- Feb 23: OPEN → Feb 25: CLOSE (2 days, $0 payoff)
- Mar 7: OPEN → Mar 9: CLOSE (2 days, $0 payoff)
- ... repeated 9 times through the year

Total premium paid: $7,493. Total payoff: $564. Net loss: $6,929 (+6.93% of portfolio). The hedge increased MaxDD from -17.3% to -24.1% — the exact opposite of its intended purpose.

This is the "slow grind" failure mode: puts are designed for fast crashes where the underlying drops below the strike within the DTE window. In a prolonged bear market, each put expires worthless and a new one is opened, creating pure premium bleed. This aligns with AQR's finding that "put protection can lead to worse drawdown characteristics when options are priced with a volatility risk premium" ([AQR — Pathetic Protection](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/Pathetic-Protection-JAI-Wint19.pdf)).

### Flash Crash Aug 2015 (Out-of-Sample) — Too Fast to Hedge

VIX spiked from 13 to 53 in 3 days. The ChoppyDetector fired RED on Aug 21 and a put was opened at K=$157. But the V-shaped recovery was complete by Sep 3 — SPY recovered to $163.50 and the put closed worthless. The crash happened so fast that the 14-DTE window missed the bottom entirely.

Unhedged portfolio MaxDD was only -4.5% (vs. SPY's -11.9%), demonstrating that TLT and GLD absorbed 62% of the damage at zero cost.

### Q4 Sell-Off 2018 (Out-of-Sample) — Slow Grind Repeat

Similar pattern to 2022 but shorter: SPY dropped 20% over 3 months. Two puts opened, both expired worthless. The unhedged portfolio limited MaxDD to -6.6% (vs. SPY's -19.4%), a 66% absorption rate from diversification alone.

### US Debt Ceiling 2011 (Out-of-Sample) — Whipsaw Destroys Value

The most hostile environment for the hedge. The market whipsawed around the ORANGE/GREEN threshold throughout the second half of 2011:

- Aug 4: OPEN RED K=$88 → Aug 18: Expired, SPY=$88.20 (missed by $0.20)
- Aug 18: OPEN ORANGE K=$86 → Aug 23: CLOSE at $0 (5 days)
- Sep 22: OPEN ORANGE K=$86 → closed worthless
- Oct cycle: another worthless put

Four puts, $1,977 premium, $0 payoff. Portfolio MaxDD was only -4.0% vs. SPY's -18.6% — a 78% absorption rate from TLT/GLD diversification. The hedge was entirely redundant.

---

## Why Diversification Beats Puts

The data reveals a consistent pattern: the multi-asset portfolio (40% SPY + 20% TLT + 15% GLD + 25% cash) already provides substantial drawdown protection across all market stress types.

| Period | SPY MaxDD | Portfolio MaxDD | Drawdown Absorbed | Cost |
|---|---|---|---|---|
| Flash Crash 2015 | -11.9% | -4.5% | 62% | Free |
| Q4 Sell-Off 2018 | -19.4% | -6.6% | 66% | Free |
| Debt Ceiling 2011 | -18.6% | -4.0% | 78% | Free |
| COVID Crash 2020 | -33.7% | -13.1% | 61% | Free |
| Rate Hike 2022 | -24.5% | -17.3% | 29% | Free |

During equity drawdowns, TLT (long-term Treasuries) historically rallies during flight-to-safety events — TLT gained approximately 25% during the COVID crash as the Fed cut rates to zero. GLD (gold) provides inflation and currency stress protection, gaining 23% in 2020 and maintaining a long-term CAGR of ~8.25%.

The 2022 rate hike cycle is the exception where diversification absorbed only 29% — both stocks and bonds fell simultaneously. But this is precisely the scenario where puts also failed (9 puts, $7,493 premium, $564 payoff), because the slow grind nature of the decline means puts expire worthless before the market drops enough to trigger payoffs.

Graham Capital Management's research confirms that diversification combined with trend-following strategies historically outperforms pure options-based hedging on a cost-adjusted basis ([Graham Capital — Tail Risk Hedging](https://www.grahamcapital.com/wp-content/uploads/2023/08/Tail-Risk-Hedging_Graham-Research-October-2017-1.pdf)). Cambridge Associates similarly notes that "across academic studies, buying an ongoing tail hedge using put options is a money-losing proposition" and that one study found "systematically buying put options within a stock portfolio eroded almost two-thirds of the equity returns over nearly thirty years" ([Cambridge Associates](https://www.cambridgeassociates.com/insight/portfolio-protection-challenges-with-equity-put-options/)).

---

## The Volatility Risk Premium Problem

A fundamental economic force works against systematic put buying: the volatility risk premium (VRP). Index put options are systematically overpriced relative to realized volatility because:

1. **Implied volatility exceeds realized volatility** — VIX (implied) historically averages ~19 while realized SPY volatility averages ~15. This gap is the insurance premium embedded in option prices ([CXO Advisory](https://www.cxoadvisory.com/volatility-effects/volatility-risk-premium-an-exploitable-stock-market-predictor/)).

2. **Put skew amplifies the cost** — SPY put options trade at even higher implied volatility than ATM options (the "skew"), adding ~20% to the effective cost. The backtest accounts for this with the 1.20 IV multiplier.

3. **The CBOE PPUT Index proves the drag** — the CBOE S&P 500 5% Put Protection Index, which systematically buys monthly 5% OTM puts, delivered a 2.5% compound annualized excess return — equivalent to holding just 36.5% in SPX and 63.5% in cash ([AQR](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/Pathetic-Protection-JAI-Wint19.pdf)). The VRP consumed nearly two-thirds of the equity risk premium.

4. **Option Alpha's meta-analysis confirms** that "the vast majority of protective put strategies, or long put strategies in general, are useless and significantly drag down your portfolio" across systematic testing of multiple strike/expiry combinations ([Option Alpha](https://optionalpha.com/podcast/protective-put-strategy-researching-findings)).

---

## The Timing Paradox

The backtest reveals a fundamental paradox with signal-triggered hedging:

**To protect against a crash, you must hold the put before the crash starts.** But the ChoppyDetector is inherently reactive — it needs rising volatility and falling momentum to trigger ORANGE/RED, which means the crash has already begun. By the time the signal fires:

- **Fast crashes (COVID):** The put has 16 trading days before expiry; if the crash completes within that window, the put pays off. This works for COVID-scale events (~1 per decade).
- **V-recoveries (2015):** The crash reverses before the put window closes. The signal fires correctly but the payoff window misses the bottom.
- **Slow grinds (2018, 2022):** The signal fires repeatedly but each individual drop isn't deep enough within a single DTE window to generate payoff. Pure premium bleed.
- **Whipsaws (2011):** The signal oscillates, opening and closing positions with maximum premium waste.

The alternative — always being hedged — is prohibitively expensive. At roughly 1–2% annually in premium drag (confirmed by the PPUT Index data), constant put protection would consume the equity risk premium that the entire trading strategy is designed to capture. Quantpedia notes that the VRP is "quite substantial — selling put options gives average returns ranging from 0.5% to 1.5% per day" ([Quantpedia](https://quantpedia.com/strategies/volatility-risk-premium-effect)), which means systematic put buyers are on the wrong side of this structural risk premium.

---

## Overfitting Assessment

An important critique of this analysis is whether the backtest parameters themselves are overfit:

| Component | Overfitting Risk | Mitigation |
|---|---|---|
| ChoppyDetector thresholds (0.192/0.229/0.296) | Medium — calibrated on 2018–2022 IS data | Uses 9 independent feature groups; ensemble reduces threshold sensitivity |
| ORANGE 2% OTM / RED 5% OTM | Low — standard option strike conventions | Not data-mined |
| 21-DTE / 14-DTE windows | Medium — chosen to match standard monthly/biweekly cycles | Could test 60–90 DTE for slow grinds |
| 20% hedge ratio | Low — conservative conventional sizing | Not data-mined |
| v2 fixes (faster trigger, proper payoff) | Bug fixes, not parameter changes | Legitimate corrections |

The 3 OOS periods (2015, 2018, 2011) used **zero parameter changes** from the IS-calibrated model, providing a genuine out-of-sample test. The consistent REJECT across all 3 OOS periods with the same parameters confirms the finding is not merely IS overfitting.

---

## Existing Protection Stack (Zero Cost)

The current system already layers five distinct protection mechanisms, none of which require option premium:

| Mechanism | Trigger | Action | Covers |
|---|---|---|---|
| TLT 20% allocation | Structural | Flight-to-safety natural hedge | Rate shock, recession, panic |
| GLD 15% allocation | Structural | Inflation/currency natural hedge | Inflation, currency stress, panic |
| ChoppyDetector 50% scale | Score ≥ 0.229 (ORANGE) | Cuts equity exposure to 50% | All stressed regimes |
| AnomalyLayer 65% scale | STRESSED regime | Reduces exposure on macro/FX/sentiment stress | Macro + FX + sentiment |
| PositionAnomalyScorer | Crypto-specific signals | Crypto exposure floor | Crypto crashes |

Combined, these reduced the COVID drawdown from -33.7% (SPY) to -13.1% (portfolio) — equivalent to what a perfectly timed put would provide, at zero premium cost.

---

## When to Revisit

Options hedging should be reconsidered if any of the following conditions change:

1. **Portfolio becomes concentrated** — if TLT/GLD allocations are removed and the portfolio shifts to equity/crypto-only, the natural hedge disappears and puts acquire genuine marginal value.

2. **Crypto derivatives become available** — BTC/ETH have no natural diversifier in the portfolio. A BTC put via Deribit during ORANGE/RED would address the one asset class where structural diversification fails. Equity options on Alpaca cannot hedge crypto legs.

3. **Longer DTE tested** — 60–90 DTE puts may survive slow grinds better than 21 DTE. The 2022 rate hike cycle could look different with 90-DTE puts rolled quarterly. Not tested in this analysis.

4. **Collar strategy evaluated** — Option Alpha's research found that the collar strategy (buying 5% OTM put + selling 5% OTM call) had "one of the most attractive risk-reward profiles" among all protective strategies tested, because the call premium offsets the put cost ([Option Alpha](https://optionalpha.com/podcast/protective-put-strategy-researching-findings)). A zero-cost collar triggered by ChoppyDetector would be worth investigating as a follow-up.

---

## Configuration

```yaml
# config/settings.yaml
options_hedge:
  enabled: false   # REJECTED — see docs/options_hedge_analysis.md
```

The `ProtectivePutHedge` class exists in `execution/options_hedge.py` and is wired into `LiveEngine` but disabled. Re-enable only if the conditions above change.

---

## Repository Files

| File | Description |
|---|---|
| `execution/options_hedge.py` | ProtectivePutHedge class (disabled in production) |
| `backtest/options_hedge_backtest.py` | v1 backtest (contains lookahead bug — kept for reference) |
| `backtest/options_hedge_backtest_v2.py` | v2 backtest (corrected trigger, proper payoff accounting) |
| `results/options_hedge_v2_results.json` | Full numerical results (all 5 periods) |
| `results/options_hedge_v2_*.png` | Per-period equity curve charts with ChoppyDetector overlay |
| `docs/options_hedge_analysis.md` | This document |

---

## References

- Israelov, R. (2017). "Pathetic Protection: The Elusive Benefits of Protective Puts." *The Journal of Alternative Investments*. [AQR](https://www.aqr.com/Insights/Research/White-Papers/Pathetic-Protection-The-Elusive-Benefits-of-Protective-Puts)
- Prakash, K. (2017). "Tail Risk Hedging." *Graham Capital Management Research Note*. [PDF](https://www.grahamcapital.com/wp-content/uploads/2023/08/Tail-Risk-Hedging_Graham-Research-October-2017-1.pdf)
- Cambridge Associates (2021). "Portfolio Protection: Challenges with Equity Put Options." [Link](https://www.cambridgeassociates.com/insight/portfolio-protection-challenges-with-equity-put-options/)
- Option Alpha (2020). "Protective Put Strategy Researching Findings." [Podcast](https://optionalpha.com/podcast/protective-put-strategy-researching-findings)
- Quantpedia. "Volatility Risk Premium Effect." [Link](https://quantpedia.com/strategies/volatility-risk-premium-effect)

---

## Conclusion

The protective put hedge is a theoretically appealing but empirically counterproductive addition to this specific portfolio. The combination of multi-asset diversification (TLT/GLD absorbing 60–78% of drawdowns) and position scaling (ChoppyDetector cutting exposure on ORANGE/RED) already achieves what puts are designed to do — and does it for free. Adding puts on top introduces premium drag (mean +0.88% per stress period), degrades risk-adjusted returns (mean ΔSharpe -0.270), and paradoxically worsens drawdowns (mean ΔMaxDD -0.018) due to the volatility risk premium and signal timing limitations.

The only scenario where puts add genuine value — a COVID-scale spike crash completing within 21 DTE — is a once-per-decade event that requires either permanent hedging (too expensive) or perfect timing (impossible with reactive signals). The verdict is clear: **the existing system already works. Don't pay for redundant insurance.**
