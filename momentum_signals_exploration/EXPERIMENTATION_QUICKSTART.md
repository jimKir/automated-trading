# Momentum Signals Experimentation Quickstart

Your complete toolkit for experimenting with ranking methods and expanding to larger universes is ready. This guide walks you through the exact steps to get started.

## What You Have

✅ **Core Momentum Scanner** (in root folder)
- Hourly momentum calculation
- Multi-universe support (500 to 5000+ symbols)
- 6 advanced filters
- 3 ranking methods
- Multi-channel alerts

✅ **Analysis Toolkit** (in `analysis/` folder)
- Ranking methods comparison
- DataBento integration testing
- Backtesting framework
- Interactive dashboard

✅ **Complete Documentation**
- README.md - Full scanner documentation
- EXPLORATION_GUIDE.md - Experimentation roadmap
- ANALYSIS_TOOLS_README.md - How to use analysis tools
- This file - Quick start guide

## Prerequisites (5 minutes)

```bash
# 1. Install dependencies
cd /path/to/momentum_signals_exploration
pip install -r requirements.lock --break-system-packages

# 2. Set Alpaca credentials (IMPORTANT - don't commit these)
export APCA_API_KEY_ID="your-api-key-id"
export APCA_API_SECRET_KEY="your-secret-key"

# 3. Verify credentials work
python main.py --universe sp500 --action scan
```

If you see results (top 20 gainers/losers), you're good to go.

## Experiment 1: Compare Ranking Methods (30 minutes)

**Goal:** See how different ranking methods rank the same symbols differently.

```bash
# Step 1: Run ranking comparison on live data
cd analysis
python ranking_comparison.py --universe sp500 --symbols 500

# Step 2: Open the generated report
cat ranking_comparison_report.json
```

**What to look for:**
- How many symbols appear in top-20 across all 3 methods? (target: 8-12)
- Which method gives the highest confidence signals?
- Do different methods rank different symbols?

**Example Output:**
```
📊 Method 1: Standard Ranking (by momentum score)
  Top symbol: TSLA (momentum: 0.0245)

🔍 Method 2: Volume-Weighted Momentum
  Top symbol: NVDA (volume: 145000000)

🔍 Method 3: Surprise Factor (Breakout Signal)
  Top symbol: AAPL (surprise: high volume + big move)

📊 Top-20 Overlap:
  All 3 methods: 9 symbols
  Consensus signals: ['TSLA', 'NVDA', 'AAPL', ...]
```

**Next:** Use the consensus signals (all 3 methods agree) for your highest confidence trades.

---

## Experiment 2: Test DataBento for Scaling (45 minutes)

**Goal:** Verify DataBento works as a scalable alternative for 5000+ symbols.

```bash
# Step 1: Test data consistency (Compare Alpaca vs DataBento)
cd analysis
python databento_integration.py --test-consistency --symbols 50

# Step 2: Test latency at different sizes
python databento_integration.py --test-latency

# Step 3: Consolidate findings
cat databento_integration_report.json | grep -A 20 "consistency_test"
```

**What to look for:**
- Are Alpaca and DataBento prices consistent? (target: <0.5% difference)
- Which source is faster? (usually DataBento)
- Can you scan 5000 symbols in <2 minutes?

**Expected Results:**
```
✅ Data Consistency:
   Avg price diff: 0.003% ✓
   Avg volume diff: 1.2% ✓

⏱️ Latency:
   100 symbols: 0.45s (Alpaca) vs 0.42s (DataBento)
   500 symbols: 2.1s (Alpaca) vs 1.8s (DataBento) ✓

📊 Scaling to 5000:
   Estimated scan time: 1-2 minutes ✓
```

**Next:** Use Alpaca for real-time (S&P 500), DataBento for bulk (extended universes).

---

## Experiment 3: Backtest Ranking Methods (2-4 hours)

**Goal:** See which ranking method performed best historically.

```bash
# Step 1: Backtest on 30 days of data
cd analysis
python backtest_ranking_methods.py --period 1h --lookback 30

# Step 2: Extend to 60 days
python backtest_ranking_methods.py --period 1h --lookback 60

# Step 3: Review results
cat ranking_backtest_report.json | grep -A 30 "comparison"
```

**What to look for:**
- Which method has highest win rate? (target: >55%)
- Which has best Sharpe ratio? (target: >1.5)
- Is performance consistent across timeframes?

**Expected Results:**
```
BACKTEST RESULTS
================
#1: surprise_factor
  Score: 0.95
  Win rate: 58%
  Avg return: 0.25%
  Sharpe ratio: 1.8

#2: volume_weighted
  Score: 0.82
  Win rate: 55%
  Avg return: 0.18%
  Sharpe ratio: 1.6

#3: standard
  Score: 0.68
  Win rate: 52%
  Avg return: 0.12%
  Sharpe ratio: 1.2
```

**Next:** Deploy the winner (usually Surprise Factor) to production.

---

## Experiment 4: Generate Analysis Dashboard (15 minutes)

**Goal:** Consolidate all analysis into an interactive report.

```bash
# Step 1: Generate all analysis outputs
cd analysis
python analysis_dashboard.py

# Step 2: Open in browser
open momentum_analysis_dashboard.html
# Or: xdg-open momentum_analysis_dashboard.html (Linux)
# Or: start momentum_analysis_dashboard.html (Windows)
```

**Dashboard includes:**
- Side-by-side ranking method comparison
- DataBento integration status
- Backtest performance rankings
- High-confidence consensus signals
- Deployment recommendations
- Production checklist

---

## Production Deployment Checklist

Once analysis is complete, follow this to deploy:

```bash
# Step 1: Review analysis dashboard
# (Make sure surprise_factor is the winner)
open analysis/momentum_analysis_dashboard.html

# Step 2: Update main scanner config for best method
# (Already defaults to all 3, you can optimize in filters.json)
cat config/filters.json

# Step 3: Test on live data one final time
python main.py --universe sp500 --top-n 20

# Step 4: Commit to GitHub
git add momentum_signals_exploration/
git commit -m "feat: add ranking methods experimentation toolkit

- Ranking comparison tool for live method testing
- DataBento integration testing and validation
- Backtesting framework for ranking method performance
- Interactive analysis dashboard
- Complete experimentation guide"

git push origin main

# Step 5: Deploy to production scheduler
# (See MOMENTUM_SCANNER_GUIDE.md for deployment options)
```

---

## Common Scenarios & How to Handle Them

### Scenario 1: Different Methods Give Different Top Symbols

**What to do:**
- Use consensus signals (all 3 methods agree) for highest confidence
- Create filter for "all 3 methods ranked in top 20"
- Use alternative methods for secondary signals

```python
from analysis.ranking_comparison import RankingComparison

comp = RankingComparison()
results = comp.run_scan('sp500', 500)
overlap = comp.analyze_ranking_overlap(results, top_n=20)
print(f"Consensus signals: {overlap['symbols_in_all_three']}")
```

### Scenario 2: Alpaca Rate Limits (200 req/min)

**What to do:**
- Use DataBento for universes >500 symbols
- Run S&P 500 with Alpaca (no limits)
- Keep 5-minute delay between scans

```python
scanner = MomentumScanner(data_source='databento')
results = scanner.run_full_scan('all', 5000)  # DataBento handles bulk
```

### Scenario 3: Live Results Don't Match Backtest

**Common reasons:**
- Slippage (execution delayed by 1-2 seconds)
- Fills below/above target price
- Symbols moving fast (especially small caps)

**What to do:**
- Monitor first 1-2 weeks closely
- Track daily win rate vs backtest (target: 45%+)
- Increase filters (min_volume, min_liquidity) if needed
- Switch to consensus signals only

### Scenario 4: Want to Use with Volatility Predictor

Already set up for you!

```python
from examples.trading_integration import MomentumTradingStrategy
from core.hybrid_predictor import HybridPredictor

strategy = MomentumTradingStrategy()
predictor = HybridPredictor()

# Scanner finds momentum signals
momentum_signals = strategy.scan_and_filter('sp500')

# Volatility predictor sizes positions
signals_with_vol = strategy.check_volatility(momentum_signals, predictor)

# Generate trades
trades = strategy.generate_signals(signals_with_vol)
```

---

## Quick Reference Commands

```bash
# Scanner basics
python main.py --universe sp500 --action scan
python main.py --action show-symbols --universe sp500

# Analysis - Quick comparison
cd analysis && python ranking_comparison.py

# Analysis - Deep dive
cd analysis && python ranking_comparison.py --symbols 500
cd analysis && python databento_integration.py
cd analysis && python backtest_ranking_methods.py --lookback 60

# Dashboard
cd analysis && python analysis_dashboard.py
open momentum_analysis_dashboard.html

# View reports
cat ranking_comparison_report.json | python -m json.tool
cat databento_integration_report.json | python -m json.tool
cat ranking_backtest_report.json | python -m json.tool
```

---

## Typical Timeline

| Phase | Task | Duration | Output |
|-------|------|----------|--------|
| Setup | Install, credentials, first scan | 15 min | Verify scanner works |
| Experiment 1 | Ranking methods comparison | 30 min | `ranking_comparison_report.json` |
| Experiment 2 | DataBento testing | 45 min | `databento_integration_report.json` |
| Experiment 3 | Backtesting (30-60 days) | 2-4 hr | `ranking_backtest_report.json` |
| Analysis | Dashboard & consolidation | 15 min | `momentum_analysis_dashboard.html` |
| Deployment | Review checklist, commit, deploy | 30 min | Live scanning |
| Validation | Paper trading, monitoring | 1-2 wk | Fine-tuned filters |

**Total: ~8-10 hours to go from "let's experiment" to "live in production"**

---

## Key Metrics to Track

Once deployed, monitor these daily:

- **Win Rate** - % of signals with positive next return (target: 50%+)
- **Avg Return** - Average return per signal (target: 0.15%+)
- **Max Drawdown** - Worst consecutive losing streak (limit: <5 losses)
- **Data Freshness** - Time from signal to data (target: <10 seconds)
- **Symbol Coverage** - How many symbols in universe you're tracking (target: 500+)

---

## Getting Help

If something doesn't work:

1. **Check the logs:** `python main.py 2>&1 | head -50`
2. **Review README.md** - Troubleshooting section
3. **Check ANALYSIS_TOOLS_README.md** - Per-tool guide
4. **Review example code** - `examples/trading_integration.py`

---

## Next Level: Custom Experiments

Once comfortable with the basics:

1. **Create custom filter profiles** - Edit `config/filters.json`
2. **Test alternative universes** - Edit `symbols.py`
3. **Combine ranking methods** - Use weighted averaging
4. **Integrate with your portfolio** - Connect to `examples/trading_integration.py`
5. **Add position sizing** - Combine with volatility predictor

---

**You're ready to experiment! Start with Experiment 1 and work through to production. Good luck! 🚀**
