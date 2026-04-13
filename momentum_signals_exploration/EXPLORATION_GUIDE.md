# Momentum Signals Exploration

This folder contains the momentum scanner with experimental features and analysis tools for optimizing ranking methods and data sources.

## Current Status

✅ **Core Implementation Complete:**
- Hourly momentum scanner (Alpaca API)
- Symbol universe support (500 to 5000+)
- 6 filter types (volume, price, magnitude, direction, liquidity, blacklist)
- 3 ranking methods (standard, volume-weighted, surprise factor)
- Alert system (console, Slack, email, webhook)
- Configuration-driven approach

## Experimentation Phase

### 1. Ranking Methods Deep-Dive

**Implemented Rankings:**
- `rank_by_metric()` - Standard sorting by momentum score
- `rank_by_volume_weighted_momentum()` - High volume = more confidence
- `rank_by_surprise_factor()` - Big moves on high volume (breakout signal)

**To Experiment:**
```python
from scanner import MomentumScanner
from filters import RankingEngine

scanner = MomentumScanner()
results = scanner.run_full_scan('sp500', 100)

# Test different ranking methods
engine = RankingEngine()
ranked1 = engine.rank_by_metric(results, metric='combined')
ranked2 = engine.rank_by_volume_weighted_momentum(results)
ranked3 = engine.rank_by_surprise_factor(results)

# Compare results - same symbols ranked differently
```

**Next Steps:**
- Analyze which ranking method correlates with next-hour/next-day returns
- Test inverse rankings (lowest momentum, biggest losers)
- Create hybrid ranking (weighted average of multiple methods)
- Backtest each ranking on historical data

### 2. Data Source Expansion

**Current Sources:**
- **Alpaca**: Free, 200 req/min, unlimited historical, S&P 500 easy
- **DataBento**: Free tier covers 5000+ symbols with 2-year history

**To Experiment:**
```python
# Test Alpaca for S&P 500
scanner_alpaca = MomentumScanner(data_source='alpaca')
sp500_results = scanner_alpaca.run_full_scan('sp500')

# Test DataBento for extended universe
scanner_databento = MomentumScanner(data_source='databento')
russell3k_results = scanner_databento.run_full_scan('all')

# Compare results and latency
```

**Next Steps:**
- Measure query latency and data freshness for each source
- Compare price data consistency between sources
- Test DataBento subscription tiers for cost/coverage trade-off
- Build fallback logic (use Alpaca if DataBento unavailable)

### 3. Filter Optimization

**Current Filters:**
- min_volume: 100,000
- min_price: $5.00
- max_price: $1,000.00
- min_magnitude: 0.5% (0.005)
- min_liquidity_score: 500,000

**To Experiment:**
```python
# Test different filter thresholds
filter_profiles = {
    'conservative': {'min_volume': 500000, 'min_magnitude': 0.01},
    'aggressive': {'min_volume': 50000, 'min_magnitude': 0.001},
    'large_cap': {'min_price': 10, 'min_liquidity_score': 1000000},
    'small_cap': {'min_price': 5, 'max_price': 100}
}

for profile_name, filters in filter_profiles.items():
    # Load config with filters
    # Run scan
    # Compare results
```

**Next Steps:**
- Create filter profiles for different trading strategies
- Measure false positive rates (filtered symbols that reverse)
- Optimize for your risk tolerance

### 4. Performance Metrics

**To Track:**
- Symbols in top-20 before filters vs after filters
- Average magnitude of filtered results
- Ratio of gainers/losers
- Data freshness (latency from exchange to our fetch)
- API cost per scan

**Output Format:**
```json
{
  "scan_timestamp": "2026-04-02T14:30:00Z",
  "universe": "sp500",
  "symbols_scanned": 500,
  "symbols_after_filters": 120,
  "top_gainers": [...],
  "top_losers": [...],
  "metrics": {
    "avg_momentum": 0.015,
    "median_volume": 2500000,
    "data_freshness_ms": 250
  }
}
```

## Quick Start for Experimentation

```bash
# 1. Install dependencies
pip install -r requirements.lock --break-system-packages

# 2. Set Alpaca credentials
export APCA_API_KEY_ID="your-key"
export APCA_API_SECRET_KEY="your-secret"

# 3. Run single scan with all ranking methods
python main.py --universe sp500 --action scan

# 4. Or use the examples
python examples/trading_integration.py
```

## Files Structure

- `scanner.py` - Core MomentumScanner class (hourly data, momentum calculation)
- `filters.py` - MomentumFilters and RankingEngine for ranking experimentation
- `alerts.py` - Multi-channel alert system
- `symbols.py` - Symbol universe management (500 to 5000+)
- `main.py` - CLI entry point
- `config.json` - Configuration (data source, filters, alerts)
- `config/filters.json` - Alternative filter profiles
- `examples/trading_integration.py` - Integration with volatility predictor
- `README.md` - Full documentation

## Next Integration Points

1. **Backtest Engine**: Historical analysis of ranking method performance
2. **Volatility Predictor**: Combine momentum signals with volatility predictions
3. **Position Sizing**: Use momentum magnitude for position sizing
4. **Risk Management**: Alert on unexpected reversals or volume drops

## Notes

- All free tier APIs (Alpaca 200/min, DataBento free tier)
- Paper trading ready (no real money required for testing)
- Horizontal scaling to 5000+ symbols via DataBento
- Zero OpEx for basic usage
