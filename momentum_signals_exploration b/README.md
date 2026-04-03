# Hourly Momentum Scanner

Production-ready scanner to identify equities with highest momentum on an hourly basis.

**Cost:** $0 (Alpaca free tier)
**Universe:** 500-5000+ US equities
**Frequency:** Hourly (run every hour at :00)
**Latency:** 5-10 seconds per scan

---

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
pip install alpaca-trade-api pandas numpy schedule requests python-dotenv
```

### 2. Set Up Alpaca Account

```bash
# Create account at https://alpaca.markets (free, $2k+ required)
# Get API credentials from dashboard
export APCA_API_KEY_ID="your_key"
export APCA_API_SECRET_KEY="your_secret"
```

### 3. Run a Scan

```bash
# Scan S&P 500 once
python main.py --universe sp500 --action scan

# Show top gainers + losers
# Results printed to console + (optionally) Slack/Email
```

### 4. Schedule Hourly Scans

```bash
# Run scan every hour
python main.py --universe sp500 --schedule-hourly

# Press Ctrl+C to stop
```

---

## Features

### 1. **Multiple Universes**

```bash
# S&P 500 (500 symbols)
python main.py --universe sp500

# Sector leaders (100 symbols)
python main.py --universe sectors

# Nasdaq 100 (100 symbols)
python main.py --universe nasdaq100

# All US equities (5000+ symbols) - requires DataBento
python main.py --universe all --data-source databento

# Custom CSV
python main.py --universe symbols.csv
```

### 2. **Momentum Metrics Calculated**

```
Per symbol:
├─ Intra-hour momentum: (close - open) / open
├─ Hourly return: (current close - previous close) / previous close
├─ Price: Current price
└─ Volume: Trading volume
```

### 3. **Smart Filters**

```json
{
  "min_volume": 100000,        // Remove low-liquidity stocks
  "min_price": 5.0,            // Avoid penny stocks
  "max_price": 1000.0,         // Avoid ultra-expensive
  "min_magnitude": 0.005,      // Only 0.5%+ moves
  "direction": "both",         // up/down/both
  "min_liquidity_score": 500000 // price × volume filter
}
```

### 4. **Alert Channels**

```
├─ Console (always)
├─ Slack (optional)
├─ Email (optional)
└─ Custom webhook (optional)
```

### 5. **Advanced Ranking**

```python
# Standard momentum ranking
ranked = scanner.get_top_gainers(top_n=20)

# Volume-weighted momentum (high volume = more confidence)
ranked = RankingEngine.rank_by_volume_weighted_momentum(results)

# Surprise factor (big moves on high volume)
ranked = RankingEngine.rank_by_surprise_factor(results)
```

---

## Configuration

### Main Config (`config.json`)

```json
{
  "data_source": "alpaca",    // or "databento"
  "universe": "sp500",
  "filters": { ... },
  "alerts": {
    "console": true,
    "slack": true,
    "email": true
  },
  "slack_webhook": "https://...",
  "email_to": "you@example.com",
  "smtp_config": { ... }
}
```

### Filters Config (`config/filters.json`)

Override default filters:

```json
{
  "min_volume": 500000,    // More selective
  "min_magnitude": 0.01,   // Only 1%+ moves
  "direction": "up"        // Only gainers
}
```

---

## Usage Examples

### Example 1: Scan S&P 500, Get Top 20 Gainers

```bash
python main.py --universe sp500 --action scan --top-n 20
```

**Output:**
```
HOURLY MOMENTUM SCAN - 2024-01-15 14:00:00
==================================================
🚀 TOP GAINERS:
Symbol   Momentum    Hourly %    Price       Volume
NVDA     +2.45%      +1.82%      $842.50     5,234,000
MSFT     +1.78%      +0.95%      $418.25     4,102,000
...
```

### Example 2: Scan with Custom Filters

```bash
python main.py \
  --universe sp500 \
  --action scan \
  --filters-config config/filters.json
```

Filter high-volume momentum:
```json
{
  "min_volume": 1000000,
  "min_magnitude": 0.01,
  "direction": "up"
}
```

### Example 3: Schedule Hourly with Slack Alerts

```bash
# Edit config.json with your Slack webhook
python main.py --universe sp500 --schedule-hourly
```

Every hour at :00, scan runs and posts to Slack:
```
📊 Hourly Momentum Scan - 14:00:00
Top Gainers:
• NVDA: +2.45% ($842.50)
• MSFT: +1.78% ($418.25)
...
```

### Example 4: Show Available Symbols

```bash
python main.py --universe sp500 --action show-symbols
```

---

## API Integration Details

### Alpaca (FREE Tier)

```
Rate limit: 200 requests/minute
You get: 1 call = all 500 symbols
Cost: $0 (requires $2k+ account minimum)
```

**Example:**
```python
from alpaca_trade_api import REST

api = REST()
bars = api.get_barset(
    symbols=['AAPL', 'MSFT', ...],  # Up to 500
    timeframe='1h',
    limit=5
)
```

### DataBento (FREE Tier)

```
Rate limit: Generous
You get: 1 call = 5000+ symbols
Cost: $0 (free tier available)
```

**Better for large universes (5000+ symbols).**

---

## Production Setup

### 1. System Scheduler (Cron)

```bash
# Edit crontab
crontab -e

# Add this line (runs at top of every hour)
0 * * * * cd /path/to/momentum_scanner && python main.py --universe sp500 --action scan >> logs/scan.log 2>&1
```

### 2. Docker Container

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Run scan every hour
CMD python main.py --universe sp500 --schedule-hourly
```

```bash
# Build and run
docker build -t momentum-scanner .
docker run -e APCA_API_KEY_ID=... -e APCA_API_SECRET_KEY=... momentum-scanner
```

### 3. With Trading Integration

```python
# Use with your trading system
from scanner import MomentumScanner

scanner = MomentumScanner()
gainers, losers = scanner.run_full_scan(symbols, top_n=20)

# Get top gainer
top_symbol = gainers[0][0]
top_momentum = gainers[0][1]['intra_momentum']

# Send to your trading engine
if top_momentum > 0.02:  # 2%+ momentum
    order = place_buy_order(top_symbol, shares=10)
```

---

## Output Formats

### Console

```
HOURLY MOMENTUM SCAN - 2024-01-15 14:00:00
==================================================
🚀 TOP GAINERS:
Symbol   Momentum    Hourly %    Price       Volume
NVDA     +2.45%      +1.82%      $842.50     5,234,000
```

### Slack

```
📊 Hourly Momentum Scan - 14:00:00

Top Gainers:
• NVDA: +2.45% ($842.50)
• MSFT: +1.78% ($418.25)

Top Losers:
• XYZ: -1.55% ($25.30)
```

### Email (HTML)

```
Nice HTML table with colors:
✓ Green for gainers
✗ Red for losers
```

### Webhook (JSON)

```json
{
  "timestamp": "2024-01-15T14:00:00Z",
  "gainers": [
    {"symbol": "NVDA", "momentum": 0.0245, "price": 842.50},
    {"symbol": "MSFT", "momentum": 0.0178, "price": 418.25}
  ],
  "losers": [...]
}
```

---

## Cost Breakdown

| Universe | API Calls/Hour | Cost/Month | Latency |
|----------|---|---|---|
| S&P 500 (500) | 1 | $0 | 5 sec |
| All stocks (5000) | 1 | $0 | 10 sec |
| Real-time (paid) | N/A | $50+ | <1 sec |

**Total monthly cost: $0** ✨

---

## Troubleshooting

### "Authentication error"

```bash
# Check Alpaca credentials
export APCA_API_KEY_ID="xxx"
export APCA_API_SECRET_KEY="yyy"

# Test connection
python -c "from alpaca_trade_api import REST; print(REST().get_account())"
```

### "Rate limit exceeded"

You hit API limits. Wait a minute and retry.

```
For S&P 500 scan: Should never happen (1 call/hour)
For 5000 symbols: May need batching (DataBento instead)
```

### "Empty results"

```bash
# Check filters are not too restrictive
python main.py --universe sp500 --top-n 50  # Show more

# Check data source has data
python -c "from symbols import get_symbol_list; print(get_symbol_list('sp500')[:10])"
```

---

## Files

```
momentum_scanner/
├─ scanner.py           # Core scanning engine
├─ filters.py           # Filtering & ranking
├─ alerts.py            # Alert delivery
├─ symbols.py           # Universe definitions
├─ main.py              # Entry point
├─ config.json          # Main configuration
├─ config/
│  └─ filters.json      # Filter configuration
├─ logs/                # Log files (created on run)
└─ README.md            # This file
```

---

## Next Steps

1. **Run your first scan:** `python main.py --universe sp500`
2. **Integrate with Slack:** Set `slack_webhook` in `config.json`
3. **Schedule hourly:** `python main.py --universe sp500 --schedule-hourly`
4. **Connect to trading system:** Use `MomentumScanner` class directly

---

## Advanced Usage

### Custom Ranking

```python
from filters import RankingEngine

# Use volume-weighted momentum
ranked = RankingEngine.rank_by_volume_weighted_momentum(results, volume_weight=0.4)

# Use surprise factor
ranked = RankingEngine.rank_by_surprise_factor(results)
```

### Multi-Timeframe

```python
# Current system: 1-hour bars
# Could extend to:
# - 15-minute bars
# - 5-minute bars
# - Daily bars
```

### Integration with Volatility Model

```python
# Combine momentum scan with volatility predictions
from hybrid_predictor import HybridPredictor

predictor = HybridPredictor(...)

for symbol, metrics in gainers[:5]:
    volatility = predictor.predict(features)
    if volatility < 0.02:  # Low volatility
        place_order(symbol)
```

---

## Support & Maintenance

- Logs saved to `logs/scan.log`
- Check Alpaca API status: https://status.alpaca.markets
- Update symbol universe monthly (S&P 500 changes)
- Monitor alert delivery (Slack/Email)

**That's it! Happy scanning! 🚀**
