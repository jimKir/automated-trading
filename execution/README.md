# execution/

Live trading engine and broker adapters.

## `live_engine.py` -- LiveEngine

Main orchestrator that wires all modules into a continuous trading loop (default 60-second cycle).

**Trading cycle pipeline:**
1. Fetch account info from broker
2. Check Intraday Shock Detector (VIX-based scaling)
3. Run circuit breaker checks (drawdown, daily loss)
4. Determine if rebalance is due (daily/weekly/biweekly/adaptive)
5. Fetch 400 days of data
6. Generate signals via SignalGenerator (regime dispatch)
7. Apply EWS portfolio-wide scale
8. Apply vol-engine per-symbol scales
9. Apply PositionAnomalyScorer per-symbol scales
10. Compute target weights via Portfolio
11. Generate and execute delta orders
12. Set stop-losses

**Rebalance cadence:** Configurable as daily, weekly, biweekly, monthly, or adaptive (biweekly in GREEN, weekly in YELLOW+). VIX-spike forced rebalance on +20% single-day jumps.

**Broker selection:** `get_broker(config)` routes to AlpacaBroker (paper/live), PaperBroker (local sim), IBKRBroker, or BinanceBroker based on mode and available credentials.

## `hourly_entry_timer.py` -- HourlyEntryTimer

Intraday entry timing gate. Wired into LiveEngine but provides minimal OOS edge.

- **Equity:** Preferred entry at 12:00 ET (checks VWAP position + 3-bar momentum). Hard fallback at 13:05 ET.
- **Crypto:** Session window 14:00-17:00 UTC, enters if RSI < 45.
- **Bypassed:** GLD, TLT, SHY, AGG, IEF, BND (always enter immediately).

## Broker Adapters

All implement `BrokerBase` abstract interface (Order, AccountInfo dataclasses).

| Broker | File | Asset Classes | Library |
|---|---|---|---|
| Alpaca | `alpaca_broker.py` | US equities, limited crypto | `alpaca-py` |
| Binance | `binance_broker.py` | Crypto (spot + futures) | `python-binance` |
| IBKR | `ibkr_broker.py` | All (equities, futures, forex, crypto) | `ib_insync` |
| Paper | `paper_broker.py` | All (local simulation) | yfinance for prices |

`broker_base.py` defines: `OrderType` (MARKET/LIMIT/STOP/STOP_LIMIT), `OrderSide` (BUY/SELL), `OrderStatus`, `Order` dataclass, `AccountInfo` dataclass.

PaperBroker simulates fills with configurable commission (0.1%) and slippage (0.05%), starting equity 25k.
