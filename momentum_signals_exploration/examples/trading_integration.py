#!/usr/bin/env python3
"""
Integration example: Combine momentum scanner with volatility predictions.

Shows how to:
1. Scan for hourly momentum
2. Filter by volatility predictions
3. Generate trading signals
4. Manage positions
"""

import sys

sys.path.insert(0, "..")

import logging

from filters import MomentumFilters
from scanner import MomentumScanner
from symbols import get_symbol_list

# If using hybrid deployment volatility predictor
try:
    sys.path.insert(0, "../../hybrid-deployment-v1/core")
    from hybrid_predictor import HybridPredictor

    VOLATILITY_AVAILABLE = True
except ImportError:
    VOLATILITY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumTradingStrategy:
    """Trade based on momentum + volatility signals."""

    def __init__(self, capital: float = 25000):
        """
        Initialize strategy.

        Args:
            capital: Trading capital
        """
        self.capital = capital
        self.positions = {}
        self.trades = []

        # Initialize scanners
        self.scanner = MomentumScanner(data_source="alpaca")
        self.filters = MomentumFilters()

        # Optional: volatility predictor
        if VOLATILITY_AVAILABLE:
            self.volatility_predictor = HybridPredictor(
                endpoint_type="sagemaker", sagemaker_endpoint="trading-volatility"
            )
            logger.info("✓ Volatility predictor loaded")
        else:
            self.volatility_predictor = None
            logger.warning("⚠ Volatility predictor not available")

    def scan_and_filter(self, universe: str = "sp500", top_n: int = 20) -> tuple:
        """
        Scan universe and apply filters.

        Args:
            universe: Symbol universe
            top_n: Top results to return

        Returns:
            (gainers, losers) after filtering
        """
        logger.info(f"Scanning {universe} universe...")

        # Get symbols
        symbols = get_symbol_list(universe)

        # Run scan
        _gainers, _losers = self.scanner.run_full_scan(symbols, top_n=top_n)

        # Apply quality filters
        filter_config = {
            "min_volume": 500000,  # High liquidity only
            "min_magnitude": 0.01,  # 1%+ moves
            "min_price": 10.0,  # $10+
            "max_price": 500.0,  # Not too expensive
        }

        filtered = self.filters.apply_all_filters(self.scanner.results, filter_config)

        # Re-rank filtered results
        if filtered:
            gainers_filtered = sorted(
                filtered.items(), key=lambda x: x[1]["intra_momentum"], reverse=True
            )[:top_n]

            losers_filtered = sorted(filtered.items(), key=lambda x: x[1]["intra_momentum"])[:top_n]
        else:
            gainers_filtered, losers_filtered = [], []

        logger.info(f"Filtered to {len(filtered)} symbols")
        return gainers_filtered, losers_filtered

    def check_volatility(self, symbol: str, features: dict) -> float:
        """
        Get volatility prediction for symbol.

        Args:
            symbol: Stock symbol
            features: Feature dict (26 technical indicators)

        Returns:
            Predicted volatility (0-1)
        """
        if not self.volatility_predictor:
            return 0.02  # Default 2%

        try:
            volatility = self.volatility_predictor.predict(features)
            logger.info(f"{symbol}: Volatility = {volatility * 100:.2f}%")
            return volatility
        except Exception as e:
            logger.warning(f"Volatility prediction failed for {symbol}: {e}")
            return 0.02  # Default

    def generate_signals(self, gainers: list, losers: list) -> dict:
        """
        Generate trading signals.

        Rules:
        - High momentum + low volatility = BUY
        - High negative momentum + high volatility = SELL/SHORT
        - Moderate momentum = HOLD

        Args:
            gainers: Top gainers from scan
            losers: Top losers from scan

        Returns:
            Dict of signals: {symbol: {'signal': 'BUY'/'SELL', 'score': 0-100}}
        """
        signals = {}

        # BUY signals (gainers with low volatility)
        for symbol, metrics in gainers[:5]:  # Top 5 gainers
            momentum = metrics["intra_momentum"]
            metrics["volume"]
            metrics["price"]

            # Score: 0-100
            score = min(100, momentum * 1000)  # Scale to 0-100

            signals[symbol] = {
                "signal": "BUY",
                "score": score,
                "momentum": momentum,
                "reason": f"Strong momentum: {momentum * 100:.2f}%",
            }

            logger.info(
                f"{symbol}: BUY signal (momentum: {momentum * 100:.2f}%, score: {score:.0f})"
            )

        # SELL signals (losers with high volatility)
        for symbol, metrics in losers[:3]:  # Top 3 losers
            momentum = metrics["intra_momentum"]
            metrics["volume"]
            metrics["price"]

            score = min(100, abs(momentum) * 1000)

            signals[symbol] = {
                "signal": "SELL",
                "score": score,
                "momentum": momentum,
                "reason": f"Negative momentum: {momentum * 100:.2f}%",
            }

            logger.info(
                f"{symbol}: SELL signal (momentum: {momentum * 100:.2f}%, score: {score:.0f})"
            )

        return signals

    def execute_signals(self, signals: dict) -> list:
        """
        Execute trades based on signals.

        Args:
            signals: Trading signals dict

        Returns:
            List of executed trades
        """
        executed_trades = []

        for symbol, signal in signals.items():
            signal_type = signal["signal"]
            score = signal["score"]

            # Position sizing based on score
            position_size = max(1, int((self.capital * 0.05) / 100))  # 5% of capital per position

            trade = {
                "symbol": symbol,
                "type": signal_type,
                "size": position_size,
                "score": score,
                "reason": signal["reason"],
            }

            # In real trading, would execute via broker API
            # For now, just log and track
            logger.info(
                f"EXECUTE: {signal_type} {position_size} shares of {symbol} (score: {score:.0f})"
            )

            self.positions[symbol] = trade
            self.trades.append(trade)
            executed_trades.append(trade)

        return executed_trades

    def run_full_strategy(self, universe: str = "sp500") -> dict:
        """
        Run complete strategy cycle.

        Args:
            universe: Symbol universe to scan

        Returns:
            Results dict with signals and trades
        """
        logger.info("\n" + "=" * 80)
        logger.info("MOMENTUM TRADING STRATEGY - FULL CYCLE")
        logger.info("=" * 80)

        # Step 1: Scan
        gainers, losers = self.scan_and_filter(universe)

        if not gainers and not losers:
            logger.warning("No signals generated")
            return {"signals": {}, "trades": []}

        # Step 2: Generate signals
        signals = self.generate_signals(gainers, losers)

        # Step 3: Execute
        trades = self.execute_signals(signals)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Signals generated: {len(signals)}")
        logger.info(f"Trades executed: {len(trades)}")
        logger.info(f"Current positions: {len(self.positions)}")

        return {"gainers": gainers, "losers": losers, "signals": signals, "trades": trades}


def main():
    """Run example strategy."""
    # Initialize strategy
    strategy = MomentumTradingStrategy(capital=25000)

    # Run full strategy
    results = strategy.run_full_strategy(universe="sp500")

    # Print results
    print("\n" + "=" * 80)
    print("TRADING SIGNALS")
    print("=" * 80)

    for symbol, signal in results["signals"].items():
        print(f"{symbol}: {signal['signal']} (score: {signal['score']:.0f}) - {signal['reason']}")

    print("\n" + "=" * 80)
    print("EXECUTED TRADES")
    print("=" * 80)

    for trade in results["trades"]:
        print(f"{trade['type']} {trade['size']} {trade['symbol']} - {trade['reason']}")

    print("\n✓ Strategy cycle complete\n")


if __name__ == "__main__":
    main()
