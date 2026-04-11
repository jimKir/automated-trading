#!/usr/bin/env python3
"""
Hourly Momentum Scanner - Identify top gainers/losers across US equities.

Efficiently scans 500-5000 symbols for momentum every hour.
Cost: $0 (Alpaca free tier)
"""

import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumScanner:
    """Core momentum scanning engine."""

    def __init__(self, data_source="alpaca"):
        """
        Initialize scanner.

        Args:
            data_source: 'alpaca' or 'databento'
        """
        self.data_source = data_source
        self.results = {}

        if data_source == "alpaca":
            from alpaca_trade_api import REST

            self.api = REST()
        elif data_source == "databento":
            import databento as db

            self.api = db.Historical()

    def fetch_hourly_bars(self, symbols: list[str], limit: int = 5) -> dict:
        """
        Fetch hourly bars for all symbols.

        Args:
            symbols: List of stock symbols
            limit: Number of hours to fetch

        Returns:
            Dict with OHLCV data for each symbol
        """
        if self.data_source == "alpaca":
            return self._fetch_alpaca(symbols, limit)
        return self._fetch_databento(symbols, limit)

    def _fetch_alpaca(self, symbols: list[str], limit: int) -> dict:
        """Fetch from Alpaca API using current API (get_bars instead of deprecated get_barset)."""
        logger.info(f"Fetching {len(symbols)} symbols from Alpaca...")

        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=limit)

            bars_dict = {}
            for symbol in symbols:
                try:
                    bars = self.api.get_bars(symbol, timeframe="1h", start=start_time, end=end_time)

                    if bars is None or len(bars) == 0:
                        logger.debug(f"No data for {symbol}")
                        continue

                    # Convert bars to list of dicts
                    bars_list = []
                    for bar in bars:
                        # Handle Bar object from alpaca-trade-api
                        if hasattr(bar, "__dict__"):
                            bar_dict = {
                                "open": float(bar.open),
                                "high": float(bar.high),
                                "low": float(bar.low),
                                "close": float(bar.close),
                                "volume": int(bar.volume),
                                "timestamp": bar.timestamp,
                            }
                        elif isinstance(bar, dict):
                            bar_dict = bar
                        else:
                            continue
                        bars_list.append(bar_dict)

                    if len(bars_list) > 0:
                        bars_dict[symbol] = bars_list
                        logger.debug(f"  {symbol}: {len(bars_list)} bars")

                except Exception as e:
                    logger.debug(f"Could not fetch {symbol}: {str(e)[:100]}")
                    continue

            logger.info(f"✓ Fetched {len(bars_dict)} symbols")
            return bars_dict
        except Exception as e:
            logger.error(f"✗ Fetch failed: {e}")
            raise

    def _fetch_databento(self, symbols: list[str], limit: int) -> dict:
        """Fetch from DataBento API."""
        logger.info(f"Fetching {len(symbols)} symbols from DataBento...")

        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=limit)

            data = self.api.timeseries.get_range(
                dataset="GLBX.MBO",
                symbols=symbols,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                timeframe="1h",
            )
            logger.info("✓ Fetched data")
            return data
        except Exception as e:
            logger.error(f"✗ Fetch failed: {e}")
            raise

    def calculate_momentum(self, bars: dict) -> dict:
        """
        Calculate momentum metrics for each symbol.

        Args:
            bars: OHLCV data from API

        Returns:
            Dict with momentum scores and metrics
        """
        logger.info("Calculating momentum...")
        momentum_scores = {}

        for symbol, bar_data in bars.items():
            try:
                # Handle both list and dict formats
                if isinstance(bar_data, list) and len(bar_data) >= 2:
                    # List format (Alpaca get_bars returns this)
                    last_bar = bar_data[-1]
                    prev_bar = bar_data[-2]

                    # Handle both dict and object formats
                    if isinstance(last_bar, dict):
                        open_price = last_bar.get("open") or last_bar.get("o")
                        close_price = last_bar.get("close") or last_bar.get("c")
                        prev_close = prev_bar.get("close") or prev_bar.get("c")
                        volume = last_bar.get("volume") or last_bar.get("v")
                    else:
                        # Object format
                        open_price = getattr(last_bar, "open", None) or getattr(last_bar, "o", None)
                        close_price = getattr(last_bar, "close", None) or getattr(
                            last_bar, "c", None
                        )
                        prev_close = getattr(prev_bar, "close", None) or getattr(
                            prev_bar, "c", None
                        )
                        volume = getattr(last_bar, "volume", None) or getattr(last_bar, "v", None)

                elif isinstance(bar_data, dict) and len(bar_data) >= 2:
                    # Dict format
                    bars_list = list(bar_data.values()) if isinstance(bar_data, dict) else bar_data
                    if len(bars_list) < 2:
                        continue
                    last_bar = bars_list[-1]
                    prev_bar = bars_list[-2]

                    if isinstance(last_bar, dict):
                        open_price = last_bar.get("open") or last_bar.get("o")
                        close_price = last_bar.get("close") or last_bar.get("c")
                        prev_close = prev_bar.get("close") or prev_bar.get("c")
                        volume = last_bar.get("volume") or last_bar.get("v")
                    else:
                        open_price = getattr(last_bar, "open", None) or getattr(last_bar, "o", None)
                        close_price = getattr(last_bar, "close", None) or getattr(
                            last_bar, "c", None
                        )
                        prev_close = getattr(prev_bar, "close", None) or getattr(
                            prev_bar, "c", None
                        )
                        volume = getattr(last_bar, "volume", None) or getattr(last_bar, "v", None)
                else:
                    continue

                # Ensure we have valid numbers
                if not all([open_price, close_price, prev_close, volume]):
                    continue

                # Calculate metrics
                intra_hour_momentum = (close_price - open_price) / open_price
                hourly_return = (close_price - prev_close) / prev_close

                momentum_scores[symbol] = {
                    "intra_momentum": intra_hour_momentum,  # This hour open->close
                    "hourly_return": hourly_return,  # Previous close->current close
                    "combined_momentum": intra_hour_momentum + hourly_return,
                    "price": close_price,
                    "volume": volume,
                    "open": open_price,
                    "close": close_price,
                }
            except (KeyError, TypeError, IndexError, AttributeError) as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        logger.info(f"✓ Calculated momentum for {len(momentum_scores)} symbols")
        self.results = momentum_scores
        return momentum_scores

    def rank_by_momentum(self, metric: str = "intra_momentum", top_n: int = 20) -> list[tuple]:
        """
        Rank symbols by momentum metric.

        Args:
            metric: 'intra_momentum', 'hourly_return', or 'combined_momentum'
            top_n: Number of top results to return

        Returns:
            List of (symbol, metrics_dict) tuples, sorted by metric
        """
        ranked = sorted(self.results.items(), key=lambda x: x[1][metric], reverse=True)
        return ranked[:top_n]

    def get_top_gainers(self, top_n: int = 20) -> list[tuple]:
        """Get top gainers by intra-hour momentum."""
        return self.rank_by_momentum("intra_momentum", top_n)

    def get_top_losers(self, top_n: int = 20) -> list[tuple]:
        """Get top losers by intra-hour momentum."""
        ranked = sorted(self.results.items(), key=lambda x: x[1]["intra_momentum"])
        return ranked[:top_n]

    def format_results(self, results: list[tuple]) -> str:
        """
        Format results for display.

        Args:
            results: List of (symbol, metrics) tuples

        Returns:
            Formatted string
        """
        output = []
        output.append("=" * 80)
        output.append(
            f"{'Symbol':<8} {'Intra %':<10} {'Hourly %':<10} {'Price':<10} {'Volume':<15}"
        )
        output.append("-" * 80)

        for symbol, metrics in results:
            output.append(
                f"{symbol:<8} "
                f"{metrics['intra_momentum'] * 100:>8.2f}% "
                f"{metrics['hourly_return'] * 100:>8.2f}% "
                f"${metrics['price']:>8.2f} "
                f"{metrics['volume']:>13,.0f}"
            )

        output.append("=" * 80)
        return "\n".join(output)

    def run_full_scan(self, symbols: list[str], top_n: int = 20) -> tuple[list[tuple], list[tuple]]:
        """
        Run complete scan: fetch -> calculate -> rank.

        Args:
            symbols: List of symbols to scan
            top_n: Number of top results to return

        Returns:
            (top_gainers, top_losers)
        """
        logger.info(f"Starting full scan of {len(symbols)} symbols...")
        start_time = datetime.now()

        # Fetch data
        bars = self.fetch_hourly_bars(symbols)

        # Calculate momentum
        self.calculate_momentum(bars)

        # Get results
        gainers = self.get_top_gainers(top_n)
        losers = self.get_top_losers(top_n)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ Scan complete in {elapsed:.1f} seconds")

        return gainers, losers
