#!/usr/bin/env python3
"""
Hourly Momentum Scanner - Main entry point.

Usage:
    python main.py --universe sp500 --action scan
    python main.py --universe all --action scan --filters config/filters.json
    python main.py --universe sp500 --schedule-hourly
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import schedule
import time

from scanner import MomentumScanner
from filters import MomentumFilters, RankingEngine
from alerts import AlertManager
from symbols import get_symbol_list

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_file: str = 'config.json') -> dict:
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent / config_file

    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            'data_source': 'alpaca',
            'filters': {
                'min_volume': 100000,
                'min_price': 5.0,
                'max_price': 1000.0,
                'min_magnitude': 0.005,
                'direction': 'both'
            },
            'alerts': {
                'console': True,
                'slack': False,
                'email': False
            }
        }


def run_scan(universe: str = 'sp500',
            filters_config: dict = None,
            data_source: str = 'alpaca',
            top_n: int = 20) -> tuple:
    """
    Run momentum scan.

    Args:
        universe: 'sp500', 'sectors', 'all', or path to CSV
        filters_config: Filter configuration dict
        data_source: 'alpaca' or 'databento'
        top_n: Number of top results

    Returns:
        (gainers, losers) tuples
    """
    logger.info("=" * 80)
    logger.info(f"HOURLY MOMENTUM SCANNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Get symbols
    logger.info(f"Loading {universe} universe...")
    symbols = get_symbol_list(universe)
    logger.info(f"✓ Loaded {len(symbols)} symbols")

    # Initialize scanner
    scanner = MomentumScanner(data_source=data_source)

    # Run scan
    gainers, losers = scanner.run_full_scan(symbols, top_n=top_n)

    # Apply filters if provided
    if filters_config:
        logger.info("Applying filters...")
        filtered_results = MomentumFilters().apply_all_filters(
            scanner.results,
            filters_config
        )

        # Re-rank filtered results
        gainers = sorted(
            filtered_results.items(),
            key=lambda x: x[1]['intra_momentum'],
            reverse=True
        )[:top_n]

        losers = sorted(
            filtered_results.items(),
            key=lambda x: x[1]['intra_momentum']
        )[:top_n]

    return gainers, losers


def send_alerts(gainers: tuple, losers: tuple, alert_config: dict = None) -> None:
    """Send alerts via configured channels."""
    if not alert_config:
        alert_config = {}

    alert_manager = AlertManager(alert_config)
    alert_manager.send_all_alerts(gainers, losers)


def schedule_hourly_scan(universe: str = 'sp500',
                        filters_config: dict = None,
                        alert_config: dict = None) -> None:
    """
    Schedule scan to run every hour.

    Args:
        universe: Symbol universe
        filters_config: Filter configuration
        alert_config: Alert configuration
    """
    def job():
        try:
            logger.info("Running scheduled scan...")
            gainers, losers = run_scan(universe, filters_config)
            send_alerts(gainers, losers, alert_config)
        except Exception as e:
            logger.error(f"Scan failed: {e}", exc_info=True)

    # Schedule for every hour at :00
    schedule.every().hour.at(":00").do(job)

    logger.info("Scheduler started. Running scans at top of every hour.")
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Hourly Momentum Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan S&P 500 once
  python main.py --universe sp500 --action scan

  # Scan all stocks with filters
  python main.py --universe all --action scan --filters-config config/filters.json

  # Schedule hourly scans
  python main.py --universe sp500 --schedule-hourly

  # Scan sector leaders
  python main.py --universe sectors --action scan
        """
    )

    parser.add_argument(
        '--universe',
        choices=['sp500', 'sectors', 'all', 'nasdaq100'],
        default='sp500',
        help='Symbol universe to scan'
    )

    parser.add_argument(
        '--action',
        choices=['scan', 'show-symbols'],
        default='scan',
        help='Action to perform'
    )

    parser.add_argument(
        '--schedule-hourly',
        action='store_true',
        help='Schedule scan to run every hour'
    )

    parser.add_argument(
        '--data-source',
        choices=['alpaca', 'databento'],
        default='alpaca',
        help='Data source'
    )

    parser.add_argument(
        '--filters-config',
        default='config/filters.json',
        help='Path to filters config JSON'
    )

    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to main config JSON'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top results to show'
    )

    args = parser.parse_args()

    # Load main config
    main_config = load_config(args.config)

    # Load filters if provided
    filters_config = None
    if Path(args.filters_config).exists():
        with open(args.filters_config, 'r') as f:
            filters_config = json.load(f)
    elif args.filters_config != 'config/filters.json':
        logger.warning(f"Filters config not found: {args.filters_config}")

    # Actions
    if args.action == 'show-symbols':
        symbols = get_symbol_list(args.universe)
        print(f"\n{args.universe.upper()} Universe ({len(symbols)} symbols):")
        for i, symbol in enumerate(symbols, 1):
            print(f"  {i:3d}. {symbol}")
        print()

    elif args.action == 'scan':
        gainers, losers = run_scan(
            universe=args.universe,
            filters_config=filters_config,
            data_source=args.data_source,
            top_n=args.top_n
        )
        send_alerts(gainers, losers, main_config.get('alerts', {}))

    elif args.schedule_hourly:
        schedule_hourly_scan(
            universe=args.universe,
            filters_config=filters_config,
            alert_config=main_config.get('alerts', {})
        )


if __name__ == '__main__':
    main()
