#!/usr/bin/env python3
"""Script to run historical data backfill."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run market data backfill")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        help="Symbols to backfill",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(date.today() - timedelta(days=730)).isoformat(),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--vendors",
        nargs="+",
        default=["databento", "alpaca"],
        help="Vendors to use",
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=["ohlcv-1d"],
        help="Schemas to fetch",
    )
    args = parser.parse_args()

    print("Starting backfill:")
    print(f"  Symbols: {args.symbols}")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Vendors: {args.vendors}")
    print(f"  Schemas: {args.schemas}")
    print()

    try:
        from flows.backfill_flow import backfill_flow

        results = backfill_flow(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            vendors=args.vendors,
            schemas=args.schemas,
        )

        for vendor, summary in results.items():
            print(
                f"  {vendor}: {summary.completed_chunks}/{summary.total_chunks} chunks "
                f"({summary.success_rate:.0%} success)"
            )
    except Exception as exc:
        print(f"Backfill failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
