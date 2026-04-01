"""Tests for analytics lake."""

from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa

from market_data.storage.analytics_lake import AnalyticsLake, OHLCV_SCHEMA
from market_data.storage.cloud_storage import LocalStorageBackend


class TestAnalyticsLake:
    def test_write_and_read(self, analytics_lake: AnalyticsLake) -> None:
        now = datetime.now(tz=timezone.utc)
        table = pa.Table.from_pylist(
            [
                {
                    "timestamp_utc": 1704067200000000000,
                    "symbol_id": 1,
                    "open": 150.0,
                    "high": 155.0,
                    "low": 149.0,
                    "close": 153.0,
                    "volume": 1000000,
                    "vwap": 152.0,
                    "trade_count": 5000,
                    "ingestion_time": now,
                }
            ],
            schema=OHLCV_SCHEMA,
        )

        path = analytics_lake.write_table(
            table=table,
            asset_class="equity",
            schema_name="ohlcv-1d",
            symbol_id=1,
            year=2024,
            month=1,
        )
        assert path.endswith(".parquet")

        result = analytics_lake.read_table(
            asset_class="equity",
            schema_name="ohlcv-1d",
            symbol_id=1,
            year=2024,
            month=1,
        )
        assert result is not None
        assert result.num_rows == 1
        assert result.column("close")[0].as_py() == 153.0

    def test_read_nonexistent(self, analytics_lake: AnalyticsLake) -> None:
        result = analytics_lake.read_table(
            asset_class="equity",
            schema_name="ohlcv-1d",
            symbol_id=999,
            year=2024,
            month=1,
        )
        assert result is None

    def test_read_date_range(self, analytics_lake: AnalyticsLake) -> None:
        now = datetime.now(tz=timezone.utc)
        for month in [1, 2, 3]:
            table = pa.Table.from_pylist(
                [
                    {
                        "timestamp_utc": 1704067200000000000 + month * 2_592_000_000_000_000,
                        "symbol_id": 1,
                        "open": 150.0 + month,
                        "high": 155.0,
                        "low": 149.0,
                        "close": 153.0 + month,
                        "volume": 1000000,
                        "vwap": 152.0,
                        "trade_count": 5000,
                        "ingestion_time": now,
                    }
                ],
                schema=OHLCV_SCHEMA,
            )
            analytics_lake.write_table(
                table=table,
                asset_class="equity",
                schema_name="ohlcv-1d",
                symbol_id=1,
                year=2024,
                month=month,
            )

        result = analytics_lake.read_date_range(
            asset_class="equity",
            schema_name="ohlcv-1d",
            symbol_id=1,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=3,
        )
        assert result is not None
        assert result.num_rows == 3

    def test_list_files(self, analytics_lake: AnalyticsLake) -> None:
        now = datetime.now(tz=timezone.utc)
        table = pa.Table.from_pylist(
            [
                {
                    "timestamp_utc": 1704067200000000000,
                    "symbol_id": 1,
                    "open": 150.0,
                    "high": 155.0,
                    "low": 149.0,
                    "close": 153.0,
                    "volume": 1000000,
                    "vwap": 152.0,
                    "trade_count": 5000,
                    "ingestion_time": now,
                }
            ],
            schema=OHLCV_SCHEMA,
        )
        analytics_lake.write_table(
            table=table,
            asset_class="equity",
            schema_name="ohlcv-1d",
            symbol_id=1,
            year=2024,
            month=1,
        )

        files = analytics_lake.list_files(asset_class="equity")
        assert len(files) == 1
        assert files[0].endswith(".parquet")
