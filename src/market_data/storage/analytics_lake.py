"""Normalized Parquet analytics lake for vendor-agnostic data storage."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

if TYPE_CHECKING:
    from market_data.storage.cloud_storage import StorageBackend

logger = structlog.get_logger(__name__)

# Normalized equity trades schema
TRADES_SCHEMA = pa.schema(
    [
        pa.field("timestamp_utc", pa.int64()),  # UTC nanoseconds
        pa.field("symbol_id", pa.int32()),  # Internal symbol ID (FK)
        pa.field("price", pa.float64()),  # Trade price
        pa.field("size", pa.int32()),  # Trade size (shares)
        pa.field("exchange_id", pa.int8()),  # Exchange code
        pa.field("conditions", pa.string()),  # Trade conditions
        pa.field("sequence_number", pa.int64()),  # Exchange sequence number
        pa.field("ingestion_time", pa.timestamp("ns")),  # When record was ingested
    ]
)

# Normalized OHLCV bars schema
OHLCV_SCHEMA = pa.schema(
    [
        pa.field("timestamp_utc", pa.int64()),
        pa.field("symbol_id", pa.int32()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("vwap", pa.float64()),
        pa.field("trade_count", pa.int32()),
        pa.field("ingestion_time", pa.timestamp("ns")),
    ]
)

# Normalized quotes schema
QUOTES_SCHEMA = pa.schema(
    [
        pa.field("timestamp_utc", pa.int64()),
        pa.field("symbol_id", pa.int32()),
        pa.field("bid_price", pa.float64()),
        pa.field("bid_size", pa.int32()),
        pa.field("ask_price", pa.float64()),
        pa.field("ask_size", pa.int32()),
        pa.field("exchange_id", pa.int8()),
        pa.field("ingestion_time", pa.timestamp("ns")),
    ]
)

SCHEMA_MAP: dict[str, pa.Schema] = {
    "trades": TRADES_SCHEMA,
    "ohlcv-1m": OHLCV_SCHEMA,
    "ohlcv-1h": OHLCV_SCHEMA,
    "ohlcv-1d": OHLCV_SCHEMA,
    "quotes": QUOTES_SCHEMA,
    "tbbo": QUOTES_SCHEMA,
}


class AnalyticsLake:
    """Normalized Parquet analytics lake.

    Converts all timestamps to UTC nanoseconds, maps symbols to internal IDs,
    and writes with Snappy compression and 128MB row groups.

    Path format: /analytics/<asset_class>/<schema>/<year>/<month>/<symbol_id>.parquet

    Args:
        storage: Storage backend to use.
        base_path: Base path prefix for analytics data.
        compression: Parquet compression codec.
        row_group_size_mb: Target row group size in MB.
    """

    def __init__(
        self,
        storage: StorageBackend,
        base_path: str = "analytics",
        compression: str = "snappy",
        row_group_size_mb: int = 128,
    ) -> None:
        self.storage = storage
        self.base_path = base_path
        self.compression = compression
        self.row_group_size = row_group_size_mb * 1024 * 1024  # Convert to bytes

    def write_table(
        self,
        table: pa.Table,
        asset_class: str,
        schema_name: str,
        symbol_id: int,
        year: int,
        month: int,
    ) -> str:
        """Write a PyArrow table to the analytics lake as Parquet.

        Args:
            table: PyArrow table with normalized data.
            asset_class: Asset class (equity/option/future/crypto).
            schema_name: Data schema name (trades/ohlcv-1d/etc).
            symbol_id: Internal symbol identifier.
            year: Data year.
            month: Data month.

        Returns:
            Storage path where data was written.
        """
        path = self._build_path(
            asset_class=asset_class,
            schema_name=schema_name,
            symbol_id=symbol_id,
            year=year,
            month=month,
        )

        buf = io.BytesIO()
        pq.write_table(
            table,
            buf,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )
        self.storage.write(path, buf.getvalue())

        logger.info(
            "analytics_data_written",
            path=path,
            rows=table.num_rows,
            size_bytes=buf.tell(),
        )
        return path

    def append_records(
        self,
        records: list[dict[str, Any]],
        asset_class: str,
        schema_name: str,
        symbol_id: int,
        year: int,
        month: int,
    ) -> str:
        """Append records to an existing or new Parquet file.

        Reads existing data if present, appends new records, rewrites.

        Args:
            records: List of record dictionaries.
            asset_class: Asset class.
            schema_name: Data schema.
            symbol_id: Internal symbol ID.
            year: Data year.
            month: Data month.

        Returns:
            Storage path where data was written.
        """
        arrow_schema = SCHEMA_MAP.get(schema_name)
        if arrow_schema is None:
            raise ValueError(f"Unknown schema: {schema_name}")

        path = self._build_path(
            asset_class=asset_class,
            schema_name=schema_name,
            symbol_id=symbol_id,
            year=year,
            month=month,
        )

        # Read existing data if present
        existing_table = None
        if self.storage.exists(path):
            existing_data = self.storage.read(path)
            existing_table = pq.read_table(io.BytesIO(existing_data))

        # Create new table from records
        new_table = pa.Table.from_pylist(records, schema=arrow_schema)

        # Combine
        if existing_table is not None:
            combined = pa.concat_tables([existing_table, new_table])
        else:
            combined = new_table

        return self.write_table(
            table=combined,
            asset_class=asset_class,
            schema_name=schema_name,
            symbol_id=symbol_id,
            year=year,
            month=month,
        )

    def read_table(
        self,
        asset_class: str,
        schema_name: str,
        symbol_id: int,
        year: int,
        month: int,
        columns: list[str] | None = None,
    ) -> pa.Table | None:
        """Read a Parquet table from the analytics lake.

        Args:
            asset_class: Asset class.
            schema_name: Data schema.
            symbol_id: Internal symbol ID.
            year: Data year.
            month: Data month.
            columns: Optional column filter.

        Returns:
            PyArrow table or None if not found.
        """
        path = self._build_path(
            asset_class=asset_class,
            schema_name=schema_name,
            symbol_id=symbol_id,
            year=year,
            month=month,
        )

        if not self.storage.exists(path):
            return None

        data = self.storage.read(path)
        return pq.read_table(io.BytesIO(data), columns=columns)

    def read_date_range(
        self,
        asset_class: str,
        schema_name: str,
        symbol_id: int,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        columns: list[str] | None = None,
    ) -> pa.Table | None:
        """Read data across multiple months.

        Args:
            asset_class: Asset class.
            schema_name: Data schema.
            symbol_id: Internal symbol ID.
            start_year: Start year.
            start_month: Start month.
            end_year: End year.
            end_month: End month.
            columns: Optional column filter.

        Returns:
            Combined PyArrow table or None if no data found.
        """
        tables: list[pa.Table] = []
        year, month = start_year, start_month

        while (year, month) <= (end_year, end_month):
            table = self.read_table(
                asset_class=asset_class,
                schema_name=schema_name,
                symbol_id=symbol_id,
                year=year,
                month=month,
                columns=columns,
            )
            if table is not None:
                tables.append(table)

            month += 1
            if month > 12:
                month = 1
                year += 1

        if not tables:
            return None
        return pa.concat_tables(tables)

    def list_files(
        self,
        asset_class: str | None = None,
        schema_name: str | None = None,
    ) -> list[str]:
        """List analytics Parquet files with optional filters.

        Args:
            asset_class: Filter by asset class.
            schema_name: Filter by schema.

        Returns:
            List of matching file paths.
        """
        prefix = self.base_path
        if asset_class:
            prefix = f"{prefix}/{asset_class}"
        if schema_name:
            prefix = f"{prefix}/{schema_name}"

        return [f for f in self.storage.list_files(prefix) if f.endswith(".parquet")]

    def _build_path(
        self,
        asset_class: str,
        schema_name: str,
        symbol_id: int,
        year: int,
        month: int,
    ) -> str:
        """Build analytics lake path.

        Format: <base_path>/<asset_class>/<schema>/<year>/<month>/<symbol_id>.parquet
        """
        return (
            f"{self.base_path}/{asset_class}/{schema_name}/{year}/{month:02d}/{symbol_id}.parquet"
        )
