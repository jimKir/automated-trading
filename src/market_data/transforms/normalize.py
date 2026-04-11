"""Data normalization: DBN to Parquet, timestamp standardization, symbol ID resolution."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from market_data.storage.analytics_lake import OHLCV_SCHEMA, TRADES_SCHEMA
from market_data.storage.symbol_master import SymbolMaster

logger = structlog.get_logger(__name__)


class DataNormalizer:
    """Normalize raw vendor data to unified Parquet format.

    Converts:
    - Databento DBN → Parquet with normalized schema
    - Alpaca JSON → Parquet with normalized schema
    - All timestamps to UTC nanoseconds
    - All symbols mapped to internal IDs via SymbolMaster

    Args:
        symbol_master: Symbol master for ID resolution.
    """

    def __init__(self, symbol_master: SymbolMaster) -> None:
        self.symbol_master = symbol_master

    def normalize_databento_trades(self, dbn_path: str | Path, symbol: str) -> pa.Table:
        """Convert Databento DBN trades to normalized Parquet format.

        Args:
            dbn_path: Path to DBN file.
            symbol: Symbol ticker for ID resolution.

        Returns:
            Normalized PyArrow table.
        """
        symbol_id = self._resolve_symbol_id(symbol)
        now = datetime.now(tz=UTC)

        try:
            import databento as db

            store = db.DBNStore.from_file(str(dbn_path))
            df = store.to_df()

            records = []
            for _, row in df.iterrows():
                ts_ns = int(row.get("ts_event", 0))
                records.append(
                    {
                        "timestamp_utc": ts_ns,
                        "symbol_id": symbol_id,
                        "price": float(row.get("price", 0)) / 1e9,  # DBN uses fixed-point
                        "size": int(row.get("size", 0)),
                        "exchange_id": int(row.get("publisher_id", 0)) % 128,
                        "conditions": str(row.get("conditions", "")),
                        "sequence_number": int(row.get("sequence", 0)),
                        "ingestion_time": now,
                    }
                )

            return pa.Table.from_pylist(records, schema=TRADES_SCHEMA)
        except ImportError:
            logger.warning("databento_sdk_not_available_for_normalization")
            return pa.table(
                {col: [] for col in TRADES_SCHEMA.names},
                schema=TRADES_SCHEMA,
            )

    def normalize_databento_ohlcv(self, dbn_path: str | Path, symbol: str) -> pa.Table:
        """Convert Databento DBN OHLCV bars to normalized Parquet format.

        Args:
            dbn_path: Path to DBN file.
            symbol: Symbol ticker.

        Returns:
            Normalized PyArrow table.
        """
        symbol_id = self._resolve_symbol_id(symbol)
        now = datetime.now(tz=UTC)

        try:
            import databento as db

            store = db.DBNStore.from_file(str(dbn_path))
            df = store.to_df()

            records = []
            for _, row in df.iterrows():
                records.append(
                    {
                        "timestamp_utc": int(row.get("ts_event", 0)),
                        "symbol_id": symbol_id,
                        "open": float(row.get("open", 0)) / 1e9,
                        "high": float(row.get("high", 0)) / 1e9,
                        "low": float(row.get("low", 0)) / 1e9,
                        "close": float(row.get("close", 0)) / 1e9,
                        "volume": int(row.get("volume", 0)),
                        "vwap": 0.0,
                        "trade_count": 0,
                        "ingestion_time": now,
                    }
                )

            return pa.Table.from_pylist(records, schema=OHLCV_SCHEMA)
        except ImportError:
            logger.warning("databento_sdk_not_available_for_normalization")
            return pa.table(
                {col: [] for col in OHLCV_SCHEMA.names},
                schema=OHLCV_SCHEMA,
            )

    def normalize_alpaca_bars(self, json_path: str | Path, symbol: str) -> pa.Table:
        """Convert Alpaca JSON bars to normalized Parquet format.

        Args:
            json_path: Path to JSON file with Alpaca bars.
            symbol: Symbol ticker.

        Returns:
            Normalized PyArrow table.
        """
        symbol_id = self._resolve_symbol_id(symbol)
        now = datetime.now(tz=UTC)

        with open(json_path) as f:
            bars = json.load(f)

        records = []
        for bar in bars:
            ts_str = bar.get("timestamp", bar.get("t", ""))
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            ts_ns = int(ts_dt.timestamp() * 1e9)

            records.append(
                {
                    "timestamp_utc": ts_ns,
                    "symbol_id": symbol_id,
                    "open": float(bar.get("open", bar.get("o", 0))),
                    "high": float(bar.get("high", bar.get("h", 0))),
                    "low": float(bar.get("low", bar.get("l", 0))),
                    "close": float(bar.get("close", bar.get("c", 0))),
                    "volume": int(bar.get("volume", bar.get("v", 0))),
                    "vwap": float(bar.get("vwap", bar.get("vw", 0)) or 0),
                    "trade_count": int(bar.get("trade_count", bar.get("n", 0)) or 0),
                    "ingestion_time": now,
                }
            )

        return pa.Table.from_pylist(records, schema=OHLCV_SCHEMA)

    def normalize_to_parquet(
        self,
        table: pa.Table,
        output_path: str | Path,
        compression: str = "snappy",
        row_group_size: int = 128 * 1024 * 1024,
    ) -> Path:
        """Write a normalized table to Parquet file.

        Args:
            table: Normalized PyArrow table.
            output_path: Output file path.
            compression: Compression codec.
            row_group_size: Row group size in bytes.

        Returns:
            Path to written Parquet file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(
            table,
            str(output_path),
            compression=compression,
            row_group_size=row_group_size,
        )
        logger.info(
            "parquet_written",
            path=str(output_path),
            rows=table.num_rows,
        )
        return output_path

    def _resolve_symbol_id(self, symbol: str) -> int:
        """Resolve a ticker to internal symbol_id.

        Auto-creates if not found.

        Args:
            symbol: Ticker symbol.

        Returns:
            Internal symbol_id.
        """
        from market_data.storage.symbol_master import SymbolRecord

        existing = self.symbol_master.get_by_ticker(symbol)
        if existing:
            return existing.symbol_id

        # Auto-create with minimal info
        record = SymbolRecord(symbol_id=0, ticker=symbol, asset_class="equity")
        return self.symbol_master.upsert_symbol(record)
