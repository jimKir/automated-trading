"""Raw data lake manager for immutable, append-only raw data storage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from market_data.storage.cloud_storage import StorageBackend

logger = structlog.get_logger(__name__)


class RawDataLake:
    """Raw data lake manager.

    Implements immutable, append-only writes with metadata sidecar JSON
    containing SHA-256 checksums.

    Path format: /raw/<vendor>/<asset_class>/<schema>/<year>/<month>/<day>/<symbol>.<ext>

    Args:
        storage: Storage backend to use.
        base_path: Base path prefix for raw data.
    """

    def __init__(
        self,
        storage: StorageBackend,
        base_path: str = "raw",
    ) -> None:
        self.storage = storage
        self.base_path = base_path
        self.logger = logger

    def write(
        self,
        data: bytes,
        vendor: str,
        asset_class: str,
        schema: str,
        symbol: str,
        date: datetime,
        extension: str = "dbn",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write raw data to the lake with metadata sidecar.

        Immutable append-only: raises error if file already exists.

        Args:
            data: Raw data bytes to write.
            vendor: Data vendor name (databento/alpaca).
            asset_class: Asset class (equity/option/future/crypto).
            schema: Data schema (trades/quotes/ohlcv-1d/etc).
            symbol: Symbol ticker.
            date: Date of the data.
            extension: File extension.
            metadata: Additional metadata to store in sidecar.

        Returns:
            Storage path where data was written.

        Raises:
            FileExistsError: If data file already exists (immutable).
        """
        data_path = self._build_path(
            vendor=vendor,
            asset_class=asset_class,
            schema=schema,
            symbol=symbol,
            date=date,
            extension=extension,
        )

        if self.storage.exists(data_path):
            raise FileExistsError(
                f"Raw data already exists at {data_path}. Raw lake is immutable and append-only."
            )

        # Compute checksum
        checksum = hashlib.sha256(data).hexdigest()

        # Write data file
        self.storage.write(data_path, data)

        # Write metadata sidecar
        sidecar = {
            "vendor": vendor,
            "asset_class": asset_class,
            "schema": schema,
            "symbol": symbol,
            "date": date.isoformat(),
            "extension": extension,
            "checksum_sha256": checksum,
            "size_bytes": len(data),
            "ingestion_time": datetime.now(tz=UTC).isoformat(),
            "data_path": data_path,
            **(metadata or {}),
        }
        sidecar_path = data_path + ".meta.json"
        self.storage.write(sidecar_path, json.dumps(sidecar, indent=2).encode())

        self.logger.info(
            "raw_data_written",
            path=data_path,
            size_bytes=len(data),
            checksum=checksum[:16],
        )
        return data_path

    def write_file(
        self,
        local_path: str | Path,
        vendor: str,
        asset_class: str,
        schema: str,
        symbol: str,
        date: datetime,
        extension: str = "dbn",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write a local file to the raw data lake.

        Args:
            local_path: Path to local file.
            vendor: Data vendor name.
            asset_class: Asset class.
            schema: Data schema.
            symbol: Symbol ticker.
            date: Date of the data.
            extension: File extension.
            metadata: Additional metadata.

        Returns:
            Storage path where data was written.
        """
        data = Path(local_path).read_bytes()
        return self.write(
            data=data,
            vendor=vendor,
            asset_class=asset_class,
            schema=schema,
            symbol=symbol,
            date=date,
            extension=extension,
            metadata=metadata,
        )

    def read(
        self,
        vendor: str,
        asset_class: str,
        schema: str,
        symbol: str,
        date: datetime,
        extension: str = "dbn",
    ) -> bytes:
        """Read raw data from the lake.

        Args:
            vendor: Data vendor name.
            asset_class: Asset class.
            schema: Data schema.
            symbol: Symbol ticker.
            date: Date of the data.
            extension: File extension.

        Returns:
            Raw data bytes.
        """
        data_path = self._build_path(
            vendor=vendor,
            asset_class=asset_class,
            schema=schema,
            symbol=symbol,
            date=date,
            extension=extension,
        )
        return self.storage.read(data_path)

    def read_metadata(
        self,
        vendor: str,
        asset_class: str,
        schema: str,
        symbol: str,
        date: datetime,
        extension: str = "dbn",
    ) -> dict[str, Any]:
        """Read metadata sidecar for a raw data file.

        Args:
            vendor: Data vendor name.
            asset_class: Asset class.
            schema: Data schema.
            symbol: Symbol ticker.
            date: Date of the data.
            extension: File extension.

        Returns:
            Metadata dictionary.
        """
        data_path = self._build_path(
            vendor=vendor,
            asset_class=asset_class,
            schema=schema,
            symbol=symbol,
            date=date,
            extension=extension,
        )
        sidecar_path = data_path + ".meta.json"
        data = self.storage.read(sidecar_path)
        return json.loads(data)

    def verify_checksum(
        self,
        vendor: str,
        asset_class: str,
        schema: str,
        symbol: str,
        date: datetime,
        extension: str = "dbn",
    ) -> bool:
        """Verify data integrity using stored checksum.

        Args:
            vendor: Data vendor name.
            asset_class: Asset class.
            schema: Data schema.
            symbol: Symbol ticker.
            date: Date of the data.
            extension: File extension.

        Returns:
            True if checksum matches, False otherwise.
        """
        data = self.read(
            vendor=vendor,
            asset_class=asset_class,
            schema=schema,
            symbol=symbol,
            date=date,
            extension=extension,
        )
        metadata = self.read_metadata(
            vendor=vendor,
            asset_class=asset_class,
            schema=schema,
            symbol=symbol,
            date=date,
            extension=extension,
        )
        computed = hashlib.sha256(data).hexdigest()
        return computed == metadata.get("checksum_sha256", "")

    def list_files(
        self,
        vendor: str | None = None,
        asset_class: str | None = None,
        schema: str | None = None,
        year: int | None = None,
    ) -> list[str]:
        """List raw data files with optional filters.

        Args:
            vendor: Filter by vendor.
            asset_class: Filter by asset class.
            schema: Filter by schema.
            year: Filter by year.

        Returns:
            List of matching file paths.
        """
        prefix = self.base_path
        if vendor:
            prefix = f"{prefix}/{vendor}"
        if asset_class:
            prefix = f"{prefix}/{asset_class}"
        if schema:
            prefix = f"{prefix}/{schema}"
        if year:
            prefix = f"{prefix}/{year}"

        all_files = self.storage.list_files(prefix)
        return [f for f in all_files if not f.endswith(".meta.json")]

    def _build_path(
        self,
        vendor: str,
        asset_class: str,
        schema: str,
        symbol: str,
        date: datetime,
        extension: str,
    ) -> str:
        """Build raw data lake path.

        Format: <base_path>/<vendor>/<asset_class>/<schema>/<year>/<month>/<day>/<symbol>.<ext>
        """
        return (
            f"{self.base_path}/{vendor}/{asset_class}/{schema}/"
            f"{date.year}/{date.month:02d}/{date.day:02d}/"
            f"{symbol}.{extension}"
        )
