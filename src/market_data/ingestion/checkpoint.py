"""Checkpoint manager for resumable downloads."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CheckpointState:
    """State of a checkpoint for a download operation."""

    vendor: str
    symbol: str
    schema: str
    start_date: str
    end_date: str
    last_processed_date: str | None = None
    records_processed: int = 0
    bytes_processed: int = 0
    status: str = "in_progress"  # in_progress, completed, failed
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_key(self) -> str:
        """Unique key for this checkpoint."""
        return f"{self.vendor}:{self.symbol}:{self.schema}:{self.start_date}:{self.end_date}"


class CheckpointManager:
    """Manage checkpoints for resumable data downloads.

    Uses SQLite for persistence. Tracks progress per symbol/schema/date range.
    Supports checkpointing every N records or N megabytes.

    Args:
        db_path: Path to SQLite checkpoint database.
        checkpoint_every_records: Checkpoint after this many records.
        checkpoint_every_mb: Checkpoint after this many megabytes.
    """

    def __init__(
        self,
        db_path: str | Path = "checkpoints.db",
        checkpoint_every_records: int = 1000,
        checkpoint_every_mb: int = 100,
    ) -> None:
        self.db_path = Path(db_path)
        self.checkpoint_every_records = checkpoint_every_records
        self.checkpoint_every_mb = checkpoint_every_mb
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the checkpoint database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_key TEXT PRIMARY KEY,
                    vendor TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    schema TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    last_processed_date TEXT,
                    records_processed INTEGER DEFAULT 0,
                    bytes_processed INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'in_progress',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.commit()

    def save(self, state: CheckpointState) -> None:
        """Save or update a checkpoint.

        Args:
            state: Checkpoint state to persist.
        """
        state.updated_at = datetime.utcnow().isoformat()
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (checkpoint_key, vendor, symbol, schema, start_date, end_date,
                 last_processed_date, records_processed, bytes_processed, status,
                 created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.checkpoint_key,
                    state.vendor,
                    state.symbol,
                    state.schema,
                    state.start_date,
                    state.end_date,
                    state.last_processed_date,
                    state.records_processed,
                    state.bytes_processed,
                    state.status,
                    state.created_at,
                    state.updated_at,
                    json.dumps(state.metadata),
                ),
            )
            conn.commit()
        logger.debug(
            "checkpoint_saved",
            key=state.checkpoint_key,
            records=state.records_processed,
        )

    def load(
        self, vendor: str, symbol: str, schema: str, start_date: str, end_date: str
    ) -> CheckpointState | None:
        """Load a checkpoint if it exists.

        Args:
            vendor: Vendor name.
            symbol: Symbol ticker.
            schema: Data schema.
            start_date: Start date string.
            end_date: End date string.

        Returns:
            CheckpointState if found, None otherwise.
        """
        key = f"{vendor}:{symbol}:{schema}:{start_date}:{end_date}"
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_key = ?", (key,)
            ).fetchone()

        if row is None:
            return None

        return CheckpointState(
            vendor=row["vendor"],
            symbol=row["symbol"],
            schema=row["schema"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            last_processed_date=row["last_processed_date"],
            records_processed=row["records_processed"],
            bytes_processed=row["bytes_processed"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )

    def should_checkpoint(self, records: int, bytes_size: int) -> bool:
        """Check if a checkpoint should be taken based on records or bytes.

        Args:
            records: Records processed since last checkpoint.
            bytes_size: Bytes processed since last checkpoint.

        Returns:
            True if checkpoint should be taken.
        """
        if records >= self.checkpoint_every_records:
            return True
        return bytes_size >= self.checkpoint_every_mb * 1024 * 1024

    def mark_completed(self, state: CheckpointState) -> None:
        """Mark a checkpoint as completed.

        Args:
            state: Checkpoint state to mark as completed.
        """
        state.status = "completed"
        self.save(state)

    def mark_failed(self, state: CheckpointState) -> None:
        """Mark a checkpoint as failed.

        Args:
            state: Checkpoint state to mark as failed.
        """
        state.status = "failed"
        self.save(state)

    def get_incomplete(self, vendor: str | None = None) -> list[CheckpointState]:
        """Get all incomplete checkpoints.

        Args:
            vendor: Optional vendor filter.

        Returns:
            List of incomplete checkpoint states.
        """
        query = "SELECT * FROM checkpoints WHERE status = 'in_progress'"
        params: list[str] = []
        if vendor:
            query += " AND vendor = ?"
            params.append(vendor)

        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            CheckpointState(
                vendor=row["vendor"],
                symbol=row["symbol"],
                schema=row["schema"],
                start_date=row["start_date"],
                end_date=row["end_date"],
                last_processed_date=row["last_processed_date"],
                records_processed=row["records_processed"],
                bytes_processed=row["bytes_processed"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def clear_completed(self) -> int:
        """Remove all completed checkpoints.

        Returns:
            Number of checkpoints removed.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM checkpoints WHERE status = 'completed'")
            conn.commit()
            return cursor.rowcount
