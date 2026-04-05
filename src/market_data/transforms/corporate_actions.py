"""Corporate actions: split and dividend adjustments applied on read."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SplitRecord:
    """Stock split record."""

    symbol_id: int
    ex_date: str
    split_from: float  # e.g., 1 (old shares)
    split_to: float    # e.g., 4 (new shares)


@dataclass
class DividendRecord:
    """Dividend record."""

    symbol_id: int
    ex_date: str
    amount: float
    dividend_type: str = "cash"  # cash, stock, special


class CorporateActionsManager:
    """Manage corporate action adjustments (splits and dividends).

    Adjustments are applied on read via view layer, not mutating stored data.
    Maintains a table of historical splits and dividends per symbol.

    Args:
        db_path: Path to SQLite database.
    """

    def __init__(self, db_path: str | Path = "corporate_actions.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the corporate actions database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS splits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER NOT NULL,
                    ex_date TEXT NOT NULL,
                    split_from REAL NOT NULL,
                    split_to REAL NOT NULL,
                    UNIQUE(symbol_id, ex_date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dividends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER NOT NULL,
                    ex_date TEXT NOT NULL,
                    amount REAL NOT NULL,
                    dividend_type TEXT DEFAULT 'cash',
                    UNIQUE(symbol_id, ex_date, dividend_type)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_splits_symbol
                ON splits(symbol_id, ex_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dividends_symbol
                ON dividends(symbol_id, ex_date)
            """)
            conn.commit()

    def add_split(self, split: SplitRecord) -> None:
        """Record a stock split.

        Args:
            split: Split record to add.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO splits (symbol_id, ex_date, split_from, split_to)
                VALUES (?, ?, ?, ?)
                """,
                (split.symbol_id, split.ex_date, split.split_from, split.split_to),
            )
            conn.commit()
        logger.info(
            "split_recorded",
            symbol_id=split.symbol_id,
            ex_date=split.ex_date,
            ratio=f"{split.split_from}:{split.split_to}",
        )

    def add_dividend(self, dividend: DividendRecord) -> None:
        """Record a dividend.

        Args:
            dividend: Dividend record to add.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO dividends
                (symbol_id, ex_date, amount, dividend_type)
                VALUES (?, ?, ?, ?)
                """,
                (
                    dividend.symbol_id,
                    dividend.ex_date,
                    dividend.amount,
                    dividend.dividend_type,
                ),
            )
            conn.commit()

    def get_split_adjustment_factor(
        self,
        symbol_id: int,
        as_of_date: str,
        target_date: str | None = None,
    ) -> float:
        """Calculate cumulative split adjustment factor.

        Returns a multiplier to apply to historical prices to make them
        comparable to prices on the as_of_date.

        Args:
            symbol_id: Internal symbol ID.
            as_of_date: Date to adjust prices relative to.
            target_date: Start date for adjustment window. If None, all splits used.

        Returns:
            Cumulative adjustment factor (multiply prices by this).
        """
        query = """
            SELECT split_from, split_to FROM splits
            WHERE symbol_id = ? AND ex_date <= ?
        """
        params: list[Any] = [symbol_id, as_of_date]
        if target_date:
            query += " AND ex_date > ?"
            params.append(target_date)

        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(query, params).fetchall()

        factor = 1.0
        for split_from, split_to in rows:
            factor *= split_to / split_from
        return factor

    def get_dividend_adjustment(
        self,
        symbol_id: int,
        start_date: str,
        end_date: str,
    ) -> float:
        """Calculate cumulative dividend adjustment.

        Args:
            symbol_id: Internal symbol ID.
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            Total dividend amount in the range.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(amount), 0) FROM dividends
                WHERE symbol_id = ? AND ex_date BETWEEN ? AND ?
                """,
                (symbol_id, start_date, end_date),
            ).fetchone()
        return row[0] if row else 0.0

    def adjust_prices(
        self,
        prices: np.ndarray,
        dates: list[str],
        symbol_id: int,
        adjustment_mode: str = "split",
        as_of_date: str | None = None,
    ) -> np.ndarray:
        """Apply corporate action adjustments to price array.

        Args:
            prices: Array of prices to adjust.
            dates: Corresponding dates for each price.
            symbol_id: Internal symbol ID.
            adjustment_mode: One of 'raw', 'split', 'dividend', 'all'.
            as_of_date: Reference date for adjustment. Defaults to today.

        Returns:
            Adjusted price array.
        """
        if adjustment_mode == "raw":
            return prices.copy()

        adjusted = prices.copy().astype(np.float64)
        ref_date = as_of_date or date.today().isoformat()

        if adjustment_mode in ("split", "all"):
            for i, d in enumerate(dates):
                factor = self.get_split_adjustment_factor(
                    symbol_id=symbol_id,
                    as_of_date=ref_date,
                    target_date=d,
                )
                adjusted[i] *= factor

        if adjustment_mode in ("dividend", "all"):
            for i, d in enumerate(dates):
                div_adj = self.get_dividend_adjustment(
                    symbol_id=symbol_id,
                    start_date=d,
                    end_date=ref_date,
                )
                adjusted[i] -= div_adj

        return adjusted

    def get_splits(self, symbol_id: int) -> list[SplitRecord]:
        """Get all splits for a symbol.

        Args:
            symbol_id: Internal symbol ID.

        Returns:
            List of SplitRecord objects.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM splits WHERE symbol_id = ? ORDER BY ex_date",
                (symbol_id,),
            ).fetchall()

        return [
            SplitRecord(
                symbol_id=row["symbol_id"],
                ex_date=row["ex_date"],
                split_from=row["split_from"],
                split_to=row["split_to"],
            )
            for row in rows
        ]

    def get_dividends(self, symbol_id: int) -> list[DividendRecord]:
        """Get all dividends for a symbol.

        Args:
            symbol_id: Internal symbol ID.

        Returns:
            List of DividendRecord objects.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM dividends WHERE symbol_id = ? ORDER BY ex_date",
                (symbol_id,),
            ).fetchall()

        return [
            DividendRecord(
                symbol_id=row["symbol_id"],
                ex_date=row["ex_date"],
                amount=row["amount"],
                dividend_type=row["dividend_type"],
            )
            for row in rows
        ]
