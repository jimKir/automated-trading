"""Symbol master table for mapping vendor tickers to internal IDs."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SymbolRecord:
    """A record in the symbol master table."""

    symbol_id: int
    ticker: str
    figi: str | None = None
    isin: str | None = None
    cusip: str | None = None
    asset_class: str = "equity"
    primary_exchange: str | None = None
    listing_date: str | None = None
    delisting_date: str | None = None
    name: str | None = None


@dataclass
class OptionsContract:
    """An options contract record."""

    contract_id: int
    underlying_symbol_id: int
    strike: float
    expiration: str
    option_type: str  # call/put
    contract_size: int = 100
    multiplier: float = 1.0


class SymbolMaster:
    """Symbol master table backed by SQLite.

    Maps vendor tickers to internal symbol_id. Supports daily updates
    from Databento symbology and Alpaca assets. Includes options contract master.

    Args:
        db_path: Path to SQLite database file.
    """

    def __init__(self, db_path: str | Path = "symbol_master.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the symbol master database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    figi TEXT,
                    isin TEXT,
                    cusip TEXT,
                    asset_class TEXT NOT NULL DEFAULT 'equity',
                    primary_exchange TEXT,
                    listing_date TEXT,
                    delisting_date TEXT,
                    name TEXT,
                    UNIQUE(ticker, asset_class)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_contracts (
                    contract_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    underlying_symbol_id INTEGER NOT NULL,
                    strike REAL NOT NULL,
                    expiration TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    contract_size INTEGER DEFAULT 100,
                    multiplier REAL DEFAULT 1.0,
                    FOREIGN KEY (underlying_symbol_id) REFERENCES symbols(symbol_id),
                    UNIQUE(underlying_symbol_id, strike, expiration, option_type)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbols_ticker
                ON symbols(ticker)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbols_figi
                ON symbols(figi)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_options_underlying
                ON options_contracts(underlying_symbol_id)
            """)
            conn.commit()

    def upsert_symbol(self, record: SymbolRecord) -> int:
        """Insert or update a symbol record.

        Args:
            record: Symbol record to upsert.

        Returns:
            The symbol_id of the inserted/updated record.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO symbols (ticker, figi, isin, cusip, asset_class,
                                     primary_exchange, listing_date, delisting_date, name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, asset_class)
                DO UPDATE SET
                    figi = excluded.figi,
                    isin = excluded.isin,
                    cusip = excluded.cusip,
                    primary_exchange = excluded.primary_exchange,
                    listing_date = excluded.listing_date,
                    delisting_date = excluded.delisting_date,
                    name = excluded.name
                """,
                (
                    record.ticker,
                    record.figi,
                    record.isin,
                    record.cusip,
                    record.asset_class,
                    record.primary_exchange,
                    record.listing_date,
                    record.delisting_date,
                    record.name,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT symbol_id FROM symbols WHERE ticker = ? AND asset_class = ?",
                (record.ticker, record.asset_class),
            ).fetchone()
            return row[0] if row else 0

    def get_by_ticker(self, ticker: str, asset_class: str = "equity") -> SymbolRecord | None:
        """Look up a symbol by ticker.

        Args:
            ticker: Symbol ticker.
            asset_class: Asset class.

        Returns:
            SymbolRecord or None if not found.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM symbols WHERE ticker = ? AND asset_class = ?",
                (ticker, asset_class),
            ).fetchone()

        if row is None:
            return None
        return SymbolRecord(
            symbol_id=row["symbol_id"],
            ticker=row["ticker"],
            figi=row["figi"],
            isin=row["isin"],
            cusip=row["cusip"],
            asset_class=row["asset_class"],
            primary_exchange=row["primary_exchange"],
            listing_date=row["listing_date"],
            delisting_date=row["delisting_date"],
            name=row["name"],
        )

    def get_by_id(self, symbol_id: int) -> SymbolRecord | None:
        """Look up a symbol by internal ID.

        Args:
            symbol_id: Internal symbol ID.

        Returns:
            SymbolRecord or None if not found.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM symbols WHERE symbol_id = ?", (symbol_id,)
            ).fetchone()

        if row is None:
            return None
        return SymbolRecord(
            symbol_id=row["symbol_id"],
            ticker=row["ticker"],
            figi=row["figi"],
            isin=row["isin"],
            cusip=row["cusip"],
            asset_class=row["asset_class"],
            primary_exchange=row["primary_exchange"],
            listing_date=row["listing_date"],
            delisting_date=row["delisting_date"],
            name=row["name"],
        )

    def get_symbol_id(self, ticker: str, asset_class: str = "equity") -> int | None:
        """Get the internal symbol_id for a ticker.

        Args:
            ticker: Symbol ticker.
            asset_class: Asset class.

        Returns:
            symbol_id or None if not found.
        """
        record = self.get_by_ticker(ticker, asset_class)
        return record.symbol_id if record else None

    def list_symbols(
        self,
        asset_class: str | None = None,
        active_only: bool = False,
    ) -> list[SymbolRecord]:
        """List all symbols with optional filters.

        Args:
            asset_class: Filter by asset class.
            active_only: Only return symbols without a delisting date.

        Returns:
            List of SymbolRecord objects.
        """
        query = "SELECT * FROM symbols WHERE 1=1"
        params: list[Any] = []
        if asset_class:
            query += " AND asset_class = ?"
            params.append(asset_class)
        if active_only:
            query += " AND delisting_date IS NULL"

        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            SymbolRecord(
                symbol_id=row["symbol_id"],
                ticker=row["ticker"],
                figi=row["figi"],
                isin=row["isin"],
                cusip=row["cusip"],
                asset_class=row["asset_class"],
                primary_exchange=row["primary_exchange"],
                listing_date=row["listing_date"],
                delisting_date=row["delisting_date"],
                name=row["name"],
            )
            for row in rows
        ]

    def upsert_options_contract(self, contract: OptionsContract) -> int:
        """Insert or update an options contract.

        Args:
            contract: Options contract record.

        Returns:
            The contract_id.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO options_contracts
                (underlying_symbol_id, strike, expiration, option_type, contract_size, multiplier)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(underlying_symbol_id, strike, expiration, option_type)
                DO UPDATE SET
                    contract_size = excluded.contract_size,
                    multiplier = excluded.multiplier
                """,
                (
                    contract.underlying_symbol_id,
                    contract.strike,
                    contract.expiration,
                    contract.option_type,
                    contract.contract_size,
                    contract.multiplier,
                ),
            )
            conn.commit()
            row = conn.execute(
                """
                SELECT contract_id FROM options_contracts
                WHERE underlying_symbol_id = ? AND strike = ? AND expiration = ? AND option_type = ?
                """,
                (
                    contract.underlying_symbol_id,
                    contract.strike,
                    contract.expiration,
                    contract.option_type,
                ),
            ).fetchone()
            return row[0] if row else 0

    def get_options_contracts(
        self, underlying_symbol_id: int
    ) -> list[OptionsContract]:
        """Get all options contracts for an underlying symbol.

        Args:
            underlying_symbol_id: Symbol ID of the underlying.

        Returns:
            List of OptionsContract records.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM options_contracts WHERE underlying_symbol_id = ?",
                (underlying_symbol_id,),
            ).fetchall()

        return [
            OptionsContract(
                contract_id=row["contract_id"],
                underlying_symbol_id=row["underlying_symbol_id"],
                strike=row["strike"],
                expiration=row["expiration"],
                option_type=row["option_type"],
                contract_size=row["contract_size"],
                multiplier=row["multiplier"],
            )
            for row in rows
        ]

    def bulk_upsert(self, records: list[SymbolRecord]) -> int:
        """Bulk insert/update symbols.

        Args:
            records: List of symbol records to upsert.

        Returns:
            Number of records processed.
        """
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            for record in records:
                conn.execute(
                    """
                    INSERT INTO symbols (ticker, figi, isin, cusip, asset_class,
                                         primary_exchange, listing_date, delisting_date, name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, asset_class)
                    DO UPDATE SET
                        figi = excluded.figi,
                        isin = excluded.isin,
                        cusip = excluded.cusip,
                        primary_exchange = excluded.primary_exchange,
                        listing_date = excluded.listing_date,
                        delisting_date = excluded.delisting_date,
                        name = excluded.name
                    """,
                    (
                        record.ticker,
                        record.figi,
                        record.isin,
                        record.cusip,
                        record.asset_class,
                        record.primary_exchange,
                        record.listing_date,
                        record.delisting_date,
                        record.name,
                    ),
                )
            conn.commit()

        logger.info("bulk_upsert_complete", count=len(records))
        return len(records)

    @property
    def symbol_count(self) -> int:
        """Total number of symbols in the master."""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
            return row[0] if row else 0
