"""Tests for symbol master."""

from __future__ import annotations

from pathlib import Path

from market_data.storage.symbol_master import (
    OptionsContract,
    SymbolMaster,
    SymbolRecord,
)


class TestSymbolMaster:
    def test_upsert_and_lookup(self, tmp_dir: Path) -> None:
        sm = SymbolMaster(db_path=tmp_dir / "sm.db")
        record = SymbolRecord(symbol_id=0, ticker="AAPL", asset_class="equity", name="Apple Inc.")
        sid = sm.upsert_symbol(record)
        assert sid > 0

        result = sm.get_by_ticker("AAPL")
        assert result is not None
        assert result.ticker == "AAPL"
        assert result.name == "Apple Inc."
        assert result.symbol_id == sid

    def test_get_by_id(self, tmp_dir: Path) -> None:
        sm = SymbolMaster(db_path=tmp_dir / "sm.db")
        sid = sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="MSFT", asset_class="equity"))
        result = sm.get_by_id(sid)
        assert result is not None
        assert result.ticker == "MSFT"

    def test_get_symbol_id(self, tmp_dir: Path) -> None:
        sm = SymbolMaster(db_path=tmp_dir / "sm.db")
        sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="GOOGL", asset_class="equity"))
        sid = sm.get_symbol_id("GOOGL")
        assert sid is not None
        assert sid > 0

        missing = sm.get_symbol_id("ZZZZ")
        assert missing is None

    def test_list_symbols(self, tmp_dir: Path) -> None:
        sm = SymbolMaster(db_path=tmp_dir / "sm.db")
        sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="AAPL", asset_class="equity"))
        sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="BTC", asset_class="crypto"))
        sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="MSFT", asset_class="equity"))

        equities = sm.list_symbols(asset_class="equity")
        assert len(equities) == 2

        all_symbols = sm.list_symbols()
        assert len(all_symbols) == 3

    def test_bulk_upsert(self, tmp_dir: Path) -> None:
        sm = SymbolMaster(db_path=tmp_dir / "sm.db")
        records = [
            SymbolRecord(symbol_id=0, ticker=f"SYM{i}", asset_class="equity")
            for i in range(10)
        ]
        count = sm.bulk_upsert(records)
        assert count == 10
        assert sm.symbol_count == 10

    def test_upsert_updates_existing(self, tmp_dir: Path) -> None:
        sm = SymbolMaster(db_path=tmp_dir / "sm.db")
        sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="AAPL", asset_class="equity", name="Old"))
        sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="AAPL", asset_class="equity", name="Apple Inc."))
        result = sm.get_by_ticker("AAPL")
        assert result is not None
        assert result.name == "Apple Inc."
        assert sm.symbol_count == 1

    def test_options_contract(self, tmp_dir: Path) -> None:
        sm = SymbolMaster(db_path=tmp_dir / "sm.db")
        sid = sm.upsert_symbol(SymbolRecord(symbol_id=0, ticker="AAPL", asset_class="equity"))
        contract = OptionsContract(
            contract_id=0,
            underlying_symbol_id=sid,
            strike=150.0,
            expiration="2024-12-20",
            option_type="call",
        )
        cid = sm.upsert_options_contract(contract)
        assert cid > 0

        contracts = sm.get_options_contracts(sid)
        assert len(contracts) == 1
        assert contracts[0].strike == 150.0
