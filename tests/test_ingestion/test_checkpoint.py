"""Tests for checkpoint manager."""

from __future__ import annotations

from pathlib import Path

from market_data.ingestion.checkpoint import CheckpointManager, CheckpointState


class TestCheckpointManager:
    def test_save_and_load(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(db_path=tmp_dir / "checkpoints.db")
        state = CheckpointState(
            vendor="databento",
            symbol="AAPL",
            schema="ohlcv-1d",
            start_date="2024-01-01",
            end_date="2024-01-31",
            records_processed=500,
            status="in_progress",
        )
        mgr.save(state)
        loaded = mgr.load("databento", "AAPL", "ohlcv-1d", "2024-01-01", "2024-01-31")
        assert loaded is not None
        assert loaded.records_processed == 500
        assert loaded.status == "in_progress"

    def test_mark_completed(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(db_path=tmp_dir / "checkpoints.db")
        state = CheckpointState(
            vendor="databento",
            symbol="AAPL",
            schema="ohlcv-1d",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        mgr.save(state)
        mgr.mark_completed(state)
        loaded = mgr.load("databento", "AAPL", "ohlcv-1d", "2024-01-01", "2024-01-31")
        assert loaded is not None
        assert loaded.status == "completed"

    def test_get_incomplete(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(db_path=tmp_dir / "checkpoints.db")
        state1 = CheckpointState(
            vendor="databento",
            symbol="AAPL",
            schema="ohlcv-1d",
            start_date="2024-01-01",
            end_date="2024-01-31",
            status="in_progress",
        )
        state2 = CheckpointState(
            vendor="databento",
            symbol="MSFT",
            schema="ohlcv-1d",
            start_date="2024-01-01",
            end_date="2024-01-31",
            status="in_progress",
        )
        mgr.save(state1)
        mgr.save(state2)
        mgr.mark_completed(state2)

        incomplete = mgr.get_incomplete("databento")
        assert len(incomplete) == 1
        assert incomplete[0].symbol == "AAPL"

    def test_should_checkpoint_by_records(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(
            db_path=tmp_dir / "checkpoints.db",
            checkpoint_every_records=100,
        )
        assert not mgr.should_checkpoint(50, 0)
        assert mgr.should_checkpoint(100, 0)
        assert mgr.should_checkpoint(150, 0)

    def test_should_checkpoint_by_bytes(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(
            db_path=tmp_dir / "checkpoints.db",
            checkpoint_every_mb=1,
        )
        assert not mgr.should_checkpoint(0, 500_000)
        assert mgr.should_checkpoint(0, 1_048_576)

    def test_clear_completed(self, tmp_dir: Path) -> None:
        mgr = CheckpointManager(db_path=tmp_dir / "checkpoints.db")
        state = CheckpointState(
            vendor="databento",
            symbol="AAPL",
            schema="ohlcv-1d",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        mgr.save(state)
        mgr.mark_completed(state)
        removed = mgr.clear_completed()
        assert removed == 1
