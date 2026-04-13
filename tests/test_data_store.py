"""
Tests for data.data_store — unified local/S3 data loading.
"""

import pandas as pd
import pytest

from data.data_store import DataStore, _use_s3, reset_store


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton before each test."""
    reset_store()
    yield
    reset_store()


@pytest.fixture
def sample_parquet(tmp_path):
    """Create a sample SPY parquet file in a temp directory."""
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0], "volume": [1000, 1100, 1200]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    df.to_parquet(tmp_path / "SPY.parquet")
    return tmp_path


class TestDataStoreLocal:
    def test_load_spy_local(self, sample_parquet):
        """DataStore loads SPY from local parquet."""
        store = DataStore(local_dir=str(sample_parquet), use_s3=False)
        result = store.load("SPY")
        assert result is not None
        assert len(result) == 3
        assert "close" in result.columns

    def test_returns_none_for_missing_symbol(self, tmp_path):
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        result = store.load("NONEXISTENT")
        assert result is None

    def test_date_filter(self, tmp_path):
        df = pd.DataFrame(
            {"close": range(100)},
            index=pd.date_range("2020-01-01", periods=100),
        )
        df.to_parquet(tmp_path / "SPY.parquet")
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        result = store.load("SPY", start_date="2020-03-01")
        assert result is not None
        assert result.index.min() >= pd.Timestamp("2020-03-01")

    def test_date_filter_end(self, tmp_path):
        df = pd.DataFrame(
            {"close": range(100)},
            index=pd.date_range("2020-01-01", periods=100),
        )
        df.to_parquet(tmp_path / "SPY.parquet")
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        result = store.load("SPY", end_date="2020-02-01")
        assert result is not None
        assert result.index.max() <= pd.Timestamp("2020-02-01")

    def test_load_universe(self, tmp_path):
        for sym in ["SPY", "QQQ"]:
            df = pd.DataFrame(
                {"close": [100.0]},
                index=pd.date_range("2024-01-01", periods=1),
            )
            df.to_parquet(tmp_path / f"{sym}.parquet")
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        result = store.load_universe(["SPY", "QQQ", "MISSING"])
        assert "SPY" in result
        assert "QQQ" in result
        assert "MISSING" not in result

    def test_list_available_local(self, tmp_path):
        for sym in ["SPY", "QQQ", "IWM"]:
            df = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
            df.to_parquet(tmp_path / f"{sym}.parquet")
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        available = store.list_available()
        assert len(available) == 3

    def test_columns_normalised_to_lowercase(self, tmp_path):
        df = pd.DataFrame(
            {"Close": [100.0], "Volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        df.to_parquet(tmp_path / "SPY.parquet")
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        result = store.load("SPY")
        assert "close" in result.columns
        assert "volume" in result.columns
        assert "Close" not in result.columns

    def test_save_and_reload(self, tmp_path):
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        df = pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        ok = store.save("TEST", df)
        assert ok is True
        result = store.load("TEST")
        assert result is not None
        assert len(result) == 2

    def test_symbol_to_filename_slashes(self, tmp_path):
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        assert store._symbol_to_filename("BTC/USD") == "BTC_USD.parquet"
        assert store._symbol_to_filename("BTC-USD") == "BTC_USD.parquet"
        assert store._symbol_to_filename("SPY") == "SPY.parquet"

    def test_list_available_empty_dir(self, tmp_path):
        store = DataStore(local_dir=str(tmp_path / "nonexistent"), use_s3=False)
        assert store.list_available() == []


class TestDataStoreAutoDetect:
    def test_uses_local_when_not_ec2(self, monkeypatch):
        monkeypatch.delenv("DATA_SOURCE", raising=False)
        # On dev machine (not EC2), should use local
        assert _use_s3() is False

    def test_uses_s3_when_env_set(self, monkeypatch):
        monkeypatch.setenv("DATA_SOURCE", "s3")
        assert _use_s3() is True

    def test_uses_local_when_env_set_local(self, monkeypatch):
        monkeypatch.setenv("DATA_SOURCE", "local")
        assert _use_s3() is False

    def test_uses_local_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DATA_SOURCE", "LOCAL")
        assert _use_s3() is False

    def test_uses_s3_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("DATA_SOURCE", "S3")
        assert _use_s3() is True

    def test_constructor_respects_use_s3_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DATA_SOURCE", raising=False)
        store = DataStore(local_dir=str(tmp_path), use_s3=True)
        assert store.use_s3 is True

        store2 = DataStore(local_dir=str(tmp_path), use_s3=False)
        assert store2.use_s3 is False

    def test_default_s3_config(self, tmp_path):
        store = DataStore(local_dir=str(tmp_path), use_s3=False)
        assert store.s3_bucket == "trading-data-380277571671-eu-north-1-an"
        assert store.s3_region == "eu-north-1"
        assert store.s3_prefix == "historical/daily"
