"""Tests for cloud storage backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from market_data.storage.cloud_storage import (
    CloudStorageFactory,
    LocalStorageBackend,
)


class TestLocalStorageBackend:
    def test_write_and_read(self, tmp_dir: Path) -> None:
        storage = LocalStorageBackend(base_path=str(tmp_dir))
        data = b"test data content"
        storage.write("test/file.bin", data)

        result = storage.read("test/file.bin")
        assert result == data

    def test_exists(self, tmp_dir: Path) -> None:
        storage = LocalStorageBackend(base_path=str(tmp_dir))
        assert not storage.exists("nonexistent.txt")

        storage.write("exists.txt", b"hello")
        assert storage.exists("exists.txt")

    def test_list_files(self, tmp_dir: Path) -> None:
        storage = LocalStorageBackend(base_path=str(tmp_dir))
        storage.write("dir/a.txt", b"a")
        storage.write("dir/b.txt", b"b")
        storage.write("other/c.txt", b"c")

        files = storage.list_files("dir")
        assert len(files) == 2

    def test_delete(self, tmp_dir: Path) -> None:
        storage = LocalStorageBackend(base_path=str(tmp_dir))
        storage.write("deleteme.txt", b"data")
        assert storage.exists("deleteme.txt")

        storage.delete("deleteme.txt")
        assert not storage.exists("deleteme.txt")

    def test_get_size(self, tmp_dir: Path) -> None:
        storage = LocalStorageBackend(base_path=str(tmp_dir))
        data = b"0" * 100
        storage.write("sized.bin", data)
        assert storage.get_size("sized.bin") == 100

    def test_write_file_and_read_file(self, tmp_dir: Path) -> None:
        storage = LocalStorageBackend(base_path=str(tmp_dir / "storage"))
        source = tmp_dir / "source.txt"
        source.write_bytes(b"file content")

        # write_file(local_path, remote_path)
        storage.write_file(str(source), "uploaded.txt")
        assert storage.exists("uploaded.txt")

        dest = tmp_dir / "downloaded.txt"
        storage.read_file("uploaded.txt", str(dest))
        assert dest.read_bytes() == b"file content"


class TestCloudStorageFactory:
    def test_create_local(self, tmp_dir: Path) -> None:
        storage = CloudStorageFactory.create(
            provider="local",
            base_path=str(tmp_dir),
        )
        assert isinstance(storage, LocalStorageBackend)

    def test_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown storage provider"):
            CloudStorageFactory.create(provider="unknown_cloud")
