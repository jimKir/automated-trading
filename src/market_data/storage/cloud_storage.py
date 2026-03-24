"""Cloud storage abstraction supporting local filesystem, Azure Blob, AWS S3, GCS."""

from __future__ import annotations

import abc
import os
import shutil
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class StorageBackend(abc.ABC):
    """Abstract storage backend interface.

    Provides a uniform API for local and cloud storage operations.
    """

    @abc.abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write bytes to the given path."""

    @abc.abstractmethod
    def read(self, path: str) -> bytes:
        """Read bytes from the given path."""

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists."""

    @abc.abstractmethod
    def list_files(self, prefix: str) -> list[str]:
        """List files under a prefix."""

    @abc.abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file at the given path."""

    @abc.abstractmethod
    def write_file(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a local file to storage."""

    @abc.abstractmethod
    def read_file(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from storage to local path."""

    @abc.abstractmethod
    def get_size(self, path: str) -> int:
        """Get size of a file in bytes."""


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend.

    Args:
        base_path: Root directory for all storage operations.
    """

    def __init__(self, base_path: str | Path = "/data") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        """Resolve a storage path to a local filesystem path."""
        return self.base_path / path.lstrip("/")

    def write(self, path: str, data: bytes) -> None:
        """Write bytes to local filesystem."""
        full_path = self._resolve(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)

    def read(self, path: str) -> bytes:
        """Read bytes from local filesystem."""
        return self._resolve(path).read_bytes()

    def exists(self, path: str) -> bool:
        """Check if file exists on local filesystem."""
        return self._resolve(path).exists()

    def list_files(self, prefix: str) -> list[str]:
        """List files under a prefix on local filesystem."""
        full_path = self._resolve(prefix)
        if not full_path.exists():
            return []
        return [
            str(p.relative_to(self.base_path))
            for p in full_path.rglob("*")
            if p.is_file()
        ]

    def delete(self, path: str) -> None:
        """Delete a file from local filesystem."""
        full_path = self._resolve(path)
        if full_path.exists():
            full_path.unlink()

    def write_file(self, local_path: str | Path, remote_path: str) -> None:
        """Copy a local file to storage location."""
        dest = self._resolve(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(dest))

    def read_file(self, remote_path: str, local_path: str | Path) -> None:
        """Copy a storage file to local path."""
        src = self._resolve(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(local_path))

    def get_size(self, path: str) -> int:
        """Get file size on local filesystem."""
        full_path = self._resolve(path)
        return full_path.stat().st_size if full_path.exists() else 0


class AzureBlobStorageBackend(StorageBackend):
    """Azure Blob Storage backend.

    Args:
        account_name: Azure storage account name.
        account_key: Azure storage account key.
        container_name: Blob container name.
    """

    def __init__(
        self,
        account_name: str | None = None,
        account_key: str | None = None,
        container_name: str = "market-data",
    ) -> None:
        self.account_name = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT", "")
        self.account_key = account_key or os.environ.get("AZURE_STORAGE_KEY", "")
        self.container_name = container_name
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Azure Blob client."""
        if self._client is None:
            from azure.storage.blob import BlobServiceClient

            conn_str = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.account_name};"
                f"AccountKey={self.account_key};"
                f"EndpointSuffix=core.windows.net"
            )
            self._client = BlobServiceClient.from_connection_string(conn_str)
        return self._client

    def _container_client(self) -> Any:
        return self._get_client().get_container_client(self.container_name)

    def write(self, path: str, data: bytes) -> None:
        """Write bytes to Azure Blob Storage."""
        blob = self._container_client().get_blob_client(path)
        blob.upload_blob(data, overwrite=True)

    def read(self, path: str) -> bytes:
        """Read bytes from Azure Blob Storage."""
        blob = self._container_client().get_blob_client(path)
        return blob.download_blob().readall()

    def exists(self, path: str) -> bool:
        """Check if blob exists."""
        blob = self._container_client().get_blob_client(path)
        try:
            blob.get_blob_properties()
            return True
        except Exception:
            return False

    def list_files(self, prefix: str) -> list[str]:
        """List blobs under a prefix."""
        blobs = self._container_client().list_blobs(name_starts_with=prefix)
        return [b.name for b in blobs]

    def delete(self, path: str) -> None:
        """Delete a blob."""
        blob = self._container_client().get_blob_client(path)
        blob.delete_blob()

    def write_file(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a local file to Azure Blob."""
        with open(local_path, "rb") as f:
            self.write(remote_path, f.read())

    def read_file(self, remote_path: str, local_path: str | Path) -> None:
        """Download a blob to local path."""
        data = self.read(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(data)

    def get_size(self, path: str) -> int:
        """Get blob size."""
        blob = self._container_client().get_blob_client(path)
        props = blob.get_blob_properties()
        return props.size


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend.

    Args:
        bucket: S3 bucket name.
        region: AWS region.
    """

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
    ) -> None:
        self.bucket = bucket or os.environ.get("S3_BUCKET", "market-data")
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the S3 client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def write(self, path: str, data: bytes) -> None:
        """Write bytes to S3."""
        self._get_client().put_object(Bucket=self.bucket, Key=path, Body=data)

    def read(self, path: str) -> bytes:
        """Read bytes from S3."""
        resp = self._get_client().get_object(Bucket=self.bucket, Key=path)
        return resp["Body"].read()

    def exists(self, path: str) -> bool:
        """Check if object exists in S3."""
        try:
            self._get_client().head_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False

    def list_files(self, prefix: str) -> list[str]:
        """List objects under a prefix in S3."""
        resp = self._get_client().list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]

    def delete(self, path: str) -> None:
        """Delete an object from S3."""
        self._get_client().delete_object(Bucket=self.bucket, Key=path)

    def write_file(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a local file to S3."""
        self._get_client().upload_file(str(local_path), self.bucket, remote_path)

    def read_file(self, remote_path: str, local_path: str | Path) -> None:
        """Download an S3 object to local path."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._get_client().download_file(self.bucket, remote_path, str(local_path))

    def get_size(self, path: str) -> int:
        """Get S3 object size."""
        resp = self._get_client().head_object(Bucket=self.bucket, Key=path)
        return resp["ContentLength"]


class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend.

    Args:
        bucket: GCS bucket name.
        project: GCP project ID.
    """

    def __init__(
        self,
        bucket: str | None = None,
        project: str | None = None,
    ) -> None:
        self.bucket_name = bucket or os.environ.get("GCS_BUCKET", "market-data")
        self.project = project or os.environ.get("GCS_PROJECT", "")
        self._client: Any = None

    def _get_bucket(self) -> Any:
        """Lazy-initialize the GCS bucket."""
        if self._client is None:
            from google.cloud import storage

            client = storage.Client(project=self.project)
            self._client = client.bucket(self.bucket_name)
        return self._client

    def write(self, path: str, data: bytes) -> None:
        """Write bytes to GCS."""
        blob = self._get_bucket().blob(path)
        blob.upload_from_string(data)

    def read(self, path: str) -> bytes:
        """Read bytes from GCS."""
        blob = self._get_bucket().blob(path)
        return blob.download_as_bytes()

    def exists(self, path: str) -> bool:
        """Check if blob exists in GCS."""
        return self._get_bucket().blob(path).exists()

    def list_files(self, prefix: str) -> list[str]:
        """List blobs under a prefix in GCS."""
        blobs = self._get_bucket().list_blobs(prefix=prefix)
        return [b.name for b in blobs]

    def delete(self, path: str) -> None:
        """Delete a blob from GCS."""
        self._get_bucket().blob(path).delete()

    def write_file(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a local file to GCS."""
        blob = self._get_bucket().blob(remote_path)
        blob.upload_from_filename(str(local_path))

    def read_file(self, remote_path: str, local_path: str | Path) -> None:
        """Download a GCS blob to local path."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob = self._get_bucket().blob(remote_path)
        blob.download_to_filename(str(local_path))

    def get_size(self, path: str) -> int:
        """Get blob size in GCS."""
        blob = self._get_bucket().blob(path)
        blob.reload()
        return blob.size or 0


class CloudStorageFactory:
    """Factory for creating storage backends.

    Supports local filesystem, Azure Blob, AWS S3, and GCS.

    Args:
        provider: Storage provider name (local/azure/s3/gcs).
        **kwargs: Provider-specific configuration.

    Returns:
        Configured StorageBackend instance.
    """

    _PROVIDERS: dict[str, type[StorageBackend]] = {
        "local": LocalStorageBackend,
        "azure": AzureBlobStorageBackend,
        "s3": S3StorageBackend,
        "gcs": GCSStorageBackend,
    }

    @classmethod
    def create(cls, provider: str = "local", **kwargs: Any) -> StorageBackend:
        """Create a storage backend for the given provider.

        Args:
            provider: Storage provider (local, azure, s3, gcs).
            **kwargs: Provider-specific arguments.

        Returns:
            StorageBackend instance.

        Raises:
            ValueError: If provider is not supported.
        """
        backend_cls = cls._PROVIDERS.get(provider.lower())
        if backend_cls is None:
            raise ValueError(
                f"Unknown storage provider: {provider}. "
                f"Supported: {list(cls._PROVIDERS.keys())}"
            )
        logger.info("creating_storage_backend", provider=provider)
        return backend_cls(**kwargs)

    @classmethod
    def register(cls, name: str, backend_cls: type[StorageBackend]) -> None:
        """Register a custom storage backend.

        Args:
            name: Provider name.
            backend_cls: Backend class implementing StorageBackend.
        """
        cls._PROVIDERS[name.lower()] = backend_cls
