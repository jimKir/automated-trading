"""
DataStore — unified data loading from local disk or S3.

Transparently reads parquet files from local disk or S3 based on environment.
All parquet reads for the historical/daily universe go through this module.

Usage:
    from data.data_store import DataStore, get_store

    store = get_store()           # auto-detects local vs S3
    spy = store.load('SPY')       # returns pd.DataFrame or None
    all_data = store.load_universe(['SPY', 'QQQ', 'IWM'])

Environment detection:
    DATA_SOURCE=s3     -> always use S3
    DATA_SOURCE=local  -> always use local (default on non-EC2)
    (unset)            -> auto-detect: S3 on EC2/ECS, local otherwise

Config priority: env var > settings.yaml > defaults
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Defaults — can be overridden via settings.yaml or constructor args
S3_BUCKET = 'trading-data-380277571671-eu-north-1-an'
S3_PREFIX = 'historical/daily'
S3_REGION = 'eu-north-1'
LOCAL_DATA_DIR = str(Path(__file__).parent.parent / 'data' / 'historical' / 'daily')


def _is_ec2() -> bool:
    """Detect if running on EC2/ECS via instance metadata endpoint."""
    try:
        import urllib.request
        urllib.request.urlopen(
            'http://169.254.169.254/latest/meta-data/', timeout=0.5
        )
        return True
    except Exception:
        return False


def _use_s3() -> bool:
    """Returns True if data should be loaded from S3."""
    ds = os.environ.get('DATA_SOURCE', '').lower()
    if ds == 's3':
        return True
    if ds == 'local':
        return False
    return _is_ec2()


def _load_settings_yaml() -> dict:
    """Load data config from settings.yaml if available."""
    try:
        import yaml
        settings_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
        if settings_path.exists():
            with open(settings_path) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get('data', {})
    except Exception:
        pass
    return {}


class DataStore:
    """
    Unified parquet data loader.
    Transparently reads from local disk or S3 based on environment.

    Environment variables:
        DATA_SOURCE=s3     -> always use S3
        DATA_SOURCE=local  -> always use local (default on non-EC2)
        AWS_DEFAULT_REGION -> defaults to eu-north-1
    """

    def __init__(
        self,
        local_dir: str | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
        s3_region: str | None = None,
        use_s3: bool | None = None,
    ):
        # Load from settings.yaml as base, then override with explicit args
        yaml_cfg = _load_settings_yaml()

        self.local_dir = Path(local_dir or yaml_cfg.get('local_dir', LOCAL_DATA_DIR))
        self.s3_bucket = s3_bucket or yaml_cfg.get('s3_bucket', S3_BUCKET)
        self.s3_prefix = (s3_prefix or yaml_cfg.get('s3_prefix', S3_PREFIX)).rstrip('/')
        self.s3_region = s3_region or yaml_cfg.get('s3_region', S3_REGION)
        self.use_s3 = use_s3 if use_s3 is not None else _use_s3()
        self._s3_client = None

        source = 'S3' if self.use_s3 else 'local'
        logger.info(f"[DataStore] Using {source} data source")
        if self.use_s3:
            logger.info(f"[DataStore] s3://{self.s3_bucket}/{self.s3_prefix}/")
        else:
            logger.info(f"[DataStore] {self.local_dir}/")

    def _get_s3_client(self):
        """Lazy S3 client init — only imported if actually using S3."""
        if self._s3_client is None:
            import boto3
            self._s3_client = boto3.client('s3', region_name=self.s3_region)
        return self._s3_client

    def _symbol_to_filename(self, symbol: str) -> str:
        """Convert symbol to parquet filename. Handles BTC/USD -> BTC_USD etc."""
        return symbol.replace('/', '_').replace('-', '_') + '.parquet'

    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load daily OHLCV data for a symbol.
        Returns None if symbol not found (never raises).

        Args:
            symbol: e.g. 'SPY', 'BTC/USD', 'BTC-USD'
            start_date: optional filter 'YYYY-MM-DD'
            end_date: optional filter 'YYYY-MM-DD'
        """
        fname = self._symbol_to_filename(symbol)

        try:
            if self.use_s3:
                df = self._load_from_s3(fname)
            else:
                df = self._load_from_local(fname)

            if df is None:
                return None

            # Normalise column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Apply date filters
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]

            return df

        except Exception as e:
            logger.warning(f"[DataStore] Failed to load {symbol}: {e}")
            return None

    def _load_from_local(self, fname: str) -> Optional[pd.DataFrame]:
        """Load parquet from local disk — tries multiple naming conventions."""
        # Build list of candidate filenames to try
        base = fname.replace('.parquet', '')
        candidates = [
            fname,                                          # BTC_USD.parquet (canonical)
            base.replace('_', '-') + '.parquet',            # BTC-USD.parquet
            base.split('_')[0] + '.parquet',                # BTC.parquet (no suffix)
            base.lower() + '.parquet',                      # btc_usd.parquet
            base.upper() + '.parquet',                      # BTC_USD.parquet
            base.lower().replace('_', '-') + '.parquet',    # btc-usd.parquet
            base.split('_')[0].upper() + '.parquet',        # BTC.parquet (uppercase, no suffix)
            base.replace('=', '_') + '.parquet',            # ES_F.parquet (futures)
            base.replace('=F', '') + '.parquet',            # ES.parquet (futures without F)
            base.lstrip('^') + '.parquet',                  # VIX.parquet (strip ^ prefix)
        ]
        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        for candidate in unique_candidates:
            path = self.local_dir / candidate
            if path.exists():
                if candidate != fname:
                    logger.debug(f"[DataStore] {fname} -> found as {candidate}")
                return pd.read_parquet(path)

        logger.debug(f"[DataStore] Not found locally: {fname}")
        return None

    def _load_from_s3(self, fname: str) -> Optional[pd.DataFrame]:
        """Load parquet from S3 using s3fs or boto3 fallback."""
        s3_path = f's3://{self.s3_bucket}/{self.s3_prefix}/{fname}'

        # Try s3fs first (faster, supports pandas read_parquet directly)
        try:
            import s3fs
            fs = s3fs.S3FileSystem(
                client_kwargs={'region_name': self.s3_region}
            )
            if not fs.exists(s3_path):
                logger.debug(f"[DataStore] Not found in S3: {s3_path}")
                return None
            with fs.open(s3_path, 'rb') as f:
                return pd.read_parquet(f)
        except ImportError:
            pass  # fall through to boto3

        # Fallback: boto3 download to memory
        try:
            client = self._get_s3_client()
            key = f'{self.s3_prefix}/{fname}'
            response = client.get_object(Bucket=self.s3_bucket, Key=key)
            return pd.read_parquet(io.BytesIO(response['Body'].read()))
        except Exception as e:
            # Check for NoSuchKey specifically
            if 'NoSuchKey' in str(type(e).__name__) or 'NoSuchKey' in str(e):
                logger.debug(f"[DataStore] Not found in S3: {fname}")
            else:
                logger.warning(f"[DataStore] S3 load failed for {fname}: {e}")
            return None

    def load_universe(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple symbols. Returns dict of {symbol: DataFrame}.
        Silently skips missing symbols.
        """
        result = {}
        for symbol in symbols:
            df = self.load(symbol, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                result[symbol] = df
            else:
                logger.warning(f"[DataStore] Skipping {symbol} — no data available")
        return result

    def list_available(self) -> List[str]:
        """List all available symbols."""
        if self.use_s3:
            return self._list_s3()
        return self._list_local()

    def _list_local(self) -> List[str]:
        if not self.local_dir.exists():
            return []
        return [f.stem.replace('_', '/') for f in self.local_dir.glob('*.parquet')]

    def _list_s3(self) -> List[str]:
        try:
            client = self._get_s3_client()
            response = client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f'{self.s3_prefix}/'
            )
            files = [
                obj['Key'].split('/')[-1]
                for obj in response.get('Contents', [])
            ]
            return [
                f.replace('.parquet', '').replace('_', '/')
                for f in files if f.endswith('.parquet')
            ]
        except Exception as e:
            logger.warning(f"[DataStore] Cannot list S3: {e}")
            return []

    def save(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Save DataFrame as parquet. Saves locally; optionally syncs to S3.
        Used by daily_data_update.py and historical_store.py.
        """
        fname = self._symbol_to_filename(symbol)

        # Always save locally first
        self.local_dir.mkdir(parents=True, exist_ok=True)
        local_path = self.local_dir / fname
        df.to_parquet(local_path)

        # If in S3 mode, also upload
        if self.use_s3:
            try:
                import boto3
                client = boto3.client('s3', region_name=self.s3_region)
                key = f'{self.s3_prefix}/{fname}'
                client.upload_file(str(local_path), self.s3_bucket, key)
                logger.info(f"[DataStore] Uploaded {symbol} to S3")
            except Exception as e:
                logger.warning(f"[DataStore] S3 upload failed for {symbol}: {e}")
                return False
        return True


# Module-level singleton — import and use directly
_store: Optional[DataStore] = None


def get_store() -> DataStore:
    """Get or create the global DataStore instance."""
    global _store
    if _store is None:
        _store = DataStore()
    return _store


def reset_store():
    """Reset the singleton (useful for testing)."""
    global _store
    _store = None
