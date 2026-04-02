"""
Market Data Catalogue
=====================
Central registry for all fetched market data: what was fetched, from where,
at what frequency, covering what date range, stored where on disk.

Every fetch call writes a catalogue entry. The catalogue answers:
  - "Do I already have SPY daily bars from Alpaca for 2023?"
  - "What's the freshest OPRA imbalance data I have for AAPL?"
  - "Show me all data sources I have for NVDA"

Usage:
    from src.market_data.catalogue import DataCatalogue

    cat = DataCatalogue()

    # Log a fetch
    cat.record(
        source    = "databento",
        dataset   = "XNAS.ITCH",
        schema    = "imbalance",
        symbols   = ["AAPL", "MSFT"],
        frequency = "daily",
        start     = "2023-01-01",
        end       = "2026-03-21",
        cache_path= "~/.databento_cache/imbalance/abc123.json",
        rows      = 1764,
    )

    # Check before fetching
    entry = cat.find(source="databento", schema="imbalance",
                     symbol="AAPL", date="2024-06-15")
    if entry:
        # load from entry["cache_path"] — skip API call
        ...

    # Inspect everything
    cat.summary()
    cat.find_all(source="alpaca")
"""
from __future__ import annotations

import json
import hashlib
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ── Catalogue file location ───────────────────────────────────────────────────
CATALOGUE_DIR  = Path.home() / ".databento_cache"
CATALOGUE_FILE = CATALOGUE_DIR / "catalogue.json"

# Frequency normalisation
FREQ_ALIASES = {
    "1m": "1min", "1min": "1min", "minute": "1min",
    "1h": "1hour", "1hr": "1hour", "hourly": "1hour",
    "1d": "1day", "daily": "1day", "day": "1day",
    "1w": "1week", "weekly": "1week", "week": "1week",
    "tick": "tick", "trade": "tick", "trades": "tick",
    "snapshot": "snapshot", "imbalance": "snapshot",
    "stats": "snapshot", "statistics": "snapshot",
    "ohlcv-1m": "1min", "ohlcv-1d": "1day",
    "ohlcv-1h": "1hour", "cbbo-1m": "1min",
}


def _norm_freq(freq: str) -> str:
    return FREQ_ALIASES.get(freq.lower(), freq.lower())


def _date_str(d: Union[str, date, datetime, None]) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _entry_key(source: str, dataset: str, schema: str,
                symbols: List[str], start: str, end: str) -> str:
    """Deterministic hash key for a catalogue entry."""
    parts = f"{source}|{dataset}|{schema}|{','.join(sorted(symbols))}|{start}|{end}"
    return hashlib.md5(parts.encode()).hexdigest()[:16]


# ── Main Catalogue Class ──────────────────────────────────────────────────────

class DataCatalogue:
    """
    Persistent JSON catalogue of all fetched market data.

    Schema of each entry:
    {
      "key":        "abc123def456",       # unique identifier
      "source":     "databento",          # databento | alpaca | yfinance | fred
      "dataset":    "XNAS.ITCH",          # provider dataset name
      "schema":     "imbalance",          # data schema / table
      "frequency":  "snapshot",           # 1min | 1day | tick | snapshot etc.
      "symbols":    ["AAPL","MSFT"],      # list of symbols
      "start":      "2023-01-01",         # coverage start (YYYY-MM-DD)
      "end":        "2026-03-21",         # coverage end (YYYY-MM-DD)
      "rows":       1764,                 # approximate row count
      "cache_path": "~/.databento_cache/imbalance/abc.json",
      "fetched_at": "2026-04-02T14:32:11",
      "notes":      "closing auction window 19:50-19:59 UTC",
      "tags":       ["signal", "microstructure"]
    }
    """

    def __init__(self, catalogue_path: Optional[Path] = None):
        self._path = catalogue_path or CATALOGUE_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, dict] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, dict]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self):
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, default=str)
            )
        except Exception as e:
            print(f"  [Catalogue] Warning: could not save catalogue: {e}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(
        self,
        source:     str,
        dataset:    str,
        schema:     str,
        symbols:    Union[List[str], str],
        start:      Union[str, date, datetime],
        end:        Union[str, date, datetime],
        frequency:  str            = "1day",
        rows:       int            = 0,
        cache_path: Optional[str]  = None,
        notes:      str            = "",
        tags:       List[str]      = None,
        extra:      Dict[str, Any] = None,
    ) -> str:
        """
        Record a data fetch in the catalogue.
        Returns the entry key.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = sorted(set(symbols))
        start_s = _date_str(start)
        end_s   = _date_str(end)
        freq    = _norm_freq(frequency)

        key = _entry_key(source, dataset, schema, symbols, start_s, end_s)

        entry = {
            "key":        key,
            "source":     source.lower(),
            "dataset":    dataset,
            "schema":     schema,
            "frequency":  freq,
            "symbols":    symbols,
            "start":      start_s,
            "end":        end_s,
            "rows":       rows,
            "cache_path": str(cache_path) if cache_path else None,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "notes":      notes,
            "tags":       tags or [],
            "extra":      extra or {},
        }

        # Merge if entry already exists (update end date and row count)
        if key in self._data:
            existing = self._data[key]
            # Extend coverage if new data covers more dates
            if end_s and (existing.get("end") or "") < end_s:
                existing["end"] = end_s
            existing["rows"]       = max(existing.get("rows", 0), rows)
            existing["fetched_at"] = entry["fetched_at"]
            if notes:
                existing["notes"]  = notes
        else:
            self._data[key] = entry

        self._save()
        return key

    def update_end(self, key: str, new_end: Union[str, date], rows: int = 0):
        """Extend the end date of an existing entry (incremental fetch)."""
        if key not in self._data:
            return
        self._data[key]["end"]        = _date_str(new_end)
        self._data[key]["fetched_at"] = datetime.now().isoformat(timespec="seconds")
        if rows:
            self._data[key]["rows"] = rows
        self._save()

    # ── Read ──────────────────────────────────────────────────────────────────

    def find(
        self,
        source:  Optional[str] = None,
        dataset: Optional[str] = None,
        schema:  Optional[str] = None,
        symbol:  Optional[str] = None,
        date:    Optional[Union[str, "date"]] = None,
    ) -> Optional[dict]:
        """
        Find the best matching catalogue entry.
        Returns the first match or None.

        date: if provided, checks that the entry covers this date.
        """
        date_s = _date_str(date) if date else None

        for entry in self._data.values():
            if source  and entry.get("source")  != source.lower():  continue
            if dataset and entry.get("dataset") != dataset:          continue
            if schema  and entry.get("schema")  != schema:           continue
            if symbol  and symbol not in entry.get("symbols", []):   continue
            if date_s:
                s = entry.get("start") or "0000-01-01"
                e = entry.get("end")   or "9999-12-31"
                if not (s <= date_s <= e):
                    continue
            return entry
        return None

    def find_all(
        self,
        source:    Optional[str] = None,
        schema:    Optional[str] = None,
        symbol:    Optional[str] = None,
        frequency: Optional[str] = None,
        tag:       Optional[str] = None,
    ) -> List[dict]:
        """Return all entries matching the given filters."""
        freq_norm = _norm_freq(frequency) if frequency else None
        results = []
        for entry in self._data.values():
            if source    and entry.get("source")    != source.lower():    continue
            if schema    and entry.get("schema")    != schema:             continue
            if symbol    and symbol not in entry.get("symbols", []):       continue
            if freq_norm and entry.get("frequency") != freq_norm:          continue
            if tag       and tag not in entry.get("tags", []):             continue
            results.append(entry)
        return sorted(results, key=lambda x: x.get("fetched_at", ""))

    def has(
        self,
        source:  str,
        schema:  str,
        symbol:  str,
        date:    Union[str, "date"],
    ) -> bool:
        """Quick check: do we already have this data point?"""
        return self.find(source=source, schema=schema,
                         symbol=symbol, date=date) is not None

    def coverage(self, source: str, schema: str, symbol: str) -> Optional[tuple]:
        """Return (start, end) coverage for a symbol, or None if not found."""
        entry = self.find(source=source, schema=schema, symbol=symbol)
        if entry:
            return entry.get("start"), entry.get("end")
        return None

    # ── Display ───────────────────────────────────────────────────────────────

    def summary(self, source: Optional[str] = None):
        """Print a human-readable summary of the catalogue."""
        entries = self.find_all(source=source)

        if not entries:
            print("  [Catalogue] Empty — no data recorded yet.")
            return

        print()
        print("=" * 80)
        print(f"  DATA CATALOGUE  ({len(entries)} entries)"
              + (f"  filter: source={source}" if source else ""))
        print("=" * 80)
        print(f"  {'Source':<12} {'Dataset':<18} {'Schema':<14} {'Freq':<10} "
              f"{'Symbols':>7} {'Start':<12} {'End':<12} {'Rows':>8}")
        print("  " + "─" * 80)

        by_source: Dict[str, List[dict]] = {}
        for e in entries:
            by_source.setdefault(e["source"], []).append(e)

        for src in sorted(by_source):
            for e in sorted(by_source[src],
                            key=lambda x: (x["dataset"], x["schema"])):
                syms = e.get("symbols", [])
                sym_str = syms[0] if len(syms) == 1 else f"{len(syms)} syms"
                rows    = e.get("rows", 0)
                rows_s  = f"{rows:,}" if rows else "—"
                print(f"  {e['source']:<12} {e['dataset']:<18} {e['schema']:<14} "
                      f"{e['frequency']:<10} {sym_str:>7} "
                      f"{e.get('start','?'):<12} {e.get('end','?'):<12} {rows_s:>8}")
                if e.get("notes"):
                    print(f"  {'':12} {e['notes']}")
        print("=" * 80)
        print(f"  Catalogue file: {self._path}")
        print()

    def __repr__(self):
        return f"DataCatalogue({len(self._data)} entries, path={self._path})"


# ── Module-level singleton ────────────────────────────────────────────────────
_catalogue: Optional[DataCatalogue] = None

def get_catalogue() -> DataCatalogue:
    """Get (or create) the module-level catalogue singleton."""
    global _catalogue
    if _catalogue is None:
        _catalogue = DataCatalogue()
    return _catalogue
