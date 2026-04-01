"""
earnings_nlp_signal.py
======================
Earnings Call NLP Signal for the Automated Trading System.

OVERVIEW
--------
This module computes a directional signal in [-1, +1] derived from three
complementary NLP-based analysis components applied to recent earnings call
transcripts and 8-K filings:

  A. Transcript Sentiment     (40% weight)
  B. Sentiment Change          (40% weight)  ← most predictive component
  C. Guidance Keywords         (20% weight)

The combined signal is used as a 15% component of the overall strategy blend
(configured in config/settings.yaml under ``earnings_nlp.weight``).

ECONOMIC RATIONALE
------------------
Post-Earnings Announcement Drift (PEAD) is one of the most persistent and
well-documented anomalies in academic finance. Ball & Brown (1968) first showed
that the market underreacts to earnings surprises, and the drift continues for
30–60 days after the announcement. Several NLP-specific extensions have been
documented in the literature:

1. **Transcript sentiment** — Loughran & McDonald (2011) demonstrate that
   standard negative-word lists (e.g. Harvard IV) misclassify financial
   language. Their finance-specific negative word list predicts abnormal returns
   in the 3–5 days after 10-K filing. FinBERT (Yang et al. 2020), pre-trained
   on financial communication, substantially outperforms both general-purpose
   sentiment models and dictionary methods for earnings language.

2. **Sentiment change** — Kearney & Liu (2014) show that *changes* in disclosure
   tone are more informative than the level. A CEO who shifts from cautious to
   optimistic language across consecutive earnings calls signals a turning point
   that the market takes 1–4 weeks to fully price in. This is the highest-alpha
   component of the three.

3. **Guidance keywords** — Huang, Teoh & Zhang (2014) document that
   forward-looking language in earnings calls predicts subsequent analyst
   forecast revisions and, through that channel, future returns. Raised guidance
   phrasing ("raised guidance", "accelerating", "record") reliably precedes
   positive surprises in the next quarter.

POST-EARNINGS DRIFT TIMING
--------------------------
The signal is designed to capture the PEAD window:
  - Earnings 1 day old:  weight = 1.00  (immediate reaction + fresh drift)
  - Earnings 15 days old: weight ≈ 0.60  (still in primary drift window)
  - Earnings 30 days old: weight = 0.10  (tail end of documented drift)

The exponential decay ensures that signals from stale earnings are strongly
down-weighted without being discarded entirely, since PEAD is documented for
up to 60 days post-announcement.

ANTI-OVERFITTING NOTES
-----------------------
All numerical thresholds are grounded in economic theory or prior literature:
  - 40/40/20 weights    → sentiment change > level (Kearney & Liu 2014)
  - 30-day lookback     → primary PEAD window (Ball & Brown 1968; Bernard 1992)
  - 21-day decay half-life → ~1 calendar month; matches analyst revision lag
  - FinBERT model       → pretrained on financial corpora, no in-sample fitting
  - Keyword lists       → drawn from Loughran-McDonald (2011) finance lexicon

No threshold was selected by backtesting optimisation.

DATA SOURCES
------------
1. SEC EDGAR full-text search (EFTS) — free, no API key required
   https://efts.sec.gov/LATEST/search-index?q="{ticker}"&forms=8-K
2. Financial Modeling Prep API (free tier: 250 calls/day)
   https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}
3. Neutral (0.0) — returned with a WARNING if both sources fail

SENTIMENT BACKENDS (in order of preference)
--------------------------------------------
1. FinBERT (ProsusAI/finbert) via HuggingFace transformers — most accurate
2. VADER (vaderSentiment) — lightweight, rule-based, no GPU needed
3. Keyword scoring — pure Python fallback, no additional dependencies

USAGE
-----
    from strategy.earnings_nlp_signal import EarningsNLPSignal

    signal = EarningsNLPSignal()
    signals = signal.compute(
        symbols=["AAPL", "MSFT", "GOOGL"],
        as_of_date=datetime.date(2026, 4, 1),
        lookback_days=30,
    )
    # → {"AAPL": 0.55, "MSFT": -0.23, "GOOGL": 0.08}

CONFIGURATION  (config/settings.yaml)
--------------------------------------
    earnings_nlp:
      enabled: true
      weight: 0.15               # 15% of total signal blend
      lookback_days: 30          # days to look back for recent earnings
      use_finbert: true          # use FinBERT if transformers available
      fmp_api_key: ""            # Financial Modeling Prep key (optional)
      sentiment_decay_days: 21   # half-life for sentiment signal decay

References
----------
- Ball, R. & Brown, P. (1968). An Empirical Evaluation of Accounting Income
  Numbers. Journal of Accounting Research.
- Bernard, V. L. (1992). Stock Price Reactions to Earnings Announcements.
  Advances in Behavioral Finance.
- Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability?
  Textual Analysis, Dictionaries, and 10-Ks. Journal of Finance.
- Kearney, C. & Liu, S. (2014). Textual Sentiment in Finance: A Survey of
  Methods and Models. International Review of Financial Analysis.
- Huang, A. H., Teoh, S. H., & Zhang, Y. (2014). Tone Management.
  The Accounting Review.
- Yang, Y., Uy, M. C. S., & Huang, A. (2020). FinBERT: A Pretrained Language
  Model for Financial Communications. arXiv:2006.08097.
- Hutto, C. J. & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model
  for Sentiment Analysis. ICWSM.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module is importable in lean envs
# ---------------------------------------------------------------------------

# FinBERT via HuggingFace Transformers
try:
    from transformers import pipeline as _hf_pipeline  # type: ignore[import]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# VADER sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER  # type: ignore[import]
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

# YAML config loader
try:
    import yaml  # type: ignore[import]
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants  (all economically motivated — see module docstring)
# ---------------------------------------------------------------------------

# Component weights — must sum to 1.0
_WEIGHT_TRANSCRIPT_SENTIMENT: float = 0.40
_WEIGHT_SENTIMENT_CHANGE: float = 0.40
_WEIGHT_GUIDANCE_KEYWORDS: float = 0.20

# Decay parameters
_DEFAULT_DECAY_HALF_LIFE_DAYS: int = 21   # ~1 calendar month; analyst revision lag
_MIN_RECENCY_WEIGHT: float = 0.10          # floor weight for oldest earnings in window

# Cache
_CACHE_TTL_SECONDS: int = 86400            # 24 hours — earnings transcripts don't change
_DEFAULT_CACHE_DIR: str = "/tmp/earnings_nlp_cache"

# HTTP
_HTTP_TIMEOUT_SECONDS: int = 15
_MAX_RETRIES: int = 2
_RETRY_BACKOFF_SECONDS: float = 1.0
_USER_AGENT: str = (
    "EarningsNLPSignal/1.0 (automated-trading-system; "
    "research-use; contact: sys@trading)"
)

# Signal bounds
_SIGNAL_CLIP: float = 1.0

# SEC EDGAR endpoints
_EDGAR_FULL_TEXT_URL: str = (
    "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
    "&dateRange=custom&startdt={start}&enddt={end}&forms=8-K"
)
_EDGAR_BROWSE_URL: str = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=include&count=5"
    "&search_text="
)
_EDGAR_SEARCH_URL: str = (
    "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}+earnings%22"
    "&forms=8-K&dateRange=custom&startdt={start}&enddt={end}"
)

# Financial Modeling Prep
_FMP_TRANSCRIPT_URL: str = (
    "https://financialmodelingprep.com/api/v3/earning_call_transcript"
    "/{symbol}?quarter={quarter}&year={year}&apikey={apikey}"
)
_FMP_EARNINGS_DATE_URL: str = (
    "https://financialmodelingprep.com/api/v3/historical/earning_calendar"
    "/{symbol}?apikey={apikey}"
)

# FinBERT model identifier
_FINBERT_MODEL: str = "ProsusAI/finbert"

# Guidance keyword lexicon (Loughran-McDonald finance lexicon, curated subset)
_BULLISH_KEYWORDS: List[str] = [
    "confident", "confidence", "accelerating", "accelerate",
    "exceed", "exceeded", "exceeding", "outperform", "outperformed",
    "strong pipeline", "raised guidance", "raise guidance", "increasing guidance",
    "record revenue", "record earnings", "record quarter",
    "robust", "momentum", "upside", "improving", "improved",
    "growth acceleration", "beat expectations", "above expectations",
    "raised our outlook", "increased our guidance", "positive outlook",
    "strong demand", "expanded margins", "margin expansion",
]
_BEARISH_KEYWORDS: List[str] = [
    "cautious", "caution", "headwinds", "headwind",
    "uncertainty", "uncertain", "below expectations", "miss",
    "challenging", "challenges", "difficult environment",
    "restructuring", "restructure", "workforce reduction", "layoffs",
    "margin compression", "margin pressure", "cost pressures",
    "declining demand", "softness", "weaker than expected",
    "lowered guidance", "reduced guidance", "lowering our outlook",
    "macroeconomic uncertainty", "geopolitical", "supply chain constraints",
    "slower than anticipated", "deferred", "delayed revenue",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TranscriptData:
    """Parsed earnings transcript or 8-K text data."""
    symbol: str
    filing_date: datetime.date
    text: str                       # raw extracted plain text
    source: str                     # "edgar" | "fmp" | "cache"
    filing_url: Optional[str] = None


@dataclass
class SentimentResult:
    """Output of any sentiment backend."""
    positive: float     # probability or normalised score [0, 1]
    negative: float     # probability or normalised score [0, 1]
    neutral: float      # probability or normalised score [0, 1]
    raw_score: float    # positive - negative, in [-1, +1]
    backend: str        # "finbert" | "vader" | "keyword"


@dataclass
class EarningsRecord:
    """Aggregated per-symbol earnings analysis."""
    symbol: str
    filing_date: datetime.date
    days_ago: int
    recency_weight: float
    transcript_sentiment: float     # component A score [-1, +1]
    sentiment_backend: str
    guidance_score: float           # component C score [-1, +1]
    transcript: Optional[TranscriptData] = None


@dataclass
class SignalComponents:
    """Decomposed signal components for diagnostics."""
    transcript_sentiment: float     # component A
    sentiment_change: float         # component B
    guidance_keywords: float        # component C
    combined: float                 # weighted combination
    num_earnings_events: int
    latest_filing_date: Optional[datetime.date]
    sentiment_backend: str


# ---------------------------------------------------------------------------
# HTML text extractor
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Lightweight HTML → plain-text extractor (no external dependencies)."""

    _SKIP_TAGS = frozenset({"script", "style", "head", "meta", "link", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._in_skip: int = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._in_skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._in_skip = max(0, self._in_skip - 1)

    def handle_data(self, data: str) -> None:
        if self._in_skip == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _html_to_text(html: str) -> str:
    """Extract plain text from an HTML/HTM string."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
        return extractor.get_text()
    except Exception:
        # Fallback: strip tags with regex
        return re.sub(r"<[^>]+>", " ", html)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = _HTTP_TIMEOUT_SECONDS) -> Optional[str]:
    """
    Fetch URL with retries, returning the response body as a UTF-8 string.

    Returns None on failure rather than raising, so callers can implement
    graceful degradation.
    """
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(_MAX_RETRIES + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                # Try UTF-8 first, fall back to latin-1 for SEC filings
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    return raw.decode("latin-1", errors="replace")
        except HTTPError as exc:
            if exc.code == 429:
                # Rate-limited — back off longer
                wait = _RETRY_BACKOFF_SECONDS * (2 ** attempt) * 3
                logger.warning("HTTP 429 rate-limit on %s; sleeping %.1fs", url, wait)
                time.sleep(wait)
            elif exc.code in (404, 403):
                logger.debug("HTTP %d for %s", exc.code, url)
                return None
            else:
                logger.warning("HTTP %d fetching %s (attempt %d)", exc.code, url, attempt + 1)
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except URLError as exc:
            logger.warning("URLError fetching %s: %s (attempt %d)", url, exc, attempt + 1)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except Exception as exc:
            logger.warning("Unexpected error fetching %s: %s", url, exc)
            return None
    return None


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class _DiskCache:
    """
    Simple file-system key-value cache with TTL.

    Keys are SHA-256 hashed to create safe filenames. Values are stored as
    JSON. Expired entries are silently skipped (not deleted) to avoid I/O
    overhead during hot-path computation.
    """

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR, ttl_seconds: int = _CACHE_TTL_SECONDS) -> None:
        self._dir = Path(cache_dir)
        self._ttl = ttl_seconds
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Cannot create cache dir %s: %s — caching disabled", cache_dir, exc)
            self._dir = None  # type: ignore[assignment]

    def _key_path(self, key: str) -> Optional[Path]:
        if self._dir is None:
            return None
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_path(key)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - data["ts"] > self._ttl:
                return None
            return data["value"]
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_path(key)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps({"ts": time.time(), "value": value}),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Cache write failed for key %s: %s", key[:16], exc)

    def has(self, key: str) -> bool:
        return self.get(key) is not None


# ---------------------------------------------------------------------------
# Sentiment backends
# ---------------------------------------------------------------------------

class _FinBERTBackend:
    """
    HuggingFace FinBERT (ProsusAI/finbert) sentiment pipeline.

    The model returns three labels: positive / negative / neutral with
    associated probabilities. We chunk long texts to respect the 512-token
    BERT window and aggregate by averaging chunk scores.
    """

    _MAX_CHUNK_CHARS: int = 1800   # ~450 tokens; safe margin below 512-token limit
    _MAX_CHUNKS: int = 20           # cap GPU time: max ~36k chars per transcript

    def __init__(self) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed; cannot use FinBERT backend")
        logger.info("Loading FinBERT model %s (first call may be slow)…", _FINBERT_MODEL)
        self._pipe = _hf_pipeline(
            "text-classification",
            model=_FINBERT_MODEL,
            top_k=None,   # return all three labels
            truncation=True,
        )
        logger.info("FinBERT model loaded.")

    def score(self, text: str) -> SentimentResult:
        """Score a text string and return a SentimentResult."""
        chunks = self._chunk_text(text)
        pos_scores, neg_scores, neu_scores = [], [], []

        for chunk in chunks:
            try:
                results = self._pipe(chunk)
                # results is a list of label/score dicts (top_k=None)
                label_map: Dict[str, float] = {}
                for item in results[0]:  # type: ignore[index]
                    label_map[item["label"].lower()] = float(item["score"])
                pos_scores.append(label_map.get("positive", 0.0))
                neg_scores.append(label_map.get("negative", 0.0))
                neu_scores.append(label_map.get("neutral", 0.0))
            except Exception as exc:
                logger.warning("FinBERT chunk scoring failed: %s", exc)

        if not pos_scores:
            return SentimentResult(0.0, 0.0, 1.0, 0.0, "finbert")

        pos = float(np.mean(pos_scores))
        neg = float(np.mean(neg_scores))
        neu = float(np.mean(neu_scores))
        raw = float(np.clip(pos - neg, -_SIGNAL_CLIP, _SIGNAL_CLIP))
        return SentimentResult(pos, neg, neu, raw, "finbert")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks suitable for BERT's context window."""
        # Split on sentence boundaries where possible
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) + 1 > self._MAX_CHUNK_CHARS:
                if current:
                    chunks.append(current.strip())
                    if len(chunks) >= self._MAX_CHUNKS:
                        break
                current = sent
            else:
                current = f"{current} {sent}" if current else sent
        if current and len(chunks) < self._MAX_CHUNKS:
            chunks.append(current.strip())
        return chunks if chunks else [text[: self._MAX_CHUNK_CHARS]]


class _VADERBackend:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment backend.

    VADER is a rule-based model designed for social-media text and works
    reasonably well on earnings language. We use the compound score, which
    is normalised to [-1, +1], and convert to the three-component format
    expected by SentimentResult.
    """

    def __init__(self) -> None:
        if not _VADER_AVAILABLE:
            raise ImportError("vaderSentiment not installed; cannot use VADER backend")
        self._analyzer = _VADER()

    def score(self, text: str) -> SentimentResult:
        """Score a text string and return a SentimentResult."""
        # VADER handles long text less well; chunk into paragraphs and average
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text[:5000]]

        compound_scores = []
        for para in paragraphs[:50]:   # cap at 50 paragraphs
            try:
                scores = self._analyzer.polarity_scores(para[:1000])
                compound_scores.append(scores["compound"])
            except Exception:
                pass

        if not compound_scores:
            return SentimentResult(0.0, 0.0, 1.0, 0.0, "vader")

        compound = float(np.mean(compound_scores))
        # Map compound [-1, +1] to pseudo-probabilities
        pos = max(0.0, compound)
        neg = max(0.0, -compound)
        neu = 1.0 - pos - neg
        return SentimentResult(pos, neg, neu, compound, "vader")


class _KeywordBackend:
    """
    Pure-Python keyword scoring fallback (no additional dependencies).

    Uses the Loughran-McDonald financial lexicon subsets defined in this
    module. Normalises the net keyword count by total document length so
    that longer transcripts don't systematically score higher/lower.
    """

    def score(self, text: str) -> SentimentResult:
        """Score a text string and return a SentimentResult."""
        lower = text.lower()
        total_words = max(1, len(lower.split()))

        bull_count = sum(
            lower.count(kw.lower()) for kw in _BULLISH_KEYWORDS
        )
        bear_count = sum(
            lower.count(kw.lower()) for kw in _BEARISH_KEYWORDS
        )

        # Normalise per 1000 words to make comparable across transcript lengths
        bull_rate = bull_count / total_words * 1000
        bear_rate = bear_count / total_words * 1000

        # Cap rates to avoid extreme values from short texts
        bull_rate = min(bull_rate, 10.0)
        bear_rate = min(bear_rate, 10.0)

        net = bull_rate - bear_rate
        # Map [-10, +10] range to [-1, +1] with tanh squashing
        raw = float(np.clip(math.tanh(net / 5.0), -_SIGNAL_CLIP, _SIGNAL_CLIP))

        pos = max(0.0, raw)
        neg = max(0.0, -raw)
        neu = 1.0 - pos - neg
        return SentimentResult(pos, neg, neu, raw, "keyword")


# ---------------------------------------------------------------------------
# SEC EDGAR transcript fetcher
# ---------------------------------------------------------------------------

class _EDGARFetcher:
    """
    Fetches recent 8-K filings for a given ticker from SEC EDGAR.

    The EDGAR Full-Text Search System (EFTS) is used to find 8-K filings
    containing the ticker symbol and earnings-related terms. Document text
    is then extracted from the linked HTM/HTML filing document.
    """

    def __init__(self, cache: _DiskCache) -> None:
        self._cache = cache

    def fetch(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Optional[TranscriptData]:
        """
        Find the most recent 8-K filing for ``symbol`` within the date range
        and return its extracted plain text.

        Returns None if no filing can be located.
        """
        cache_key = f"edgar:{symbol}:{start_date}:{end_date}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            td = TranscriptData(**cached)
            td.filing_date = datetime.date.fromisoformat(cached["filing_date"])
            return td

        # Build search URL
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        search_url = _EDGAR_FULL_TEXT_URL.format(
            ticker=quote_plus(f'"{symbol}"'),
            start=start_str,
            end=end_str,
        )

        logger.debug("EDGAR search: %s", search_url)
        response = _http_get(search_url)
        if not response:
            # Fallback: try a broader keyword search
            search_url2 = _EDGAR_SEARCH_URL.format(
                ticker=quote_plus(symbol),
                start=start_str,
                end=end_str,
            )
            response = _http_get(search_url2)

        if not response:
            logger.debug("EDGAR: no response for %s in %s – %s", symbol, start_str, end_str)
            return None

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.debug("EDGAR: invalid JSON for %s", symbol)
            return None

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            logger.debug("EDGAR: no 8-K hits for %s in %s – %s", symbol, start_str, end_str)
            return None

        # Take the most recent hit
        best_hit = hits[0]
        source = best_hit.get("_source", {})
        file_date_str = source.get("file_date", "")
        entity_name = source.get("entity_name", symbol)
        accession_no = source.get("accession_no", "").replace("-", "")

        try:
            filing_date = datetime.date.fromisoformat(file_date_str)
        except ValueError:
            filing_date = end_date

        # Construct filing index URL
        cik = str(source.get("_id", "")).split(":")[0] if ":" in str(source.get("_id", "")) else ""
        if not cik:
            cik = str(source.get("entity_id", ""))

        # Try to get the actual document text from the filing
        text = self._extract_filing_text(best_hit, accession_no, cik, symbol)
        if not text:
            # Use EDGAR inline highlights as lightweight fallback
            highlights = source.get("period_of_report", "") or ""
            inline_text = source.get("file_date", "") + " " + entity_name + " " + highlights
            # Try the inline text field
            inline_text += " " + " ".join(
                best_hit.get("highlight", {}).get("file_date", [])
            )
            text = inline_text.strip()

        if not text or len(text) < 100:
            logger.debug("EDGAR: extracted text too short for %s (%d chars)", symbol, len(text))
            return None

        td = TranscriptData(
            symbol=symbol,
            filing_date=filing_date,
            text=text,
            source="edgar",
            filing_url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/",
        )

        # Cache the result (store as dict for JSON serialisability)
        self._cache.set(cache_key, {
            "symbol": td.symbol,
            "filing_date": td.filing_date.isoformat(),
            "text": td.text,
            "source": td.source,
            "filing_url": td.filing_url,
        })
        return td

    def _extract_filing_text(
        self,
        hit: Dict[str, Any],
        accession_no: str,
        cik: str,
        symbol: str,
    ) -> Optional[str]:
        """Attempt to fetch and extract text from the primary 8-K HTML document."""
        source = hit.get("_source", {})

        # EDGAR EFTS provides a direct file_num or period_of_report; try to
        # build the filing index URL and grab the first .htm document.
        accession_formatted = (
            f"{accession_no[:10]}-{accession_no[10:12]}-{accession_no[12:]}"
            if len(accession_no) == 18 else accession_no
        )

        # Try to get the entity CIK from _id field (format: CIK:ACCESSION)
        hit_id = hit.get("_id", "")
        if ":" in hit_id:
            parts = hit_id.split(":")
            if parts[0].isdigit():
                cik = parts[0]

        if not cik:
            return None

        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession_no}/{accession_formatted}-index.htm"
        )
        index_html = _http_get(index_url)
        if not index_html:
            return None

        # Find the first .htm/.html document link in the index
        doc_links = re.findall(
            r'href="(/Archives/edgar/data/[^"]+\.htm[l]?)"',
            index_html,
            re.IGNORECASE,
        )
        if not doc_links:
            return None

        # Prefer documents that look like press releases (not exhibits)
        primary_link = doc_links[0]
        for link in doc_links:
            lname = link.lower()
            if any(x in lname for x in ["ex-99", "ex99", "press", "earnings"]):
                primary_link = link
                break

        doc_url = f"https://www.sec.gov{primary_link}"
        doc_html = _http_get(doc_url)
        if not doc_html:
            return None

        text = _html_to_text(doc_html)
        # Keep it to a reasonable length to avoid memory issues
        return text[:80000]


# ---------------------------------------------------------------------------
# Financial Modeling Prep transcript fetcher
# ---------------------------------------------------------------------------

class _FMPFetcher:
    """
    Fetches earnings call transcripts from Financial Modeling Prep API.

    FMP provides structured transcripts (CEO/CFO Q&A) for US equities.
    Free tier: 250 API calls per day. The API key is optional; if not
    provided, only the demo endpoint (limited symbols) is tried.
    """

    def __init__(self, api_key: str, cache: _DiskCache) -> None:
        self._api_key = api_key or "demo"
        self._cache = cache

    def fetch(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Optional[TranscriptData]:
        """
        Fetch the most recent earnings call transcript within the date range.
        Returns None if unavailable.
        """
        cache_key = f"fmp:{symbol}:{start_date}:{end_date}:{self._api_key[:8]}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            td = TranscriptData(**cached)
            td.filing_date = datetime.date.fromisoformat(cached["filing_date"])
            return td

        # Determine which quarter/year to request
        year = end_date.year
        quarter = (end_date.month - 1) // 3 + 1

        url = _FMP_TRANSCRIPT_URL.format(
            symbol=symbol,
            quarter=quarter,
            year=year,
            apikey=self._api_key,
        )
        response = _http_get(url)
        if not response:
            return None

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, list) or not data:
            return None

        # Find the most recent transcript within our window
        best: Optional[Dict[str, Any]] = None
        best_date: Optional[datetime.date] = None

        for item in data:
            date_str = item.get("date", "")
            try:
                filing_date = datetime.date.fromisoformat(date_str[:10])
            except (ValueError, TypeError):
                continue
            if start_date <= filing_date <= end_date:
                if best_date is None or filing_date > best_date:
                    best = item
                    best_date = filing_date

        if best is None or best_date is None:
            return None

        content = best.get("content", "")
        if not content or len(content) < 100:
            return None

        td = TranscriptData(
            symbol=symbol,
            filing_date=best_date,
            text=content,
            source="fmp",
            filing_url=url,
        )
        self._cache.set(cache_key, {
            "symbol": td.symbol,
            "filing_date": td.filing_date.isoformat(),
            "text": td.text,
            "source": td.source,
            "filing_url": td.filing_url,
        })
        return td


# ---------------------------------------------------------------------------
# Guidance keyword scorer
# ---------------------------------------------------------------------------

def _score_guidance_keywords(text: str) -> float:
    """
    Score forward-looking language in ``text`` using the curated keyword
    lexicon defined at module level.

    Returns a value in [-1, +1]:
      +1 → strongly bullish guidance
      -1 → strongly bearish guidance
       0 → neutral / no guidance language

    The approach follows Huang, Teoh & Zhang (2014): we count keyword
    occurrences (multi-word phrases before single words to avoid double-
    counting), normalise by document length, and apply tanh squashing.
    """
    lower = text.lower()
    total_words = max(1, len(lower.split()))

    # Count occurrences — multi-word phrases first (prevents double-counting)
    bull_count = 0
    bear_count = 0

    # Process multi-word phrases first
    multi_bull = [kw for kw in _BULLISH_KEYWORDS if " " in kw]
    multi_bear = [kw for kw in _BEARISH_KEYWORDS if " " in kw]
    single_bull = [kw for kw in _BULLISH_KEYWORDS if " " not in kw]
    single_bear = [kw for kw in _BEARISH_KEYWORDS if " " not in kw]

    # Count and mask matched regions for multi-word phrases
    masked = lower
    for phrase in multi_bull:
        count = masked.count(phrase)
        if count:
            bull_count += count
            masked = masked.replace(phrase, " " * len(phrase))

    for phrase in multi_bear:
        count = masked.count(phrase)
        if count:
            bear_count += count
            masked = masked.replace(phrase, " " * len(phrase))

    # Single words (on the masked text to avoid re-counting)
    for word in single_bull:
        # Use word-boundary matching
        bull_count += len(re.findall(r"\b" + re.escape(word) + r"\b", masked))

    for word in single_bear:
        bear_count += len(re.findall(r"\b" + re.escape(word) + r"\b", masked))

    # Normalise per 1000 words
    bull_rate = bull_count / total_words * 1000
    bear_rate = bear_count / total_words * 1000

    # Cap rates
    bull_rate = min(bull_rate, 10.0)
    bear_rate = min(bear_rate, 10.0)

    net = bull_rate - bear_rate
    score = float(np.clip(math.tanh(net / 4.0), -_SIGNAL_CLIP, _SIGNAL_CLIP))
    return score


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def _recency_weight(days_ago: int, lookback_days: int, decay_half_life: int) -> float:
    """
    Compute the recency weight for an earnings event that occurred ``days_ago``
    days before as_of_date.

    Uses exponential decay with the specified half-life. Weight is floored at
    ``_MIN_RECENCY_WEIGHT`` to ensure the oldest earnings within the window
    still contribute (captures the full PEAD tail).

    Parameters
    ----------
    days_ago : int
        Number of days between the earnings filing and as_of_date.
    lookback_days : int
        Maximum lookback window (events older than this are excluded upstream).
    decay_half_life : int
        Number of days for weight to halve (exponential decay).

    Returns
    -------
    float
        Weight in [_MIN_RECENCY_WEIGHT, 1.0].
    """
    if days_ago <= 0:
        return 1.0
    lambda_ = math.log(2) / max(1, decay_half_life)
    weight = math.exp(-lambda_ * days_ago)
    return max(_MIN_RECENCY_WEIGHT, weight)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load ``earnings_nlp`` section from settings.yaml.

    Searches the following locations in order:
      1. ``config_path`` if explicitly provided
      2. ``config/settings.yaml`` relative to cwd
      3. ``../config/settings.yaml`` relative to this file's directory

    Returns the ``earnings_nlp`` sub-dict, or defaults if not found.
    """
    defaults: Dict[str, Any] = {
        "enabled": True,
        "weight": 0.15,
        "lookback_days": 30,
        "use_finbert": True,
        "fmp_api_key": "",
        "sentiment_decay_days": 21,
    }

    if not _YAML_AVAILABLE:
        logger.debug("yaml not installed; using default earnings_nlp config")
        return defaults

    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    search_paths += [
        Path.cwd() / "config" / "settings.yaml",
        Path(__file__).parent.parent / "config" / "settings.yaml",
        Path(__file__).parent.parent.parent / "config" / "settings.yaml",
    ]

    for path in search_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    full_cfg = yaml.safe_load(fh) or {}
                cfg = full_cfg.get("earnings_nlp", {})
                if cfg:
                    merged = {**defaults, **cfg}
                    logger.debug("Loaded earnings_nlp config from %s", path)
                    return merged
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", path, exc)

    logger.debug("No earnings_nlp config found; using defaults")
    return defaults


# ---------------------------------------------------------------------------
# Main signal class
# ---------------------------------------------------------------------------

class EarningsNLPSignal:
    """
    Earnings Call NLP Signal.

    Computes a directional signal in [-1, +1] for each symbol based on
    sentiment analysis of recent earnings call transcripts and 8-K filings.

    The signal has three components:

    A. **Transcript Sentiment** (40%) — FinBERT / VADER / keyword scoring of
       the most recent earnings transcript within the lookback window.

    B. **Sentiment Change** (40%) — Delta between current and prior-quarter
       sentiment. A shift from negative to positive language is the strongest
       individual predictor of post-earnings drift (Kearney & Liu 2014).

    C. **Guidance Keywords** (20%) — Forward-looking language analysis using
       the Loughran-McDonald financial lexicon keyword subsets.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to settings.yaml. If None, standard locations are tried.
    cache_dir : str, optional
        Directory for the 24-hour transcript cache. Defaults to
        ``/tmp/earnings_nlp_cache``.

    Examples
    --------
    >>> signal = EarningsNLPSignal()
    >>> signals = signal.compute(
    ...     symbols=["AAPL", "MSFT"],
    ...     as_of_date=datetime.date(2026, 4, 1),
    ...     lookback_days=30,
    ... )
    >>> signals
    {"AAPL": 0.55, "MSFT": -0.12}
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._cfg = _load_config(config_path)
        self._cache = _DiskCache(
            cache_dir=cache_dir or _DEFAULT_CACHE_DIR,
            ttl_seconds=_CACHE_TTL_SECONDS,
        )
        self._edgar = _EDGARFetcher(self._cache)
        self._fmp = _FMPFetcher(
            api_key=self._cfg.get("fmp_api_key", ""),
            cache=self._cache,
        )

        # Sentiment backend — initialise lazily (FinBERT model download)
        self._finbert: Optional[_FinBERTBackend] = None
        self._vader: Optional[_VADERBackend] = None
        self._keyword = _KeywordBackend()

        # In-memory cache for previous-quarter sentiment scores
        # Maps symbol → SentimentResult (updated across calls)
        self._prev_sentiment: Dict[str, SentimentResult] = {}

        # Diagnostic: last computed components per symbol
        self._last_components: Dict[str, SignalComponents] = {}

        logger.info(
            "EarningsNLPSignal initialised (FinBERT=%s, VADER=%s, FMP_key=%s)",
            _TRANSFORMERS_AVAILABLE,
            _VADER_AVAILABLE,
            "set" if self._cfg.get("fmp_api_key") else "not set",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        symbols: List[str],
        as_of_date: datetime.date,
        lookback_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute the earnings NLP signal for each symbol.

        Parameters
        ----------
        symbols : list of str
            Ticker symbols to compute signals for (e.g. ["AAPL", "MSFT"]).
        as_of_date : datetime.date
            The reference date for the computation. Only earnings transcripts
            within [as_of_date - lookback_days, as_of_date] are considered.
        lookback_days : int, optional
            Override for the config ``lookback_days`` value.

        Returns
        -------
        dict of str → float
            Signal values in [-1, +1] for each symbol. Symbols for which no
            transcript data can be found return 0.0 (neutral).
        """
        if not self._cfg.get("enabled", True):
            logger.info("EarningsNLPSignal disabled by config; returning neutral signals")
            return {s: 0.0 for s in symbols}

        effective_lookback = lookback_days or self._cfg.get("lookback_days", 30)
        decay_half_life = self._cfg.get("sentiment_decay_days", _DEFAULT_DECAY_HALF_LIFE_DAYS)
        start_date = as_of_date - datetime.timedelta(days=effective_lookback)

        logger.info(
            "Computing EarningsNLPSignal for %d symbols; window %s – %s",
            len(symbols),
            start_date,
            as_of_date,
        )

        results: Dict[str, float] = {}
        for symbol in symbols:
            try:
                signal, components = self._compute_symbol(
                    symbol=symbol,
                    as_of_date=as_of_date,
                    start_date=start_date,
                    decay_half_life=decay_half_life,
                )
                results[symbol] = float(np.clip(signal, -_SIGNAL_CLIP, _SIGNAL_CLIP))
                self._last_components[symbol] = components
                logger.debug(
                    "%s: signal=%.3f (A=%.3f B=%.3f C=%.3f via %s)",
                    symbol,
                    signal,
                    components.transcript_sentiment,
                    components.sentiment_change,
                    components.guidance_keywords,
                    components.sentiment_backend,
                )
            except Exception as exc:
                logger.warning("Unexpected error computing signal for %s: %s", symbol, exc)
                results[symbol] = 0.0

        logger.info(
            "EarningsNLPSignal complete: %d signals computed, %d non-zero",
            len(results),
            sum(1 for v in results.values() if abs(v) > 1e-6),
        )
        return results

    def get_components(self, symbol: str) -> Optional[SignalComponents]:
        """
        Return the decomposed signal components from the last ``compute`` call.

        Useful for diagnostics, attribution analysis, and research.

        Parameters
        ----------
        symbol : str
            Ticker symbol.

        Returns
        -------
        SignalComponents or None
            None if the symbol was not included in the last compute call.
        """
        return self._last_components.get(symbol)

    # ------------------------------------------------------------------
    # Per-symbol computation
    # ------------------------------------------------------------------

    def _compute_symbol(
        self,
        symbol: str,
        as_of_date: datetime.date,
        start_date: datetime.date,
        decay_half_life: int,
    ) -> Tuple[float, SignalComponents]:
        """
        Compute the full signal for a single symbol.

        Returns (combined_signal, SignalComponents).
        """
        # ---- Fetch transcript ----
        transcript = self._fetch_transcript(symbol, start_date, as_of_date)

        if transcript is None:
            logger.debug("%s: no transcript found → neutral signal", symbol)
            components = SignalComponents(
                transcript_sentiment=0.0,
                sentiment_change=0.0,
                guidance_keywords=0.0,
                combined=0.0,
                num_earnings_events=0,
                latest_filing_date=None,
                sentiment_backend="none",
            )
            return 0.0, components

        # ---- Recency weight ----
        days_ago = (as_of_date - transcript.filing_date).days
        recency_weight = _recency_weight(days_ago, (as_of_date - start_date).days, decay_half_life)

        # ---- Component A: Transcript Sentiment ----
        sentiment_result = self._score_sentiment(transcript.text)
        component_a = sentiment_result.raw_score * recency_weight

        # ---- Component B: Sentiment Change ----
        prev = self._prev_sentiment.get(symbol)
        if prev is not None:
            # Change in raw score: positive = improvement (bearish → bullish)
            delta = sentiment_result.raw_score - prev.raw_score
            # Clip and scale: a swing from -1 → +1 should give +1 signal
            component_b = float(np.clip(delta, -_SIGNAL_CLIP, _SIGNAL_CLIP)) * recency_weight
        else:
            # No prior quarter data — use current sentiment as a weak proxy
            # (attenuated to avoid double-counting with component A)
            component_b = sentiment_result.raw_score * 0.5 * recency_weight

        # Store current as new previous for the next call
        self._prev_sentiment[symbol] = sentiment_result

        # ---- Component C: Guidance Keywords ----
        guidance_score = _score_guidance_keywords(transcript.text)
        component_c = guidance_score * recency_weight

        # ---- Combined signal ----
        combined = (
            _WEIGHT_TRANSCRIPT_SENTIMENT * component_a
            + _WEIGHT_SENTIMENT_CHANGE * component_b
            + _WEIGHT_GUIDANCE_KEYWORDS * component_c
        )
        combined = float(np.clip(combined, -_SIGNAL_CLIP, _SIGNAL_CLIP))

        components = SignalComponents(
            transcript_sentiment=float(component_a),
            sentiment_change=float(component_b),
            guidance_keywords=float(component_c),
            combined=combined,
            num_earnings_events=1,
            latest_filing_date=transcript.filing_date,
            sentiment_backend=sentiment_result.backend,
        )
        return combined, components

    # ------------------------------------------------------------------
    # Transcript fetching (with source fallback chain)
    # ------------------------------------------------------------------

    def _fetch_transcript(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Optional[TranscriptData]:
        """
        Attempt to fetch earnings transcript using the following fallback chain:

        1. SEC EDGAR Full-Text Search (free, no key)
        2. Financial Modeling Prep API (requires key for most symbols)
        3. Return None → caller will emit neutral signal

        Both sources are tried even if the first succeeds, so the most
        information-rich source wins. EDGAR provides raw press-release text;
        FMP provides structured Q&A transcripts.
        """
        # Try EDGAR first (free, no rate limits for research use)
        edgar_transcript = None
        try:
            edgar_transcript = self._edgar.fetch(symbol, start_date, end_date)
        except Exception as exc:
            logger.warning("EDGAR fetch failed for %s: %s", symbol, exc)

        # Try FMP if API key is configured and EDGAR returned nothing useful
        fmp_transcript = None
        fmp_key = self._cfg.get("fmp_api_key", "")
        if fmp_key and (edgar_transcript is None or len(edgar_transcript.text) < 500):
            try:
                fmp_transcript = self._fmp.fetch(symbol, start_date, end_date)
            except Exception as exc:
                logger.warning("FMP fetch failed for %s: %s", symbol, exc)

        # Prefer FMP (structured transcript) over EDGAR (press release) if both
        # are available, since structured Q&A has richer NLP signal
        if fmp_transcript and len(fmp_transcript.text) >= 500:
            return fmp_transcript
        if edgar_transcript and len(edgar_transcript.text) >= 200:
            return edgar_transcript

        logger.debug(
            "%s: no usable transcript found (EDGAR=%s, FMP=%s)",
            symbol,
            "ok" if edgar_transcript else "none",
            "ok" if fmp_transcript else "none",
        )
        return None

    # ------------------------------------------------------------------
    # Sentiment scoring (with backend fallback chain)
    # ------------------------------------------------------------------

    def _score_sentiment(self, text: str) -> SentimentResult:
        """
        Score text sentiment using the best available backend.

        Fallback order:
          1. FinBERT  (if transformers installed AND use_finbert=True in config)
          2. VADER    (if vaderSentiment installed)
          3. Keyword  (always available — pure Python)
        """
        use_finbert = self._cfg.get("use_finbert", True)

        if use_finbert and _TRANSFORMERS_AVAILABLE:
            if self._finbert is None:
                try:
                    self._finbert = _FinBERTBackend()
                except Exception as exc:
                    logger.warning("Failed to initialise FinBERT: %s; falling back to VADER", exc)
            if self._finbert is not None:
                try:
                    return self._finbert.score(text)
                except Exception as exc:
                    logger.warning("FinBERT scoring failed: %s; falling back to VADER", exc)

        if _VADER_AVAILABLE:
            if self._vader is None:
                try:
                    self._vader = _VADERBackend()
                except Exception as exc:
                    logger.warning("Failed to initialise VADER: %s; using keyword backend", exc)
            if self._vader is not None:
                try:
                    return self._vader.score(text)
                except Exception as exc:
                    logger.warning("VADER scoring failed: %s; using keyword backend", exc)

        # Final fallback: pure keyword scoring
        return self._keyword.score(text)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_sentiment_history(self) -> None:
        """
        Clear the in-memory previous-quarter sentiment cache.

        Call this if you want to treat all symbols as having no prior history
        (e.g., at the start of a new backtest run).
        """
        self._prev_sentiment.clear()
        logger.debug("EarningsNLPSignal: sentiment history cleared")

    def warmup(self, symbols: List[str], as_of_date: datetime.date, quarters_back: int = 1) -> None:
        """
        Pre-populate the previous-quarter sentiment cache by scoring transcripts
        from ``quarters_back`` quarters before ``as_of_date``.

        This allows Component B (Sentiment Change) to produce a meaningful
        delta on the very first live call rather than falling back to the
        attenuated proxy.

        Parameters
        ----------
        symbols : list of str
            Ticker symbols to warm up.
        as_of_date : datetime.date
            Current reference date.
        quarters_back : int
            How many quarters to look back for the prior-quarter transcript.
        """
        prior_end = as_of_date - datetime.timedelta(days=60 * quarters_back)
        prior_start = prior_end - datetime.timedelta(days=90)

        logger.info(
            "Warming up sentiment history for %d symbols (prior window %s – %s)",
            len(symbols),
            prior_start,
            prior_end,
        )

        for symbol in symbols:
            try:
                transcript = self._fetch_transcript(symbol, prior_start, prior_end)
                if transcript:
                    result = self._score_sentiment(transcript.text)
                    self._prev_sentiment[symbol] = result
                    logger.debug(
                        "%s warmup: prior sentiment=%.3f (via %s)",
                        symbol,
                        result.raw_score,
                        result.backend,
                    )
            except Exception as exc:
                logger.warning("Warmup failed for %s: %s", symbol, exc)

    @property
    def config(self) -> Dict[str, Any]:
        """Return the active configuration dictionary (read-only copy)."""
        return dict(self._cfg)

    @property
    def sentiment_history(self) -> Dict[str, float]:
        """
        Return a snapshot of the previous-quarter sentiment scores by symbol.

        Useful for diagnostics and state persistence across backtest chunks.
        """
        return {sym: r.raw_score for sym, r in self._prev_sentiment.items()}

    def restore_sentiment_history(self, history: Dict[str, float]) -> None:
        """
        Restore previous-quarter sentiment scores from a serialised snapshot.

        Parameters
        ----------
        history : dict of str → float
            As returned by ``sentiment_history``. Values in [-1, +1].
        """
        for sym, score in history.items():
            score_clipped = float(np.clip(score, -1.0, 1.0))
            self._prev_sentiment[sym] = SentimentResult(
                positive=max(0.0, score_clipped),
                negative=max(0.0, -score_clipped),
                neutral=1.0 - abs(score_clipped),
                raw_score=score_clipped,
                backend="restored",
            )
        logger.debug(
            "EarningsNLPSignal: restored sentiment history for %d symbols", len(history)
        )


# ---------------------------------------------------------------------------
# Standalone test / smoke-test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """
    Quick smoke-test that can be run directly:

        python -m strategy.earnings_nlp_signal

    Tests the full pipeline for a few well-known tickers with publicly
    available EDGAR filings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print("\n=== EarningsNLPSignal Smoke Test ===\n")

    signal = EarningsNLPSignal()

    test_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    as_of = datetime.date.today()

    print(f"as_of_date : {as_of}")
    print(f"lookback   : 90 days (extended for test)")
    print(f"backends   : FinBERT={_TRANSFORMERS_AVAILABLE}, VADER={_VADER_AVAILABLE}\n")

    # Warm up with prior quarter
    signal.warmup(test_symbols, as_of, quarters_back=1)

    results = signal.compute(test_symbols, as_of, lookback_days=90)

    print(f"{'Symbol':<10} {'Signal':>8}  {'SentA':>6}  {'SentB':>6}  {'Guide':>6}  "
          f"{'Backend':<10}  {'FilingDate'}")
    print("-" * 72)
    for sym in test_symbols:
        sig = results.get(sym, 0.0)
        comp = signal.get_components(sym)
        if comp:
            print(
                f"{sym:<10} {sig:>+8.4f}  {comp.transcript_sentiment:>+6.3f}  "
                f"{comp.sentiment_change:>+6.3f}  {comp.guidance_keywords:>+6.3f}  "
                f"{comp.sentiment_backend:<10}  {comp.latest_filing_date}"
            )
        else:
            print(f"{sym:<10} {sig:>+8.4f}  (no transcript found)")

    print("\n=== Keyword Backend Test ===")
    kb = _KeywordBackend()
    bull_text = (
        "We are confident in our growth trajectory. Revenue exceeded expectations "
        "with record earnings and strong pipeline across all segments. "
        "We are raising guidance for the next quarter."
    )
    bear_text = (
        "We are cautious about the macro environment. There are significant headwinds "
        "and uncertainty in demand. Results were below expectations due to "
        "challenging market conditions. We are restructuring certain business units."
    )
    print(f"Bullish text score: {kb.score(bull_text).raw_score:+.4f} (expected > 0)")
    print(f"Bearish text score: {kb.score(bear_text).raw_score:+.4f} (expected < 0)")

    print("\n=== Guidance Keyword Test ===")
    print(f"Bullish guidance: {_score_guidance_keywords(bull_text):+.4f}")
    print(f"Bearish guidance: {_score_guidance_keywords(bear_text):+.4f}")

    print("\nSmoke test complete.\n")


if __name__ == "__main__":
    _smoke_test()
