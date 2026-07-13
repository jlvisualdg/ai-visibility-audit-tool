"""
AI engine visibility check.

The pattern here is "mock-first, wire one real engine end-to-end, then
add the rest" — directly from the aeo-tracking skill. Why:

1. Mock engines let the entire pipeline run without any API key. You can
   see a real report before spending a dollar.
2. Wiring 4 engines at once means debugging 4 different JSON shapes at
   once. Wire one, ship, then add the next.
3. Mock engines let tests assert exact output. Real engines can't.

v1: Perplexity via OpenRouter is real (cheap, search-grounded, citations
in a known annotation format). The other three are mocks with the
protocol in place — drop in a real client when you have a key.

Engine protocol (informal):
    class Engine:
        name: str
        def query(self, topic: str) -> EngineResult: ...
        def is_real(self) -> bool: ...

EngineResult:
    text: str             # raw response text
    citations: list[str]  # extracted cited URLs
    model: str
    latency_ms: int
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
PERPLEXITY_MODEL = os.environ.get("PERPLEXITY_MODEL", "perplexity/sonar-pro").strip()
PASSES_PER_QUERY = int(os.environ.get("PASSES_PER_QUERY", "3"))

# The prompt template for buyer-intent topic queries. Engineered to elicit
# vendor names + brief descriptions + URLs. The phrase "List them with
# specific company names and domains" pushes models toward citing.
DEFAULT_PROMPT_TEMPLATE = (
    "I'm researching {topic}. "
    "What are the top vendors, platforms, or services a buyer would consider? "
    "List them with brief pros/cons. Include specific company names and domains."
)

# How to detect a domain in free text (fallback for engines that don't
# surface citations). Conservative — looks for "domain.tld" patterns.
DOMAIN_RE = re.compile(
    r"\b((?:[a-z0-9-]+\.)+[a-z]{2,})\b",
    re.IGNORECASE,
)

# Common URL-prefixes to strip when extracting domains
URL_PREFIXES = (
    "https://", "http://", "www.",
)

# Domains that aren't real "competitor" mentions (search engine homepages,
# documentation, social media, etc.) — kept out of the citation matrix
# when generating the competitor list, but still counted in raw citations.
NOISE_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "twitter.com", "x.com",
    "wikipedia.org", "linkedin.com", "reddit.com", "amazon.com",
    "apple.com", "microsoft.com", "github.com",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EngineResult:
    """The output of one engine call for one topic."""
    text: str
    citations: list[str]           # cited domains (e.g. "bhsq.com")
    model: str = ""
    latency_ms: int = 0
    error: Optional[str] = None
    # v1.2 — brand visibility tracking
    brand_mentioned: bool = False
    brand_mention_count: int = 0
    first_mention_section: Optional[int] = None    # 1-indexed section number
    url_citation_count: int = 0                    # how many times target URL appears
    first_competitor: Optional[str] = None          # top brand mentioned before target
    first_competitor_url: Optional[str] = None
    competitors: list[dict] = field(default_factory=list)  # [{name, url}, ...]


@dataclass
class TopicResult:
    """The aggregated result for one topic across one engine (one or more passes)."""
    topic: str
    engine: str
    passes: list[EngineResult] = field(default_factory=list)
    cited_sources: list[str] = field(default_factory=list)   # deduped union
    covered: bool = False                                    # target domain present
    pass_count: int = 0
    # v1.2 — aggregates
    best_brand_mentions: int = 0
    best_url_citations: int = 0
    best_first_section: Optional[int] = None
    top_competitor: Optional[str] = None

    @property
    def coverage_rate(self) -> float:
        """Fraction of passes where the target domain appeared."""
        if not self.passes:
            return 0.0
        return sum(1 for p in self.passes if _target_in(p.citations)) / len(self.passes)


@dataclass
class CitationMatrix:
    """The full topic x engine matrix for one audit."""
    domain: str
    topics: list[str] = field(default_factory=list)
    engines: list[str] = field(default_factory=list)
    results: list[TopicResult] = field(default_factory=list)
    all_competitors: dict = field(default_factory=dict)  # domain -> count

    @property
    def total_cells(self) -> int:
        return len(self.topics) * len(self.engines)

    @property
    def covered_cells(self) -> int:
        return sum(1 for r in self.results if r.covered)

    @property
    def coverage_pct(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return round(100.0 * self.covered_cells / self.total_cells, 1)

    def result_for(self, topic: str, engine: str) -> Optional[TopicResult]:
        for r in self.results:
            if r.topic == topic and r.engine == engine:
                return r
        return None


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


class Engine(Protocol):
    name: str
    def query(self, topic: str) -> EngineResult: ...
    def is_real(self) -> bool: ...


# ---------------------------------------------------------------------------
# Perplexity via OpenRouter (real)
# ---------------------------------------------------------------------------


class PerplexityEngine:
    """
    Perplexity Sonar via OpenRouter. Uses the OpenAI-compatible chat
    completions endpoint.

    Citation field shape (the gotcha — documented in the aeo-tracking skill):
        data["choices"][0]["message"]["annotations"][i]["url_citation"]["url"]

    NOT data["citations"] — that key is absent for Perplexity via
    OpenRouter. This is OpenAI-style annotation format.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, target_domain: str = ""):
        self.api_key = (api_key or OPENROUTER_API_KEY).strip()
        self.model = (model or PERPLEXITY_MODEL).strip()
        self.target_domain = target_domain
        self.brand_name = target_domain.split(".")[0].replace("-", " ").title() if target_domain else ""
        if not self.api_key:
            raise ValueError(
                "PerplexityEngine needs OPENROUTER_API_KEY. "
                "Set it in .env or pass api_key=..."
            )

    @property
    def name(self) -> str:
        return "perplexity"

    def is_real(self) -> bool:
        return True

    def query(self, topic: str) -> EngineResult:
        prompt = DEFAULT_PROMPT_TEMPLATE.format(topic=topic)
        started = time.time()
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    # Recommended by OpenRouter for ranking
                    "HTTP-Referer": "https://boringmarketer.com",
                    "X-Title": "AEO Audit Tool",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            return EngineResult(
                text="",
                citations=[],
                model=self.model,
                latency_ms=_elapsed_ms(started),
                error=f"HTTP error: {e}",
            )
        except ValueError as e:
            return EngineResult(
                text="",
                citations=[],
                model=self.model,
                latency_ms=_elapsed_ms(started),
                error=f"JSON decode error: {e}",
            )

        # Extract content + annotations
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content", "") or ""
            annotations = message.get("annotations") or []
        except (KeyError, IndexError, TypeError) as e:
            return EngineResult(
                text="",
                citations=[],
                model=self.model,
                latency_ms=_elapsed_ms(started),
                error=f"Unexpected response shape: {e}",
            )

        # Citations live in annotations[i].url_citation.url
        citations = []
        for ann in annotations:
            url_cite = ann.get("url_citation") or {}
            url = url_cite.get("url")
            if url:
                domain = _domain_from_url(url)
                if domain:
                    citations.append(domain)

        # Also try the cheaper fallback: regex the text for "domain.tld" mentions
        # for engines that don't surface annotations reliably.
        if not citations:
            citations = list(dict.fromkeys(
                d for d in DOMAIN_RE.findall(text)
                if not _is_noise_domain(d)
            ))

        return EngineResult(
            text=text,
            citations=citations,
            model=self.model,
            latency_ms=_elapsed_ms(started),
            brand_mentioned=self.target_domain.lower() in text.lower() if self.target_domain else False,
            brand_mention_count=text.lower().count(self.target_domain.lower()) if self.target_domain else 0,
            url_citation_count=sum(1 for c in citations if self.target_domain in c) if self.target_domain else 0,
            first_mention_section=_find_first_mention_section(text, self.target_domain, self.brand_name),
            first_competitor=(_find_first_competitor(text, self.brand_name) or {}).get("name"),
            first_competitor_url=(_find_first_competitor(text, self.brand_name) or {}).get("url"),
            competitors=_find_all_competitors(text, self.brand_name) if self.brand_name else [],
        )


# ---------------------------------------------------------------------------
# Mock engines (deterministic, no network)
# ---------------------------------------------------------------------------


class MockEngine:
    """
    A deterministic mock engine. Returns plausible citations + text so
    the entire pipeline works without API keys.

    Set `mock_topic_responses` to a {topic: EngineResult} dict to inject
    test data. Otherwise returns generic but realistic-looking output.
    """

    def __init__(self, slot: str, mock_topic_responses: Optional[dict] = None):
        # e.g. "chatgpt (mock)" — the (mock) suffix is critical so the
        # report never accidentally shows mock output as a real result.
        self._slot = slot
        self._name = f"{slot} (mock)"
        self._mock_topic_responses = mock_topic_responses or {}

    @property
    def name(self) -> str:
        return self._name

    def is_real(self) -> bool:
        return False

    def query(self, topic: str) -> EngineResult:
        if topic in self._mock_topic_responses:
            return self._mock_topic_responses[topic]
        return _default_mock_response(self._slot, topic)


def _default_mock_response(slot: str, topic: str) -> EngineResult:
    """
    Default mock response for a slot + topic. Returns a plausible
    EngineResult so the citation matrix and reporter work end-to-end.
    """
    # Slot-specific plausible "top vendors" — these are invented. The
    # point is to exercise the matrix and report, not to give real data.
    slot_vendors = {
        "chatgpt": [
            "vendor-alpha.com", "vendor-beta.io", "vendor-gamma.co",
            "vendor-delta.net", "vendor-epsilon.com",
        ],
        "claude": [
            "platform-one.com", "platform-two.io", "platform-three.net",
            "platform-four.co", "platform-five.com",
        ],
        "gemini": [
            "solution-a.com", "solution-b.io", "solution-c.co",
            "solution-d.net", "solution-e.com",
        ],
        "perplexity": [
            "research-x.com", "research-y.io", "research-z.co",
            "research-q.net", "research-p.com",
        ],
    }
    vendors = slot_vendors.get(slot, ["example-vendor.com"])

    text = (
        f"Here are the top vendors and platforms for {topic}, based on "
        f"current market data:\n\n"
    )
    for i, v in enumerate(vendors, 1):
        text += f"{i}. {v} — A leading provider in this space with strong customer reviews.\n"
    text += (
        f"\nThese companies are commonly cited in industry reports and "
        f"buyer reviews. Note that market positioning changes frequently."
    )

    return EngineResult(
        text=text,
        citations=vendors,
        model=f"{slot}-mock-v1",
        latency_ms=120,
    )


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def build_engines(
    target_domain: str = "",
    api_keys: Optional[dict] = None,
) -> list[Engine]:
    """
    Build the v1 engine set: Perplexity real (if key), three mocks.

    Returns a list in the order: [perplexity, chatgpt, claude, gemini].
    """
    keys = api_keys or {}
    engines: list[Engine] = []

    # Perplexity — real if key present, else mock
    perplexity_key = (keys.get("openrouter") or OPENROUTER_API_KEY).strip()
    if perplexity_key:
        try:
            engines.append(PerplexityEngine(api_key=perplexity_key, target_domain=target_domain))
        except ValueError:
            engines.append(MockEngine("perplexity"))
    else:
        engines.append(MockEngine("perplexity"))

    # ChatGPT — real via OpenRouter search-preview
    try:
        from src.engines import ChatGPTEngine
        engines.append(ChatGPTEngine(api_key=perplexity_key, target_domain=target_domain))
    except Exception:
        engines.append(MockEngine("chatgpt"))

    # Claude — removed: doesn't return citations via OpenRouter
    # engines.append(MockEngine("claude"))

    # Gemini — removed: doesn't return citations via OpenRouter
    # engines.append(MockEngine("gemini"))

    return engines


# ---------------------------------------------------------------------------
# Citation matrix builder
# ---------------------------------------------------------------------------


def build_citation_matrix(
    domain: str,
    topics: list[str],
    passes: int = None,
    engines: Optional[list[Engine]] = None,
) -> CitationMatrix:
    """
    Build the topic x engine citation matrix for `domain`.

    For each (topic, engine) pair, runs `passes` queries (default 3)
    and aggregates the results. Stores the per-pass results in
    TopicResult.passes so the report can show "2/3 runs cited" instead
    of a single boolean.
    """
    domain = _normalize_domain(domain)
    if not topics:
        topics = [f"Best {domain.split('.')[0].capitalize()} Solutions"]

    if engines is None:
        engines = build_engines(domain)

    if passes is None:
        passes = PASSES_PER_QUERY

    matrix = CitationMatrix(
        domain=domain,
        topics=topics,
        engines=[e.name for e in engines],
    )

    # Set the target domain on each engine so they can check coverage.
    # (Engines don't actually use this; we check after the fact.)
    target_norm = _normalize_domain_for_match(domain)

    for engine in engines:
        for topic in topics:
            result = TopicResult(topic=topic, engine=engine.name, pass_count=passes)
            for _ in range(passes):
                er = engine.query(topic)
                # Determine if target domain is in this pass's citations
                if _target_in(er.citations, target=target_norm):
                    result.covered = True
                result.passes.append(er)
                time.sleep(0.2)  # polite delay between calls

            # Aggregate citations across passes (deduped, preserving order)
            seen: set[str] = set()
            for er in result.passes:
                for c in er.citations:
                    if c not in seen:
                        seen.add(c)
                        result.cited_sources.append(c)

            # v1.2 — brand visibility aggregates
            best_mentions = 0
            best_citations = 0
            best_section = None
            top_comp = None
            for er in result.passes:
                if er.brand_mention_count > best_mentions:
                    best_mentions = er.brand_mention_count
                if er.url_citation_count > best_citations:
                    best_citations = er.url_citation_count
                if er.first_mention_section is not None:
                    if best_section is None or er.first_mention_section < best_section:
                        best_section = er.first_mention_section
                if er.first_competitor and top_comp is None:
                    top_comp = er.first_competitor
            result.best_brand_mentions = best_mentions
            result.best_url_citations = best_citations
            result.best_first_section = best_section
            result.top_competitor = top_comp

            matrix.results.append(result)

    # Build competitor frequency map across all cells
    competitor_counts: dict[str, int] = {}
    for r in matrix.results:
        for src in r.cited_sources:
            if _normalize_domain_for_match(src) == target_norm:
                continue  # don't count self
            competitor_counts[src] = competitor_counts.get(src, 0) + 1
    matrix.all_competitors = dict(
        sorted(competitor_counts.items(), key=lambda x: -x[1])
    )

    return matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def _normalize_domain(d: str) -> str:
    d = (d or "").strip().lower()
    for p in ("https://", "http://"):
        if d.startswith(p):
            d = d[len(p):]
    d = d.split("/")[0]
    return d


def _normalize_domain_for_match(d: str) -> str:
    """More aggressive normalization for comparison: strip www., lowercase."""
    d = _normalize_domain(d)
    if d.startswith("www."):
        d = d[4:]
    return d


def _domain_from_url(url: str) -> Optional[str]:
    """Extract bare domain from a URL. Returns None on failure."""
    try:
        host = urlparse(url).netloc.lower()
    except (ValueError, TypeError):
        return None
    if not host:
        # Try to salvage: maybe the input was just a domain with no scheme
        if "." in url and " " not in url:
            return url.lower().split("/")[0]
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _is_noise_domain(d: str) -> bool:
    d = _normalize_domain_for_match(d)
    return d in NOISE_DOMAINS


def _target_in(citations: list[str], target: Optional[str] = None) -> bool:
    """
    Is the target domain in this list of citations? Uses normalized
    substring matching (target normalized, citation normalized). Phase 1
    of brand_cited detection — see README for the limit.
    """
    if target is None:
        return False
    target_norm = _normalize_domain_for_match(target)
    target_bare = target_norm.split(".")[0]  # e.g. "paretotalent"
    for c in citations:
        c_norm = _normalize_domain_for_match(c)
        if c_norm == target_norm:
            return True
        # Subdomain match: vendor.paretotalent.com -> paretotalent.com
        if c_norm.endswith("." + target_norm):
            return True
        # Brand-name-as-substring: rare but cheap. Require a word
        # boundary (dot or hyphen) on either side so "example" doesn't
        # match "notexample.com".
        if target_bare and len(target_bare) > 4:
            if (
                f".{target_bare}." in f".{c_norm}."
                or f"-{target_bare}." in f".{c_norm}."
                or c_norm.startswith(f"{target_bare}.")
                or c_norm.endswith(f".{target_bare}")
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# v1.2 — position tracking + competitor extraction
# ---------------------------------------------------------------------------


def _find_first_mention_section(text: str, domain: str, brand: str) -> Optional[int]:
    """Find which numbered section/paragraph first mentions the brand or domain."""
    if not domain and not brand:
        return None
    sections = [s.strip() for s in re.split(r'\n(?=#{1,3}\s|\d+\.\s|\*\*)|(?<=\n)\n(?=[A-Z])', text) if s.strip()]
    targets = []
    if brand:
        targets.append(brand.lower())
    if domain:
        targets.append(domain.lower())
    for i, section in enumerate(sections):
        for t in targets:
            if t in section.lower():
                return i + 1
    return None


def _find_first_competitor(text: str, brand: str) -> Optional[dict]:
    """Find the first competitor brand mentioned (capitalized multi-word phrase with URL)."""
    if not brand:
        return None
    brand_lower = brand.lower()
    sections = [s.strip() for s in re.split(r'\n(?=#{1,3}\s|\d+\.\s|\*\*)|(?<=\n)\n(?=[A-Z])', text) if s.strip()]
    for section in sections:
        if brand_lower in section.lower():
            break  # brand itself — skip to next section
        urls = re.findall(r'https?://[^\s<>"\')\\]]+', section)
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', section)
        for cap in caps:
            if cap.lower() != brand_lower and len(cap) > 5:
                return {"name": cap, "url": urls[0] if urls else None}
    return None


def _find_all_competitors(text: str, brand: str) -> list[dict]:
    """Extract all competitor brands mentioned with their URLs."""
    if not brand:
        return []
    brand_lower = brand.lower()
    competitors: list[dict] = []
    seen: set[str] = set()
    sections = [s.strip() for s in re.split(r'\n(?=#{1,3}\s|\d+\.\s|\*\*)|(?<=\n)\n(?=[A-Z])', text) if s.strip()]
    noise = {"linkedin", "facebook", "youtube", "google", "apple", "microsoft", "amazon",
             "twitter", "x", "reddit", "wikipedia", "github", "instagram", "tiktok"}
    for section in sections:
        urls = re.findall(r'https?://[^\s<>"\')\\]]+', section)
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', section)
        for cap in caps:
            cap_lower = cap.lower()
            if cap_lower != brand_lower and cap_lower not in noise and len(cap) > 5 and cap_lower not in seen:
                seen.add(cap_lower)
                competitors.append({"name": cap, "url": urls[0] if urls else None})
    return competitors[:5]
