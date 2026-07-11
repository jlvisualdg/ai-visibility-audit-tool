"""
Domain crawler with AI-readiness signal extraction.

The crawler is intentionally lightweight: requests + BeautifulSoup, polite
delays, respects robots.txt for AI bot directives. It does not aim to
replicate a full Screaming Frog audit — it extracts the ~8 signals that
actually correlate with AI engine citation rates:

- answer_capsules (H2/H3 followed by direct <p> answer)
- stat_density (numbers per 100 words)
- authorship (visible bylines, author schema, meta tags)
- schema (JSON-LD / microdata)
- health_score (composite)
- thin_pages, missing_anchor_text, question_headings
- has_llms_txt, robots_blocks_ai
"""

from __future__ import annotations

import re
import time
import json
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Polite, identifiable crawler
DEFAULT_HEADERS = {
    "User-Agent": (
        "AEO-Audit-Tool/0.1 (+https://boringmarketer.com) "
        "Python-requests"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

# A small set of seed URLs to try. The crawler picks the first one that
# returns 2xx and uses it as the entry point. This is how it handles
# sites where the bare domain redirects to /about or similar.
SEED_PATHS = ["/", "/about", "/services", "/products", "/solutions"]

# Patterns that suggest a question heading — useful for spotting pages
# that *want* to be cited for buyer questions but don't follow through
# with answer capsules.
QUESTION_HEADING_RE = re.compile(
    r"^\s*(what|how|why|when|where|who|which|is|are|can|do|does|should)\b",
    re.IGNORECASE,
)

# Pattern for "data points" — numbers, percentages, currency. Used in
# stat density. Intentionally loose: catches "47%", "$1,200", "3.5x".
STAT_RE = re.compile(
    r"""
    (?:
        \$\d[\d,]*(?:\.\d+)?[kmb]?    # $1, $1.2k, $3.5M
      | \d+(?:\.\d+)?%                  # 47%
      | \d+(?:\.\d+)?x                  # 3.5x
      | \b\d{2,}(?:,\d{3})*(?:\.\d+)?\b # 100, 1,200, 1.2M
    )
    """,
    re.VERBOSE,
)

# AI bot user agents we look for in robots.txt directives
AI_BOT_USER_AGENTS = [
    "GPTBot",
    "ChatGPT-User",
    "OpenAI",
    "OAI-SearchBot",
    "Claude-Web",
    "ClaudeBot",
    "anthropic-ai",
    "PerplexityBot",
    "Perplexity-User",
    "Google-Extended",
    "Google-CloudVertexBot",
    "Applebot-Extended",
    "Amazonbot",
    "Bytespider",
    "CCBot",
    "cohere-ai",
    "FacebookExternalHit",
    "magpie-crawler",
    "meta-externalagent",
    "omgili",
    "omgilibot",
    "PetalBot",
    "Scrapy",
    "Twitterbot",
    "TurnitinBot",
    "YandexAdditional",
    "YandexAdditionalBot",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PageSnapshot:
    """A single crawled page."""
    url: str
    status_code: int
    title: str = ""
    meta_description: str = ""
    word_count: int = 0
    answer_capsules: int = 0
    stat_density: float = 0.0
    has_authorship: bool = False
    has_schema: bool = False
    has_question_headings: bool = False
    question_heading_count: int = 0
    missing_anchor_text: int = 0
    internal_links: int = 0
    h2_count: int = 0
    h3_count: int = 0
    headings: list[str] = field(default_factory=list)   # h1 + h2 text
    schema_types: list[str] = field(default_factory=list)       # JSON-LD @type values
    canonical_url: str = ""
    broken_links_on_page: int = 0
    size_kb: float = 0.0
    response_ms: int = 0
    redirect_hops: int = 0
    # Accessibility / agent readability (HTML-level)
    semantic_landmarks: int = 0
    aria_labeled_elements: int = 0
    images_with_alt: int = 0
    images_total: int = 0
    form_inputs_labeled: int = 0
    form_inputs_total: int = 0
    heading_skip_levels: int = 0
    agent_readability_score: int = 0


@dataclass
class CrawlIssue:
    """A specific crawl-level issue found on a page or across the site."""
    category: str       # "thin_content" | "access_hygiene" | "schema" | ...
    severity: str       # "critical" | "high" | "medium" | "low"
    detail: str
    url: Optional[str] = None


@dataclass
class CrawlResult:
    """The result of crawling one domain."""
    domain: str
    pages_crawled: int = 0
    pages_analyzed: list[PageSnapshot] = field(default_factory=list)
    answer_capsules: int = 0
    stat_density: float = 0.0
    authorship_pages: int = 0
    schema_pages: int = 0
    health_score: int = 0
    total_word_count: int = 0
    thin_pages: list[str] = field(default_factory=list)
    missing_anchor_text: list[str] = field(default_factory=list)
    question_headings: list[str] = field(default_factory=list)
    has_llms_txt: bool = False
    robots_blocks_ai: bool = False
    ai_bots_allowed: int = 0
    ai_bots_blocked: int = 0
    robots_text: str = ""
    issues: list[CrawlIssue] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    title: str = ""
    meta_description: str = ""
    # v1.1 signals
    schema_types_found: list[str] = field(default_factory=list)
    broken_links: list[str] = field(default_factory=list)
    total_size_kb: float = 0.0
    avg_response_ms: int = 0
    max_redirect_hops: int = 0
    has_about_page: bool = False
    has_contact_page: bool = False
    # Agent readability aggregates
    avg_agent_readability: int = 0
    pages_with_landmarks: int = 0
    total_images_missing_alt: int = 0

    @property
    def total_pages(self) -> int:
        return len(self.pages_analyzed)

    @property
    def total_issues(self) -> int:
        return len(self.issues)

    @property
    def issues_by_severity(self) -> dict:
        out = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in self.issues:
            out[issue.severity] = out.get(issue.severity, 0) + 1
        return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def crawl_domain(domain: str, max_pages: int = 10, timeout: int = 15) -> CrawlResult:
    """
    Crawl `domain` and extract AI-readiness signals.

    Picks the first responsive URL from SEED_PATHS, then BFS-crawls internal
    links up to `max_pages` pages. Polite delays (0.5s) between requests.
    Stops on robots.txt disallow (for our user agent) and on consecutive
    errors.
    """
    domain = _normalize_domain(domain)
    result = CrawlResult(domain=domain)

    # 1. Check robots.txt + llms.txt first
    robots_info = _check_robots_txt(domain, timeout=timeout)
    result.robots_text = robots_info["text"]
    result.robots_blocks_ai = robots_info["blocks_ai"]
    result.ai_bots_allowed = robots_info["ai_bots_allowed"]
    result.ai_bots_blocked = robots_info["ai_bots_blocked"]
    result.has_llms_txt = _check_special_file(domain, "/llms.txt", timeout=timeout)

    if robots_info["blocks_self"]:
        result.errors.append(
            f"robots.txt disallows the audit user-agent from {domain}"
        )
        # Still proceed — most sites allow our agent; but flag it

    # 2. Find a responsive seed URL
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    seed_url = None
    for path in SEED_PATHS:
        candidate = f"https://{domain}{path}"
        try:
            r = session.get(candidate, timeout=timeout, allow_redirects=True)
            if r.ok and "text/html" in r.headers.get("Content-Type", ""):
                seed_url = r.url
                break
        except requests.RequestException:
            continue

    if not seed_url:
        # Try http as a last resort
        for path in SEED_PATHS:
            candidate = f"http://{domain}{path}"
            try:
                r = session.get(candidate, timeout=timeout, allow_redirects=True)
                if r.ok and "text/html" in r.headers.get("Content-Type", ""):
                    seed_url = r.url
                    break
            except requests.RequestException:
                continue

    if not seed_url:
        result.errors.append(f"Could not reach {domain} on any of: {SEED_PATHS}")
        result.health_score = 0
        return result

    # 3. BFS crawl
    seen: set[str] = set()
    queue: list[str] = [seed_url]
    consecutive_errors = 0

    while queue and len(result.pages_analyzed) < max_pages and consecutive_errors < 3:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        if not _is_internal(url, domain):
            continue

        try:
            r = session.get(url, timeout=timeout, allow_redirects=True)
            if not r.ok:
                consecutive_errors += 1
                result.errors.append(f"HTTP {r.status_code} on {url}")
                continue
            if "text/html" not in r.headers.get("Content-Type", ""):
                continue

            consecutive_errors = 0
            redirect_hops = len(r.history)
            resp_ms = int(r.elapsed.total_seconds() * 1000)
            snap = _extract_page_signals(r.text, r.url, size_bytes=len(r.content), response_ms=resp_ms, redirect_hops=redirect_hops)
            result.pages_analyzed.append(snap)
            result.pages_crawled += 1

            # Pick up new internal links
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith("#") or href.startswith("mailto:"):
                    continue
                absolute = urljoin(r.url, href)
                # Strip fragment
                absolute = absolute.split("#")[0]
                if _is_internal(absolute, domain) and absolute not in seen:
                    queue.append(absolute)

            time.sleep(0.5)  # polite delay

        except requests.RequestException as e:
            consecutive_errors += 1
            result.errors.append(f"Request error on {url}: {e}")

    # 4. Aggregate
    _aggregate(result)
    return result


def generate_buyer_topics(
    result: CrawlResult,
    n: int = 4,
    page_text_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> list[str]:
    """
    Heuristic buyer-intent topic generation from crawled content.

    No LLM dependency. Strategy:
    - Pull titles, H1/H2, and prominent nouns/phrases from crawled pages
    - Score phrases for buyer-intent signals: comparison, pricing, vendor,
      best/top, alternatives, "for [audience]"
    - Return top N as topic strings

    Topic strings are deliberately phrased as "buyer questions" the site
    should be cited for, e.g. "EA Matching Vendor Recommendations",
    "Best Remote Chief of Staff Platforms".

    `page_text_lookup` is an optional callable that, given a URL, returns
    the page's full text. If provided, we extract h1/h2 from there too.
    """
    corpus: list[str] = []
    for page in result.pages_analyzed:
        if page.title:
            corpus.append(page.title)
        if page.meta_description:
            corpus.append(page.meta_description)
        for h in page.headings:
            corpus.append(h)

    if not corpus:
        # Fallback: a generic set based on the domain
        return [
            f"Best {result.domain.split('.')[0].capitalize()} alternatives",
            f"{result.domain.split('.')[0].capitalize()} reviews and pricing",
            f"How to choose a solution like {result.domain}",
            f"{result.domain.split('.')[0].capitalize()} vs competitors",
        ]

    text = " \n ".join(corpus).lower()

    # Buyer-intent signal words. A phrase that contains one of these
    # (or is a phrase like "X platform/service/tool") is much more likely
    # to be a real buyer topic than a brand word alone.
    intent_triggers = {
        "platform", "platforms", "service", "services", "tool", "tools",
        "software", "solution", "solutions", "vendor", "vendors",
        "agency", "agencies", "company", "companies", "provider", "providers",
        "alternative", "alternatives", "review", "reviews", "pricing",
        "matching", "match", "staff", "remote", "assistant", "virtual",
        "chief", "executive", "talent", "recruiting", "hiring",
        "delegation", "delegate", "system", "systems", "founder", "founders",
        "best", "top", "compare", "comparison", "vs",
    }

    # Exclude bare brand words AND category words that don't make a
    # good buyer-intent question on their own.
    exclude_words = {
        "talent", "pareto", "home", "about", "contact", "blog", "post",
        "page", "site", "website", "click", "here", "read", "more",
        "menu", "login", "sign", "up", "in", "out",
    }

    # Tokenize on non-letters
    tokens = [
        t for t in re.split(r"[^a-z0-9]+", text)
        if t and t not in exclude_words and len(t) > 2
    ]

    # 1-grams + 2-grams + 3-grams
    from collections import Counter
    unigrams = Counter(tokens)
    bigrams = Counter(zip(tokens, tokens[1:]))
    trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))

    scored: list[tuple[float, str]] = []

    # Trigrams score highest when they contain 2+ intent triggers
    for phrase, count in trigrams.most_common(40):
        if sum(1 for t in phrase if t in intent_triggers) >= 2:
            title = " ".join(w.capitalize() for w in phrase)
            scored.append((count * 3.0, title))

    # Bigrams score high when they contain an intent trigger
    for phrase, count in bigrams.most_common(60):
        if sum(1 for t in phrase if t in intent_triggers) >= 1:
            title = " ".join(w.capitalize() for w in phrase)
            # Avoid adding bigrams that are already a substring of an
            # existing scored trigram
            if not any(title in s for _, s in scored):
                scored.append((count * 2.0, title))

    # If we have nothing intent-rich, fall back to the most common bigrams
    if not scored:
        for phrase, count in bigrams.most_common(n):
            title = " ".join(w.capitalize() for w in phrase)
            scored.append((float(count), title))

    # Dedupe, sort, return top n
    seen: set[str] = set()
    topics: list[str] = []
    for _, title in sorted(scored, key=lambda x: -x[0]):
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        topics.append(title)
        if len(topics) >= n:
            break

    # If we still have < n, pad with the bare-domain fallbacks
    bare = result.domain.split(".")[0].capitalize()
    fallbacks = [
        f"Best {bare} Alternatives",
        f"{bare} Reviews and Pricing",
        f"How to Choose a Solution Like {bare}",
        f"{bare} vs Competitors",
    ]
    for f in fallbacks:
        if len(topics) >= n:
            break
        if f.lower() not in {t.lower() for t in topics}:
            topics.append(f)

    return topics[:n]


# ---------------------------------------------------------------------------
# Page-level signal extraction
# ---------------------------------------------------------------------------


def _extract_page_signals(html: str, url: str, size_bytes: int = 0, response_ms: int = 0, redirect_hops: int = 0) -> PageSnapshot:
    """Extract AI-readiness signals from one page's HTML."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    word_count = len(text.split())

    # Title + meta description
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_desc_tag.get("content", "").strip() if meta_desc_tag else ""

    # Canonical URL
    canonical_url = ""
    canonical_tag = soup.find("link", rel="canonical")
    if canonical_tag and canonical_tag.get("href"):
        canonical_url = canonical_tag["href"].strip()

    snap = PageSnapshot(
        url=url,
        status_code=200,
        title=title,
        meta_description=meta_description,
        word_count=word_count,
        canonical_url=canonical_url,
        size_kb=round(size_bytes / 1024, 1) if size_bytes else 0.0,
        response_ms=response_ms,
        redirect_hops=redirect_hops,
    )

    # Answer capsules: H2/H3 followed by a <p> with at least 20 chars within
    # the next 200 characters of the HTML
    h2_h3 = soup.find_all(["h2", "h3"])
    snap.h2_count = len(soup.find_all("h2"))
    snap.h3_count = len(soup.find_all("h3"))

    answer_capsule_count = 0
    question_heading_count = 0

    for heading in h2_h3:
        heading_text = heading.get_text(" ", strip=True)
        if QUESTION_HEADING_RE.match(heading_text):
            question_heading_count += 1

        # Find the next <p> sibling
        nxt = heading.find_next("p")
        if nxt:
            p_text = nxt.get_text(" ", strip=True)
            if 20 <= len(p_text) <= 400:
                answer_capsule_count += 1

    snap.answer_capsules = answer_capsule_count
    snap.has_question_headings = question_heading_count > 0
    snap.question_heading_count = question_heading_count

    # Capture h1 + h2 text for topic generation downstream
    headings: list[str] = []
    for h in soup.find_all(["h1", "h2"]):
        text = h.get_text(" ", strip=True)
        if 8 <= len(text) <= 100:
            headings.append(text)
    snap.headings = headings

    # Stat density
    snap.stat_density = _compute_stat_density(text)

    # Authorship
    snap.has_authorship = _has_authorship(soup)

    # Schema + schema types
    snap.has_schema, snap.schema_types = _has_schema_with_types(soup)

    # Anchor text + internal links
    snap.internal_links, snap.missing_anchor_text, snap.broken_links_on_page = _analyze_anchors(soup, url)

    # Agent readability signals
    _extract_a11y_signals(snap, soup)

    return snap


def _compute_stat_density(text: str) -> float:
    """Number of data points per 100 words."""
    words = text.split()
    if not words:
        return 0.0
    stats = len(STAT_RE.findall(text))
    return round(stats / max(len(words), 1) * 100, 2)


def _has_authorship(soup: BeautifulSoup) -> bool:
    """Check for author byline, author schema, or author meta tag."""
    # Visible byline patterns
    for sel in [
        '[rel="author"]',
        '[class*="author" i]',
        '[itemprop="author"]',
    ]:
        if soup.select_one(sel):
            return True
    # Meta tag
    for meta in soup.find_all("meta"):
        if (meta.get("name", "").lower() == "author"
                or meta.get("property", "").lower() == "article:author"):
            return True
    # JSON-LD Person/Author schema
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(tag.string or "{}")
            if _jsonld_has_author(data):
                return True
        except (json.JSONDecodeError, TypeError):
            continue
    return False


def _jsonld_has_author(obj) -> bool:
    """Recursively check JSON-LD blob for author/Person entries."""
    if isinstance(obj, dict):
        t = obj.get("@type", "")
        if isinstance(t, str) and "person" in t.lower():
            return True
        if isinstance(t, list) and any("person" in str(x).lower() for x in t):
            return True
        for k, v in obj.items():
            if k.lower() in ("author", "creator") and v:
                return True
            if _jsonld_has_author(v):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if _jsonld_has_author(item):
                return True
    return False


def _has_schema_with_types(soup: BeautifulSoup) -> tuple[bool, list[str]]:
    """Check for JSON-LD or microdata schema. Returns (has_schema, [@type values])."""
    types: list[str] = []
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string or not script.string.strip():
            continue
        try:
            data = json.loads(script.string.strip())
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    t = item.get("@type")
                    if isinstance(t, str):
                        types.append(t)
                    elif isinstance(t, list):
                        types.extend(t)
        except (json.JSONDecodeError, AttributeError):
            pass
    for el in soup.find_all(attrs={"itemtype": True}):
        raw = el.get("itemtype", "")
        t = raw.rstrip("/").split("/")[-1]
        if t and t not in types:
            types.append(t)
    return len(types) > 0, types


def _has_schema(soup: BeautifulSoup) -> bool:
    """Legacy wrapper."""
    has, _ = _has_schema_with_types(soup)
    return has


def _analyze_anchors(soup: BeautifulSoup, base_url: str) -> tuple[int, int, int]:
    """Count internal links, missing/generic anchor text, and empty/broken href links.
    Returns (internal_links, missing_anchor, broken_href_count)."""
    GENERIC_ANCHORS = {
        "", "click here", "read more", "learn more", "here", "this",
        "link", "more", "continue", "continue reading", "view more",
    }
    base_host = urlparse(base_url).netloc
    internal = 0
    missing = 0
    broken = 0
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            broken += 1
            continue
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        absolute = urljoin(base_url, href)
        if urlparse(absolute).netloc == base_host:
            internal += 1
            anchor_text = a.get_text(" ", strip=True).lower()
            if anchor_text in GENERIC_ANCHORS or not anchor_text:
                missing += 1
    return internal, missing, broken


# ---------------------------------------------------------------------------
# Aggregate + score
# ---------------------------------------------------------------------------


def _aggregate(result: CrawlResult) -> None:
    """Roll up page-level snapshots into site-level signals and score."""
    pages = result.pages_analyzed
    if not pages:
        result.health_score = 0
        return

    # Rollups
    result.answer_capsules = sum(p.answer_capsules for p in pages)
    result.authorship_pages = sum(1 for p in pages if p.has_authorship)
    result.schema_pages = sum(1 for p in pages if p.has_schema)
    result.total_word_count = sum(p.word_count for p in pages)

    # Average stat density across pages with content
    pages_with_content = [p for p in pages if p.word_count > 50]
    if pages_with_content:
        result.stat_density = round(
            sum(p.stat_density for p in pages_with_content) / len(pages_with_content),
            2,
        )

    # Site title/description: use the first page we crawled
    result.title = pages[0].title
    result.meta_description = pages[0].meta_description

    # Thin pages
    result.thin_pages = [p.url for p in pages if p.word_count < 300]

    # Missing anchor text (flag the URLs where it's worst)
    bad_anchors = sorted(
        (p for p in pages if p.missing_anchor_text > 2),
        key=lambda p: -p.missing_anchor_text,
    )
    result.missing_anchor_text = [p.url for p in bad_anchors[:5]]

    # Question headings (sample)
    result.question_headings = [
        p.url for p in pages if p.has_question_headings
    ][:5]

    # v1.1 signals
    result.total_size_kb = round(sum(p.size_kb for p in pages), 1)
    result.avg_response_ms = int(sum(p.response_ms for p in pages) / len(pages))
    result.max_redirect_hops = max((p.redirect_hops for p in pages), default=0)

    all_types: list[str] = []
    for p in pages:
        for t in p.schema_types:
            if t not in all_types:
                all_types.append(t)
    result.schema_types_found = all_types

    result.broken_links = [p.url for p in pages if p.broken_links_on_page > 0]

    about_paths = ("/about", "/about-us", "/company", "/our-story", "/team")
    contact_paths = ("/contact", "/contact-us", "/get-in-touch", "/support", "/help")
    for p in pages:
        path = urlparse(p.url).path.rstrip("/").lower()
        if any(path == ap or path.endswith(ap) for ap in about_paths):
            result.has_about_page = True
        if any(path == cp or path.endswith(cp) for cp in contact_paths):
            result.has_contact_page = True

    # Agent readability aggregates
    if pages:
        result.avg_agent_readability = int(sum(p.agent_readability_score for p in pages) / len(pages))
        result.pages_with_landmarks = sum(1 for p in pages if p.semantic_landmarks > 0)
        result.total_images_missing_alt = sum(p.images_total - p.images_with_alt for p in pages)

    # Issues
    issues: list[CrawlIssue] = []

    if result.answer_capsules == 0 and result.total_pages >= 3:
        issues.append(CrawlIssue(
            category="answer_capsules",
            severity="high",
            detail=f"No answer-first content patterns detected across {result.total_pages} pages",
        ))

    if result.broken_links:
        issues.append(CrawlIssue(
            category="technical",
            severity="medium",
            detail=f"{len(result.broken_links)} page(s) found with empty or broken href links",
        ))

    if result.max_redirect_hops > 1:
        issues.append(CrawlIssue(
            category="technical",
            severity="low",
            detail=f"Max {result.max_redirect_hops} redirect hops detected — chains over 1 hop degrade crawl efficiency",
        ))

    if not result.has_about_page:
        issues.append(CrawlIssue(
            category="entity_definition",
            severity="medium",
            detail="No About page detected — AI engines use About pages for entity grounding and brand context",
        ))

    if not result.has_contact_page:
        issues.append(CrawlIssue(
            category="entity_definition",
            severity="low",
            detail="No Contact page detected — missing NAP (name/address/phone) harms local entity signals",
        ))

    if result.pages_with_landmarks == 0 and result.total_pages >= 3:
        issues.append(CrawlIssue(
            category="accessibility",
            severity="medium",
            detail="No semantic landmarks (nav, main, header, footer) detected — AI agents rely on landmarks for page structure",
        ))

    if result.total_images_missing_alt > 0:
        issues.append(CrawlIssue(
            category="accessibility",
            severity="low",
            detail=f"{result.total_images_missing_alt} images missing alt text — AI agents use alt text for image context",
        ))

    if result.thin_pages:
        issues.append(CrawlIssue(
            category="thin_content",
            severity="high",
            detail=f"{len(result.thin_pages)} of {result.total_pages} pages have under 300 words",
        ))

    if result.missing_anchor_text:
        issues.append(CrawlIssue(
            category="internal_linking",
            severity="medium",
            detail=f"{len(result.missing_anchor_text)} pages have internal links with missing or generic anchor text",
        ))

    if result.schema_pages == 0:
        issues.append(CrawlIssue(
            category="schema",
            severity="medium",
            detail="No JSON-LD or microdata schema markup detected on crawled pages",
        ))

    if result.authorship_pages == 0 and result.total_pages >= 3:
        issues.append(CrawlIssue(
            category="authorship",
            severity="medium",
            detail="No author bylines or Person schema detected on crawled pages",
        ))

    if not result.has_llms_txt:
        issues.append(CrawlIssue(
            category="access_hygiene",
            severity="low",
            detail="No /llms.txt found — LLM-readable site summary unavailable",
        ))

    if result.robots_blocks_ai:
        issues.append(CrawlIssue(
            category="access_hygiene",
            severity="critical",
            detail=(
                f"robots.txt blocks {result.ai_bots_blocked} of {len(AI_BOT_USER_AGENTS)} "
                f"tracked AI crawlers — only {result.ai_bots_allowed} explicitly allowed"
            ),
        ))

    result.issues = issues
    result.health_score = _compute_health_score(result)


def _compute_health_score(result: CrawlResult) -> int:
    """Composite 0-100 health score from site-level signals."""
    if result.total_pages == 0:
        return 0

    score = 100

    # Big deductions
    if result.answer_capsules == 0:
        score -= 20
    if result.schema_pages == 0:
        score -= 10
    if result.authorship_pages == 0:
        score -= 10

    # Small deductions per thin page (cap at 15)
    score -= min(15, len(result.thin_pages) * 3)

    # Penalty for blocking AI
    if result.robots_blocks_ai:
        score -= 20

    # Bonus for having the new files
    if result.has_llms_txt:
        score += 3

    # v1.1 signals
    if result.has_about_page:
        score += 3
    if result.has_contact_page:
        score += 2
    if result.max_redirect_hops > 1:
        score -= 3
    if result.broken_links:
        score -= min(10, len(result.broken_links) * 2)
    if result.avg_response_ms > 3000:
        score -= 10
    elif result.avg_response_ms > 1500:
        score -= 5

    # Agent readability
    if result.pages_with_landmarks == 0:
        score -= 5
    score += min(5, result.pages_with_landmarks)

    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_domain(domain: str) -> str:
    """Strip scheme, path, and trailing slash from a domain string."""
    d = domain.strip().lower()
    for prefix in ("https://", "http://"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.split("/")[0]
    return d


def _is_internal(url: str, domain: str) -> bool:
    """Is `url` on `domain` (same host or subdomain)?"""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return False
    if not host:
        return False
    # Exact match or subdomain
    return host == domain or host.endswith("." + domain)


def _check_robots_txt(domain: str, timeout: int = 10) -> dict:
    """
    Check robots.txt for AI bot directives. Returns:
    {
        "text": str,              # raw robots.txt
        "blocks_ai": bool,        # True if ANY tracked AI bot is blocked
        "blocks_self": bool,      # blocks our audit user-agent
        "ai_bots_allowed": int,   # count of tracked AI bots explicitly allowed
        "ai_bots_blocked": int,   # count of tracked AI bots explicitly blocked
        "total_ai_bots_tracked": int,
    }
    Parses both Allow and Disallow directives.
    """
    out = {
        "text": "",
        "blocks_ai": False,
        "blocks_self": False,
        "ai_bots_allowed": 0,
        "ai_bots_blocked": 0,
        "total_ai_bots_tracked": len(AI_BOT_USER_AGENTS),
    }
    try:
        r = requests.get(
            f"https://{domain}/robots.txt",
            headers=DEFAULT_HEADERS,
            timeout=timeout,
        )
        if not r.ok:
            return out
        out["text"] = r.text
    except requests.RequestException:
        return out

    bot_decisions: dict[str, bool] = {}  # True=blocked, False=allowed

    current_agents: list[str] = []
    for raw_line in out["text"].splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("user-agent:"):
            current_agents = [line.split(":", 1)[1].strip()]
        elif line.lower().startswith("disallow:"):
            value = line.split(":", 1)[1].strip()
            if value in ("/", ""):
                for agent in current_agents:
                    if agent == "*":
                        out["blocks_ai"] = True
                        for bot in AI_BOT_USER_AGENTS:
                            if bot not in bot_decisions:
                                bot_decisions[bot] = True
                    elif agent in AI_BOT_USER_AGENTS:
                        out["blocks_ai"] = True
                        bot_decisions[agent] = True
                    if agent == DEFAULT_HEADERS["User-Agent"].split(" ")[0] or agent == "*":
                        out["blocks_self"] = True
        elif line.lower().startswith("allow:"):
            value = line.split(":", 1)[1].strip()
            if value in ("/", ""):
                for agent in current_agents:
                    if agent == "*":
                        for bot in AI_BOT_USER_AGENTS:
                            if bot not in bot_decisions:
                                bot_decisions[bot] = False
                    elif agent in AI_BOT_USER_AGENTS:
                        if agent not in bot_decisions:
                            bot_decisions[agent] = False

    out["ai_bots_blocked"] = sum(1 for v in bot_decisions.values() if v)
    out["ai_bots_allowed"] = sum(1 for v in bot_decisions.values() if not v)

    return out


def _check_special_file(domain: str, path: str, timeout: int = 10) -> bool:
    """Check if a special file (e.g. /llms.txt) exists."""
    try:
        r = requests.head(
            f"https://{domain}{path}",
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        if r.ok:
            return True
        # Some servers return 405 for HEAD; fall back to a ranged GET
        r = requests.get(
            f"https://{domain}{path}",
            headers={**DEFAULT_HEADERS, "Range": "bytes=0-100"},
            timeout=timeout,
            allow_redirects=True,
        )
        return r.ok
    except requests.RequestException:
        return False


def _extract_a11y_signals(snap: PageSnapshot, soup: BeautifulSoup) -> None:
    """Extract accessibility / agent-readability signals from HTML (no headless browser)."""
    # Semantic landmarks
    LANDMARKS = {"nav", "main", "header", "footer", "aside", "article", "section"}
    snap.semantic_landmarks = sum(
        1 for tag in LANDMARKS if soup.find(tag)
    )

    # ARIA-labeled elements
    snap.aria_labeled_elements = len(soup.find_all(
        attrs={"aria-label": True}
    )) + len(soup.find_all(
        attrs={"aria-labelledby": True}
    ))

    # Images
    imgs = soup.find_all("img")
    snap.images_total = len(imgs)
    snap.images_with_alt = sum(
        1 for img in imgs if img.get("alt") and img["alt"].strip()
    )

    # Form inputs with labels
    inputs = soup.find_all(["input", "select", "textarea"])
    snap.form_inputs_total = len(inputs)
    snap.form_inputs_labeled = sum(
        1 for inp in inputs
        if inp.get("id") and soup.find("label", attrs={"for": inp["id"]})
        or inp.get("aria-label")
        or inp.get("aria-labelledby")
        or inp.find_parent("label")
    )

    # Heading hierarchy gaps
    heading_tags = [int(h.name[1]) for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])]
    skip_levels = 0
    for i in range(1, len(heading_tags)):
        if heading_tags[i] > heading_tags[i-1] + 1:
            skip_levels += 1
    snap.heading_skip_levels = skip_levels

    # Composite agent readability score (0-100)
    score = 50  # baseline
    if snap.semantic_landmarks >= 3:
        score += 15
    elif snap.semantic_landmarks >= 1:
        score += 5
    if snap.images_total > 0:
        alt_pct = snap.images_with_alt / snap.images_total
        score += int(alt_pct * 15)
    if snap.form_inputs_total > 0:
        labeled_pct = snap.form_inputs_labeled / snap.form_inputs_total
        score += int(labeled_pct * 10)
    else:
        score += 10  # no forms = no penalty
    if snap.heading_skip_levels == 0:
        score += 10
    elif snap.heading_skip_levels <= 2:
        score += 5
    score += min(5, snap.aria_labeled_elements)

    snap.agent_readability_score = max(0, min(100, score))
