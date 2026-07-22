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


# Targeted crawl — path candidates for each page type
_ABOUT_PATHS = ["/about", "/about-us", "/company", "/our-story", "/team", "/who-we-are",
                "/leadership", "/our-team", "/people", "/founders", "/founder"]
_CONTACT_PATHS = ["/contact", "/contact-us", "/get-in-touch", "/reach-us", "/support", "/help"]

# URL path patterns that suggest a service/product page
_SERVICE_PATH_RE = re.compile(
    r"/(services?|solutions?|products?|offer(?:ing)?s?|platform|tools?|"
    r"capabilities|work|what-we-do|how-it-works|packages?|plans?)",
    re.IGNORECASE,
)
# Paths to exclude from service page candidates
_NON_SERVICE_PATH_RE = re.compile(
    r"/(about|contact|blog|news|press|media|careers?|jobs?|login|sign[- ]?(?:in|up)|"
    r"register|privacy|terms|faq|help|support|legal|cookie)",
    re.IGNORECASE,
)

# Social media domains for credibility checks
_SOCIAL_DOMAINS = (
    "facebook.com", "twitter.com", "x.com", "linkedin.com",
    "instagram.com", "youtube.com", "tiktok.com", "threads.net",
)

# Phone number regex (loose — catches US formats + international +XX prefix)
_PHONE_RE = re.compile(
    r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)

# Credentials/degree keywords for people extraction
_CREDENTIALS_RE = re.compile(
    r"\b(phd|ph\.d|md|m\.d|cpa|cfa|mba|licensed|certified|degree|j\.?d\.?|esq)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PersonSignal:
    """A person found on a crawled page."""
    name: str
    role: str = ""
    has_bio: bool = False
    has_credentials: bool = False  # mentions degrees, licenses, certifications
    source_url: str = ""


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
    # Page type tag set during targeted crawl
    page_type: str = ""  # "homepage" | "about" | "contact" | "service" | "other"
    # Credibility / trustworthiness signals
    has_privacy_link: bool = False
    has_terms_link: bool = False
    has_nap_signals: bool = False   # phone number or physical address
    has_trust_signals: bool = False  # testimonials, reviews, client logos
    has_social_links: bool = False
    social_channel_count: int = 0
    # Schema quality signals
    schema_quality_score: int = 0
    schema_missing_props: list[str] = field(default_factory=list)
    # NEEATT-specific signals (populated by _extract_credibility_signals)
    has_media_mentions: bool = False
    has_stat_counters: bool = False
    has_ratings_widget: bool = False
    has_corp_registration: bool = False
    # Raw HTML for targeted pages (about/homepage only) — used for people extraction
    raw_html: str = field(default="", repr=False)


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
    # Access/read restrictions encountered during crawl (set in _aggregate)
    rate_limited: bool = False          # any HTTP 429 seen
    access_restricted: bool = False     # any HTTP 401/403 seen
    core_pages_only: bool = False       # broad crawl skipped due to rate limiting
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
    # Targeted page URLs (set during directed crawl)
    homepage_url: str = ""
    about_url: str = ""
    contact_url: str = ""
    service_url: str = ""
    service_keyword: str = ""
    # Credibility audit (derived from homepage snapshot)
    has_ssl: bool = False
    has_privacy_policy: bool = False
    has_terms: bool = False
    has_contact_info_on_homepage: bool = False
    has_trust_signals_on_homepage: bool = False
    has_social_links: bool = False
    social_channel_count: int = 0
    credibility_score: int = 0
    # NEEATT sub-scores (0-100 each, populated in _compute_credibility_score)
    neeatt_notability: int = 0
    neeatt_experience: int = 0
    neeatt_expertise: int = 0
    neeatt_authoritativeness: int = 0
    neeatt_trustworthiness: int = 0
    neeatt_transparency: int = 0
    # Derived signals for report display
    has_media_mentions: bool = False     # "as seen on" / press logos on homepage
    has_stat_counters: bool = False      # "1,200+ projects" style counters
    has_ratings_widget: bool = False     # Trustpilot / Google rating widget
    has_corp_registration: bool = False  # EIN, LLC, incorporation mention
    people_with_bios: int = 0           # count of people with bios (from people list)
    people_with_credentials: int = 0    # count of people with credential keywords
    # Schema quality (from homepage)
    schema_quality_score: int = 0
    schema_missing_props: list[str] = field(default_factory=list)
    # People found on about page
    people: list[PersonSignal] = field(default_factory=list)

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

    @property
    def data_incomplete(self) -> bool:
        """True if access/read restrictions likely made the crawl partial."""
        return self.rate_limited or self.access_restricted

    @property
    def access_error_summary(self) -> list[str]:
        """Human-readable summary of the specific read issues encountered.

        Buckets the raw ``errors`` list by HTTP status / failure type so the
        report can show 'N requests rate-limited (HTTP 429)' rather than a wall
        of URLs.
        """
        import re as _re
        n_429 = n_403 = n_404 = n_5xx = n_other = 0
        n_req_err = 0
        for e in self.errors:
            m = _re.search(r"HTTP (\d{3})", e)
            if m:
                code = int(m.group(1))
                if code == 429:
                    n_429 += 1
                elif code in (401, 403):
                    n_403 += 1
                elif code == 404:
                    n_404 += 1
                elif 500 <= code <= 599:
                    n_5xx += 1
                else:
                    n_other += 1
            elif "Request error" in e or "Could not reach" in e:
                n_req_err += 1
        summary: list[str] = []
        if n_429:
            summary.append(f"{n_429} request{'s' if n_429 != 1 else ''} rate-limited (HTTP 429)")
        if n_403:
            summary.append(f"{n_403} request{'s' if n_403 != 1 else ''} access-restricted (HTTP 401/403)")
        if n_5xx:
            summary.append(f"{n_5xx} request{'s' if n_5xx != 1 else ''} hit a server error (HTTP 5xx)")
        if n_404:
            summary.append(f"{n_404} page{'s' if n_404 != 1 else ''} not found (HTTP 404)")
        if n_other:
            summary.append(f"{n_other} request{'s' if n_other != 1 else ''} returned an unexpected status")
        if n_req_err:
            summary.append(f"{n_req_err} request{'s' if n_req_err != 1 else ''} failed to connect or timed out")
        return summary


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def crawl_domain(domain: str, max_pages: int = 10, timeout: int = 15, main_keyword: str = "") -> CrawlResult:
    """
    Crawl `domain` and extract AI-readiness signals.

    Phase 1 — targeted fetches (always attempted first):
      homepage → about → contact → primary service page

    Phase 2 — BFS over remaining page budget.

    Polite 0.5s delays between requests. Stops on consecutive errors.

    Args:
        main_keyword: Optional hint (e.g. first BOTF topic) used to score
                      which nav link is the primary service page. When omitted
                      the crawler derives a keyword from the homepage H1/title.
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

    # 2. Set up session
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    seen: set[str] = set()

    # ── Phase 1: Targeted fetches ──────────────────────────────────────────

    # 2a. Homepage (required — bail if unreachable)
    homepage_snap, homepage_soup = _fetch_and_analyze(
        session, f"https://{domain}/", domain, timeout, "homepage", result
    )
    if homepage_snap is None:
        homepage_snap, homepage_soup = _fetch_and_analyze(
            session, f"http://{domain}/", domain, timeout, "homepage", result
        )
    if homepage_snap is None:
        result.errors.append(f"Could not reach {domain}")
        _aggregate(result)  # classifies any 429/403 so the report explains why
        return result

    result.pages_analyzed.append(homepage_snap)
    result.pages_crawled += 1
    result.homepage_url = homepage_snap.url
    result.has_ssl = homepage_snap.url.startswith("https://")
    seen.add(homepage_snap.url)

    # Seed keyword for service page discovery
    if not main_keyword and homepage_soup is not None:
        main_keyword = _derive_main_keyword(homepage_soup, domain)

    def _targeted_fetch(candidates: list[str], page_type: str) -> tuple:
        """Try each candidate URL; return (snap, soup) for the first that works."""
        for url in candidates:
            if url in seen:
                continue
            snap, soup = _fetch_and_analyze(session, url, domain, timeout, page_type, result)
            if snap is not None:
                return snap, soup
        return None, None

    # 2b. About page
    if len(result.pages_analyzed) < max_pages:
        about_candidates = []
        if homepage_soup is not None:
            found = _find_nav_url(homepage_soup, domain, _ABOUT_PATHS)
            if found:
                about_candidates.append(found)
        about_candidates += [f"https://{domain}{p}" for p in _ABOUT_PATHS]

        about_snap, _ = _targeted_fetch(about_candidates, "about")

        # Last resort: scan footer links for about-like pages
        if about_snap is None and homepage_soup is not None:
            footer = homepage_soup.find("footer")
            if footer:
                for a in footer.find_all("a", href=True):
                    href = a["href"].strip()
                    if not href:
                        continue
                    text = a.get_text(" ", strip=True).lower()
                    if any(kw in text or kw in href.lower() for kw in ("about", "team", "people", "story", "who we are")):
                        abs_url = urljoin(result.homepage_url, href).split("#")[0]
                        if _is_internal(abs_url, result.domain) and abs_url not in seen:
                            about_snap, _ = _fetch_and_analyze(session, abs_url, domain, timeout, "about", result)
                            if about_snap:
                                break

        if about_snap:
            result.pages_analyzed.append(about_snap)
            result.pages_crawled += 1
            result.about_url = about_snap.url
            seen.add(about_snap.url)

    # 2c. Contact page
    if len(result.pages_analyzed) < max_pages:
        contact_candidates = []
        if homepage_soup is not None:
            found = _find_nav_url(homepage_soup, domain, _CONTACT_PATHS)
            if found:
                contact_candidates.append(found)
        contact_candidates += [f"https://{domain}{p}" for p in _CONTACT_PATHS]

        contact_snap, _ = _targeted_fetch(contact_candidates, "contact")
        if contact_snap:
            result.pages_analyzed.append(contact_snap)
            result.pages_crawled += 1
            result.contact_url = contact_snap.url
            seen.add(contact_snap.url)

    # 2d. Service page (matches the business's primary keyword)
    if len(result.pages_analyzed) < max_pages and homepage_soup is not None:
        service_url = _find_service_page_url(homepage_soup, domain, main_keyword)
        if service_url and service_url not in seen:
            service_snap, _ = _fetch_and_analyze(
                session, service_url, domain, timeout, "service", result
            )
            if service_snap:
                result.pages_analyzed.append(service_snap)
                result.pages_crawled += 1
                result.service_url = service_snap.url
                result.service_keyword = main_keyword
                seen.add(service_snap.url)

    # ── Phase 2: BFS for remaining page budget ─────────────────────────────

    # Rate-limit fallback: if the targeted core-page phase already tripped an
    # HTTP 429, skip the broad BFS crawl entirely. Continuing would hammer a
    # site that's already throttling us; instead we analyze core pages only.
    if any("HTTP 429" in e for e in result.errors):
        result.core_pages_only = True
        _aggregate(result)
        return result

    # Seed queue from homepage links
    bfs_queue: list[str] = []
    if homepage_soup is not None:
        for a in homepage_soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("mailto:"):
                continue
            absolute = urljoin(result.homepage_url, href).split("#")[0]
            if _is_internal(absolute, domain) and absolute not in seen:
                bfs_queue.append(absolute)

    consecutive_errors = 0
    while bfs_queue and len(result.pages_analyzed) < max_pages and consecutive_errors < 3:
        url = bfs_queue.pop(0)
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
                # Rate-limit fallback: stop the broad crawl immediately on 429
                # to avoid overloading the site. Core pages are already analyzed.
                if r.status_code == 429:
                    result.core_pages_only = True
                    break
                continue
            if "text/html" not in r.headers.get("Content-Type", ""):
                continue

            consecutive_errors = 0
            snap = _extract_page_signals(
                r.text, r.url,
                size_bytes=len(r.content),
                response_ms=int(r.elapsed.total_seconds() * 1000),
                redirect_hops=len(r.history),
            )
            snap.page_type = "other"
            result.pages_analyzed.append(snap)
            result.pages_crawled += 1

            # Expand queue from this page
            link_soup = BeautifulSoup(r.text, "lxml")
            for a in link_soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith("#") or href.startswith("mailto:"):
                    continue
                absolute = urljoin(r.url, href).split("#")[0]
                if _is_internal(absolute, domain) and absolute not in seen:
                    bfs_queue.append(absolute)

            time.sleep(0.5)

        except requests.RequestException as e:
            consecutive_errors += 1
            result.errors.append(f"Request error on {url}: {e}")

    # 3. Aggregate
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
    schema_analysis = _analyze_schema_graph(soup)
    snap.schema_quality_score = schema_analysis["quality_score"]
    snap.schema_missing_props = schema_analysis["missing_props"]

    # Anchor text + internal links
    snap.internal_links, snap.missing_anchor_text, snap.broken_links_on_page = _analyze_anchors(soup, url)

    # Agent readability signals
    _extract_a11y_signals(snap, soup)

    # Credibility signals (used on every page; aggregated from homepage)
    cred = _extract_credibility_signals(soup, url)
    snap.has_privacy_link = cred["has_privacy_link"]
    snap.has_terms_link = cred["has_terms_link"]
    snap.has_nap_signals = cred["has_nap_signals"]
    snap.has_trust_signals = cred["has_trust_signals"]
    snap.has_social_links = cred["has_social_links"]
    snap.social_channel_count = cred.get("social_channel_count", 0)
    snap.has_media_mentions = cred.get("has_media_mentions", False)
    snap.has_stat_counters = cred.get("has_stat_counters", False)
    snap.has_ratings_widget = cred.get("has_ratings_widget", False)
    snap.has_corp_registration = cred.get("has_corp_registration", False)

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
    """Check for JSON-LD or microdata schema. Returns (has_schema, [@type values]).
    Handles @graph nesting where JSON-LD wraps types in a graph array."""
    types: list[str] = []

    def _extract_types(obj):
        """Recursively extract @type values from nested JSON-LD."""
        if isinstance(obj, dict):
            t = obj.get("@type")
            if isinstance(t, str):
                types.append(t)
            elif isinstance(t, list):
                types.extend(t)
            # Check @graph (common pattern: {"@graph": [{@type: "WebSite"}, ...]})
            if "@graph" in obj:
                _extract_types(obj["@graph"])
            # Recurse into nested dicts
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    _extract_types(v)
        elif isinstance(obj, list):
            for item in obj:
                _extract_types(item)

    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string or not script.string.strip():
            continue
        try:
            data = json.loads(script.string.strip())
            _extract_types(data)
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


# Known LocalBusiness subtypes (non-exhaustive but covers common ones)
_LOCAL_BUSINESS_SUBTYPES = {
    "localbusiness", "restaurant", "foodestablishment", "store", "hotel",
    "lodgingbusiness", "dentist", "physician", "medicalorganization",
    "hospital", "pharmacy", "accountingservice", "financialservice",
    "realestate", "realestateagent", "legalservice", "lawyer",
    "autodealer", "autorepair", "beautysalon", "hairsalon", "spa",
    "gym", "fitnesscenter", "library", "museum", "park",
    "professionaleservice", "homegoodsstore", "clothingstore",
    "electronicsstore", "florist", "movingcompany", "plumber",
    "electrician", "generalcontractor", "roofingcontractor",
    "insuranceagency", "travelagency", "veterinarycare",
}


def _analyze_schema_graph(soup: BeautifulSoup) -> dict:
    """
    Parse all JSON-LD script tags and score the entity graph quality.

    Returns:
        {
            "quality_score": int (0-100),
            "entities": list[str],        # deduped @type values found
            "missing_props": list[str],   # human-readable descriptions of absences
        }
    """
    # Collect all entity objects from JSON-LD
    entities: list[dict] = []

    def _collect_entities(obj):
        if isinstance(obj, dict):
            if "@type" in obj:
                entities.append(obj)
            if "@graph" in obj and isinstance(obj["@graph"], list):
                for item in obj["@graph"]:
                    _collect_entities(item)
            for v in obj.values():
                if isinstance(v, dict):
                    _collect_entities(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            _collect_entities(item)
        elif isinstance(obj, list):
            for item in obj:
                _collect_entities(item)

    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string or not script.string.strip():
            continue
        try:
            data = json.loads(script.string.strip())
            _collect_entities(data)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Build lookup by normalized @type
    def _types_of(entity: dict) -> list[str]:
        t = entity.get("@type", "")
        if isinstance(t, str):
            return [t]
        if isinstance(t, list):
            return [str(x) for x in t]
        return []

    def _find_entities_by_type(type_name: str) -> list[dict]:
        """Find entities whose @type matches type_name (case-insensitive)."""
        name_lower = type_name.lower()
        return [e for e in entities if any(t.lower() == name_lower for t in _types_of(e))]

    def _has_type(type_name: str) -> bool:
        return len(_find_entities_by_type(type_name)) > 0

    def _has_local_business_type() -> tuple[bool, list[dict]]:
        """Returns (found, matching_entities) for LocalBusiness or any known subtype."""
        matches = [
            e for e in entities
            if any(t.lower() in _LOCAL_BUSINESS_SUBTYPES for t in _types_of(e))
        ]
        return len(matches) > 0, matches

    # Scoring
    score = 0
    missing: list[str] = []

    # --- Organization checks ---
    org_entities = _find_entities_by_type("Organization")
    has_org = len(org_entities) > 0

    if has_org:
        score += 15
        org = org_entities[0]

        if org.get("name"):
            score += 5
        else:
            missing.append("Organization schema missing name property")

        if org.get("url"):
            score += 5
        else:
            missing.append("Organization schema missing url property")

        if org.get("logo"):
            score += 5
        else:
            missing.append("Organization missing logo property")

        if org.get("contactPoint"):
            score += 5
        else:
            missing.append("Organization missing contactPoint property")

        same_as = org.get("sameAs", [])
        if isinstance(same_as, str):
            same_as = [same_as]
        if isinstance(same_as, list) and len(same_as) >= 1:
            score += 10
        else:
            missing.append("No sameAs links to social profiles or entity databases")

        if org.get("@id"):
            score += 5
        else:
            missing.append("Organization missing @id (entity identifier)")
    else:
        missing.append("Organization schema missing")

    # --- WebSite checks ---
    website_entities = _find_entities_by_type("WebSite")
    has_website = len(website_entities) > 0

    if has_website:
        score += 10
        ws = website_entities[0]

        # SearchAction potential action
        potential_action = ws.get("potentialAction")
        has_search_action = False
        if isinstance(potential_action, dict):
            action_type = potential_action.get("@type", "")
            if isinstance(action_type, str) and "searchaction" in action_type.lower():
                has_search_action = True
        elif isinstance(potential_action, list):
            for action in potential_action:
                if isinstance(action, dict) and "searchaction" in str(action.get("@type", "")).lower():
                    has_search_action = True
                    break
        if has_search_action:
            score += 5
        else:
            missing.append("WebSite missing potentialAction SearchAction (enables sitelinks search)")

        # WebSite linked to Org via publisher or @id cross-ref
        ws_publisher = ws.get("publisher")
        org_id = org_entities[0].get("@id") if org_entities else None
        ws_linked = False
        if ws_publisher:
            if isinstance(ws_publisher, dict) and ws_publisher.get("@id") and org_id:
                ws_linked = ws_publisher["@id"] == org_id
            elif isinstance(ws_publisher, dict) and ws_publisher.get("@type"):
                ws_linked = True  # at least has a publisher reference
        if not ws_linked and org_id:
            # Check if website's @id or about references org
            if ws.get("@id") or ws.get("about"):
                ws_linked = True
        if ws_linked:
            score += 5
        else:
            missing.append("WebSite not linked to Organization via publisher property")
    else:
        missing.append("WebSite schema missing")

    # --- LocalBusiness checks ---
    has_lb, lb_entities = _has_local_business_type()
    if has_lb:
        score += 10
        lb = lb_entities[0]

        if lb.get("address"):
            score += 5
        else:
            missing.append("LocalBusiness missing address property")

        if lb.get("geo"):
            score += 5
        else:
            missing.append("LocalBusiness missing geo (lat/lng) property")

        if lb.get("openingHours") or lb.get("openingHoursSpecification"):
            score += 5
        else:
            missing.append("LocalBusiness missing openingHours or openingHoursSpecification")
    # No missing_props for LocalBusiness absence — it only applies to local businesses

    # Dedupe entity types for reporting
    all_type_strings: list[str] = []
    for e in entities:
        for t in _types_of(e):
            if t and t not in all_type_strings:
                all_type_strings.append(t)

    # Normalize: LocalBusiness points only apply to local sites.
    # Without LB, max achievable is 70 (Org 50 + WebSite 20). Renormalize to 100
    # so a fully-complete national/SaaS site can score 100, not 70.
    max_applicable = 95 if has_lb else 70
    normalized = round(min(score, max_applicable) * 100 / max_applicable) if max_applicable else 0

    return {
        "quality_score": max(0, min(100, normalized)),
        "entities": all_type_strings,
        "missing_props": missing,
    }


def _extract_people(soup: BeautifulSoup, url: str) -> list[PersonSignal]:
    """
    Extract person signals from a page (typically an about/team page).

    Sources checked (in order):
      1. JSON-LD Person objects
      2. Microdata [itemtype*="Person"]
      3. Elements with team/staff/member/person/bio class names

    Returns up to 10 PersonSignal objects, deduplicated by name.
    """
    people: list[PersonSignal] = []
    seen_names: set[str] = set()

    def _add_person(name: str, role: str = "", has_bio: bool = False,
                    has_credentials: bool = False) -> None:
        key = name.strip().lower()
        if not key or key in seen_names:
            return
        # Must look like a real name: 2-5 words, each word capitalized-ish
        words = name.strip().split()
        if len(words) < 2 or len(words) > 5:
            return
        seen_names.add(key)
        people.append(PersonSignal(
            name=name.strip(),
            role=role.strip(),
            has_bio=has_bio,
            has_credentials=has_credentials,
            source_url=url,
        ))

    # 1. JSON-LD Person objects
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string.strip())
        except (json.JSONDecodeError, AttributeError):
            continue

        def _walk_jsonld(obj):
            if isinstance(obj, dict):
                types = obj.get("@type", "")
                if isinstance(types, str):
                    types = [types]
                if isinstance(types, list) and any("person" in str(t).lower() for t in types):
                    name = obj.get("name", "")
                    role = obj.get("jobTitle", "")
                    if name:
                        _add_person(name, role)
                for v in obj.values():
                    _walk_jsonld(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk_jsonld(item)

        _walk_jsonld(data)

    # 2. Microdata [itemtype*="Person"]
    for item in soup.find_all(attrs={"itemtype": True}):
        if "person" not in item.get("itemtype", "").lower():
            continue
        name_el = item.find(attrs={"itemprop": "name"})
        role_el = item.find(attrs={"itemprop": "jobTitle"})
        if name_el:
            name = name_el.get_text(" ", strip=True)
            role = role_el.get_text(" ", strip=True) if role_el else ""
            _add_person(name, role)

    # 3. Class-based team/staff/bio sections
    _TEAM_SELECTORS = [
        '[class*="team" i]', '[class*="staff" i]', '[class*="member" i]',
        '[class*="person" i]', '[class*="bio" i]',
    ]
    for selector in _TEAM_SELECTORS:
        for container in soup.select(selector):
            # Find a name from the first heading-like element with 2-5 words
            name = ""
            role = ""
            for tag in container.find_all(["h2", "h3", "h4", "strong"]):
                text = tag.get_text(" ", strip=True)
                words = text.split()
                if 2 <= len(words) <= 5:
                    name = text
                    break

            if not name:
                continue

            # Find role from sibling or role-class element
            role_el = container.find(
                attrs={"class": lambda c: c and any(
                    kw in " ".join(c).lower() for kw in ("title", "role", "position")
                )}
            ) if hasattr(container, "find") else None
            if role_el:
                role = role_el.get_text(" ", strip=True)
            else:
                # Try next sibling <p>
                name_tag = container.find(["h2", "h3", "h4", "strong"])
                if name_tag:
                    sib = name_tag.find_next_sibling("p")
                    if sib:
                        role = sib.get_text(" ", strip=True)

            # Bio and credentials from full container text
            container_text = container.get_text(" ", strip=True)
            has_bio = len(container_text.split()) >= 50
            has_creds = bool(_CREDENTIALS_RE.search(container_text))

            _add_person(name, role, has_bio, has_creds)

        if len(people) >= 10:
            break

    return people[:10]


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
    # ── Access / read-restriction detection ──────────────────────────────
    # Parse recorded fetch errors for HTTP status codes so the report can warn
    # that signals may be incomplete when the site throttled or blocked us.
    _n429 = _n403 = 0
    for e in result.errors:
        m = re.search(r"HTTP (\d{3})", e)
        if not m:
            continue
        code = int(m.group(1))
        if code == 429:
            _n429 += 1
        elif code in (401, 403):
            _n403 += 1
    if _n429:
        result.rate_limited = True
        result.issues.append(CrawlIssue(
            category="access_hygiene",
            severity="high",
            detail=(
                f"Site rate-limited the crawler (HTTP 429 on {_n429} request"
                f"{'s' if _n429 != 1 else ''}) — indexability signals below may be "
                "incomplete. Rate limits are enforced by request rate, so AI "
                "crawlers (GPTBot, PerplexityBot, ClaudeBot) indexing this site can "
                "be throttled the same way."
            ),
        ))
    if _n403:
        result.access_restricted = True
        result.issues.append(CrawlIssue(
            category="access_hygiene",
            severity="high",
            detail=(
                f"Site returned access-restricted responses (HTTP 401/403 on "
                f"{_n403} request{'s' if _n403 != 1 else ''}) — some pages could not "
                "be read, so signals below may be incomplete."
            ),
        ))

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

    # has_about_page / has_contact_page — from targeted URLs first, then URL scan
    if result.about_url:
        result.has_about_page = True
    if result.contact_url:
        result.has_contact_page = True
    if not result.has_about_page or not result.has_contact_page:
        about_paths = ("/about", "/about-us", "/company", "/our-story", "/team")
        contact_paths = ("/contact", "/contact-us", "/get-in-touch", "/support", "/help")
        for p in pages:
            path = urlparse(p.url).path.rstrip("/").lower()
            if any(path == ap or path.endswith(ap) for ap in about_paths):
                result.has_about_page = True
            if any(path == cp or path.endswith(cp) for cp in contact_paths):
                result.has_contact_page = True

    # Credibility fields — from homepage snapshot (first page tagged "homepage",
    # or the first page as fallback for BFS-only runs)
    homepage_snap = next(
        (p for p in pages if p.page_type == "homepage"),
        pages[0] if pages else None,
    )
    if homepage_snap:
        result.has_privacy_policy = homepage_snap.has_privacy_link
        result.has_terms = homepage_snap.has_terms_link
        result.has_contact_info_on_homepage = homepage_snap.has_nap_signals
        result.has_trust_signals_on_homepage = homepage_snap.has_trust_signals
        result.has_social_links = homepage_snap.has_social_links
        result.social_channel_count = homepage_snap.social_channel_count
        # NEEATT-specific signals
        result.has_media_mentions = homepage_snap.has_media_mentions
        result.has_stat_counters = homepage_snap.has_stat_counters
        result.has_ratings_widget = homepage_snap.has_ratings_widget
        result.has_corp_registration = homepage_snap.has_corp_registration

    # Schema quality — taken from homepage (most important page for entity graph)
    if homepage_snap:
        result.schema_quality_score = homepage_snap.schema_quality_score
        result.schema_missing_props = homepage_snap.schema_missing_props

    # Agent readability aggregates
    if pages:
        result.avg_agent_readability = int(sum(p.agent_readability_score for p in pages) / len(pages))
        result.pages_with_landmarks = sum(1 for p in pages if p.semantic_landmarks > 0)
        result.total_images_missing_alt = sum(p.images_total - p.images_with_alt for p in pages)

    # People extraction from about page
    about_snap_for_people = next((p for p in pages if p.page_type == "about"), None)
    if about_snap_for_people and about_snap_for_people.raw_html:
        from bs4 import BeautifulSoup as _BS
        about_soup = _BS(about_snap_for_people.raw_html, "lxml")
        result.people = _extract_people(about_soup, about_snap_for_people.url)

    # People counts for NEEATT scoring (set here for report template access)
    result.people_with_bios = sum(1 for p in result.people if p.has_bio)
    result.people_with_credentials = sum(1 for p in result.people if p.has_credentials)

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
    elif result.pages_with_landmarks > 0 and result.pages_with_landmarks < result.total_pages // 2:
        issues.append(CrawlIssue(
            category="accessibility",
            severity="low",
            detail=f"Partial semantic landmark coverage ({result.pages_with_landmarks}/{result.total_pages} pages) — add nav/main/header/footer elements to remaining pages",
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

    # Schema quality issues (richer than the old binary check)
    schema_q = result.schema_quality_score
    if schema_q == 0:
        issues.append(CrawlIssue(
            category="schema",
            severity="critical",
            detail="No schema markup on homepage — AI engines cannot establish entity identity for this domain",
        ))
    elif schema_q < 40:
        # Emit individual issues for the top missing props
        for mp in result.schema_missing_props[:3]:
            issues.append(CrawlIssue(
                category="schema",
                severity="high",
                detail=mp,
            ))
        issues.append(CrawlIssue(
            category="schema",
            severity="high",
            detail=f"Homepage schema is incomplete (quality score {schema_q}/100) — critical entity properties are missing",
        ))
    elif schema_q < 70:
        for mp in result.schema_missing_props[:2]:
            issues.append(CrawlIssue(
                category="schema",
                severity="medium",
                detail=mp,
            ))

    # Partial coverage check for non-homepage pages
    if result.schema_pages < result.total_pages // 2:
        issues.append(CrawlIssue(
            category="schema",
            severity="low",
            detail=f"Partial schema coverage ({result.schema_pages}/{result.total_pages} pages) — extend JSON-LD markup to all pages for consistent AI entity signals",
        ))

    if result.authorship_pages == 0 and result.total_pages >= 3:
        issues.append(CrawlIssue(
            category="authorship",
            severity="medium",
            detail="No author bylines or Person schema detected on crawled pages",
        ))
    elif result.authorship_pages > 0 and result.authorship_pages < result.total_pages // 2:
        issues.append(CrawlIssue(
            category="authorship",
            severity="low",
            detail=f"Partial authorship coverage ({result.authorship_pages}/{result.total_pages} pages) — AI engines prefer consistent author signals across all content pages",
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

    # ── Response time threshold ───────────────────────────────────────────
    if result.avg_response_ms > 5000:
        issues.append(CrawlIssue(
            category="performance",
            severity="critical",
            detail=f"Extremely slow avg response time ({result.avg_response_ms}ms) — AI crawlers time out at ~5s and will skip slow pages",
        ))
    elif result.avg_response_ms > 3000:
        issues.append(CrawlIssue(
            category="performance",
            severity="high",
            detail=f"Slow avg response time ({result.avg_response_ms}ms) — ideal is under 3s to ensure reliable AI crawler access",
        ))
    elif result.avg_response_ms > 1000:
        issues.append(CrawlIssue(
            category="performance",
            severity="medium",
            detail=f"Moderate avg response time ({result.avg_response_ms}ms) — under 1s is ideal; slower responses risk AI crawler timeouts",
        ))

    # ── Credibility / baseline trust checks ──────────────────────────────
    if not result.has_ssl:
        issues.append(CrawlIssue(
            category="credibility",
            severity="critical",
            detail="Site not served over HTTPS — AI crawlers and modern browsers treat HTTP as insecure",
        ))

    if not result.has_privacy_policy:
        issues.append(CrawlIssue(
            category="credibility",
            severity="medium",
            detail="No privacy policy link found on homepage — required for compliance and baseline credibility",
        ))

    if not result.has_terms:
        issues.append(CrawlIssue(
            category="credibility",
            severity="low",
            detail="No terms of service link found — baseline legal credibility signal for AI entity evaluation",
        ))

    if not result.has_contact_info_on_homepage:
        issues.append(CrawlIssue(
            category="credibility",
            severity="medium",
            detail="No phone number or address on homepage — NAP (name/address/phone) strengthens entity recognition",
        ))

    if not result.has_trust_signals_on_homepage:
        issues.append(CrawlIssue(
            category="credibility",
            severity="medium",
            detail=(
                "No testimonials, reviews, or client logos detected on homepage — "
                "social proof signals influence AI recommendation likelihood"
            ),
        ))

    if not result.has_social_links:
        issues.append(CrawlIssue(
            category="credibility",
            severity="low",
            detail="No social media links found — social presence is a minor entity authority signal for AI engines",
        ))

    result.issues = issues
    result.health_score = _compute_health_score(result)
    result.credibility_score = _compute_credibility_score(result)


def _compute_health_score(result: CrawlResult) -> int:
    """Indexability sub-score (0-100) — technical AI-readiness signals only.

    Credibility signals (about/contact/NAP/trust/social/privacy/terms) have
    been moved to _compute_credibility_score.  This function now reflects
    purely technical crawlability/indexability factors.

    Caps:
      - Score cannot exceed 60 if any critical severity issues exist.
      - Score cannot exceed 75 if any high severity issues exist.
      - Score cannot exceed 85 if any medium severity issues exist.
    """
    if result.total_pages == 0:
        return 0

    score = 100

    # Content quality
    if result.answer_capsules == 0:
        score -= 20

    # Schema quality (0-100) replaces binary page-count check
    schema_q = result.schema_quality_score
    if schema_q == 0:
        score -= 15
    elif schema_q < 40:
        score -= 10
    elif schema_q < 70:
        score -= 5

    # Small deductions per thin page (cap at 15)
    score -= min(15, len(result.thin_pages) * 3)

    # AI access
    if result.robots_blocks_ai:
        score -= 20

    # AI discoverability
    if result.has_llms_txt:
        score += 5

    # Technical health
    if result.max_redirect_hops > 1:
        score -= 3
    if result.broken_links:
        score -= min(10, len(result.broken_links) * 2)
    if result.avg_response_ms > 3000:
        score -= 10
    elif result.avg_response_ms > 1500:
        score -= 5

    # Security
    if not result.has_ssl:
        score -= 10

    # Agent readability
    if result.pages_with_landmarks == 0:
        score -= 5
    score += min(5, result.pages_with_landmarks)
    if result.total_images_missing_alt > 10:
        score -= 3

    # ── Apply caps based on issue severity ──
    severities = result.issues_by_severity

    # Cap at 60 if any critical issues
    if severities.get("critical", 0) > 0:
        score = min(score, 60)
    # Cap at 75 if any high issues
    elif severities.get("high", 0) > 0:
        score = min(score, 75)
    # Cap at 85 if any medium issues
    elif severities.get("medium", 0) > 0:
        score = min(score, 85)

    return max(0, min(100, score))


def _compute_credibility_score(result: CrawlResult) -> int:
    """Credibility sub-score (0-100) using the NEEATT rubric.

    Dimensions and weights:
      Notability        15% — media mentions, press, awards, partnerships
      Experience        15% — years in business, project/client counters
      Expertise         20% — authorship, staff bios, credentials, value prop
      Authoritativeness 20% — third-party ratings, reviews, trust signals
      Trustworthiness   15% — NAP, SSL, about/contact pages, corp registration
      Transparency      15% — privacy policy, terms, social links
    """
    # --- Notability (0-100) ---
    notability = 0
    if result.has_media_mentions:
        notability += 70
    if result.has_about_page:
        notability += 30  # about page is entity anchor

    # --- Experience (0-100) ---
    experience = 0
    if result.has_stat_counters:
        experience += 70
    if result.people_with_bios > 0:
        experience += 30

    # --- Expertise (0-100) ---
    expertise = 0
    if result.authorship_pages > 0:
        ratio = result.authorship_pages / max(result.total_pages, 1)
        expertise += int(50 * min(ratio * 2, 1.0))  # up to 50pts at 50%+ coverage
    if result.people_with_credentials > 0:
        expertise += 30
    if result.people_with_bios >= 2:
        expertise += 20

    # --- Authoritativeness (0-100) ---
    authoritativeness = 0
    if result.has_ratings_widget:
        authoritativeness += 50
    if result.has_trust_signals_on_homepage:
        authoritativeness += 50

    # --- Trustworthiness (0-100) ---
    trustworthiness = 0
    if result.has_ssl:
        trustworthiness += 25
    if result.has_contact_info_on_homepage:
        trustworthiness += 25
    if result.has_about_page:
        trustworthiness += 20
    if result.has_contact_page:
        trustworthiness += 15
    if result.has_corp_registration:
        trustworthiness += 15

    # --- Transparency (0-100) ---
    transparency = 0
    if result.has_privacy_policy:
        transparency += 35
    if result.has_terms:
        transparency += 25
    sc = result.social_channel_count
    if sc >= 3:
        transparency += 40
    elif sc >= 1:
        transparency += 20

    # Store sub-scores
    result.neeatt_notability = max(0, min(100, notability))
    result.neeatt_experience = max(0, min(100, experience))
    result.neeatt_expertise = max(0, min(100, expertise))
    result.neeatt_authoritativeness = max(0, min(100, authoritativeness))
    result.neeatt_trustworthiness = max(0, min(100, trustworthiness))
    result.neeatt_transparency = max(0, min(100, transparency))

    # Weighted composite
    composite = (
        0.15 * result.neeatt_notability
        + 0.15 * result.neeatt_experience
        + 0.20 * result.neeatt_expertise
        + 0.20 * result.neeatt_authoritativeness
        + 0.15 * result.neeatt_trustworthiness
        + 0.15 * result.neeatt_transparency
    )
    return max(0, min(100, round(composite)))


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

    # If robots.txt exists but has no explicit AI bot directives at all,
    # all bots are implicitly allowed (no blocks = all green).
    if not bot_decisions and not out["blocks_ai"]:
        out["ai_bots_allowed"] = len(AI_BOT_USER_AGENTS)

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


# ---------------------------------------------------------------------------
# Targeted crawl helpers
# ---------------------------------------------------------------------------


def _fetch_and_analyze(
    session: requests.Session,
    url: str,
    domain: str,
    timeout: int,
    page_type: str,
    result: Optional["CrawlResult"] = None,
) -> tuple:
    """
    Fetch `url` and return (PageSnapshot, BeautifulSoup) or (None, None).

    The soup is returned so the caller can extract nav links for further
    discovery without re-parsing. When `result` is supplied, HTTP/connection
    failures are recorded in ``result.errors`` so access restrictions
    (rate limits, 403s) are surfaced in the report.
    """
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        if not r.ok:
            if result is not None:
                result.errors.append(f"HTTP {r.status_code} on {url}")
            return None, None
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None, None
        snap = _extract_page_signals(
            r.text, r.url,
            size_bytes=len(r.content),
            response_ms=int(r.elapsed.total_seconds() * 1000),
            redirect_hops=len(r.history),
        )
        snap.page_type = page_type
        # Store raw HTML for targeted pages so people extraction can re-parse
        if page_type in ("about", "homepage"):
            snap.raw_html = r.text
        soup = BeautifulSoup(r.text, "lxml")
        return snap, soup
    except requests.RequestException as e:
        if result is not None:
            result.errors.append(f"Request error on {url}: {e}")
        return None, None


def _find_nav_url(soup: BeautifulSoup, domain: str, path_patterns: list) -> Optional[str]:
    """
    Scan all <a> tags for the first link whose path matches one of
    `path_patterns` (exact or ends-with match). Returns an absolute URL or None.
    """
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        if href.startswith("/"):
            path = href.split("?")[0].rstrip("/").lower()
            if any(path == p or path.endswith(p) for p in path_patterns):
                return f"https://{domain}{href.split('?')[0]}"
        else:
            try:
                parsed = urlparse(href)
                if not _is_internal(href, domain):
                    continue
                path = parsed.path.rstrip("/").lower()
                if any(path == p or path.endswith(p) for p in path_patterns):
                    return href.split("?")[0]
            except ValueError:
                continue
    return None


def _find_service_page_url(soup: BeautifulSoup, domain: str, main_keyword: str = "") -> Optional[str]:
    """
    Find the best service/product page from the homepage's nav links.

    Scoring:
      +10  URL path matches a known service-type pattern
      +3×N keyword overlap between nav link text/path and main_keyword words
      -1×depth  prefer top-level pages

    Non-service paths (about, contact, blog, privacy…) are excluded.
    """
    keyword_words: set[str] = set()
    if main_keyword:
        keyword_words = {w for w in re.split(r"\W+", main_keyword.lower()) if len(w) > 2}

    candidates: list[tuple[int, str]] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue

        # Resolve to absolute URL
        if href.startswith("/"):
            abs_url = f"https://{domain}{href.split('?')[0]}"
        elif _is_internal(href, domain):
            abs_url = href.split("?")[0]
        else:
            continue

        path = urlparse(abs_url).path.rstrip("/").lower()
        if not path or path == "/":
            continue
        if _NON_SERVICE_PATH_RE.search(path):
            continue

        score = 0
        if _SERVICE_PATH_RE.search(path):
            score += 10

        if keyword_words:
            path_words = set(re.split(r"[-/_]", path))
            anchor_words = {
                w for w in re.split(r"\W+", a.get_text(" ", strip=True).lower())
                if len(w) > 2
            }
            overlap = len(keyword_words & (path_words | anchor_words))
            score += overlap * 3

        depth = path.count("/")
        score -= depth

        if score > 0:
            candidates.append((score, abs_url))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _derive_main_keyword(soup: BeautifulSoup, domain: str) -> str:
    """
    Derive a rough primary service keyword from homepage H1 or title.
    Used as a hint for service page discovery when no keyword is supplied.
    """
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(" ", strip=True)
        if 5 < len(text) < 100:
            return text
    title_tag = soup.find("title")
    if title_tag:
        text = title_tag.get_text(" ", strip=True)
        for sep in (" | ", " - ", " — ", " :: ", " · "):
            if sep in text:
                text = text.split(sep)[0]
        if 5 < len(text) < 100:
            return text
    return domain.split(".")[0]


_NOTABILITY_PATTERNS = [
    r"\bas\s+seen\s+on\b", r"\bfeatured\s+(?:in|on)\b", r"\bpress\b",
    r"\bmedia\s+(?:coverage|mention|feature)\b", r"\bin\s+the\s+news\b",
    r"\bpublished\s+in\b", r"\brecognized\s+by\b", r"\baward(?:ed|s)?\b",
    r"\bpartner(?:ship)?s?\s+with\b",
]

_EXPERIENCE_PATTERNS = [
    r"\b\d[\d,]*\+?\s*(?:year|yr)s?\s+(?:of\s+)?(?:experience|expertise|in\s+(?:business|industry))\b",
    r"\b\d[\d,]*\+?\s*(?:project|client|customer|case|deal|home|unit|property|contract)s?\b",
    r"\b\d[\d,]*\+?\s*(?:team\s+member|employee|staff|professional)s?\b",
    r"\bsince\s+(?:19|20)\d\d\b",
]

_RATINGS_PATTERNS = [
    r"\btrustpilot\b", r"\bgoogle\s+(?:review|rating)\b",
    r"\b(?:4|5)[\.,]\d\s*(?:out\s+of\s*5|stars?|\/\s*5)\b",
    r"\bverified\s+review", r"\bbbb\s+accredited\b",
    r"\bang(?:ie|i['']s)\s+list\b",
]

_CORP_REGISTRATION_PATTERNS = [
    r"\bein[:\s#]?\s*\d{2}[-\s]\d{7}\b",   # EIN with actual number (not bare "Inc.")
    r"\b(?:company|business)\s+(?:number|no\.?|reg\.?|#)\s*\d+",
    r"\b(?:incorporated|corporation)\b",
    r"\bregistered\s+(?:business|company|in\s+[A-Z][a-z]+)\b",
    r"\bregistration\s+(?:number|no\.?)\b",
    r"\bcompany\s+(?:number|no\.?|reg\.?)\b",
]


def _extract_credibility_signals(soup: BeautifulSoup, url: str) -> dict:
    """
    Extract credibility and trust signals from a page's HTML.

    Returns a dict of booleans:
      has_privacy_link, has_terms_link, has_nap_signals,
      has_trust_signals, has_social_links, social_channel_count,
      has_media_mentions, has_stat_counters, has_ratings_widget,
      has_corp_registration
    """
    all_links = soup.find_all("a", href=True)
    page_text = soup.get_text(" ", strip=True).lower()

    # Privacy policy link
    has_privacy = any(
        "privacy" in (a.get("href", "") + " " + a.get_text()).lower()
        for a in all_links
    )

    # Terms of service / terms & conditions
    has_terms = any(
        any(kw in (a.get("href", "") + " " + a.get_text()).lower()
            for kw in ("terms", "/tos", "conditions", "/legal"))
        for a in all_links
    )

    # Check footer specifically for NAP (that's where it lives most often)
    footer = soup.find("footer")
    footer_text = footer.get_text(" ", strip=True).lower() if footer else ""

    # Email detection
    has_email = bool(re.search(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", page_text))

    # NAP requires phone OR physical address — email alone is not NAP
    has_nap = bool(_PHONE_RE.search(page_text))
    if not has_nap:
        has_nap = bool(re.search(
            r"\b\d{1,5}\s+\w[\w\s]{2,30}"
            r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|court|ct|place|pl)\b",
            page_text,
        ))
    if not has_nap and footer_text:
        has_nap = bool(_PHONE_RE.search(footer_text))

    # Trust signals — scan body content only (exclude nav/header to avoid false positives
    # from link text like "Read reviews →" or "Review our terms")
    _trust_scope = soup.find("main") or soup.find("article") or soup.find("body") or soup
    for _excl_tag in ("nav", "header"):
        for _el in _trust_scope.find_all(_excl_tag):
            _el.decompose()
    _trust_text = _trust_scope.get_text(" ", strip=True).lower()
    _TRUST_PATTERNS = [
        r"\btestimonial", r"\breview[s\s]", r"\bratings?\s+(?:widget|score|badge|\d)",
        r"\b\d+\s*stars?\b",
        r"\bclient\s+(?:story|spotlight|result|success)",
        r"\bcase\s+stud", r"\btrusted\s+by", r"\bused\s+by",
        r"\bcertif(?:ied|ication)", r"\baward(?:ed|s?\s+(?:by|winner|finalist))",
        r"\b\d{1,3}[\s,]\d{3}\+?\s*(?:customer|client|user)",
    ]
    has_trust = any(re.search(p, _trust_text) for p in _TRUST_PATTERNS)

    # Social media links — count distinct platform domains
    _CREDIBILITY_SOCIAL_PLATFORMS = (
        "facebook.com", "twitter.com", "x.com", "linkedin.com",
        "instagram.com", "youtube.com", "tiktok.com", "pinterest.com",
    )
    seen_platforms: set[str] = set()
    for a in all_links:
        href = a.get("href", "")
        for platform in _CREDIBILITY_SOCIAL_PLATFORMS:
            if platform in href:
                seen_platforms.add(platform)
    social_channel_count = len(seen_platforms)
    has_social = social_channel_count >= 1

    # NEEATT: Notability — media mentions / press logos
    has_media_mentions = any(re.search(p, page_text) for p in _NOTABILITY_PATTERNS)

    # NEEATT: Experience — stat counters (quantified track record)
    has_stat_counters = any(re.search(p, page_text) for p in _EXPERIENCE_PATTERNS)

    # NEEATT: Authoritativeness — third-party ratings/reviews widget
    has_ratings_widget = any(re.search(p, page_text) for p in _RATINGS_PATTERNS)

    # NEEATT: Transparency — corporate registration signals
    has_corp_registration = any(re.search(p, page_text) for p in _CORP_REGISTRATION_PATTERNS)

    return {
        "has_privacy_link": has_privacy,
        "has_terms_link": has_terms,
        "has_nap_signals": has_nap,
        "has_trust_signals": has_trust,
        "has_social_links": has_social,
        "social_channel_count": social_channel_count,
        "has_media_mentions": has_media_mentions,
        "has_stat_counters": has_stat_counters,
        "has_ratings_widget": has_ratings_widget,
        "has_corp_registration": has_corp_registration,
    }
