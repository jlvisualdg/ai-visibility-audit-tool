"""
AI-powered buyer-intent topic generation.

Replaces the heuristic keyword-extraction approach with LLM reverse-engineering:
1. Scrape homepage content
2. Feed to gpt-4o-mini to reverse-engineer the business into 5 unbranded BOTF queries
3. Queries are solution-aware, keyword-driven, and NEVER mention the brand name

CRITICAL RULE: No brand names in queries. No comparisons. The queries must be what
a buyer types BEFORE they know the company exists.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
TOPICGEN_MODEL = os.environ.get("TOPICGEN_MODEL", "openai/gpt-4o-mini").strip()
TOPICGEN_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Public API ──


def generate_botf_topics(
    domain: str,
    homepage_html: Optional[str] = None,
    api_key: Optional[str] = None,
    n: int = 4,
    timeout: int = 30,
) -> list[str]:
    """
    Generate `n` unbranded, bottom-of-funnel buyer queries for a domain.

    If homepage_html is provided, uses that directly. Otherwise scrapes the homepage.
    Falls back to heuristic generation if the LLM call fails.

    Post-processes the result to replace 2 of the 5 queries at positions [0] and [3]
    with keyword-variation queries based on the business's primary service keyword,
    detected via an additional LLM extraction pass.
    """
    key = (api_key or OPENROUTER_API_KEY).strip()
    if not key:
        return _fallback_topics(domain, n)

    # 1. Get homepage content
    if not homepage_html:
        try:
            r = requests.get(
                f"https://{domain}",
                headers={"User-Agent": "AEO-Audit/1.1"},
                timeout=10,
            )
            if r.ok:
                homepage_html = r.text
        except requests.RequestException:
            pass

    if not homepage_html:
        return _fallback_topics(domain, n)

    # 2. Extract clean text
    soup = BeautifulSoup(homepage_html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "svg", "img"]):
        tag.decompose()
    body = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()[:3000]
    title = soup.title.get_text(strip=True) if soup.title else ""
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag:
        meta_desc = meta_tag.get("content", "")

    # 3. LLM prompt for initial 5 queries
    prompt = f"""You are analyzing a business website to generate high-intent, bottom-of-funnel search queries that a ready-to-buy customer would type into ChatGPT or Perplexity.

BUSINESS:
Title: {title}
Meta: {meta_desc}
Homepage: {body}

TASK: Generate {n} queries that would lead a buyer to this company's solution.

MANDATORY RULES — queries will be rejected if they violate ANY of these:
1. NO brand names. NO company names. These are unbranded.
2. USE solution-category keywords: "executive assistant matching service", "remote operator agency", "virtual assistant staffing platform" — not generic pain words like "help" or "workload"
3. HIGH-VOLUME service keywords: target the specific service category (e.g. "virtual chief of staff" or "executive assistant agency"), not vague terms like "top talent" or "skilled operator"
4. COMMERCIAL BOTF language only: "best X for Y", "hire a X", "X service pricing", "X company reviews", "top X agencies 2026", "cost of X", "find a X provider". NO informational queries starting with "how to..." or "what is..."
5. 6-15 words each. Natural buyer language. No fluff.
6. Based strictly on the services this business actually offers.

BAD examples: "how to delegate tasks", "need help managing workload", "where to get matched with top talent"
GOOD examples: "best executive assistant matching service for founders", "top virtual EA agencies for startups 2026", "hire a remote right-hand operator pricing"

Return ONLY a JSON array of {n} strings. Format: ["query1", "query2", ...]"""

    # 4. Call LLM for initial topics
    topics = _call_llm_for_topics(key, prompt, timeout)
    if topics is None:
        return _fallback_topics(domain, n)

    # 5. Validate and retry: reject informational queries starting with
    #    "how to", "cost of", "reviews of", etc.  Up to 2 retries.
    topics = _validate_and_retry_topics(topics, key, prompt, timeout)

    # 6. Post-process: replace positions [0] and [2] with keyword-variation queries
    if len(topics) >= 4:
        _inject_keyword_variations(topics, body, key, timeout)

    return topics[:n]


# ── LLM call helpers ──


def _call_llm_for_topics(api_key: str, prompt: str, timeout: int) -> Optional[list[str]]:
    """Make a single LLM call for topic generation.  Returns list of topics or None."""
    try:
        r = requests.post(
            TOPICGEN_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": TOPICGEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400,
            },
            timeout=timeout,
        )
        if not r.ok:
            return None

        raw = r.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            topics = json.loads(match.group())
            if isinstance(topics, list) and len(topics) >= 1:
                return topics
        return None
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None


# ── Validation: reject informational queries ──

# Patterns that indicate an informational (non-commercial) query.
# Case-insensitive matching against the start of the query string.
_INFORMATIONAL_PATTERNS: list[str] = [
    r"how\s+to\b",
    r"how\s+do\b",
    r"how\s+can\b",
    r"how\s+does\b",
    r"how\s+much\b",
    r"how\s+many\b",
    r"how\s+long\b",
    r"how\s+would\b",
    r"how\s+should\b",
    r"how\s+are\b",
    r"how\s+is\b",
    r"what\s+is\b",
    r"what\s+are\b",
    r"what\s+does\b",
    r"what\s+do\b",
    r"what\s+can\b",
    r"why\s+is\b",
    r"why\s+are\b",
    r"why\s+do\b",
    r"why\s+should\b",
    r"where\s+to\b",
    r"where\s+can\b",
    r"where\s+do\b",
    r"when\s+to\b",
    r"when\s+should\b",
    r"cost\s+of\b",
    r"price\s+of\b",
    r"pricing\s+of\b",
    r"reviews?\s+of\b",
    r"review\s+of\b",
    r"definition\s+of\b",
    r"meaning\s+of\b",
    r"difference\s+between\b",
    r"explain\b",
    r"describe\b",
    r"define\b",
]

_INFORMATIONAL_RE = re.compile(
    r"^\s*(?:" + "|".join(_INFORMATIONAL_PATTERNS) + r")",
    re.IGNORECASE,
)

MAX_VALIDATION_RETRIES = 2


def _count_informational_queries(topics: list[str]) -> int:
    """Return the number of topics that match informational patterns."""
    count = 0
    for t in topics:
        if _INFORMATIONAL_RE.match(t):
            count += 1
    return count


def _validate_and_retry_topics(
    topics: list[str],
    api_key: str,
    prompt: str,
    timeout: int,
) -> list[str]:
    """Validate generated topics and retry if any are informational queries.

    If any topic matches informational patterns (starting with "how to",
    "cost of", "reviews of", etc.), re-call the LLM up to MAX_VALIDATION_RETRIES
    times.  If a retry produces cleaner results, those are used.  Otherwise
    the original topics are returned as-is.
    """
    bad_count = _count_informational_queries(topics)
    if bad_count == 0:
        return topics  # all clean

    for attempt in range(MAX_VALIDATION_RETRIES):
        retry_prompt = (
            prompt
            + f"\n\nCRITICAL: Your previous response contained {bad_count} informational query(s). "
            + "Informational queries (starting with 'how to', 'cost of', 'reviews of', 'what is', "
            + "'why are', 'where to', etc.) are NOT acceptable. "
            + "Generate ONLY commercial BOTF queries using patterns like 'best X for Y', "
            + "'hire a X', 'top X agencies 2026', 'find a X provider', 'X service pricing'. "
            + f"Return ONLY a JSON array of {len(topics)} strings."
        )
        retry = _call_llm_for_topics(api_key, retry_prompt, timeout)
        if retry is None:
            continue  # LLM call failed, try again

        new_bad = _count_informational_queries(retry)
        if new_bad < bad_count:
            topics = retry
            bad_count = new_bad
            if bad_count == 0:
                return topics  # all clean now

    return topics  # return best we have (original or improved)


def _inject_keyword_variations(
    topics: list[str],
    body_text: str,
    api_key: str,
    timeout: int,
) -> None:
    """
    Post-process topics list in-place: extract the business's primary service
    keyword from the homepage text, generate 2 variation queries, and slot them
    into positions [0] and [3] of the topics list.

    Silently returns without modification if any step fails (LLM unavailable,
    malformed response, etc.).
    """
    # Step A: Extract the primary service keyword
    kw_prompt = (
        "Extract the single primary service keyword from this business homepage text.\n"
        "This should be the main 2-6 word phrase describing what service/product the business sells.\n"
        'Example: for "virtual executive assistant matching for founders" → "executive assistant matching"\n'
        'Example: for "AI-powered chatbot platform for ecommerce" → "AI chatbot platform"\n'
        "Return ONLY the keyword phrase, nothing else — no quotes, no explanation.\n\n"
        f"Homepage excerpt:\n{body_text[:1500]}"
    )

    try:
        kw_r = requests.post(
            TOPICGEN_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": TOPICGEN_MODEL,
                "messages": [{"role": "user", "content": kw_prompt}],
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=timeout,
        )
        if not kw_r.ok:
            return
        main_kw = (
            kw_r.json()["choices"][0]["message"]["content"]
            .strip()
            .strip('"')
            .strip("'")
        )
        if not main_kw:
            return
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return

    # Step B: Generate 2 variation queries from the main keyword
    var_prompt = (
        f'Generate exactly 2 different bottom-of-funnel search queries built around the keyword "{main_kw}".\n'
        "Must be 6-15 words each, unbranded, natural commercial/buyer language.\n"
        'Use BOTF phrasing: "best X for Y", "hire a X service", "top X agencies 2026", "X service pricing", '
        '"cost of X for business", "find a X provider".\n'
        "The two queries should use DIFFERENT phrasings and angles — do not just swap one word.\n"
        "Do NOT include the brand name or any specific company name.\n"
        'Return ONLY a JSON array of 2 strings. Example: ["query one here", "query two here"]'
    )

    try:
        var_r = requests.post(
            TOPICGEN_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": TOPICGEN_MODEL,
                "messages": [{"role": "user", "content": var_prompt}],
                "temperature": 0.5,
                "max_tokens": 150,
            },
            timeout=timeout,
        )
        if not var_r.ok:
            return

        var_raw = var_r.json()["choices"][0]["message"]["content"].strip()
        var_match = re.search(r"\[.*?\]", var_raw, re.DOTALL)
        if var_match:
            variations = json.loads(var_match.group())
            if isinstance(variations, list) and len(variations) >= 2:
                # Slot into positions [0] and [2]
                topics[0] = variations[0]
                topics[2] = variations[1]
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return


def _fallback_topics(domain: str, n: int) -> list[str]:
    """Basic fallback when LLM is unavailable."""
    bare = domain.split(".")[0].replace("-", " ").title()
    return [
        f"best {bare.lower()} alternatives",
        f"{bare} reviews and pricing",
        f"how to choose a solution like {bare}",
        f"{bare} vs competitors",
    ][:n]