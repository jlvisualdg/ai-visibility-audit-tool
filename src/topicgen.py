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

# gpt-4o-mini pricing: $0.15/M input, $0.60/M output
_TOPICGEN_IN_PRICE = 0.15
_TOPICGEN_OUT_PRICE = 0.60
_topicgen_cost_usd: float = 0.0


def get_topicgen_cost() -> float:
    return _topicgen_cost_usd


def reset_topicgen_cost() -> None:
    global _topicgen_cost_usd
    _topicgen_cost_usd = 0.0


def _accumulate_cost(response_json: dict) -> None:
    global _topicgen_cost_usd
    usage = response_json.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    _topicgen_cost_usd += (in_tok * _TOPICGEN_IN_PRICE + out_tok * _TOPICGEN_OUT_PRICE) / 1_000_000

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
        return _fallback_topics(domain, n, homepage_html, key)

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
        return _fallback_topics(domain, n, "", key)

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

    # 3. LLM prompt for initial queries
    #
    # This is a SIMULATION: we reverse-engineer the business's core service from
    # its scraped homepage, then generate commercial-intent queries engineered so
    # an AI engine responds by RECOMMENDING A LIST OF SPECIFIC COMPANIES/BRANDS.
    # Every query must be the kind that produces a ranked roster of named vendors —
    # that is the only signal the audit can measure. Pricing/cost and informational
    # queries make engines explain concepts instead of naming brands, so they are banned.
    prompt = f"""You are reverse-engineering a business from its website to run a brand-visibility simulation.

BUSINESS:
Title: {title}
Meta: {meta_desc}
Homepage: {body}

GOAL: First identify this business's ONE core service or product category. Then generate {n} search queries that a ready-to-buy customer would type into ChatGPT or Perplexity — queries engineered so the AI answers by RECOMMENDING A LIST OF SPECIFIC COMPANIES, BRANDS, OR PROVIDERS in that category.

MANDATORY RULES — a query is rejected if it violates ANY of these:
1. NO brand names, NO company names. Queries are unbranded (the buyer does not yet know any vendor).
2. Must reliably elicit a RANKED LIST OF NAMED VENDORS. Use recommendation phrasing:
   "best X for Y", "top X companies/agencies/providers 2026", "leading X for Y",
   "most recommended X services", "which X is best for Y", "find the best X provider for Y".
3. Anchor on the specific SERVICE CATEGORY keyword (e.g. "executive assistant matching service",
   "personal injury law firm", "managed IT provider") — never vague pain words ("help", "support").
4. STRICTLY BANNED — these do NOT produce brand lists:
   - Pricing/cost queries ("X pricing", "cost of X", "how much does X cost", "X rates").
   - Informational queries ("how to...", "what is...", "why...", "guide to...").
5. 6-15 words each. Natural commercial buyer language. No fluff.
6. Based strictly on the core service this business actually offers.

BAD (rejected): "how to delegate tasks", "executive assistant service pricing", "cost of hiring a VA", "what is a virtual assistant"
GOOD (accepted): "best executive assistant matching service for founders", "top virtual EA agencies for startups 2026", "leading remote chief of staff providers for CEOs", "most recommended executive assistant firms for busy founders"

Return ONLY a JSON array of {n} strings. Format: ["query1", "query2", ...]"""

    # 4. Call LLM for initial topics
    topics = _call_llm_for_topics(key, prompt, timeout)
    if topics is None:
        return _fallback_topics(domain, n, homepage_html, key)

    # 5. Validate and retry: reject informational queries starting with
    #    "how to", "cost of", "reviews of", etc.  Up to 2 retries.
    topics = _validate_and_retry_topics(topics, key, prompt, timeout)

    # 6. Post-process: replace positions [0] and [2] with keyword-variation queries
    if len(topics) >= 4:
        _inject_keyword_variations(topics, body, key, timeout)

    # 7. Hard post-filter: remove any remaining informational queries
    topics = _hard_filter_informational(topics, n)

    # 8. Location injection: if local business, append city to all queries
    city = _extract_city(title, body[:500], key, timeout)
    if city:
        topics = _inject_location(topics, city)

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

        response_json = r.json()
        _accumulate_cost(response_json)
        raw = (response_json["choices"][0]["message"].get("content") or "").strip()
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            topics = json.loads(match.group())
            if isinstance(topics, list) and len(topics) >= 1:
                return topics
        return None
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
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

# Pricing/cost phrasing anywhere in a query — these make engines explain fee
# structures instead of naming brands, so they never yield a brand recommendation.
_PRICING_RE = re.compile(
    r"\b(pricing|price|prices|cost|costs|rate|rates|fee|fees|cheap|"
    r"cheapest|affordable|how\s+much|per\s+hour|per\s+month)\b",
    re.IGNORECASE,
)


def _is_brand_eliciting(query: str) -> bool:
    """A query is brand-eliciting only if it is neither informational nor pricing-oriented."""
    return not _INFORMATIONAL_RE.match(query) and not _PRICING_RE.search(query)


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
        kw_json = kw_r.json()
        _accumulate_cost(kw_json)
        main_kw = (
            (kw_json["choices"][0]["message"].get("content") or "")
            .strip()
            .strip('"')
            .strip("'")
        )
        if not main_kw:
            return
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        return

    # Step B: Generate 2 variation queries from the main keyword.
    # These MUST elicit a list of named brands — no pricing/cost phrasing (which
    # makes engines explain fees instead of recommending companies).
    var_prompt = (
        f'Generate exactly 2 different commercial search queries built around the keyword "{main_kw}".\n'
        "Must be 6-15 words each, unbranded, natural commercial/buyer language.\n"
        "Each query must be phrased so an AI engine answers by RECOMMENDING A LIST OF SPECIFIC "
        "COMPANIES/BRANDS. Use phrasing like: \"best X for Y\", \"top X companies/agencies 2026\", "
        '"leading X providers for Y", "most recommended X services", "which X is best for Y".\n'
        "BANNED: pricing/cost queries and informational (how/what/why) queries — they do not yield brand lists.\n"
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

        var_json = var_r.json()
        _accumulate_cost(var_json)
        var_raw = (var_json["choices"][0]["message"].get("content") or "").strip()
        var_match = re.search(r"\[.*?\]", var_raw, re.DOTALL)
        if var_match:
            variations = json.loads(var_match.group())
            if isinstance(variations, list) and len(variations) >= 2:
                # Slot into positions [0] and [2]
                topics[0] = variations[0]
                topics[2] = variations[1]
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        return


def _hard_filter_informational(topics: list[str], n: int) -> list[str]:
    """Replace any query that won't elicit a brand list (informational OR pricing).

    Runs after LLM validation retries as a hard backstop. Non-brand-eliciting
    entries are swapped for generic ranking-style placeholders so we always
    return n brand-eliciting topics.
    """
    _GENERIC_BOTF = [
        "best service providers in this category 2026",
        "top rated companies in this industry",
        "most recommended providers for this type of service",
        "leading agencies to hire in this space",
    ]
    result = []
    generic_idx = 0
    for t in topics:
        if not _is_brand_eliciting(t):
            # Replace with a generic ranking-style query
            result.append(_GENERIC_BOTF[generic_idx % len(_GENERIC_BOTF)])
            generic_idx += 1
        else:
            result.append(t)
    return result[:n]


def _extract_city(title: str, body_excerpt: str, api_key: str, timeout: int) -> Optional[str]:
    """Extract the primary city this business serves, if it's a local business.

    Returns the city name (e.g. 'Baltimore') or None for national/global brands.
    City takes priority over state per style guide.
    """
    if not api_key:
        return None
    prompt = (
        "If this is a LOCAL business serving a specific city, return ONLY the city name "
        "(e.g. 'Baltimore', 'Chicago', 'Miami'). "
        "Return NOTHING (empty string) if this is a national brand, online-only, or serves broadly.\n\n"
        f"Title: {title}\nText: {body_excerpt}"
    )
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
                "temperature": 0.0,
                "max_tokens": 20,
            },
            timeout=timeout,
        )
        if not r.ok:
            return None
        resp = r.json()
        _accumulate_cost(resp)
        city = (resp["choices"][0]["message"].get("content") or "").strip().strip('"').strip("'").strip(".")
        # Sanity check: should be 1-3 words, no punctuation beyond comma
        if city and 2 <= len(city) <= 30 and "\n" not in city:
            # Strip trailing state or country additions (e.g. "Baltimore, MD" → "Baltimore")
            city = city.split(",")[0].strip()
            return city if city else None
        return None
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        return None


def _inject_location(topics: list[str], city: str) -> list[str]:
    """Append 'in [city]' to each topic that doesn't already mention the city."""
    city_lower = city.lower()
    result = []
    for t in topics:
        if city_lower in t.lower():
            result.append(t)
        else:
            result.append(f"{t} in {city}")
    return result


def _fallback_topics(domain: str, n: int, homepage_html: str = "", api_key: str = "") -> list[str]:
    """Fallback when LLM is unavailable.

    Generates UNBRANDED commercial BOTF queries using service keywords
    extracted from the homepage HTML. NEVER uses the brand/domain name.
    """
    # Try to extract service keywords from homepage
    service_keyword = ""
    if homepage_html:
        try:
            soup = BeautifulSoup(homepage_html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "svg", "img"]):
                tag.decompose()
            body = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()[:2000]
            title = soup.title.get_text(strip=True) if soup.title else ""
            meta_desc = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag:
                meta_desc = meta_tag.get("content", "")

            # Use LLM to extract the service keyword even in fallback
            key = (api_key or OPENROUTER_API_KEY).strip()
            if key:
                kw_prompt = (
                    "Extract the single primary service keyword from this business homepage.\n"
                    "This should be the 2-6 word phrase describing what the business sells.\n"
                    'Return ONLY the keyword phrase, nothing else.\n\n'
                    f"Title: {title}\nMeta: {meta_desc}\nHomepage: {body[:1000]}"
                )
                try:
                    kw_r = requests.post(
                        TOPICGEN_URL,
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": TOPICGEN_MODEL,
                            "messages": [{"role": "user", "content": kw_prompt}],
                            "temperature": 0.0,
                            "max_tokens": 50,
                        },
                        timeout=15,
                    )
                    if kw_r.ok:
                        service_keyword = (
                            (kw_r.json()["choices"][0]["message"].get("content") or "")
                            .strip().strip('"').strip("'").strip(".")
                        )
                except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
                    pass
        except Exception:
            pass

    # If we got a service keyword, build unbranded brand-eliciting queries around it
    if service_keyword and len(service_keyword) > 2:
        kw = service_keyword.lower()
        return [
            f"best {kw} companies for businesses 2026",
            f"top {kw} providers to consider",
            f"most recommended {kw} services",
            f"leading {kw} agencies to hire",
        ][:n]

    # Last resort: generic commercial queries (still unbranded, brand-eliciting)
    return [
        "best service providers in this category 2026",
        "top rated companies in this industry",
        "most recommended providers for this type of service",
        "leading agencies to hire in this space",
    ][:n]