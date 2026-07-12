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
    n: int = 5,
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
    try:
        r = requests.post(
            TOPICGEN_URL,
            headers={
                "Authorization": f"Bearer {key}",
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
            return _fallback_topics(domain, n)

        raw = r.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            topics = json.loads(match.group())
            if isinstance(topics, list) and len(topics) >= 1:
                topics = topics[:n]
            else:
                return _fallback_topics(domain, n)
        else:
            return _fallback_topics(domain, n)
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return _fallback_topics(domain, n)

    # 5. Post-process: replace positions [0] and [3] with keyword-variation queries
    if len(topics) >= 5:
        _inject_keyword_variations(topics, body, key, timeout)

    return topics[:n]


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
                # Slot into positions [0] and [3]
                topics[0] = variations[0]
                topics[3] = variations[1]
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