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

    # 3. LLM prompt
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

    # 4. Call LLM
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
                return topics[:n]
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        pass

    return _fallback_topics(domain, n)


def _fallback_topics(domain: str, n: int) -> list[str]:
    """Basic fallback when LLM is unavailable."""
    bare = domain.split(".")[0].replace("-", " ").title()
    return [
        f"best {bare.lower()} alternatives",
        f"{bare} reviews and pricing",
        f"how to choose a solution like {bare}",
        f"{bare} vs competitors",
    ][:n]
