"""
LLM-based brand entity extraction from AI engine responses.

Replaces regex/heuristic capitalized-phrase extraction with a semantic pass
that identifies the actual companies / brands / products a response recommends,
attaches each to its cited URL when the answer provides one, and works for ANY
business vertical with ZERO hardcoded term lists.

Why LLM instead of regex: distinguishing a real brand ("Miller & Zois") from a
capitalized category label ("Case Expenses", "Standard Rate", "Court") cannot be
done reliably by pattern matching without a per-industry blocklist. A language
model knows the difference natively, for every vertical.

Contract mirrors scoring.extract_brand_mentions() so it is a drop-in upgrade:
    extract_brands_llm(...) -> (brands, positions, brand_urls)
      brands      : ordered, deduped list of brand names (first appearance)
      positions   : 1-indexed positions of the TARGET brand within `brands`
      brand_urls  : {brand_name: bare_domain} for brands the answer linked/cited

Falls back to the deterministic regex extractor when the LLM key is missing or
any step fails, so an audit never hard-crashes on extraction.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

from src.scoring import (
    extract_brand_mentions,
    _extract_target_tokens,
    _find_target_positions,
)

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
BRAND_EXTRACT_MODEL = os.environ.get(
    "BRAND_EXTRACT_MODEL", "openai/gpt-4o-mini"
).strip()
BRAND_EXTRACT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Bare-domain regex for pulling domains written inline in the response text
# (engines don't always list a brand's own site in their formal citations).
_TEXT_DOMAIN_RE = re.compile(r"\b((?:[a-z0-9-]+\.)+[a-z]{2,})\b", re.IGNORECASE)

# gpt-4o-mini pricing: $0.15/M input, $0.60/M output
_IN_PRICE = 0.15
_OUT_PRICE = 0.60
_brand_extract_cost_usd: float = 0.0


def get_brand_extract_cost() -> float:
    return _brand_extract_cost_usd


def reset_brand_extract_cost() -> None:
    global _brand_extract_cost_usd
    _brand_extract_cost_usd = 0.0


def _accumulate_cost(response_json: dict) -> None:
    global _brand_extract_cost_usd
    usage = response_json.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    _brand_extract_cost_usd += (in_tok * _IN_PRICE + out_tok * _OUT_PRICE) / 1_000_000


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """An AI assistant answered a buyer's commercial question. Extract every SPECIFIC company, brand, product, or service provider that the answer presents as a recommendation or option — in order of first appearance.

STRICT RULES:
- Return ONLY real named entities: proper nouns that name an actual business, product, or provider a buyer could choose.
- DO NOT return generic category labels, section headings, descriptions, features, or abstract concepts. Things to EXCLUDE (never brands, regardless of industry): pricing/fee types, service categories, document or process names, generic role/place words, evaluation criteria, and any phrase that describes a KIND of thing rather than naming a specific company.
- Write each name EXACTLY as it appears (keep "&", punctuation, capitalization).
- If the answer links or attributes a website/domain to an entity, put that domain in "url"; otherwise use null.
- Deduplicate: list each entity once, at its first position.

RESPONSE TEXT:
<<<
{text}
>>>

DOMAINS THE ENGINE CITED AS SOURCES (use to attach a url; NOT every domain is a brand):
{citations}

Return ONLY a JSON array, no prose, no markdown fences:
[{{"name": "Example Co", "url": "example.com"}}, {{"name": "Other Brand", "url": null}}]"""


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------


def _domain_matches_brand(brand: str, domain: str) -> bool:
    """True if `domain` plausibly belongs to `brand` (brand name is in the host).

    Guarantees a brand→URL link is the brand's OWN site (e.g. "Trilogy" ->
    trilogyproducts.com), not a third-party article that merely mentions it
    (e.g. "CeraVe" -> skinhealthfoundation.org). Generic, no hardcoding.
    """
    from src.scoring import _brand_variants
    dom = _bare_domain(domain) or domain
    dn = re.sub(r"[^a-z0-9]", "", dom.lower())
    if not dn:
        return False
    # Match if any connector variant of the brand (with/without "and") of
    # length >= 4 appears in the domain host.
    return any(len(v) >= 4 and v in dn for v in _brand_variants(brand))


def _augment_urls_from_citations(
    brands: list[str],
    brand_urls: dict[str, str],
    citations: list[str],
) -> dict[str, str]:
    """Tie recommended brands to domains that appear in the engine's citations.

    The LLM attaches a URL only when the answer text makes the link obvious. But
    engines also return a separate citations/annotations list, and a brand's own
    domain often appears there even when the prose didn't spell it out (e.g. the
    brand "Trilogy" alongside a cited "trilogyproducts.com"). For any brand still
    missing a URL, we match its normalized name against each citation domain.

    We only ever link to domains ACTUALLY present in the citations — never guess
    a domain that isn't there. Fully generic across verticals.
    """
    if not citations:
        return brand_urls

    cite_domains = [d for d in (_bare_domain(c) for c in citations) if d]

    for b in brands:
        if brand_urls.get(b):
            continue
        for dom in cite_domains:
            # Link only when the brand name appears in the citation domain
            # ("trilogy" in "trilogyproducts.com") — its own site, not a source.
            if _domain_matches_brand(b, dom):
                brand_urls[b] = dom
                break

    return brand_urls


def _bare_domain(url: Optional[str]) -> Optional[str]:
    """Reduce a URL or domain string to its bare host (no scheme/www/path)."""
    if not url or not isinstance(url, str):
        return None
    u = url.strip().lower()
    if not u or u in {"null", "none", "n/a"}:
        return None
    for proto in ("https://", "http://"):
        if u.startswith(proto):
            u = u[len(proto):]
    if u.startswith("www."):
        u = u[4:]
    u = u.split("/")[0].split("?")[0].strip()
    # Must look like a domain
    if "." not in u or " " in u:
        return None
    return u or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_brands_llm(
    text: str,
    citations: Optional[list[str]] = None,
    target_domain: str = "",
    target_brand_name: str = "",
    *,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[list[str], list[int], dict[str, str]]:
    """Extract recommended brands from an AI response via an LLM pass.

    Returns (brands, positions, brand_urls). On any failure, falls back to the
    deterministic regex extractor (with an empty url map).
    """
    citations = citations or []

    if not text or not text.strip():
        return ([], [], {})

    key = (api_key or OPENROUTER_API_KEY).strip()
    if not key:
        return _regex_fallback(text, target_domain, target_brand_name)

    prompt = _PROMPT_TEMPLATE.format(
        text=text[:6000],
        citations=", ".join(citations[:25]) if citations else "(none provided)",
    )

    try:
        r = requests.post(
            BRAND_EXTRACT_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": BRAND_EXTRACT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 600,
            },
            timeout=timeout,
        )
        if not r.ok:
            return _regex_fallback(text, target_domain, target_brand_name)
        r.encoding = "utf-8"
        response_json = r.json()
        _accumulate_cost(response_json)
        raw = response_json["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return _regex_fallback(text, target_domain, target_brand_name)

    entities = _parse_entities(raw)
    if entities is None:
        return _regex_fallback(text, target_domain, target_brand_name)

    # Build ordered, deduped brand list + url map
    brands: list[str] = []
    brand_urls: dict[str, str] = {}
    seen: set[str] = set()
    for ent in entities:
        name = (ent.get("name") or "").strip()
        if not name or not (2 <= len(name) <= 60):
            continue
        key_norm = name.lower()
        if key_norm in seen:
            continue
        seen.add(key_norm)
        brands.append(name)
        dom = _bare_domain(ent.get("url"))
        # Only keep the LLM's URL if it's actually the brand's own domain —
        # not a third-party source that merely mentions the brand.
        if dom and _domain_matches_brand(name, dom):
            brand_urls[name] = dom

    # Fill in any brand→domain links still missing by matching brand names
    # against every domain we can see — the engine's formal citations PLUS any
    # domains written inline in the response text (their own site only).
    domain_pool = list(citations) + _TEXT_DOMAIN_RE.findall(text)
    brand_urls = _augment_urls_from_citations(brands, brand_urls, domain_pool)

    # Compute target positions. Include the website-derived brand name as a
    # token so an AI response's shorter form ("Pinder Plotkin") still matches
    # the site's "Pinder Plotkin LLC" via fuzzy normalization.
    target_tokens = _extract_target_tokens(target_domain)
    if target_brand_name and target_brand_name not in target_tokens:
        target_tokens.append(target_brand_name)
    positions = _find_target_positions(brands, target_tokens)

    return (brands, positions, brand_urls)


def _parse_entities(raw: str) -> Optional[list[dict]]:
    """Parse the model's JSON array. Returns None if unparseable."""
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    # Keep only dict entries with a name
    return [d for d in data if isinstance(d, dict) and d.get("name")]


def _regex_fallback(
    text: str,
    target_domain: str,
    target_brand_name: str,
) -> Tuple[list[str], list[int], dict[str, str]]:
    """Deterministic fallback using the legacy regex extractor."""
    try:
        brands, positions = extract_brand_mentions(text, target_domain, target_brand_name)
        return (brands, positions, {})
    except Exception:
        return ([], [], {})
