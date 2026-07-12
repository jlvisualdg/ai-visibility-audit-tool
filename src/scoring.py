"""
Brand mention extraction with position tracking.

Extracts multi-word capitalized brand names from AI engine response text,
filters false positives, deduplicates while preserving order, and tracks
where the target brand appears.

Returns:
    (unique_brands, target_positions) where:
    - unique_brands: deduplicated brand names in order of first appearance
    - target_positions: 1-indexed occurrence positions of the target brand
      among ALL brand mentions (before dedup), in order of appearance
"""

from __future__ import annotations

import re
from typing import Tuple


# ---------------------------------------------------------------------------
# Stop words — lowercase. A multi-word phrase where every word is a stop
# word is treated as a false positive and filtered out.
# ---------------------------------------------------------------------------

STOP_WORDS: set[str] = {
    "a", "an", "the",
    "and", "but", "or", "for", "nor", "so", "yet",
    "in", "on", "at", "by", "to", "of", "with", "from", "as",
    "it", "is", "be", "are", "was", "were", "been", "has", "had", "have",
    "we", "he", "she", "they", "you",
    "this", "that", "these", "those",
    "its", "his", "her", "our", "your", "my", "their",
    "can", "may", "will", "would", "could", "should",
    "do", "does", "did", "not", "no",
    "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only",
    "if", "then", "than", "also", "just", "now", "up", "out",
    "when", "where", "which", "who", "whom", "whose", "what",
    "how", "why", "very", "too", "any",
}

# Additional phrases known to be false positives even when capitalized
# (e.g., common English phrases that happen to be multi-word capitalized)
FALSE_POSITIVE_PHRASES: set[str] = {
    "new york", "los angeles", "san francisco", "las vegas",
    "united states", "united kingdom", "south africa", "north america",
    "european union", "middle east", "south east",
}


# ---------------------------------------------------------------------------
# Regex: multi-word capitalized brand name
# ---------------------------------------------------------------------------

# Multi-word capitalized phrases joined by plain spaces (no "and"/"&").
# "Acme Corp", "Pareto Talent", "Global Tech Solutions" — these are
# individual brands.  "and" / "&" between capitalized phrases act as
# list separators, not intra-brand connectors (e.g. "Acme Corp and
# Global Tech" should produce TWO brands, not one).
#
# Uses a negative lookahead per space to reject " and " / " & " bridges.
PLAIN_BRAND = re.compile(
    r"\b([A-Z][a-z]+"             # first capitalized word (2+ chars)
    r"(?:"                        # one or more additional words
    r"\s+(?!and\b|&)"             #   space NOT followed by "and" or "&"
    r"[A-Z][a-z]+"                #   capitalized word (2+ chars)
    r")+"
    r")",
)

# "&" / "and" compound brands — names where a connector is integral
# to the brand identity.  "Johnson & Johnson", "Procter & Gamble",
# "Procter and Gamble".  These are detected separately so they aren't
# broken into pieces.
CONNECTOR_BRAND = re.compile(
    r"\b([A-Z][a-z]+"             # first word (must be capitalized)
    r"\s+(?i:&|and)\s+"           # connector — case-insensitive
    r"[A-Z][a-z]+"                # second word (must be capitalized)
    r")",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_brand_mentions(
    response_text: str,
    target_domain: str,
) -> Tuple[list[str], list[int]]:
    """Extract brand names and track target brand positions.

    Scans `response_text` for multi-word capitalized phrases that look like
    brand names (e.g. "Pareto Talent", "Acme Corp", "Johnson & Johnson"),
    filters out false positives, and returns deduplicated brands in order
    of first appearance plus the 1-indexed occurrence positions of the
    target brand.

    Target matching: if `target_domain` is "paretotalent.com", we look for
    "paretotalent" or "pareto talent" (case-insensitive) anywhere in the
    response text. Each occurrence's position among all brand mentions
    (before deduplication) is recorded.

    Args:
        response_text: The AI engine response text to scan.
        target_domain: Domain being audited (e.g. "paretotalent.com").

    Returns:
        (unique_brands, target_positions) tuple.

    Examples:
        >>> extract_brand_mentions("Acme Corp and Global Tech lead the market.", "globaltech.com")
        (['Acme Corp', 'Global Tech'], [2])

        >>> extract_brand_mentions("Pareto Talent was cited. Acme Corp also. Pareto Talent again.", "paretotalent.com")
        (['Pareto Talent', 'Acme Corp'], [1, 3])
    """
    if not response_text or not response_text.strip():
        return ([], [])

    # ----- Step 1: find all multi-word capitalized candidates -----
    raw_matches: list[tuple[int, int, str]] = []  # (start, end, phrase)

    # Plain space-separated brands (e.g. "Acme Corp", "Global Tech")
    for m in PLAIN_BRAND.finditer(response_text):
        raw_matches.append((m.start(), m.end(), m.group(1)))

    # Connector brands (e.g. "Johnson & Johnson", "Procter and Gamble")
    # Guard: skip connector matches whose start falls inside a plain-brand
    # span — these are fragments like "Corp and Global" from
    # "Acme Corp and Global Tech", not real compound brands.
    plain_spans = [(s, e) for s, e, _ in raw_matches]
    for m in CONNECTOR_BRAND.finditer(response_text):
        if not any(s <= m.start() < e for s, e in plain_spans):
            raw_matches.append((m.start(), m.end(), m.group(1)))

    # Sort by position in text to preserve order
    raw_matches.sort(key=lambda x: x[0])
    phrases = [p for _, _, p in raw_matches]

    # ----- Step 2: filter false positives -----
    filtered: list[str] = []
    for phrase in phrases:
        if _is_plausible_brand(phrase):
            filtered.append(phrase)

    # ----- Step 3: deduplicate while preserving order -----
    seen: set[str] = set()
    unique_brands: list[str] = []
    for phrase in filtered:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            unique_brands.append(phrase)

    # ----- Step 4: find target brand positions -----
    # Extract target search tokens from the domain
    target_tokens = _extract_target_tokens(target_domain)
    target_positions = _find_target_positions(filtered, target_tokens)

    return (unique_brands, target_positions)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_plausible_brand(phrase: str) -> bool:
    """Return True if `phrase` looks like a real brand, not a false positive.

    Filters:
      - Phrases where every word is a stop word (e.g. "For The").
      - Known false-positive geographic / generic phrases.
      - Phrases ≤ 5 characters total (too short to be meaningful).
    """
    words = phrase.split()
    if len(phrase) <= 5:
        return False
    if all(w.lower() in STOP_WORDS for w in words):
        return False
    if phrase.lower() in FALSE_POSITIVE_PHRASES:
        return False
    # At least one word must be a "content" word (not in stop-words)
    if not any(w.lower() not in STOP_WORDS for w in words):
        return False
    return True


def _extract_target_tokens(domain: str) -> list[str]:
    """Extract search tokens from a domain for case-insensitive matching.

    "paretotalent.com" -> ["paretotalent"]
    "acme-corp.com"    -> ["acmecorp", "acme corp"]
    "acmecorp.com"     -> ["acmecorp"]

    Also includes common substrings of the bare domain when the domain
    is a compound (e.g. "pareto" from "paretotalent").
    """
    if not domain:
        return []

    bare = _normalize_domain(domain).split(".")[0]
    tokens = [bare.lower()]

    # If the bare domain has hyphens, add a space-separated variant
    if "-" in bare:
        tokens.append(bare.replace("-", " ").lower())

    # For compound domains like "paretotalent", also try extracting
    # common substrings (first part of a known split)
    # We try a heuristic: look for common suffixes like "talent", "roofing",
    # "corp", "inc", "group", "solutions", "tech", "software", "media",
    # "consulting", "capital", "partners", "ventures", "labs", "studios"
    common_suffixes = [
        "talent", "roofing", "corp", "inc", "group", "solutions", "tech",
        "software", "media", "consulting", "capital", "partners", "ventures",
        "labs", "studios", "agency", "creative", "marketing", "digital",
        "design", "studio", "systems", "cloud", "data", "health", "care",
        "finance", "legal", "education", "energy", "food", "travel",
    ]
    for suffix in common_suffixes:
        if bare.lower().endswith(suffix) and len(bare) > len(suffix) + 1:
            prefix = bare[:-len(suffix)]
            tokens.append(f"{prefix} {suffix}".lower())
            # Also add just the prefix as a search token
            if len(prefix) >= 3:
                tokens.append(prefix.lower())
            break  # only try the first matching suffix

    return tokens


def _find_target_positions(
    brands: list[str],
    target_tokens: list[str],
) -> list[int]:
    """Find 1-indexed positions where the target brand appears in `brands`.

    Both the brand name and target tokens are normalized before comparison:
    "&" / "and" connectors are collapsed to spaces so that "Procter & Gamble"
    matches a token of "procter gamble" from domain "procter-gamble.com".
    """
    if not target_tokens:
        return []

    positions: list[int] = []
    for i, brand in enumerate(brands, start=1):
        brand_norm = _normalize_brand_for_match(brand)
        for token in target_tokens:
            token_norm = _normalize_brand_for_match(token)
            if token_norm in brand_norm or brand_norm in token_norm:
                positions.append(i)
                break  # each brand phrase counts at most once per mention
    return positions


def _normalize_brand_for_match(s: str) -> str:
    """Normalize a brand name or token for comparison.

    - Lowercase
    - Collapse " & " and " and " to a single space
    - Strip extra whitespace
    """
    s = s.lower().strip()
    s = re.sub(r"\s*(?:&|and)\s*", " ", s)
    return " ".join(s.split())  # collapse multiple spaces


def _normalize_domain(d: str) -> str:
    """Strip scheme, path, and www. prefix from a domain string."""
    d = (d or "").strip().lower()
    for prefix in ("https://", "http://"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.split("/")[0]
    if d.startswith("www."):
        d = d[4:]
    return d


# ---------------------------------------------------------------------------
# Scoring — AI Presence Score
# ---------------------------------------------------------------------------

import statistics
from typing import Optional

# Adjustable constants
MAX_MENTION_MULTIPLIER: int = 3
"""Maximum mention multiplier.  3 means mentions are capped at 3×."""

MIN_BRAND_LENGTH: int = 3
"""Minimum character length for a string to be considered a brand name."""


def calculate_ai_presence_score(
    mention_count: int,
    first_position: Optional[int],
    total_brands: int,
    *,
    max_mention_multiplier: int = MAX_MENTION_MULTIPLIER,
) -> float:
    """
    Calculate the AI Presence score for a single query.

    Formula
    -------
    Position_Score  = (total_brands - first_position + 1) / total_brands * 100
    Mention_Multiplier = min(mention_count, max_mention_multiplier) / max_mention_multiplier
    Query_Score     = min(Position_Score × Mention_Multiplier, 100)

    Parameters
    ----------
    mention_count : int
        How many times the target brand was mentioned in the response.
    first_position : int or None
        1-indexed position of the target brand among all brands (1 = first).
        None means the brand was not found — score will be 0.0.
    total_brands : int
        Total distinct brands identified in the response.
    max_mention_multiplier : int
        Override for the mention cap (default MAX_MENTION_MULTIPLIER).

    Returns
    -------
    float
        Score in [0, 100].
    """
    if total_brands <= 0 or first_position is None:
        return 0.0

    position_score = (total_brands - first_position + 1) / total_brands * 100.0
    multiplier = min(mention_count, max_mention_multiplier) / max_mention_multiplier
    raw = position_score * multiplier
    return min(raw, 100.0)


def calculate_overall_ai_presence(query_scores: list[float]) -> float:
    """
    Aggregate per-query scores into the Overall AI Presence metric.

    Overall_AI_Presence = mean(all Query_Scores)

    Parameters
    ----------
    query_scores : list[float]
        One calculate_ai_presence_score result per query.

    Returns
    -------
    float
        Overall AI Presence in [0, 100].
    """
    if not query_scores:
        return 0.0
    return statistics.mean(query_scores)


# ---------------------------------------------------------------------------
# Aggregate results — end-to-end audit summary
# ---------------------------------------------------------------------------


def _is_target_brand(brand: str, target_tokens: list[str]) -> bool:
    """Return True if *brand* matches the target domain tokens."""
    if not target_tokens:
        return False
    brand_norm = _normalize_brand_for_match(brand)
    if not brand_norm:
        return False
    for token in target_tokens:
        token_norm = _normalize_brand_for_match(token)
        if not token_norm:
            continue
        if token_norm in brand_norm or brand_norm in token_norm:
            return True
    return False


def _domain_contains_citation(citation: str, target_domain: str) -> bool:
    """Return True if *citation* contains the normalized target domain."""
    norm_domain = _normalize_domain(target_domain)
    norm_cite = citation.lower().strip()
    bare = norm_domain.split(".")[0]
    return bare in norm_cite or norm_domain in norm_cite


def aggregate_results(results: list[dict], target_domain: str) -> dict:
    """Compute the end-to-end audit summary from a flat list of query results.

    Each result dict must have the shape produced by ``execute_all()``:
        {topic, engine, text, citations, latency_ms, error,
         brand_mentions, positions, target_mention_count}

    Returns a dict with exactly five keys:
        ai_presence_pct : float 0–100
            Mean of per-query ``calculate_ai_presence_score`` values.
        best_brand : str
            Brand (excluding the target brand) with the highest cumulative
            AI presence score across all queries.  Returns ``'None'`` when
            no non-target brands are found.
        best_model : str
            Engine name ('Perplexity', 'ChatGPT', 'Claude', or 'Gemini')
            with the highest mean AI presence score.
        citation_count : int
            Total count of citations that contain the target domain.
        best_topic : str
            Topic with the highest mean AI presence score across its
            4-engine pass.  Returns ``''`` when *results* is empty.
    """
    if not results:
        return {
            "ai_presence_pct": 0.0,
            "best_brand": "None",
            "best_model": "",
            "citation_count": 0,
            "best_topic": "",
        }

    target_tokens = _extract_target_tokens(target_domain)

    # ---- Per-query AI presence scores ----
    query_scores: list[float] = []
    for r in results:
        mention_count = r.get("target_mention_count", 0)
        positions = r.get("positions", []) or []
        first_position = positions[0] if positions else None
        brand_mentions = r.get("brand_mentions", []) or []
        total_brands = len(brand_mentions)

        score = calculate_ai_presence_score(
            mention_count=mention_count,
            first_position=first_position,
            total_brands=total_brands,
        )
        query_scores.append(score)

    # ---- AI presence pct ----
    ai_presence_pct = statistics.mean(query_scores) if query_scores else 0.0

    # ---- Per-query scores keyed by engine and topic ----
    engine_scores: dict[str, list[float]] = {}
    topic_scores: dict[str, list[float]] = {}
    for r, score in zip(results, query_scores):
        eng = r.get("engine", "")
        topic = r.get("topic", "")
        engine_scores.setdefault(eng, []).append(score)
        topic_scores.setdefault(topic, []).append(score)

    # ---- Best model ----
    best_model = ""
    best_model_mean = -1.0
    for eng, scores in engine_scores.items():
        m = statistics.mean(scores)
        if m > best_model_mean:
            best_model_mean = m
            best_model = eng

    # ---- Best topic ----
    best_topic = ""
    best_topic_mean = -1.0
    for topic, scores in topic_scores.items():
        m = statistics.mean(scores)
        if m > best_topic_mean:
            best_topic_mean = m
            best_topic = topic

    # ---- Best brand (excluding target) ----
    # For each non-target brand, accumulate a presence-style score across all
    # queries.  We treat each appearance of the brand in a result's
    # brand_mentions list as a mention (mention_count=1) and use its 1-indexed
    # position.  The score per appearance is:
    #     (total_brands - position + 1) / total_brands * 100
    # which is position_score without the mention multiplier.
    brand_cumulative: dict[str, float] = {}
    for r in results:
        brand_mentions = r.get("brand_mentions", []) or []
        total_brands = len(brand_mentions)
        if total_brands <= 0:
            continue
        for pos, brand in enumerate(brand_mentions, start=1):
            if _is_target_brand(brand, target_tokens):
                continue
            # position_score for a single mention
            ps = (total_brands - pos + 1) / total_brands * 100.0
            brand_cumulative[brand] = brand_cumulative.get(brand, 0.0) + ps

    if brand_cumulative:
        best_brand = max(brand_cumulative, key=brand_cumulative.__getitem__)
    else:
        best_brand = "None"

    # ---- Citation count ----
    citation_count = 0
    for r in results:
        citations = r.get("citations", []) or []
        for cite in citations:
            if _domain_contains_citation(cite, target_domain):
                citation_count += 1

    return {
        "ai_presence_pct": round(ai_presence_pct, 2),
        "best_brand": best_brand,
        "best_model": best_model,
        "citation_count": citation_count,
        "best_topic": best_topic,
    }