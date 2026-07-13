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

# Multi-word AI-ism phrases that are common in LLM responses — these are
# conversational filler / hedging phrases, not brand names.  Checked via
# case-insensitive substring match because they often appear embedded in
# longer prose (e.g. "Key considerations include ...").
AI_ISM_PHRASES: list[str] = [
    "key considerations",
    "it depends",
    "keep in mind",
    "it's worth noting",
    "it is worth noting",
    "it's important to",
    "it is important to",
    "in conclusion",
    "in summary",
    "to summarize",
    "on the other hand",
    "that being said",
    "having said that",
    "at the end of the day",
    "the bottom line",
    "the fact that",
    "one thing to consider",
    "something to keep in mind",
    "let's dive into",
    "let me explain",
    "here's the thing",
    "here is the thing",
    "the key takeaway",
    "the main takeaway",
    "what this means",
    "what does this mean",
    "to put it simply",
    "to be clear",
    "worth mentioning",
    "it goes without saying",
    "needless to say",
    "as mentioned earlier",
    "as noted above",
    "if you mean",
    "on average",
    "most cases",
    "in some cases",
    "insurance coverage",
    "top vendors",
    "top providers",
    "top options",
    "best options",
    "best providers",
    "top brands",
    "best brands",
    "top companies",
    "best companies",
    "top rated",
    "best rated",
    "key features",
    "main features",
    "important factors",
    "key factors",
    "things to consider",
    "factors to consider",
    "depending on your",
    "depending on the",
    "based on your",
    "based on the",
    "worth noting",
    "please note",
    "important to note",
    "make sure",
    "before choosing",
    "when choosing",
    "consider these",
    "pro tip", "quick tip",
    "medication costs", "delivery methods",
    "online telehealth subscriptions",
    "top online", "perimenopause care",
    "specialized menopause care",
    "personalized hrt",
    "comprehensive hormone",
    "hormone testing",
    "virtual visits",
    "in-person visits",
    "generic alternatives",
    "brand name",
    "out-of-pocket costs",
    "without insurance",
    "with insurance",
    "monthly subscription",
    "annual subscription",
    "per month",
    "per visit",
    "per session",
    "price range",
    "cost breakdown",
    "payment options",
    "sliding scale",
    "membership model",
    "subscription model",
    "dose form",
    "pill form",
    "patch form",
    "cream form",
    "gel form",
    "injectable form",
    "oral medication",
    "topical medication",
    "transdermal patch",
    "vaginal cream",
    "compounded bioidentical",
    "synthetic hormones",
    "bioidentical hormones",
    "plant-based estrogen",
    "synthetic estrogen",
    "natural progesterone",
    "synthetic progesterone",
    "low-dose estrogen",
    "combination therapy",
    "estrogen-only therapy",
    "estrogen pills", "oral estrogen",
    "estrogen cream", "progesterone pills",
    "alternative & longevity",
    "female hormonal health",
    "menopausal & perimenopausal",
    "virtual telehealth subscriptions",
    "hormonal health",
    "telehealth subscriptions",
    "bioidentical hormone",
    "hormone pellets",
    "estrogen patches",
    "progesterone cream",
    "testosterone therapy",
    "thyroid medication",
    "birth control",
    "fertility treatments",
    "mental wellness",
    "sexual health",
    "digestive health",
    "immune support",
    "stress management",
    "sleep aids",
    "pain management",
    "chronic conditions",
    "acute care",
    "urgent care",
    "primary physician",
    "specialist referral",
    "lab testing",
    "blood work",
    "diagnostic imaging",
    "preventive screening",
    "annual exam",
    "wellness exam",
    "physical exam",
]
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
    target_brand_name: str = "",
) -> Tuple[list[str], list[int]]:
    """Extract brand names from AI response text using contextual analysis.

    Brand recommendations in AI responses appear in specific structural contexts:
    1. Numbered/bulleted list items (e.g. "1. Midi Health - ...")
    2. Bold or linked markdown (e.g. "**Midi Health**" or "[Midi Health](url)")
    3. After recommendation verbs ("recommend", "suggest", "consider", "top pick")
    4. As standalone capitalized names NOT part of a descriptive phrase

    Descriptive phrases like "Hormone Replacement Therapy" or "Online Directories"
    are filtered out because they appear in prose context, not recommendation context.

    Extraction strategy:
    1. Split response into blocks (double-newline separated)
    2. Identify "recommendation blocks" (lists, bold names, linked names)
    3. Extract capitalized phrases only from recommendation blocks
    4. Filter through plausibility checks (stop words, AI-isms, descriptive terms)
    5. Track target brand positions

    Args:
        response_text: The AI engine response text to scan.
        target_domain: Domain being audited (e.g. "bywinona.com").
        target_brand_name: Brand name extracted from the scraped page title.

    Returns:
        (unique_brands, target_positions) tuple.
    """
    if not response_text or not response_text.strip():
        return ([], [])

    # ----- Step 1: Contextual block splitting -----
    # Split on double newlines to get logical blocks
    blocks = re.split(r'\n\s*\n', response_text)

    # ----- Step 2: Classify blocks and extract candidates -----
    # A "recommendation block" is one that contains list items, bold names,
    # linked names, or appears after a recommendation verb.
    all_candidates: list[str] = []  # ordered list of candidate brand names

    for block in blocks:
        block_candidates = _extract_brands_from_block(block)
        all_candidates.extend(block_candidates)

    # Also scan for bold/linked brand names across the entire text
    # (these can appear inline in prose too)
    bold_brands = _extract_bold_linked_brands(response_text)
    for b in bold_brands:
        if b not in all_candidates:
            all_candidates.append(b)

    # ----- Step 3: Filter false positives -----
    filtered: list[str] = []
    for phrase in all_candidates:
        if _is_plausible_brand(phrase) and _is_not_descriptive(phrase):
            filtered.append(phrase)

    # ----- Step 4: Deduplicate while preserving order -----
    seen: set[str] = set()
    unique_brands: list[str] = []
    for phrase in filtered:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            unique_brands.append(phrase)

    # ----- Step 5: Find target brand positions -----
    target_tokens = _extract_target_tokens(target_domain)
    if target_brand_name and target_brand_name.lower() not in target_tokens:
        target_tokens.append(target_brand_name.lower())
    target_positions = _find_target_positions(filtered, target_tokens)

    return (unique_brands, target_positions)


# ---------------------------------------------------------------------------

# Regex for detecting numbered-list item prefixes like "1.", "2)", "1 -", etc.
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.\)]\s*|[-*]\s+)(.+)", re.MULTILINE)

# Regex for bold markdown: **Brand Name** or __Brand Name__
_BOLD_RE = re.compile(r"\*\*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\*\*")

# Regex for linked markdown: [Brand Name](url)
_LINKED_RE = re.compile(r"\[([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\]\(")

# Regex for connector brands in list items
_LIST_CONNECTOR_RE = re.compile(
    r"(?:\d+[.\)]\s*|[-*]\s+)"
    r"([A-Z][a-z]+"
    r"\s+(?i:&|and)\s+"
    r"[A-Z][a-z]+"
    r")"
)

# Descriptive phrases that are NOT brands — these are category descriptions,
# treatment names, or structural labels commonly capitalized in AI responses.
_DESCRIPTIVE_TERMS: set[str] = {
    "hormone replacement therapy", "hormone therapy",
    "telehealth platforms", "telehealth providers",
    "online directories", "national provider directories",
    "dedicated menopause telehealth", "dedicated menopause telehealth platforms",
    "specialized telehealth platforms", "specialized telehealth",
    "mental health", "primary care",
    "blood pressure", "bone health",
    "medical history", "medical professional",
    "healthcare provider", "health care",
    "clinical trials", "case studies",
    "treatment options", "treatment plan",
    "patient portal", "patient support",
    "customer reviews", "customer support",
    "free consultation", "initial consultation",
    "board certified", "board-certified",
    "insurance coverage", "prescription medication",
    "active ingredients", "medical advice",
    "emergency services", "emergency room",
    "preventive care", "preventative care",
    "wellness programs", "wellness centers",
    "fitness programs", "nutrition plans",
    "skin care", "hair care",
    "weight loss", "weight management",
    "menopause relief", "menopause symptoms",
    "perimenopause symptoms",
    "estrogen levels", "estrogen therapy",
    "progesterone levels", "progesterone therapy",
    "top picks", "top choices", "top options",
    "best picks", "best choices", "best options",
    "key features", "main benefits",
    "pro tip", "quick tip",
    "medication costs", "delivery methods",
    "online telehealth subscriptions",
    "top online", "perimenopause care",
    "specialized menopause care",
    "personalized hrt",
    "comprehensive hormone",
    "hormone testing",
    "virtual visits",
    "in-person visits",
    "generic alternatives",
    "brand name",
    "out-of-pocket costs",
    "without insurance",
    "with insurance",
    "monthly subscription",
    "annual subscription",
    "per month",
    "per visit",
    "per session",
    "price range",
    "cost breakdown",
    "payment options",
    "sliding scale",
    "membership model",
    "subscription model",
    "dose form",
    "pill form",
    "patch form",
    "cream form",
    "gel form",
    "injectable form",
    "oral medication",
    "topical medication",
    "transdermal patch",
    "vaginal cream",
    "compounded bioidentical",
    "synthetic hormones",
    "bioidentical hormones",
    "plant-based estrogen",
    "synthetic estrogen",
    "natural progesterone",
    "synthetic progesterone",
    "low-dose estrogen",
    "combination therapy",
    "estrogen-only therapy",
    "cyclic therapy",
    "continuous therapy",
    "estrogen pills", "oral estrogen",
    "estrogen cream", "progesterone pills",
    "alternative & longevity",
    "female hormonal health",
    "menopausal & perimenopausal",
    "virtual telehealth subscriptions",
    "hormonal health",
    "telehealth subscriptions",
    "bioidentical hormone",
    "hormone pellets",
    "estrogen patches",
    "progesterone cream",
    "testosterone therapy",
    "thyroid medication",
    "birth control",
    "fertility treatments",
    "mental wellness",
    "sexual health",
    "digestive health",
    "immune support",
    "stress management",
    "sleep aids",
    "pain management",
    "chronic conditions",
    "acute care",
    "urgent care",
    "primary physician",
    "specialist referral",
    "lab testing",
    "blood work",
    "diagnostic imaging",
    "preventive screening",
    "annual exam",
    "wellness exam",
    "physical exam",
}

# City names that get capitalized but aren't brands
_GEO_NAMES: set[str] = {
    "falls church", "new york", "los angeles", "san francisco",
    "las vegas", "united states", "united kingdom", "south africa",
    "north america", "european union", "middle east", "south east",
    "washington dc", "washington d.c.",
}


def _extract_brands_from_block(block: str) -> list[str]:
    """Extract candidate brand names from a single text block.

    Detects brands in:
    - Numbered/bulleted list items: "1. Midi Health - ..."
    - Bold text: "**Midi Health** offers..."
    - Linked text: "[Midi Health](https://...)"
    - Connector brands in lists: "1. Johnson & Johnson - ..."
    """
    candidates: list[str] = []
    seen_lower: set[str] = set()

    def _add(phrase: str):
        if phrase and phrase.lower() not in seen_lower:
            candidates.append(phrase)
            seen_lower.add(phrase.lower())

    # 1. List items: "1. Brand Name ..." or "- Brand Name ..."
    for m in _LIST_ITEM_RE.finditer(block):
        item_text = m.group(1).strip()
        # Extract the first capitalized phrase from the list item
        # (this is typically the brand name before the description)
        brand_match = PLAIN_BRAND.match(item_text)
        if brand_match:
            _add(brand_match.group(1))
        else:
            # Try connector brand in list
            conn_match = _LIST_CONNECTOR_RE.match(block[block.index(item_text)-3:])
            if conn_match:
                _add(conn_match.group(1))
            # Single capitalized word as brand (e.g. "Winona")
            elif re.match(r'^[A-Z][a-z]{2,}\b', item_text):
                single = re.match(r'^([A-Z][a-z]{2,})\b', item_text)
                if single:
                    _add(single.group(1))

    # 2. Bold brands in this block
    for m in _BOLD_RE.finditer(block):
        _add(m.group(1))

    # 3. Linked brands in this block
    for m in _LINKED_RE.finditer(block):
        _add(m.group(1))

    # 4. Plain brand regex (catches brands in any context within the block)
    for m in PLAIN_BRAND.finditer(block):
        _add(m.group(1))

    # 5. Connector brands (Johnson & Johnson)
    for m in CONNECTOR_BRAND.finditer(block):
        _add(m.group(1))

    return candidates


def _extract_bold_linked_brands(text: str) -> list[str]:
    """Extract brand names that appear in bold or as link text across the full response."""
    brands: list[str] = []
    seen_lower: set[str] = set()

    for m in _BOLD_RE.finditer(text):
        phrase = m.group(1)
        if phrase.lower() not in seen_lower:
            brands.append(phrase)
            seen_lower.add(phrase.lower())

    for m in _LINKED_RE.finditer(text):
        phrase = m.group(1)
        if phrase.lower() not in seen_lower:
            brands.append(phrase)
            seen_lower.add(phrase.lower())

    return brands


def _is_not_descriptive(phrase: str) -> bool:
    """Return True if the phrase is NOT a descriptive term, category, or geo name.

    This catches phrases that are capitalized in AI responses but are
    descriptions/treatments rather than brand names:
    - "Hormone Replacement Therapy" (treatment name)
    - "Online Directories" (category)
    - "Falls Church" (city)
    """
    phrase_lower = phrase.lower()

    # Check descriptive terms
    if phrase_lower in _DESCRIPTIVE_TERMS:
        return False

    # Check geo names
    if phrase_lower in _GEO_NAMES:
        return False

    # Check if any descriptive term is a substring
    for term in _DESCRIPTIVE_TERMS:
        if phrase_lower == term or phrase_lower.startswith(term + " "):
            return False

    # Check if phrase contains common descriptive suffixes
    # Note: "health", "care" are NOT included here because many real brand
    # names end in these words (Midi Health, Everly Health, Hims & Hers).
    descriptive_suffixes = [
        "therapy", "treatment", "providers", "directories", "platforms",
        "programs", "options", "solutions",
        "medication", "relief", "symptoms",
        "trials", "studies", "reviews", "support", "consultation",
        "pro tip", "quick tip",
        "medication costs", "delivery methods",
        "online telehealth subscriptions",
        "top online", "perimenopause care",
        "specialized menopause care",
        "personalized hrt",
        "comprehensive hormone",
        "hormone testing",
        "virtual visits",
        "in-person visits",
        "generic alternatives",
        "brand name",
        "out-of-pocket costs",
        "without insurance",
        "with insurance",
        "monthly subscription",
        "annual subscription",
        "per month",
        "per visit",
        "per session",
        "price range",
        "cost breakdown",
        "payment options",
        "sliding scale",
        "membership model",
        "subscription model",
        "dose form",
        "pill form",
        "patch form",
        "cream form",
        "gel form",
        "injectable form",
        "oral medication",
        "topical medication",
        "transdermal patch",
        "vaginal cream",
        "compounded bioidentical",
        "synthetic hormones",
        "bioidentical hormones",
        "plant-based estrogen",
        "synthetic estrogen",
        "natural progesterone",
        "synthetic progesterone",
        "low-dose estrogen",
        "combination therapy",
        "estrogen-only therapy",
        "cyclic therapy",
        "continuous therapy",
        ]
    words = phrase_lower.split()
    if len(words) >= 2 and words[-1] in descriptive_suffixes:
        return False

    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_plausible_brand(phrase: str) -> bool:
    """Return True if `phrase` looks like a real brand, not a false positive.

    Filters:
      - Phrases where every word is a stop word (e.g. "For The").
      - Known false-positive geographic / generic phrases.
      - AI-ism conversational filler phrases (substring match).
      - Phrases ≤ 5 characters total (too short to be meaningful).
    """
    words = phrase.split()
    if len(phrase) <= 5:
        return False
    if all(w.lower() in STOP_WORDS for w in words):
        return False
    phrase_lower = phrase.lower()
    if phrase_lower in FALSE_POSITIVE_PHRASES:
        return False
    # Check AI-ism phrases via case-insensitive substring match —
    # these often appear embedded in longer text like "Key Considerations for ..."
    for ai_phrase in AI_ISM_PHRASES:
        if ai_phrase in phrase_lower:
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
        "pro tip", "quick tip",
        "medication costs", "delivery methods",
        "online telehealth subscriptions",
        "top online", "perimenopause care",
        "specialized menopause care",
        "personalized hrt",
        "comprehensive hormone",
        "hormone testing",
        "virtual visits",
        "in-person visits",
        "generic alternatives",
        "brand name",
        "out-of-pocket costs",
        "without insurance",
        "with insurance",
        "monthly subscription",
        "annual subscription",
        "per month",
        "per visit",
        "per session",
        "price range",
        "cost breakdown",
        "payment options",
        "sliding scale",
        "membership model",
        "subscription model",
        "dose form",
        "pill form",
        "patch form",
        "cream form",
        "gel form",
        "injectable form",
        "oral medication",
        "topical medication",
        "transdermal patch",
        "vaginal cream",
        "compounded bioidentical",
        "synthetic hormones",
        "bioidentical hormones",
        "plant-based estrogen",
        "synthetic estrogen",
        "natural progesterone",
        "synthetic progesterone",
        "low-dose estrogen",
        "combination therapy",
        "estrogen-only therapy",
        "cyclic therapy",
        "continuous therapy",
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

    # ---- AI Presence: average coverage % across all query x engine cells ----
    # A cell is "covered" if the target brand was mentioned in the answer text
    # (brand recommendation) OR the target domain was cited as a source (URL citation).
    # AI Presence = covered_cells / total_cells * 100
    covered_count = 0
    for r in results:
        mention_count = r.get("target_mention_count", 0)
        citations = r.get("citations", []) or []
        has_citation = any(_domain_contains_citation(c, target_domain) for c in citations)
        if mention_count > 0 or has_citation:
            covered_count += 1

    total_count = len(results)
    ai_presence_pct = (100.0 * covered_count / total_count) if total_count > 0 else 0.0

    # ---- Per-query scores keyed by engine and topic (for best_model/best_topic) ----
    # Use covered (1.0) or not (0.0) as the per-cell score
    engine_scores: dict[str, list[float]] = {}
    topic_scores: dict[str, list[float]] = {}
    for r in results:
        eng = r.get("engine", "")
        topic = r.get("topic", "")
        mention_count = r.get("target_mention_count", 0)
        citations = r.get("citations", []) or []
        has_citation = any(_domain_contains_citation(c, target_domain) for c in citations)
        cell_score = 1.0 if (mention_count > 0 or has_citation) else 0.0
        engine_scores.setdefault(eng, []).append(cell_score)
        topic_scores.setdefault(topic, []).append(cell_score)

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
    # Brand recommendations: brand names explicitly written in AI answer text.
    # Scored by position among all recommended brands.
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

    # ---- Citation count (target domain cited as source) ----
    # Citations: target/competitor domains returned as sources/annotations.
    # This is independent from brand recommendations.
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


# ---------------------------------------------------------------------------
# Extract recommended brands — list-style vs inline prose detection
# ---------------------------------------------------------------------------

# Regex for detecting numbered-list item prefixes like "1.", "2)", "1 -", etc.
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-•]\s+)(.+)", re.MULTILINE)


def extract_recommended_brands(response_text: str) -> dict:
    """Detect list-style brand recommendations vs inline prose mentions.

    Splits the response text by double-newlines into logical blocks.  Blocks
    that contain at least two numbered-list / bullet items are classified as
    "list-style" recommendations; all other blocks are "inline prose".

    Within list-style blocks, we extract brand mentions via the same
    PLAIN_BRAND / CONNECTOR_BRAND regex pipeline used by extract_brand_mentions(),
    then pass them through _is_plausible_brand() filtering.

    Args:
        response_text: The AI engine response text to analyze.

    Returns:
        dict with keys:
          - list_brands: list[str] — brands from list-style recommendation blocks
          - inline_brands: list[str] — brands from inline prose blocks
          - list_blocks: int — count of list-style blocks found
          - inline_blocks: int — count of inline prose blocks found
    """
    if not response_text or not response_text.strip():
        return {
            "list_brands": [],
            "inline_brands": [],
            "list_blocks": 0,
            "inline_blocks": 0,
        }

    # Split on 2+ consecutive newlines to get logical blocks
    blocks = re.split(r"\n\s*\n+", response_text.strip())
    if not blocks:
        return {"list_brands": [], "inline_brands": [], "list_blocks": 0, "inline_blocks": 0}

    list_brands: list[str] = []
    inline_brands: list[str] = []
    list_blocks = 0
    inline_blocks = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Detect list-style: block has at least 2 lines matching numbered/bullet patterns
        lines = block.split("\n")
        list_lines = [ln for ln in lines if _LIST_ITEM_RE.match(ln)]

        if len(list_lines) >= 2:
            # List-style block — extract brands from list lines specifically
            list_blocks += 1
            for ln in list_lines:
                match = _LIST_ITEM_RE.match(ln)
                if match:
                    item_text = match.group(1)
                    brands_in_item = _extract_brands_from_text(item_text)
                    for b in brands_in_item:
                        if b not in list_brands:
                            list_brands.append(b)
        else:
            # Inline prose block — extract brands from the full block
            inline_blocks += 1
            brands_in_block = _extract_brands_from_text(block)
            for b in brands_in_block:
                if b not in inline_brands:
                    inline_brands.append(b)

    return {
        "list_brands": list_brands,
        "inline_brands": inline_brands,
        "list_blocks": list_blocks,
        "inline_blocks": inline_blocks,
    }


def _extract_brands_from_text(text: str) -> list[str]:
    """Extract plausible brand names from a short text snippet.

    Uses the same regex patterns as extract_brand_mentions() but returns
    only the unique, filtered brand list (no position tracking).
    """
    if not text:
        return []

    raw_matches: list[tuple[int, int, str]] = []

    for m in PLAIN_BRAND.finditer(text):
        raw_matches.append((m.start(), m.end(), m.group(1)))

    plain_spans = [(s, e) for s, e, _ in raw_matches]
    for m in CONNECTOR_BRAND.finditer(text):
        if not any(s <= m.start() < e for s, e in plain_spans):
            raw_matches.append((m.start(), m.end(), m.group(1)))

    raw_matches.sort(key=lambda x: x[0])
    phrases = [p for _, _, p in raw_matches]

    seen: set[str] = set()
    brands: list[str] = []
    for phrase in phrases:
        if _is_plausible_brand(phrase):
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                brands.append(phrase)

    return brands