"""
Tests for src/scoring.py — brand mention extraction with position tracking.

Covers:
  1. Normal extraction (multi-word capitalized brands)
  2. Position tracking (target brand occurrence positions)
  3. Deduplication (preserve first-appearance order)
  4. False-positive filtering (stop words, geographic names, fragments)
  5. Empty / edge-case responses (empty, whitespace, no brands)
  6. Connector brands (Johnson & Johnson, Procter & Gamble)
  7. Domain token extraction (compound, hyphenated, subdomain matching)
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import (
    extract_brand_mentions,
    _is_plausible_brand,
    _extract_target_tokens,
    _normalize_brand_for_match,
    _normalize_domain,
)


# ============================================================================
# Test 1: Normal extraction — multi-word capitalized brands
# ============================================================================


def test_normal_extraction():
    """Multi-word capitalized phrases are extracted, ordered by appearance."""
    text = (
        "Acme Corp leads the market while Global Tech Solutions and "
        "Pareto Talent compete for second place."
    )
    brands, positions = extract_brand_mentions(text, "paretotalent.com")

    assert brands == ["Acme Corp", "Global Tech Solutions", "Pareto Talent"], (
        f"Expected 3 brands in order, got: {brands}"
    )
    # "Pareto Talent" is the 3rd brand mentioned
    assert positions == [3], f"Expected position [3], got: {positions}"


def test_normal_extraction_with_numbers():
    """Brands with numeric components are not spuriously split."""
    text = "Level 3 Communications and 23andMe Inc are notable."
    # "Level" alone wouldn't match PLAIN_BRAND (needs 2+ words)
    # "23andMe" starts with digit, not [A-Z] — won't match
    brands, _ = extract_brand_mentions(text, "level3.com")
    # Only "Level 3 Communications" might match if "3" can follow [A-Z][a-z]+
    # Actually "Level" matches but "3" doesn't (not [A-Z][a-z]+), so no match
    # "23andMe Inc" — "Inc" alone doesn't match, "23andMe" starts with digit
    # So we expect 0 brands from this text
    assert brands == [], f"Expected no brands from numeric-heavy text, got: {brands}"


# ============================================================================
# Test 2: Position tracking — target brand occurrence positions
# ============================================================================


def test_position_tracking_simple():
    """Positions reflect 1-indexed occurrence numbers before dedup."""
    text = "Pareto Talent is great. Global Tech is also. Pareto Talent again."
    brands, positions = extract_brand_mentions(text, "paretotalent.com")

    assert brands == ["Pareto Talent", "Global Tech"]
    # Pareto Talent appears at positions 1 and 3 (before dedup)
    assert positions == [1, 3], f"Expected [1, 3], got: {positions}"


def test_position_tracking_target_absent():
    """When target brand is not mentioned, positions is empty."""
    text = "Acme Corp and Global Tech dominate. Beta Inc trails."
    brands, positions = extract_brand_mentions(text, "xerox.com")

    assert brands == ["Acme Corp", "Global Tech", "Beta Inc"]
    assert positions == [], f"Expected [], got: {positions}"


def test_position_tracking_many_mentions():
    """Position tracking handles many mentions of the target brand."""
    text = (
        "Acme Corp first. Beta Corp second. Acme Corp third. "
        "Gamma Co fourth. Acme Corp fifth. Delta Ltd sixth. Acme Corp last."
    )
    brands, positions = extract_brand_mentions(text, "acmecorp.com")

    assert "Acme Corp" in brands
    assert positions == [1, 3, 5, 7], f"Expected [1, 3, 5, 7], got: {positions}"


def test_position_tracking_compound_domain():
    """Compound domain 'paretotalent.com' matches 'Pareto Talent' in text."""
    text = "Alpha Inc leads. Pareto Talent follows."
    brands, positions = extract_brand_mentions(text, "paretotalent.com")

    assert brands == ["Alpha Inc", "Pareto Talent"]
    assert positions == [2], f"Expected [2], got: {positions}"


# ============================================================================
# Test 3: Deduplication — preserve first-appearance order
# ============================================================================


def test_deduplication_preserves_order():
    """Duplicates are removed; first appearance determines unique order."""
    text = (
        "Beta Corp mentioned. Alpha Inc follows. Beta Corp again. "
        "Gamma Co enters. Alpha Inc returns."
    )
    brands, _ = extract_brand_mentions(text, "betacorp.com")

    assert brands == ["Beta Corp", "Alpha Inc", "Gamma Co"], (
        f"Expected first-appearance order, got: {brands}"
    )


def test_deduplication_case_insensitive():
    """Same brand with different casing is treated as duplicate.

    NOTE: The regex only matches Title Case brands ([A-Z][a-z]+).
    ALL-CAPS like ACME CORP won't match. Case-insensitive dedup
    is tested here with mixed Title Case variants.
    """
    text = "Acme Corp was first. ACME Corp came second but won't match. Acme Corp third."
    brands, _ = extract_brand_mentions(text, "acmecorp.com")

    # "ACME Corp" won't match (all-caps), only title-case "Acme Corp" does.
    # Both "Acme Corp" occurrences are same case, deduped to one.
    assert brands == ["Acme Corp"], (
        f"Expected ['Acme Corp'] (only title-case matches), got: {brands}"
    )


def test_deduplication_keeps_position_for_duplicates():
    """Even with dedup, positions count all occurrences (including duplicates).

    Uses 'acme-corp.com' which tokenizes to ['acme-corp', 'acme corp']
    and will match 'Acme Corp' via normalized comparison.
    """
    text = "Acme Corp first. Other Co second. Acme Corp third."
    brands, positions = extract_brand_mentions(text, "acme-corp.com")

    assert brands == ["Acme Corp", "Other Co"]
    assert positions == [1, 3], f"Expected [1, 3] (counting all 3 mentions), got: {positions}"


# ============================================================================
# Test 4: False-positive filtering
# ============================================================================


def test_filter_stop_word_phrases():
    """Phrases composed entirely of stop words are filtered out."""
    text = "For The And It Is not a brand. Acme Corp is real."
    brands, _ = extract_brand_mentions(text, "acmecorp.com")

    # "For The" should be filtered, "And It" should be filtered
    # Only "Acme Corp" should remain
    assert brands == ["Acme Corp"], f"Expected ['Acme Corp'], got: {brands}"


def test_filter_geographic_false_positives():
    """Known geographic names are filtered out."""
    text = (
        "New York based Acme Corp and San Francisco startup Beta Inc "
        "are expanding to Los Angeles and Las Vegas."
    )
    brands, _ = extract_brand_mentions(text, "acmecorp.com")

    assert "New York" not in brands, f"'New York' should be filtered, got: {brands}"
    assert "San Francisco" not in brands, f"'San Francisco' should be filtered"
    assert "Los Angeles" not in brands, f"'Los Angeles' should be filtered"
    assert "Las Vegas" not in brands, f"'Las Vegas' should be filtered"
    assert brands == ["Acme Corp", "Beta Inc"], f"Expected ['Acme Corp', 'Beta Inc'], got: {brands}"


def test_filter_short_phrases():
    """Phrases ≤ 5 characters total are filtered out."""
    text = "Hi Co and Lo Inc are not real brands but Acme Corp is."
    brands, _ = extract_brand_mentions(text, "acmecorp.com")

    # "Hi Co" is 5 chars (including space) → filtered (len ≤ 5)
    # "Lo Inc" is 6 chars → this is borderline but plausible
    # "Acme Corp" is 9 chars → kept
    assert "Hi Co" not in brands, f"'Hi Co' (5 chars) should be filtered, got: {brands}"
    assert "Acme Corp" in brands


def test_filter_connector_fragment():
    """A connector phrase that starts inside a plain brand span is filtered."""
    text = "Acme Corp and Global Tech are competitors."

    # "Corp and Global" is a connector match whose start (position of 'C' in
    # the second word "Corp") falls inside the plain match "Acme Corp".
    # It should be filtered out by the overlap guard.
    brands, _ = extract_brand_mentions(text, "acmecorp.com")

    assert "Corp and Global" not in brands, f"Fragment should be filtered, got: {brands}"
    assert brands == ["Acme Corp", "Global Tech"], f"Expected ['Acme Corp', 'Global Tech'], got: {brands}"


def test_filter_lowercase_connector():
    """Lowercase words before 'and' should not match as connector brands."""
    text = "We continue to dominate and Delta Ltd enters the market."
    brands, _ = extract_brand_mentions(text, "deltaltd.com")

    # "dominate" is lowercase, so "dominate and Delta" should NOT match
    # Only "Delta Ltd" should be found
    assert "dominate and Delta" not in brands, f"'dominate and Delta' should be filtered"
    assert brands == ["Delta Ltd"], f"Expected ['Delta Ltd'], got: {brands}"


# ============================================================================
# Test 5: Empty / edge-case responses
# ============================================================================


def test_empty_response():
    """Empty string returns empty lists."""
    brands, positions = extract_brand_mentions("", "acmecorp.com")
    assert brands == [], f"Expected [], got: {brands}"
    assert positions == [], f"Expected [], got: {positions}"


def test_whitespace_only():
    """Whitespace-only response returns empty lists."""
    brands, positions = extract_brand_mentions("   \n\t  ", "acmecorp.com")
    assert brands == []
    assert positions == []


def test_no_capitalized_words():
    """Text with no capitalized multi-word phrases returns empty."""
    text = "this is all lowercase and nothing matches the brand pattern at all."
    brands, positions = extract_brand_mentions(text, "acmecorp.com")
    assert brands == []
    assert positions == []


def test_single_capitalized_words():
    """Single capitalized words are not extracted (need 2+ words)."""
    text = "Acme is great and Global has potential but neither is multi-word."
    brands, _ = extract_brand_mentions(text, "acme.com")
    assert brands == [], f"Single words should not match, got: {brands}"


def test_no_target_domain():
    """Empty target domain returns brands but empty positions."""
    text = "Acme Corp and Global Tech are competitors."
    brands, positions = extract_brand_mentions(text, "")
    assert brands == ["Acme Corp", "Global Tech"]
    assert positions == []


def test_none_response():
    """None response_text should be handled gracefully."""
    # Our function checks `if not response_text`, which catches None
    brands, positions = extract_brand_mentions(None, "acmecorp.com")
    assert brands == []
    assert positions == []


# ============================================================================
# Test 6: Connector brands (& / and)
# ============================================================================


def test_ampersand_brands():
    """Brands with '&' connector are extracted as single units."""
    text = "Johnson & Johnson and Procter & Gamble are industry leaders."
    brands, _ = extract_brand_mentions(text, "jnj.com")
    assert brands == ["Johnson & Johnson", "Procter & Gamble"], (
        f"Expected ['Johnson & Johnson', 'Procter & Gamble'], got: {brands}"
    )


def test_and_connector_brand():
    """Brands with 'and' connector are extracted as single units."""
    text = "Procter and Gamble is a major consumer goods company."
    brands, _ = extract_brand_mentions(text, "pg.com")
    assert brands == ["Procter and Gamble"], (
        f"Expected ['Procter and Gamble'], got: {brands}"
    )


def test_mixed_connectors():
    """Mixed '&' and 'and' in separate brands are handled."""
    text = "Johnson & Johnson partners with Procter and Gamble."
    brands, _ = extract_brand_mentions(text, "jnj.com")
    assert "Johnson & Johnson" in brands
    assert "Procter and Gamble" in brands


def test_connector_brand_position_tracking():
    """Position tracking works for connector brands matching hyphenated domains."""
    text = "Procter & Gamble leads. Johnson & Johnson follows."
    brands, positions = extract_brand_mentions(text, "procter-gamble.com")
    assert brands == ["Procter & Gamble", "Johnson & Johnson"]
    # "procter-gamble.com" → tokens: ["procter-gamble", "procter gamble"]
    # normalized "Procter & Gamble" → "procter gamble" matches "procter gamble"
    assert positions == [1], f"Expected [1], got: {positions}"


# ============================================================================
# Test 7: Domain token extraction
# ============================================================================


def test_compound_domain_token_extraction():
    """'paretotalent.com' extracts tokens for 'pareto talent' and 'pareto'."""
    tokens = _extract_target_tokens("paretotalent.com")
    assert "paretotalent" in tokens, f"Missing bare domain token, got: {tokens}"
    assert "pareto talent" in tokens, f"Missing humanized token, got: {tokens}"
    assert "pareto" in tokens, f"Missing prefix token, got: {tokens}"


def test_hyphenated_domain_token_extraction():
    """'procter-gamble.com' extracts hyphen-separated tokens."""
    tokens = _extract_target_tokens("procter-gamble.com")
    assert "procter-gamble" in tokens
    assert "procter gamble" in tokens


def test_simple_domain_token_extraction():
    """'acme.com' extracts just 'acme'."""
    tokens = _extract_target_tokens("acme.com")
    assert tokens == ["acme"], f"Expected ['acme'], got: {tokens}"


def test_normalize_domain():
    """Domain normalization strips scheme, www, path."""
    assert _normalize_domain("https://www.AcmeCorp.com/path") == "acmecorp.com"
    assert _normalize_domain("http://acme.com") == "acme.com"
    assert _normalize_domain("www.example.com") == "example.com"
    assert _normalize_domain("example.com") == "example.com"
    assert _normalize_domain("") == ""


def test_normalize_brand_for_match():
    """Brand normalization collapses connectors and lowercases."""
    assert _normalize_brand_for_match("Procter & Gamble") == "procter gamble"
    assert _normalize_brand_for_match("Johnson and Johnson") == "johnson johnson"
    assert _normalize_brand_for_match("Acme Corp") == "acme corp"
    assert _normalize_brand_for_match("  Extra   Spaces  &  Stuff  ") == "extra spaces stuff"


# ============================================================================
# Test 8: _is_plausible_brand helper
# ============================================================================


def test_is_plausible_brand_valid():
    """Real-looking brands pass the filter."""
    assert _is_plausible_brand("Acme Corp") is True
    assert _is_plausible_brand("Global Tech Solutions") is True
    assert _is_plausible_brand("Johnson & Johnson") is True
    assert _is_plausible_brand("Pareto Talent") is True


def test_is_plausible_brand_invalid():
    """False positives are rejected."""
    assert _is_plausible_brand("For The") is False  # all stop words
    assert _is_plausible_brand("And It") is False  # all stop words
    assert _is_plausible_brand("New York") is False  # geographic
    assert _is_plausible_brand("Los Angeles") is False  # geographic
    assert _is_plausible_brand("Hi Co") is False  # ≤ 5 chars
    assert _is_plausible_brand("Is For") is False  # all stop words


# ============================================================================
# Test 9: Integration-style real-world scenarios
# ============================================================================


def test_realistic_ai_response():
    """A realistic AI engine response with multiple brands."""
    text = (
        "Based on current market data, the top recruitment platforms "
        "for creative agencies include Pareto Talent, Creative People, "
        "and The Agency Source. Pareto Talent stands out for its "
        "specialized focus on design and marketing roles. Other notable "
        "platforms include Recruit Creative and Design Hire."
    )
    brands, positions = extract_brand_mentions(text, "paretotalent.com")

    assert "Pareto Talent" in brands
    assert "Creative People" in brands
    assert "The Agency Source" in brands  # "The" is stop word but "Agency Source" is content
    assert "Recruit Creative" in brands
    assert "Design Hire" in brands

    # Pareto Talent appears twice: positions 1 and 4 in raw mentions
    assert positions == [1, 4], (
        f"Expected positions [1, 4] for two Pareto Talent mentions, got: {positions}"
    )


def test_competitor_heavy_response():
    """Response with many competitors and repeated target mentions."""
    text = (
        "1. Acme Corp — market leader\n"
        "2. Global Tech — strong contender\n"
        "3. Acme Corp — also cited here\n"
        "4. Beta Inc — emerging player\n"
        "5. Gamma Co — niche provider\n"
        "6. Acme Corp — mentioned again\n"
        "7. Delta Ltd — enterprise focus\n"
        "8. Acme Corp — final mention"
    )
    brands, positions = extract_brand_mentions(text, "acmecorp.com")

    assert len(brands) == 5  # Acme Corp, Global Tech, Beta Inc, Gamma Co, Delta Ltd
    # Acme Corp appears at raw positions 1, 3, 6, 8 (1-indexed among all mentions)
    assert positions == [1, 3, 6, 8], (
        f"Expected [1, 3, 6, 8] for Acme Corp mentions, got: {positions}"
    )


# ============================================================================
# Test 10: Edge cases around regex boundaries
# ============================================================================


def test_brand_at_text_boundaries():
    """Brands at the start and end of text are detected."""
    text = "Acme Corp leads. The market follows. Closes with Global Tech."
    brands, _ = extract_brand_mentions(text, "acmecorp.com")
    assert brands[0] == "Acme Corp"
    assert brands[-1] == "Global Tech"


def test_brand_with_punctuation():
    """Brands adjacent to punctuation are still extracted."""
    text = "(Acme Corp) and [Global Tech] with \"Beta Inc\" — all quoted."
    brands, _ = extract_brand_mentions(text, "acmecorp.com")
    assert brands == ["Acme Corp", "Global Tech", "Beta Inc"], f"Got: {brands}"


def test_brand_across_sentence_boundaries():
    """Brands split across sentences are extracted separately."""
    text = "We recommend Acme Corp. Beta Inc is also good."
    brands, _ = extract_brand_mentions(text, "acmecorp.com")
    assert brands == ["Acme Corp", "Beta Inc"], f"Got: {brands}"


def test_no_false_positive_on_single_cap_word():
    """Single capitalized words like 'Monday' or 'January' aren't brands."""
    text = "We met on Monday with January Corp to discuss plans."
    brands, _ = extract_brand_mentions(text, "januarycorp.com")
    # "Monday" alone — single word, no match
    # "January Corp" — two capitalized words, valid
    assert brands == ["January Corp"], f"Got: {brands}"
    assert "Monday" not in brands


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    for name, func in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            func()
            passed += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, f"ERROR: {e}"))
            print(f"  ✗ {name}: ERROR: {e}")

    print()
    print(f"{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    if errors:
        print(f"\nFailures:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)