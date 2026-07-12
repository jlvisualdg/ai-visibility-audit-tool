"""
Validate the scoring system against the documented examples in
docs/scoring_system.md.

Each test case is extracted verbatim from the "Worked Examples" section.
Run this after any change to MAX_MENTION_MULTIPLIER or the scoring logic.

Usage:
    python tests/validate_scoring.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'from src.scoring import ...' works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import calculate_ai_presence_score, calculate_overall_ai_presence

TOLERANCE = 0.01


# -----------------------------------------------------------------------
# Test cases — one per Worked Example in docs/scoring_system.md
# -----------------------------------------------------------------------

def test_example_1_best_case():
    """Example 1: 1st of 4 with 3 mentions → 100.00 %"""
    score = calculate_ai_presence_score(
        mention_count=3,
        first_position=1,
        total_brands=4,
    )
    assert abs(score - 100.00) < TOLERANCE, (
        f"Example 1 failed: expected 100.00, got {score}"
    )


def test_example_2_mid_mention():
    """Example 2: 3rd of 4 with 2 mentions → 33.33 %"""
    score = calculate_ai_presence_score(
        mention_count=2,
        first_position=3,
        total_brands=4,
    )
    assert abs(score - 33.33) < TOLERANCE, (
        f"Example 2 failed: expected 33.33, got {score}"
    )


def test_example_3_first_single():
    """Example 3: 1st of 5 with 1 mention → 33.33 %"""
    score = calculate_ai_presence_score(
        mention_count=1,
        first_position=1,
        total_brands=5,
    )
    assert abs(score - 33.33) < TOLERANCE, (
        f"Example 3 failed: expected 33.33, got {score}"
    )


def test_example_4_last_mention():
    """Example 4: 4th of 4 with 1 mention → 8.33 %"""
    score = calculate_ai_presence_score(
        mention_count=1,
        first_position=4,
        total_brands=4,
    )
    assert abs(score - 8.33) < TOLERANCE, (
        f"Example 4 failed: expected 8.33, got {score}"
    )


def test_example_5_brand_not_found():
    """Example 5: brand absent → 0.00 %"""
    score = calculate_ai_presence_score(
        mention_count=0,
        first_position=None,
        total_brands=5,
    )
    assert abs(score - 0.0) < TOLERANCE, (
        f"Example 5 failed: expected 0.00, got {score}"
    )


# -----------------------------------------------------------------------
# Additional edge-case tests
# -----------------------------------------------------------------------

def test_no_brands_in_response():
    """Zero total brands — score is 0.0 even if position is given."""
    score = calculate_ai_presence_score(1, 1, 0)
    assert score == 0.0


def test_cap_at_100():
    """Score never exceeds 100 even with extreme inputs."""
    score = calculate_ai_presence_score(
        mention_count=100,
        first_position=1,
        total_brands=1,
    )
    assert score <= 100.0, f"Score exceeded cap: {score}"


def test_overall_mean_aggregation():
    """Overall AI Presence = mean of query scores."""
    scores = [100.0, 33.33, 33.33, 8.33, 0.0]
    overall = calculate_overall_ai_presence(scores)
    expected = sum(scores) / len(scores)  # 34.998
    assert abs(overall - expected) < TOLERANCE, (
        f"Aggregation failed: expected {expected}, got {overall}"
    )


def test_overall_empty_list():
    """Empty list → 0.0."""
    assert calculate_overall_ai_presence([]) == 0.0


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_example_1_best_case,
        test_example_2_mid_mention,
        test_example_3_first_single,
        test_example_4_last_mention,
        test_example_5_brand_not_found,
        test_no_brands_in_response,
        test_cap_at_100,
        test_overall_mean_aggregation,
        test_overall_empty_list,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ✓ {t.__doc__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__doc__}")
            print(f"    {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("VALIDATION: All tests pass within 0.01 tolerance.")
    else:
        print(f"VALIDATION FAILED: {failed} test(s) failed.")
        sys.exit(1)