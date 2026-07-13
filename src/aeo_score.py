"""
Composite AEO Score calculator.

    AEO Score = 0.45 × Visibility + 0.30 × Citation + 0.25 × Indexability

Each sub-score is 0–100.  The composite is an integer in [0, 100].

Sub-score definitions
---------------------
Visibility (0-100):
    Position/mention-weighted mean across all query×engine cells.
    Already computed by aggregate_results() as ai_presence_pct.

Citation (0-100):
    Fraction of query×engine cells where the target domain appears at
    least once in the structured source list (not just prose mentions).
    Independent from visibility — a brand can be cited without being
    explicitly recommended, and vice versa.

Indexability (0-100):
    Crawler health score — reflects how well the site is structured for
    AI crawling (schema, authorship, answer capsules, robots/llms.txt,
    credibility signals, etc.).
"""
from __future__ import annotations

WEIGHT_VISIBILITY: float   = 0.45
WEIGHT_CITATION: float     = 0.30
WEIGHT_INDEXABILITY: float = 0.25


def compute_citation_score(results: list[dict], target_domain: str) -> float:
    """Citation coverage score (0–100).

    Counts the fraction of query×engine cells where the target domain
    appears at least once in the cited source list.
    """
    from src.scoring import _domain_contains_citation

    if not results:
        return 0.0
    cells_with_citation = sum(
        1 for r in results
        if any(
            _domain_contains_citation(c, target_domain)
            for c in (r.get("citations", []) or [])
        )
    )
    return round(100.0 * cells_with_citation / len(results), 2)


def compute_aeo_score(
    visibility_score: float,
    citation_score: float,
    indexability_score: float,
) -> int:
    """Composite AEO Score (integer, 0–100).

    Formula:
        AEO = 0.45 × Visibility + 0.30 × Citation + 0.25 × Indexability
    """
    raw = (
        WEIGHT_VISIBILITY   * visibility_score
        + WEIGHT_CITATION   * citation_score
        + WEIGHT_INDEXABILITY * indexability_score
    )
    return max(0, min(100, round(raw)))


def compute_all(
    results: list[dict],
    target_domain: str,
    ai_presence_pct: float,
    indexability_score: float,
) -> dict:
    """Compute all three sub-scores and the composite AEO score.

    Args:
        results:           Flat list of query×engine dicts from execute_all().
        target_domain:     The domain being audited.
        ai_presence_pct:   Weighted visibility score from aggregate_results().
        indexability_score: crawl.health_score (0–100).

    Returns dict with keys:
        aeo_score           int   0-100 composite
        visibility_score    float 0-100
        citation_score      float 0-100
        indexability_score  float 0-100
    """
    vis  = round(float(ai_presence_pct), 2)
    cite = compute_citation_score(results, target_domain)
    idx  = round(float(indexability_score), 2)
    aeo  = compute_aeo_score(vis, cite, idx)

    return {
        "aeo_score":          aeo,
        "visibility_score":   vis,
        "citation_score":     cite,
        "indexability_score": idx,
    }


def score_label(score: int) -> str:
    """Return a human-readable label for a 0–100 AEO score."""
    if score >= 75:
        return "Strong"
    if score >= 50:
        return "Emerging"
    if score >= 25:
        return "Limited"
    return "Critical"


def score_color_class(score: int) -> str:
    """Return a CSS modifier class name for the score tier."""
    if score >= 75:
        return "score--strong"
    if score >= 50:
        return "score--emerging"
    if score >= 25:
        return "score--limited"
    return "score--critical"
