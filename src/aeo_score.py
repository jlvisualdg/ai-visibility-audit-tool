"""
Composite AEO Score calculator.

    AEO Score = 0.50 × Visibility + 0.25 × Credibility + 0.25 × Indexability

Weights are defined in src/config.py and imported here.
"""
from __future__ import annotations
from src.config import (
    WEIGHT_VISIBILITY, WEIGHT_CREDIBILITY, WEIGHT_INDEXABILITY,
    THRESHOLD_STRONG, THRESHOLD_DEVELOPING,
)

# ---------------------------------------------------------------------------
# Backward-compat aliases (old 3-bucket weights kept for any code that
# imported them directly from this module)
# ---------------------------------------------------------------------------
WEIGHT_CITATION: float = WEIGHT_CREDIBILITY   # legacy alias


def compute_citation_score(results: list[dict], target_domain: str) -> float:
    """Citation coverage score (0–100).  Kept for backward compatibility."""
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
    credibility_score: float,
    indexability_score: float,
) -> int:
    """Composite AEO Score (integer, 0–100).

    Formula:
        AEO = 0.50 × Visibility + 0.25 × Credibility + 0.25 × Indexability
    """
    raw = (
        WEIGHT_VISIBILITY   * visibility_score
        + WEIGHT_CREDIBILITY * credibility_score
        + WEIGHT_INDEXABILITY * indexability_score
    )
    return max(0, min(100, round(raw)))


def compute_all(
    results: list[dict],
    target_domain: str,
    ai_presence_pct: float,
    indexability_score: float,
    credibility_score: float = 0.0,
) -> dict:
    """Compute all three sub-scores and the composite AEO score.

    Args:
        results:            Flat list of query×engine dicts from execute_all().
        target_domain:      The domain being audited.
        ai_presence_pct:    Weighted visibility score from aggregate_results().
        indexability_score: crawl.health_score (0–100).
        credibility_score:  crawl.credibility_score (0–100).

    Returns dict with keys:
        aeo_score           int   0-100 composite
        visibility_score    float 0-100
        credibility_score   float 0-100
        indexability_score  float 0-100
        citation_score      float 0-100  (legacy — same as credibility_score)
    """
    vis  = round(float(ai_presence_pct), 2)
    cred = round(float(credibility_score), 2)
    idx  = round(float(indexability_score), 2)
    aeo  = compute_aeo_score(vis, cred, idx)

    return {
        "aeo_score":          aeo,
        "visibility_score":   vis,
        "credibility_score":  cred,
        "citation_score":     cred,   # legacy alias so existing callers don't break
        "indexability_score": idx,
    }


def score_label(score: int) -> str:
    """Return STRONG / DEVELOPING / CRITICAL label."""
    if score >= THRESHOLD_STRONG:
        return "Strong"
    if score >= THRESHOLD_DEVELOPING:
        return "Developing"
    return "Critical"


def score_color_class(score: int) -> str:
    """Return a CSS modifier class name for the score tier."""
    if score >= THRESHOLD_STRONG:
        return "score--strong"
    if score >= THRESHOLD_DEVELOPING:
        return "score--developing"
    return "score--critical"


def bucket_label(score: int) -> str:
    """Badge label: STRONG / DEVELOPING / CRITICAL."""
    if score >= THRESHOLD_STRONG:
        return "STRONG"
    if score >= THRESHOLD_DEVELOPING:
        return "DEVELOPING"
    return "CRITICAL"
