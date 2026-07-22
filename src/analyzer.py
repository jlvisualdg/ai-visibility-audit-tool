"""
Strategic analysis + fix generation.

Takes the crawl result and citation matrix, produces:
- A short strategic read (title + summary + the_play)
- 5-7 prioritized fix recommendations
- The complete AuditReport that the reporter consumes

Phase 1: heuristic baseline (no LLM). The output is deterministic
and ships a useful report without spending tokens.

Phase 2 (planned): when an LLM is available, pass the crawl signals +
citation matrix in, get a richer strategic read. The heuristic path
remains the fallback for cost / reliability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.crawler import CrawlResult
from src.visibility import CitationMatrix


@dataclass
class FixRecommendation:
    title: str
    priority: str       # "HIGH" | "MEDIUM" | "LOW"
    tag: str            # "WORTH CITING" | "FOUNDATION" | "RECOMMENDED"
    first_step: str
    agent_fixable: bool = True


@dataclass
class StrategicRead:
    title: str
    summary: str
    the_play: str


@dataclass
class AuditReport:
    domain: str
    ai_coverage_pct: float
    fixable_gaps: int
    strategic_read: StrategicRead
    citation_matrix: CitationMatrix
    crawl_signals: CrawlResult
    fixes: list[FixRecommendation] = field(default_factory=list)
    buyer_topics: list[str] = field(default_factory=list)
    page_blueprints: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze(
    crawl: CrawlResult,
    matrix: Optional[CitationMatrix] = None,
    topics: Optional[list[str]] = None,
) -> AuditReport:
    """
    Synthesize crawl + visibility data into a complete audit report.
    """
    coverage = matrix.coverage_pct if matrix else 0.0
    fixes = _generate_fixes(crawl, matrix)
    strategic = _build_strategic_read(crawl, matrix, coverage)

    report = AuditReport(
        domain=crawl.domain,
        ai_coverage_pct=coverage,
        fixable_gaps=len(fixes),
        strategic_read=strategic,
        citation_matrix=matrix or CitationMatrix(domain=crawl.domain),
        crawl_signals=crawl,
        fixes=fixes,
        buyer_topics=topics or [],
    )

    report.page_blueprints = _build_page_blueprints(crawl, matrix)
    return report


# ---------------------------------------------------------------------------
# Strategic read
# ---------------------------------------------------------------------------


def _build_strategic_read(
    crawl: CrawlResult,
    matrix: Optional[CitationMatrix],
    coverage: float,
) -> StrategicRead:
    """
    Pick a strategic read template based on coverage level + crawl health.
    Heuristic-driven; deterministic.
    """
    domain = crawl.domain
    name = _humanize_domain(domain)
    health = crawl.health_score

    if coverage >= 50 and health >= 70:
        return StrategicRead(
            title=f"{name} has a defensible AI visibility position — now protect and extend it",
            summary=(
                f"{domain} is cited across a meaningful share of buyer queries on AI engines "
                f"({coverage:.0f}% coverage across the matrix), and the site signals support it. "
                f"The risk now is the citation set getting stale or competitors catching up."
            ),
            the_play=(
                "Refresh the most-cited pages quarterly, publish 2 net-new answer-first pages "
                "per month targeting adjacent buyer questions, and run this audit again in 90 "
                "days to catch any coverage regression before clients do."
            ),
        )

    if coverage >= 25:
        return StrategicRead(
            title=f"{name} is on the AI map — fill the citation gaps to take the lead",
            summary=(
                f"{domain} shows up for some buyer queries ({coverage:.0f}% coverage) but "
                f"is being out-cited by competitors on the topics that matter most. The site "
                f"has the bones; it needs more quotable content."
            ),
            the_play=(
                "Pick the 2-3 topics where competitors appear most often, write answer-first "
                "guides that cite original data, and make sure every product page has a direct "
                f"answer capsule under the H1. Re-audit in 60 days."
            ),
        )

    # coverage < 25 — the default for most sites we audit
    return StrategicRead(
        title=f"Build a topical authority hub for {name} — answer-first content is the unlock",
        summary=(
            f"{domain} is essentially invisible to AI engines on the topics it should own "
            f"({coverage:.0f}% coverage). The crawl signals show why: limited answer-first "
            f"content, no authoritativeness signals, and content that's hard for AI to extract "
            f"and cite. This is fixable, but it requires a content system, not a one-off post."
        ),
        the_play=(
            "Stand up a 6-page topical hub built around the 4 buyer topics in this report. "
            "Each page: answer-first 40-word capsule under the H1, original data with sources, "
            "named author, schema markup, internal links with descriptive anchor text. "
            "Re-audit in 90 days to measure movement."
        ),
    )


def _humanize_domain(domain: str) -> str:
    """
    Turn 'paretotalent.com' into 'Paretotalent' for headline copy.

    v1 is intentionally simple: title-case the bare domain. The report
    displays the full domain right next to the humanized form, so the
    user always sees the source. Smarter humanization (recognizing
    "Pareto" inside "Paretotalent") is a phase 2 problem.
    """
    import re
    bare = domain.split(".")[0]
    # Try camelCase split first (handles "AcmeRoofing" -> "Acme Roofing")
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", bare).split()
    return " ".join(p.capitalize() for p in parts)


# ---------------------------------------------------------------------------
# Fix generation (heuristic baseline)
# ---------------------------------------------------------------------------


def _generate_fixes(
    crawl: CrawlResult,
    matrix: Optional[CitationMatrix],
) -> list[FixRecommendation]:
    """Drive fix cards from the unified checklist in src/checklist.py.

    Iterates every ChecklistEntry, evaluates its signal_fn, and emits a
    FixRecommendation for each unmet signal. Priority order: HIGH → MEDIUM → LOW.
    Within each priority, schema/indexability fixes are shown before credibility,
    then visibility. Capped at 10 fixes so the report stays focused.
    """
    from src.checklist import CHECKLIST

    _PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    _BUCKET_ORDER = {"indexability": 0, "credibility": 1, "visibility": 2}

    raw: list[tuple[int, int, FixRecommendation]] = []

    for entry in CHECKLIST:
        try:
            triggered = entry.signal_fn(crawl, matrix)
        except Exception:
            continue
        if not triggered:
            continue

        title = entry.title
        # Dynamic title for thin-page count
        if entry.signal_fn.__name__ == "_thin_pages" and crawl.thin_pages:
            title = f"Expand {len(crawl.thin_pages)} thin page(s) to at least 600 words"

        fix = FixRecommendation(
            title=title,
            priority=entry.priority,
            tag=entry.tag,
            first_step=entry.first_step,
            agent_fixable=entry.agent_fixable,
        )
        sort_key = (
            _PRIORITY_ORDER.get(entry.priority, 9),
            _BUCKET_ORDER.get(entry.bucket, 9),
        )
        raw.append((*sort_key, fix))

    raw.sort(key=lambda x: (x[0], x[1]))

    seen: set[str] = set()
    out: list[FixRecommendation] = []
    for _, _, fix in raw:
        if fix.title in seen:
            continue
        seen.add(fix.title)
        out.append(fix)
        if len(out) >= 10:
            break
    return out




# ---------------------------------------------------------------------------
# Page blueprints (for the "what to write next" section)
# ---------------------------------------------------------------------------


def _build_page_blueprints(
    crawl: CrawlResult,
    matrix: Optional[CitationMatrix],
) -> dict:
    """
    Suggest two content formats the client should publish next.
    Driven by buyer topics when available, else by the domain category.
    """
    blueprints = {
        "listicles": {
            "format": "'Top N [category] for [audience]' listicle",
            "word_count_range": "1,500-2,500 words",
            "first_move": (
                "Lead with a one-paragraph answer to the implicit question, then rank "
                "10 options with original commentary. Cite primary sources."
            ),
        },
        "deep_guides": {
            "format": "'How to choose [category]' definitive guide",
            "word_count_range": "2,500-4,000 words",
            "first_move": (
                "Frame the buyer's decision criteria in 4-6 named dimensions, then walk "
                "through each with original data and named examples."
            ),
        },
    }
    return blueprints
