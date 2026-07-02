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
    """
    Apply the v1 fix rules from the plan. Order: HIGH first, then MEDIUM, LOW.
    Capped at 7 fixes so the report stays focused.
    """
    fixes: list[FixRecommendation] = []

    # HIGH-priority: zero AI coverage on a real site
    if matrix is not None and matrix.coverage_pct == 0 and matrix.total_cells > 0:
        fixes.append(FixRecommendation(
            title="Build a topical authority hub with answer-first content",
            priority="HIGH",
            tag="WORTH CITING",
            first_step=(
                "Pick the 2 buyer topics with the most competitor citations and write a "
                "1,200-word answer-first guide for each. Lead with a 40-word direct answer "
                "under the H1, then expand with original data and named sources."
            ),
            agent_fixable=True,
        ))

    # HIGH: no answer capsules
    if crawl.answer_capsules == 0 and crawl.total_pages >= 3:
        fixes.append(FixRecommendation(
            title="Add answer-first sections to key pages",
            priority="HIGH",
            tag="WORTH CITING",
            first_step=(
                "For each top-10 page by traffic, add a 40-word direct answer under the H1 "
                "before the first H2. AI engines preferentially cite content that answers "
                "the question in the first paragraph."
            ),
            agent_fixable=True,
        ))

    # HIGH: thin pages
    if crawl.thin_pages:
        fixes.append(FixRecommendation(
            title=f"Expand {len(crawl.thin_pages)} thin pages to at least 600 words",
            priority="HIGH",
            tag="FOUNDATION",
            first_step=(
                "Audit the thin pages: which ones are intentionally short (e.g. contact, "
                "careers) and which are stub content? Expand the stubs to 600+ words with "
                "an answer-first section, original data, and 2-3 internal links."
            ),
            agent_fixable=True,
        ))

    # MEDIUM: missing anchor text
    if crawl.missing_anchor_text:
        fixes.append(FixRecommendation(
            title="Replace generic anchor text on internal links",
            priority="MEDIUM",
            tag="FOUNDATION",
            first_step=(
                "Search the site for 'click here', 'read more', 'learn more', and bare URLs. "
                "Replace with descriptive phrases that include the target page's primary "
                "keyword. AI engines use anchor text to understand topical relationships."
            ),
            agent_fixable=True,
        ))

    # MEDIUM: no schema
    if crawl.schema_pages == 0 and crawl.total_pages >= 3:
        fixes.append(FixRecommendation(
            title="Add JSON-LD schema markup to all key pages",
            priority="MEDIUM",
            tag="FOUNDATION",
            first_step=(
                "Add Organization schema to the homepage, Article + Person schema to blog "
                "posts, and Product/Service schema to offering pages. Use Google's Rich "
                "Results Test to validate."
            ),
            agent_fixable=True,
        ))

    # MEDIUM: no authorship
    if crawl.authorship_pages == 0 and crawl.total_pages >= 3:
        fixes.append(FixRecommendation(
            title="Add visible author bylines to content pages",
            priority="MEDIUM",
            tag="RECOMMENDED",
            first_step=(
                "Add an author byline (with photo + bio + Person schema) to every blog "
                "post and guide. AI engines weight named authorship as a trust signal, "
                "and YMYL queries in particular require it."
            ),
            agent_fixable=True,
        ))

    # MEDIUM: question headings without answers
    if crawl.question_headings and crawl.answer_capsules == 0:
        fixes.append(FixRecommendation(
            title="Convert question headings into answer-first sections",
            priority="MEDIUM",
            tag="WORTH CITING",
            first_step=(
                "The site already uses question-style H2s — good. But none of them have a "
                "direct answer paragraph under them. Add a 2-3 sentence answer immediately "
                "after each question heading."
            ),
            agent_fixable=True,
        ))

    # LOW: no ai.txt / llms_txt
    if not crawl.has_ai_txt:
        fixes.append(FixRecommendation(
            title="Publish an /ai.txt access policy",
            priority="LOW",
            tag="RECOMMENDED",
            first_step=(
                "Create /ai.txt declaring your policy for AI crawlers (GPTBot, ClaudeBot, "
                "PerplexityBot, Google-Extended). Even a permissive policy is a signal — "
                "see aietxt.org for the spec."
            ),
            agent_fixable=True,
        ))

    if not crawl.has_llms_txt:
        fixes.append(FixRecommendation(
            title="Publish a /llms.txt site summary",
            priority="LOW",
            tag="RECOMMENDED",
            first_step=(
                "Create /llms.txt with a markdown summary of your site, your top products, "
                "and your key pages. This is the format LLM-aware tools will use to "
                "discover and cite your content."
            ),
            agent_fixable=True,
        ))

    # CRITICAL: blocks AI
    if crawl.robots_blocks_ai:
        fixes.append(FixRecommendation(
            title="Unblock major AI crawlers in robots.txt",
            priority="HIGH",
            tag="CRITICAL",
            first_step=(
                "Your robots.txt currently disallows one or more of GPTBot, ClaudeBot, "
                "or PerplexityBot. This is the single biggest reason AI engines can't "
                "cite your content. Add explicit Allow rules for these user-agents."
            ),
            agent_fixable=True,
        ))

    # Sort by priority then de-dup titles
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    fixes.sort(key=lambda f: (priority_order.get(f.priority, 9), f.title))
    # De-dup
    seen: set[str] = set()
    out: list[FixRecommendation] = []
    for f in fixes:
        if f.title in seen:
            continue
        seen.add(f.title)
        out.append(f)
    return out[:7]


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
