"""
Unified signal → checklist map.

Each entry declares:
  signal_fn  : callable(crawl, matrix) -> bool   True = signal ABSENT (fix needed)
  title      : fix card title
  priority   : "HIGH" | "MEDIUM" | "LOW"
  tag        : "WORTH CITING" | "FOUNDATION" | "RECOMMENDED" | "CRITICAL"
  bucket     : "visibility" | "credibility" | "indexability"
  page_scope : short string shown in the fix card ("Homepage", "All pages", etc.)
  first_step : actionable first step copy
  agent_fixable : bool

_generate_fixes in analyzer.py iterates this table and emits a FixRecommendation
for every entry whose signal_fn returns True. This guarantees every extracted
signal — including new schema/NEEATT signals — produces a page-scoped fix card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.crawler import CrawlResult
    from src.visibility import CitationMatrix


@dataclass
class ChecklistEntry:
    signal_fn: Callable[["CrawlResult", Optional["CitationMatrix"]], bool]
    title: str
    priority: str       # "HIGH" | "MEDIUM" | "LOW"
    tag: str
    bucket: str         # "visibility" | "credibility" | "indexability"
    page_scope: str
    first_step: str
    agent_fixable: bool = True


# ---------------------------------------------------------------------------
# Visibility fixes
# ---------------------------------------------------------------------------

def _zero_coverage(crawl, matrix) -> bool:
    return matrix is not None and matrix.coverage_pct == 0 and matrix.total_cells > 0

def _no_answer_capsules(crawl, matrix) -> bool:
    return crawl.answer_capsules == 0 and crawl.total_pages >= 3

def _thin_pages(crawl, matrix) -> bool:
    return bool(crawl.thin_pages)

def _question_headings_no_answers(crawl, matrix) -> bool:
    return bool(crawl.question_headings) and crawl.answer_capsules == 0

def _missing_anchor_text(crawl, matrix) -> bool:
    return bool(crawl.missing_anchor_text)

def _no_llms_txt(crawl, matrix) -> bool:
    return not crawl.has_llms_txt

def _blocks_ai(crawl, matrix) -> bool:
    return crawl.robots_blocks_ai


# ---------------------------------------------------------------------------
# Indexability fixes (schema, technical, authorship)
# ---------------------------------------------------------------------------

def _no_schema(crawl, matrix) -> bool:
    return crawl.schema_quality_score == 0 and crawl.total_pages >= 2

def _poor_schema(crawl, matrix) -> bool:
    return 0 < crawl.schema_quality_score < 40

def _weak_schema(crawl, matrix) -> bool:
    return 40 <= crawl.schema_quality_score < 70

def _schema_missing_org(crawl, matrix) -> bool:
    return any("Organization" in p for p in crawl.schema_missing_props)

def _schema_missing_sameas(crawl, matrix) -> bool:
    return any("sameAs" in p for p in crawl.schema_missing_props)

def _schema_missing_website(crawl, matrix) -> bool:
    return any("WebSite" in p for p in crawl.schema_missing_props)

def _no_authorship(crawl, matrix) -> bool:
    return crawl.authorship_pages == 0 and crawl.total_pages >= 3

def _partial_authorship(crawl, matrix) -> bool:
    return (crawl.authorship_pages > 0
            and crawl.authorship_pages < crawl.total_pages // 2
            and crawl.total_pages >= 4)

def _slow_response(crawl, matrix) -> bool:
    return crawl.avg_response_ms > 3000

def _moderate_response(crawl, matrix) -> bool:
    return 1000 < crawl.avg_response_ms <= 3000

def _no_ssl(crawl, matrix) -> bool:
    return not crawl.has_ssl


# ---------------------------------------------------------------------------
# Credibility fixes (NEEATT)
# ---------------------------------------------------------------------------

def _no_about_page(crawl, matrix) -> bool:
    return not crawl.has_about_page

def _no_contact_page(crawl, matrix) -> bool:
    return not crawl.has_contact_page

def _no_nap(crawl, matrix) -> bool:
    return not crawl.has_contact_info_on_homepage

def _no_trust_signals(crawl, matrix) -> bool:
    return not crawl.has_trust_signals_on_homepage

def _no_ratings(crawl, matrix) -> bool:
    # Only flag if no trust signals of any kind
    return (not getattr(crawl, "has_ratings_widget", False)
            and not crawl.has_trust_signals_on_homepage)

def _no_privacy_policy(crawl, matrix) -> bool:
    return not crawl.has_privacy_policy

def _no_terms(crawl, matrix) -> bool:
    return not crawl.has_terms

def _no_social(crawl, matrix) -> bool:
    return crawl.social_channel_count == 0

def _weak_social(crawl, matrix) -> bool:
    return 1 <= crawl.social_channel_count < 3

def _no_media_mentions(crawl, matrix) -> bool:
    return not getattr(crawl, "has_media_mentions", False)

def _no_stat_counters(crawl, matrix) -> bool:
    return not getattr(crawl, "has_stat_counters", False)

def _no_people_bios(crawl, matrix) -> bool:
    return not crawl.people  # no people extracted at all

def _no_staff_credentials(crawl, matrix) -> bool:
    return (bool(crawl.people)
            and getattr(crawl, "people_with_credentials", 0) == 0)


# ---------------------------------------------------------------------------
# Master table — ordered HIGH → MEDIUM → LOW within each bucket
# ---------------------------------------------------------------------------

CHECKLIST: list[ChecklistEntry] = [

    # ── CRITICAL / HIGH — Visibility ────────────────────────────────────────
    ChecklistEntry(
        signal_fn=_blocks_ai,
        title="Unblock major AI crawlers in robots.txt",
        priority="HIGH",
        tag="CRITICAL",
        bucket="indexability",
        page_scope="robots.txt",
        first_step=(
            "Add explicit Allow: / rules for GPTBot, ClaudeBot, PerplexityBot, "
            "and Google-Extended in robots.txt. Each blocked crawler is a citation "
            "channel permanently closed off."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_ssl,
        title="Migrate to HTTPS",
        priority="HIGH",
        tag="CRITICAL",
        bucket="indexability",
        page_scope="Entire site",
        first_step=(
            "Install an SSL certificate and enforce HTTPS redirects site-wide. "
            "AI crawlers and modern browsers flag HTTP sites as insecure; many will "
            "refuse to index or cite non-HTTPS content."
        ),
    ),
    ChecklistEntry(
        signal_fn=_zero_coverage,
        title="Build a topical authority hub with answer-first content",
        priority="HIGH",
        tag="WORTH CITING",
        bucket="visibility",
        page_scope="New pages",
        first_step=(
            "Pick the 2 buyer topics with the most competitor citations and write a "
            "1,200-word answer-first guide for each. Lead with a 40-word direct answer "
            "under the H1, then expand with original data and named sources."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_answer_capsules,
        title="Add answer-first sections to key pages",
        priority="HIGH",
        tag="WORTH CITING",
        bucket="visibility",
        page_scope="Top content pages",
        first_step=(
            "For each top-10 page by traffic, add a 40-word direct answer under the H1 "
            "before the first H2. AI engines preferentially cite content that answers "
            "the question in the first paragraph."
        ),
    ),
    ChecklistEntry(
        signal_fn=_thin_pages,
        title="Expand thin pages to at least 600 words",
        priority="HIGH",
        tag="FOUNDATION",
        bucket="visibility",
        page_scope="Thin pages",
        first_step=(
            "Audit thin pages: which are intentionally short (contact, careers) and "
            "which are stub content? Expand the stubs to 600+ words with an answer-first "
            "section, original data, and 2-3 internal links."
        ),
    ),

    # ── HIGH — Schema ─────────────────────────────────────────────────────
    ChecklistEntry(
        signal_fn=_no_schema,
        title="Add JSON-LD entity schema to the homepage",
        priority="HIGH",
        tag="FOUNDATION",
        bucket="indexability",
        page_scope="Homepage",
        first_step=(
            "Add at minimum an Organization schema block to the homepage with name, url, "
            "logo, contactPoint, and sameAs links to social profiles. Without it AI engines "
            "cannot establish entity identity for this domain."
        ),
    ),
    ChecklistEntry(
        signal_fn=_poor_schema,
        title="Rebuild homepage schema — entity graph is incomplete",
        priority="HIGH",
        tag="FOUNDATION",
        bucket="indexability",
        page_scope="Homepage",
        first_step=(
            "Your schema has critical gaps. Add Organization + WebSite schemas connected "
            "via @id cross-references. Each entity needs a unique @id URI and sameAs links "
            "to social profiles and Wikidata. Use Google's Rich Results Test to validate."
        ),
    ),

    # ── MEDIUM — Schema specifics ────────────────────────────────────────
    ChecklistEntry(
        signal_fn=_schema_missing_org,
        title="Add Organization schema to homepage",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="indexability",
        page_scope="Homepage",
        first_step=(
            "Add an Organization JSON-LD block with: @id (your domain URL), name, url, "
            "logo, contactPoint, and sameAs pointing to LinkedIn, Facebook, and Wikidata. "
            "This is the foundation of your entity knowledge graph."
        ),
    ),
    ChecklistEntry(
        signal_fn=_schema_missing_sameas,
        title="Add sameAs links to Organization schema",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="indexability",
        page_scope="Homepage",
        first_step=(
            "Add a sameAs property to your Organization schema listing your verified "
            "social profiles (LinkedIn, Facebook, Twitter/X) and any entity databases "
            "(Wikidata, Crunchbase). This lets AI engines reconcile your identity across sources."
        ),
    ),
    ChecklistEntry(
        signal_fn=_schema_missing_website,
        title="Add WebSite schema linked to Organization",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="indexability",
        page_scope="Homepage",
        first_step=(
            "Add a WebSite JSON-LD block with name, url, and a publisher property that "
            "references your Organization @id. Optionally add a potentialAction SearchAction "
            "if the site has internal search."
        ),
    ),
    ChecklistEntry(
        signal_fn=_weak_schema,
        title="Complete homepage schema — missing key properties",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="indexability",
        page_scope="Homepage",
        first_step=(
            "Schema present but incomplete. Prioritize: Organization.sameAs (social + entity "
            "DB links), Organization.logo, WebSite with publisher cross-reference, and @id "
            "on every entity block. Each gap is a missed entity reconciliation opportunity."
        ),
    ),

    # ── MEDIUM — Authorship ───────────────────────────────────────────────
    ChecklistEntry(
        signal_fn=_no_authorship,
        title="Add visible author bylines to content pages",
        priority="MEDIUM",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="Blog / guide pages",
        first_step=(
            "Add an author byline (with photo, bio, and Person schema) to every blog post "
            "and guide. AI engines weight named authorship as a trust signal — YMYL queries "
            "in particular require demonstrated expertise."
        ),
    ),
    ChecklistEntry(
        signal_fn=_partial_authorship,
        title="Extend author bylines to all content pages",
        priority="MEDIUM",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="Content pages",
        first_step=(
            "Authorship is present on some pages but not consistent. Add author bylines "
            "to all blog posts, guides, and case studies. Consistency signals a deliberate "
            "editorial standard, which AI engines reward."
        ),
    ),

    # ── MEDIUM — Visibility content ────────────────────────────────────────
    ChecklistEntry(
        signal_fn=_question_headings_no_answers,
        title="Convert question headings into answer-first sections",
        priority="MEDIUM",
        tag="WORTH CITING",
        bucket="visibility",
        page_scope="Content pages",
        first_step=(
            "The site uses question-style H2s — good. But none have a direct answer "
            "paragraph underneath. Add a 2-3 sentence direct answer immediately after "
            "each question heading so AI engines can extract and cite it."
        ),
    ),
    ChecklistEntry(
        signal_fn=_missing_anchor_text,
        title="Replace generic anchor text on internal links",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="indexability",
        page_scope="All pages",
        first_step=(
            "Find 'click here', 'read more', 'learn more', and bare URLs across the site. "
            "Replace with descriptive phrases that include the target page's primary keyword. "
            "AI engines use anchor text to understand topical relationships."
        ),
    ),

    # ── MEDIUM — Credibility / NEEATT ──────────────────────────────────────
    ChecklistEntry(
        signal_fn=_no_about_page,
        title="Create an About page for entity grounding",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="credibility",
        page_scope="About page (new)",
        first_step=(
            "AI engines use the About page as the primary entity anchor for a domain. "
            "Include: company founding story and timeline, team bios with credentials, "
            "mission statement, and links to press mentions or external entity profiles."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_nap,
        title="Add NAP (name, address, phone) to homepage footer",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="credibility",
        page_scope="Homepage footer",
        first_step=(
            "Place your full business name, address, and phone number (or email) in the "
            "homepage footer. NAP consistency across your site and Google Business Profile "
            "strengthens entity recognition for local and AI searches."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_trust_signals,
        title="Add social proof to homepage (reviews, ratings, logos)",
        priority="MEDIUM",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="Homepage",
        first_step=(
            "Add at least one of: a Trustpilot/Google review widget, client logos with "
            "testimonials, or a project/customer counter (e.g. '1,200+ homes built'). "
            "Social proof signals meaningfully influence AI recommendation likelihood."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_privacy_policy,
        title="Add a privacy policy page",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="credibility",
        page_scope="Footer link",
        first_step=(
            "Create a /privacy page and link it in the site footer. Required for regulatory "
            "compliance and a baseline Transparency signal that AI entity evaluators check."
        ),
    ),

    # ── LOW — Credibility / NEEATT ─────────────────────────────────────────
    ChecklistEntry(
        signal_fn=_no_contact_page,
        title="Add a dedicated Contact page",
        priority="LOW",
        tag="FOUNDATION",
        bucket="credibility",
        page_scope="Contact page (new)",
        first_step=(
            "Create a /contact page with at minimum a contact form, phone number, and "
            "business address. Link it from the main navigation and footer."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_ratings,
        title="Add a third-party ratings widget",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="Homepage",
        first_step=(
            "Embed a Trustpilot, Google Business Profile, or BBB widget that shows "
            "your aggregate rating. Third-party ratings are the Authoritativeness signal "
            "AI engines can cross-reference independently."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_terms,
        title="Add terms of service",
        priority="LOW",
        tag="FOUNDATION",
        bucket="credibility",
        page_scope="Footer link",
        first_step=(
            "Create a /terms page and link it in the site footer alongside the privacy "
            "policy. Baseline legal transparency signal for entity evaluation."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_social,
        title="Create and link social media profiles",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="Homepage footer",
        first_step=(
            "Create profiles on LinkedIn and at least one other relevant platform, then "
            "link them from the homepage footer. Social profiles serve as external entity "
            "verification anchors (sameAs targets) for AI knowledge graphs."
        ),
    ),
    ChecklistEntry(
        signal_fn=_weak_social,
        title="Build out social presence to 3+ platforms",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="Homepage footer",
        first_step=(
            "You have social links but fewer than 3 platforms. Add LinkedIn, Facebook/Meta, "
            "and a third relevant platform (YouTube for video, X/Twitter for B2B). More "
            "verified profiles = more sameAs anchors for AI entity reconciliation."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_media_mentions,
        title="Build Notability signals — press mentions and awards",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="Homepage",
        first_step=(
            "Add an 'As Seen In' or 'Press' section above the fold with media logos or "
            "award badges. Even one credible press mention significantly lifts AI-engine "
            "perceived authority (Notability dimension of NEEATT)."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_stat_counters,
        title="Add quantified experience indicators",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="Homepage",
        first_step=(
            "Add concrete counters like '15 years in business', '500+ clients served', or "
            "'$10M in projects delivered'. These Experience signals give AI engines "
            "verifiable claims to cite when recommending your business."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_people_bios,
        title="Add team/founder profiles to the About page",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="About page",
        first_step=(
            "Add named team profiles with photo, role, and a 2-3 sentence bio. People "
            "entities are how AI engines establish Expertise — a faceless company is "
            "harder to recommend than one with named, credentialed humans behind it."
        ),
    ),
    ChecklistEntry(
        signal_fn=_no_staff_credentials,
        title="Add credentials to team bios",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="credibility",
        page_scope="About page",
        first_step=(
            "For each team member bio, add their relevant degree, license, or certification "
            "(e.g. 'CPA', 'Licensed Contractor', 'MBA'). Credentials are the Expertise "
            "dimension in NEEATT and directly influence AI recommendation for professional services."
        ),
    ),

    # ── LOW — Technical / Visibility ──────────────────────────────────────
    ChecklistEntry(
        signal_fn=_no_llms_txt,
        title="Publish a /llms.txt site summary",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="indexability",
        page_scope="/llms.txt",
        first_step=(
            "Create /llms.txt with a markdown summary of your site, top products/services, "
            "and key page URLs. This is the emerging standard format LLM-aware tools use "
            "to discover and correctly cite your content."
        ),
    ),
    ChecklistEntry(
        signal_fn=_slow_response,
        title="Reduce server response time below 3 seconds",
        priority="MEDIUM",
        tag="FOUNDATION",
        bucket="indexability",
        page_scope="Entire site",
        first_step=(
            "Profile and fix the slowest response bottleneck — typically database queries, "
            "unoptimized images, or no CDN. AI crawlers time out at ~5s; slow responses "
            "mean incomplete indexing and missed citation opportunities."
        ),
    ),
    ChecklistEntry(
        signal_fn=_moderate_response,
        title="Optimize server response time (currently over 1 second)",
        priority="LOW",
        tag="RECOMMENDED",
        bucket="indexability",
        page_scope="Entire site",
        first_step=(
            "Response times over 1s risk AI crawler timeouts on large sites. Enable server "
            "caching, optimize the largest images, and consider a CDN for static assets. "
            "Target under 500ms for reliably complete AI indexing."
        ),
    ),
]
