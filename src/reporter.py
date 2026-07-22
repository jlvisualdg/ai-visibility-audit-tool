"""
HTML report generator.

Wraps the Jinja2 template with the AuditReport and writes a timestamped
HTML file to the output directory.

v2.0: Now accepts additional v2.0 pipeline variables (ai_presence_pct,
best_brand, best_model, citation_count, best_topic, top_3_brands,
engine_data, competitive_data, topics_to_optimize, global_priorities,
crawl_signals) alongside the existing AuditReport for backward
compatibility with the report.html template.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.analyzer import AuditReport


def _brand_name(domain: str, crawl_title: str = "", crawl_meta: str = "") -> str:
    """Strict brand name extraction from scrape data.

    Priority:
    1. Page title — find phrase that matches the domain (handles tagline titles)
    2. Page title — short, clean separator parts
    3. Meta description — first 1-2 words (filtered for sentence starters)
    4. Domain — humanized fallback

    The brand name is the foundational datapoint for the entire visibility
    audit. If this is wrong, brand mention matching against AI responses
    will fail.
    """
    import re

    bare_domain = re.sub(r"^www\.", "", domain.split(".")[0].lower())
    bare_domain_clean = re.sub(r"[^a-z]", "", bare_domain)

    def _domain_match_in_text(text: str) -> str:
        """Scan text for a phrase that matches/contains the domain bare name.

        Two-pass strategy so an exact match always wins over a partial one:
        Pass 1 — exact: find longest phrase whose normalised form equals the domain.
        Pass 2 — partial: find longest phrase that is an abbreviation of the domain
                  (phrase inside domain) OR extends the domain by ≤5 chars as a prefix
                  (e.g. "Pinder Plotkin LLC" for domain "pinderplotkin").
        Normalises title separators (|, —, -) to spaces before scanning.
        """
        normalised = re.sub(r"[|–—]", " ", text)
        words = normalised.split()

        def _candidates():
            for length in range(min(5, len(words)), 0, -1):
                for i in range(len(words) - length + 1):
                    phrase = " ".join(words[i:i + length])
                    if '?' in phrase or '!' in phrase:
                        continue
                    if not (2 <= len(phrase) <= 40):
                        continue
                    _pn = re.sub(r"\s*&\s*", "and", phrase.lower())
                    phrase_norm = re.sub(r"[^a-z]", "", _pn)
                    if phrase_norm:
                        yield length, phrase, phrase_norm

        # Pass 1: exact match at any length
        for _, phrase, phrase_norm in _candidates():
            if phrase_norm == bare_domain_clean:
                return phrase

        # Pass 2: partial match (longest first)
        for length, phrase, phrase_norm in _candidates():
            if length < 2 or not bare_domain_clean:
                continue
            # phrase is an abbreviation / root contained in the domain
            if len(phrase_norm) >= 4 and phrase_norm in bare_domain_clean:
                return phrase
            # domain is a prefix of the phrase, extended by ≤5 chars (e.g. " LLC")
            if (
                len(bare_domain_clean) >= 4
                and phrase_norm.startswith(bare_domain_clean)
                and len(phrase_norm) - len(bare_domain_clean) <= 5
            ):
                return phrase
        return ""

    # 1a. Try title domain-match
    if crawl_title and bare_domain_clean:
        match = _domain_match_in_text(crawl_title)
        if match:
            return match

    # 1b. Try meta description domain-match (e.g. "Pinder Plotkin has fought…")
    if crawl_meta and bare_domain_clean:
        match = _domain_match_in_text(crawl_meta)
        if match:
            return match

    # 2. Try title: clean separator parts (first or last, picking the shorter/cleaner one)
    if crawl_title:
        title = crawl_title.strip()
        parts = re.split(r'\s*[|]\s*|\s*[-–—]\s*|\s*::\s*', title)
        generic = {"home", "homepage", "welcome", "index", "untitled", "document"}
        candidates = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.lower() in generic:
                continue
            if '?' in part or '!' in part:
                continue
            words = part.split()
            if 1 <= len(words) <= 4 and 2 <= len(part) <= 40:
                candidates.append(part)
        if candidates:
            # Prefer the shortest candidate (brand names are concise)
            return min(candidates, key=len)

    # 3. Try meta description — first 1-2 words, filtered for sentence starters
    _NON_BRAND_STARTS = {
        "injured", "are", "have", "call", "get", "find", "need", "want", "our", "we",
        "if", "do", "did", "been", "lost", "hurt", "experience", "receive", "when",
        "discover", "learn", "explore", "contact", "serving", "providing", "helping",
        "the", "a", "an", "welcome", "trusted",
    }
    if crawl_meta:
        meta = crawl_meta.strip()
        words = meta.split()[:2]
        if words and words[0].lower() not in _NON_BRAND_STARTS:
            candidate = " ".join(words)
            if 2 <= len(candidate) <= 40:
                return candidate

    # 4. Fallback: humanize domain (CamelCase split, then capitalize)
    bare = domain.split(".")[0].lstrip("www")
    # Handle hyphens and underscores
    bare = bare.replace("-", " ").replace("_", " ")
    # CamelCase split
    bare = re.sub(r"([a-z])([A-Z])", r"\1 \2", bare)
    parts = bare.split()
    return " ".join(p.capitalize() for p in parts)


def generate_report(
    report: AuditReport,
    output_dir: str = "output",
    no_ai: bool = False,
    # ── composite AEO score ──
    aeo_score: Optional[int] = None,
    visibility_score: Optional[float] = None,
    citation_score: Optional[float] = None,
    credibility_score: Optional[float] = None,
    indexability_score: Optional[float] = None,
    # ── v2.0 pipeline variables ──
    ai_presence_pct: Optional[float] = None,
    best_brand: Optional[str] = None,
    best_model: Optional[str] = None,
    citation_count: Optional[int] = None,
    best_topic: Optional[str] = None,
    top_3_brands: Optional[list[dict]] = None,
    engine_data: Optional[list[dict]] = None,
    competitive_data: Optional[list[dict]] = None,
    topics_to_optimize: Optional[list[str]] = None,
    global_priorities: Optional[list[str]] = None,
    crawl_signals: Any = None,
    # ── branding overrides ──
    brand_name: Optional[str] = None,
    brand_slogan: Optional[str] = None,
    website_url: Optional[str] = None,
    logo_svg: Optional[str] = None,
    report_date: Optional[str] = None,
) -> str:
    """Render the dark-themed HTML report and write it to `output_dir`.

    Args:
        report: The AuditReport dataclass instance.
        output_dir: Where to write the HTML file.
        no_ai: Whether AI engines were skipped.
        ai_presence_pct: v2.0 — overall AI presence percentage (0-100).
        best_brand: v2.0 — top performing non-target brand.
        best_model: v2.0 — engine with highest AI presence.
        citation_count: v2.0 — total target domain citations.
        best_topic: v2.0 — topic with highest AI presence.
        top_3_brands: v2.0 — top 3 non-target brands [{name, count}, ...].
        engine_data: v2.0 — per-engine breakdown.
        competitive_data: v2.0 — per-topic competitive breakdown.
        topics_to_optimize: v2.0 — queries with zero target presence.
        global_priorities: v2.0 — 3 strategic priority items.
        crawl_signals: v2.0 — CrawlResult (alternative access path).
        brand_name: Override for brand display name.
        brand_slogan: Override for tagline.
        website_url: Override for website URL.
        logo_svg: Override for logo SVG string.
        report_date: Override for report date.

    Returns:
        Absolute path of the generated HTML file.
    """
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.html")

    # Compute defaults
    generated_at = report_date or datetime.now().strftime("%Y-%m-%d %H:%M")

    # Fallback composite score from old formula when not supplied
    if aeo_score is None:
        from src.aeo_score import compute_aeo_score, score_label, score_color_class, bucket_label
        _vis = float(ai_presence_pct or 0)
        _idx = float(getattr(report.crawl_signals, "health_score", 0))
        _cred = float(getattr(report.crawl_signals, "credibility_score", 0))
        aeo_score = compute_aeo_score(_vis, _cred, _idx)
        visibility_score  = _vis
        credibility_score = _cred
        citation_score    = 0.0
        indexability_score = _idx
    else:
        from src.aeo_score import score_label, score_color_class, bucket_label

    _vis_f  = round(float(visibility_score or 0), 1)
    _cred_f = round(float(credibility_score or 0), 1)
    _idx_f  = round(float(indexability_score or 0), 1)

    html = template.render(
        report=report,
        generated_at=generated_at,
        no_ai=no_ai,
        # ── composite AEO score ──
        aeo_score=aeo_score,
        aeo_label=score_label(aeo_score),
        aeo_color_class=score_color_class(aeo_score),
        visibility_score=_vis_f,
        visibility_label=bucket_label(int(_vis_f)),
        credibility_score=_cred_f,
        credibility_label=bucket_label(int(_cred_f)),
        citation_score=round(float(citation_score or 0), 1),
        indexability_score=_idx_f,
        indexability_label=bucket_label(int(_idx_f)),
        # ── branding ──
        brand_name=brand_name or _brand_name(report.domain, getattr(report.crawl_signals, 'title', ''), getattr(report.crawl_signals, 'meta_description', '')),
        brand_slogan=brand_slogan or "Answer Engine Optimization Audit",
        website_url=website_url or f"https://{report.domain}",
        # ── v2.0 verdict metrics ──
        ai_presence_pct=ai_presence_pct,
        best_brand=best_brand,
        best_model=best_model,
        citation_count=citation_count,
        best_topic=best_topic,
        # ── v2.0 derived data ──
        top_3_brands=top_3_brands or [],
        engine_data=engine_data or [],
        competitive_data=competitive_data or [],
        topics_to_optimize=topics_to_optimize or [],
        global_priorities=global_priorities or [],
        crawl_signals=crawl_signals,
        logo_svg=logo_svg,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_domain = report.domain.replace("/", "_").replace(":", "_")
    out_path = out_dir / f"{safe_domain}-audit-{timestamp}.html"
    out_path.write_text(html, encoding="utf-8")

    return str(out_path.resolve())