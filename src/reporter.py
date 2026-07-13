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
    1. Page title — split on separators (|, -, —, ::), take first part
    2. Meta description — first 1-2 words if title fails
    3. Domain — humanized fallback only

    The brand name is the foundational datapoint for the entire visibility
    audit. If this is wrong, brand mention matching against AI responses
    will fail.
    """
    import re

    # 1. Try page title first
    if crawl_title:
        title = crawl_title.strip()
        # Split on common title separators
        parts = re.split(r'\s*[|]\s*|\s*[-–—]\s*|\s*::\s*', title)
        if parts and parts[0].strip():
            candidate = parts[0].strip()
            # Reject generic words
            generic = {"home", "homepage", "welcome", "index", "untitled", "document"}
            if candidate.lower() not in generic and 2 <= len(candidate) <= 40:
                return candidate

    # 2. Try meta description
    if crawl_meta:
        meta = crawl_meta.strip()
        # Take first 1-2 words that look like a brand
        words = meta.split()[:2]
        if words:
            candidate = " ".join(words)
            if (2 <= len(candidate) <= 40
                    and not candidate.lower().startswith(("the ", "a ", "an ", "welcome"))):
                return candidate

    # 3. Fallback: humanize domain
    bare = domain.split(".")[0]
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", bare).split()
    return " ".join(p.capitalize() for p in parts)


def generate_report(
    report: AuditReport,
    output_dir: str = "output",
    no_ai: bool = False,
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

    html = template.render(
        report=report,
        generated_at=generated_at,
        no_ai=no_ai,
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