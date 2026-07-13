#!/usr/bin/env python3
"""
AI Visibility Audit Tool — CLI entry point.

Examples:

    # Pure mock mode (no API keys)
    python audit.py --domain example.com --no-ai --max-pages 3

    # Real audit with Perplexity via OpenRouter
    OPENROUTER_API_KEY=sk-or-v1-... python audit.py --domain paretotalent.com

    # Custom output dir
    python audit.py --domain example.com --no-ai --output ./reports

The report lands at: <output>/<domain>-audit-<timestamp>.html
"""

import random
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.analyzer import analyze
from src.crawler import crawl_domain
from src.reporter import generate_report

console = Console()

# ---------------------------------------------------------------------------
# v2.0 — Mock result generator (for --no-ai flag)
# ---------------------------------------------------------------------------

# Simulated brand sets per engine for mock mode — varied so the report
# shows realistic competitive differentiation across engines.
_MOCK_ENGINE_BRANDS: dict[str, list[str]] = {
    "Perplexity": [
        "Brand Alpha", "Brand Beta", "Brand Gamma", "Brand Delta",
        "Brand Epsilon", "Brand Zeta", "Brand Eta", "Brand Theta",
    ],
    "ChatGPT": [
        "Brand Epsilon", "Brand Iota", "Brand Gamma", "Brand Kappa",
        "Brand Alpha", "Brand Lambda", "Brand Mu", "Brand Nu",
    ],
    "Gemini": [
        "Brand Beta", "Brand Alpha", "Brand Iota", "Brand Nu",
        "Brand Kappa", "Brand Epsilon", "Brand Lambda", "Brand Zeta",
    ],
}

# Which topics does the target brand appear in (per engine, for mock)?
_MOCK_TARGET_PRESENCE: dict[str, set[int]] = {
    "Perplexity": {0, 2},       # topics index 0 and 2
    "ChatGPT": {0, 1},          # topics 0 and 1
    "Gemini": {1, 3},           # topics 1 and 3
}


def _generate_mock_results(domain: str, topics: list[str]) -> list[dict]:
    """Generate synthetic results (4 topics × 3 engines) for --no-ai mode.

    Each result matches the dict shape produced by ``execute_all()``.
    Varied presence across engines/topics so the report shows realistic patterns.
    """
    engines = ["Perplexity", "ChatGPT", "Gemini"]
    target_tokens = _extract_target_tokens(domain)
    target_brand = " ".join(w.capitalize() for w in target_tokens[0].split()) if target_tokens else domain.split(".")[0].title()

    results: list[dict] = []
    rng = random.Random(42)  # deterministic seed for reproducibility

    for engine in engines:
        brands = list(_MOCK_ENGINE_BRANDS.get(engine, ["Acme Corp", "Global Inc"]))
        presence_set = _MOCK_TARGET_PRESENCE.get(engine, set())

        for i, topic in enumerate(topics):
            topic_has_target = i in presence_set

            # Build brand mentions list — target brand first when present
            mentions = []
            if topic_has_target:
                mentions.append(target_brand)

            # Add 4-6 other brands (shuffled, varying per topic)
            topic_brands = list(brands)
            rng.shuffle(topic_brands)
            n_brands = rng.randint(4, 6)
            for b in topic_brands[:n_brands]:
                if b != target_brand:
                    mentions.append(b)

            # Positions of target brand (1-indexed)
            positions = [1] if topic_has_target else []

            # Generate fake citation domains
            citation_domains = [
                b.lower().replace(" ", "-") + ".com" for b in mentions[1:5]
            ]
            if topic_has_target:
                citation_domains.insert(0, domain)

            # Fake response text
            text_lines = [f"Here are the top vendors for {topic}:"]
            for j, b in enumerate(mentions, 1):
                text_lines.append(f"{j}. {b} — A leading provider in this space.")
            text = "\n".join(text_lines)

            results.append({
                "topic": topic,
                "engine": engine,
                "text": text,
                "citations": citation_domains,
                "latency_ms": rng.randint(80, 250),
                "error": None,
                "brand_mentions": mentions,
                "positions": positions,
                "target_mention_count": len(positions),
            })

    return results


def _extract_target_tokens(domain: str) -> list[str]:
    """Extract search tokens from a domain (same logic as scoring.py)."""
    import re
    if not domain:
        return []
    bare = _normalize_domain(domain).split(".")[0]
    tokens = [bare.lower()]
    if "-" in bare:
        tokens.append(bare.replace("-", " ").lower())
    common_suffixes = [
        "talent", "roofing", "corp", "inc", "group", "solutions", "tech",
        "software", "media", "consulting", "capital", "partners", "ventures",
        "labs", "studios", "agency", "creative", "marketing", "digital",
        "design", "studio", "systems", "cloud", "data", "health", "care",
        "finance", "legal", "education", "energy", "food", "travel",
    ]
    for suffix in common_suffixes:
        if bare.lower().endswith(suffix) and len(bare) > len(suffix) + 1:
            prefix = bare[:-len(suffix)]
            tokens.append(f"{prefix} {suffix}".lower())
            if len(prefix) >= 3:
                tokens.append(prefix.lower())
            break
    return tokens


def _normalize_domain(d: str) -> str:
    """Strip scheme, path, and www. prefix."""
    d = (d or "").strip().lower()
    for prefix in ("https://", "http://"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.split("/")[0]
    if d.startswith("www."):
        d = d[4:]
    return d


# ---------------------------------------------------------------------------
# v2.0 — Build CitationMatrix from flat execute_all results
# ---------------------------------------------------------------------------

def _build_citation_matrix_from_results(
    results: list[dict],
    domain: str,
    topics: list[str],
) -> "CitationMatrix":
    """Convert flat v2.0 results into a CitationMatrix the template expects.

    Uses lowercase engine names (e.g. 'perplexity') to match template's
    selectattr filters on engine name.
    """
    from src.visibility import CitationMatrix, TopicResult

    # Map display names back to lowercase for template compatibility
    _ENGINE_DISPLAY_TO_LOWERCASE = {
        "Perplexity": "perplexity",
        "ChatGPT": "chatgpt",
        "Claude": "claude",
        "Gemini": "gemini",
    }

    engines = list(dict.fromkeys(
        _ENGINE_DISPLAY_TO_LOWERCASE.get(r["engine"], r["engine"].lower())
        for r in results
    ))
    matrix = CitationMatrix(domain=domain, topics=topics, engines=engines)

    target_tokens = _extract_target_tokens(domain)

    for r in results:
        engine_key = _ENGINE_DISPLAY_TO_LOWERCASE.get(r["engine"], r["engine"].lower())
        target_mentioned = r.get("target_mention_count", 0) > 0
        citations = r.get("citations", []) or []
        brand_mentions = r.get("brand_mentions", []) or []
        positions = r.get("positions", []) or []

        # Find top competitor (first non-target brand in mentions)
        top_competitor = None
        for b in brand_mentions:
            if not _is_target_brand(b, target_tokens):
                top_competitor = b
                break

        # Count URL citations containing target domain
        url_citation_count = sum(
            1 for c in citations if _domain_contains_citation(c, domain)
        )

        tr = TopicResult(
            topic=r["topic"],
            engine=engine_key,
            pass_count=1,
            covered=target_mentioned,
            cited_sources=citations,
            passes=[],
            best_brand_mentions=r.get("target_mention_count", 0),
            best_url_citations=url_citation_count,
            best_first_section=positions[0] if positions else None,
            top_competitor=top_competitor,
        )
        matrix.results.append(tr)

    # Build all_competitors — count non-target brand appearances across all results
    competitor_counts: dict[str, int] = {}
    for r in results:
        for b in r.get("brand_mentions", []) or []:
            if _is_target_brand(b, target_tokens):
                continue
            competitor_counts[b] = competitor_counts.get(b, 0) + 1
    matrix.all_competitors = dict(
        sorted(competitor_counts.items(), key=lambda x: -x[1])
    )

    return matrix


def _is_target_brand(brand: str, target_tokens: list[str]) -> bool:
    """Return True if brand matches the target domain tokens."""
    import re
    if not target_tokens:
        return False
    brand_norm = brand.lower().strip()
    brand_norm = re.sub(r"\s*(?:&|and)\s*", " ", brand_norm)
    brand_norm = " ".join(brand_norm.split())
    for token in target_tokens:
        token_norm = re.sub(r"\s*(?:&|and)\s*", " ", token.lower())
        token_norm = " ".join(token_norm.split())
        if token_norm in brand_norm or brand_norm in token_norm:
            return True
    return False


def _domain_contains_citation(citation: str, target_domain: str) -> bool:
    """Return True if citation contains the target domain."""
    norm_domain = _normalize_domain(target_domain)
    norm_cite = citation.lower().strip()
    bare = norm_domain.split(".")[0]
    return bare in norm_cite or norm_domain in norm_cite


# ---------------------------------------------------------------------------
# v2.0 — Derive additional variables for the report
# ---------------------------------------------------------------------------

def _derive_v2_variables(results: list[dict], aggregate: dict, domain: str) -> dict:
    """Compute v2.0 report variables from results and aggregate metrics.

    Returns a dict with:
        top_3_brands, engine_data, competitive_data, crawl_signals,
        topics_to_optimize, global_priorities
    """
    target_tokens = _extract_target_tokens(domain)

    # --- top_3_brands: top 3 non-target brands by cumulative mentions ---
    brand_counts: dict[str, int] = {}
    for r in results:
        for b in r.get("brand_mentions", []) or []:
            if _is_target_brand(b, target_tokens):
                continue
            brand_counts[b] = brand_counts.get(b, 0) + 1
    top_3_brands = [
        {"name": name, "count": count}
        for name, count in sorted(brand_counts.items(), key=lambda x: -x[1])[:3]
    ]

    # --- engine_data: group results by engine ---
    engine_data: dict[str, dict] = {}
    for r in results:
        eng = r["engine"]
        if eng not in engine_data:
            engine_data[eng] = {
                "engine": eng,
                "total_mentions": 0,
                "total_citations": 0,
                "topics_covered": 0,
                "total_topics": 0,
            }
        ed = engine_data[eng]
        ed["total_topics"] += 1
        ed["total_mentions"] += r.get("target_mention_count", 0)
        ed["total_citations"] += sum(
            1 for c in (r.get("citations", []) or [])
            if _domain_contains_citation(c, domain)
        )
        if r.get("target_mention_count", 0) > 0:
            ed["topics_covered"] += 1
    engine_data_list = sorted(engine_data.values(), key=lambda x: x["engine"])

    # --- competitive_data: per-query breakdown ---
    # Count both brand recommendations (mentions in text) and citations (URL sources)
    competitive_data: list[dict] = []
    for topic in dict.fromkeys(r["topic"] for r in results):
        topic_results = [r for r in results if r["topic"] == topic]
        mentions = sum(r.get("target_mention_count", 0) for r in topic_results)
        citations = sum(
            sum(1 for c in (r.get("citations", []) or []) if _domain_contains_citation(c, domain))
            for r in topic_results
        )
        # A topic is "covered" if target brand is mentioned in text OR cited as source
        engines_covered = sum(
            1 for r in topic_results
            if r.get("target_mention_count", 0) > 0
            or any(_domain_contains_citation(c, domain) for c in (r.get("citations", []) or []))
        )
        competitive_data.append({
            "topic": topic,
            "target_mentions": mentions,
            "target_citations": citations,
            "engines_covered": engines_covered,
            "total_engines": len(topic_results),
        })

    # --- topics_to_optimize: queries where target has 0 presence
    # (neither brand mentioned in text nor domain cited as source) ---
    topics_to_optimize = [
        cd["topic"] for cd in competitive_data
        if cd["target_mentions"] == 0 and cd["target_citations"] == 0
    ]

    # --- global_priorities: 3 items based on performance ---
    ai_presence = aggregate.get("ai_presence_pct", 0)
    if ai_presence >= 50:
        global_priorities = [
            "Protect and extend your defensible AI visibility position",
            "Refresh most-cited pages quarterly with new data",
            "Publish 2 net-new answer-first pages per month",
        ]
    elif ai_presence >= 25:
        global_priorities = [
            "Fill the citation gaps on your strongest topics",
            "Write answer-first guides with original data",
            "Ensure every product page has a direct answer capsule",
        ]
    else:
        global_priorities = [
            "Stand up a 6-page topical hub around buyer topics",
            "Add answer-first 40-word capsules under H1 headings",
            "Add original data, named authors, and schema markup",
        ]

    return {
        "top_3_brands": top_3_brands,
        "engine_data": engine_data_list,
        "competitive_data": competitive_data,
        "topics_to_optimize": topics_to_optimize,
        "global_priorities": global_priorities,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--domain", "-d",
    required=True,
    help="Domain to audit (e.g. paretotalent.com). Scheme and path are stripped.",
)
@click.option(
    "--max-pages",
    default=10,
    show_default=True,
    help="Max pages to crawl. Default 10 keeps audit runtime under ~30 seconds.",
)
@click.option(
    "--no-ai",
    is_flag=True,
    help="Skip AI engine citation checks. Uses deterministic mock data instead.",
)
@click.option(
    "--passes",
    default=None,
    type=int,
    help="Passes per (engine, topic) pair. Default from $PASSES_PER_QUERY (3).",
)
@click.option(
    "--output", "-o",
    default="output",
    show_default=True,
    help="Output directory for the HTML report.",
)
def main(domain: str, max_pages: int, no_ai: bool, passes: int, output: str):
    """Run a full AI visibility audit on a domain (v2.0 pipeline)."""
    # Normalize domain early so all subsequent steps see the same string
    from src.crawler import _normalize_domain
    domain = _normalize_domain(domain)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        # ── 1. Crawl ──
        task = progress.add_task(f"[cyan]Crawling {domain}...", total=5)
        crawl = crawl_domain(domain, max_pages=max_pages)
        progress.update(task, advance=1)

        if crawl.errors:
            for err in crawl.errors[:3]:
                console.print(f"  [yellow]⚠[/] {err}")

        # ── 2. Topics — LLM first, fall back to heuristic ──
        progress.update(task, description="[cyan]Generating buyer topics (LLM)...")
        try:
            from src.topicgen import generate_botf_topics
            topics = generate_botf_topics(domain)
        except Exception:
            from src.crawler import generate_buyer_topics
            topics = generate_buyer_topics(crawl)
        progress.update(task, advance=1)
        console.print(f"  [dim]Topics: {', '.join(topics)}[/]")

        # ── 3. AI visibility — v2.0 pipeline ──
        skip_ai = no_ai
        progress.update(task, description="[cyan]Checking AI visibility (v2.0)...")

        if skip_ai:
            # Generate deterministic mock results matching execute_all shape
            results = _generate_mock_results(domain, topics)
            console.print("  [dim]Using mock engine data (--no-ai)[/]")
        else:
            try:
                from src.collector import execute_all
                # Extract brand name from crawl for better target matching
                from src.reporter import _brand_name
                crawl_brand = _brand_name(domain, getattr(crawl, 'title', ''), getattr(crawl, 'meta_description', ''))
                results = execute_all(topics, domain, target_brand_name=crawl_brand)
            except Exception as e:
                console.print(f"  [yellow]AI check failed ({e}), falling back to mock data[/]")
                results = _generate_mock_results(domain, topics)

        progress.update(task, advance=1)

        # Show engine state
        engines_seen = list(dict.fromkeys(r["engine"] for r in results))
        for eng in engines_seen:
            eng_results = [r for r in results if r["engine"] == eng]
            errors = [r for r in eng_results if r.get("error")]
            if errors:
                console.print(f"  [dim]Engine: {eng} [red]({len(errors)} errors)[/]")
            else:
                console.print(f"  [dim]Engine: {eng} [green]real[/]")

        # ── 4. Aggregate results (v2.0 scoring) ──
        progress.update(task, description="[cyan]Scoring visibility (v2.0)...")
        from src.scoring import aggregate_results
        aggregate = aggregate_results(results, domain)

        # ── 5. Build CitationMatrix for template compatibility ──
        matrix = _build_citation_matrix_from_results(results, domain, topics)

        # ── 6. Analyze + report ──
        progress.update(task, description="[cyan]Building report...")
        report = analyze(crawl, matrix, topics=topics)

        # ── 7. Derive v2.0 variables ──
        v2 = _derive_v2_variables(results, aggregate, domain)
        progress.update(task, advance=1)

    # Term Ownership: count of queries where target has zero presence
    # (neither mentioned in text nor cited as source) across all engines
    zero_presence_count = sum(
        1 for cd in v2.get("competitive_data", [])
        if cd.get("target_mentions", 0) == 0 and cd.get("target_citations", 0) == 0
    )
    report.fixable_gaps = zero_presence_count

    # ── Render ──
    output_path = generate_report(
        report,
        output_dir=output,
        no_ai=skip_ai,
        # v2.0 variables
        ai_presence_pct=aggregate["ai_presence_pct"],
        best_brand=aggregate["best_brand"],
        best_model=aggregate["best_model"],
        citation_count=aggregate["citation_count"],
        best_topic=aggregate["best_topic"],
        top_3_brands=v2["top_3_brands"],
        engine_data=v2["engine_data"],
        competitive_data=v2["competitive_data"],
        topics_to_optimize=v2["topics_to_optimize"],
        global_priorities=v2["global_priorities"],
        crawl_signals=crawl,
    )

    console.print()
    console.print(f"[green]✓[/] Audit complete: [bold]{domain}[/]")
    console.print(f"[green]✓[/] AI Presence:    [bold]{aggregate['ai_presence_pct']:.0f}%[/]")
    console.print(f"[green]✓[/] Best Brand:     [bold]{aggregate['best_brand']}[/]")
    console.print(f"[green]✓[/] Best Model:     [bold]{aggregate['best_model']}[/]")
    console.print(f"[green]✓[/] Citations:      [bold]{aggregate['citation_count']}[/]")
    console.print(f"[green]✓[/] Best Topic:     [bold]{aggregate['best_topic']}[/]")
    console.print(f"[green]✓[/] Health Score:   [bold]{report.crawl_signals.health_score}/100[/]")
    console.print(f"[green]✓[/] Report:         [link=file://{output_path}]{output_path}[/]")
    console.print()
    console.print(f"[dim]Open in browser: open {output_path}[/]")

    # Exit nonzero if the audit found a critical access issue
    if any(i.severity == "critical" for i in crawl.issues):
        console.print("[yellow]⚠ Critical issues found (e.g. AI bots blocked). See report.[/]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())