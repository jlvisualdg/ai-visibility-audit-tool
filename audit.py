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
from src.crawler import crawl_domain, generate_buyer_topics
from src.reporter import generate_report
from src.visibility import build_citation_matrix

console = Console()


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
    help="Skip AI engine citation checks. Crawl + analyzer + report only.",
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
    """Run a full AI visibility audit on a domain."""
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
        # 1. Crawl
        task = progress.add_task(f"[cyan]Crawling {domain}...", total=4)
        crawl = crawl_domain(domain, max_pages=max_pages)
        progress.update(task, advance=1)

        if crawl.errors:
            for err in crawl.errors[:3]:
                console.print(f"  [yellow]⚠[/] {err}")

        # 2. Topics — try LLM first, fall back to heuristic
        progress.update(task, description="[cyan]Generating buyer topics (LLM)...")
        try:
            from src.topicgen import generate_botf_topics
            topics = generate_botf_topics(domain)
        except Exception:
            topics = generate_buyer_topics(crawl)
        progress.update(task, advance=1)
        console.print(f"  [dim]Topics: {', '.join(topics)}[/]")

        # 3. AI visibility check
        skip_ai = no_ai
        progress.update(task, description="[cyan]Checking AI visibility...")
        if skip_ai:
            from src.visibility import CitationMatrix
            matrix = CitationMatrix(domain=domain, topics=topics, engines=[])
        else:
            try:
                matrix = build_citation_matrix(domain, topics, passes=passes)
            except Exception as e:
                console.print(f"  [yellow]⚠ AI check failed ({e}), continuing with crawl-only report[/]")
                from src.visibility import CitationMatrix
                matrix = CitationMatrix(domain=domain, topics=topics, engines=[])
        progress.update(task, advance=1)

        # Show engine state
        for engine in matrix.engines:
            real = "(mock)" not in engine
            marker = "[green]real[/]" if real else "[dim]mock[/]"
            console.print(f"  [dim]Engine: {engine} {marker}[/]")

        # 4. Analyze + report
        progress.update(task, description="[cyan]Building report...")
        report = analyze(crawl, matrix, topics=topics)
        progress.update(task, advance=1)

    # Render
    output_path = generate_report(report, output_dir=output, no_ai=skip_ai)
    console.print()
    console.print(f"[green]✓[/] Audit complete: [bold]{domain}[/]")
    console.print(f"[green]✓[/] AI Coverage:    [bold]{report.ai_coverage_pct:.0f}%[/]")
    console.print(f"[green]✓[/] Fixable Gaps:   [bold]{report.fixable_gaps}[/]")
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
