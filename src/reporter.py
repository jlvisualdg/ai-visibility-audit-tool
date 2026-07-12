"""
HTML report generator.

Wraps the Jinja2 template with the AuditReport and writes a timestamped
HTML file to the output directory.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.analyzer import AuditReport


def _brand_name(domain: str) -> str:
    """Humanize a domain name for branding: 'paretotalent.com' -> 'Pareto Talent'."""
    bare = domain.split(".")[0]
    # CamelCase split
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", bare).split()
    return " ".join(p.capitalize() for p in parts)


def generate_report(report: AuditReport, output_dir: str = "output", no_ai: bool = False) -> str:
    """
    Render the dark-themed HTML report and write it to `output_dir`.

    Returns the absolute path of the generated file.
    """
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.html")

    html = template.render(
        report=report,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        no_ai=no_ai,
        brand_name=_brand_name(report.domain),
        brand_slogan="AI Engine Optimization Audit",
        website_url=f"https://{report.domain}",
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_domain = report.domain.replace("/", "_").replace(":", "_")
    out_path = out_dir / f"{safe_domain}-audit-{timestamp}.html"
    out_path.write_text(html, encoding="utf-8")

    return str(out_path.resolve())
