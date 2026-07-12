"""
Smart Marketer brand identity tokens.

Extracted from aeo.co branding research — dark theme, blue/purple gradient,
Inter/system sans-serif typography, AI-answer-engine positioning.

Provides:
- COLORS, TYPOGRAPHY, BRAND_ASSETS as Python constants
- get_css_variables()     →  :root { ... } CSS block
- get_header_html(...)    →  Jinja2-compatible HTML header snippet
"""

from __future__ import annotations

from datetime import datetime
from textwrap import dedent
from typing import Dict, Final

# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------

LOGO_URL: Final[str] = "https://aeo.co/wp-content/uploads/2024/01/smart-marketer-logo.svg"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLORS: Final[Dict[str, str]] = {
    # Core brand
    "primary":       "#0066FF",   # Smart Marketer blue
    "secondary":     "#7C3AED",   # Purple (gradient partner)
    # Surfaces
    "background":    "#0A0A0A",   # Page body background
    "card":          "#111111",   # Card / panel background
    "card_hover":    "#1A1A1A",   # Card hover state
    "border":        "#222222",   # Subtle borders / dividers
    # Text
    "text":          "#FFFFFF",   # Primary body text
    "text_muted":    "#888888",   # Secondary / muted text
    "text_dim":      "#555555",   # Even more muted
    # Accent & semantic
    "accent":        "#00D4FF",   # Cyan highlight accent
    "accent_dim":    "#0099BB",   # Dimmed accent
    "success":       "#00F5A0",   # Green — scores / passes
    "warning":       "#FF8C42",   # Orange — warnings
    "danger":        "#FF4757",   # Red — errors / fails
    "info":          "#FFD93D",   # Yellow — informational
}

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

TYPOGRAPHY: Final[Dict[str, object]] = {
    # Font stacks (CSS-ready, no quotes needed inside the family string)
    "font_family": (
        'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", '
        'Roboto, "Helvetica Neue", Arial, sans-serif'
    ),
    "font_family_mono": (
        '"JetBrains Mono", "Fira Code", "SF Mono", '
        'Consolas, "Liberation Mono", monospace'
    ),
    # Sizes (px)
    "size_h1":        48,
    "size_h2":        32,
    "size_h3":        24,
    "size_body":      16,
    "size_small":     14,
    "size_caption":   12,
    # Line heights
    "line_height_tight":    1.2,
    "line_height_body":     1.6,
    "line_height_relaxed":  1.8,
    # Weights
    "weight_regular":  400,
    "weight_medium":   500,
    "weight_semibold": 600,
    "weight_bold":     700,
}

# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

GRADIENTS: Final[Dict[str, str]] = {
    "brand":   "linear-gradient(135deg, #0066FF 0%, #7C3AED 100%)",
    "success": "linear-gradient(135deg, #00F5A0 0%, #00D4FF 100%)",
    "warm":    "linear-gradient(135deg, #FF8C42 0%, #FF4757 100%)",
}

# ---------------------------------------------------------------------------
# Brand narrative / positioning
# ---------------------------------------------------------------------------

BRAND: Final[Dict[str, object]] = {
    "name":           "Smart Marketer",
    "slogan":         "Become the Answer AI Recommends",
    "url":            "https://aeo.co",
    "tagline_short":  "AI Visibility for Revenue Teams",
    "pillars": [
        "Brand Representation",
        "Research",
        "Authority",
        "Indexability",
        "Network",
    ],
    "ecosystem": [
        "Community",
        "Blueprint",
        "Show",
        "Audit",
        "Services",
        "Content Engine",
    ],
}

# ---------------------------------------------------------------------------
# Inline brand assets
# ---------------------------------------------------------------------------

BRAND_ASSETS: Final[Dict[str, str]] = {
    "logo_svg": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 60" '
        'fill="none" role="img" aria-label="Smart Marketer">\n'
        '  <defs>\n'
        '    <linearGradient id="sm-grad" x1="0%" y1="0%" x2="100%" y2="0%">\n'
        '      <stop offset="0%" stop-color="#0066FF"/>\n'
        '      <stop offset="100%" stop-color="#7C3AED"/>\n'
        '    </linearGradient>\n'
        '  </defs>\n'
        '  <text x="8" y="42" font-family="Inter, sans-serif" '
        'font-size="28" font-weight="700" letter-spacing="-0.5" '
        'fill="url(#sm-grad)">Smart Marketer</text>\n'
        '  <text x="12" y="56" font-family="Inter, sans-serif" '
        'font-size="10" font-weight="400" fill="#888888" '
        'letter-spacing="2.5" text-transform="uppercase">'
        'Become the Answer AI Recommends'
        '</text>\n'
        '</svg>'
    ),
    "logo_svg_mini": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" '
        'fill="none" role="img" aria-label="Smart Marketer icon">\n'
        '  <defs>\n'
        '    <linearGradient id="sm-grad-mini" x1="0%" y1="0%" x2="100%" y2="100%">\n'
        '      <stop offset="0%" stop-color="#0066FF"/>\n'
        '      <stop offset="100%" stop-color="#7C3AED"/>\n'
        '    </linearGradient>\n'
        '  </defs>\n'
        '  <rect x="4" y="4" width="32" height="32" rx="8" fill="url(#sm-grad-mini)"/>\n'
        '  <text x="20" y="27" font-family="Inter, sans-serif" '
        'font-size="20" font-weight="700" fill="#FFFFFF" '
        'text-anchor="middle">SM</text>\n'
        '</svg>'
    ),
}


# ====================================================================
# Helper functions for template rendering
# ====================================================================

def get_css_variables() -> str:
    """Return a ``:root { ... }`` CSS block exposing every design token.

    Designed to be injected into ``<style>`` blocks in Jinja2 templates.
    """
    lines: list[str] = [":root {"]
    # --- colours ---
    lines.append("  /* ── Colours ── */")
    for name, value in COLORS.items():
        lines.append(f"  --color-{name}: {value};")

    # --- typography ---
    lines.append("")
    lines.append("  /* ── Typography ── */")
    for name, value in TYPOGRAPHY.items():
        css_name = f"--typography-{name}"
        if isinstance(value, float):
            lines.append(f"  {css_name}: {value};")
        elif isinstance(value, int):
            lines.append(f"  {css_name}: {value}px;")
        else:
            lines.append(f"  {css_name}: {value};")

    # --- gradients ---
    lines.append("")
    lines.append("  /* ── Gradients ── */")
    for name, value in GRADIENTS.items():
        lines.append(f"  --gradient-{name}: {value};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def get_header_html(
    brand_name: str = "Smart Marketer",
    slogan: str = "Become the Answer AI Recommends",
    url: str = "https://aeo.co",
    date: str | None = None,
) -> str:
    """Return an HTML ``<header>`` block suitable for Jinja2 templates.

    Parameters
    ----------
    brand_name : str
        Brand name displayed in the header.
    slogan : str
        Tagline shown beneath the brand name.
    url : str
        Target of the "Visit" link (top-right corner).
    date : str | None
        Optional date string (e.g. ``"2026-07-12"``); defaults to today.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    html = dedent(f"""\
    <header class="sm-header">
      <div class="sm-header__left">
        <div class="sm-logo">
          {BRAND_ASSETS["logo_svg"]}
        </div>
        <div class="sm-header__titles">
          <h1 class="sm-brand-name">{brand_name}</h1>
          <p class="sm-tagline">{slogan}</p>
        </div>
      </div>
      <div class="sm-header__right">
        <span class="sm-header__date">{date}</span>
        <a href="{url}" class="sm-header__link" target="_blank" rel="noopener">
          Visit Site ↗
        </a>
      </div>
    </header>
    """)
    return html.strip()


def get_header_css() -> str:
    """Return companion CSS for the HTML produced by :func:`get_header_html`.

    Inject this into your ``<style>`` block when you call ``get_header_html``.
    """
    return dedent("""\
    .sm-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 24px 0;
        border-bottom: 1px solid var(--color-border, #222);
        margin-bottom: 40px;
        flex-wrap: wrap;
        gap: 16px;
    }
    .sm-header__left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .sm-logo svg {
        width: 200px;
        height: auto;
    }
    .sm-brand-name {
        font-size: 24px;
        font-weight: 700;
        background: var(--gradient-brand, linear-gradient(135deg, #0066FF, #7C3AED));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sm-tagline {
        font-size: var(--typography-size_small, 14px);
        color: var(--color-text_muted, #888);
        margin: 4px 0 0 0;
    }
    .sm-header__right {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .sm-header__date {
        font-size: var(--typography-size_small, 14px);
        color: var(--color-text_muted, #888);
    }
    .sm-header__link {
        font-size: var(--typography-size_small, 14px);
        color: var(--color-primary, #0066FF);
        text-decoration: none;
        border: 1px solid var(--color-border, #222);
        padding: 6px 14px;
        border-radius: 6px;
        transition: background 0.2s;
    }
    .sm-header__link:hover {
        background: var(--color-card_hover, #1A1A1A);
    }
    """)


# ---------------------------------------------------------------------------
# Convenience — list all exported token constants
# ---------------------------------------------------------------------------

__all__ = [
    "LOGO_URL",
    "COLORS",
    "TYPOGRAPHY",
    "GRADIENTS",
    "BRAND",
    "BRAND_ASSETS",
    "get_css_variables",
    "get_header_html",
    "get_header_css",
]