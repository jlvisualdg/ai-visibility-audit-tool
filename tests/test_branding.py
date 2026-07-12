"""
Tests for src.branding — brand identity tokens and helper functions.
"""

from __future__ import annotations

import re

import pytest

from src.branding import (
    BRAND,
    BRAND_ASSETS,
    COLORS,
    GRADIENTS,
    LOGO_URL,
    TYPOGRAPHY,
    get_css_variables,
    get_header_css,
    get_header_html,
)


# ---------------------------------------------------------------------------
# Constants — existence & type checks
# ---------------------------------------------------------------------------

def test_logo_url_is_string():
    assert isinstance(LOGO_URL, str)
    assert LOGO_URL.startswith("https://")


def test_colors_has_all_required_keys():
    required = {
        "primary", "secondary",
        "background", "card", "card_hover", "border",
        "text", "text_muted", "text_dim",
        "accent", "accent_dim",
        "success", "warning", "danger", "info",
    }
    assert required <= set(COLORS.keys()), f"missing keys: {required - set(COLORS.keys())}"


def test_colors_are_valid_hex():
    hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for name, value in COLORS.items():
        assert hex_pattern.match(value), f"COLORS['{name}'] = {value!r} is not a valid hex colour"


def test_typography_has_required_keys():
    required = {
        "font_family", "font_family_mono",
        "size_h1", "size_h2", "size_h3", "size_body", "size_small", "size_caption",
        "line_height_tight", "line_height_body", "line_height_relaxed",
        "weight_regular", "weight_medium", "weight_semibold", "weight_bold",
    }
    assert required <= set(TYPOGRAPHY.keys()), f"missing keys: {required - set(TYPOGRAPHY.keys())}"


def test_typography_sizes_are_positive_ints():
    size_keys = ["size_h1", "size_h2", "size_h3", "size_body", "size_small", "size_caption"]
    for key in size_keys:
        val = TYPOGRAPHY[key]
        assert isinstance(val, int), f"TYPOGRAPHY['{key}'] should be int, got {type(val).__name__}"
        assert val > 0, f"TYPOGRAPHY['{key}'] = {val} must be > 0"


def test_typography_weights_are_valid():
    weight_keys = ["weight_regular", "weight_medium", "weight_semibold", "weight_bold"]
    for key in weight_keys:
        val = TYPOGRAPHY[key]
        assert isinstance(val, int)
        assert 100 <= val <= 900


def test_h1_is_largest_size():
    assert TYPOGRAPHY["size_h1"] > TYPOGRAPHY["size_h2"]
    assert TYPOGRAPHY["size_h2"] > TYPOGRAPHY["size_h3"]
    assert TYPOGRAPHY["size_h3"] > TYPOGRAPHY["size_body"]


def test_gradients_are_css_ready():
    for name, value in GRADIENTS.items():
        assert "linear-gradient" in value, f"GRADIENTS['{name}'] missing 'linear-gradient'"


def test_brand_has_core_fields():
    assert BRAND["name"] == "Smart Marketer"
    assert BRAND["slogan"] == "Become the Answer AI Recommends"
    assert BRAND["url"] == "https://aeo.co"
    assert len(BRAND["pillars"]) == 5
    assert len(BRAND["ecosystem"]) >= 5


def test_brand_assets_contain_logos():
    assert "logo_svg" in BRAND_ASSETS
    assert "logo_svg_mini" in BRAND_ASSETS
    assert "<svg" in BRAND_ASSETS["logo_svg"]
    assert "Smart Marketer" in BRAND_ASSETS["logo_svg"]
    assert "url(#sm-grad)" in BRAND_ASSETS["logo_svg"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestGetCssVariables:

    def test_returns_string_with_root_block(self):
        css = get_css_variables()
        assert isinstance(css, str)
        assert ":root {" in css
        assert css.strip().endswith("}")

    def test_includes_color_variables(self):
        css = get_css_variables()
        for name, value in COLORS.items():
            assert f"--color-{name}: {value}" in css, f"missing --color-{name}"

    def test_includes_typography_variables(self):
        css = get_css_variables()
        assert "--typography-font_family" in css
        assert "--typography-size_h1" in css

    def test_includes_gradient_variables(self):
        css = get_css_variables()
        for name in GRADIENTS:
            assert f"--gradient-{name}" in css, f"missing --gradient-{name}"

    def test_every_line_in_block_is_valid_css_declaration(self):
        css = get_css_variables()
        inside = False
        for line in css.splitlines():
            line = line.strip()
            if line == ":root {":
                inside = True
                continue
            if line == "}":
                inside = False
                continue
            if not line or line.startswith("/*"):
                continue
            if inside:
                # Must be a CSS variable declaration
                assert line.endswith(";"), f"line missing semicolon: {line!r}"
                assert line.startswith("--"), f"line not a variable: {line!r}"
                assert ": " in line, f"line missing ': ': {line!r}"

    def test_idempotent(self):
        a = get_css_variables()
        b = get_css_variables()
        assert a == b


class TestGetHeaderHtml:

    def test_returns_string_with_header_tag(self):
        html = get_header_html()
        assert isinstance(html, str)
        assert "<header" in html

    def test_defaults_use_smart_marketer(self):
        html = get_header_html()
        assert "Smart Marketer" in html
        assert "Become the Answer AI Recommends" in html

    def test_custom_brand_and_slogan(self):
        html = get_header_html(
            brand_name="Acme Corp",
            slogan="We deliver",
            url="https://acme.com",
        )
        assert "Acme Corp" in html
        assert "We deliver" in html
        assert 'href="https://acme.com"' in html

    def test_date_appears_in_output(self):
        html = get_header_html(date="2026-07-12")
        assert "2026-07-12" in html

    def test_date_defaults_to_today(self):
        from datetime import datetime
        html = get_header_html()
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in html

    def test_includes_svg_logo(self):
        html = get_header_html()
        assert "svg" in html
        assert "linearGradient" in html


class TestGetHeaderCss:

    def test_returns_string_with_styles(self):
        css = get_header_css()
        assert ".sm-header" in css
        assert ".sm-brand-name" in css
        assert ".sm-tagline" in css

    def test_references_css_variable_names(self):
        css = get_header_css()
        # Should fall back to hardcoded values but reference variables
        assert "var(--color-border" in css or "#222" in css
        assert "var(--gradient-brand" in css or "linear-gradient" in css


# ---------------------------------------------------------------------------
# Integration / cross-checks
# ---------------------------------------------------------------------------

def test_css_variables_header_css_are_consistent():
    """CSS variable names referenced in header CSS exist in get_css_variables output."""
    variables = get_css_variables()
    header_css = get_header_css()

    # Extract var(--...) references from header CSS
    refs = set(re.findall(r"var\((--[\w-]+)", header_css))
    for ref in refs:
        assert ref in variables, f"CSS variable {ref} referenced in header CSS but not defined"


def test_colors_dark_theme():
    """Background is darker than text (dark theme invariant)."""
    assert COLORS["background"] == "#0A0A0A"
    assert COLORS["text"] == "#FFFFFF"


def test_primary_is_blue():
    assert COLORS["primary"] == "#0066FF"