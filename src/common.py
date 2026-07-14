"""
Shared helpers used by scoring.py, crawler.py, audit.py, and visibility.py.

Single canonical source — import from here; never duplicate.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Corporate/legal entity suffixes stripped before fuzzy comparison.
# ---------------------------------------------------------------------------
CORP_SUFFIXES: set[str] = {
    "llc", "l.l.c", "llp", "lllp", "lp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "ltd", "limited", "pa", "p.a", "pc",
    "p.c", "pllc", "plc", "group", "associates", "partners", "gmbh", "sa",
    "sas", "bv", "ag", "nv", "srl", "esq",
}


def normalize_domain(d: str) -> str:
    """Strip scheme, path, and www. prefix from a domain string."""
    d = (d or "").strip().lower()
    for prefix in ("https://", "http://"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.split("/")[0]
    if d.startswith("www."):
        d = d[4:]
    return d


def normalize_for_fuzzy(s: str) -> str:
    """Aggressively normalize a brand name for fuzzy identity comparison.

    Lowercases, collapses "&"/"and", drops all punctuation, strips trailing
    corporate suffixes (LLC, LLP, Inc, P.A., ...), then removes spaces entirely
    so "Pinder Plotkin LLC", "Pinder Plotkin", and domain "pinderplotkin" all
    collapse to the same key "pinderplotkin".
    """
    s = s.lower()
    s = re.sub(r"\s*(?:&|and)\s*", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    words = s.split()
    while words and words[-1] in CORP_SUFFIXES:
        words = words[:-1]
    return "".join(words)


def brand_variants(s: str) -> set[str]:
    """Normalized forms of a brand for matching against domains.

    A brand with a connector can appear in a domain two ways: dropped
    ("Miller & Zois" -> millerzois.com) or spelled out (millerandzois.com).
    We generate BOTH so either domain form matches.
    """
    s = s.lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    words = s.split()
    while words and words[-1] in CORP_SUFFIXES:
        words = words[:-1]
    keep = "".join(words)
    drop = "".join(w for w in words if w != "and")
    return {v for v in (keep, drop) if v}


def fuzzy_target_match(brand: str, target_tokens: list[str]) -> bool:
    """Return True if `brand` is the target, tolerant of suffix/spacing/case."""
    b_variants = brand_variants(brand) or {normalize_for_fuzzy(brand)}
    b_variants = {v for v in b_variants if v}
    if not b_variants:
        return False
    for token in target_tokens:
        t_variants = brand_variants(token) or {normalize_for_fuzzy(token)}
        for tn in t_variants:
            if not tn:
                continue
            for b in b_variants:
                if b == tn:
                    return True
                if len(tn) >= 5 and len(b) >= 5 and (tn in b or b in tn):
                    return True
    return False


def extract_target_tokens(domain: str) -> list[str]:
    """Extract search tokens from a domain for case-insensitive matching.

    "paretotalent.com" -> ["paretotalent"]
    "acme-corp.com"    -> ["acmecorp", "acme corp"]
    """
    if not domain:
        return []

    bare = normalize_domain(domain).split(".")[0]
    tokens = [bare.lower()]

    if "-" in bare:
        tokens.append(bare.replace("-", " ").lower())

    common_suffixes = [
        "talent", "roofing", "corp", "inc", "group", "solutions", "tech",
        "software", "media", "consulting", "capital", "partners", "ventures",
        "labs", "studios", "agency", "creative", "marketing", "digital",
        "design", "studio", "systems", "cloud", "data", "health", "care",
        "works", "hub", "base", "hq", "co", "io", "ai", "app", "apps",
        "services", "global", "pro", "plus", "now", "direct",
    ]
    for suffix in common_suffixes:
        if bare.lower().endswith(suffix) and len(bare) > len(suffix) + 1:
            prefix = bare[:-len(suffix)]
            tokens.append(f"{prefix} {suffix}".lower())
            if len(prefix) >= 3:
                tokens.append(prefix.lower())
            break

    return tokens


def is_target_brand(brand: str, target_tokens: list[str]) -> bool:
    """Return True if `brand` matches any of the target tokens (fuzzy)."""
    return fuzzy_target_match(brand, target_tokens)


def domain_contains_citation(citation: str, target_domain: str) -> bool:
    """Return True if *citation* contains the normalized target domain."""
    norm_domain = normalize_domain(target_domain)
    norm_cite = citation.lower().strip()
    bare = norm_domain.split(".")[0]
    return bare in norm_cite or norm_domain in norm_cite
