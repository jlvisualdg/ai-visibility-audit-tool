# Security Audit Report — v2.0 Pre-Release

**Project:** AI Visibility Audit Tool (`aeo-audit-tool`)
**Date:** 2026-07-12
**Auditor:** Automated security review (5-point scan)
**Scope:** `src/`, `tests/`, `audit.py`, `.env.example`, `.gitignore`
**Test suite:** 158 tests, all passing

---

## Executive Summary

The AI Visibility Audit Tool v2.0 is in **good security health** for its threat model (a CLI tool that crawls public websites and queries OpenRouter AI APIs, generating local HTML reports). Four of five checks pass cleanly; input validation has minor hardening opportunities at low severity. The overall security rating is **B+**.

---

## Findings Table

| # | Check | Result | Severity | Details |
|---|-------|--------|----------|---------|
| 1 | Hardcoded Secrets | ✅ **PASS** | — | No hardcoded API keys, passwords, or secrets found. Only hit is a usage example comment. `.env` is gitignored; `.env.example` ships with empty values. |
| 2 | Environment Variables | ✅ **PASS** | — | All `OPENROUTER_API_KEY` accesses use `os.environ.get()` with safe defaults (`""`). 3 locations audited: `src/engines.py:47`, `src/visibility.py:45`, `tests/test_integration_end_to_end.py:74`. |
| 3 | Input Validation | ⚠️ **PARTIAL PASS** | Low | Domain normalization exists (`_normalize_domain()` strips scheme/path/whitespace, handles None/empty gracefully), but no regex format validation or explicit empty-string rejection. Domain chars pass through unchecked; malformed domains fail safely at the HTTP layer. |
| 4 | API Error Handling | ✅ **PASS** | — | All engine query functions catch `RequestException`, `ValueError` (JSON decode), `KeyError`/`IndexError`/`TypeError`. CLI entry point wraps topicgen and citation matrix with try/except. All 158 tests pass. |
| 5 | Output Sanitization | ✅ **PASS** | — | Jinja2 `select_autoescape(["html", "xml"])` enabled in `reporter.py:36`. Zero `|safe` filter bypasses. All user-originated values (domain, brand_name, competitor names, topics) rendered via autoescaped `{{ }}` blocks. |

---

## Detailed Findings

### Check 1 — Hardcoded Secrets

**Methodology:**
```
rg -n '(sk-or|api_key\s*=\s*["\']|password\s*=\s*["\']|secret\s*=\s*["\'])' src/ tests/ audit.py
```

**Results:** Zero matches in source code or tests.

- `audit.py:11`: The line `OPENROUTER_API_KEY=sk-or-v1-... python audit.py --domain paretotalent.com` appears in a **docstring usage example** — it is not executable code, it is an artificial documentation key prefix. This is acceptable.
- `.env` is present in `.gitignore` (line 2: `.env`). Not committed.
- `.env.example` has empty values for all four API key fields (`OPENROUTER_API_KEY=`, `OPENAI_API_KEY=`, etc.).

**Vulnerabilities:** None.

---

### Check 2 — Environment Variables

**Methodology:** Traced every access to `OPENROUTER_API_KEY` across the codebase.

| File | Line | Access Pattern | Safe? |
|------|------|---------------|-------|
| `src/engines.py` | 47 | `os.environ.get("OPENROUTER_API_KEY", "").strip()` | ✅ |
| `src/visibility.py` | 45 | `os.environ.get("OPENROUTER_API_KEY", "").strip()` | ✅ |
| `tests/test_integration_end_to_end.py` | 74 | `os.environ.get("OPENROUTER_API_KEY", "").strip()` | ✅ |

Additionally, `tests/test_engines.py:338` patches `OPENROUTER_API_KEY` to `""` for testing the missing-key error path, and `tests/test_topicgen.py:82` similarly patches for fallback testing.

All accesses use `os.environ.get()` with a default empty string — no direct `os.environ["KEY"]` indexing that would raise `KeyError`. The `.strip()` guard handles leading/trailing whitespace.

**Vulnerabilities:** None.

---

### Check 3 — Input Validation

**Methodology:** Traced domain input flow from CLI entry (`audit.py:73`) through `_normalize_domain()` in `crawler.py:873` and `scoring.py:272`.

**What exists:**
- `audit.py:73`: `domain = _normalize_domain(domain)` normalizes early.
- `crawler.py:_normalize_domain()` (line 873): strips `https://`, `http://`, path segments, trailing slashes; lowercases.
- `scoring.py:_normalize_domain()` (line 272): Same, plus strips `www.` prefix.
- Both handle `(d or "").strip()` — None/empty doesn't crash.

**What's missing:**
1. No regex validation that input matches a domain-like pattern (e.g., `example.com`, `sub.example.co.uk`).
2. No explicit rejection of empty strings — `_normalize_domain("")` returns `""`. Click's `required=True` on the CLI flag prevents an entirely missing argument, but an explicit guard would be defense-in-depth.
3. No character blacklist for XSS/HTML-injection characters (`<`, `>`, `"`, `'`, `&`). These would pass through the normalization and end up in the report template. **However**, Jinja2 autoescaping (Check 5) neutralizes this at render time.

**Risk assessment:** Low. Malformed domains cause HTTP-level failures (DNS resolution, connection refused) rather than code injection or data leaks. XSS-prone characters are caught by autoescaping downstream. The empty-string path is reachable only via programmatic calls (not the CLI). Recommend adding a domain-format regex guard in `_normalize_domain()` for a future release.

**Vulnerabilities:** None exploitable. Hardening recommended.

---

### Check 4 — API Error Handling

**Methodology:** Reviewed all network-facing code paths and ran 158 tests.

**Error handling inventory:**

| Module | Function | Catches | Graceful? |
|--------|----------|---------|-----------|
| `src/engines.py` | `_BaseEngine.query()` | `requests.RequestException`, `ValueError` (JSON), `KeyError`/`IndexError`/`TypeError` | ✅ Returns `{error: "..."}` dict |
| `src/engines.py` | `query_all_engines()` | `ValueError` (init) | ✅ Returns `{error: "..."}` dict per-engine |
| `src/visibility.py` | `PerplexityEngine.query()` | `requests.RequestException`, `ValueError`, `KeyError`/`IndexError` | ✅ Returns `{error: "..."}` dict |
| `src/collector.py` | `execute_all()` | Engine instantiation + query failures | ✅ Populates `error` field per-result |
| `audit.py` | `main()` | `Exception` on topicgen and matrix build | ✅ Falls back to heuristic topics / empty matrix |
| `src/crawler.py` | `crawl_domain()` | `requests.RequestException` on seed URL attempts | ✅ Falls through seed paths |

**Test results:** `158 passed in 3.56s` — all unit and integration tests pass. Tests cover:
- Engine init with missing API key → `ValueError`
- LLM HTTP error → fallback paths (topicgen)
- Empty/missing API key → fallback topics
- Engine result parsing errors → graceful error dicts

**Vulnerabilities:** None.

---

### Check 5 — Output Sanitization

**Methodology:** Reviewed HTML rendering pipeline and template for XSS/injection vectors.

**Jinja2 autoescaping:**
```python
# reporter.py:34-36
env = Environment(
    loader=FileSystemLoader(str(template_dir)),
    autoescape=select_autoescape(["html", "xml"]),
)
```

`select_autoescape(["html", "xml"])` enables autoescaping for `.html` and `.xml` templates. All `{{ ... }}` expressions in `report.html` are automatically HTML-escaped. Characters `<`, `>`, `&`, `"`, `'` are converted to entities.

**Unsafe filter audit:**
```bash
rg '\|\s*safe' src/templates/report.html
# → 0 matches
```
Zero uses of the `|safe` filter that would bypass autoescaping.

**User-originated values rendered in the template:**
| Value | Source | Rendered as | Safe? |
|-------|--------|-------------|-------|
| `report.domain` | CLI input, normalized | `{{ report.domain }}` — autoescaped | ✅ |
| `brand_name` | `_brand_name(domain)` — regex + capitalize | `{{ brand_name }}` — autoescaped | ✅ |
| `website_url` | `f"https://{report.domain}"` | `{{ website_url }}` — autoescaped (also in `<a href>` context) | ✅ |
| Competitor domains | Regex from engine responses | `{{ domain }}` — autoescaped | ✅ |
| Topics | LLM or heuristic | `{{ topic }}` — autoescaped | ✅ |
| Crawl issues | Crawler signals | `{{ issue.detail }}` — autoescaped | ✅ |

**`website_url` in href attribute:** The line `<a href="{{ website_url }}">` renders the URL inside an attribute. Jinja2 autoescaping converts `"` to `&quot;`, preventing attribute injection. The URL itself is constructed as `f"https://{report.domain}"` where `domain` has been normalized (scheme/path stripped), so protocol-relative or `javascript:` injection is not possible.

**Vulnerabilities:** None.

---

## Overall Security Rating: **B+**

| Grade | Meaning |
|-------|---------|
| A | All checks pass, defense-in-depth present |
| **B+** | **All critical checks pass; one check has minor hardening opportunities** |
| B | All checks pass but with notes |
| C | One or more findings require attention before production |
| D | Multiple findings, blocking issues |
| F | Critical vulnerability present |

**Rationale for B+:** Four of five security checks pass cleanly with no vulnerabilities found. Input validation (Check 3) has low-severity hardening gaps — domain format is not regex-validated and empty strings are not explicitly rejected — but these are mitigated by downstream autoescaping and HTTP-level failure. No exploitable vulnerabilities exist in the current code.

---

## Remediation Recommendations

| Priority | Check | Recommendation | Effort |
|----------|-------|---------------|--------|
| Low | #3 | Add domain format regex validation in `_normalize_domain()` — reject empty strings and non-domain patterns (e.g., `<script>alert(1)</script>`) | 5 lines |
| Low | #3 | Add explicit empty-string rejection: `if not d: raise ValueError("domain must not be empty")` | 2 lines |
| Note | #1 | Replace `sk-or-v1-...` in `audit.py` docstring with `$OPENROUTER_API_KEY` placeholder to avoid false-positive hits in future scans | 1 line |

---

## Verification Commands

```bash
# Re-run hardcoded secrets scan (should return 0)
grep -rE 'sk-or-' src/ tests/ audit.py | grep -v 'docstring\|#.*example' || echo "PASS"

# Re-run full test suite
cd /path/to/aeo-audit-tool
python -m pytest tests/ -v

# Verify autoescaping is configured
python -c "from pathlib import Path; c = Path('src/reporter.py').read_text(); print('select_autoescape' in c)"
```