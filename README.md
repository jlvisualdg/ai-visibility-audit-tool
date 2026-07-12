# AI Visibility Audit Tool v2.0

A CLI tool that audits a target domain's **AI visibility** — how often it gets
cited when buyers ask AI engines (Perplexity, ChatGPT, Claude, Gemini) about
topics it should own. Generates a dark-themed HTML report with a citation
matrix, crawl health signals, brand presence scoring, and prioritized fix
recommendations.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-158%20passed-brightgreen)](tests/)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jlvisualdg/ai-visibility-audit-tool.git
cd ai-visibility-audit-tool

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Copy the environment template and add your OpenRouter API key
cp .env.example .env
# Edit .env and set: OPENROUTER_API_KEY=sk-or-v1-your-key-here

# 4. Run an audit
python audit.py --domain example.com --no-ai --max-pages 3    # mock mode, no API key needed
python audit.py --domain yoursite.com                         # real AI engine queries

# 5. Open the report
open output/yoursite.com-audit-*.html
```

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| **Python** | 3.10 or higher | Runtime |
| **OpenRouter API key** | `sk-or-v1-...` | Required for real AI engine queries. [Get one free →](https://openrouter.ai/keys) |
| **Dependencies** | See `requirements.txt` | See [Configuration](#configuration) below |

**No API key?** Run with `--no-ai` for a crawl-only audit that still produces a full HTML report with health scores, schema analysis, and fix recommendations.

---

## Configuration

### Environment Variables

All keys are **optional**. The tool runs fully in mock mode without them.

```env
# .env
OPENROUTER_API_KEY=sk-or-v1-your-key-here    # Required for real AI engine queries
PASSES_PER_QUERY=3                            # Passes per (engine, topic) pair (default: 3)
PERPLEXITY_MODEL=perplexity/sonar-pro         # Override the Perplexity model
ENGINE_RATE_LIMIT_SECS=2                      # Seconds between API calls (default: 2)
```

### CLI Options

```
python audit.py --help
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--domain` | `-d` | *(required)* | Domain to audit (e.g. `yoursite.com`). Scheme and path are stripped automatically. |
| `--max-pages` | — | `10` | Maximum pages to crawl. Keep ≤10 for sub-30s audits. |
| `--no-ai` | — | `false` | Skip AI engine citation checks. Crawl + analyzer + report only. |
| `--passes` | — | `3` (from env) | Number of query passes per (engine, topic) pair. |
| `--output` | `-o` | `output` | Directory for the generated HTML report. |

### Examples

```bash
# Fast mock-mode audit (no API key)
python audit.py --domain example.com --no-ai --max-pages 5

# Real multi-engine audit with custom output location
python audit.py --domain yoursite.com --max-pages 10 --output ./reports

# Quick audit with fewer pages
python audit.py --domain yoursite.com --max-pages 3 --passes 1
```

---

## Engines

Four AI engines are queried through OpenRouter (a unified API gateway). Each
engine receives the same buyer-intent topic queries and returns citations.

| Engine | Model | Real / Mock | Citation Source |
|--------|-------|-------------|-----------------|
| **Perplexity** | `perplexity/sonar` | Real (with API key) | Structured annotations |
| **ChatGPT** | `openai/gpt-4o-mini-search-preview` | Real (with API key) | Regex from text |
| **Claude** | `anthropic/claude-sonnet-4` | Real (with API key) | Regex from text |
| **Gemini** | `google/gemini-2.5-flash` | Real (with API key) | Regex from text |

Without an OpenRouter API key, all engines run in **mock mode** with
deterministic fake citations — the full pipeline works for development and
demo purposes.

---

## Output

Each audit generates two artifacts in the output directory:

### HTML Report (`output/<domain>-audit-<timestamp>.html`)

A self-contained, dark-themed HTML report with 7 sections:

| # | Section | Contents |
|---|---------|----------|
| 1 | **Header** | Brand name, slogan, domain URL, generation timestamp |
| 2 | **Verdict** | AI Presence %, Best Brand, Best Model, Citation Count, Best Topic |
| 3 | **Brand Recommendation Matrix** | 5 topics × 4 engines grid with cited/partial/not-cited status |
| 4 | **Top Recommended Brands** | Top 3 most-cited competitor brands |
| 5 | **Competitive Landscape** | Per-topic breakdown: result, mentions, citations, position, top competitor |
| 6 | **AI Indexability Audit** | Health score hero + 15 signal cards + crawl issues | 
| 7 | **Topics to Optimize** | Covered topics + zero-presence topics with dropdown toggle |

### Scoring Methodology

The **AI Presence Score** (0–100%) is computed per (topic × engine) cell from
extracted brand mentions and positions, then aggregated across all 20 cells
(5 topics × 4 engines). Full formula and worked examples at →
[docs/scoring_system.md](docs/scoring_system.md).

---

## Crawl Signals

The crawler extracts 15+ AI-readiness signals from the target domain:

- **answer_capsules** — H2/H3 headings followed by direct `<p>` answers (citation magnets)
- **stat_density** — Data points (numbers, percentages, currency) per 100 words
- **authorship_pages** — Pages with visible author bylines or author schema
- **schema_pages** — Pages with JSON-LD or microdata schema markup
- **has_llms.txt** — Presence of `/llms.txt` (emerging AI crawler standard)
- **robots_blocks_ai** — Whether `robots.txt` blocks tracked AI bots (27 bots monitored)
- **agent_readability_score** — 0–100 composite: landmarks, alt text, form labels, heading hierarchy
- **health_score** — Overall AI indexability score (0–100)
- **thin_pages** — Pages under 300 words
- **broken_links** — Empty or broken `href` attributes
- **response_time** — Average server response time in ms
- **redirect_hops** — Maximum redirect chain length
- **about_page / contact_page** — Entity-definition pages detected

---

## Testing

```bash
# Activate venv first
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Run all tests (158 tests)
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_scoring.py -v       # 47 tests — brand extraction + scoring
python -m pytest tests/test_collector.py -v     # 17 tests — collector pipeline
python -m pytest tests/test_engines.py -v       # Engine client tests
python -m pytest tests/test_template.py -v      # 26 tests — template validation
python -m pytest tests/test_branding.py -v      # 27 tests — brand tokens + helpers
python -m pytest tests/test_topicgen.py -v      # 19 tests — topic generation

# Validate scoring formula (9 tests, 0.01 tolerance)
python tests/validate_scoring.py

# Integration test (requires API key — makes 20 real API calls)
python -m pytest tests/test_integration_end_to_end.py -v
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ValueError: needs OPENROUTER_API_KEY` | No API key configured | Set `OPENROUTER_API_KEY` in `.env` or run with `--no-ai` |
| `Could not resolve host` | Domain doesn't exist or DNS failure | Verify the domain is correct and publicly accessible |
| `robots.txt disallows the audit user-agent` | Site blocks our crawler | The audit continues but flags this. Consider whitelisting the AEO-Audit-Tool UA. |
| `latin-1 codec can't encode character` | Windows encoding issue with OpenRouter | Fixed in v2.0 — `r.encoding = "utf-8"` is forced before JSON parsing |
| `All engines show (mock)` | No API key provided | Set `OPENROUTER_API_KEY` in `.env` — mock mode is the default fallback |
| Report has zero AI presence | Domain has no AI engine citations | Expected for new sites. Use the fix recommendations in the report. |
| Empty report or no topics generated | LLM topic generation failed | Tool falls back to heuristic keyword extraction — check console output for warnings |
| `ImportError: No module named 'src'` | Running tests from wrong directory | Run from the project root (`aeo-audit-tool/`) with `python -m pytest tests/` |

---

## Project Layout

```
aeo-audit-tool/
├── audit.py                     # Click CLI entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template (safe to commit)
├── README.md                    # This file
├── docs/
│   ├── scoring_system.md        # Full scoring methodology + adjustment guide
│   └── security-audit-2026-07-12.md  # v2.0 security review
├── src/
│   ├── crawler.py               # Domain crawl + 15 signal extraction (~1050 lines)
│   ├── topicgen.py              # LLM-powered unbranded BOTF query generation
│   ├── engines.py               # 4 real OpenRouter engine clients
│   ├── visibility.py            # Engine protocol + citation matrix (v1 pattern)
│   ├── collector.py             # Multi-engine × multi-topic runner + execute_all()
│   ├── scoring.py               # Brand extraction + AI Presence Score + aggregate_results()
│   ├── analyzer.py              # Strategic analysis + fix generation
│   ├── reporter.py              # Jinja2 HTML report renderer
│   ├── branding.py              # Smart Marketer brand tokens + CSS/HTML helpers
│   └── templates/
│       └── report.html          # Template v2: 7 sections, dark theme, vanilla CSS
├── output/                      # Generated reports (gitignored except .gitkeep)
└── tests/
    ├── conftest.py              # Pytest config
    ├── test_scoring.py          # Brand extraction + aggregate (47 tests)
    ├── test_collector.py        # Collector pipeline (17 tests)
    ├── test_engines.py          # Engine clients (22 assertions)
    ├── test_template.py         # Template validation (26 tests)
    ├── test_branding.py         # Brand tokens + helpers (27 tests)
    ├── test_topicgen.py         # Topic generation (19 tests)
    ├── test_integration_end_to_end.py  # Full pipeline (requires API key)
    ├── validate_scoring.py      # Scoring formula validation (9 tests)
    └── verify_prototype.py      # Ad-hoc end-to-end smoke test
```

---

## Roadmap

### v2.1 — Q3 2026
- [ ] Domain format regex validation in `_normalize_domain()`
- [ ] SQLite cache layer for repeat audits (skip re-crawls, re-queries)
- [ ] Fuzzy brand matching (embeddings-based, beyond substring)
- [ ] `llms.txt` content parsing (not just presence check)

### v2.2 — Q4 2026
- [ ] Web wrapper (Flask) for browser-based lead capture form
- [ ] Sitemap-aware crawl extension (beyond 10-page BFS)
- [ ] Background job queue for production deployment
- [ ] PDF report export

### v3.0 — 2027
- [ ] Direct platform API integration (OpenAI, Anthropic, Google) — bypass OpenRouter
- [ ] Competitive benchmarking dashboards (compare domains side-by-side)
- [ ] Scheduled recurring audits with trend tracking
- [ ] Multi-language support (topic generation in 10+ languages)

---

## Security

A comprehensive security audit was performed for the v2.0 release. See
[docs/security-audit-2026-07-12.md](docs/security-audit-2026-07-12.md) for the
full report.

**Overall rating: B+** — 4 of 5 checks pass (hardcoded secrets, env vars, error
handling, output sanitization). Input validation is functional with low-severity
hardening opportunities.

Key security properties:
- ✅ No hardcoded secrets — all API keys via `os.environ.get()`
- ✅ `.env` is gitignored; `.env.example` ships empty
- ✅ Jinja2 HTML autoescaping enabled — no `|safe` bypasses
- ✅ Graceful error handling for all network failures
- ✅ Rate limiting (2s between calls) prevents accidental API abuse

---

## Support

- **GitHub Issues:** [github.com/jlvisualdg/ai-visibility-audit-tool/issues](https://github.com/jlvisualdg/ai-visibility-audit-tool/issues)
- **Documentation:** Start with [docs/scoring_system.md](docs/scoring_system.md) for scoring methodology
- **Contributing:** Open a PR against `main` — run full test suite before submitting

---

## License

MIT © 2026 Julian Lopez