# AI Visibility Audit Tool

A CLI tool that audits a target domain's **AI visibility** — how often it gets
cited when buyers ask AI engines about topics it should own. Generates a
dark-themed HTML report with a citation matrix, crawl health signals, and
prioritized fix recommendations.

## What it does

For any domain you give it, this tool will:

1. **Crawl** the site (up to 10 pages by default) and extract AI-readiness
   signals: answer capsules, stat density, authorship, schema, robots.txt,
   content depth, internal linking hygiene.
2. **Generate 4 buyer-intent topics** the site should be cited for (heuristic
   keyword extraction, optionally LLM-enhanced).
3. **Run a citation check** across AI engines — for each topic × platform
   combination, query the engine, extract cited domains, and check whether
   the target domain appears among them.
4. **Build a citation matrix** showing topic coverage per platform and a
   ranked list of competitor domains.
5. **Synthesize a strategic read** with prioritized fixes (heuristic
   baseline, optionally LLM-enhanced for richer analysis).
6. **Render a dark-themed HTML report** ready to share with a prospect or
   hand to a content team.

## Quick start

```bash
cd ~/Desktop/aeo-audit-tool
source .venv/bin/activate
pip install -r requirements.txt

# Run in pure mock mode (no API keys needed)
python audit.py --domain example.com --no-ai --max-pages 3

# Run with the real Perplexity engine (recommended for real audits)
export OPENROUTER_API_KEY=sk-or-v1-...
python audit.py --domain paretotalent.com --max-pages 10
```

The report lands in `output/<domain>-audit-<timestamp>.html`.

## Engines and their modes

| Engine   | Default mode | Wire-up |
|----------|--------------|---------|
| Perplexity | **Real** (via OpenRouter) | Set `OPENROUTER_API_KEY`. Uses `perplexity/sonar-pro` for search-grounded responses. |
| ChatGPT   | Mock         | Set `OPENAI_API_KEY` to enable the real Responses API (uses `web_search` tool). |
| Claude    | Mock         | Set `ANTHROPIC_API_KEY` to enable the real Anthropic API (uses `web_search_20250305` tool). |
| Gemini    | Mock         | Set `GOOGLE_API_KEY` to enable the real Gemini API with `google_search` grounding. |

**v1 scope is mock-first, ONE real engine.** Perplexity via OpenRouter is the
recommended starting point because it's cheap, search-grounded, and exposes
citations in a parseable annotation format. The other three are stubbed in
mock mode with the protocol in place — they slot in one at a time as keys
become available. See `Engine protocol` below.

## Engine protocol

Every engine implements the same interface (see `src/visibility.py`):

```python
class Engine(Protocol):
    name: str                         # e.g. "perplexity" or "perplexity (mock)"
    def query(self, topic: str) -> EngineResult: ...
    def is_real(self) -> bool: ...
```

`EngineResult` carries:
- `text`: the raw model response
- `citations`: list of cited source URLs (already extracted by the engine
  client, not regex-scraped from text)
- `latency_ms`, `model`: metadata for the run log

Mock engines return deterministic fake citations so the whole pipeline
works without any API key. The mock's `name` field reflects the slot it
filled (e.g. `"chatgpt (mock)"`) so nothing in the report is mistaken for
a real result.

## Crawl signals

- **answer_capsules** — count of H2/H3 headings followed by a direct
  `<p>` answer within 200 characters (the "answer-first content"
  pattern AI engines like to cite).
- **stat_density** — data points (numbers, percentages, currency)
  per 100 words. Higher = more quotable.
- **authorship_pages** — pages with visible author bylines, author
  schema, or `author` meta tags. Trust signal.
- **schema_pages** — pages with JSON-LD or microdata schema markup.
- **health_score** — composite 0–100 score from the above + access
  hygiene (robots.txt, AI bot directives, thin pages).
- **thin_pages** — pages under 300 words.
- **missing_anchor_text** — internal links with empty or generic
  anchor text ("click here", "read more").
- **has_ai_txt / has_llms_txt / robots_blocks_ai** — accessibility
  signals for AI crawlers.

## Citation matrix

A `topics × platforms` grid. For each cell:
- `covered`: true if the target domain appears in that platform's
  cited sources for that topic
- `cited_sources`: list of competitor domains the engine cited

Plus a ranked list of competitor domains across all cells.

## Fix recommendations

The analyzer generates 5–7 prioritized fixes from the crawl + citation
data using a heuristic baseline. Examples:

- If `answer_capsules == 0` → HIGH: "Add answer-first sections to key pages"
- If `thin_pages > 0` → HIGH: "Pages with thin or non-extractable content"
- If `missing_anchor_text > 0` → MEDIUM: "Pages with internal links missing
  anchor text"
- If `ai_coverage_pct == 0` → HIGH: Strategic read about building
  topical authority


Each fix has a `priority` (HIGH / MEDIUM / LOW), a `tag`
(WORTH CITING / FOUNDATION / RECOMMENDED), a one-line `first_step`,
and an `agent_fixable` flag.

## v1 limits (read before promising clients anything)

- **Citation check uses substring matching** for `domain in cited_source`.
  It will miss misspellings, paraphrases, and brand variants. Smarter
  detection (fuzzy match, embeddings, brand-alias list) is a phase 2
  problem.
- **Gemini API with Google Search grounding is NOT the same as showing
  up in Google AI Overviews.** AI Overviews is the box in Google Search
  results and has no public API. The Gemini API is the closest
  programmatic proxy, but ranking and source selection can differ.
  Don't let your clients conflate the two.
- **Buyer topic generation is heuristic** when no LLM is available —
  keyword extraction + service/category patterns. Good enough for
  v1; richer when an LLM is wired in.
- **Crawl is capped at 10 pages by default** to keep audit runtime
  under ~30 seconds. Large sites need a sitemap-aware extension.
- **No caching** — every audit re-crawls and re-queries. Add a SQLite
  cache layer for client re-runs (planned for v2).
- **Mock engines return plausible but invented citations.** Always
  check the `name` field for `(mock)` suffix before reporting results
  to a client.

## Project layout

```
aeo-audit-tool/
├── audit.py                  # CLI entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── crawler.py            # Domain crawl + signal extraction
│   ├── visibility.py         # Engine protocol + 4 clients + citation matrix
│   ├── analyzer.py           # Strategic analysis + fix generation
│   ├── reporter.py           # HTML report generation
│   └── templates/
│       └── report.html       # Jinja2 dark-themed report template
├── output/                   # Generated reports land here
└── tests/
```

## Development

```bash
source .venv/bin/activate
python audit.py --domain example.com --no-ai --max-pages 3   # smoke test
python tests/verify_prototype.py                             # ad-hoc end-to-end verification
```

## Future

- Web wrapper (Flask) for browser-based lead capture form
- Background job queue for production deployment
- Cache layer for repeat audits
- Smarter brand-cited detection (fuzzy match, embeddings, brand aliases)
- LLM-enhanced strategic read + LLM-enhanced topic generation (heuristic
  currently produces noun-phrase bigrams from headings; an LLM pass
  would turn them into proper buyer-intent questions like "best
  delegation system for founders")
- Wire real Claude, ChatGPT, and Gemini engines (one at a time, with
  regression tests per the aeo-tracking skill methodology)
