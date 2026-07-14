# AEO Audit Tool — Comprehensive Fix & Improvement Plan

**Date:** 2026-07-13
**Author:** Engineering review (full-codebase pass)
**Decisions locked by product owner:**

1. **Web shape:** Embeddable lead-capture form → **async job API** (FastAPI + background worker + polling). Report stored and emailed.
2. **AI sourcing:** **OpenRouter** for Perplexity + ChatGPT; **DataForSEO** for Gemini citations (`DATAFORSEO_BASE64` already in `.env`).
3. **Headline metric:** **Composite weighted AEO Score (0–100)** = Visibility + Citation + Indexability sub-scores.
4. **Sequencing:** **Correctness first**, then web-enable.

> Goal, restated: a tool that lives on the AEO/Smart Marketer site, runs efficiently, and handles many concurrent audits (respecting OpenRouter limits). Each audit: scrape the site → derive 4 bottom-of-funnel buyer queries → run them through 3 AI platforms → simulate a ready-to-buy customer → score the target brand's **explicit recommendations** and **citations** (target domain + competitor domains) → roll up into one AEO score across several audit items.

---

## 0. What works today (keep it)

The end-to-end CLI pipeline runs and produces a branded HTML report:

`audit.py` → `crawler.crawl_domain()` → `topicgen.generate_botf_topics()` → `collector.execute_all()` → `scoring.aggregate_results()` → `analyzer.analyze()` → `reporter.generate_report()`.

- **Crawler** (`crawler.py`) extracts a solid set of AI-indexability signals (answer capsules, schema, authorship, robots/AI-bot blocking across 27 bots, llms.txt, a11y/agent-readability, thin pages, health score).
- **Topic generation** (`topicgen.py`) reverse-engineers 4 unbranded BOTF queries via `gpt-4o-mini`, validates against informational patterns, retries, and injects keyword variations. Good design.
- **Report template** is on-brand (Smart Marketer light design system) and self-contained.
- **Graceful degradation:** every network path has try/except and mock fallbacks.

The bones are good. The problems are (a) a headline metric that contradicts its own docs, (b) brand extraction that has been overfit into an unmaintainable state, (c) a fully sequential execution model that can't scale, and (d) no web/concurrency layer for the stated product.

---

## 1. Findings

### 1A. Correctness bugs (fix first — these erode trust in the number)

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| C1 | **High** | **Headline metric doesn't match the documented methodology.** `docs/scoring_system.md` specifies a position/mention-weighted `Query_Score`, and `calculate_ai_presence_score()` implements it and is validated by `validate_scoring.py`. But `aggregate_results()` actually computes `ai_presence_pct` as a **binary coverage %** (`covered_cells / total_cells`). The documented, tested formula is **never used** in the real pipeline. | `scoring.py:1059` (aggregate) vs `scoring.py:967` (unused fn) + `docs/scoring_system.md` |
| C2 | **High** | **Brand extraction is overfit to two clients** (a menopause/HRT telehealth brand and pricing pages). `AI_ISM_PHRASES`, `_DESCRIPTIVE_TERMS`, and `descriptive_suffixes` contain ~200 hardcoded phrases like `"perimenopause care"`, `"bioidentical hormone"`, `"winona pricing breakdown"`. It will silently mis-extract for any other industry. | `scoring.py:55-259`, `397-571`, `719-772` |
| C3 | **Medium** | **`common_suffixes` list in `_extract_target_tokens()` is corrupted** with pasted menopause/pricing phrases (e.g. checks whether a domain "ends with `perimenopause care`"). Nonsensical — copy-paste contamination across three functions. | `scoring.py:838-891` (also duplicated in `audit.py:141`) |
| C4 | **Medium** | **`--passes` / `PASSES_PER_QUERY` is advertised but ignored.** `execute_all()` runs exactly one pass per cell (`pass_count=1` hardcoded). README + `.env.example` promise variance-aware multi-pass. | `collector.py:110`, `audit.py:470` |
| C5 | **Medium** | **Citation quality is inconsistent across engines.** Perplexity returns structured `annotations`; ChatGPT/others fall back to `_extract_citations_from_text()` which regexes *any* domain-like token from prose. So `citation_count` mixes real source citations with incidental domain mentions, and isn't comparable across platforms. | `engines.py:100-110, 231-233` |
| C6 | **Low** | **robots.txt group parsing bug**: `User-Agent:` resets `current_agents` to a single item, so stacked user-agent lines sharing one rule block aren't grouped. `blocks_self` compares against `"AEO-Audit-Tool/0.1"` (includes version), so a rule naming just `AEO-Audit-Tool` never matches. | `crawler.py:973, 986` |
| C7 | **Low** | **`PageSnapshot.status_code` hardcoded to 200**; several per-page signals (`heading_skip_levels`, `form_inputs_labeled`, etc.) are computed but never aggregated (write-only). Health-score cap comments drift from code (says 80, caps at 85). | `crawler.py:469, 664, 834` |

### 1B. Architecture debt

| # | Finding | Detail |
|---|---------|--------|
| A1 | **Two overlapping engine stacks.** `visibility.py` (legacy v1: `EngineResult`, `MockEngine`, `build_engines`, `build_citation_matrix`, OpenRouter `ClaudeEngine`/`GeminiEngine`) and `engines.py`+`collector.py` (v2 flat-dict). The real pipeline uses v2 then **converts** to v1 `CitationMatrix` in `audit.py` just to satisfy the template. `visibility.py`'s engine machinery is largely dead. |
| A2 | **Four copies of the same helpers.** `_normalize_domain`, `_extract_target_tokens`, `_is_target_brand`, `_domain_contains_citation` are duplicated across `audit.py`, `scoring.py`, `visibility.py`, `crawler.py` — three of them drift independently. |
| A3 | **Three layers compute overlapping aggregates.** `scoring.aggregate_results()`, `audit._derive_v2_variables()`, and the Jinja template itself all derive "top brands / coverage / best topic." Logic belongs in one place. |
| A4 | **Dead code:** `engines.query_all_engines()`, `engines.ClaudeEngine/GeminiEngine`, most of `visibility.py`, `_has_schema` legacy wrapper, `page_text_lookup` param. |
| A5 | **Config sprawl + doc drift.** README claims 4 engines incl. Claude and "5 topics × 4 engines = 20 cells"; reality is 3 engines × 4 topics = 12 cells, Claude removed. `.env.example` lists unused direct-provider keys and omits `DATAFORSEO_BASE64`. |

### 1C. Performance (the "run efficiently / concurrent" ask)

| # | Finding | Impact |
|---|---------|--------|
| P1 | **Engine calls are fully sequential** with a **2s sleep between every call** (`collector.py:162`) *plus* a global 2s rate-limiter inside engines (`engines.py:71`). 12 cells → ~24s+ of pure sleeping before latency. DataForSEO Gemini calls have a 120s timeout each, serialized. A single audit can take 60-120s. | Slow; unusable under concurrent load. |
| P2 | **Crawler is single-threaded BFS** with `time.sleep(0.5)` per page, `queue.pop(0)` (O(n)), and **parses each page's HTML twice**. | Adds seconds per audit. |
| P3 | **No shared concurrency model or backpressure.** Nothing coordinates parallel audits or respects a global OpenRouter budget. | Concurrent audits would blow rate limits. |
| P4 | **No caching / dedupe / storage.** Every audit re-crawls and re-queries even for a domain audited minutes ago. Reports only written to disk. | Wasted API spend; no history. |

### 1D. Product gaps (the "live on my website" ask)

- No web service, no lead-capture form, no async job model, no result persistence, no email delivery.
- No input validation hardening for a public endpoint (SSRF on crawl target, domain format, rate limiting per IP).
- No observability (structured logs, per-audit cost/latency, error surfacing).

---

## 2. Target architecture

```
                       ┌─────────────────────────────────────────────┐
Browser (your site)    │  Embeddable form (JS snippet / iframe)        │
   email + domain ─────►  POST /api/audits  →  { audit_id, status }    │
   poll status     ◄────  GET  /api/audits/{id}                        │
   view report     ◄────  GET  /r/{id}  (HTML)                         │
                       └───────────────────┬─────────────────────────┘
                                           │ enqueue
                                  ┌────────▼─────────┐
                                  │  Job queue        │  (in-proc asyncio
                                  │  + worker pool    │   queue → Redis/RQ later)
                                  └────────┬─────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
     ┌────────────────┐          ┌──────────────────┐        ┌──────────────────┐
     │ crawl (async)  │          │ topicgen (LLM)   │        │ engine runner     │
     │ httpx + gather │          │                  │        │ (async, bounded)  │
     └────────────────┘          └──────────────────┘        └────────┬─────────┘
                                                                       │ 3 platforms × 4 queries
                                                     ┌─────────────────┼─────────────────┐
                                                     ▼                 ▼                 ▼
                                              OpenRouter        OpenRouter         DataForSEO
                                              (Perplexity)      (ChatGPT)          (Gemini)
                                                     └─────────────────┼─────────────────┘
                                                                       ▼
                                              ┌────────────────────────────────────────┐
                                              │ analysis: brand extraction (LLM) +       │
                                              │ scoring (visibility / citation / index)  │
                                              └───────────────────┬──────────────────────┘
                                                                  ▼
                                              ┌────────────────────────────────────────┐
                                              │ composite AEO score → report → store →   │
                                              │ email deliver                            │
                                              └────────────────────────────────────────┘

Cross-cutting: shared config, one rate-limiter/budget per provider, SQLite store, structured logs.
```

**Key principle:** one canonical result model flows through the whole pipeline. Kill the v1↔v2 conversion in `audit.py`.

---

## 3. Phased plan

Each phase is independently shippable and leaves the tree green. Correctness (Phases 0–3) lands before the web layer (Phases 4–5), per the locked sequencing.

### Phase 0 — Foundations & cleanup (no behavior change)

- **0.1 `src/common.py`** — single home for `normalize_domain`, `extract_target_tokens`, `is_target_brand`, `domain_matches`, brand-name normalization. Delete the 3–4 duplicates; import from here everywhere.
- **0.2 `src/config.py`** — centralize env (`OPENROUTER_API_KEY`, `DATAFORSEO_BASE64`, model IDs, rate limits, concurrency caps) into a typed settings object loaded once. No scattered `os.environ.get`.
- **0.3 Delete dead code** — remove `engines.query_all_engines`, unused `ClaudeEngine`/`GeminiEngine` (OpenRouter), the unused half of `visibility.py`, `_has_schema` wrapper, `page_text_lookup`. Keep only the dataclasses the template needs (move them to a `models.py`).
- **0.4 Canonical result model** — define `EngineResult`, `QueryResult`, `AuditResult` dataclasses in `src/models.py`. Retire the `_build_citation_matrix_from_results` conversion shim.
- **0.5 Docs truth-up** — fix README (3 platforms, 4 queries, 12 cells, DataForSEO requirement) and `.env.example` (add `DATAFORSEO_BASE64`, drop unused keys).

*Exit:* tests green, no duplicated helpers, one result model, honest docs.

### Phase 1 — Scoring correctness

- **1.1 Decide and implement ONE visibility formula.** Adopt the documented **position + mention weighted** `Query_Score` as the *Visibility sub-score* (it already exists and is tested). Wire `aggregate_results()` to actually call `calculate_ai_presence_score()` per cell and average — resolving C1. Keep binary coverage as a secondary "reach %" stat, clearly labeled.
- **1.2 Separate the two metrics cleanly** (the product's core distinction):
  - **Brand Recommendation** = target brand explicitly named in the answer text (position-aware). This is the Visibility sub-score.
  - **Citation** = target domain appears as a *source*. Score target vs. competitor domains; report share-of-citations.
- **1.3 Competitor scoring** — rank competitor brands (recommendations) and competitor domains (citations) with the same math, so the report shows "who's beating you and by how much."
- **1.4 Update `docs/scoring_system.md` + `validate_scoring.py`** to match the shipped formula exactly. The doc and the code must agree (this is the whole point of C1).

### Phase 2 — Brand extraction rebuild (kills C2/C3)

- **2.1 Replace the regex+denylist extractor with an LLM structured-extraction pass.** For each engine response, one cheap LLM call (`gpt-4o-mini`, JSON mode) returns `{recommended_brands: [{name, position, is_target, mention_count}], cited_domains: [...]}`. This generalizes to any industry — no per-client phrase lists.
- **2.2 Keep the regex extractor as a deterministic fallback** (LLM down / budget guard), but strip the overfit menopause/pricing phrase lists down to a small, genuinely-generic stopword/AI-ism set.
- **2.3 Fix `_extract_target_tokens`** — remove the corrupted `common_suffixes` entries (C3); derive tokens from domain + scraped brand name only.
- **2.4 Golden tests** — snapshot real responses from 3–4 diverse domains (e.g. `paretotalent.com`, `bywinona.com`, a SaaS, an agency) as fixtures; assert extraction quality so we never re-overfit.

### Phase 3 — Composite AEO score + report

- **3.1 Composite score** (`src/aeo_score.py`):
  ```
  AEO Score (0–100) = 0.45 · Visibility + 0.30 · Citation + 0.25 · Indexability
  ```
  - **Visibility** — mean position/mention-weighted brand-recommendation score across 12 cells.
  - **Citation** — target's share of brand-relevant citations (target + competitor domains), across cells.
  - **Indexability** — existing crawl `health_score`, recalibrated.
  - Weights live in `config.py`, documented, tunable. Show the 3 sub-scores as a breakdown so the number is explainable.
- **3.2 Report refresh** — headline hero = AEO Score with the 3 sub-score meters; keep the brand-recommendation matrix (4 queries × 3 platforms), competitor tables, citation share, and indexability cards. Move all aggregate computation out of Jinja into the scoring layer (fixes A3); template only renders.
- **3.3 Analyzer** — regenerate fixes/priorities from the composite sub-scores (e.g. low Citation → digital-PR/linkable-asset plays; low Indexability → schema/answer-capsule plays).

### Phase 4 — Concurrency, rate limits & performance (the efficiency ask)

- **4.1 Async engine runner** (`src/runner.py`) — convert engine calls to `httpx.AsyncClient` and run the 12 cells with `asyncio.gather` under a bounded semaphore. Remove the blanket `time.sleep(2)` in `collector.py`.
- **4.2 Per-provider rate limiting / budget** (see §4 detail below) — a shared async token-bucket per provider (OpenRouter, DataForSEO) so that *all* in-flight audits collectively respect the limit. This is what makes "many requests at a time" safe.
- **4.3 Retries with backoff** — honor OpenRouter `429` + `Retry-After`; exponential backoff with jitter; cap attempts. Distinguish retryable (429/5xx/timeout) from fatal (4xx auth).
- **4.4 Multi-pass support** — implement `--passes` (C4): run N passes per cell concurrently, aggregate with variance shown. Default 1 for web (cost), 3 for client deliverables.
- **4.5 Crawler speedup** — async fetch with `httpx` + `asyncio.gather` (bounded), `deque` queue, single HTML parse per page, drop the per-page sleep (rely on concurrency cap + politeness delay only where needed) (P2).
- **4.6 Caching** — SQLite cache keyed by (domain, day) for crawl + engine responses; short TTL; skip re-work for repeat audits (P4).

### Phase 5 — Web service, storage & delivery (the "on my website" ask)

- **5.1 FastAPI app** (`src/web/`):
  - `POST /api/audits` `{domain, email, consent}` → validates, dedupes, enqueues, returns `{audit_id}`.
  - `GET /api/audits/{id}` → `{status: queued|running|done|error, progress, aeo_score?}`.
  - `GET /r/{id}` → the rendered HTML report.
- **5.2 Job queue + worker** — start with an in-process `asyncio` queue + worker pool bounded by the global concurrency cap; design the interface so it can swap to Redis/RQ/Celery when traffic warrants. Each job = one `AuditResult` through the Phase 0–3 pipeline on the Phase 4 async runner.
- **5.3 Persistence** — SQLite (`audits` table: id, domain, email, status, score json, created_at, report_path). Migratable to Postgres.
- **5.4 Lead capture + email** — store email as the lead; email the report link on completion (transactional provider, e.g. Resend/SES — needs a decision + key). Double-opt-in / consent checkbox for compliance.
- **5.5 Embeddable front-end** — a small JS snippet or iframe for the AEO site: form → submit → progress → report link. Styled with the existing design system.
- **5.6 Public-endpoint hardening:**
  - **SSRF guard** on the crawl target — resolve DNS, reject private/loopback/link-local IP ranges, block non-http(s) schemes and redirects to internal hosts. (Critical: you're fetching user-supplied URLs server-side.)
  - Domain-format validation (closes the security-audit C3 gap).
  - Per-IP + per-email rate limiting on `POST /api/audits`.
  - Secrets only via env; never in responses or logs.

### Phase 6 — Observability, tests, deploy

- Structured logging per audit (crawl time, per-engine latency, tokens/cost, errors), a `/health` endpoint, and a simple metrics counter.
- Test matrix: unit (scoring, extraction, rate-limiter), integration (mocked HTTP for the full pipeline), a small live smoke test behind an env flag. Fix the misleading `validate_scoring.py` (now validates the *used* formula).
- Deploy: containerize (Dockerfile), `uvicorn`/`gunicorn` workers behind the host's reverse proxy, env-based config, `output/` → object storage or DB blob.

---

## 4. Concurrency & OpenRouter rate-limit design (detail)

OpenRouter enforces limits by (a) requests/interval tied to your credit balance and (b) per-model throughput, and returns **HTTP 429** (sometimes with `Retry-After`) plus daily caps on free-tier models. Design so that *N concurrent audits* never exceed those collectively:

1. **Global async token-bucket per provider** (module-level singletons): `openrouter_limiter`, `dataforseo_limiter`. Every engine call `await limiter.acquire()` before firing. Configure rate from `config.py` (e.g. OpenRouter N req/s, DataForSEO lower — it's slower/costlier).
2. **Bounded concurrency** at two levels: per-audit (`asyncio.Semaphore` over the 12 cells) *and* global (worker pool size). The global cap × per-audit cap ≤ provider budget.
3. **Queue, don't drop.** Audits beyond capacity wait in the job queue; the API returns `queued` immediately (async model already supports this).
4. **Backoff on 429/5xx** with jitter and `Retry-After`; a circuit-breaker that pauses a provider briefly after repeated 429s so one hot audit doesn't starve others.
5. **Cost/token budget guard** per audit and per day (config) — refuse or downgrade (fewer passes) when exceeded, and log it (no silent truncation).
6. **Prefer paid, citation-returning models** for reliability; keep model IDs in config so they're swappable without code changes. Re-validate that the chosen ChatGPT/Perplexity models still return usable citations.

---

## 5. File-by-file change map (summary)

| File | Action |
|------|--------|
| `src/common.py` | **New** — shared domain/brand helpers (dedupe A2). |
| `src/config.py` | **New** — typed settings/env, model IDs, rate + concurrency caps. |
| `src/models.py` | **New** — canonical `EngineResult`/`QueryResult`/`AuditResult`. |
| `src/runner.py` | **New** — async, rate-limited engine runner (replaces sequential `collector.execute_all`). |
| `src/aeo_score.py` | **New** — composite score + sub-scores. |
| `src/ratelimit.py` | **New** — async token-bucket + backoff per provider. |
| `src/web/` | **New** — FastAPI app, routes, job queue, worker, storage, email, embed snippet. |
| `src/scoring.py` | **Rewrite** — thin: call weighted formula; LLM extraction + slimmed fallback; drop overfit lists (C1/C2/C3). |
| `src/engines.py` | **Slim** — async OpenRouter client only; delete dead engines/`query_all_engines` (A4). |
| `src/gemini_dataforseo.py` | **Keep**, make async, route through the DataForSEO limiter. |
| `src/collector.py` | **Retire** → folded into `runner.py`. |
| `src/visibility.py` | **Retire** engine machinery; move needed dataclasses to `models.py` (A1). |
| `src/crawler.py` | **Refactor** — async fetch, `deque`, single parse, fix robots parsing (P2/C6/C7). |
| `src/analyzer.py` | **Update** — fixes derived from sub-scores. |
| `src/reporter.py` / `templates/report.html` | **Update** — AEO hero + sub-scores; move aggregates out of Jinja (A3). |
| `audit.py` | **Simplify** — CLI wraps the same pipeline the web worker uses; drop the mock/conversion sprawl. |
| `docs/scoring_system.md`, `README.md`, `.env.example` | **Truth-up** (A5, C1). |
| `tests/` | **Expand** — golden extraction fixtures, rate-limiter tests, mocked pipeline; fix `validate_scoring.py`. |

---

## 6. Risks & open items (need a decision before/at that phase)

1. **Email provider** for report delivery (Resend / SES / Postmark?) — needs an account + key. Blocks Phase 5.4.
2. **Hosting/runtime** for the web service (where does it deploy — your existing site host, a VPS, a PaaS?) — shapes Phase 6 and the queue choice (in-proc vs Redis).
3. **DataForSEO cost/latency**: Gemini calls are slow (up to 120s) and metered. Confirm budget, and whether Gemini is required for every audit or can be optional/async-deferred.
4. **Model selection**: confirm current OpenRouter model IDs still return citations (Perplexity annotations, ChatGPT search-preview). Re-validate before Phase 1.
5. **LLM extraction cost**: one extra `gpt-4o-mini` call per cell (~12/audit) — cheap, but include in the per-audit budget guard.
6. **Compliance**: storing emails = lead data. Consent checkbox + privacy note + unsubscribe. Confirm what your site already does.

---

## 7. Suggested execution order (correctness-first)

1. Phase 0 (foundations) → 2. Phase 1 (scoring truth) → 3. Phase 2 (extraction rebuild) → 4. Phase 3 (composite + report) — **now the number is trustworthy.**
5. Phase 4 (async + rate limits) → 6. Phase 5 (web + leads) → 7. Phase 6 (observability + deploy) — **now it scales on your site.**

Phases 0–3 are pure correctness/cleanup with no new infra and can land as small PRs. Phase 4 is the pivot to async and should be its own reviewed PR because it touches every network call.
