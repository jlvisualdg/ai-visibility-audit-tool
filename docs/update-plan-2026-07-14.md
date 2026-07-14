# AEO Audit Tool — Merged Update & Report-Restructure Plan

**Date:** 2026-07-14
**Supersedes/merges:** [`improvement-plan-2026-07-13.md`](improvement-plan-2026-07-13.md) (engineering) + [`redesign-spec-2026-07.md`](redesign-spec-2026-07.md) (report/messaging)

## Locked decisions (product owner, 2026-07-14)

| # | Decision | Value |
|---|----------|-------|
| D1 | **Sequencing** | **Correctness first, then redesign.** Fix the scoring/extraction bugs so the redesigned buckets display trustworthy numbers, *then* restructure the report. |
| D2 | **3-bucket composite weights** | **Visibility 50% · Credibility 25% · Indexability 25%.** |
| D3 | **Brand identity** | **Keep Smart Marketer** in header/footer/CTA. Only the *score label* is renamed to **"BE THE ANSWER® AEO Score"** (redesign Change 1A). |
| D4 | **Credibility bucket source** | Split signals **by meaning, not by web-accessibility**: Credibility = About page, **social-channel count**, authorship, trust/reviews, NAP, contact. Indexability keeps the technical/crawlability signals. Recalibrate `health_score` so nothing is double-counted. |

---

## Guiding model — the new score decomposition

The composite changes from `Visibility(45) + Citation(30) + Indexability(25)` to a **three-bucket** model. "Citation" is no longer a top-level bucket — it folds into **Visibility**. A brand-new **Credibility** bucket is carved out of signals currently buried inside `health_score`.

```
BE THE ANSWER® AEO Score (0–100)
  = 0.50 · Visibility   (green)
  + 0.25 · Credibility  (amber)
  + 0.25 · Indexability (blue)
```

### Bucket → data-field mapping (map every existing field to exactly one bucket)

| Bucket | Colour | Fields (all already collected unless noted) |
|--------|--------|----------------------------------------------|
| **Visibility** 🟢 | `#065F46` / bg `#D1FAE5` / border `#6EE7B7` | Brand Recommendation Matrix (`citation_matrix.results`, `target_brand_position`), Top Recommended Brands (`all_competitors`), Competitive Landscape, Topics to Optimize (`zero_topics`), `ai_presence_pct`, citation share |
| **Credibility** 🟡 | `#92400E` / bg `#FEF3C7` / border `#FCD34D` | `authorship_pages`, **social-channel count (NEW — see §Phase R1)**, `has_trust_signals_on_homepage` (reviews/testimonials), `has_contact_info_on_homepage` (NAP), `has_about_page`, `has_contact_page`, `has_privacy_policy`/`has_terms` |
| **Indexability** 🔵 | `#1E40AF` / bg `#DBEAFE` / border `#93C5FD` | `answer_capsules`, `stat_density`, `schema_pages`/`schema_types_found`, `thin_pages`, `has_llms_txt`, `robots_blocks_ai`/AI-bots, `avg_response_ms`, `avg_agent_readability`, `pages_with_landmarks`, `total_images_missing_alt`, `broken_links`, `max_redirect_hops`, `has_ssl` |

Bucket badge thresholds (redesign): **STRONG** 70–100 · **DEVELOPING** 40–69 · **CRITICAL** 0–39. Replaces today's 4-tier Strong/Emerging/Limited/Critical in [`aeo_score.py`](../src/aeo_score.py).

---

## PART A — Correctness first (engineering, blocks the redesign)

Only the subset of the July-13 improvement plan that the buckets *depend on*. Full architecture/web items (§A1–A5, Phases 4–6 of that plan) are deferred to Part C.

### A0 — Foundations (no behaviour change)
- `src/common.py` — consolidate the 4 duplicated helpers (`normalize_domain`, `extract_target_tokens`, `is_target_brand`, `domain_matches`) that drift across `audit.py`/`scoring.py`/`visibility.py`/`crawler.py`.
- `src/config.py` — one typed settings object (API keys, model IDs, rate limits, **bucket weights D2**). Weights must live here, not hardcoded.

### A1 — Fix C1: the headline metric actually runs (High)
`aggregate_results()` computes a **binary coverage %** while the documented, tested position/mention-weighted `calculate_ai_presence_score()` is **never called**. Wire the weighted formula in as the **Visibility** sub-score; keep binary coverage as a secondary "reach %" stat. Update [`docs/scoring_system.md`](scoring_system.md) + `validate_scoring.py` to match the shipped formula.

### A2 — Fix C2/C3: brand-extraction overfit (High/Med)
~200 hardcoded menopause/HRT/pricing phrases in `scoring.py` mis-extract for any other industry, and `common_suffixes` in `_extract_target_tokens` is copy-paste-corrupted. Replace with an LLM structured-extraction pass (`gpt-4o-mini`, JSON mode) returning `{recommended_brands, cited_domains}`; keep a slim generic-stopword regex fallback. Add golden-fixture tests across 3–4 diverse domains so we never re-overfit.

### A3 — Credibility sub-score extraction (D4 — new, blocks redesign Change 2)
- Add `credibility_score` (0–100) computed from the **meaning-based** credibility set in the mapping table above.
- **Recalibrate `_compute_health_score`** ([crawler.py:1168](../src/crawler.py:1168)): remove the credibility bonuses currently mixed in (`has_about_page +5`, `has_contact_page +3`, `has_contact_info +3`, `has_trust_signals +5`, `has_social_links +2`, privacy/terms) so Indexability = purely technical and nothing is double-counted. Renormalize the technical deductions to still span 0–100.
- Extend `compute_all()` in [`aeo_score.py`](../src/aeo_score.py) to return `visibility_score`, `credibility_score`, `indexability_score` and the D2-weighted composite. Drop the standalone `citation_score` top-level output (fold citation share into Visibility).

*Exit for Part A:* tests green; composite = 3 buckets with correct numbers; docs honest.

---

## PART B — Report restructure (the 3 approved redesign changes + messaging)

Template: [`src/templates/report.html`](../src/templates/report.html). Reporter: [`src/reporter.py`](../src/reporter.py). **All aggregate computation moves to Python (scoring/aeo_score); the template only renders** (fixes improvement-plan A3).

### R1 — Social-channel count (the one sanctioned new data point)
`_credibility_signals` ([crawler.py:1643](../src/crawler.py:1643)) currently returns `has_social_links: bool`. Change to **count distinct platforms** among LinkedIn, Instagram, Facebook, X/Twitter, YouTube, TikTok, Pinterest on the homepage.
- `social_channel_count ≥ 3` → ✅ pass; `< 3` → ❌ credibility failure → **auto-generates a P1 fix card**.
- Display count + threshold as a scored item in the Credibility section.

### R2 — Change 1: rename + cut disclaimer
- **1A:** Score hero label → **"BE THE ANSWER® AEO Score"** everywhere (currently `{{ aeo_label }} AEO Score` renders as "Critical AEO Score" at [report.html:1501](../src/templates/report.html:1501)). Separate the fixed brand label from the tier badge (STRONG/DEVELOPING/CRITICAL).
- **1B:** Delete the `.data-warning` block ([report.html:1556–1578](../src/templates/report.html:1556)) and its CSS ([report.html:416–459](../src/templates/report.html:416)). Cut clean, no replacement.

### R3 — Change 2: three score buckets
- Under the composite hero, a **horizontal trio** of sub-score cards — each filled with its bucket bg colour, bordered in its bucket border colour, showing the 0–100 sub-score + STRONG/DEVELOPING/CRITICAL badge.
- Each of the three report sections gets a **4px solid left-border accent** in its bucket colour:
  - 🟢 Visibility section wraps the existing Matrix + Top Brands + Competitive Landscape + Topics (unchanged logic per "What Not to Change").
  - 🟡 Credibility section — **new** — renders authorship, social-channel count, trust/reviews, NAP, about/contact as scored items.
  - 🔵 Indexability section — the existing "Agent Indexability Audit" cards, minus the credibility signals moved to amber.
- Add bucket colour tokens to `:root`.

### R4 — Change 3: unified "Your AEO Fix List"
Replace "Prioritized Indexability Issues" ([report.html:2039](../src/templates/report.html:2039)) **and** "Topics to Optimize" pills with one section. Every card traces to a real finding — no generic advice.
- **Unify the two fix sources** — `analyzer._generate_fixes()` (HIGH/MED/LOW `FixRecommendation`) and `crawl.issues` (`CrawlIssue` severity+category) — into one card model: `{bucket, priority (P1/P2/P3), title, meaning, action}`.
- Map `category`/source → bucket (credibility categories → amber, technical → blue, zero-presence topics → green). Map severity → priority. Social-count failure → forced P1.
- **Sort:** P1 → P2 → P3; within priority, Indexability → Credibility → Visibility.
- Card: 4px left border in bucket colour, white bg, subtle shadow, bucket badge + priority badge.

*Exit for Part B:* report renders 3 buckets + unified fix list; matrix/landscape/top-brands/CTA untouched; template contains no aggregate logic.

---

## PART C — Deferred (post-redesign, from July-13 plan)

Not blocking the redesign; schedule after Parts A+B ship: architecture cleanup (kill dual v1/v2 engine stacks, `visibility.py` dead code, conversion shim), performance (async runner, per-provider token-bucket rate limiting, `--passes` C4, crawler concurrency, SQLite cache), and the web product (FastAPI async job API, persistence, lead-capture form + email, SSRF hardening).

---

## Open items needing a later decision
- **Visibility internal blend** — the bucket now contains both brand-recommendation and citation. Proposed: Visibility sub-score = 0.6·recommendation + 0.4·citation-share (tunable in `config.py`). Confirm the split.
- **Credibility scoring curve** — how many of the ~7 credibility signals = STRONG? Proposed: linear % of signals passed, with social-count and authorship weighted double.
