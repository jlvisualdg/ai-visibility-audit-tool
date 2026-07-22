# Part D — Build Plan (2026-07-22)

Approved by Julian. Deepens the two shallowest scoring inputs (schema, credibility)
and closes the biggest data-capture gap (people/entity extraction), then wires every
signal to a page-scoped fix. Build order is **1 → 2 → 3 → 4** (4 depends on 1–3).

> **Baseline:** work on top of `main` @ `792f9a4` (Part A+B recovered). If the Part A+B
> markers (`bucket_label`, `_compute_credibility_score`, `header-bta-badge`) are ever
> missing, restore the 6 files from commit `01f3eeb` (a clean superset).

---

## Context — what's shallow today

- **Schema:** `_has_schema_with_types` only collects `@type` strings; `schema_pages` is a
  page count. No graph, no completeness, no connectivity check.
- **Credibility:** `_extract_credibility_signals` returns ~5 booleans via brittle regex-OR
  over lowercased homepage text; `_compute_credibility_score` is a flat datapoint tally.
- **People/entities:** the crawler captures `has_about_page` + `about_url` but extracts
  **zero people** — no founder/team names, roles, bios, or credentials.
- **Fixes:** `_generate_fixes` in `analyzer.py` is a hand-written if-ladder, disconnected
  from most extracted signals.

---

## 1. Schema graph validator → quality score (not count)

**Files:** `src/crawler.py` (new `_analyze_schema_graph`), `src/aeo_score.py`, `src/templates/report.html`

- Parse homepage JSON-LD into the full object graph (follow `@graph`), not just `@type`.
- Score a **0–100 schema quality** on presence + completeness + connectivity:
  - Core entities on homepage: `Organization`, `WebSite`, `LocalBusiness` (or specific
    subtype) — presence.
  - `Organization` completeness: `name`, `logo`, `url`, `contactPoint`, `sameAs[]`
    (each property scores).
  - `WebSite`: nested/linked to Org; `potentialAction` = `SearchAction`.
  - Connectivity: every main entity has `@id`; `publisher`/`author` resolve to the
    Org `@id`.
  - `sameAs` targets point to social profiles / Wikidata.
- New fields on `PageSnapshot`/`CrawlResult`: `schema_entities`, `schema_quality_score`,
  `schema_missing_props`.
- Each missing property emits a homepage-scoped `CrawlIssue`.
- Feeds the Indexability bucket; surface in the Indexability section + fix list.

## 2. NEEATT credibility rework

**Files:** `src/crawler.py` (`_extract_credibility_signals` → NEEATT extractors,
`_compute_credibility_score`), `src/templates/report.html` (credibility section)

Extract signals, split by page scope:

- **Homepage:** media logos / "as seen on" (Notability); stat counters e.g. `1,200+`
  (Experience); value-prop clarity (Expertise); ratings/review widgets — Trustpilot,
  Google (Authoritativeness); footer NAP + cert/payment badges (Trust); privacy/terms/
  contact links (Transparency).
- **About:** press/Wiki links; company timeline / case studies; staff bios + degrees/
  licenses; memberships / board seats; corporate registration (EIN/LLC); real-team-photo
  vs stock-image heuristic.

Rebuild `_compute_credibility_score` as a **NEEATT rubric** — 6 sub-scores (Notability,
Experience, Expertise, Authoritativeness, Trustworthiness, Transparency) weighted into a
0–100 credibility score — replacing the current flat 8-datapoint tally. Regroup the
report's credibility section under the 6 dimensions; each unmet signal → fix card.

## 3. Input-capture fallbacks

**Files:** `src/crawler.py`

- About discovery: add `/leadership`, `/our-team`, `/people`, `/founders`; if path guesses
  + nav link both fail, scan homepage **footer** links and `sitemap.xml`.
- People extraction: pull names/roles/bios from About/team pages into a
  `people: list[Person]` structure (name, role, has_bio, has_credentials).
- NAP: add email + footer-scoped scan; structured address capture.

## 4. Unified signal → checklist map + fix rewrite

**Files:** new `src/checklist.py`, `src/analyzer.py` (`_generate_fixes` rewritten)

- One declarative table: `signal → checklist item → bucket → priority → page scope → fix copy`.
- Rewrite `_generate_fixes` as a loop over unmet signals against that table (replacing the
  if-ladder), so **every** extracted datapoint deterministically drives a specific,
  page-scoped fix card. Guarantees the new schema/NEEATT signals from steps 1–2 reach the
  fix list.

---

## Per-step done criteria

Each step ships with unit tests and a live re-run on `www.micromatic.com` to sanity-check
output before moving to the next.

## Known pre-existing issue (out of scope, do not confuse with new work)

`tests/test_topicgen.py` has ~7 failures from `mock` `side_effect` iterator exhaustion
raising `StopIteration` under Python 3.14. Unrelated to Part D; fix separately if desired.
