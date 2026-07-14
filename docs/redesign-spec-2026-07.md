# AEO Audit Tool — Redesign Spec

**Agency:** AEO.co · **Brand:** BE THE ANSWER®  
**Status:** Approved, not yet implemented · **Date:** July 2026

> Implementation note: changes are to the audit output/template only — core data extraction and scoring logic are not modified. Read the existing template and data schema before touching layout. Map every current data field before reorganising.

---

## Change 1 — Rename the Score + Cut the Disclaimer

### A. Rename score label
`"Critical AEO Score"` → `"BE THE ANSWER® AEO Score"`  
Apply everywhere the label appears in the report.

### B. Remove the data disclaimer entirely
Find and delete the block starting with:
> "Some data may be inaccurate as the indexability signals below reflect only the 32 pages we were able to read…"

No replacement copy. Cut it clean.

---

## Change 2 — Three Score Buckets

Replace the single score structure with three visually distinct sub-scores. Every existing data point must be assigned to **exactly one** bucket. Do not create new data requirements — only reorganise and relabel what already exists.

### The Three Buckets

| Bucket | Colour | Background | Border | What goes here |
|--------|--------|-----------|--------|----------------|
| **Indexability** | `#1E40AF` | `#DBEAFE` | `#93C5FD` | All current "Agent Indexability Audit" data: content structure, schema & entity, response zone, agent readability scores |
| **Credibility** | `#92400E` | `#FEF3C7` | `#FCD34D` | Authorship signals, NEEATT-related checks, social media presence, E-E-A-T signals, review signals, author bios, credentials |
| **Visibility** | `#065F46` | `#D1FAE5` | `#6EE7B7` | Brand Recommendation Matrix, Top Recommended Brands, Competitive Landscape, Topics to Optimize |

### Visual distinction rules
- Each bucket has its own section with a **4px solid left-border accent** in the bucket colour
- The three sub-score cards sit beneath the main BE THE ANSWER® AEO Score as a **horizontal trio**, each filled with its bucket background colour and bordered in its bucket border colour
- Score badge labels: **STRONG** (70–100) · **DEVELOPING** (40–69) · **CRITICAL** (0–39)
- The composite BE THE ANSWER® AEO Score = weighted average of all three sub-scores

### Social media links — Credibility rule
Within the Credibility bucket, add a social media presence check:
- Count distinct social platform profile links on the homepage (LinkedIn, Instagram, Facebook, X/Twitter, YouTube, TikTok, Pinterest)
- **3 or more links** = ✅ green — threshold met
- **Fewer than 3 links** = ❌ red — credibility failure
- Display the count and threshold result as a scored item inside the Credibility section
- If fewer than 3, this automatically generates a **P1 card** in the Fix List (see Change 3)

---

## Change 3 — Comprehensive Prioritised Fix List

Replace the existing "Prioritized Indexability Issues" list and "Topics to Optimize" pills with a single unified section: **"Your AEO Fix List"**

This list is a comprehensive reflection of **every issue detected across the full audit** — not just indexability issues. Every fix card traces to a specific detected finding.

### Card anatomy

```
┌──────────────────────────────────────────────────────┐
│  [BUCKET BADGE]          [PRIORITY BADGE]            │
│  Issue title                                         │
│  What this means for AI visibility (1–2 lines)       │
│  → Recommended action (specific, imperative)         │
└──────────────────────────────────────────────────────┘
```

**Bucket badge** — colour-matched to the bucket:
- 🔵 INDEXABILITY
- 🟡 CREDIBILITY
- 🟢 VISIBILITY

**Priority badge:**
- **P1** — Fix First (red)
- **P2** — High Impact (orange)
- **P3** — Authority Builder (slate)

**Sort order:** P1 first → P2 → P3. Within each priority: Indexability → Credibility → Visibility.

**Card styling:** 4px left border in bucket colour · white card background · subtle box-shadow.

**Source rule:** Every card must trace to a specific issue detected in the audit data. No generic advice cards. If a data point produced a finding → card. If it passed → no card.

---

## What Not to Change

- Brand Recommendation Matrix logic or display
- Competitive Landscape
- Top Recommended Brands section
- CTA footer
- Any existing data collection or scoring logic beyond the bucket reorganisation above
- Any copy outside of the two label changes in Change 1
