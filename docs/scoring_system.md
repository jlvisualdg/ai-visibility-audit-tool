# AI Visibility Audit — Scoring System

## Overview

This document defines the complete scoring methodology for the AI Visibility
Audit Tool.  Every score is derived from **live data** — real AI engine
responses are parsed for brand mentions, and positions are extracted
deterministically.  There is no AI-summarisation step in the scoring pipeline:
it's pure arithmetic on extracted facts.

---

## Core Metric: AI Presence Score (0–100 %)

### Per-Query Formula

```
Position_Score = (Total_Brands − Your_Position + 1) ÷ Total_Brands × 100
Mention_Multiplier = min(Mention_Count, MAX_MENTION_MULTIPLIER) ÷ MAX_MENTION_MULTIPLIER
Query_Score = min(Position_Score × Mention_Multiplier, 100)
```

| Term | Meaning |
|---|---|
| `Total_Brands` | Number of distinct brands identified in the engine's response |
| `Your_Position` | 1-indexed rank of the target brand (1 = first brand named) |
| `Mention_Count` | How many times the target brand appears in the response text |
| `MAX_MENTION_MULTIPLIER` | Hard cap on the mention multiplier (default **3**) |

### Position Weighting Rationale

AI engines tend to list the most authoritative / relevant sources **first**.
Users scanning an AI response trust earlier recommendations more, and
click-through rates are highest for the first-cited brand.  The linear
decay weighting reflects this:

| Position | Formula | Score |
|---|---|---|
| 1st of 4 | (4 − 1 + 1) ÷ 4 | **100.00 %** |
| 2nd of 4 | (4 − 2 + 1) ÷ 4 | **75.00 %** |
| 3rd of 4 | (4 − 3 + 1) ÷ 4 | **50.00 %** |
| 4th of 4 | (4 − 4 + 1) ÷ 4 | **25.00 %** |

General case — position *N* of *T* brands:

```
Position_N_of_T = (T − N + 1) ÷ T × 100
```

### Mention Multiplier

Multiple mentions signal a **stronger recommendation** — the engine keeps
returning to your brand as it elaborates on the topic.  However, the
multiplier is capped at 3× so that a brand cannot inflate its score by
stuffing its name into every sentence.

| Mentions | Multiplier (cap = 3) | Effect |
|---|---|---|
| 0 | 0.00 | No credit |
| 1 | 0.33 | One-third credit |
| 2 | 0.67 | Two-thirds credit |
| 3+ | 1.00 | Full credit |

The formula: `min(Mention_Count, 3) ÷ 3`

This means **a single mention at position 1 is worth the same as three
mentions at position 2** (both ≈ 33.33 % on a query with 2+ brands).

### Final Query Score Cap

```
Query_Score = min(Position_Score × Mention_Multiplier, 100)
```

The cap at 100 prevents any single query from exceeding a perfect score.
In practice the cap only binds when:

- The brand is **1st of 1** (monopoly mention) **and** has 3+ mentions.
- The brand is **1st of 2** **and** has 2 mentions at position 1.

---

## Aggregation: Overall AI Presence

After calculating a `Query_Score` for every (topic × engine) combination,
the **Overall AI Presence** is the arithmetic mean:

```
Overall_AI_Presence = mean(all Query_Scores)
```

**Example pipeline:**

| Topic | Engine | Score |
|---|---|---|
| "best x for y" | Perplexity | 33.33 |
| "best x for y" | ChatGPT | 50.00 |
| "top z agencies" | Perplexity | 66.67 |
| "top z agencies" | ChatGPT | 0.00 |

Overall = (33.33 + 50.00 + 66.67 + 0.00) ÷ 4 = **37.50 %**

---

## Worked Examples

> All examples use `MAX_MENTION_MULTIPLIER = 3`.

### Example 1 — Best Case (1st with 3+ mentions)

```
Mentions = 3, Position = 1st, Total Brands = 4
Position_Score   = (4 − 1 + 1) ÷ 4 × 100 = 100.00 %
Mention_Multiplier = min(3, 3) ÷ 3 = 1.00
Query_Score      = 100.00 × 1.00 = 100.00 %
```

### Example 2 — Middle Mention (3rd of 4, 2 mentions)

```
Mentions = 2, Position = 3rd, Total Brands = 4
Position_Score   = (4 − 3 + 1) ÷ 4 × 100 = 50.00 %
Mention_Multiplier = min(2, 3) ÷ 3 = 0.6667
Query_Score      = 50.00 × 0.6667 = 33.33 %
```

### Example 3 — First with Single Mention

```
Mentions = 1, Position = 1st, Total Brands = 5
Position_Score   = (5 − 1 + 1) ÷ 5 × 100 = 100.00 %
Mention_Multiplier = min(1, 3) ÷ 3 = 0.3333
Query_Score      = 100.00 × 0.3333 = 33.33 %
```

### Example 4 — Last Mention

```
Mentions = 1, Position = 4th, Total Brands = 4
Position_Score   = (4 − 4 + 1) ÷ 4 × 100 = 25.00 %
Mention_Multiplier = min(1, 3) ÷ 3 = 0.3333
Query_Score      = 25.00 × 0.3333 = 8.33 %
```

### Example 5 — Brand Not Found

```
Mentions = 0, Position = None, Total Brands = 5
Query_Score      = 0.00  (brand absent)
```

---

## Adjustment Guide

### MAX_MENTION_MULTIPLIER

| File | Constant |
|---|---|
| `src/scoring.py` | `MAX_MENTION_MULTIPLIER = 3` |
| `src/scoring.py` → `calculate_ai_presence_score()` | keyword arg `max_mention_multiplier` |

**What it does:** Caps the number of mentions that contribute to the
multiplier.  With the default of 3, the multiplier reaches 1.0 at 3
mentions.  A mention count of 1 yields 0.33×, 2 yields 0.67×, 3+ yields
1.0×.

**When to change it:**

| New value | Effect |
|---|---|
| `1` | Any mention gets full credit (no penalty for single mentions).  Makes the system more generous. |
| `5` | Requires 5 mentions for full credit.  Useful when responses are very long and multiple mentions are common (e.g., academic papers). |
| `10` | Very strict — mentions must be extremely frequent.  Rarely useful outside highly specialised domains. |

**How to change it call-by-call** (without editing the constant):

```python
from src.scoring import calculate_ai_presence_score

score = calculate_ai_presence_score(
    mention_count=2,
    first_position=1,
    total_brands=5,
    max_mention_multiplier=5,   # override
)
```

**Warning:** If you change the constant globally, re-run the validation
suite (`python tests/validate_scoring.py`) and update the expected values
in this document.

### MIN_BRAND_LENGTH

| File | Constant |
|---|---|
| `src/scoring.py` | `MIN_BRAND_LENGTH = 3` |

**What it does:** Filters out candidate brand-name strings shorter than
this many characters during brand extraction from raw response text.
Prevents noise tokens like "AI", "Co", or "Inc" from being counted as
separate brands.

**When to change it:**

| New value | Effect |
|---|---|
| `1` | No length filter — every capitalised token is a potential brand.  Very noisy. |
| `5` | Aggressive filter — brands with short names (e.g. "Uber", "Zap") may be missed. |
| `10` | Extremely aggressive — only multi-word company names will pass.  Not recommended. |

---

## Data Pipeline

```
                      ┌──────────────────┐
                      │  5 BOTF queries  │  (generated by src/topicgen.py)
                      └────────┬─────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌───────────┐    ┌───────────┐    ┌───────────┐
      │Perplexity │    │ ChatGPT   │    │ Claude    │  ←  also Gemini
      │ (20 resp) │    │ (20 resp) │    │ (20 resp) │      = 4 engines
      └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Extract brands  │  src/scoring.py
                    │ + positions     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Per-query score │  calculate_ai_presence_score()
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Overall mean    │  calculate_overall_ai_presence()
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  HTML report    │  src/reporter.py
                    └─────────────────┘
```

---

## Validation

Run the validation suite:

```bash
python tests/validate_scoring.py
```

This exercises all five worked examples from this document with a
tolerance of **0.01** (i.e., results must match to within one one-hundredth
of a percentage point).  If you change `MAX_MENTION_MULTIPLIER` you must
update both this document and the test expectations.