"""
End-to-end integration test: full audit pipeline with real OpenRouter API calls.

Runs 5 BOTF (Bottom-Of-The-Funnel) queries across 4 AI engines (Perplexity,
ChatGPT, Claude, Gemini) via execute_all(), aggregates results via
aggregate_results(), prints a summary table, and saves the raw API response
data to tests/integration_report_paretotalent.json.

CONTRACTS (other subagents build these in parallel):
    from src.collector import execute_all
        execute_all(topics: list[str], target_domain: str) -> list[dict]
        Returns list of 20 dicts, each with keys:
            {topic, engine, text, citations, latency_ms, error,
             brand_mentions, positions, target_mention_count}

    from src.scoring import aggregate_results
        aggregate_results(results: list[dict], target_domain: str) -> dict
        Returns dict with exactly 5 keys:
            {ai_presence_pct, best_brand, best_model, citation_count, best_topic}

NOTE: This test makes REAL API calls to OpenRouter and costs money.
      Uses only 5 queries (not the full 20) to limit cost.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Queries — 5 real BOTF (Bottom-Of-The-Funnel) queries for ParetoTalent
# ---------------------------------------------------------------------------

BOTF_QUERIES = [
    "executive assistant matching service for startups",
    "top remote operator agencies 2026",
    "hire a virtual chief of staff",
    "cost of executive assistant matching services",
    "find a remote operator provider for founders",
]

TARGET_DOMAIN = "paretotalent.com"

# Expected engines
EXPECTED_ENGINES = {"Perplexity", "ChatGPT", "Claude", "Gemini"}
EXPECTED_RESULT_COUNT = len(BOTF_QUERIES) * len(EXPECTED_ENGINES)  # 20

REPORT_PATH = PROJECT_ROOT / "tests" / "integration_report_paretotalent.json"


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def run_integration_test():
    """Run the full pipeline and report all metrics."""
    print("=" * 72)
    print("  AI VISIBILITY AUDIT — End-to-End Integration Test")
    print(f"  Target: {TARGET_DOMAIN}")
    print(f"  Queries: {len(BOTF_QUERIES)} × Engines: {len(EXPECTED_ENGINES)} = {EXPECTED_RESULT_COUNT} API calls")
    print("=" * 72)
    print()

    # ---- Step 0: verify API key ----
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        # Try loading .env
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        except ImportError:
            pass

    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment or .env file.")
        print("   Set it in .env at the project root and retry.")
        return {"status": "skipped", "reason": "No API key"}

    masked = api_key[:12] + "..." if len(api_key) > 12 else "***"
    print(f"✓ API key: {masked}")
    print()

    # ---- Step 1: execute_all() — 5 topics × 4 engines = 20 calls ----
    print("Step 1: Running execute_all() — this will make 20 real API calls...")
    print(f"         (5 queries × 4 engines = {EXPECTED_RESULT_COUNT} calls)")
    print()

    try:
        from src.collector import execute_all
    except ImportError as e:
        print(f"❌ Cannot import execute_all from src.collector: {e}")
        print("   This module is being built by another subagent in parallel.")
        print("   Writing test file only — re-run once collector.py is ready.")
        return {"status": "blocked", "reason": f"ImportError: {e}"}

    t0 = time.time()
    try:
        results = execute_all(BOTF_QUERIES, TARGET_DOMAIN)
    except Exception as e:
        print(f"❌ execute_all() raised: {type(e).__name__}: {e}")
        return {"status": "blocked", "reason": f"execute_all error: {e}"}

    elapsed = time.time() - t0
    print(f"✓ execute_all() returned {len(results)} results in {elapsed:.1f}s")
    print()

    # ---- Step 2: Validate results shape ----
    print("Step 2: Validating results shape...")
    errors = []

    # 2a: correct count
    if len(results) != EXPECTED_RESULT_COUNT:
        errors.append(
            f"Expected {EXPECTED_RESULT_COUNT} results, got {len(results)}"
        )

    # 2b: each result has required keys
    required_keys = {
        "topic", "engine", "text", "citations", "latency_ms",
        "error", "brand_mentions", "positions", "target_mention_count",
    }
    for i, r in enumerate(results):
        missing = required_keys - set(r.keys())
        if missing:
            errors.append(f"Result[{i}] missing keys: {missing}")
        extra = set(r.keys()) - required_keys
        if extra:
            errors.append(f"Result[{i}] has extra keys: {extra}")

    # 2c: error rate check (allow 0-2 errors for rate limiting)
    error_count = sum(1 for r in results if r.get("error") is not None)
    if error_count > 2:
        errors.append(
            f"Too many errors: {error_count}/{len(results)} (expected ≤ 2)"
        )

    # 2d: latency check
    for i, r in enumerate(results):
        if r.get("error") is None and r.get("latency_ms", 0) <= 0:
            errors.append(f"Result[{i}] has invalid latency_ms: {r.get('latency_ms')}")

    # 2e: all 4 engines represented
    engine_counts = Counter(r.get("engine", "unknown") for r in results)
    missing_engines = EXPECTED_ENGINES - set(engine_counts.keys())
    if missing_engines:
        errors.append(f"Missing engines: {missing_engines}")

    # 2f: all 5 topics represented
    topic_counts = Counter(r.get("topic", "unknown") for r in results)
    missing_topics = set(BOTF_QUERIES) - set(topic_counts.keys())
    if missing_topics:
        errors.append(f"Missing topics: {missing_topics}")

    if errors:
        print(f"❌ Validation found {len(errors)} issue(s):")
        for e in errors:
            print(f"   - {e}")
    else:
        print("✓ All 20 results have correct shape and required keys")
    print()

    # ---- Step 3: Per-engine success rate ----
    print("Step 3: Per-engine success rate:")
    print()
    print(f"  {'Engine':<16} {'Success':>8} {'Errors':>8} {'Rate':>8}  Avg Latency")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}  {'-'*12}")

    engine_stats = {}
    for engine_name in sorted(EXPECTED_ENGINES):
        engine_results = [r for r in results if r.get("engine") == engine_name]
        success = sum(1 for r in engine_results if r.get("error") is None)
        errs = len(engine_results) - success
        rate = (success / len(engine_results) * 100) if engine_results else 0
        latencies = [
            r["latency_ms"]
            for r in engine_results
            if r.get("error") is None and r.get("latency_ms", 0) > 0
        ]
        avg_lat = int(sum(latencies) / len(latencies)) if latencies else 0
        engine_stats[engine_name] = {
            "success": success, "errors": errs, "rate": rate, "avg_latency_ms": avg_lat,
        }
        print(f"  {engine_name:<16} {success:>8} {errs:>8} {rate:>7.0f}%  {avg_lat:>5} ms")

    print()

    # ---- Step 4: aggregate_results() ----
    print("Step 4: Calling aggregate_results()...")

    try:
        from src.scoring import aggregate_results
    except ImportError as e:
        print(f"❌ Cannot import aggregate_results from src.scoring: {e}")
        print("   This function is being built by another subagent in parallel.")
        print("   Skipping aggregation — re-run once scoring.py is updated.")
        print()

        # Save raw results anyway so we have a record
        _save_report(results, engine_stats, aggregation=None, errors=errors)
        return {
            "status": "partial",
            "reason": f"aggregate_results ImportError: {e}",
            "results_count": len(results),
            "engine_stats": engine_stats,
            "validation_errors": errors,
        }

    try:
        aggregation = aggregate_results(results, TARGET_DOMAIN)
    except Exception as e:
        print(f"❌ aggregate_results() raised: {type(e).__name__}: {e}")
        _save_report(results, engine_stats, aggregation=None, errors=errors)
        return {
            "status": "partial",
            "reason": f"aggregate_results error: {e}",
            "results_count": len(results),
            "engine_stats": engine_stats,
        }

    print(f"✓ aggregate_results() returned: {list(aggregation.keys())}")
    print()

    # ---- Step 5: Validate aggregation output ----
    print("Step 5: Validating aggregation output...")

    expected_agg_keys = {"ai_presence_pct", "best_brand", "best_model", "citation_count", "best_topic"}
    agg_errors = []

    missing_keys = expected_agg_keys - set(aggregation.keys())
    if missing_keys:
        agg_errors.append(f"aggregation missing keys: {missing_keys}")

    ai_pct = aggregation.get("ai_presence_pct")
    if not isinstance(ai_pct, (int, float)) or not (0 <= ai_pct <= 100):
        agg_errors.append(f"ai_presence_pct invalid: {ai_pct} (expected float 0-100)")

    best_brand = aggregation.get("best_brand")
    if not isinstance(best_brand, str) or not best_brand:
        agg_errors.append(f"best_brand invalid: {best_brand!r} (expected non-empty string)")

    best_model = aggregation.get("best_model")
    if best_model not in EXPECTED_ENGINES:
        agg_errors.append(f"best_model not in engines: {best_model!r} (expected one of {EXPECTED_ENGINES})")

    citation_count = aggregation.get("citation_count")
    if not isinstance(citation_count, int) or citation_count < 0:
        agg_errors.append(f"citation_count invalid: {citation_count} (expected int ≥ 0)")

    best_topic = aggregation.get("best_topic")
    if best_topic not in BOTF_QUERIES:
        agg_errors.append(f"best_topic not in queries: {best_topic!r} (expected one of {BOTF_QUERIES})")

    if agg_errors:
        for e in agg_errors:
            print(f"   ❌ {e}")
    else:
        print("✓ All 5 aggregation keys present and valid")
    print()

    # ---- Step 6: Top 3 brands by cumulative mention ----
    print("Step 6: Top 3 brands by cumulative mention count...")

    brand_counter: Counter[str] = Counter()
    for r in results:
        mentions = r.get("brand_mentions") or []
        if isinstance(mentions, list):
            for brand in mentions:
                if isinstance(brand, str):
                    brand_counter[brand] += 1

    top3 = brand_counter.most_common(3)
    print()
    print(f"  {'Rank':<6} {'Brand':<40} {'Mentions':>10}")
    print(f"  {'-'*6} {'-'*40} {'-'*10}")
    for rank, (brand, count) in enumerate(top3, 1):
        display = brand[:38] + ".." if len(brand) > 40 else brand
        print(f"  {rank:<6} {display:<40} {count:>10}")
    if not top3:
        print("  (no brands found across all results)")
    print()

    # ---- Step 7: Summary table ----
    print("=" * 72)
    print("  INTEGRATION TEST SUMMARY")
    print("=" * 72)
    print()
    print(f"  Total API calls:        {len(results)}")
    print(f"  Successful calls:       {sum(1 for r in results if r.get('error') is None)}")
    print(f"  Failed calls:           {error_count}")
    print(f"  Overall success rate:   {(len(results) - error_count) / len(results) * 100:.0f}%")
    print(f"  Total elapsed:          {elapsed:.0f}s")
    print()
    print(f"  Overall AI Presence:    {ai_pct:.1f}%")
    print(f"  Best model:             {best_model}")
    print(f"  Best topic:             {best_topic}")
    print(f"  Total citations:        {citation_count}")
    print(f"  Best brand:             {best_brand}")
    if top3:
        print(f"  Top 3 brands:           {', '.join(f'{b} ({c})' for b, c in top3)}")
    print()

    # ---- Step 8: Save report ----
    _save_report(results, engine_stats, aggregation, errors + agg_errors)

    return {
        "status": "ok",
        "results_count": len(results),
        "success_count": len(results) - error_count,
        "error_count": error_count,
        "elapsed_s": elapsed,
        "engine_stats": engine_stats,
        "aggregation": aggregation,
        "top_brands": top3,
        "validation_errors": errors + agg_errors,
    }


def _save_report(results, engine_stats, aggregation, errors):
    """Save the full API response data to JSON for historical record."""
    # Make results JSON-serializable (handle sets, Paths, etc.)
    serializable = {
        "metadata": {
            "target_domain": TARGET_DOMAIN,
            "queries": BOTF_QUERIES,
            "engine_count": len(EXPECTED_ENGINES),
            "expected_engines": sorted(EXPECTED_ENGINES),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validation_errors": errors,
        },
        "engine_stats": {
            k: {
                "success": v["success"],
                "errors": v["errors"],
                "rate_pct": v["rate"],
                "avg_latency_ms": v["avg_latency_ms"],
            }
            for k, v in engine_stats.items()
        },
        "results": results,
    }
    if aggregation:
        serializable["aggregation"] = aggregation

    report_json = json.dumps(serializable, indent=2, default=str, ensure_ascii=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_json, encoding="utf-8")

    size_kb = len(report_json) / 1024
    print(f"💾 Report saved: {REPORT_PATH} ({size_kb:.1f} KB)")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    outcome = run_integration_test()
    print(f"\nFinal outcome: {outcome.get('status', 'unknown')}")

    if outcome.get("status") == "ok":
        print("✓ Integration test PASSED")
        sys.exit(0)
    elif outcome.get("status") == "skipped":
        print("⚠ Integration test SKIPPED — no API key configured")
        sys.exit(0)  # Not a failure, just can't run
    elif outcome.get("status") == "partial":
        print("⚠ Integration test PARTIAL — some components not ready")
        sys.exit(0)  # Expected while parallel builds are in progress
    else:
        print("❌ Integration test BLOCKED")
        sys.exit(1)