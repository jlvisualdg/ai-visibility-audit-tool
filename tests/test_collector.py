"""
Tests for src/collector.py — multi-engine × multi-topic collector
and the aggregate_results() summary in src/scoring.py.

Covers:
  1. execute_all returns 20 results with correct shape
  2. Brand mentions attached to each result
  3. aggregate_results computes all 5 summary keys correctly
  4. Edge case: all misses → ai_presence_pct=0, best_brand='None'
  5. Edge case: all hits on one brand → best_brand is that brand
  6. Engine failure handling
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers — build mock engine query results
# ---------------------------------------------------------------------------


def _make_result(topic, engine, text="", citations=None, error=None,
                 brand_mentions=None, positions=None, latency_ms=100):
    """Build a result dict matching the shape from collect.execute_all()."""
    return {
        "topic": topic,
        "engine": engine,
        "text": text,
        "citations": citations or [],
        "latency_ms": latency_ms,
        "error": error,
        "brand_mentions": brand_mentions or [],
        "positions": positions or [],
        "target_mention_count": len(positions or []),
    }


# ---------------------------------------------------------------------------
# Mock engine that returns canned responses
# ---------------------------------------------------------------------------


class FakeEngine:
    """A mock engine that returns a pre-configured result per topic."""

    def __init__(self, name, response_map=None):
        self.name = name
        self._response_map = response_map or {}

    def query(self, topic):
        """Return a canned result dict for the topic, or a default."""
        return self._response_map.get(topic, {
            "text": f"Mock response from {self.name} for '{topic}'",
            "citations": [],
            "engine": self.name,
            "latency_ms": 50,
            "error": None,
        })

    def is_real(self):
        return False


# ---------------------------------------------------------------------------
# Tests for execute_all()
# ---------------------------------------------------------------------------


class TestExecuteAll:
    """Tests for collector.execute_all() with mocked engines."""

    def test_returns_20_results_for_5_topics(self):
        """5 topics × 4 engines = 20 result dicts."""
        from src.collector import execute_all

        topics = ["topic A", "topic B", "topic C", "topic D", "topic E"]

        # Patch all four engine classes to return FakeEngine instances
        with patch("src.collector.PerplexityEngine") as mock_p, \
             patch("src.collector.ChatGPTEngine") as mock_c, \
             patch("src.collector.ClaudeEngine") as mock_cl, \
             patch("src.collector.GeminiEngine") as mock_g:

            for mock_cls, name in [
                (mock_p, "perplexity"),
                (mock_c, "chatgpt"),
                (mock_cl, "claude"),
                (mock_g, "gemini"),
            ]:
                fake = FakeEngine(name)
                mock_cls.return_value = fake

            results = execute_all(
                topics, "example.com", sleep_between=0.0
            )

        assert len(results) == 20, f"Expected 20 results, got {len(results)}"
        engines_seen = {r["engine"] for r in results}
        assert engines_seen == {"Perplexity", "ChatGPT", "Claude", "Gemini"}
        topics_seen = {r["topic"] for r in results}
        assert topics_seen == set(topics)

    def test_each_result_has_required_keys(self):
        """Every result dict has all 9 required keys."""
        from src.collector import execute_all

        topics = ["topic X", "topic Y"]

        with patch("src.collector.PerplexityEngine") as mock_p, \
             patch("src.collector.ChatGPTEngine") as mock_c, \
             patch("src.collector.ClaudeEngine") as mock_cl, \
             patch("src.collector.GeminiEngine") as mock_g:

            for mock_cls, name in [
                (mock_p, "perplexity"),
                (mock_c, "chatgpt"),
                (mock_cl, "claude"),
                (mock_g, "gemini"),
            ]:
                fake = FakeEngine(name)
                mock_cls.return_value = fake

            results = execute_all(
                topics, "example.com", sleep_between=0.0
            )

        required = {
            "topic", "engine", "text", "citations", "latency_ms",
            "error", "brand_mentions", "positions", "target_mention_count",
        }
        for r in results:
            assert set(r.keys()) == required, (
                f"Missing keys: {required - set(r.keys())}"
            )

    def test_brand_mentions_extracted_from_text(self):
        """Brand mentions are extracted when the mock response has brand text."""
        from src.collector import execute_all

        topics = ["best CRM"]

        brand_text = (
            "Acme Corp leads the market. Pareto Talent follows. "
            "Acme Corp mentioned again."
        )

        with patch("src.collector.PerplexityEngine") as mock_p, \
             patch("src.collector.ChatGPTEngine") as mock_c, \
             patch("src.collector.ClaudeEngine") as mock_cl, \
             patch("src.collector.GeminiEngine") as mock_g:

            for mock_cls, name in [
                (mock_p, "perplexity"),
                (mock_c, "chatgpt"),
                (mock_cl, "claude"),
                (mock_g, "gemini"),
            ]:
                fake = FakeEngine(name, {
                    "best CRM": {
                        "text": brand_text,
                        "citations": ["acme.com", "competitor.com"],
                        "engine": name,
                        "latency_ms": 100,
                        "error": None,
                    }
                })
                mock_cls.return_value = fake

            results = execute_all(
                topics, "paretotalent.com", sleep_between=0.0
            )

        # All 4 results should have extracted brands
        for r in results:
            assert "Acme Corp" in r["brand_mentions"], (
                f"Engine {r['engine']}: missing 'Acme Corp' in {r['brand_mentions']}"
            )

        # The first perplexity result should have Pareto Talent with position tracking
        first = results[0]
        assert "Pareto Talent" in first["brand_mentions"]
        # Pareto Talent is at position 2 (after Acme Corp)
        assert 2 in first["positions"], (
            f"Expected position 2 for Pareto Talent, got {first['positions']}"
        )
        assert first["target_mention_count"] >= 1

    def test_engine_failure_returns_error_result(self):
        """When an engine fails, the result has error populated, empty lists."""
        from src.collector import execute_all

        topics = ["topic X"]

        with patch("src.collector.PerplexityEngine") as mock_p, \
             patch("src.collector.ChatGPTEngine") as mock_c, \
             patch("src.collector.ClaudeEngine") as mock_cl, \
             patch("src.collector.GeminiEngine") as mock_g:

            # First engine works
            fake_p = FakeEngine("perplexity")
            mock_p.return_value = fake_p

            # Second engine raises during query
            fake_c = FakeEngine("chatgpt")
            fake_c.query = MagicMock(side_effect=RuntimeError("Boom"))
            mock_c.return_value = fake_c

            # Third engine works
            fake_cl = FakeEngine("claude")
            mock_cl.return_value = fake_cl

            # Fourth engine raises during instantiation
            mock_g._engine_name = "gemini"
            mock_g.side_effect = ValueError("No API key")

            results = execute_all(
                topics, "example.com", sleep_between=0.0
            )

        # Should still have exactly 4 results (one per engine)
        assert len(results) == 4

        # ChatGPT result should have error
        chatgpt_results = [r for r in results if r["engine"] == "ChatGPT"]
        assert len(chatgpt_results) == 1
        assert chatgpt_results[0]["error"] is not None
        assert chatgpt_results[0]["brand_mentions"] == []
        assert chatgpt_results[0]["positions"] == []

        # Gemini result should have error (instantiation failed)
        gemini_results = [r for r in results if r["engine"] == "Gemini"]
        assert len(gemini_results) == 1
        assert gemini_results[0]["error"] is not None
        assert "No API key" in gemini_results[0]["error"]
        assert gemini_results[0]["brand_mentions"] == []


# ---------------------------------------------------------------------------
# Tests for aggregate_results()
# ---------------------------------------------------------------------------


class TestAggregateResults:
    """Tests for scoring.aggregate_results()."""

    def test_computes_all_five_keys(self):
        """Returns dict with exactly the 5 required keys."""
        from src.scoring import aggregate_results

        results = [
            _make_result("topic A", "Perplexity",
                         brand_mentions=["Acme Corp", "Pareto Talent"],
                         positions=[2],
                         citations=["competitor.com"]),
            _make_result("topic A", "ChatGPT",
                         brand_mentions=["Global Tech", "Pareto Talent"],
                         positions=[2],
                         citations=[]),
            _make_result("topic B", "Perplexity",
                         brand_mentions=["Acme Corp", "Beta Inc"],
                         positions=[],
                         citations=["paretotalent.com"]),
            _make_result("topic B", "ChatGPT",
                         brand_mentions=["Beta Inc"],
                         positions=[],
                         citations=[]),
        ]

        agg = aggregate_results(results, "paretotalent.com")

        assert set(agg.keys()) == {
            "ai_presence_pct", "best_brand", "best_model",
            "citation_count", "best_topic",
        }
        assert isinstance(agg["ai_presence_pct"], float)
        assert isinstance(agg["best_brand"], str)
        assert isinstance(agg["best_model"], str)
        assert isinstance(agg["citation_count"], int)
        assert isinstance(agg["best_topic"], str)

    def test_ai_presence_pct_with_hits(self):
        """Target brand appearing yields non-zero AI presence."""
        from src.scoring import aggregate_results

        # Pareto Talent appears in 3 of 4 results at good positions
        results = [
            _make_result("t1", "A",
                         brand_mentions=["Pareto Talent", "Acme Corp"],
                         positions=[1]),  # position 1 → high score
            _make_result("t2", "B",
                         brand_mentions=["Acme Corp", "Pareto Talent"],
                         positions=[2]),  # position 2 → moderate score
            _make_result("t3", "C",
                         brand_mentions=["Global Tech"],  # no target
                         positions=[]),
            _make_result("t4", "D",
                         brand_mentions=["Pareto Talent", "Beta Inc"],
                         positions=[1]),  # position 1 → high score
        ]

        agg = aggregate_results(results, "paretotalent.com")

        # Score 1: (2-1+1)/2*100 = 100 * min(1,3)/3 = 33.33
        # Score 2: (2-2+1)/2*100 = 50 * 1/3 = 16.67
        # Score 3: 0 (no mention)
        # Score 4: (2-1+1)/2*100 = 100 * 1/3 = 33.33
        # Mean: (33.33 + 16.67 + 0 + 33.33) / 4 = 20.83
        assert agg["ai_presence_pct"] > 0
        assert agg["ai_presence_pct"] < 100

    def test_all_misses_ai_presence_zero(self):
        """When target brand never appears, ai_presence_pct is 0."""
        from src.scoring import aggregate_results

        results = [
            _make_result("t1", "A",
                         brand_mentions=["Acme Corp", "Global Tech"],
                         positions=[]),
            _make_result("t2", "B",
                         brand_mentions=["Beta Inc"],
                         positions=[]),
        ]

        agg = aggregate_results(results, "paretotalent.com")
        assert agg["ai_presence_pct"] == 0.0, (
            f"Expected 0.0, got {agg['ai_presence_pct']}"
        )

    def test_all_misses_no_brands_best_brand_none(self):
        """When zero brands are found across all results, best_brand is 'None'."""
        from src.scoring import aggregate_results

        results = [
            _make_result("t1", "A", brand_mentions=[], positions=[]),
            _make_result("t2", "B", brand_mentions=[], positions=[]),
        ]

        agg = aggregate_results(results, "paretotalent.com")
        assert agg["ai_presence_pct"] == 0.0
        assert agg["best_brand"] == "None", (
            f"Expected 'None', got '{agg['best_brand']}'"
        )

    def test_best_brand_excludes_target(self):
        """best_brand is the non-target brand with highest cumulative score."""
        from src.scoring import aggregate_results

        # Acme Corp appears first in many results → should win
        results = [
            _make_result("t1", "A",
                         brand_mentions=["Acme Corp", "Pareto Talent", "Beta Inc"],
                         positions=[2]),  # Pareto Talent at pos 2
            _make_result("t2", "B",
                         brand_mentions=["Acme Corp", "Global Tech"],
                         positions=[]),
            _make_result("t3", "C",
                         brand_mentions=["Acme Corp", "Beta Inc", "Pareto Talent"],
                         positions=[3]),
        ]

        agg = aggregate_results(results, "paretotalent.com")

        # Acme Corp: (3-1+1)/3*100 + (2-1+1)/2*100 + (3-1+1)/3*100
        # = 100 + 100 + 100 = 300
        # Beta Inc: (3-3+1)/3*100 + (3-2+1)/3*100 = 33.33 + 66.67 = 100
        # Global Tech: (2-2+1)/2*100 = 50
        assert agg["best_brand"] == "Acme Corp", (
            f"Expected 'Acme Corp', got '{agg['best_brand']}'"
        )

    def test_all_hits_on_one_brand(self):
        """When only one brand appears, it becomes best_brand."""
        from src.scoring import aggregate_results

        results = [
            _make_result("t1", "A",
                         brand_mentions=["Acme Corp"],
                         positions=[]),
            _make_result("t2", "B",
                         brand_mentions=["Acme Corp"],
                         positions=[]),
        ]

        agg = aggregate_results(results, "paretotalent.com")
        assert agg["best_brand"] == "Acme Corp"

    def test_best_model_computed_correctly(self):
        """Engine with highest mean AI presence wins."""
        from src.scoring import aggregate_results

        results = [
            # Perplexity: 2 hits at position 1
            _make_result("t1", "Perplexity",
                         brand_mentions=["Pareto Talent", "Acme Corp"],
                         positions=[1]),
            _make_result("t2", "Perplexity",
                         brand_mentions=["Pareto Talent"],
                         positions=[1]),
            # ChatGPT: 0 hits
            _make_result("t1", "ChatGPT",
                         brand_mentions=["Acme Corp"],
                         positions=[]),
            _make_result("t2", "ChatGPT",
                         brand_mentions=["Beta Inc"],
                         positions=[]),
        ]

        agg = aggregate_results(results, "paretotalent.com")
        assert agg["best_model"] == "Perplexity", (
            f"Expected 'Perplexity', got '{agg['best_model']}'"
        )

    def test_citation_count_counts_target_domain(self):
        """Citation count sums citations that contain the target domain."""
        from src.scoring import aggregate_results

        results = [
            _make_result("t1", "A",
                         brand_mentions=["Acme Corp"],
                         positions=[],
                         citations=["paretotalent.com", "competitor.com", "pareto.co"]),
            _make_result("t2", "B",
                         brand_mentions=["Beta Inc"],
                         positions=[],
                         citations=["other.com"]),
            _make_result("t3", "C",
                         brand_mentions=["Gamma Co"],
                         positions=[],
                         citations=["paretotalent.com", "blog.paretotalent.com"]),
        ]

        agg = aggregate_results(results, "paretotalent.com")

        # "paretotalent.com" in result 1, "pareto.co" doesn't match,
        # "paretotalent.com" and "blog.paretotalent.com" in result 3 → 3 total
        assert agg["citation_count"] == 3, (
            f"Expected 3, got {agg['citation_count']}"
        )

    def test_citation_count_with_empty_citations(self):
        """No citations yields citation_count=0."""
        from src.scoring import aggregate_results

        results = [
            _make_result("t1", "A", brand_mentions=[], citations=[]),
        ]

        agg = aggregate_results(results, "example.com")
        assert agg["citation_count"] == 0

    def test_best_topic_computed_correctly(self):
        """Topic with highest mean AI presence wins."""
        from src.scoring import aggregate_results

        results = [
            # topic "buyer intent" scores high
            _make_result("buyer intent", "A",
                         brand_mentions=["Pareto Talent", "Acme Corp"],
                         positions=[1]),
            _make_result("buyer intent", "B",
                         brand_mentions=["Pareto Talent"],
                         positions=[1]),
            # topic "pricing" scores low
            _make_result("pricing", "A",
                         brand_mentions=["Acme Corp"],
                         positions=[]),
            _make_result("pricing", "B",
                         brand_mentions=["Beta Inc"],
                         positions=[]),
        ]

        agg = aggregate_results(results, "paretotalent.com")
        assert agg["best_topic"] == "buyer intent", (
            f"Expected 'buyer intent', got '{agg['best_topic']}'"
        )

    def test_empty_results_returns_sensible_defaults(self):
        """Empty results list returns all-zero/sentinel defaults."""
        from src.scoring import aggregate_results

        agg = aggregate_results([], "example.com")
        assert agg == {
            "ai_presence_pct": 0.0,
            "best_brand": "None",
            "best_model": "",
            "citation_count": 0,
            "best_topic": "",
        }

    def test_results_with_errors_handled_gracefully(self):
        """Results where engines errored out don't break aggregation."""
        from src.scoring import aggregate_results

        results = [
            _make_result("t1", "A",
                         brand_mentions=["Pareto Talent"],
                         positions=[1]),
            _make_result("t2", "B",
                         brand_mentions=[], positions=[], error="Timeout",
                         text="", citations=[]),
        ]

        agg = aggregate_results(results, "paretotalent.com")

        # Should still compute without crashing
        assert agg["ai_presence_pct"] >= 0
        assert agg["best_model"] == "A"  # only engine with a score
        assert agg["citation_count"] == 0


# ---------------------------------------------------------------------------
# Test: aggregate_results uses the 5 specified keys exactly
# ---------------------------------------------------------------------------


def test_aggregate_results_returns_exact_five_keys():
    """The returned dict has exactly the 5 documented keys."""
    from src.scoring import aggregate_results

    results = [
        _make_result("t1", "Perplexity",
                     brand_mentions=["Acme Corp", "Pareto Talent"],
                     positions=[2]),
    ]

    agg = aggregate_results(results, "paretotalent.com")

    required_keys = {
        "ai_presence_pct", "best_brand", "best_model",
        "citation_count", "best_topic",
    }
    assert set(agg.keys()) == required_keys, (
        f"Expected keys {required_keys}, got {set(agg.keys())}"
    )
    assert len(agg) == 5, f"Expected exactly 5 keys, got {len(agg)}"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    for name, func in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            func()
            passed += 1
            print(f"  \u2713 {name}")
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  \u2717 {name}: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, f"ERROR: {e}"))
            print(f"  \u2717 {name}: ERROR: {e}")

    print()
    print(f"{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    if errors:
        print(f"\nFailures:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)