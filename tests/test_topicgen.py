"""
Tests for src/topicgen.py — AI-powered buyer-intent topic generation.

Covers:
  1. Fallback when no API key provided
  2. Fallback when no homepage HTML available
  3. Basic LLM topic generation (mocked requests)
  4. Keyword-variation post-processing (positions [0] and [3] replaced)
  5. Graceful degradation when keyword extraction fails
  6. Graceful degradation when variation generation fails
  7. Output always has correct length
"""

from __future__ import annotations

import json
from unittest.mock import patch, Mock

import pytest


# ── Helpers ──


def _mock_openrouter_response(content: str) -> dict:
    """Build a realistic OpenRouter JSON response."""
    return {
        "id": "gen-test-123",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


SAMPLE_HOMEPAGE_HTML = """
<html>
<head><title>Pareto Talent — Executive Assistant Matching</title>
<meta name="description" content="We match founders with top-tier executive assistants.">
</head>
<body>
<h1>Virtual Executive Assistant Matching for Founders</h1>
<p>Pareto Talent connects busy founders with pre-vetted, full-time remote executive assistants.
Our matching platform uses data-driven algorithms to pair you with the perfect EA.</p>
</body>
</html>
"""

INITIAL_TOPICS_JSON = json.dumps([
    "best executive assistant matching service for startup founders",
    "hire a virtual executive assistant for growing business",
    "top executive assistant staffing agencies 2026",
    "remote executive assistant service pricing and reviews",
    "find pre-vetted executive assistant for founders",
])

KEYWORD_RESPONSE = "executive assistant matching"

VARIATION_TOPICS_JSON = json.dumps([
    "best executive assistant matching platform for SaaS founders",
    "hire an executive assistant matching agency for startups 2026",
])


# ── Fallback tests ──


class TestFallback:
    """Tests for fallback behavior when LLM is unavailable."""

    def test_no_api_key_returns_fallback(self):
        """When no API key is provided, fallback topics are returned."""
        from src.topicgen import generate_botf_topics

        with patch("src.topicgen.OPENROUTER_API_KEY", ""):
            topics = generate_botf_topics("paretotalent.com", api_key="")
        assert len(topics) == 4  # fallback returns up to 4
        assert all(isinstance(t, str) for t in topics)
        assert any("pareto" in t.lower() for t in topics)

    def test_no_homepage_returns_fallback(self):
        """When scraping fails and no HTML provided, fallback is returned."""
        from src.topicgen import generate_botf_topics

        with patch("src.topicgen.requests.get") as mock_get:
            mock_get.side_effect = __import__("requests").RequestException("Network error")
            topics = generate_botf_topics("paretotalent.com", api_key="sk-test")

        assert len(topics) == 4
        assert any("pareto" in t.lower() for t in topics)

    def test_fallback_domain_parsing(self):
        """Fallback strips TLD and title-cases the domain."""
        from src.topicgen import _fallback_topics

        topics = _fallback_topics("my-company.io", 4)
        assert topics[0] == "best my company alternatives"
        assert topics[1] == "My Company reviews and pricing"


# ── Basic LLM generation ──


class TestBasicGeneration:
    """Tests for the core LLM topic generation path."""

    def test_returns_5_topics_from_llm(self):
        """When LLM succeeds, 5 topics are returned."""
        from src.topicgen import generate_botf_topics

        with patch("src.topicgen.requests.get") as mock_get:
            mock_get.return_value = Mock(ok=True, text=SAMPLE_HOMEPAGE_HTML)

            with patch("src.topicgen.requests.post") as mock_post:
                mock_post.return_value = Mock(ok=True)
                mock_post.return_value.json.return_value = _mock_openrouter_response(
                    INITIAL_TOPICS_JSON
                )

                topics = generate_botf_topics(
                    "paretotalent.com",
                    homepage_html=SAMPLE_HOMEPAGE_HTML,
                    api_key="sk-test",
                )

        assert len(topics) == 5
        assert all(isinstance(t, str) for t in topics)
        assert all(len(t) > 0 for t in topics)

    def test_all_topics_are_strings(self):
        """Every returned topic is a non-empty string."""
        from src.topicgen import generate_botf_topics

        with patch("src.topicgen.requests.post") as mock_post:
            mock_post.return_value = Mock(ok=True)
            mock_post.return_value.json.return_value = _mock_openrouter_response(
                INITIAL_TOPICS_JSON
            )

            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        for topic in topics:
            assert isinstance(topic, str)
            assert topic.strip() != ""

    def test_llm_failure_falls_back(self):
        """When the initial LLM call fails, fallback is returned."""
        from src.topicgen import generate_botf_topics

        with patch("src.topicgen.requests.post") as mock_post:
            mock_post.side_effect = __import__("requests").RequestException("Timeout")

            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        assert len(topics) == 4  # fallback
        assert any("pareto" in t.lower() for t in topics)

    def test_llm_http_error_falls_back(self):
        """When the LLM returns a non-200, fallback is returned."""
        from src.topicgen import generate_botf_topics

        with patch("src.topicgen.requests.post") as mock_post:
            mock_post.return_value = Mock(ok=False, status_code=500)

            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        assert len(topics) == 4  # fallback


# ── Keyword variation post-processing ──


class TestKeywordVariations:
    """Tests for the keyword-variation injection into positions [0] and [3]."""

    def _setup_triple_mock(self, topics_json, kw_response, var_json):
        """
        Set up 3 sequential mocked POST responses:
          1. Initial topic generation
          2. Keyword extraction
          3. Variation generation
        Returns the mock_post that can be used for assertions.
        """
        mock_post = Mock()
        # We need different responses for each call
        responses = [
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(topics_json))),
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(kw_response))),
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(var_json))),
        ]
        mock_post.side_effect = responses
        return mock_post, responses

    def test_positions_0_and_3_are_variations(self):
        """Positions [0] and [3] contain keyword-variation queries."""
        from src.topicgen import generate_botf_topics

        mock_post, _ = self._setup_triple_mock(
            INITIAL_TOPICS_JSON,
            KEYWORD_RESPONSE,
            VARIATION_TOPICS_JSON,
        )

        with patch("src.topicgen.requests.post", mock_post):
            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        assert len(topics) == 5
        assert topics[0] == "best executive assistant matching platform for SaaS founders"
        assert topics[3] == "hire an executive assistant matching agency for startups 2026"
        # Positions [1], [2], [4] keep original values
        assert topics[1] == "hire a virtual executive assistant for growing business"
        assert topics[2] == "top executive assistant staffing agencies 2026"
        assert topics[4] == "find pre-vetted executive assistant for founders"

    def test_variations_are_unbranded(self):
        """Keyword-variation queries do not contain the domain brand."""
        from src.topicgen import generate_botf_topics

        mock_post, _ = self._setup_triple_mock(
            INITIAL_TOPICS_JSON,
            KEYWORD_RESPONSE,
            VARIATION_TOPICS_JSON,
        )

        with patch("src.topicgen.requests.post", mock_post):
            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        for topic in topics:
            assert "pareto" not in topic.lower(), f"Brand name found in: {topic}"

    def test_keyword_extraction_failure_preserves_original_topics(self):
        """When keyword extraction fails, original 5 topics are preserved as-is."""
        from src.topicgen import generate_botf_topics

        mock_post = Mock()
        # First call succeeds (initial topics), second call fails (keyword extraction)
        responses = [
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(INITIAL_TOPICS_JSON))),
            Mock(ok=False, status_code=500),  # keyword extraction fails
        ]
        mock_post.side_effect = responses

        with patch("src.topicgen.requests.post", mock_post):
            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        assert len(topics) == 5
        assert topics[0] == "best executive assistant matching service for startup founders"
        assert topics[3] == "remote executive assistant service pricing and reviews"

    def test_variation_generation_failure_preserves_original_topics(self):
        """When variation generation fails, original 5 topics are preserved."""
        from src.topicgen import generate_botf_topics

        mock_post = Mock()
        responses = [
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(INITIAL_TOPICS_JSON))),
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(KEYWORD_RESPONSE))),
            Mock(ok=False, status_code=500),  # variation generation fails
        ]
        mock_post.side_effect = responses

        with patch("src.topicgen.requests.post", mock_post):
            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        assert len(topics) == 5
        assert topics[0] == "best executive assistant matching service for startup founders"
        assert topics[3] == "remote executive assistant service pricing and reviews"

    def test_keyword_extraction_exception_preserves_original(self):
        """When keyword extraction raises an exception, original topics preserved."""
        from src.topicgen import generate_botf_topics

        mock_post = Mock()
        responses = [
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(INITIAL_TOPICS_JSON))),
            __import__("requests").RequestException("Connection reset"),  # exception
        ]
        mock_post.side_effect = responses

        with patch("src.topicgen.requests.post", mock_post):
            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        assert len(topics) == 5

    def test_variation_single_returns_original(self):
        """When variation generation returns fewer than 2 queries, original preserved."""
        from src.topicgen import generate_botf_topics

        mock_post = Mock()
        responses = [
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(INITIAL_TOPICS_JSON))),
            Mock(ok=True, json=Mock(return_value=_mock_openrouter_response(KEYWORD_RESPONSE))),
            Mock(
                ok=True,
                json=Mock(
                    return_value=_mock_openrouter_response(
                        json.dumps(["only one query here"])
                    )
                ),
            ),
        ]
        mock_post.side_effect = responses

        with patch("src.topicgen.requests.post", mock_post):
            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        assert len(topics) == 5
        # Original topics preserved because we only got 1 variation
        assert topics[0] == "best executive assistant matching service for startup founders"


# ── Edge cases / invariants ──


class TestInvariants:
    """Edge case and invariant tests."""

    def test_scraped_htmL_works_with_keyword_variations(self):
        """When homepage is scraped (not provided), keyword variations still work."""
        from src.topicgen import generate_botf_topics

        with patch("src.topicgen.requests.get") as mock_get:
            mock_get.return_value = Mock(ok=True, text=SAMPLE_HOMEPAGE_HTML)

            mock_post, _ = TestKeywordVariations()._setup_triple_mock(
                INITIAL_TOPICS_JSON,
                KEYWORD_RESPONSE,
                VARIATION_TOPICS_JSON,
            )

            with patch("src.topicgen.requests.post", mock_post):
                topics = generate_botf_topics(
                    "paretotalent.com",
                    api_key="sk-test",
                )

            assert len(topics) == 5
            assert topics[0] == "best executive assistant matching platform for SaaS founders"
            assert topics[3] == "hire an executive assistant matching agency for startups 2026"

    def test_module_imports_cleanly(self):
        """The topicgen module imports without errors."""
        import src.topicgen  # noqa: F401

    def test_fallback_func_is_callable(self):
        """_fallback_topics is importable and callable."""
        from src.topicgen import _fallback_topics

        result = _fallback_topics("example.com", 3)
        assert len(result) == 3
        assert all(isinstance(t, str) for t in result)

    def test_inject_keyword_variations_is_callable(self):
        """_inject_keyword_variations is importable."""
        from src.topicgen import _inject_keyword_variations

        assert callable(_inject_keyword_variations)

    def test_n_parameter_respected(self):
        """The `n` parameter limits output count."""
        from src.topicgen import generate_botf_topics

        with patch("src.topicgen.requests.post") as mock_post:
            mock_post.return_value = Mock(ok=True)
            mock_post.return_value.json.return_value = _mock_openrouter_response(
                INITIAL_TOPICS_JSON
            )

            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
                n=3,
            )

        assert len(topics) == 3  # limited by n

    def test_no_brand_names_in_output(self):
        """None of the generated topics contain the brand name (Pareto)."""
        from src.topicgen import generate_botf_topics

        mock_post, _ = TestKeywordVariations()._setup_triple_mock(
            INITIAL_TOPICS_JSON,
            KEYWORD_RESPONSE,
            VARIATION_TOPICS_JSON,
        )

        with patch("src.topicgen.requests.post", mock_post):
            topics = generate_botf_topics(
                "paretotalent.com",
                homepage_html=SAMPLE_HOMEPAGE_HTML,
                api_key="sk-test",
            )

        for topic in topics:
            assert "pareto" not in topic.lower(), f"Brand name leaked into: {topic}"