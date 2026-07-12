"""Tests for src/engines.py — real OpenRouter engine clients."""

import time
from unittest.mock import patch

import pytest


def _mock_openrouter_response(content: str = "", annotations: list = None) -> dict:
    """Build a realistic OpenRouter JSON response."""
    return {
        "id": "gen-test-123",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "annotations": annotations or [],
                },
                "finish_reason": "stop",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Unit tests (mock network)
# ---------------------------------------------------------------------------


class TestPerplexityEngineUnit:
    """Test PerplexityEngine with mocked HTTP responses."""

    def test_returns_dict_with_all_keys(self):
        """query() returns a dict with text, citations, engine, latency_ms, error."""
        from src.engines import PerplexityEngine

        engine = PerplexityEngine(api_key="sk-test")
        fake_response = _mock_openrouter_response(
            content="Top vendors: Example Corp (example.com), Acme Inc (acme.io)."
        )

        with patch("src.engines._post_to_openrouter", return_value=fake_response):
            with patch("src.engines._wait_for_rate_limit"):
                result = engine.query("virtual assistant service")

        assert isinstance(result, dict)
        assert set(result.keys()) == {"text", "citations", "engine", "latency_ms", "error"}
        assert result["text"] != ""
        assert result["engine"] == "perplexity"
        assert result["error"] is None
        assert isinstance(result["latency_ms"], int)
        assert isinstance(result["citations"], list)

    def test_extracts_citations_from_annotations(self):
        """Perplexity annotations produce citation domains."""
        from src.engines import PerplexityEngine

        engine = PerplexityEngine(api_key="sk-test")
        fake_response = _mock_openrouter_response(
            content="Some text.",
            annotations=[
                {"url_citation": {"url": "https://www.vendor-alpha.com/page"}},
                {"url_citation": {"url": "https://vendor-beta.io/other"}},
            ],
        )

        with patch("src.engines._post_to_openrouter", return_value=fake_response):
            with patch("src.engines._wait_for_rate_limit"):
                result = engine.query("virtual assistant service")

        assert "vendor-alpha.com" in result["citations"]
        assert "vendor-beta.io" in result["citations"]

    def test_fallback_regex_citations_when_no_annotations(self):
        """When annotations are missing, domains are extracted from text."""
        from src.engines import PerplexityEngine

        engine = PerplexityEngine(api_key="sk-test")
        fake_response = _mock_openrouter_response(
            content="Check out example.com and test-site.org for more info.",
            annotations=[],
        )

        with patch("src.engines._post_to_openrouter", return_value=fake_response):
            with patch("src.engines._wait_for_rate_limit"):
                result = engine.query("virtual assistant service")

        assert "example.com" in result["citations"]
        assert "test-site.org" in result["citations"]

    def test_handles_http_error_gracefully(self):
        """HTTP errors are caught and returned as error string, not raised."""
        from src.engines import PerplexityEngine

        engine = PerplexityEngine(api_key="sk-test")

        with patch("src.engines._post_to_openrouter") as mock_post:
            mock_post.side_effect = __import__("requests").RequestException("Timeout")
            with patch("src.engines._wait_for_rate_limit"):
                result = engine.query("virtual assistant service")

        assert result["text"] == ""
        assert result["citations"] == []
        assert result["engine"] == "perplexity"
        assert result["error"] is not None
        assert "HTTP error" in result["error"]

    def test_handles_bad_response_shape(self):
        """Malformed JSON response is caught and returned as error."""
        from src.engines import PerplexityEngine

        engine = PerplexityEngine(api_key="sk-test")

        with patch("src.engines._post_to_openrouter", return_value={"choices": []}):
            with patch("src.engines._wait_for_rate_limit"):
                result = engine.query("virtual assistant service")

        assert result["error"] is not None
        assert "Unexpected response shape" in result["error"]


class TestChatGPTEngineUnit:
    """Test ChatGPTEngine with mocked HTTP responses."""

    def test_engine_name_is_chatgpt(self):
        from src.engines import ChatGPTEngine

        engine = ChatGPTEngine(api_key="sk-test")
        assert engine.name == "chatgpt"
        assert engine.is_real() is True

    def test_uses_correct_model(self):
        from src.engines import ChatGPTEngine

        engine = ChatGPTEngine(api_key="sk-test")
        assert engine._model == "openai/gpt-4o-mini-search-preview"

    def test_query_returns_expected_dict(self):
        from src.engines import ChatGPTEngine

        engine = ChatGPTEngine(api_key="sk-test")
        fake_response = _mock_openrouter_response(
            content="Top vendors include ManyChat (manychat.com) and Tidio (tidio.com)."
        )

        with patch("src.engines._post_to_openrouter", return_value=fake_response):
            with patch("src.engines._wait_for_rate_limit"):
                result = engine.query("virtual assistant service")

        assert result["engine"] == "chatgpt"
        assert result["error"] is None
        assert isinstance(result["latency_ms"], int)


class TestClaudeEngineUnit:
    """Test ClaudeEngine with mocked HTTP responses."""

    def test_engine_name_is_claude(self):
        from src.engines import ClaudeEngine

        engine = ClaudeEngine(api_key="sk-test")
        assert engine.name == "claude"

    def test_uses_correct_model(self):
        from src.engines import ClaudeEngine

        engine = ClaudeEngine(api_key="sk-test")
        assert engine._model == "anthropic/claude-sonnet-4"

    def test_query_with_no_citations_falls_back_to_regex(self):
        from src.engines import ClaudeEngine

        engine = ClaudeEngine(api_key="sk-test")
        fake_response = _mock_openrouter_response(
            content="I recommend checking out driftscape.com and wanderlog.com for travel planning.",
            annotations=[],
        )

        with patch("src.engines._post_to_openrouter", return_value=fake_response):
            with patch("src.engines._wait_for_rate_limit"):
                result = engine.query("virtual assistant service")

        assert result["engine"] == "claude"
        assert "driftscape.com" in result["citations"]
        assert "wanderlog.com" in result["citations"]


class TestGeminiEngineUnit:
    """Test GeminiEngine with mocked HTTP responses."""

    def test_engine_name_is_gemini(self):
        from src.engines import GeminiEngine

        engine = GeminiEngine(api_key="sk-test")
        assert engine.name == "gemini"

    def test_uses_correct_model(self):
        from src.engines import GeminiEngine

        engine = GeminiEngine(api_key="sk-test")
        assert engine._model == "google/gemini-2.5-flash"

    def test_query_handles_empty_content(self):
        from src.engines import GeminiEngine

        engine = GeminiEngine(api_key="sk-test")
        fake_response = _mock_openrouter_response(content="", annotations=[])

        with patch("src.engines._post_to_openrouter", return_value=fake_response):
            with patch("src.engines._wait_for_rate_limit"):
                result = engine.query("virtual assistant service")

        assert result["engine"] == "gemini"
        assert result["text"] == ""
        assert result["citations"] == []
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Bulk runner tests
# ---------------------------------------------------------------------------


class TestQueryAllEngines:
    """Test the query_all_engines() bulk runner."""

    @pytest.fixture(autouse=True)
    def _patch_api(self):
        """Patch _post_to_openrouter globally so we don't hit the network."""
        mock_response = _mock_openrouter_response(
            content="Top vendors for this topic.",
            annotations=[],
        )
        with patch("src.engines._post_to_openrouter", return_value=mock_response):
            with patch("src.engines._wait_for_rate_limit"):
                yield

    def test_runs_all_four_by_default(self):
        from src.engines import query_all_engines

        results = query_all_engines("virtual assistant service", api_key="sk-test")
        assert len(results) == 4
        engine_names = {r["engine"] for r in results}
        assert engine_names == {"perplexity", "chatgpt", "claude", "gemini"}

    def test_can_limit_to_subset(self):
        from src.engines import query_all_engines

        results = query_all_engines(
            "virtual assistant service",
            api_key="sk-test",
            engines=["perplexity", "claude"],
        )
        assert len(results) == 2
        assert {r["engine"] for r in results} == {"perplexity", "claude"}

    def test_unknown_engine_returns_error(self):
        from src.engines import query_all_engines

        results = query_all_engines(
            "virtual assistant service",
            api_key="sk-test",
            engines=["bogus"],
        )
        assert len(results) == 1
        assert results[0]["error"] is not None
        assert "Unknown engine" in results[0]["error"]

    def test_all_results_have_required_keys(self):
        from src.engines import query_all_engines

        results = query_all_engines("virtual assistant service", api_key="sk-test")
        for r in results:
            assert set(r.keys()) == {"text", "citations", "engine", "latency_ms", "error"}


# ---------------------------------------------------------------------------
# Rate limit test
# ---------------------------------------------------------------------------


class TestRateLimit:
    """Test the global rate limiter."""

    def test_rate_limit_enforces_delay(self):
        """_wait_for_rate_limit blocks for at least the configured interval."""
        from src.engines import _wait_for_rate_limit
        import src.engines as eng_mod

        # Reset the last-call time so we're sure to hit the delay
        eng_mod._last_call_time = time.time()

        start = time.time()
        _wait_for_rate_limit()
        elapsed = time.time() - start

        # It should wait roughly 2 seconds (the configured rate limit)
        assert elapsed >= 1.8, f"Only waited {elapsed:.2f}s, expected >= ~2s"


# ---------------------------------------------------------------------------
# Imports + smoke tests (verifies file loads cleanly)
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    """The engines module imports without errors."""
    import src.engines  # noqa: F401


def test_all_four_classes_exist():
    """All four engine classes are defined and importable."""
    from src.engines import (
        PerplexityEngine,
        ChatGPTEngine,
        ClaudeEngine,
        GeminiEngine,
    )

    for cls in [PerplexityEngine, ChatGPTEngine, ClaudeEngine, GeminiEngine]:
        assert cls is not None


def test_all_engines_reject_empty_key():
    """Instantiation without a key raises ValueError."""
    from src.engines import (
        PerplexityEngine,
        ChatGPTEngine,
        ClaudeEngine,
        GeminiEngine,
    )

    # Temporarily unset the env var so the test is reliable
    with patch.dict("os.environ", {}, clear=True):
        with patch("src.engines.OPENROUTER_API_KEY", ""):
            for cls in [PerplexityEngine, ChatGPTEngine, ClaudeEngine, GeminiEngine]:
                with pytest.raises(ValueError, match="needs OPENROUTER_API_KEY"):
                    cls(api_key="")