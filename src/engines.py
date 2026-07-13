"""
Real OpenRouter engine clients for AI visibility audits.

Each engine wraps the OpenRouter chat completions API and exposes a
uniform `query(topic) -> dict` interface. This module coexists with
src/visibility.py (which has its own PerplexityEngine + mock engines).

Engine protocol (informal):
    class Engine:
        name: str
        def query(self, topic: str) -> dict: ...

Return dict shape:
    {text: str, citations: list[str], engine: str, latency_ms: int, error: str|None}

Rate limiting: 2 seconds between successive API calls (enforced globally
via a module-level rate limiter shared by all four engines).

Model map:
    PerplexityEngine  -> perplexity/sonar
    ChatGPTEngine     -> openai/gpt-4o-mini-search-preview
    ClaudeEngine      -> anthropic/claude-sonnet-4
    GeminiEngine      -> google/gemini-2.5-flash

Citation handling:
  - Perplexity: data['choices'][0]['message']['annotations'][i]['url_citation']['url']
  - Other engines: may not return real citations — returns empty list gracefully.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# The query IS the prompt — don't wrap it. The topicgen already generates
# complete buyer-intent queries. Wrapping them in a research question changes
# what the engine responds to and degrades brand recommendation quality.
DEFAULT_PROMPT_TEMPLATE = "{topic}"

# Rate limiting — shared across all engines so concurrent callers wait in line.
_RATE_LIMIT_SECS = float(os.environ.get("ENGINE_RATE_LIMIT_SECS", "2"))
_last_call_time: float = 0.0

# Regex to extract domains from free text (fallback for non-citation engines)
DOMAIN_RE = re.compile(
    r"\b((?:[a-z0-9-]+\.)+[a-z]{2,})\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_rate_limit() -> None:
    """Block until at least _RATE_LIMIT_SECS have passed since the last API call."""
    global _last_call_time
    now = time.time()
    wait = _last_call_time + _RATE_LIMIT_SECS - now
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.time()


def _elapsed_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def _domain_from_url(url: str) -> Optional[str]:
    """Extract bare domain from a URL. Returns None on failure."""
    try:
        host = urlparse(url).netloc.lower()
    except (ValueError, TypeError):
        return None
    if not host:
        if "." in url and " " not in url:
            return url.lower().split("/")[0]
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _extract_citations_from_text(text: str) -> list[str]:
    """Extract unique domain mentions from free text as a fallback."""
    domains = DOMAIN_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for d in domains:
        d_lower = d.lower()
        if d_lower not in seen:
            seen.add(d_lower)
            result.append(d_lower)
    return result


def _post_to_openrouter(
    model: str,
    messages: list[dict],
    api_key: str,
) -> dict:
    """POST to OpenRouter chat completions and return the parsed JSON response.

    Raises requests.RequestException on network/HTTP errors,
    ValueError on JSON decode errors.
    """
    r = requests.post(
        OPENROUTER_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://boringmarketer.com",
            "X-Title": "AEO Audit Tool - Engines",
        },
        json={
            "model": model,
            "messages": messages,
        },
        timeout=60,
    )
    r.raise_for_status()
    # Force UTF-8: OpenRouter sends Unicode but requests may default to
    # ISO-8859-1 (latin-1) when the server omits charset, which causes
    # "latin-1 codec can't encode character" on em-dashes etc.
    r.encoding = "utf-8"
    return r.json()


# ---------------------------------------------------------------------------
# Base engine
# ---------------------------------------------------------------------------


class _BaseEngine:
    """Common scaffolding for all OpenRouter-based engines."""

    _model: str
    _engine_name: str

    def __init__(self, api_key: Optional[str] = None):
        key = (api_key or OPENROUTER_API_KEY).strip()
        if not key:
            raise ValueError(
                f"{type(self).__name__} needs OPENROUTER_API_KEY. "
                "Set it in .env or pass api_key=..."
            )
        self._api_key = key

    @property
    def name(self) -> str:
        return self._engine_name

    def is_real(self) -> bool:
        return True

    def query(self, topic: str) -> dict:
        """Call the engine for `topic` and return a result dict.

        Returns:
            dict with keys: text, citations, engine, latency_ms, error.
        """
        prompt = DEFAULT_PROMPT_TEMPLATE.format(topic=topic)
        started = time.time()

        try:
            _wait_for_rate_limit()
            data = _post_to_openrouter(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self._api_key,
            )
        except requests.RequestException as e:
            return {
                "text": "",
                "citations": [],
                "engine": self._engine_name,
                "latency_ms": _elapsed_ms(started),
                "error": f"HTTP error: {e}",
            }
        except ValueError as e:
            return {
                "text": "",
                "citations": [],
                "engine": self._engine_name,
                "latency_ms": _elapsed_ms(started),
                "error": f"JSON decode error: {e}",
            }

        # --- Parse response ---
        text = ""
        citations: list[str] = []

        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content", "") or ""
            # Try to extract citations from annotations (Perplexity-specific path)
            annotations = message.get("annotations") or []
            for ann in annotations:
                url_cite = ann.get("url_citation") or {}
                url = url_cite.get("url")
                if url:
                    domain = _domain_from_url(url)
                    if domain:
                        citations.append(domain)
        except (KeyError, IndexError, TypeError) as e:
            return {
                "text": "",
                "citations": [],
                "engine": self._engine_name,
                "latency_ms": _elapsed_ms(started),
                "error": f"Unexpected response shape: {e}",
            }

        # Fallback: regex the text for domain mentions if no annotations
        if not citations:
            citations = _extract_citations_from_text(text)

        return {
            "text": text,
            "citations": citations,
            "engine": self._engine_name,
            "latency_ms": _elapsed_ms(started),
            "error": None,
        }


# ---------------------------------------------------------------------------
# Concrete engines
# ---------------------------------------------------------------------------


class PerplexityEngine(_BaseEngine):
    """Perplexity Sonar via OpenRouter — search-grounded with citations."""

    _model = "perplexity/sonar"
    _engine_name = "perplexity"


class ChatGPTEngine(_BaseEngine):
    """ChatGPT (GPT-4o-mini search preview) via OpenRouter."""

    _model = "openai/gpt-4o-mini-search-preview"
    _engine_name = "chatgpt"


class ClaudeEngine(_BaseEngine):
    """Claude Sonnet 4 via OpenRouter."""

    _model = "anthropic/claude-sonnet-4"
    _engine_name = "claude"


class GeminiEngine(_BaseEngine):
    """Gemini 2.5 Flash via OpenRouter."""

    _model = "google/gemini-2.5-flash"
    _engine_name = "gemini"


# ---------------------------------------------------------------------------
# Bulk runner
# ---------------------------------------------------------------------------


def query_all_engines(
    topic: str,
    api_key: Optional[str] = None,
    engines: Optional[list[str]] = None,
) -> list[dict]:
    """Run all four engines against `topic` and return a list of result dicts.

    Args:
        topic: The topic to query (e.g. 'virtual assistant service').
        api_key: OpenRouter API key override.
        engines: Subset of engine names to run (default: all four).

    Returns:
        List of result dicts, one per engine.
    """
    engine_map = {
        "perplexity": PerplexityEngine,
        "chatgpt": ChatGPTEngine,
        "claude": ClaudeEngine,
        "gemini": GeminiEngine,
    }

    names = engines or list(engine_map.keys())
    results: list[dict] = []

    for name in names:
        cls = engine_map.get(name)
        if cls is None:
            results.append({
                "text": "",
                "citations": [],
                "engine": name,
                "latency_ms": 0,
                "error": f"Unknown engine: {name}",
            })
            continue

        try:
            engine = cls(api_key=api_key)
            result = engine.query(topic)
        except ValueError as e:
            result = {
                "text": "",
                "citations": [],
                "engine": name,
                "latency_ms": 0,
                "error": str(e),
            }

        results.append(result)

    return results