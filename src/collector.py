"""
Multi-engine × multi-topic collector for AI visibility audits.

Runs every engine against every topic, extracts brand mentions from each
response, and returns a flat list of structured result dicts.
"""

from __future__ import annotations

import time
from typing import Optional

from src.engines import (
    PerplexityEngine,
    ChatGPTEngine,
)
from src.gemini_dataforseo import GeminiDataForSEOEngine
from src.scoring import extract_brand_mentions

# ---------------------------------------------------------------------------
# Engine name map — lowercase internal → display name
# ---------------------------------------------------------------------------

ENGINE_DISPLAY_NAMES: dict[str, str] = {
    "perplexity": "Perplexity",
    "chatgpt": "ChatGPT",
    "gemini": "Gemini",
}


def execute_all(
    topics: list[str],
    target_domain: str,
    *,
    api_key: Optional[str] = None,
    sleep_between: float = 2.0,
) -> list[dict]:
    """Run all four engines against every topic and extract brand mentions.

    Loop order: for engine in engines: for topic in topics (5 topics × 4 engines = 20 results).

    Each result dict shape:
        {
            topic: str,
            engine: str,           # display name: 'Perplexity', 'ChatGPT', 'Claude', 'Gemini'
            text: str,
            citations: list[str],
            latency_ms: int,
            error: str | None,
            brand_mentions: list[str],   # from extract_brand_mentions
            positions: list[int],         # from extract_brand_mentions
            target_mention_count: int,    # len(positions)
        }

    Engine failures are handled gracefully — the result dict will have
    error populated and empty lists for citations/brand_mentions/positions.
    """
    # Only engines that return real citations/sources.
    # Claude removed: doesn't return URL annotations via OpenRouter.
    # Gemini uses DataForSEO LLM Scraper API for real citations.
    class _GeminiWrapper:
        """Wraps GeminiDataForSEOEngine to match the constructor signature."""
        _engine_name = "gemini"
        def __init__(self, api_key=None):
            self._inner = GeminiDataForSEOEngine(target_domain=target_domain)
        @property
        def name(self):
            return self._inner.name
        def is_real(self):
            return self._inner.is_real()
        def query(self, topic):
            return self._inner.query(topic)

    engine_classes = [PerplexityEngine, ChatGPTEngine, _GeminiWrapper]

    results: list[dict] = []

    for cls in engine_classes:
        engine_name = ""  # will be set below

        try:
            engine = cls(api_key=api_key)
            engine_name = engine.name
        except ValueError as e:
            # Engine instantiation failed (e.g. missing API key)
            display_name = ENGINE_DISPLAY_NAMES.get(
                cls._engine_name, cls._engine_name
            )
            for topic in topics:
                results.append({
                    "topic": topic,
                    "engine": display_name,
                    "text": "",
                    "citations": [],
                    "latency_ms": 0,
                    "error": str(e),
                    "brand_mentions": [],
                    "positions": [],
                    "target_mention_count": 0,
                })
            continue

        display_name = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)

        for topic in topics:
            # Rate limit is handled inside engine.query(), but we also sleep
            # between calls as a safety net
            try:
                raw = engine.query(topic)
            except Exception as e:
                raw = {
                    "text": "",
                    "citations": [],
                    "engine": engine_name,
                    "latency_ms": 0,
                    "error": str(e),
                }

            # ── Two independent metrics ──
            # 1. BRAND RECOMMENDATIONS: brand names explicitly written in the
            #    AI's answer text (what the user reads). Extracted via regex
            #    on capitalized phrases, filtered for AI-isms.
            # 2. CITATIONS: domains/URLs returned as sources/annotations by
            #    the engine (technical source references). These are separate
            #    from brand recommendations — a domain can be cited without
            #    the brand name being written in the answer, and vice versa.
            text = raw.get("text", "") or ""
            citations = raw.get("citations", []) or []
            error = raw.get("error")

            if error:
                brand_mentions: list[str] = []
                positions: list[int] = []
            else:
                try:
                    # Brand recommendations: from text only
                    brand_mentions, positions = extract_brand_mentions(
                        text, target_domain
                    )
                except Exception:
                    brand_mentions = []
                    positions = []

            results.append({
                "topic": topic,
                "engine": display_name,
                "text": text,
                "citations": raw.get("citations", []),
                "latency_ms": raw.get("latency_ms", 0),
                "error": error,
                "brand_mentions": brand_mentions,
                "positions": positions,
                "target_mention_count": len(positions),
            })

            # Sleep between calls to avoid hammering APIs
            time.sleep(sleep_between)

    return results