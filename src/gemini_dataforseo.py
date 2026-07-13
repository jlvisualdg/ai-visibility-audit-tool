"""
DataForSEO Gemini LLM Scraper engine.

Uses the DataForSEO API to get real Gemini responses with cited sources,
brand mentions, and structured content. This replaces the OpenRouter
Gemini engine which returned no citations.

API docs: https://docs.dataforseo.com/v3/ai_optimization/gemini/llm_scraper/live/advanced/
"""

from __future__ import annotations

import base64
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

DATAFORSEO_BASE64 = os.environ.get("DATAFORSEO_BASE64", "").strip()
DATAFORSEO_ENDPOINT = "https://api.dataforseo.com/v3/ai_optimization/gemini/llm_scraper/live/advanced"


class GeminiDataForSEOEngine:
    """Real Gemini engine via DataForSEO LLM Scraper API."""

    _engine_name = "gemini"

    def __init__(self, target_domain: str = "", api_key: Optional[str] = None):
        cred = (api_key or DATAFORSEO_BASE64).strip()
        if not cred:
            raise ValueError(
                "GeminiDataForSEOEngine needs DATAFORSEO_BASE64. "
                "Set it in .env or pass api_key=..."
            )
        self._cred = cred
        self._target_domain = target_domain

    @property
    def name(self) -> str:
        return self._engine_name

    def is_real(self) -> bool:
        return True

    def query(self, topic: str) -> dict:
        """Query Gemini via DataForSEO and return structured result.

        Returns dict with: text, citations, engine, latency_ms, error.
        """
        started = time.time()

        # Build auth header
        auth_header = f"Basic {self._cred}"

        # Build task payload — DataForSEO expects an array of tasks
        payload = [{
            "keyword": topic,
            "location_name": "United States",
            "language_name": "English",
        }]

        try:
            r = requests.post(
                DATAFORSEO_ENDPOINT,
                json=payload,
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            r.raise_for_status()
            r.encoding = "utf-8"
            data = r.json()
        except requests.RequestException as e:
            return {
                "text": "",
                "citations": [],
                "engine": self._engine_name,
                "latency_ms": int((time.time() - started) * 1000),
                "error": f"HTTP error: {e}",
                "cost_usd": 0.0,
            }
        except ValueError as e:
            return {
                "text": "",
                "citations": [],
                "engine": self._engine_name,
                "latency_ms": int((time.time() - started) * 1000),
                "error": f"JSON decode error: {e}",
                "cost_usd": 0.0,
            }

        # Parse DataForSEO response structure
        text_parts: list[str] = []
        citations: list[str] = []

        try:
            tasks = data.get("tasks", [])
            if not tasks:
                return {
                    "text": "",
                    "citations": [],
                    "engine": self._engine_name,
                    "latency_ms": int((time.time() - started) * 1000),
                    "error": "No tasks in response",
                    "cost_usd": 0.0,
                }

            task = tasks[0]
            task_cost = float(task.get("cost") or 0.0)

            if task.get("status_code") != 20000:
                return {
                    "text": "",
                    "citations": [],
                    "engine": self._engine_name,
                    "latency_ms": int((time.time() - started) * 1000),
                    "error": f"DataForSEO error: {task.get('status_message', 'unknown')}",
                    "cost_usd": task_cost,
                }

            results = task.get("result", [])
            if not results:
                return {
                    "text": "",
                    "citations": [],
                    "engine": self._engine_name,
                    "latency_ms": int((time.time() - started) * 1000),
                    "error": "No results in task",
                    "cost_usd": task_cost,
                }

            result = results[0]

            # Extract items (gemini_text, gemini_table, etc.)
            items = result.get("items", [])
            for item in items:
                item_type = item.get("type", "")
                if item_type == "gemini_text":
                    md = item.get("markdown", "")
                    original = item.get("original_text", "")
                    if md:
                        text_parts.append(md)
                    elif original:
                        text_parts.append(original)

                    # Extract sources from this item
                    sources = item.get("sources") or []
                    for src in sources:
                        domain = src.get("domain", "")
                        if domain:
                            citations.append(domain)

                elif item_type == "gemini_table":
                    md = item.get("markdown", "")
                    if md:
                        text_parts.append(md)

            # Also extract from top-level sources
            top_sources = result.get("sources", []) or []
            for src in top_sources:
                domain = src.get("domain", "")
                if domain and domain not in citations:
                    citations.append(domain)

        except (KeyError, IndexError, TypeError) as e:
            return {
                "text": "",
                "citations": [],
                "engine": self._engine_name,
                "latency_ms": int((time.time() - started) * 1000),
                "error": f"Parse error: {e}",
                "cost_usd": 0.0,
            }

        text = "\n\n".join(text_parts)

        return {
            "text": text,
            "citations": citations,
            "engine": self._engine_name,
            "latency_ms": int((time.time() - started) * 1000),
            "error": None,
            "cost_usd": task_cost,
        }
