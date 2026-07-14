"""
Central configuration for the AEO Audit Tool.

Bucket weights and score thresholds live here so they are never duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

# Bucket weights for the 3-bucket AEO Score
WEIGHT_VISIBILITY: float = 0.50
WEIGHT_CREDIBILITY: float = 0.25
WEIGHT_INDEXABILITY: float = 0.25

# Score thresholds (3-tier system)
THRESHOLD_STRONG: int = 70      # 70-100 = STRONG
THRESHOLD_DEVELOPING: int = 40  # 40-69 = DEVELOPING
                                 # 0-39  = CRITICAL


@dataclass
class Settings:
    openrouter_api_key: str
    dataforseo_base64: str
    passes_per_query: int
    engine_rate_limit_secs: float
    perplexity_model: str
    chatgpt_model: str
    gemini_model: str
    max_pages: int


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        dataforseo_base64=os.environ.get("DATAFORSEO_BASE64", ""),
        passes_per_query=int(os.environ.get("PASSES_PER_QUERY", "1")),
        engine_rate_limit_secs=float(os.environ.get("ENGINE_RATE_LIMIT_SECS", "2")),
        perplexity_model=os.environ.get("PERPLEXITY_MODEL", "perplexity/sonar"),
        chatgpt_model=os.environ.get("CHATGPT_MODEL", "openai/gpt-4o-mini-search-preview"),
        gemini_model=os.environ.get("GEMINI_MODEL", "google/gemini-2.5-flash"),
        max_pages=int(os.environ.get("MAX_PAGES", "10")),
    )
