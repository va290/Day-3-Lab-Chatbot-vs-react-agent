"""
Provider factory: build an LLMProvider from environment variables so the rest
of the app (chatbot, agent, test runner) is provider-agnostic.

Switch providers by editing .env:
    DEFAULT_PROVIDER=local | openai | google
"""

import os
from typing import Optional

from src.core.llm_provider import LLMProvider


def _load_env_files() -> None:
    """Load .env (cloud creds: LLM_ENDPOINT / API_KEY / MODEL live here)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_provider(provider: Optional[str] = None) -> LLMProvider:
    """Instantiate the configured LLM provider. Imports are lazy so you only
    need the SDK for the provider you actually use."""
    _load_env_files()
    provider = (provider or os.getenv("DEFAULT_PROVIDER", "openai")).lower()

    if provider in ("cloud", "openai_compat"):
        from src.core.cloud_provider import CloudProvider
        return CloudProvider(
            model_name=os.getenv("MODEL", os.getenv("DEFAULT_MODEL", "")),
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("LLM_ENDPOINT"),
        )

    if provider == "local":
        from src.core.local_provider import LocalProvider
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=model_path)

    if provider == "openai":
        from src.core.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model_name=os.getenv("DEFAULT_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider in ("google", "gemini"):
        from src.core.gemini_provider import GeminiProvider
        return GeminiProvider(
            model_name=os.getenv("DEFAULT_MODEL", "gemini-1.5-flash"),
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    raise ValueError(f"Unknown DEFAULT_PROVIDER '{provider}'. Use: local | openai | google.")
