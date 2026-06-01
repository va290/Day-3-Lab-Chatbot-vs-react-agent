"""
CloudProvider: adapter around the mentor-provided OpenAI-compatible endpoint
(see the `env` file: LLM_ENDPOINT / API_KEY / MODEL).

The mentor's sample `llm_provider.py` returns a plain string from generate().
This adapter keeps the exact same connection approach (OpenAI client pointed at
a custom base_url) but conforms to our project's `LLMProvider` ABC: generate()
returns {content, usage, latency_ms, provider} so the telemetry tracker can
record tokens, latency and cost just like the other providers.
"""

import time
from typing import Dict, Any, Optional, Generator

from openai import OpenAI

from src.core.llm_provider import LLMProvider


class CloudProvider(LLMProvider):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(model_name, api_key)
        self.base_url = base_url
        # OpenAI-compatible client; base_url points at the free cloud endpoint.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _extract_usage(response) -> Dict[str, int]:
        """OpenAI-compatible endpoints usually return `usage`, but some free
        gateways omit it — fall back to zeros so telemetry never crashes."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        latency_ms = int((time.time() - start_time) * 1000)
        content = response.choices[0].message.content or ""

        return {
            "content": content,
            "usage": self._extract_usage(response),
            "latency_ms": latency_ms,
            "provider": "cloud",
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
