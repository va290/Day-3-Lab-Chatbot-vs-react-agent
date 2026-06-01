"""
CloudProvider: adapter around the mentor-provided OpenAI-compatible endpoint
(see .env: LLM_ENDPOINT / API_KEY / MODEL).

The mentor's sample `llm_provider.py` returned a plain string. This adapter
keeps the same connection approach (OpenAI client pointed at a custom base_url)
but conforms to our project's `LLMProvider` ABC: generate() returns
{content, usage, latency_ms, provider} so telemetry can record tokens, latency
and cost like the other providers.

Resilience: free cloud endpoints frequently drop connections / rate-limit, so
the network call is wrapped in an explicit retry-with-exponential-backoff loop.
Each retry is logged as an `LLM_RETRY` telemetry event. The OpenAI SDK's own
retries are disabled (max_retries=0) so our loop is the single source of truth.
"""

import time
from typing import Dict, Any, Optional, Generator

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

# Transient failures worth retrying (connection drop, timeout, 429, 5xx).
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class CloudProvider(LLMProvider):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ):
        super().__init__(model_name, api_key)
        self.base_url = base_url
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # max_retries=0: disable the SDK's silent retries so our explicit loop
        # (which logs LLM_RETRY) is the only retry mechanism.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

    # ----------------------------------------------------------- resilience
    def _create_completion(self, *, stream: bool, messages):
        """Call the chat-completions API with retry + exponential backoff.
        Logs an LLM_RETRY event for every retry and LLM_ERROR if it gives up."""
        attempt = 0
        while True:
            try:
                return self.client.chat.completions.create(
                    model=self.model_name, messages=messages, stream=stream,
                )
            except _RETRYABLE as e:
                attempt += 1
                if attempt > self.max_retries:
                    logger.log_event("LLM_ERROR", {
                        "provider": "cloud", "model": self.model_name,
                        "attempts": attempt, "error_type": type(e).__name__,
                        "error": str(e)[:200],
                    })
                    raise
                delay = self.backoff_base * (2 ** (attempt - 1))  # 1s, 2s, 4s, ...
                logger.log_event("LLM_RETRY", {
                    "provider": "cloud", "model": self.model_name,
                    "attempt": attempt, "max_retries": self.max_retries,
                    "delay_s": delay, "error_type": type(e).__name__,
                    "error": str(e)[:200],
                })
                time.sleep(delay)

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

    # --------------------------------------------------------------- generate
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._create_completion(stream=False, messages=messages)

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

        # Retry covers establishing the stream; a mid-stream drop is not resumed.
        stream = self._create_completion(stream=True, messages=messages)

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
