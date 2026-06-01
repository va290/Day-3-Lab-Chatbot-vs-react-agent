"""
Baseline Chatbot (Phase 2 of the lab).

This is deliberately the *naive* approach: a single LLM call with no tools and no
reasoning loop. It can answer simple questions, but on multi-step e-commerce
queries it has no way to look up real prices/stock/discounts, so it will
hallucinate numbers. That failure is the whole point — it motivates the ReAct
agent. Telemetry is logged so the chatbot and agent can be compared on the same
metrics (tokens, latency, cost).
"""

from typing import Optional

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class Chatbot:
    """A minimal single-turn chatbot with no tools and no ReAct loop."""

    SYSTEM_PROMPT = (
        "You are a helpful e-commerce assistant. Answer the user's question "
        "directly and concisely."
    )

    def __init__(self, llm: LLMProvider, system_prompt: Optional[str] = None):
        self.llm = llm
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    def run(self, user_input: str) -> str:
        logger.log_event("CHATBOT_START", {"input": user_input, "model": self.llm.model_name})

        result = self.llm.generate(user_input, system_prompt=self.system_prompt)
        content = (result.get("content") or "").strip()

        tracker.track_request(
            provider=result.get("provider", "unknown"),
            model=self.llm.model_name,
            usage=result.get("usage", {}),
            latency_ms=result.get("latency_ms", 0),
        )

        logger.log_event("CHATBOT_END", {"answer": content})
        return content


if __name__ == "__main__":
    # Quick interactive smoke test: `python -m src.agent.chatbot`
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from src.core.factory import build_provider

    bot = Chatbot(build_provider())
    print("Baseline Chatbot (Ctrl-C to exit)\n")
    try:
        while True:
            q = input("You: ")
            if not q.strip():
                continue
            print(f"Bot: {bot.run(q)}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
