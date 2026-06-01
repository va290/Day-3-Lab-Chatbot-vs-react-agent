from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

from openai import OpenAI
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = ROOT_DIR / ".env"


def load_env_config(env_path: Path = DEFAULT_ENV_PATH) -> Dict[str, str]:
    load_dotenv(dotenv_path=env_path)

    config = {
        "LLM_ENDPOINT": os.getenv("LLM_ENDPOINT", "").strip(),
        "API_KEY": os.getenv("API_KEY", "").strip(),
        "MODEL": os.getenv("MODEL", "").strip(),
    }

    missing = [key for key, value in config.items() if not value]
    if missing:
        raise ValueError(
            f"Missing required env values: {', '.join(missing)}. "
            f"Expected them in {env_path} or exported environment variables."
        )

    return config


class LLMProvider:
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        env_path: Path = DEFAULT_ENV_PATH,
    ) -> None:
        config = load_env_config(env_path)

        self.model_name = model_name or config["MODEL"]
        self.api_key = api_key or config["API_KEY"]
        self.base_url = base_url or config["LLM_ENDPOINT"]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the model configured in .env.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Say hello in one short sentence.",
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens instead of waiting for the full response.",
    )
    args = parser.parse_args()

    provider = LLMProvider()

    if args.stream:
        for token in provider.stream(args.prompt, system_prompt=args.system):
            print(token, end="", flush=True)
        print()
        return

    print(provider.generate(args.prompt, system_prompt=args.system))


if __name__ == "__main__":
    main()
