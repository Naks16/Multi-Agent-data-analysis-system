"""Thin wrapper around the Anthropic SDK.

Two methods:
- complete(system, prompt)          -> plain text + token usage
- complete_json(system, prompt, M)  -> a validated pydantic model M
  (with one "repair" retry if the reply isn't valid JSON / doesn't match M)

Tests replace `complete()` with a fake, so no network calls happen in tests.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


@dataclass
class LLMReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMJsonError(Exception):
    """Raised when the model's reply is still invalid after the repair retry."""


def extract_json(text: str) -> str:
    """Pull the JSON out of a reply that may be wrapped in ```json fences or prose."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Otherwise take everything from the first { or [ to the last } or ].
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end < start:
        return text.strip()
    return text[start:end + 1]


class LLM:
    def __init__(self, model: str | None = None, max_tokens: int = 16000):
        # Both come from the environment; nothing is hardcoded.
        self.model = model or os.environ.get("MODEL_NAME")
        if not self.model:
            raise RuntimeError("Set the MODEL_NAME environment variable (see .env.example).")
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def complete(self, system: str, prompt: str) -> LLMReply:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined this request (stop_reason='refusal').")
        # The reply can contain thinking blocks as well as text; keep only the text.
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMReply(text, response.usage.input_tokens, response.usage.output_tokens)

    def complete_json(
        self,
        system: str,
        prompt: str,
        model_cls: type[M],
        extra_check: Callable[[M], str | None] | None = None,
    ) -> tuple[M, dict]:
        """Ask for JSON, validate it with `model_cls`, and retry once with the error.

        `extra_check` lets the caller add rules pydantic can't express (it
        returns an error message, or None if the object is fine).
        Returns (parsed_object, {"input_tokens": .., "output_tokens": ..}).
        """
        usage = {"input_tokens": 0, "output_tokens": 0}
        current_prompt = prompt

        for attempt in range(2):  # first try + one repair retry
            reply = self.complete(system, current_prompt)
            usage["input_tokens"] += reply.input_tokens
            usage["output_tokens"] += reply.output_tokens

            try:
                parsed = model_cls.model_validate(json.loads(extract_json(reply.text)))
                problem = extra_check(parsed) if extra_check else None
                if problem is None:
                    return parsed, usage
            except (json.JSONDecodeError, ValidationError) as e:
                problem = str(e)

            current_prompt = (
                f"{prompt}\n\n---\nYour previous reply was:\n{reply.text}\n\n"
                f"It was rejected because:\n{problem}\n\n"
                "Reply again with ONLY the corrected JSON, no other text."
            )

        raise LLMJsonError(f"Invalid JSON after repair retry: {problem}")
