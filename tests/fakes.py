"""Test double for the LLM, shared by the planner/analyst/graph tests."""

from analyst.llm import LLM, LLMReply


class FakeLLM(LLM):
    """Returns scripted replies instead of calling the API.

    `replies` is a list of strings returned in order by complete(). Every prompt
    received is saved in `self.prompts` so tests can inspect it.
    """

    def __init__(self, replies: list[str]):
        # Deliberately skip LLM.__init__: no API key, no client.
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> LLMReply:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return LLMReply(self.replies.pop(0), input_tokens=10, output_tokens=5)
