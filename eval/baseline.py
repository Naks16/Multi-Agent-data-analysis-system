"""Single-LLM baseline: one call, with the question and the full CSV pasted into the prompt.

This is the "just paste the data into a chatbot" approach the agent pipeline is compared
against. No planning, no code execution, no fact-checking.
"""

from pathlib import Path

from analyst.llm import LLM

SYSTEM_PROMPT = """You are a data analyst. Answer the question using the CSV data you are given.
Give a direct answer with the key numbers, in a few short paragraphs."""


def run_baseline(csv_path: str | Path, question: str, llm: LLM) -> tuple[str, dict]:
    """Return (answer text, token usage)."""
    csv_text = Path(csv_path).read_text(encoding="utf-8")
    reply = llm.complete(SYSTEM_PROMPT, f"Question: {question}\n\nCSV data:\n{csv_text}")
    return reply.text, {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}
