"""Planner agent: schema summary + question -> 3-6 concrete analysis steps.

The LLM never sees the full CSV, only `summarize_csv()`'s compact summary.
"""

import re
import time
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from analyst.llm import LLM
from analyst.state import AgentState
from analyst.trace import log_event


# --------------------------------------------------------------------------- #
# Schema summary (what the LLM is allowed to see about the data)
# --------------------------------------------------------------------------- #

def summarize_csv(csv_path: str, max_distinct: int = 20) -> str:
    df = pd.read_csv(csv_path)
    parts = [
        f"Rows: {len(df)}, Columns: {len(df.columns)}",
        "\nColumn dtypes:\n" + df.dtypes.to_string(),
        "\nNull counts:\n" + df.isna().sum().to_string(),
        "\nFirst 5 rows:\n" + df.head(5).to_string(index=False),
        "\ndescribe():\n" + df.describe(include="all").to_string(),
    ]
    # List the values of low-cardinality text columns (e.g. region names) so the
    # LLM can write correct filters without seeing the data.
    text_columns = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_columns:
        values = df[col].dropna().unique()
        if len(values) <= max_distinct:
            parts.append(f"\nDistinct values of '{col}': {sorted(map(str, values))}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #

class Step(BaseModel):
    id: str
    description: str = Field(min_length=10)
    output_type: Literal["number", "table", "chart"]
    is_statistical_test: bool = False


class Plan(BaseModel):
    steps: list[Step] = Field(min_length=3, max_length=6)

    @model_validator(mode="after")
    def ids_are_unique(self):
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"step ids must be unique, got {ids}")
        return self


# Words that suggest the question is about a change over time or a comparison.
CHANGE_WORDS = re.compile(
    r"\b(why|drop|dropped|fall|fell|decline|declined|decrease|increase|rise|rose|grow|growth|"
    r"change|changed|compare|comparison|versus|vs|differ|difference|higher|lower|impact)\b",
    re.IGNORECASE,
)


def needs_statistical_test(question: str) -> bool:
    return bool(CHANGE_WORDS.search(question))


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a senior data analyst planning an analysis.
You will get a summary of a CSV file (not the full data) and a question.
Plan 3-6 concrete steps that a pandas/scipy script can compute from the CSV.

Rules:
- Each step must be computable from the columns that exist. Name the exact columns.
- Prefer steps that isolate the cause (break the metric down by each categorical column).
- If the question is about a change or a comparison, include at least one step that runs a
  statistical test (e.g. Welch t-test or Mann-Whitney U of the target period vs. prior periods)
  and set "is_statistical_test": true on that step.
- output_type is one of "number", "table", "chart".

Reply with ONLY this JSON, no other text:
{"steps": [{"id": "s1", "description": "...", "output_type": "table", "is_statistical_test": false}]}"""


def build_prompt(schema_summary: str, question: str) -> str:
    return f"Question: {question}\n\nData summary:\n{schema_summary}"


def make_plan(llm: LLM, schema_summary: str, question: str) -> tuple[Plan, dict]:
    """Return (validated plan, token usage). Raises LLMJsonError if the LLM can't comply."""

    def check_stat_test(plan: Plan) -> str | None:
        if needs_statistical_test(question) and not any(s.is_statistical_test for s in plan.steps):
            return ("The question involves a change or comparison, so at least one step must "
                    "run a statistical test with \"is_statistical_test\": true.")
        return None

    return llm.complete_json(
        SYSTEM_PROMPT, build_prompt(schema_summary, question), Plan, extra_check=check_stat_test
    )


# --------------------------------------------------------------------------- #
# LangGraph node
# --------------------------------------------------------------------------- #

def planner_node(state: AgentState, llm: LLM) -> dict:
    start = time.perf_counter()
    schema_summary = summarize_csv(state["csv_path"])
    plan, usage = make_plan(llm, schema_summary, state["question"])
    steps = [s.model_dump() for s in plan.steps]

    event = log_event(
        state["run_dir"], "planner",
        input_summary=f"question={state['question']!r}, schema_summary={len(schema_summary)} chars",
        output=steps,
        duration_s=round(time.perf_counter() - start, 2),
        tokens=usage,
    )
    return {
        "schema_summary": schema_summary,
        "plan": steps,
        "trace": state.get("trace", []) + [event],
    }
