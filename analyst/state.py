"""Shared state passed between the LangGraph nodes.

LangGraph gives each node the current state and merges the dict the node
returns back into it. A TypedDict is enough here: it documents the fields and
lets editors type-check them, without extra machinery.
"""

from typing import TypedDict


class PlanStep(TypedDict):
    id: str                 # e.g. "s1"
    description: str        # what to compute, in plain English
    output_type: str        # "number" | "table" | "chart"
    is_statistical_test: bool


class StepResult(TypedDict):
    step_id: str
    description: str
    code: str               # last code that was run (successful or not)
    stdout: str             # JSON printed by the code (empty if failed)
    error: str | None       # last error message, if the step failed
    retries: int            # how many times we re-asked the LLM after a failure
    success: bool
    charts: list[str]       # PNG paths created by this step


class AgentState(TypedDict, total=False):
    # Inputs
    csv_path: str
    question: str
    run_dir: str            # runs/<timestamp>/ - report, charts and trace go here
    schema_summary: str     # compact text description of the CSV (never the full data)

    # Planner -> Analyst
    plan: list[PlanStep]

    # Analyst -> Reporter/Critic
    step_results: list[StepResult]
    charts: list[str]

    # Reporter <-> Critic loop
    report: str
    critic_feedback: dict   # {"approved": bool, "issues": [...]}
    revision_count: int

    # Observability: one dict per agent step (also written to trace.jsonl)
    trace: list[dict]
