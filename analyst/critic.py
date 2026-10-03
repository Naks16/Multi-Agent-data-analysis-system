"""Critic agent: checks every claim in the draft report against the computed outputs.

Returns {"approved": bool, "issues": [...]}. The graph sends a rejected report back
to the Reporter (max 2 revisions). If it is still rejected after that, the critic
appends the remaining issues to report.md as caveats.
"""

import time
from pathlib import Path

from pydantic import BaseModel, model_validator

from analyst.llm import LLM
from analyst.reporter import format_evidence
from analyst.state import AgentState
from analyst.trace import log_event

MAX_REVISIONS = 2


class CriticVerdict(BaseModel):
    approved: bool
    issues: list[str] = []

    @model_validator(mode="after")
    def approved_means_no_issues(self):
        if self.approved and self.issues:
            raise ValueError("approved must be false when there are issues")
        if not self.approved and not self.issues:
            raise ValueError("a rejected report must list at least one issue")
        return self


SYSTEM_PROMPT = """You are a strict fact-checker for a data analysis report.

You get the report and the step outputs (the ONLY trusted evidence). Check every number and
every conclusion in the report:
- A number is supported only if it appears in a step output, or follows from them by simple
  arithmetic (e.g. a percentage change of two reported values). Allow normal rounding.
- A finding must cite a step whose output actually supports it.
- A causal claim ("X caused Y") is unsupported unless worded as an association.
- p-values and test names must match the step outputs.
- Only computed values in the step outputs are evidence. Facts from outside the data (e.g. how
  the data was collected) are unsupported if stated as fact; they are acceptable only when
  clearly worded as a possible explanation (e.g. "this may be because...").
- Ignore the auto-generated lines in the Limitations section about missing values, failed
  steps and causation; they are added by code.

Reply with ONLY this JSON:
{"approved": true, "issues": []}
or
{"approved": false, "issues": ["Unsupported: '<claim>' - <why, and which step says otherwise>"]}"""


def should_revise(state: AgentState) -> bool:
    """True if the report was rejected and we still have revisions left."""
    return not state["critic_feedback"]["approved"] and state.get("revision_count", 0) < MAX_REVISIONS


def critic_node(state: AgentState, llm: LLM) -> dict:
    start = time.perf_counter()
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Step outputs:\n\n{format_evidence(state['step_results'])}\n\n"
        f"---\nReport to check:\n\n{state['report']}"
    )
    verdict, usage = llm.complete_json(SYSTEM_PROMPT, prompt, CriticVerdict)
    feedback = verdict.model_dump()
    update = {"critic_feedback": feedback}

    # Out of revisions and still not approved: ship the report with the issues listed.
    if not should_revise({**state, **update}) and not verdict.approved:
        caveats = "\n".join(f"- {issue}" for issue in verdict.issues)
        report = (state["report"] + "\n## Unresolved reviewer caveats\n\n"
                  f"The automated fact-checker still flagged these claims after "
                  f"{MAX_REVISIONS} revisions; treat them with caution:\n\n{caveats}\n")
        Path(state["run_dir"], "report.md").write_text(report, encoding="utf-8")
        update["report"] = report

    event = log_event(
        state["run_dir"], "critic",
        input_summary=f"report of {len(state['report'])} chars, revision {state.get('revision_count', 0)}",
        output=feedback,
        duration_s=round(time.perf_counter() - start, 2),
        tokens=usage,
    )
    update["trace"] = state.get("trace", []) + [event]
    return update
