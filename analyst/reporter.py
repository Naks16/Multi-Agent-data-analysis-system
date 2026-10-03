"""Reporter agent: turns step_results into report.md.

The LLM returns *structured* content (short answer, findings with step ids,
test results, limitations). Python then renders the markdown, so the required
sections, chart embeds, failed-step list and data-quality notes are always
present and correct no matter what the LLM writes.
"""

import json
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from analyst.llm import LLM
from analyst.state import AgentState, StepResult
from analyst.trace import log_event

MAX_EVIDENCE_CHARS = 3000  # per step, when showing outputs to the LLM


class Finding(BaseModel):
    text: str = Field(min_length=5)
    step_id: str


class StatTest(BaseModel):
    step_id: str
    test_name: str
    p_value: float
    interpretation: str = Field(min_length=5)


class ReportDraft(BaseModel):
    short_answer: str = Field(min_length=5)
    key_findings: list[Finding] = []
    statistical_tests: list[StatTest] = []
    limitations: list[str] = []


SYSTEM_PROMPT = """You write the findings of a data analysis for a business reader.

Strict rule: every number and every conclusion must come from the step outputs you are given.
Do not invent numbers, do not round differently than the outputs, and do not claim causes the
data cannot show. If the outputs don't answer something, say so.
Knowledge from outside the data (e.g. how it was collected) may be mentioned only as a
possible explanation ("this may be because..."), never stated as a fact.

Reply with ONLY this JSON:
{
  "short_answer": "2-3 sentence direct answer to the question",
  "key_findings": [{"text": "finding with its numbers", "step_id": "s2"}],
  "statistical_tests": [{"step_id": "s4", "test_name": "Mann-Whitney U", "p_value": 0.0001,
                         "interpretation": "plain-English meaning, incl. whether it is significant at 5%"}],
  "limitations": ["data quality issues, assumptions, what the data cannot tell us"]
}
Each key finding must cite the id of the step whose output supports it.
Do not list missing-value counts, failed steps or "correlation is not causation" under
limitations: those lines are added automatically."""


def format_evidence(step_results: list[StepResult]) -> str:
    """Successful step outputs, labelled by step id. Shared with the Critic."""
    blocks = []
    for r in step_results:
        if r["success"]:
            blocks.append(f"[{r['step_id']}] {r['description']}\n{r['stdout'][:MAX_EVIDENCE_CHARS]}")
    return "\n\n".join(blocks) or "(no successful steps)"


def build_prompt(state: AgentState) -> str:
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Step outputs (JSON printed by the analysis code):\n\n{format_evidence(state['step_results'])}"
    )
    feedback = state.get("critic_feedback")
    if feedback and not feedback["approved"]:
        issues = "\n".join(f"- {i}" for i in feedback["issues"])
        prompt += (
            f"\n\n---\nA reviewer rejected your previous report:\n\n{state['report']}\n\n"
            f"Issues to fix:\n{issues}\n\n"
            "Fix every issue. Remove any claim you cannot support with the step outputs."
        )
    return prompt


def make_draft(llm: LLM, state: AgentState) -> tuple[ReportDraft, dict]:
    ok_ids = {r["step_id"] for r in state["step_results"] if r["success"]}

    def check_citations(draft: ReportDraft) -> str | None:
        cited = [f.step_id for f in draft.key_findings] + [t.step_id for t in draft.statistical_tests]
        bad = [c for c in cited if c not in ok_ids]
        if bad:
            return f"These step ids are unknown or failed: {bad}. Only cite: {sorted(ok_ids)}"
        if not draft.key_findings:
            return "Give at least one key finding."
        return None

    return llm.complete_json(SYSTEM_PROMPT, build_prompt(state), ReportDraft, extra_check=check_citations)


# --------------------------------------------------------------------------- #
# Markdown rendering (pure Python, no LLM)
# --------------------------------------------------------------------------- #

def missing_value_notes(csv_path: str) -> list[str]:
    nulls = pd.read_csv(csv_path).isna().sum()
    return [f"Column `{col}` has {n} missing values." for col, n in nulls.items() if n > 0]


def render_report(state: AgentState, draft: ReportDraft) -> str:
    lines = [
        "# Analysis Report", "",
        f"**Question:** {state['question']}", "",
        "## Short answer", "", draft.short_answer, "",
        "## Key findings", "",
    ]
    lines += [f"- {f.text} *(source: step {f.step_id})*" for f in draft.key_findings] or ["- None."]

    lines += ["", "## Statistical tests", ""]
    if draft.statistical_tests:
        lines += ["| Test | Step | p-value | Significant at 5%? | Interpretation |",
                  "|---|---|---|---|---|"]
        for t in draft.statistical_tests:
            significant = "Yes" if t.p_value < 0.05 else "No"
            lines.append(f"| {t.test_name} | {t.step_id} | {t.p_value:.4g} | {significant} | {t.interpretation} |")
    else:
        lines.append("No statistical tests were run successfully.")

    if state.get("charts"):
        lines += ["", "## Charts", ""]
        # report.md lives in the same folder as the PNGs, so use bare file names.
        lines += [f"![{Path(c).stem}]({Path(c).name})\n" for c in state["charts"]]

    limitations = []
    for r in state["step_results"]:
        if not r["success"]:
            last_error_line = (r["error"] or "").strip().splitlines()[0] if r["error"] else "unknown error"
            limitations.append(f"Step {r['step_id']} (\"{r['description']}\") failed after "
                               f"{r['retries']} retries: {last_error_line}")
    limitations += missing_value_notes(state["csv_path"])
    limitations += draft.limitations
    limitations.append("This analysis shows associations in the data; it does not prove causation.")

    # The LLM sometimes repeats a line the code already adds; keep the first copy only.
    seen = set()
    lines += ["", "## Limitations", ""]
    for lim in limitations:
        if lim.strip().lower() not in seen:
            seen.add(lim.strip().lower())
            lines.append(f"- {lim}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# LangGraph node
# --------------------------------------------------------------------------- #

def reporter_node(state: AgentState, llm: LLM) -> dict:
    start = time.perf_counter()
    is_revision = bool(state.get("critic_feedback"))
    revision_count = state.get("revision_count", 0) + (1 if is_revision else 0)

    if any(r["success"] for r in state["step_results"]):
        draft, usage = make_draft(llm, state)
    else:
        draft = ReportDraft(short_answer="The question could not be answered: every analysis step failed.")
        usage = {"input_tokens": 0, "output_tokens": 0}

    report = render_report(state, draft)
    Path(state["run_dir"], "report.md").write_text(report, encoding="utf-8")

    event = log_event(
        state["run_dir"], "reporter",
        input_summary=f"{len(state['step_results'])} step results, revision {revision_count}",
        output=json.loads(draft.model_dump_json()),
        duration_s=round(time.perf_counter() - start, 2),
        tokens=usage,
    )
    return {
        "report": report,
        "revision_count": revision_count,
        "trace": state.get("trace", []) + [event],
    }
