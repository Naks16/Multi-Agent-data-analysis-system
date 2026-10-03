"""Analyst agent: for each plan step, the LLM writes pandas/scipy code that we run
in the sandboxed executor. Failures are fed back to the LLM (max 3 retries per step);
after that the step is marked failed and we move on.
"""

import json

from pydantic import BaseModel, Field

from analyst.executor import run_code
from analyst.llm import LLM, LLMJsonError
from analyst.state import AgentState, PlanStep, StepResult
from analyst.trace import log_event

MAX_RETRIES = 3
MAX_CONTEXT_CHARS = 1500  # how much of earlier steps' output we show the LLM
# Step outputs are the evidence for the report, so they must contain computed values,
# not sentences the LLM wrote. Labels like "Mann-Whitney U" or "Upper West Side South"
# are short; a text value with this many words or more is treated as prose and rejected.
MAX_WORDS_IN_TEXT_VALUE = 8


class CodeReply(BaseModel):
    code: str = Field(min_length=1)


SYSTEM_PROMPT = """You write Python code for ONE step of a data analysis.

The code runs in a restricted sandbox:
- Two variables already exist: CSV_PATH (the input CSV) and OUTPUT_DIR (folder for charts).
  Do not redefine them.
- You may import: pandas, numpy, scipy, matplotlib, json, math, statistics, datetime.
- You may NOT import os, sys, subprocess, socket, requests, shutil, pathlib, or use
  eval/exec/__import__, or access dunder attributes like __class__.
- Print EXACTLY ONE JSON object to stdout with print(json.dumps(result, default=str)) and print
  nothing else. Convert numpy/pandas values to plain Python (float(), int(), .to_dict()).
- Keep the output small: at most ~20 rows for any table. Round floats to 4 significant digits.
- The JSON must contain ONLY values computed from the data, plus short labels (column names,
  category values, test names). NO notes, explanations, interpretations or conclusions:
  interpreting results is the reporter's job, and anything you write by hand is not evidence.
- Handle missing values explicitly (e.g. dropna) and report how many rows you dropped.
- For statistical tests, include in the JSON: test name, groups compared, group sizes,
  group means, the statistic and the p_value.
- For charts: plt.savefig(f"{OUTPUT_DIR}/<step_id>_<short_name>.png", dpi=100,
  bbox_inches="tight") then plt.close(). Give every chart a title and axis labels.

Reply with ONLY this JSON: {"code": "<the python code>"}"""


def build_prompt(state: AgentState, step: PlanStep, previous: list[StepResult]) -> str:
    context = ""
    for r in previous:
        if r["success"]:
            context += f"\n[{r['step_id']}] {r['description']}\n{r['stdout'][:MAX_CONTEXT_CHARS]}\n"
    return (
        f"Overall question: {state['question']}\n\n"
        f"Data summary:\n{state['schema_summary']}\n\n"
        f"Results of earlier steps (for context):{context or ' none yet'}\n\n"
        f"Your step (id={step['id']}, expected output: {step['output_type']}):\n"
        f"{step['description']}"
    )


def build_retry_prompt(base_prompt: str, code: str, error: str) -> str:
    return (
        f"{base_prompt}\n\n---\nYour previous code failed.\n\nCode:\n{code}\n\n"
        f"Error:\n{error}\n\nFix the problem and reply with the full corrected code as JSON."
    )


def find_prose(value) -> str | None:
    """Return the first string inside a JSON value that looks like a sentence, else None."""
    if isinstance(value, str):
        return value if len(value.split()) >= MAX_WORDS_IN_TEXT_VALUE else None
    if isinstance(value, dict):
        children = list(value.keys()) + list(value.values())
    elif isinstance(value, list):
        children = value
    else:
        return None  # numbers, booleans, null
    for child in children:
        found = find_prose(child)
        if found:
            return found
    return None


def check_stdout(stdout: str) -> str | None:
    """Return an error message if stdout isn't valid, prose-free JSON, else None."""
    if not stdout.strip():
        return "The code printed nothing. It must print one JSON object."
    try:
        output = json.loads(stdout)
    except json.JSONDecodeError as e:
        return f"stdout is not valid JSON ({e}). stdout was:\n{stdout[:500]}"
    prose = find_prose(output)
    if prose:
        return (f"The output contains free text: {prose[:200]!r}. Print only computed values "
                "and short labels - no notes, explanations or conclusions.")
    return None


def run_step(llm: LLM, state: AgentState, step: PlanStep, previous: list[StepResult]) -> StepResult:
    run_dir = state["run_dir"]
    base_prompt = build_prompt(state, step, previous)
    prompt = base_prompt
    code, error = "", None

    for attempt in range(MAX_RETRIES + 1):  # 1 first try + up to 3 retries
        tokens = {"input_tokens": 0, "output_tokens": 0}
        result = None
        try:
            reply, tokens = llm.complete_json(SYSTEM_PROMPT, prompt, CodeReply)
            code = reply.code
            result = run_code(code, state["csv_path"], run_dir)
            error = result.error if not result.success else check_stdout(result.stdout)
            if error and result.stderr:
                error += f"\n\nstderr (last 2000 chars):\n{result.stderr[-2000:]}"
        except LLMJsonError as e:
            error = f"LLM did not return valid JSON: {e}"

        log_event(
            run_dir, "analyst",
            step_id=step["id"],
            attempt=attempt,
            input_summary=step["description"],
            code=code,
            stdout=result.stdout if result else "",
            stderr=result.stderr if result else "",
            error=error,
            duration_s=round(result.duration_s, 2) if result else 0,
            tokens=tokens,
        )

        if error is None:
            charts = [f for f in result.new_files if f.lower().endswith(".png")]
            return StepResult(
                step_id=step["id"], description=step["description"], code=code,
                stdout=result.stdout.strip(), error=None, retries=attempt,
                success=True, charts=charts,
            )

        prompt = build_retry_prompt(base_prompt, code, error)

    return StepResult(
        step_id=step["id"], description=step["description"], code=code,
        stdout="", error=error, retries=MAX_RETRIES, success=False, charts=[],
    )


def analyst_node(state: AgentState, llm: LLM) -> dict:
    results: list[StepResult] = []
    for step in state["plan"]:
        results.append(run_step(llm, state, step, previous=results))

    charts = [c for r in results for c in r["charts"]]
    # Every attempt (code, stdout, stderr) is already in trace.jsonl; keep only a
    # per-step summary in state so it stays small.
    summary = log_event(
        state["run_dir"], "analyst",
        input_summary=f"{len(state['plan'])} steps",
        output=[{k: r[k] for k in ("step_id", "success", "retries")} for r in results],
    )
    return {
        "step_results": results,
        "charts": charts,
        "trace": state.get("trace", []) + [summary],
    }
