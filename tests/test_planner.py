import json

import pandas as pd
import pytest

from analyst.llm import LLMJsonError
from analyst.planner import make_plan, needs_statistical_test, planner_node, summarize_csv
from fakes import FakeLLM

QUESTION = "Why did sales drop in March?"

GOOD_PLAN = {"steps": [
    {"id": "s1", "description": "Total revenue per month", "output_type": "table"},
    {"id": "s2", "description": "Revenue per month by region", "output_type": "chart"},
    {"id": "s3", "description": "Mann-Whitney U test of March daily revenue vs Jan-Feb",
     "output_type": "number", "is_statistical_test": True},
]}


@pytest.fixture
def tiny_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({
        "date": [f"2025-0{m}-01" for m in (1, 2, 3)] * 2,
        "region": ["North"] * 3 + ["West"] * 3,
        "revenue": [100, 105, 98, 120, 118, 20],
    }).to_csv(path, index=False)
    return path


def test_valid_json_parses_into_steps():
    llm = FakeLLM([json.dumps(GOOD_PLAN)])
    plan, usage = make_plan(llm, "schema", QUESTION)
    assert [s.id for s in plan.steps] == ["s1", "s2", "s3"]
    assert plan.steps[1].output_type == "chart"
    assert usage == {"input_tokens": 10, "output_tokens": 5}


def test_json_inside_code_fence_and_prose_is_accepted():
    llm = FakeLLM([f"Here is the plan:\n```json\n{json.dumps(GOOD_PLAN)}\n```"])
    plan, _ = make_plan(llm, "schema", QUESTION)
    assert len(plan.steps) == 3


def test_invalid_json_gets_one_repair_retry():
    llm = FakeLLM(["{not json", json.dumps(GOOD_PLAN)])
    plan, usage = make_plan(llm, "schema", QUESTION)
    assert len(plan.steps) == 3
    assert "rejected because" in llm.prompts[1]          # error was fed back
    assert usage["input_tokens"] == 20                    # both calls counted


@pytest.mark.parametrize("bad_plan", [
    {"steps": GOOD_PLAN["steps"][:2]},                                     # too few steps
    {"steps": [{**s, "output_type": "pie"} for s in GOOD_PLAN["steps"]]},  # bad enum
    {"steps": [{**s, "id": "s1"} for s in GOOD_PLAN["steps"]]},            # duplicate ids
])
def test_schema_violations_raise_after_repair(bad_plan):
    llm = FakeLLM([json.dumps(bad_plan), json.dumps(bad_plan)])
    with pytest.raises(LLMJsonError):
        make_plan(llm, "schema", QUESTION)
    assert len(llm.prompts) == 2  # exactly one repair retry


def test_change_question_requires_a_statistical_test():
    no_test = {"steps": [{**s, "is_statistical_test": False} for s in GOOD_PLAN["steps"]]}
    llm = FakeLLM([json.dumps(no_test), json.dumps(GOOD_PLAN)])
    plan, _ = make_plan(llm, "schema", QUESTION)
    assert any(s.is_statistical_test for s in plan.steps)
    assert "statistical test" in llm.prompts[1]


def test_needs_statistical_test():
    assert needs_statistical_test("Why did sales drop in March?")
    assert needs_statistical_test("Compare North vs South")
    assert not needs_statistical_test("What are the top 5 products by revenue?")


def test_summary_does_not_contain_full_data(tmp_path):
    path = tmp_path / "big.csv"
    pd.DataFrame({"x": range(1000), "secret_marker": [f"row{i}" for i in range(1000)]}).to_csv(path, index=False)
    summary = summarize_csv(path)
    assert "row0" in summary          # head(5) is included
    assert "row999" not in summary    # but not the rest


def test_planner_node_updates_state_and_writes_trace(tiny_csv, tmp_path):
    llm = FakeLLM([json.dumps(GOOD_PLAN)])
    state = {"csv_path": str(tiny_csv), "question": QUESTION, "run_dir": str(tmp_path), "trace": []}
    update = planner_node(state, llm)
    assert len(update["plan"]) == 3
    assert "Distinct values of 'region'" in update["schema_summary"]
    assert update["trace"][0]["agent"] == "planner"
    assert (tmp_path / "trace.jsonl").exists()
