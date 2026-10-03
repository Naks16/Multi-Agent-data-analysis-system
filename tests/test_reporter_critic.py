import json

import pandas as pd
import pytest

from analyst.critic import CriticVerdict, critic_node, should_revise
from analyst.reporter import reporter_node
from fakes import FakeLLM

OK_STEP = {"step_id": "s1", "description": "Total revenue", "code": "", "stdout": '{"total": 600}',
           "error": None, "retries": 0, "success": True, "charts": []}
FAILED_STEP = {"step_id": "s2", "description": "Revenue by region", "code": "",
               "stdout": "", "error": "Exit code 1: KeyError: 'region'", "retries": 3,
               "success": False, "charts": []}
DRAFT = {"short_answer": "Total revenue is 600.",
         "key_findings": [{"text": "Total revenue is 600.", "step_id": "s1"}]}


@pytest.fixture
def state(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"revenue": [100, None, 500]}).to_csv(csv, index=False)
    return {"csv_path": str(csv), "question": "What is total revenue?", "run_dir": str(tmp_path),
            "step_results": [OK_STEP, FAILED_STEP], "charts": [], "trace": []}


def test_report_lists_failed_steps_and_missing_values(state):
    update = reporter_node(state, FakeLLM([json.dumps(DRAFT)]))
    report = update["report"]
    assert 'Step s2 ("Revenue by region") failed after 3 retries' in report
    assert "Column `revenue` has 1 missing values." in report
    assert "does not prove causation" in report
    assert update["revision_count"] == 0


def test_duplicate_limitations_are_removed(state):
    draft = {**DRAFT, "limitations": [
        "This analysis shows associations in the data; it does not prove causation."]}
    report = reporter_node(state, FakeLLM([json.dumps(draft)]))["report"]
    assert report.count("does not prove causation") == 1


def test_reporter_cannot_cite_a_failed_step(state):
    bad = {**DRAFT, "key_findings": [{"text": "West is down 80%.", "step_id": "s2"}]}
    llm = FakeLLM([json.dumps(bad), json.dumps(DRAFT)])
    update = reporter_node(state, llm)
    assert "unknown or failed" in llm.prompts[1]
    assert "s2)*" not in update["report"]


def test_revision_increments_count(state):
    state.update(report="old", revision_count=0,
                 critic_feedback={"approved": False, "issues": ["bad number"]})
    llm = FakeLLM([json.dumps(DRAFT)])
    update = reporter_node(state, llm)
    assert update["revision_count"] == 1
    assert "bad number" in llm.prompts[0]


def test_verdict_must_be_consistent():
    with pytest.raises(ValueError):
        CriticVerdict(approved=True, issues=["something"])
    with pytest.raises(ValueError):
        CriticVerdict(approved=False, issues=[])


def test_should_revise():
    rejected = {"approved": False, "issues": ["x"]}
    assert should_revise({"critic_feedback": rejected, "revision_count": 1})
    assert not should_revise({"critic_feedback": rejected, "revision_count": 2})
    assert not should_revise({"critic_feedback": {"approved": True, "issues": []}, "revision_count": 0})


def test_critic_sees_evidence_and_report(state):
    state["report"] = "# Report\nTotal revenue is 600."
    llm = FakeLLM([json.dumps({"approved": True, "issues": []})])
    update = critic_node(state, llm)
    assert update["critic_feedback"]["approved"]
    assert '{"total": 600}' in llm.prompts[0]          # successful output is evidence
    assert "KeyError" not in llm.prompts[0]            # failed step is not
    assert "Total revenue is 600." in llm.prompts[0]
