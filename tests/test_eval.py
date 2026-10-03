"""The eval harness: scoring rules, the baseline, and one full run_one per system (fake LLM)."""

import pandas as pd

from eval.baseline import run_baseline
from eval.cases import CASES, Case, NumberCheck, number_matches, numbers_in, score
from eval.run_eval import answer_part, render_results, run_one
from fakes import FakeLLM
from test_graph import APPROVE, DRAFT, scripted_llm, tiny_csv  # noqa: F401 (tiny_csv is a fixture)


def test_numbers_in_handles_commas_currency_and_percent():
    assert numbers_in("Revenue $1,234.50, churn 27.5%, r=0.83") == [1234.5, 27.5, 0.83]


def test_number_matches_tolerance_and_percent_form():
    assert number_matches("about 174.3 dollars", 174.34, 0.02, is_rate=False)
    assert not number_matches("about 150 dollars", 174.34, 0.02, is_rate=False)
    assert number_matches("Basic churns 27.5% of customers", 0.275, 0.02, is_rate=True)
    assert number_matches("Basic churn rate is 0.28", 0.275, 0.02, is_rate=True)


def test_null_case_accepts_no_change_and_rejects_a_claimed_effect():
    null_case = next(c for c in CASES if c.id == "redesign_null")
    assert score(null_case, "The change is not statistically significant (p=0.97).") == {"says no real change": True}
    assert score(null_case, "The redesign increased conversion by 12%.") == {"says no real change": False}


def test_number_check_uses_value_computed_from_csv(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"x": [10.0, 20.0]}).to_csv(path, index=False)
    case = Case("t", path, "q?", keywords={}, numbers=[NumberCheck("mean x", lambda df: df.x.mean())])
    assert score(case, "the mean is 15") == {"mean x": True}
    assert score(case, "the mean is 20") == {"mean x": False}


def test_answer_part_drops_charts_and_limitations():
    report = "# R\n\n## Short answer\n\nWest fell.\n\n## Charts\n\n![a](a.png)\n\n## Limitations\n\n- West x\n"
    assert answer_part(report) == "## Short answer\n\nWest fell.\n\n"


def test_baseline_sends_question_and_full_csv(tiny_csv):  # noqa: F811
    llm = FakeLLM(["West revenue fell in March."])
    answer, tokens = run_baseline(tiny_csv, "Why?", llm)
    assert answer == "West revenue fell in March."
    assert tokens == {"input_tokens": 10, "output_tokens": 5}
    assert "Question: Why?" in llm.prompts[0]
    assert tiny_csv.read_text() in llm.prompts[0]


def test_run_one_scores_both_systems(tiny_csv, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.chdir(tmp_path)  # the pipeline writes runs/ into the current folder
    case = Case("tiny", tiny_csv, "Why did revenue drop in March?", keywords={"names West": [r"\bwest\b"]})

    pipeline = run_one("pipeline", case, scripted_llm(DRAFT, APPROVE))
    assert pipeline["passed"] and pipeline["critic_approved"]
    assert pipeline["steps_ok"] == pipeline["steps_total"] == 3
    assert pipeline["tokens"]["input_tokens"] > 0  # summed from trace.jsonl

    baseline = run_one("baseline", case, FakeLLM(["Revenue fell everywhere."]))
    assert not baseline["passed"] and baseline["checks"] == {"names West": False}


class RefusingLLM(FakeLLM):
    def complete(self, system, prompt):
        raise RuntimeError("The model declined this request")


def test_run_one_records_a_crash_as_failed(tiny_csv):  # noqa: F811
    case = Case("tiny", tiny_csv, "Why?", keywords={"names West": [r"\bwest\b"]})
    result = run_one("baseline", case, RefusingLLM([]))
    assert not result["passed"]
    assert result["error"].startswith("RuntimeError")


def test_render_results_counts_passes():
    case = CASES[0]
    run = lambda ok: {"passed": ok, "checks": {"names West region": ok}, "error": None, "seconds": 2.0,  # noqa: E731
                      "tokens": {"input_tokens": 100, "output_tokens": 10},
                      "critic_approved": True, "steps_ok": 3, "steps_total": 3}
    md = render_results([case], {case.id: {"pipeline": [run(True)], "baseline": [run(False)]}}, 1, "m")
    assert "| pipeline | 1/1 | 1/1 |" in md
    assert "| baseline | 0/1 | 0/1 |" in md
    assert "missed: names West region" in md
