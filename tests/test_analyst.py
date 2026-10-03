import json

import pandas as pd
import pytest

from analyst.analyst import MAX_RETRIES, analyst_node, run_step
from fakes import FakeLLM

STEP = {"id": "s1", "description": "Total revenue", "output_type": "number",
        "is_statistical_test": False}

GOOD_CODE = (
    "import json\nimport pandas as pd\n"
    "df = pd.read_csv(CSV_PATH)\n"
    "print(json.dumps({'total_revenue': int(df['revenue'].sum())}))"
)
CHART_CODE = (
    "import json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
    "df = pd.read_csv(CSV_PATH)\n"
    "df.plot(y='revenue')\nplt.savefig(f'{OUTPUT_DIR}/s1_revenue.png')\nplt.close()\n"
    "print(json.dumps({'rows': len(df)}))"
)


def code_reply(code: str) -> str:
    return json.dumps({"code": code})


@pytest.fixture
def state(tmp_path):
    csv = tmp_path / "tiny.csv"
    pd.DataFrame({"revenue": [100, 200, 300]}).to_csv(csv, index=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return {"csv_path": str(csv), "question": "What is total revenue?",
            "schema_summary": "revenue: int", "run_dir": str(run_dir), "trace": []}


def test_step_succeeds_first_time(state):
    result = run_step(FakeLLM([code_reply(GOOD_CODE)]), state, STEP, previous=[])
    assert result["success"]
    assert result["retries"] == 0
    assert json.loads(result["stdout"]) == {"total_revenue": 600}
    assert result["code"] == GOOD_CODE


def test_error_is_fed_back_and_step_retries(state):
    broken = "import pandas as pd\npd.read_csv(CSV_PATH)['no_such_column']"
    llm = FakeLLM([code_reply(broken), code_reply(GOOD_CODE)])
    result = run_step(llm, state, STEP, previous=[])
    assert result["success"]
    assert result["retries"] == 1
    assert "KeyError" in llm.prompts[1]   # the error reached the LLM


def test_blocked_code_counts_as_failure_and_is_retried(state):
    llm = FakeLLM([code_reply("import os\nprint('{}')"), code_reply(GOOD_CODE)])
    result = run_step(llm, state, STEP, previous=[])
    assert result["success"] and result["retries"] == 1
    assert "safety check" in llm.prompts[1]


def test_non_json_stdout_is_retried(state):
    llm = FakeLLM([code_reply("print('total is 600')"), code_reply(GOOD_CODE)])
    result = run_step(llm, state, STEP, previous=[])
    assert result["success"] and result["retries"] == 1


def test_free_text_in_output_is_rejected_and_retried(state):
    with_note = (
        "import json\nimport pandas as pd\ndf = pd.read_csv(CSV_PATH)\n"
        "print(json.dumps({'total': int(df['revenue'].sum()),"
        " 'note': 'Cash tips are not recorded by the meter so ignore them'}))"
    )
    llm = FakeLLM([code_reply(with_note), code_reply(GOOD_CODE)])
    result = run_step(llm, state, STEP, previous=[])
    assert result["success"] and result["retries"] == 1
    assert "free text" in llm.prompts[1]
    assert "note" not in result["stdout"]


def test_short_labels_are_allowed(state):
    labels = (
        "import json\n"
        "print(json.dumps({'test': 'Mann-Whitney U (two-sided)', 'zone': 'Upper West Side South',"
        " 'groups': ['credit card', 'cash'], 'p_value': 0.01}))"
    )
    result = run_step(FakeLLM([code_reply(labels)]), state, STEP, previous=[])
    assert result["success"] and result["retries"] == 0


def test_step_marked_failed_after_max_retries(state):
    broken = code_reply("raise ValueError('boom')")
    llm = FakeLLM([broken] * (MAX_RETRIES + 1))
    result = run_step(llm, state, STEP, previous=[])
    assert not result["success"]
    assert result["retries"] == MAX_RETRIES
    assert "boom" in result["error"]
    assert llm.replies == []  # exactly 1 + MAX_RETRIES attempts were made


def test_node_continues_after_failed_step_and_collects_charts(state):
    state["plan"] = [STEP, {**STEP, "id": "s2"}]
    replies = [code_reply("raise ValueError('boom')")] * (MAX_RETRIES + 1) + [code_reply(CHART_CODE)]
    update = analyst_node(state, FakeLLM(replies))

    s1, s2 = update["step_results"]
    assert not s1["success"] and s2["success"]
    assert len(update["charts"]) == 1 and update["charts"][0].endswith("s1_revenue.png")

    lines = (open(f"{state['run_dir']}/trace.jsonl").read().strip().splitlines())
    attempts = [json.loads(l) for l in lines if json.loads(l).get("attempt") is not None]
    assert len(attempts) == MAX_RETRIES + 2   # 4 attempts for s1, 1 for s2
    assert all("code" in a and "tokens" in a for a in attempts)
