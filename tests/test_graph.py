"""End-to-end: the whole LangGraph pipeline on a tiny CSV with a scripted (fake) LLM.

The executor is real, so the generated code actually runs.
"""

import json

import pandas as pd
import pytest

from analyst.graph import run_analysis, stream_analysis
from fakes import FakeLLM

QUESTION = "Why did revenue drop in March?"

PLAN = {"steps": [
    {"id": "s1", "description": "Total revenue per month", "output_type": "table"},
    {"id": "s2", "description": "Revenue per month by region as a chart", "output_type": "chart"},
    {"id": "s3", "description": "Mann-Whitney U test: March revenue vs Jan-Feb revenue",
     "output_type": "number", "is_statistical_test": True},
]}

S1_CODE = """import json
import pandas as pd
df = pd.read_csv(CSV_PATH)
print(json.dumps({"revenue_by_month": df.groupby("month")["revenue"].sum().to_dict()}))"""

S2_CODE = """import json
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(CSV_PATH)
df.pivot_table(index="month", columns="region", values="revenue", aggfunc="sum").plot()
plt.title("Revenue by region"); plt.xlabel("month"); plt.ylabel("revenue")
plt.savefig(f"{OUTPUT_DIR}/s2_region.png"); plt.close()
print(json.dumps({"chart": "s2_region.png"}))"""

S3_CODE = """import json
import pandas as pd
from scipy import stats
df = pd.read_csv(CSV_PATH)
march, before = df[df.month == 3]["revenue"], df[df.month < 3]["revenue"]
res = stats.mannwhitneyu(march, before)
print(json.dumps({"test": "Mann-Whitney U", "statistic": float(res.statistic),
                  "p_value": float(res.pvalue)}))"""

DRAFT = {
    "short_answer": "Revenue fell in March, driven by the West region.",
    "key_findings": [{"text": "March revenue was lower than Jan and Feb.", "step_id": "s1"}],
    "statistical_tests": [{"step_id": "s3", "test_name": "Mann-Whitney U", "p_value": 0.03,
                           "interpretation": "March differs from Jan-Feb at the 5% level."}],
    "limitations": ["Only three months of data."],
}
REJECT = {"approved": False, "issues": ["Unsupported: 'driven by the West region' - no step shows this"]}
APPROVE = {"approved": True, "issues": []}


@pytest.fixture
def tiny_csv(tmp_path):
    rows = []
    for month in (1, 2, 3):
        for day in range(1, 6):
            rows.append({"month": month, "region": "North", "revenue": 100 + day})
            rows.append({"month": month, "region": "West", "revenue": (20 if month == 3 else 120) + day})
    path = tmp_path / "tiny.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def scripted_llm(*reporter_critic_replies) -> FakeLLM:
    replies = [json.dumps(PLAN)] + [json.dumps({"code": c}) for c in (S1_CODE, S2_CODE, S3_CODE)]
    return FakeLLM(replies + [json.dumps(r) for r in reporter_critic_replies])


def read_trace(run_dir) -> list[dict]:
    with open(f"{run_dir}/trace.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_full_run_with_one_revision_reaches_end(tiny_csv, tmp_path):
    llm = scripted_llm(DRAFT, REJECT, DRAFT, APPROVE)
    final = run_analysis(tiny_csv, QUESTION, llm, runs_dir=tmp_path / "runs")

    assert llm.replies == []                      # every scripted reply was used
    assert final["critic_feedback"]["approved"]
    assert final["revision_count"] == 1
    assert all(r["success"] for r in final["step_results"])
    assert "Unsupported: 'driven by the West region'" in llm.prompts[-2]  # issues reached the reporter

    report = open(f"{final['run_dir']}/report.md", encoding="utf-8").read()
    for section in ("## Short answer", "## Key findings", "## Statistical tests",
                    "## Charts", "## Limitations"):
        assert section in report
    assert "*(source: step s1)*" in report
    assert "![s2_region](s2_region.png)" in report
    assert "| Mann-Whitney U | s3 | 0.03 | Yes |" in report

    agents = [e["agent"] for e in read_trace(final["run_dir"])]
    assert agents[0] == "planner"
    assert agents[-4:] == ["reporter", "critic", "reporter", "critic"]


def test_stream_yields_each_agent_in_order(tiny_csv, tmp_path):
    llm = scripted_llm(DRAFT, REJECT, DRAFT, APPROVE)
    events = list(stream_analysis(tiny_csv, QUESTION, llm, runs_dir=tmp_path / "runs"))

    assert [agent for agent, _ in events] == [
        "planner", "analyst", "reporter", "critic", "reporter", "critic"]
    _, final = events[-1]
    assert final["critic_feedback"]["approved"]
    assert len(final["plan"]) == 3                 # earlier updates are kept in the merged state


def test_stops_after_two_revisions_and_adds_caveats(tiny_csv, tmp_path):
    llm = scripted_llm(DRAFT, REJECT, DRAFT, REJECT, DRAFT, REJECT)
    final = run_analysis(tiny_csv, QUESTION, llm, runs_dir=tmp_path / "runs")

    assert llm.replies == []
    assert not final["critic_feedback"]["approved"]
    assert final["revision_count"] == 2
    report = open(f"{final['run_dir']}/report.md", encoding="utf-8").read()
    assert "## Unresolved reviewer caveats" in report
    assert "driven by the West region' - no step shows this" in report
