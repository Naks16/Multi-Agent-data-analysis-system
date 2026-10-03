"""Web UI: upload a CSV, ask questions in a chat, and watch the agents work.

    streamlit run app.py

Each question runs the full pipeline (Planner -> Analyst -> Reporter <-> Critic)
from analyst/graph.py. The sidebar shows which agent is running; the answer and
its charts appear in the chat.
"""

from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyst.critic import should_revise
from analyst.graph import stream_analysis
from analyst.llm import LLM, LLMJsonError

load_dotenv()  # ANTHROPIC_API_KEY and MODEL_NAME
UPLOAD_DIR = Path("uploads")

AGENTS = {
    "planner": ("Planner", "Splits the question into analysis steps"),
    "analyst": ("Analyst", "Writes and runs pandas/scipy code for each step"),
    "reporter": ("Reporter", "Writes the findings, citing a step for each"),
    "critic": ("Critic", "Fact-checks the report against the computed numbers"),
}
ICONS = {"waiting": "⚪", "running": "⏳", "done": "✅"}


# --------------------------------------------------------------------------- #
# Agent status panel
# --------------------------------------------------------------------------- #

def fresh_status() -> dict:
    """agent -> (status, note). Status is one of the ICONS keys."""
    return {agent: ("waiting", "") for agent in AGENTS}


def render_agents(box, status: dict) -> None:
    with box.container():
        for agent, (name, role) in AGENTS.items():
            state, note = status[agent]
            st.markdown(f"{ICONS[state]} **{name}**" + (f" — {note}" if note else ""))
            st.caption(role)


def describe(agent: str, state: dict) -> str:
    """One-line summary of what an agent just produced."""
    if agent == "planner":
        return f"planned {len(state['plan'])} steps"
    if agent == "analyst":
        ok = sum(r["success"] for r in state["step_results"])
        return f"{ok}/{len(state['step_results'])} steps succeeded, {len(state.get('charts', []))} charts"
    if agent == "reporter":
        n = state["revision_count"]
        return "wrote the report" if n == 0 else f"revised the report (revision {n})"
    feedback = state["critic_feedback"]
    return "approved" if feedback["approved"] else f"rejected: {len(feedback['issues'])} issue(s)"


def next_agent(agent: str, state: dict) -> str | None:
    """Mirrors the edges in analyst/graph.py, so we can mark the next agent as running."""
    if agent == "critic":
        return "reporter" if should_revise(state) else None
    return {"planner": "analyst", "analyst": "reporter", "reporter": "critic"}[agent]


def run_with_live_status(csv_path: str, question: str, box) -> dict:
    status = fresh_status()
    status["planner"] = ("running", "")
    render_agents(box, status)

    final = None
    for agent, final in stream_analysis(csv_path, question, LLM()):
        status[agent] = ("done", describe(agent, final))
        nxt = next_agent(agent, final)
        if nxt:
            status[nxt] = ("running", status[nxt][1])
        render_agents(box, status)
        st.session_state.agent_status = status  # keep it after the page reruns
    return final


# --------------------------------------------------------------------------- #
# Chat messages
# --------------------------------------------------------------------------- #

def report_for_chat(report: str) -> str:
    """report.md minus the title/question (already in the chat) and the image links
    (they point at local files, so the charts are shown with st.image instead)."""
    lines = []
    for line in report.splitlines():
        if line.startswith(("# Analysis Report", "**Question:**", "![", "## Charts")):
            continue
        lines.append(line.replace("## ", "#### ", 1) if line.startswith("## ") else line)
    return "\n".join(lines).strip()


def show_message(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["error"])
        return
    st.markdown(msg["content"])
    for chart in msg.get("charts", []):
        if Path(chart).is_file():
            st.image(chart)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Multi-Agent Data Analyst", layout="wide")
st.title("Multi-Agent Data Analyst")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("agent_status", fresh_status())
st.session_state.setdefault("csv_path", None)

with st.sidebar:
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    st.subheader("Agents")
    agents_box = st.empty()
render_agents(agents_box, st.session_state.agent_status)

if uploaded is None:
    st.info("Upload a CSV in the sidebar to start.")
    st.stop()

# The backend needs a file path (generated code reads it), so save the upload to disk.
csv_path = str(UPLOAD_DIR / uploaded.name)
if st.session_state.csv_path != csv_path:
    UPLOAD_DIR.mkdir(exist_ok=True)
    Path(csv_path).write_bytes(uploaded.getvalue())
    st.session_state.csv_path = csv_path
    st.session_state.messages = []  # new dataset, new conversation
    st.session_state.agent_status = fresh_status()

with st.expander(f"Preview of {uploaded.name}"):
    st.dataframe(pd.read_csv(csv_path).head(20))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        show_message(msg)

if question := st.chat_input("Ask a question about the data"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("The agents are working (this can take a minute)..."):
                final = run_with_live_status(csv_path, question, agents_box)
            msg = {"role": "assistant", "content": report_for_chat(final["report"]),
                   "charts": final.get("charts", [])}
        except (LLMJsonError, RuntimeError, anthropic.APIError) as e:
            cause = f" (cause: {e.__cause__!r})" if e.__cause__ else ""
            msg = {"role": "assistant", "error": f"Analysis failed: {e}{cause}"}
        show_message(msg)
    st.session_state.messages.append(msg)
