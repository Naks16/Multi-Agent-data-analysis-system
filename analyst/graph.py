"""Wires the agents into a LangGraph:

    START -> planner -> analyst -> reporter -> critic --approved / out of revisions--> END
                                      ^           |
                                      +-rejected--+
"""

from langgraph.graph import END, START, StateGraph

from analyst.analyst import analyst_node
from analyst.critic import critic_node, should_revise
from analyst.llm import LLM
from analyst.planner import planner_node
from analyst.reporter import reporter_node
from analyst.state import AgentState
from analyst.trace import new_run_dir


def route_after_critic(state: AgentState) -> str:
    return "revise" if should_revise(state) else "done"


def build_graph(llm: LLM):
    graph = StateGraph(AgentState)
    # Each node receives the state; the lambdas pass in the shared LLM client.
    graph.add_node("planner", lambda state: planner_node(state, llm))
    graph.add_node("analyst", lambda state: analyst_node(state, llm))
    graph.add_node("reporter", lambda state: reporter_node(state, llm))
    graph.add_node("critic", lambda state: critic_node(state, llm))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "analyst")
    graph.add_edge("analyst", "reporter")
    graph.add_edge("reporter", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"revise": "reporter", "done": END})
    return graph.compile()


def stream_analysis(csv_path: str, question: str, llm: LLM, runs_dir: str = "runs"):
    """Run the pipeline, yielding (agent_name, state_so_far) each time an agent finishes.

    Used by the web UI to show live progress. Output goes to runs/<timestamp>/.
    """
    state: AgentState = {
        "csv_path": str(csv_path),
        "question": question,
        "run_dir": str(new_run_dir(runs_dir)),
        "revision_count": 0,
        "trace": [],
    }
    # stream_mode="updates" yields {node_name: dict_the_node_returned} after each node.
    for update in build_graph(llm).stream(state, stream_mode="updates"):
        for agent, changes in update.items():
            state = {**state, **changes}  # same merge LangGraph does (no custom reducers)
            yield agent, state


def run_analysis(csv_path: str, question: str, llm: LLM, runs_dir: str = "runs") -> AgentState:
    """Run the whole pipeline and return the final state."""
    state = None
    for _, state in stream_analysis(csv_path, question, llm, runs_dir):
        pass
    return state
