"""
build_graph.py

Assembles the full LangGraph agent: router -> SQL tool -> grounding check,
including the retry loop - if grounding fails, the specific issue is fed
back into the SQL tool for another attempt, capped at MAX_GROUNDING_RETRIES
so a persistently wrong answer doesn't loop forever.

This is what turns three independently-tested functions into one actual
agent that runs automatically end to end.
"""

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from src.graph.state import AgentState
from src.graph.router import route_question
from src.graph.grounding_check import check_grounding
from src.tools.sql_tool import run_sql_tool

MAX_GROUNDING_RETRIES = 2


def _route_after_router(state: AgentState) -> str:
    """Python/'both' tool nodes aren't built yet - fall back to a
    'not yet implemented' node rather than crashing, so the graph is
    runnable end to end today even before every route is complete."""
    return "sql_tool" if state.get("route") == "sql" else "not_implemented"


def _route_after_grounding(state: AgentState) -> str:
    """Accept the result, retry with feedback, or give up after the cap."""
    if state.get("grounded"):
        return "finalize"
    if state.get("retry_count", 0) >= MAX_GROUNDING_RETRIES:
        return "finalize"
    return "retry_sql"


def _not_implemented_node(state: AgentState) -> AgentState:
    trace = state.get("reasoning_trace", []) or []
    trace.append("Route required Python/pandas execution - not yet implemented.")
    return {
        **state,
        "final_answer": "This question needs Python-based analysis, which isn't built yet.",
        "reasoning_trace": trace,
    }


def _finalize_node(state: AgentState) -> AgentState:
    """Produces the final answer, adding an honest caveat if we're giving
    up after exhausting retries without ever reaching a grounded result -
    better to flag uncertainty than present a possibly-wrong answer as fact."""
    trace = state.get("reasoning_trace", []) or []
    answer = state.get("sql_result") or state.get("python_result") or "No answer could be produced."

    if not state.get("grounded") and state.get("retry_count", 0) >= MAX_GROUNDING_RETRIES:
        answer += (
            f"\n\n(Note: this result could not be fully verified after "
            f"{MAX_GROUNDING_RETRIES} attempts - treat it with caution.)"
        )
        trace.append("Finalized with an unverified-result caveat after exhausting retries.")
    else:
        trace.append("Finalized answer.")

    return {**state, "final_answer": answer, "reasoning_trace": trace}


def build_graph(con, llm: ChatGroq = None):
    """Build and compile the LangGraph agent.

    `con` is the DuckDB connection to query against - passed in via
    closure since it's not (and shouldn't be) part of the serializable
    agent state. `llm` can be shared across nodes for consistency /
    efficiency; if omitted, a default Groq client is created.
    """
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    def sql_node(state: AgentState) -> AgentState:
        # If we're back here after a failed grounding check, pass that
        # specific issue in as feedback rather than blindly retrying
        feedback = state.get("grounding_issue") if state.get("grounded") is False else None
        result = run_sql_tool(state, con, llm, external_feedback=feedback)
        bumped_retry = state.get("retry_count", 0) + (1 if feedback else 0)
        return {**result, "retry_count": bumped_retry}

    graph = StateGraph(AgentState)
    graph.add_node("router", lambda state: route_question(state, llm))
    graph.add_node("sql_tool", sql_node)
    graph.add_node("not_implemented", _not_implemented_node)
    graph.add_node("grounding_check", lambda state: check_grounding(state, llm))
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", _route_after_router, {
        "sql_tool": "sql_tool",
        "not_implemented": "not_implemented",
    })
    graph.add_edge("sql_tool", "grounding_check")
    graph.add_conditional_edges("grounding_check", _route_after_grounding, {
        "finalize": "finalize",
        "retry_sql": "sql_tool",
    })
    graph.add_edge("not_implemented", END)
    graph.add_edge("finalize", END)

    return graph.compile()
