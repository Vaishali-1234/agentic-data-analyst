"""
router.py

The router node: the core "agentic" decision point. Given a user's
question and a summary of the dataset's schema (from schema_inspector),
decides HOW the question should be answered - not a fixed pipeline.

Route options:
- "sql": aggregation, filtering, grouping, joins, counting
- "python": statistics, correlation, forecasting, visualization - things
            SQL can't do well
- "both": SQL narrows down the relevant data first, then Python analyzes
          that subset (e.g. "find the top constructor, then plot their
          points trend over the season")
"""

import json
from langchain_groq import ChatGroq
from src.graph.state import AgentState


ROUTER_SYSTEM_PROMPT = """You are a routing agent for a data analysis system.

Given a user's question and a summary of the available dataset schema \
(tables, columns, types, and known data quality issues), decide HOW the \
question should be answered:

- "sql": if the question needs aggregation, filtering, grouping, joining \
tables, or counting.
- "python": if the question needs statistics (correlation, regression), \
visualization, or trend analysis SQL cannot compute directly.
- "both": if the question needs SQL first to narrow down the relevant \
data, THEN Python for further analysis on that subset.

Respond ONLY with a JSON object, no other text, no markdown fences:
{"route": "sql" | "python" | "both", "reasoning": "<one sentence>"}
"""


def route_question(state: AgentState, llm: ChatGroq = None) -> AgentState:
    """LangGraph node: decides the route for the current question.

    Accepts an optional pre-built `llm` so this function is easy to test
    with a stub/mock instead of hitting the real API every time.
    """
    if llm is None:
        # llama-3.3-70b-versatile is Groq's strong general-purpose free model -
        # good enough for routing decisions and fast (Groq's whole selling point)
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    user_prompt = (
        f"Dataset schema summary:\n{state['schema_summary']}\n\n"
        f"User question: {state['question']}\n\n"
        f"Decide the route."
    )

    response = llm.invoke([
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    route, reasoning = _parse_router_response(response.content)

    trace = state.get("reasoning_trace", []) or []
    trace.append(f"Router chose '{route}': {reasoning}")

    return {
        **state,
        "route": route,
        "route_reasoning": reasoning,
        "reasoning_trace": trace,
    }


def _parse_router_response(raw_content: str) -> tuple:
    """Parse the LLM's JSON response. Falls back to 'sql' + a note if the
    response isn't valid JSON, rather than crashing the whole pipeline -
    a malformed LLM response shouldn't take down the agent."""
    try:
        # Strip markdown code fences if the model added them despite instructions
        cleaned = raw_content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        route = parsed.get("route", "sql")
        reasoning = parsed.get("reasoning", "")
        if route not in ("sql", "python", "both"):
            route = "sql"
            reasoning += " (invalid route value received, defaulted to sql)"
        return route, reasoning
    except (json.JSONDecodeError, AttributeError):
        return "sql", "Could not parse router response as JSON; defaulted to SQL."
