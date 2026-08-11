"""
grounding_check.py

The self-check node. Rather than trying to prevent every possible SQL
mistake through prompt engineering alone (which has diminishing returns -
proven by real bugs found during this project's development, e.g. an LLM
reverting to SUMming a cumulative 'standings' column despite explicit
instructions not to), this node verifies the result AFTER generation.

Given the question, the SQL query used, and the result, it asks the LLM
to sanity-check: does this result actually, plausibly answer the
question? This catches semantically-wrong-but-syntactically-valid
queries - the most dangerous failure mode, since they don't error out.
"""

import json
import re
from langchain_groq import ChatGroq
from src.graph.state import AgentState


GROUNDING_SYSTEM_PROMPT = """You are a fact-checker reviewing a data \
analysis result before it's shown to a user.

You will be given: a question, the SQL query used to answer it, and the \
result returned. Decide if the result is PLAUSIBLE and actually answers \
the question - not just syntactically valid, but semantically correct.

Common failure patterns to watch for:
- Summing a CUMULATIVE column (e.g. from a "standings" table, where a \
value like wins/points represents a running total as of that point in \
time) across multiple rows - this double-counts and produces impossibly \
large numbers.
- Using a raw count/sum where a MAX, final value, or distinct count was \
actually needed.
- A result that is numerically implausible for the domain (e.g. a sports \
team "winning" more games than were played in a season).
- Ignoring a filter the question implied (e.g. a specific year, driver, \
or category).

Respond ONLY with a JSON object, no other text:
{"grounded": true/false, "issue": "<if not grounded, explain exactly what's wrong, else empty string>"}
"""


def check_grounding(state: AgentState, llm: ChatGroq = None) -> AgentState:
    """LangGraph node: sanity-checks the SQL result against the question."""
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    user_prompt = (
        f"Question: {state['question']}\n\n"
        f"SQL query used: {state.get('sql_query', 'N/A')}\n\n"
        f"Result: {state.get('sql_result', 'N/A')}\n\n"
        f"Is this result grounded and plausible?"
    )

    response = llm.invoke([
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    grounded, issue = _parse_grounding_response(response.content)

    trace = state.get("reasoning_trace", []) or []
    if grounded:
        trace.append("Grounding check: PASSED - result appears plausible.")
    else:
        trace.append(f"Grounding check: FAILED - {issue}")

    return {
        **state,
        "grounded": grounded,
        "reasoning_trace": trace,
        # store the issue so a retry step can use it as error_context, same
        # pattern as the SQL tool's own self-correction retry
        "sql_result": state.get("sql_result"),
        "route_reasoning": state.get("route_reasoning"),
        "grounding_issue": issue,
    }


def _parse_grounding_response(raw_content: str) -> tuple:
    """Parse the LLM's grounding verdict. Fails safe: if we truly can't
    parse the response, default to grounded=False.

    Uses a regex search for the JSON object rather than only stripping
    known markdown fence patterns - real Groq/Llama responses sometimes
    add commentary before/after the JSON despite instructions not to,
    which the older prefix/suffix-only stripping missed entirely."""
    try:
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            return False, "Could not parse grounding check response (no JSON object found)."
        parsed = json.loads(match.group(0))
        return bool(parsed.get("grounded", False)), parsed.get("issue", "")
    except (json.JSONDecodeError, AttributeError):
        return False, "Could not parse grounding check response."
