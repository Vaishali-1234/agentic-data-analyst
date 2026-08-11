"""
state.py

Defines the shared state that flows through every node in the LangGraph
agent. Each node reads from and writes back to this state.
"""

from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    # Input
    question: str                    # the user's natural-language question
    schema_summary: str              # output of schema_inspector, describes available tables/columns

    # Conversation memory (for multi-turn follow-ups, added later)
    conversation_history: List[dict]

    # Routing decision
    route: Optional[str]             # "sql" | "python" | "both"
    route_reasoning: Optional[str]   # why the router chose this path

    # Tool outputs
    sql_query: Optional[str]
    sql_result: Optional[str]
    python_code: Optional[str]
    python_result: Optional[str]

    # Final output
    final_answer: Optional[str]
    reasoning_trace: List[str]       # human-readable log of steps taken, for UI display

    # Self-check / retry loop
    retry_count: int
    grounded: Optional[bool]
    grounding_issue: Optional[str]   # why grounding failed, fed back into the retry         # did the self-check pass?
