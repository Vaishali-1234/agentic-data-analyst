"""
sql_tool.py

Given a natural-language question and a schema summary, generates a SQL
query (DuckDB dialect) via LLM, executes it against the DuckDB connection,
and returns the result.

Includes ONE self-correction attempt: if the first generated query fails
to execute (syntax error, unknown column, etc.), the actual error message
is fed back to the LLM for a single retry - this is the tool-level piece
of the agent's "notice a failure and retry" behavior, separate from the
whole-pipeline self-check/grounding loop that happens later.
"""

import re
import duckdb
from langchain_groq import ChatGroq
from src.graph.state import AgentState


SQL_SYSTEM_PROMPT = """You are a SQL query generator for DuckDB.

Given a dataset schema summary and a user's question, write ONE DuckDB \
SQL query that answers it.

Rules:
- Respond with ONLY the SQL query - no explanation, no markdown fences.
- Use table and column names EXACTLY as given in the schema summary.
- If a column is flagged as "numeric-looking but stored as TEXT" or has \
missing markers like '\\N', account for this (e.g. TRY_CAST, filter out \
non-numeric values, or prefer a clean numeric alternative column if one \
is mentioned in the schema summary).
- Add a reasonable LIMIT on exploratory queries that could return many \
rows, unless the question clearly wants a full aggregate.
- IMPORTANT: some tables store CUMULATIVE running totals (e.g. a \
"standings" table where a stat like wins/points reflects the total AS \
OF that point in time, not a per-event value). NEVER use SUM() on a \
cumulative column across multiple rows - this double-counts. To count \
individual events (e.g. "how many races did X win"), prefer counting \
raw event-level rows directly (e.g. COUNT rows where a result/position \
column indicates a win) rather than summing a cumulative snapshot \
column. If unsure whether a column is cumulative, prefer the more \
granular, per-event table over a "standings"/"summary"-style table.
- If a column is flagged as quirky (text-stored numbers, missing-value \
markers) AND another column in the SAME table represents similar \
information cleanly (e.g. a numeric column with no flags and a similar \
name/purpose), PREFER the clean column over casting the quirky one.
- DuckDB CAST syntax uses the AS keyword: TRY_CAST(expr AS INTEGER). \
Do NOT use a comma, e.g. TRY_CAST(expr, INTEGER) is INVALID and will \
fail - this is not the correct syntax in DuckDB.
"""


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and stray text the LLM might add despite
    instructions - LLMs wrap output in ```sql fences surprisingly often."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip().rstrip(";")


def generate_sql(question: str, schema_summary: str, llm, error_context: str = None) -> str:
    """Ask the LLM to generate a SQL query. If error_context is given,
    this is a self-correction attempt - the LLM sees its own mistake."""
    user_prompt = f"Dataset schema summary:\n{schema_summary}\n\nQuestion: {question}"
    if error_context:
        user_prompt += (
            f"\n\nYour previous query failed with this error:\n{error_context}"
            f"\nFix the query."
        )

    response = llm.invoke([
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return _clean_sql(response.content)


def run_sql_tool(state: AgentState, con: duckdb.DuckDBPyConnection, llm: ChatGroq = None) -> AgentState:
    """LangGraph node: generates and executes a SQL query for the question."""
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    trace = state.get("reasoning_trace", []) or []
    sql_query = generate_sql(state["question"], state["schema_summary"], llm)

    try:
        result_df = con.execute(sql_query).fetchdf()
        trace.append(f"SQL tool ran: {sql_query}")
    except Exception as e:
        trace.append(f"SQL tool: first query failed ({e}) - retrying with error feedback")
        try:
            sql_query = generate_sql(
                state["question"], state["schema_summary"], llm, error_context=str(e)
            )
            result_df = con.execute(sql_query).fetchdf()
            trace.append(f"SQL tool retry succeeded: {sql_query}")
        except Exception as e2:
            trace.append(f"SQL tool: retry also failed ({e2})")
            return {
                **state,
                "sql_query": sql_query,
                "sql_result": f"SQL execution failed after retry: {e2}",
                "reasoning_trace": trace,
            }

    # Compact text form for passing to the interpreter node later -
    # the full dataframe isn't needed, a readable string is
    result_str = result_df.to_string(index=False, max_rows=20)

    return {
        **state,
        "sql_query": sql_query,
        "sql_result": result_str,
        "reasoning_trace": trace,
    }
