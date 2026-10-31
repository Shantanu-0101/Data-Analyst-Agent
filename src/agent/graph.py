"""The LangGraph StateGraph: plan -> generate -> execute -> (fix -> execute)* -> END."""
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import RetryCallback, execute_node, fix_node, generate_node, plan_node
from src.agent.state import AgentState
from src.config import MAX_RETRIES

MAX_SUMMARY_CHARS = 200  # result summaries go into later prompts, so keep them short


def should_retry(state: AgentState) -> str:
    if state["error"] and state["retries"] < MAX_RETRIES:
        return "fix"
    return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("fix", fix_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", should_retry, {"fix": "fix", "end": END})
    graph.add_edge("fix", "execute")
    return graph.compile()


def ask(
    graph,
    df: pd.DataFrame,
    schema_info: str,
    question: str,
    history: Optional[list[dict]] = None,
    *,
    llm: Optional[BaseChatModel] = None,
    on_retry: Optional[RetryCallback] = None,
) -> AgentState:
    """Answer one question. The single entry point for the UI and the walkthroughs.

    The returned state's `history` is the given history plus this turn (the given list is
    not changed), ready to pass to the next call. Raises nodes.LLMUnavailable when the LLM
    cannot be reached; errors in the generated code come back in `error` instead.
    `llm` replaces the Groq/Gemini client (for tests); `on_retry` is called before each wait
    on a rate limit.
    """
    history = list(history or [])
    state: AgentState = {
        "df": df,
        "schema_info": schema_info,
        "question": question,
        "plan": "",
        "generated_code": "",
        "result": None,
        "result_type": None,
        "error": None,
        "retries": 0,
        "history": history,
    }
    configurable = {key: value for key, value in (("llm", llm), ("on_retry", on_retry)) if value is not None}
    final = graph.invoke(state, config={"configurable": configurable})
    final["history"] = history + [{
        "question": question,
        "code": final["generated_code"],
        "result_summary": summarize_result(final),
    }]
    return final


def summarize_result(state: AgentState) -> str:
    """A one-line description of the answer, so follow-up prompts know what was shown."""
    if state["error"]:
        last_line = state["error"].strip().splitlines()[-1] if state["error"].strip() else "unknown error"
        return _short(f"failed: {last_line}")
    result, kind = state["result"], state["result_type"]
    if kind == "dataframe":
        columns = ", ".join(str(col) for col in result.columns)
        return _short(f"table, {len(result)} rows; columns: {columns}")
    if kind == "plotly":
        chart = result.data[0].type if result.data else "empty"
        title = result.layout.title.text if isinstance(result, go.Figure) else None
        return _short(f"{chart} chart" + (f" titled '{title}'" if title else ""))
    if kind == "value":
        return _short(f"value: {result}")
    return _short(str(result))


def _short(text: str) -> str:
    return text if len(text) <= MAX_SUMMARY_CHARS else text[:MAX_SUMMARY_CHARS - 3].rstrip() + "..."
