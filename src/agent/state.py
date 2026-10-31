"""The state passed between the agent graph's nodes."""
from typing import Any, Optional, TypedDict

import pandas as pd


class AgentState(TypedDict):
    df: pd.DataFrame              # The uploaded dataset
    schema_info: str              # Column names, dtypes, sample rows, stats
    question: str                 # User's plain-English question
    plan: str                     # LLM's approach description
    generated_code: str           # Python code string
    result: Optional[Any]         # Execution output (df, fig, value)
    result_type: Optional[str]    # "dataframe" | "plotly" | "value" | "text"; None on error
    error: Optional[str]          # Traceback if code failed
    retries: int                  # Fix attempts made so far
    history: list[dict]           # Earlier turns: {question, code, result_summary}
