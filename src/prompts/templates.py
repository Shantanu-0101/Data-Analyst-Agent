"""All LLM prompt templates, as str.format-style strings. No logic here beyond format_history.

Every call sends SCHEMA_PROMPT (filled in, i.e. state["schema_info"]) as the system message
and one of PLAN_PROMPT, CODE_PROMPT or FIX_PROMPT as the user message. Literal braces in a
template are written {{ }} so str.format leaves them alone.
"""
from src.config import HISTORY_TURNS

# System prefix sent with every LLM call; filled in by src.data.loader.format_schema_info.
SCHEMA_PROMPT = """You are a data analyst agent. The user uploaded a CSV with this schema:

Columns: {columns_with_dtypes}
Row count: {row_count}
Sample (first 3 rows):
{sample_rows}

Null counts: {null_summary}
Date ranges: {date_ranges}
Text column values: {category_values}
Numeric stats:
{describe_output}
"""

# Shared by CODE_PROMPT and FIX_PROMPT, since the fix call does not see the code call.
# Kept in step with src.execution.sandbox: it rejects whatever these rules forbid.
_CODE_RULES = """Rules:
- The DataFrame is already loaded as `df`. Do NOT read or write files.
- pandas is `pd`, plotly.express is `px`, numpy is `np`. They are already available: do NOT import anything.
- Use only the columns in the schema, spelled exactly as shown (names are case-sensitive).
- Store your final answer in a variable called `RESULT`.
- For charts: RESULT = px.bar(...), px.line(...), px.pie(...) etc. Give every chart a title, and readable axis names through the `labels` argument.
  - Pie charts: aggregate first, then px.pie(table, names=..., values=...).
  - Heatmaps: px.density_heatmap(df, x=..., y=..., z=..., histfunc="sum"), or px.imshow on a pivot table.
  - Trends over time: group by month with df["<date column>"].dt.to_period("M").dt.to_timestamp(), then px.line.
- For tables: RESULT = a DataFrame. Call .reset_index() after a groupby so the group names are columns.
- For a single number (a total, count or average): RESULT = the number itself, e.g. RESULT = round(df["<column>"].sum(), 2).
- For an answer that names something: RESULT = one short sentence with the key figure, e.g. RESULT = f"{{name}} has the highest average ({{value:,.2f}})."
- If the data cannot answer the question, set RESULT to a short sentence saying why.
- Columns of type datetime64 are already parsed: use .dt, not pd.to_datetime.
- Use vectorised pandas operations, not Python loops over rows: slow code is stopped.
- Do NOT use print(), st.write(), display() or fig.show().
- Do NOT define classes, use eval/exec/getattr, or use names or attributes that start with __.
"""

PLAN_PROMPT = """Previous questions in this conversation, oldest first:
{history}

New question: "{question}"

In 1-2 sentences, say how you will answer the new question with pandas and Plotly Express: which columns you will use, how you will filter, group or aggregate them, and what form the answer takes:
- a chart (name the type) when the user asks for one, or for a trend or comparison,
- a table for lists, rankings or rows,
- a single number for one total, count or average,
- a short sentence when the answer names something (e.g. which product is highest).
If the question refers to a previous answer, say how you will change that answer's code.
If the data cannot answer the question, say so. Do not write code.
"""

CODE_PROMPT = """Write Python code to answer this question: "{question}"

Plan: {plan}

Previous questions in this conversation, oldest first:
{history}

If the question refers to a previous answer ("that", "now only 2024", "same but by region"), start from that answer's code above and change it.

""" + _CODE_RULES + """
Only output the code, no explanation.
"""

FIX_PROMPT = """Your code for this question failed: "{question}"

Plan: {plan}

Error:
```
{error}
```

Code that failed:
```python
{code}
```

Fix the code. If a column or value name is wrong, use the exact names from the schema.

""" + _CODE_RULES + """
Only output the corrected code, no explanation.
"""

NO_HISTORY = "none"


def format_history(history: list[dict]) -> str:
    """Render the last HISTORY_TURNS turns for the plan and code prompts.

    Each turn is a dict with `question`, `code` and `result_summary` (code and summary may
    be missing or empty, e.g. for a turn that failed before any code was written).
    """
    turns = history[-HISTORY_TURNS:] if history else []
    if not turns:
        return NO_HISTORY
    blocks = []
    for n, turn in enumerate(turns, 1):
        lines = [f"Q{n}: {turn['question']}"]
        if turn.get("code"):
            lines.append(f"Code:\n```python\n{turn['code'].strip()}\n```")
        if turn.get("result_summary"):
            lines.append(f"Result: {turn['result_summary']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
