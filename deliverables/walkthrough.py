# %% [markdown]
# # Data Analyst Agent: walkthrough
#
# This walks through the agent step by step, outside the Streamlit app:
#
# 1. Load a CSV with pandas.
# 2. Turn it into the **schema prompt** the LLM sees on every call.
# 3. Build the LangGraph agent: plan → generate → execute → (fix → execute)*.
# 4. Ask the spec's eight sample questions, plus a follow-up that builds on the previous answer.
# 5. Watch the self-correction loop repair broken code from a real traceback.
#
# Run it as a script (`python deliverables/walkthrough.py`) or as the notebook `walkthrough.ipynb`,
# which is built from this file. It needs an LLM key in `.env` (`GROQ_API_KEY` or `GOOGLE_API_KEY`).
# Charts and a `walkthrough_results.json` summary go to `outputs/`; `generate_report.py` reads them.

# %%
import json
import sys
import time
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # in the notebook, which runs from deliverables/
    ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.agent.graph import ask, build_graph
from src.agent.nodes import get_llm
from src.config import LLM_MODEL, LLM_NAME, MAX_RETRIES, SAMPLE_CSV
from src.data.loader import dataset_summary, format_schema_info, load_csv
from src.execution.sandbox import run_code

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
IN_NOTEBOOK = "ipykernel" in sys.modules
PREVIEW_ROWS = 10  # rows of a table answer kept in the results file
if not IN_NOTEBOOK:  # model text can hold characters the Windows console encoding lacks
    sys.stdout.reconfigure(encoding="utf-8")

print(f"LLM: {LLM_NAME} · {LLM_MODEL}")

# %% [markdown]
# ## 1. Load the CSV
#
# `load_csv` accepts a path or an uploaded file, falls back to latin-1 if UTF-8 fails, and turns
# date-named columns into real datetimes, so the generated code can use `.dt` straight away.

# %%
df = load_csv(SAMPLE_CSV)
print(dataset_summary(df))
df.head()

# %% [markdown]
# ## 2. The schema prompt
#
# The LLM never sees the whole dataset. It gets this summary instead: columns and types, row count,
# a few sample rows, nulls, date ranges, category values and numeric stats. It is sent as the
# system message of every call, so the model writes code against real column names.

# %%
schema_info = format_schema_info(df)
print(schema_info)

# %% [markdown]
# ## 3. Build the agent
#
# `build_graph()` compiles the LangGraph state machine. `ask()` runs one question through it and
# returns the final state, including the `history` to pass to the next question for follow-ups.

# %%
graph = build_graph()
results = []  # one record per question, for the report


def on_retry(attempt, wait, exc):
    print(f"  ({LLM_NAME} is busy; retrying in {wait:.0f} s)")


def record(question, state, seconds, note=""):
    """Print one answer, save its chart, and keep a summary for the report."""
    kind, result = state["result_type"], state["result"]
    entry = {
        "question": question, "plan": state["plan"], "code": state["generated_code"],
        "result_type": kind, "retries": state["retries"], "error": state["error"],
        "seconds": round(seconds, 1), "note": note,
    }
    print(f"Q: {question}")
    print(f"Plan: {state['plan']}")
    print(f"Fix attempts: {state['retries']}  ·  {seconds:.1f} s")
    print("Code:\n" + state["generated_code"])
    if state["error"]:
        print("Failed after the last fix. Error:\n" + state["error"])
    elif kind == "dataframe":
        entry["columns"] = [str(col) for col in result.columns]
        entry["rows"] = json.loads(result.head(PREVIEW_ROWS).to_json(orient="values", date_format="iso"))
        entry["row_count"] = len(result)
        if not IN_NOTEBOOK:
            print(result.head(PREVIEW_ROWS).to_string())
    elif kind == "plotly":
        name = f"q{len(results) + 1:02d}"
        result.write_html(OUTPUTS / f"{name}.html", include_plotlyjs="cdn")
        result.write_image(OUTPUTS / f"{name}.png", width=900, height=500, scale=2)
        entry["chart"] = f"{name}.png"
        trace = result.data[0] if result.data else None
        # px.line draws "scatter" traces in lines mode
        entry["chart_type"] = "line" if trace and trace.type == "scatter" and "lines" in (trace.mode or "") else (
            trace.type if trace else None
        )
        print(f"Chart saved to outputs/{name}.png and .html")
    else:
        entry["value"] = result if isinstance(result, (int, float, str)) else str(result)
        print(f"Answer: {result}")
    results.append(entry)
    if IN_NOTEBOOK:  # show the real object under the cell
        from IPython.display import Image, display

        if kind == "dataframe":
            display(result.head(PREVIEW_ROWS))
        elif kind == "plotly":
            display(Image(filename=str(OUTPUTS / entry["chart"]), width=700))


def run(question, history=None, llm=None, note=""):
    start = time.perf_counter()
    state = ask(graph, df, schema_info, question, history, llm=llm, on_retry=on_retry)
    record(question, state, time.perf_counter() - start, note)
    return state


# %% [markdown]
# ## 4. The spec's sample questions
#
# Each answer is a table, a single value, text, or a Plotly chart. The agent picks the type from
# the question; the code it wrote is printed under every answer.

# %%
state = run("Show the first 5 rows")

# %%
state = run("What's the total revenue?")

# %%
state = run("Revenue by region as a bar chart")

# %%
state = run("Top 3 salespersons by quantity sold")

# %%
state = run("Monthly revenue trend")

# %% [markdown]
# ## 5. A follow-up question
#
# Passing the previous state's `history` lets the agent build on the code it just wrote.
# "Now filter that to 2024 only" means the monthly trend above.

# %%
state = run("Now filter that to 2024 only", history=state["history"], note="follow-up to the monthly trend")

# %%
state = run("Which product has the highest average revenue?")

# %%
state = run("Compare regions with a pie chart")

# %%
state = run("Show a heatmap of revenue by product and region")

# %% [markdown]
# ## 6. Self-correction
#
# Today's models rarely fail on this clean data, so the loop is hard to see. To show it, the
# wrapper below replaces the model's **first** code with a version that uses a misspelled column.
# Everything after that is real: the sandbox raises a genuine `KeyError`, and the live model's
# fix node reads the traceback and repairs the code.


# %%
class FirstCodeBroken:
    """Passes calls to the real model, except the first code-writing call (right after the plan)."""

    BROKEN = "RESULT = df.groupby('product')['revenu'].sum().sort_values(ascending=False)"

    def __init__(self, llm: BaseChatModel):
        self.llm, self.answered, self.broke = llm, 0, False

    def invoke(self, messages):
        if self.answered == 1 and not self.broke:  # count answers, not attempts: a 429 is retried
            self.broke = True
            return AIMessage(content=self.BROKEN)
        reply = self.llm.invoke(messages)
        self.answered += 1
        return reply


question = "Total revenue by product, highest first"
start = time.perf_counter()
state = ask(graph, df, schema_info, question, llm=FirstCodeBroken(get_llm()), on_retry=on_retry)
seconds = time.perf_counter() - start
# the first error is gone from the final state, so rebuild it the way the sandbox reported it
first_error = run_code(FirstCodeBroken.BROKEN, df)["error"]
print("First attempt:\n" + FirstCodeBroken.BROKEN + "\n\nSandbox error:\n" + first_error)
record(question, state, seconds, note="self-correction demo: first code broken on purpose")
results[-1].update({"broken_code": FirstCodeBroken.BROKEN, "first_error": first_error})
assert state["retries"] >= 1 and not state["error"], "the fix node should have repaired the code"

# %% [markdown]
# ## 7. Save the results
#
# `generate_report.py` builds `Project_Report.pdf` from this file and the chart PNGs.

# %%
summary = {
    "provider": LLM_NAME, "model": LLM_MODEL, "max_retries": MAX_RETRIES,
    "dataset": {"file": SAMPLE_CSV.name, **dataset_summary(df)},
    "run_at": time.strftime("%Y-%m-%d %H:%M"),
    "results": results,
}
(OUTPUTS / "walkthrough_results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
ok = sum(1 for r in results if not r["error"])
print(f"{ok} of {len(results)} questions answered; saved outputs/walkthrough_results.json")
