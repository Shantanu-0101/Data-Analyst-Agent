# Data Analyst Agent — Project Spec

A conversational AI agent that lets users upload a CSV, ask questions in plain English, and get answers as tables, charts, or summaries. The agent writes pandas/Plotly code behind the scenes, executes it, and self-corrects when the code fails.

---

## Workflow

```
User uploads CSV
       │
       ▼
Agent reads schema (columns, dtypes, sample rows, nulls, stats)
       │
       ▼
User asks a question in plain English
       │
       ▼
┌─────────────────────────────────┐
│  LangGraph Agent Loop           │
│                                 │
│  1. Plan — decide what code     │
│     to write (pandas/Plotly)    │
│                                 │
│  2. Generate — write Python     │
│     code using the schema       │
│                                 │
│  3. Execute — run code in a     │
│     sandboxed exec()            │
│                                 │
│  4. Check — did it error?       │
│     ├─ YES → read traceback,    │
│     │   fix code, retry (max 3) │
│     └─ NO  → return result      │
└─────────────────────────────────┘
       │
       ▼
Display result in Streamlit
  - DataFrame → st.dataframe()
  - Plotly fig → st.plotly_chart()
  - Number/text → st.metric() or st.write()
```

---

## Tech Stack

| Component        | Tool                          |
|------------------|-------------------------------|
| Agent framework  | LangGraph (StateGraph)        |
| LLM              | Google Gemini (free tier)     |
| Data processing  | pandas                        |
| Visualization    | Plotly Express                |
| UI               | Streamlit                     |
| Embeddings       | None (not needed)             |
| Vector store     | None (not needed)             |

---

## Project Structure

```
data-analyst-agent/
├── app.py                        # Streamlit UI
├── src/
│   ├── config.py                 # API keys, settings
│   ├── agent/
│   │   ├── graph.py              # LangGraph StateGraph definition
│   │   ├── nodes.py              # Node functions (plan, generate, execute, fix)
│   │   └── state.py              # TypedDict for agent state
│   ├── data/
│   │   └── loader.py             # CSV loading, schema extraction, preview
│   ├── execution/
│   │   └── sandbox.py            # Safe code execution (exec with restricted globals)
│   └── prompts/
│       └── templates.py          # All LLM prompt templates
├── deliverables/
│   ├── walkthrough.ipynb         # Jupyter notebook walkthrough
│   ├── walkthrough.py            # Script version
│   ├── generate_report.py        # PDF report generator
│   └── Project_Report.pdf        # Generated report
├── sample_data/
│   └── sales.csv                 # Sample CSV for demo
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Agent Graph (LangGraph)

```
          START
            │
            ▼
      ┌───────────┐
      │   plan     │  ← Decides approach: "filter + group + plot" etc.
      └─────┬─────┘
            │
            ▼
      ┌───────────┐
      │ generate   │  ← Writes pandas/Plotly code as a string
      └─────┬─────┘
            │
            ▼
      ┌───────────┐
      │  execute   │  ← Runs code via exec(), captures result or error
      └─────┬─────┘
            │
        ┌───┴───┐
        │       │
     success   error
        │       │
        ▼       ▼
      ┌────┐ ┌─────┐
      │ END│ │ fix  │  ← Reads traceback, rewrites code
      └────┘ └──┬──┘
                │
                ▼
           retries < 3?
           ├─ YES → back to execute
           └─ NO  → END (show error to user)
```

### State Schema

```python
from typing import TypedDict, Optional
import pandas as pd

class AgentState(TypedDict):
    df: pd.DataFrame              # The uploaded dataset
    schema_info: str              # Column names, dtypes, sample rows, stats
    question: str                 # User's plain-English question
    plan: str                     # LLM's approach description
    generated_code: str           # Python code string
    result: Optional[any]         # Execution output (df, fig, value)
    result_type: str              # "dataframe" | "plotly" | "value" | "text"
    error: Optional[str]          # Traceback if code failed
    retries: int                  # Current retry count
    history: list[dict]           # Chat history for context
```

---

## Node Details

### 1. `plan` node
- Input: `question` + `schema_info`
- LLM decides the approach in 1-2 sentences
- Example: "Filter rows where region == 'West', group by product, sum revenue, plot as bar chart"
- Output: sets `state["plan"]`

### 2. `generate` node
- Input: `plan` + `schema_info` + `question`
- LLM writes Python code that uses a pre-loaded variable `df` (the DataFrame)
- Code must store its output in a variable called `RESULT`
- For charts, `RESULT` must be a Plotly figure
- For tables, `RESULT` must be a DataFrame
- For single values, `RESULT` must be a string or number
- Output: sets `state["generated_code"]`

### 3. `execute` node
- Runs `generated_code` via `exec()` with restricted globals:
  ```python
  allowed_globals = {
      "pd": pd,
      "px": plotly.express,
      "np": numpy,
      "df": state["df"],  # user's data
  }
  ```
- Captures `RESULT` from the local namespace after execution
- Detects result type (DataFrame → "dataframe", Plotly Figure → "plotly", else → "value")
- On exception: captures full traceback string
- Output: sets `state["result"]` + `state["result_type"]` OR `state["error"]`

### 4. `fix` node
- Input: `generated_code` + `error` + `schema_info`
- LLM reads the traceback and rewrites the code
- Increments `retries`
- Output: sets `state["generated_code"]` (updated), routes back to `execute`

### Routing logic
```python
def should_retry(state):
    if state["error"] and state["retries"] < 3:
        return "fix"
    return "end"
```

---

## Prompt Templates

### Schema Prompt (sent with every request)
```
You are a data analyst agent. The user uploaded a CSV with this schema:

Columns: {columns_with_dtypes}
Row count: {row_count}
Sample (first 3 rows):
{sample_rows}

Null counts: {null_summary}
Numeric stats: {describe_output}
```

### Code Generation Prompt
```
Write Python code to answer this question: "{question}"

Plan: {plan}

Rules:
- The DataFrame is already loaded as `df`
- pandas is `pd`, plotly.express is `px`, numpy is `np`
- Store your final answer in a variable called `RESULT`
- For charts: RESULT = px.bar(...) or px.line(...) etc.
- For tables: RESULT = df_filtered (a DataFrame)
- For values: RESULT = "The answer is ..."
- Do NOT use print(), st.write(), or display()
- Do NOT read files — df is already loaded
- Only output the code, no explanation
```

### Fix Prompt
```
Your code failed with this error:

```
{traceback}
```

Original code:
```python
{code}
```

Fix the code. Same rules apply. Only output the corrected code.
```

---

## Streamlit UI Layout

```
┌──────────────────────────────────────────────────┐
│  Data Analyst Agent                              │
├──────────────┬───────────────────────────────────┤
│  Sidebar     │  Main area                        │
│              │                                   │
│  Upload CSV  │  Dataset Preview (first 10 rows)  │
│  [Browse]    │  ┌──────────────────────────────┐ │
│              │  │  st.dataframe()              │ │
│  Dataset     │  └──────────────────────────────┘ │
│  info:       │                                   │
│  - 1500 rows │  Schema & Stats (expandable)      │
│  - 8 columns │                                   │
│  - 0 nulls   │  ─────────────────────────────── │
│              │                                   │
│  Chat        │  Chat Interface                   │
│  history     │  ┌──────────────────────────────┐ │
│  (clickable) │  │ You: What's the top product  │ │
│              │  │      by revenue?             │ │
│              │  │                              │ │
│              │  │ Agent: [bar chart]           │ │
│              │  │ "Product X leads with $45K"  │ │
│              │  └──────────────────────────────┘ │
│              │                                   │
│              │  [Ask a question...]    [Send]    │
│              │                                   │
│              │  ▶ Show generated code (expander) │
└──────────────┴───────────────────────────────────┘
```

---

## Key Features

1. **Natural language to code** — User asks in English, agent writes pandas/Plotly
2. **Self-healing** — When code errors, agent reads traceback and retries (up to 3x)
3. **Auto-visualization** — Agent decides whether to show a table, chart, or metric
4. **Schema-aware** — Every prompt includes column names, types, and sample data so the LLM never guesses column names
5. **Code transparency** — Generated code shown in an expander so user can learn
6. **Chat history** — Previous Q&A pairs give context for follow-up questions like "now filter that to 2024 only"

---

## Sample Questions (for demo)

Using a `sales.csv` with columns: `date, product, region, revenue, quantity, salesperson`:

| Question | Expected output |
|----------|----------------|
| "Show the first 5 rows" | DataFrame |
| "What's the total revenue?" | Metric: $X |
| "Revenue by region as a bar chart" | Plotly bar chart |
| "Top 3 salespersons by quantity sold" | DataFrame (3 rows) |
| "Monthly revenue trend" | Plotly line chart |
| "Which product has the highest average revenue?" | Text answer |
| "Compare regions with a pie chart" | Plotly pie chart |
| "Show a heatmap of revenue by product and region" | Plotly heatmap |

---

## Deliverables

1. **Jupyter Notebook** (`walkthrough.ipynb`) — Step-by-step: load CSV → build schema prompt → call agent → show results
2. **Python Script** (`walkthrough.py`) — Same flow as a runnable script
3. **Streamlit App** (`app.py`) — Full interactive web app
4. **Project Report** (`Project_Report.pdf`) — Architecture, flowchart, sample outputs, learnings

---

## Dependencies

```
streamlit
langchain-google-genai
langgraph
pandas
plotly
python-dotenv
fpdf2
```

---

## .env

```
GOOGLE_API_KEY=your_gemini_key_here
```

That's it — one API key. Gemini free tier gives 15 RPM which is enough for a demo.

---

## What Makes This Agentic

This isn't a simple "send prompt → get text" project. The agent:

1. **Plans** before coding (decides approach)
2. **Acts** by writing and executing real code
3. **Observes** the result or error
4. **Reflects** on failures and self-corrects
5. **Maintains context** across a conversation

This is the Plan → Act → Observe → Reflect loop that defines an AI agent, implemented as a LangGraph StateGraph with conditional routing.
