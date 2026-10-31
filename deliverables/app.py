"""Streamlit UI: upload a CSV, ask questions in plain English, see tables, charts and answers.

Run with: streamlit run app.py
"""
import numbers

import streamlit as st

from src.agent.graph import ask, build_graph
from src.agent.nodes import LLMUnavailable
from src.config import LLM_KEY_SETTING, LLM_MODEL, LLM_NAME, MAX_RETRIES, SAMPLE_CSV, has_api_key
from src.data.loader import DataLoadError, dataset_summary, format_schema_info, load_csv

PREVIEW_ROWS = 10
SIDEBAR_QUESTION_CHARS = 60

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph()


# --- state -------------------------------------------------------------------
# df / schema_info / source: the loaded dataset; messages: one dict per answered question
# (question, plan, code, result, result_type, error, retries); history: the agent's own
# turn list, passed back into ask() for follow-up questions.

def _init_state():
    defaults = {"df": None, "schema_info": "", "source": None, "uploaded": False, "load_error": None,
                "messages": [], "history": []}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _set_data(file_or_path, source: str, uploaded: bool):
    try:
        df = load_csv(file_or_path)
    except DataLoadError as exc:
        st.session_state.load_error = f"Could not load {source}: {exc}"
        return
    st.session_state.update(
        df=df, schema_info=format_schema_info(df), source=source, uploaded=uploaded, load_error=None,
        messages=[], history=[],
    )


def _clear_data():
    st.session_state.update(
        df=None, schema_info="", source=None, uploaded=False, load_error=None, messages=[], history=[],
    )


def _on_upload():
    upload = st.session_state.uploader
    if upload is not None:
        _set_data(upload, upload.name, uploaded=True)
    elif st.session_state.uploaded:  # the user removed the file that is loaded
        _clear_data()


def _use_sample():
    _set_data(SAMPLE_CSV, SAMPLE_CSV.name, uploaded=False)


def _ask_again(question: str):
    st.session_state.pending_question = question


def _clear_chat():
    st.session_state.update(messages=[], history=[])


# --- rendering ---------------------------------------------------------------

def _escape_markdown(text: str) -> str:
    # "$45K and $30K" would otherwise render as LaTeX
    return str(text).replace("$", r"\$")


def _show_table(df):
    """st.dataframe, with date columns that hold no times shown as plain dates."""
    config = {
        col: st.column_config.DatetimeColumn(format="YYYY-MM-DD")
        for col in df.select_dtypes(include="datetime").columns
        if (df[col].dropna() == df[col].dropna().dt.normalize()).all()
    }
    st.dataframe(df, hide_index=True, column_config=config)


def _format_value(value) -> str:
    if isinstance(value, numbers.Integral):
        return f"{value:,}"
    if isinstance(value, numbers.Real):
        return f"{value:,.2f}"
    return str(value)


def _render_answer(message: dict, index: int):
    if message.get("notice"):  # the LLM itself could not be reached
        st.error(message["notice"])
        return
    if message["error"]:
        st.error(
            f"I couldn't get working code for this question after {message['retries']} "
            f"fix attempt{'s' if message['retries'] != 1 else ''}. The last error was:"
        )
        st.code(message["error"], language=None)
    else:
        result, kind = message["result"], message["result_type"]
        if kind == "dataframe":
            _show_table(result)
            st.caption(f"{len(result):,} row{'s' if len(result) != 1 else ''}")
        elif kind == "plotly":
            st.plotly_chart(result, key=f"chart-{index}")
        elif kind == "value":
            st.metric("Answer", _format_value(result))
        else:
            st.markdown(_escape_markdown(result))

    with st.expander("Show generated code"):
        if message["plan"]:
            st.markdown(f"**Plan:** {_escape_markdown(message['plan'])}")
        if message["retries"]:
            st.caption(
                f"The first code failed; fixed after {message['retries']} attempt{'s' if message['retries'] != 1 else ''}."
                if not message["error"] else f"Tried {message['retries']} fixes."
            )
        st.code(message["code"] or "# no code was generated", language="python")


def _answer(question: str) -> dict:
    """Run the agent on one question and return the message to store and render."""
    message = {"question": question, "plan": "", "code": "", "result": None, "result_type": None,
               "error": None, "retries": 0, "notice": None}
    status = st.empty()

    def on_retry(attempt, wait, exc):
        status.info(f"{LLM_NAME} is busy, retrying in {wait:.0f} seconds (attempt {attempt + 1})…")

    try:
        with st.spinner("Thinking…"):
            state = ask(
                get_graph(), st.session_state.df, st.session_state.schema_info, question,
                st.session_state.history, on_retry=on_retry,
            )
    except LLMUnavailable as exc:
        message["notice"] = str(exc)
        return message
    finally:
        status.empty()

    st.session_state.history = state["history"]
    message.update({key: state[key] for key in ("plan", "result", "result_type", "error", "retries")})
    message["code"] = state["generated_code"]
    return message


# --- page --------------------------------------------------------------------

_init_state()
has_key = has_api_key()

with st.sidebar:
    st.header("Data")
    st.file_uploader("Upload a CSV", type=["csv"], key="uploader", on_change=_on_upload)
    st.button("Use sample data", on_click=_use_sample, width="stretch")
    if st.session_state.load_error:
        st.error(st.session_state.load_error)

    if st.session_state.df is not None:
        summary = dataset_summary(st.session_state.df)
        st.subheader("Dataset info")
        st.caption(st.session_state.source)
        st.markdown(
            f"- {summary['rows']:,} rows\n- {summary['columns']} columns\n- {summary['nulls']:,} nulls"
        )

    if st.session_state.messages:
        st.subheader("Chat history")
        st.caption("Click a question to ask it again.")
        for i, message in enumerate(st.session_state.messages):
            question = message["question"]
            label = question if len(question) <= SIDEBAR_QUESTION_CHARS else question[:SIDEBAR_QUESTION_CHARS - 1] + "…"
            st.button(label, key=f"history-{i}", on_click=_ask_again, args=(question,), width="stretch")
        st.button("Clear chat", on_click=_clear_chat, type="tertiary")

    st.caption(f"Model: {LLM_NAME} · {LLM_MODEL}")

st.title("Data Analyst Agent")

if not has_key:
    st.error(f"{LLM_KEY_SETTING} is not set. Copy .env.example to .env, add your {LLM_NAME} key, then restart the app.")

df = st.session_state.df
if df is None:
    st.info("Upload a CSV in the sidebar, or try the sample sales data, to start asking questions.")
    # the sidebar starts collapsed on phones, so offer the sample here too
    st.button("Use sample data", on_click=_use_sample, key="sample-main")
    st.stop()

st.subheader("Dataset preview")
_show_table(df.head(PREVIEW_ROWS))
with st.expander("Schema & stats"):
    st.code(st.session_state.schema_info, language=None)

st.divider()

for i, message in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(_escape_markdown(message["question"]))
    with st.chat_message("assistant"):
        _render_answer(message, i)

question = st.chat_input(
    "Ask a question about your data" if has_key else f"Add a {LLM_NAME} API key to ask questions",
    disabled=not has_key,
)
question = question or st.session_state.pop("pending_question", None)

if question and has_key:
    with st.chat_message("user"):
        st.markdown(_escape_markdown(question))
    with st.chat_message("assistant"):
        message = _answer(question)
    st.session_state.messages.append(message)
    st.rerun()  # redraw from state, so the sidebar history includes this question
