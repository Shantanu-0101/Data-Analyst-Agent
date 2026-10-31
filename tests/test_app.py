"""The Streamlit app, run headless with AppTest. The LLM is replaced by a scripted fake."""
import pytest
from langchain_core.exceptions import ModelRateLimitError
from streamlit.testing.v1 import AppTest

from src.agent import nodes
from src.config import PROJECT_ROOT, SAMPLE_CSV
from src.data.loader import format_schema_info, load_csv
from tests.test_agent import ScriptedLLM

TIMEOUT_S = 30  # the first run imports pandas, plotly and langgraph


@pytest.fixture
def app():
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=TIMEOUT_S)


@pytest.fixture
def loaded(app):
    df = load_csv(SAMPLE_CSV)
    app.session_state["df"] = df
    app.session_state["schema_info"] = format_schema_info(df)
    app.session_state["source"] = SAMPLE_CSV.name
    return app


def message(**fields):
    base = {"question": "q", "plan": "p", "code": "RESULT = 1", "result": None, "result_type": None,
            "error": None, "retries": 0, "notice": None}
    return base | fields


def test_empty_state_offers_the_sample(app):
    app.run()
    assert not app.exception
    assert "Upload a CSV" in app.info[0].value
    app.button(key="sample-main").click().run()
    assert app.session_state["df"] is not None
    assert app.sidebar.subheader[0].value == "Dataset info"
    assert "1,500 rows" in app.sidebar.markdown[0].value


def test_failed_code_shows_the_last_error_and_the_code(loaded):
    loaded.session_state["messages"] = [message(error="KeyError: 'revenu'", retries=3, code="RESULT = df['revenu']")]
    loaded.run()
    assert not loaded.exception
    assert "after 3 fix attempts" in loaded.error[0].value
    codes = [block.value for block in loaded.code]
    assert "KeyError: 'revenu'" in codes and "RESULT = df['revenu']" in codes


def test_unreachable_llm_shows_the_notice(loaded):
    loaded.session_state["messages"] = [message(notice="Gemini's free-tier rate limit was reached.")]
    loaded.run()
    assert loaded.error[0].value == "Gemini's free-tier rate limit was reached."


def test_dollar_signs_are_not_read_as_latex(loaded):
    loaded.session_state["messages"] = [message(result="Laptop leads with $45K, ahead of $30K.", result_type="text")]
    loaded.run()
    assert r"\$45K" in loaded.chat_message[1].markdown[0].value


def test_question_runs_the_agent_and_lands_in_history(loaded, monkeypatch):
    llm = ScriptedLLM("Sum revenue.", "RESULT = round(df['revenue'].sum(), 2)")
    monkeypatch.setattr(nodes, "get_llm", lambda: llm)
    loaded.run()
    loaded.chat_input[0].set_value("What's the total revenue?").run()
    assert not loaded.exception
    assert loaded.metric[0].value == "1,996,526.43"
    assert loaded.session_state["history"][-1]["question"] == "What's the total revenue?"
    assert loaded.sidebar.button(key="history-0").label == "What's the total revenue?"


def test_rate_limit_becomes_a_message_not_a_crash(loaded, monkeypatch):
    monkeypatch.setattr(nodes, "_sleep", lambda s: None)
    llm = ScriptedLLM(*[ModelRateLimitError("429")] * nodes.LLM_ATTEMPTS)
    monkeypatch.setattr(nodes, "get_llm", lambda: llm)
    loaded.run()
    loaded.chat_input[0].set_value("Total?").run()
    assert not loaded.exception
    assert "rate limit" in loaded.error[0].value
