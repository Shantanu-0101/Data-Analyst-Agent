"""The agent loop and routing, with a scripted fake LLM (no network)."""
import groq
import httpx
import pandas as pd
import pytest
from langchain_core.exceptions import (
    ModelAPIError, ModelAuthenticationError, ModelNotFoundError, ModelRateLimitError,
)
from langchain_core.messages import AIMessage, SystemMessage

from src.agent import nodes
from src.agent.graph import ask, build_graph, should_retry, summarize_result
from src.agent.nodes import LLMUnavailable, invoke_llm
from src.config import MAX_RETRIES

PLAN = "Sum revenue."
GOOD = "RESULT = df['revenue'].sum()"
BAD = "RESULT = df['revenu'].sum()"  # wrong column name: KeyError


class ScriptedLLM:
    """Replies with the given texts in order; an Exception in the list is raised instead."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []  # the messages of each call

    def invoke(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)

    def prompt(self, n):
        return self.calls[n][-1].content


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["East", "West", "East"], "revenue": [100.0, 250.0, 50.0]})


@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(nodes, "_sleep", waits.append)
    return waits


def run(graph, df, llm, question="Total revenue?", history=None):
    return ask(graph, df, "SCHEMA", question, history, llm=llm)


# --- the loop ----------------------------------------------------------------

def test_good_code_first_try(graph, df):
    llm = ScriptedLLM(PLAN, f"```python\n{GOOD}\n```")
    state = run(graph, df, llm)
    assert state["retries"] == 0 and state["error"] is None
    assert state["result"] == 400.0 and state["result_type"] == "value"
    assert state["plan"] == PLAN
    assert state["generated_code"] == GOOD  # fences stripped
    assert len(llm.calls) == 2
    assert all(isinstance(call[0], SystemMessage) and call[0].content == "SCHEMA" for call in llm.calls)


def test_bad_then_good_code_takes_one_fix(graph, df):
    llm = ScriptedLLM(PLAN, BAD, GOOD)
    state = run(graph, df, llm)
    assert state["retries"] == 1 and state["error"] is None
    assert state["result"] == 400.0
    fix_prompt = llm.prompt(2)
    assert BAD in fix_prompt and "KeyError" in fix_prompt and "revenu" in fix_prompt


def test_always_bad_code_stops_after_max_retries(graph, df):
    llm = ScriptedLLM(PLAN, *[BAD] * (1 + MAX_RETRIES))
    state = run(graph, df, llm)
    assert state["retries"] == MAX_RETRIES
    assert "KeyError" in state["error"]
    assert state["result"] is None and state["result_type"] is None
    assert len(llm.calls) == 2 + MAX_RETRIES  # plan, generate, then one call per fix
    assert not llm.replies


def test_rejected_code_goes_to_fix(graph, df):
    llm = ScriptedLLM(PLAN, "import os\nRESULT = 1", GOOD)
    state = run(graph, df, llm)
    assert state["retries"] == 1 and state["result"] == 400.0
    assert "imports are not allowed" in llm.prompt(2)


def test_should_retry_routing():
    assert should_retry({"error": "boom", "retries": 0}) == "fix"
    assert should_retry({"error": "boom", "retries": MAX_RETRIES - 1}) == "fix"
    assert should_retry({"error": "boom", "retries": MAX_RETRIES}) == "end"
    assert should_retry({"error": None, "retries": 0}) == "end"


# --- history -----------------------------------------------------------------

def test_history_is_returned_with_this_turn(graph, df):
    earlier = [{"question": "Revenue by region", "code": "RESULT = df.groupby('region').sum()", "result_summary": "table"}]
    state = run(graph, df, ScriptedLLM(PLAN, GOOD), history=earlier)
    assert len(earlier) == 1  # the caller's list is not changed
    assert state["history"][:1] == earlier
    assert state["history"][1] == {"question": "Total revenue?", "code": GOOD, "result_summary": "value: 400.0"}


def test_history_reaches_plan_and_code_prompts(graph, df):
    earlier = [{"question": "Revenue by region", "code": "RESULT = df.groupby('region')['revenue'].sum()"}]
    llm = ScriptedLLM(PLAN, GOOD)
    run(graph, df, llm, question="now only East", history=earlier)
    for n in (0, 1):
        assert "Revenue by region" in llm.prompt(n)
        assert "df.groupby('region')['revenue'].sum()" in llm.prompt(n)


def test_failed_turn_is_summarised(graph, df):
    state = run(graph, df, ScriptedLLM(PLAN, *[BAD] * (1 + MAX_RETRIES)))
    assert state["history"][-1]["result_summary"].startswith("failed: KeyError")


def test_user_data_is_not_changed(graph, df):
    before = df.copy()
    run(graph, df, ScriptedLLM(PLAN, "df['revenue'] = 0\nRESULT = df['revenue'].sum()"))
    pd.testing.assert_frame_equal(df, before)


# --- summaries ---------------------------------------------------------------

def test_summaries_by_result_type(graph, df):
    table = run(graph, df, ScriptedLLM(PLAN, "RESULT = df.groupby('region', as_index=False)['revenue'].sum()"))
    assert summarize_result(table) == "table, 2 rows; columns: region, revenue"
    chart = run(graph, df, ScriptedLLM(PLAN, "RESULT = px.bar(df, x='region', y='revenue', title='Revenue')"))
    assert summarize_result(chart) == "bar chart titled 'Revenue'"
    text = run(graph, df, ScriptedLLM(PLAN, "RESULT = 'West leads. ' * 50"))
    assert summarize_result(text).startswith("West leads.") and len(summarize_result(text)) <= 200


# --- the LLM call ------------------------------------------------------------

def test_rate_limit_is_retried_with_the_servers_delay(no_sleep):
    seen = []
    llm = ScriptedLLM(ModelRateLimitError("429 RESOURCE_EXHAUSTED {'retryDelay': '37s'}"), "ok")
    assert invoke_llm(llm, [], on_retry=lambda attempt, wait, exc: seen.append((attempt, wait))) == "ok"
    assert no_sleep == [38.0] and seen == [(1, 38.0)]


def test_backoff_doubles_or_follows_the_server(no_sleep):
    llm = ScriptedLLM(ModelAPIError("503"), ModelRateLimitError("Please retry in 50s."), "ok")
    assert invoke_llm(llm, []) == "ok"
    assert no_sleep == [nodes.BACKOFF_BASE_S, 51.0]


def test_daily_quota_gives_up_at_once(no_sleep):
    error = ModelRateLimitError("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier {'retryDelay': '16514s'}")
    llm = ScriptedLLM(error, "never reached")
    with pytest.raises(LLMUnavailable, match=r"resets in about 4 h 35 min"):
        invoke_llm(llm, [])
    assert no_sleep == [] and len(llm.calls) == 1


def test_timeouts_and_dropped_connections_are_retried(no_sleep):
    llm = ScriptedLLM(httpx.ReadTimeout("timed out"), httpx.ConnectError("reset"), "ok")
    assert invoke_llm(llm, []) == "ok"
    assert len(no_sleep) == 2


def test_timeout_gives_up_with_a_readable_message(no_sleep):
    llm = ScriptedLLM(*[httpx.ReadTimeout("timed out")] * nodes.LLM_ATTEMPTS)
    with pytest.raises(LLMUnavailable, match="not responding"):
        invoke_llm(llm, [])


def test_rate_limit_gives_up_after_the_last_attempt(no_sleep):
    llm = ScriptedLLM(*[ModelRateLimitError("429")] * nodes.LLM_ATTEMPTS)
    with pytest.raises(LLMUnavailable, match="rate limit"):
        invoke_llm(llm, [])
    assert len(no_sleep) == nodes.LLM_ATTEMPTS - 1


@pytest.mark.parametrize("error, message", [
    (ModelAuthenticationError("401"), "API key"),
    (ModelNotFoundError("404"), nodes.LLM_MODEL_SETTING),
])
def test_permanent_errors_are_not_retried(no_sleep, error, message):
    llm = ScriptedLLM(error, "never reached")
    with pytest.raises(LLMUnavailable, match=message):
        invoke_llm(llm, [])
    assert no_sleep == [] and len(llm.calls) == 1


def test_llm_failure_ends_the_run(graph, df):
    with pytest.raises(LLMUnavailable):
        run(graph, df, ScriptedLLM(PLAN, ModelAuthenticationError("401")))


# --- Groq's errors -----------------------------------------------------------

def groq_error(cls, status, text):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return cls(text, response=response, body=None)


@pytest.mark.parametrize("text, wait", [
    ("Rate limit reached. Please try again in 7.66s.", 8.66),
    ("Please try again in 0m45.5s.", 46.5),
    ("Please try again in 520ms.", nodes.BACKOFF_BASE_S),  # shorter than the first backoff
])
def test_groq_rate_limit_is_retried_with_its_delay(no_sleep, text, wait):
    llm = ScriptedLLM(groq_error(groq.RateLimitError, 429, text), "ok")
    assert invoke_llm(llm, []) == "ok"
    assert no_sleep == [pytest.approx(wait)]


def test_groq_daily_limit_gives_up_at_once(no_sleep):
    llm = ScriptedLLM(groq_error(groq.RateLimitError, 429, "tokens per day (TPD). Please try again in 2h3m4s."), "x")
    with pytest.raises(LLMUnavailable, match=r"resets in about 2 h 3 min"):
        invoke_llm(llm, [])
    assert no_sleep == []


def test_groq_server_errors_and_timeouts_are_retried(no_sleep):
    request = httpx.Request("POST", "https://api.groq.com")
    llm = ScriptedLLM(groq_error(groq.InternalServerError, 503, "over capacity"), groq.APITimeoutError(request), "ok")
    assert invoke_llm(llm, []) == "ok"
    assert len(no_sleep) == 2


@pytest.mark.parametrize("cls, status, message", [
    (groq.AuthenticationError, 401, "API key"),
    (groq.NotFoundError, 404, nodes.LLM_MODEL_SETTING),
])
def test_groq_permanent_errors_are_not_retried(no_sleep, cls, status, message):
    llm = ScriptedLLM(groq_error(cls, status, "nope"), "never reached")
    with pytest.raises(LLMUnavailable, match=message):
        invoke_llm(llm, [])
    assert no_sleep == [] and len(llm.calls) == 1
