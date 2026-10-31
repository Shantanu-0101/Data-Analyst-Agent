"""The graph's node functions: plan, generate, execute and fix.

Each node takes the state (and the run's config) and returns only the keys it changes.
The LLM comes from config["configurable"]["llm"] when given, so tests can pass a fake
model; otherwise it is the shared Groq or Gemini client from get_llm().
"""
import re
import time
from functools import lru_cache
from typing import Callable, Optional

import groq
import httpx
from langchain_core.exceptions import (
    ModelAPIError, ModelAuthenticationError, ModelConnectionError, ModelError, ModelInvalidRequestError,
    ModelNotFoundError, ModelPermissionDeniedError, ModelRateLimitError, ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agent.state import AgentState
from src.config import LLM_KEY_SETTING, LLM_MODEL, LLM_MODEL_SETTING, LLM_NAME, LLM_PROVIDER, require_api_key
from src.execution.sandbox import clean_code, run_code
from src.prompts.templates import CODE_PROMPT, FIX_PROMPT, PLAN_PROMPT, format_history

LLM_ATTEMPTS = 3       # calls per prompt when the provider reports a temporary failure (429, 5xx)
BACKOFF_BASE_S = 2.0   # wait before the 2nd attempt; doubles after that
MAX_WAIT_S = 60.0      # a free-tier 429 can ask for up to a minute
REQUEST_TIMEOUT_S = 60.0  # the client waits forever by default

# A 429 says how long to wait: Gemini as "retryDelay': '37s'" or "Please retry in 37.2s",
# Groq as "Please try again in 7.66s", "in 1m23.5s", "in 2h3m4s" or "in 520ms"
_RETRY_DELAY = re.compile(
    r"(?:retryDelay['\"]?\s*:\s*['\"]|(?:retry|try again) in\s+)"
    r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+(?:\.\d+)?)s|(\d+)ms)?",
    re.IGNORECASE,
)

# on_retry(attempt that failed, seconds until the next one, the error), e.g. to tell the user
RetryCallback = Callable[[int, float, Exception], None]
_sleep = time.sleep  # replaced in tests


class LLMUnavailable(RuntimeError):
    """The LLM could not answer: bad key, unknown model, or still rate limited after retries.

    The message is written for the user. These are not problems in the generated code,
    so they end the run instead of going to the fix node.
    """


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    # slow imports; tests never need them
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=LLM_MODEL,
            temperature=0,
            api_key=require_api_key(),
            timeout=REQUEST_TIMEOUT_S,
            max_retries=0,  # invoke_llm retries, so it can report the wait
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=require_api_key(),
        timeout=REQUEST_TIMEOUT_S,
        max_retries=1,  # 1 means no SDK retries: invoke_llm retries, so it can report the wait
    )


def invoke_llm(llm: BaseChatModel, messages: list, on_retry: Optional[RetryCallback] = None) -> str:
    """Send the messages and return the reply text, retrying temporary failures with backoff."""
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            return _invoke(llm, messages)
        # timeouts and dropped connections reach us as raw httpx errors, not ModelErrors
        except (ModelError, httpx.TimeoutException, httpx.NetworkError) as exc:
            if isinstance(exc, ModelError) and not exc.is_retryable:
                raise LLMUnavailable(_describe(exc)) from exc
            asked = _server_delay(exc)
            if asked is not None and asked > MAX_WAIT_S:  # e.g. the daily quota: retrying now can't help
                raise LLMUnavailable(_describe_long_wait(asked)) from exc
            if attempt == LLM_ATTEMPTS:
                raise LLMUnavailable(_describe(exc, gave_up=True)) from exc
            wait = _wait_before_retry(exc, attempt)
            if on_retry:
                on_retry(attempt, wait, exc)
            _sleep(wait)
    raise AssertionError("unreachable")


def _invoke(llm: BaseChatModel, messages: list) -> str:
    """Call the model, turning the Groq SDK's errors into LangChain's ModelErrors.

    langchain-google-genai already raises ModelErrors; langchain-groq passes the SDK's own on.
    """
    try:
        return str(llm.invoke(messages).text)
    except groq.APITimeoutError as exc:  # before APIConnectionError, its parent
        raise ModelTimeoutError(str(exc)) from exc
    except groq.APIConnectionError as exc:
        raise ModelConnectionError(str(exc)) from exc
    except groq.APIStatusError as exc:
        fallback = ModelAPIError if exc.status_code >= 500 else ModelInvalidRequestError
        raise _GROQ_STATUS.get(exc.status_code, fallback)(str(exc)) from exc


_GROQ_STATUS = {
    401: ModelAuthenticationError,
    403: ModelPermissionDeniedError,
    404: ModelNotFoundError,
    429: ModelRateLimitError,
}


def plan_node(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    prompt = PLAN_PROMPT.format(question=state["question"], history=format_history(state.get("history")))
    return {"plan": _ask_llm(state, prompt, config).strip()}


def generate_node(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    prompt = CODE_PROMPT.format(
        question=state["question"], plan=state["plan"], history=format_history(state.get("history")),
    )
    return {"generated_code": clean_code(_ask_llm(state, prompt, config))}


def execute_node(state: AgentState) -> dict:
    # run_code returns {result, result_type, error}; error is None on success, which clears
    # the previous attempt's error
    return dict(run_code(state["generated_code"], state["df"]))


def fix_node(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    prompt = FIX_PROMPT.format(
        question=state["question"], plan=state["plan"], error=state["error"], code=state["generated_code"],
    )
    return {"generated_code": clean_code(_ask_llm(state, prompt, config)), "retries": state["retries"] + 1}


def _ask_llm(state: AgentState, prompt: str, config: Optional[RunnableConfig]) -> str:
    configurable = (config or {}).get("configurable", {})
    llm = configurable.get("llm") or get_llm()
    messages = [SystemMessage(state["schema_info"]), HumanMessage(prompt)]
    return invoke_llm(llm, messages, configurable.get("on_retry"))


def _server_delay(exc: Exception) -> Optional[float]:
    """Seconds the server asked us to wait (rounded up by 1), if its error says."""
    asked = _RETRY_DELAY.search(str(exc))
    if not asked or not any(asked.groups()):
        return None
    hours, minutes, seconds, millis = (float(part or 0) for part in asked.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000 + 1


def _wait_before_retry(exc: Exception, attempt: int) -> float:
    wait = max(BACKOFF_BASE_S * 2 ** (attempt - 1), _server_delay(exc) or 0)
    return min(wait, MAX_WAIT_S)


def _describe_long_wait(seconds: float) -> str:
    hours, minutes = divmod(round(seconds / 60), 60)
    when = f"{hours} h {minutes} min" if hours else f"{max(minutes, 1)} min"
    return (
        f"The free-tier quota for '{LLM_MODEL}' is used up; {LLM_NAME} says it resets in about {when}. "
        f"Try again then, or set {LLM_MODEL_SETTING} in .env to another model, which has its own quota."
    )


def _describe(exc: Exception, gave_up: bool = False) -> str:
    if isinstance(exc, ModelAuthenticationError):
        return f"{LLM_NAME} rejected the API key. Check {LLM_KEY_SETTING} in .env."
    if isinstance(exc, ModelPermissionDeniedError):
        return f"The API key is not allowed to use the model '{LLM_MODEL}'. Check the key's project and quota."
    if isinstance(exc, ModelNotFoundError):
        return f"{LLM_NAME} has no model named '{LLM_MODEL}'. Set {LLM_MODEL_SETTING} in .env to a current model name."
    if isinstance(exc, ModelRateLimitError):
        return f"{LLM_NAME}'s free-tier rate limit was reached. Wait a minute, then ask again."
    if gave_up:
        return f"{LLM_NAME} is not responding right now ({type(exc).__name__}). Try again in a minute."
    return f"{LLM_NAME} could not answer: {exc}"
