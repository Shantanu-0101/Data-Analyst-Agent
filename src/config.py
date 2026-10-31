"""API keys and settings, loaded from the environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# gemini-2.5-flash is closed to new keys (404 since Oct 2026); 3.8-flash was overloaded (503) when tested
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Groq when its key is set (Gemini's free tier allows only 20 requests a day per model);
# LLM_PROVIDER=gemini or =groq in .env picks one explicitly
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or ("groq" if GROQ_API_KEY else "gemini")).strip().lower()
if LLM_PROVIDER not in ("groq", "gemini"):
    raise RuntimeError(f"LLM_PROVIDER must be 'groq' or 'gemini', not '{LLM_PROVIDER}'.")

# the active provider's details, for the agent and the UI
_PROVIDERS = {
    "groq": ("Groq", GROQ_API_KEY, "GROQ_API_KEY", GROQ_MODEL, "GROQ_MODEL"),
    "gemini": ("Gemini", GOOGLE_API_KEY, "GOOGLE_API_KEY", GEMINI_MODEL, "GEMINI_MODEL"),
}
LLM_NAME, LLM_API_KEY, LLM_KEY_SETTING, LLM_MODEL, LLM_MODEL_SETTING = _PROVIDERS[LLM_PROVIDER]

MAX_RETRIES = 3        # fix attempts after the first failed execution
EXEC_TIMEOUT_S = 10    # seconds allowed for generated code to run
HISTORY_TURNS = 5      # previous Q&A turns passed to the LLM for follow-ups

SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "sales.csv"


def has_api_key() -> bool:
    return bool(LLM_API_KEY) and not LLM_API_KEY.startswith("your_")


def require_api_key() -> str:
    """Return the active provider's key, or raise a readable error if it is missing."""
    if not has_api_key():
        raise RuntimeError(
            f"{LLM_KEY_SETTING} is not set. Copy .env.example to .env and add your {LLM_NAME} key."
        )
    return LLM_API_KEY
