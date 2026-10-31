from string import Formatter

import pytest

from src.config import HISTORY_TURNS
from src.prompts.templates import (
    CODE_PROMPT, FIX_PROMPT, NO_HISTORY, PLAN_PROMPT, SCHEMA_PROMPT, format_history,
)

TEMPLATE_FIELDS = {
    "SCHEMA_PROMPT": (SCHEMA_PROMPT, {
        "columns_with_dtypes", "row_count", "sample_rows", "null_summary", "date_ranges",
        "category_values", "describe_output",
    }),
    "PLAN_PROMPT": (PLAN_PROMPT, {"question", "history"}),
    "CODE_PROMPT": (CODE_PROMPT, {"question", "plan", "history"}),
    "FIX_PROMPT": (FIX_PROMPT, {"question", "plan", "error", "code"}),
}


def fields(template):
    return {name for _, name, _, _ in Formatter().parse(template) if name is not None}


@pytest.mark.parametrize("name", TEMPLATE_FIELDS)
def test_template_has_exactly_the_expected_fields(name):
    template, expected = TEMPLATE_FIELDS[name]
    assert fields(template) == expected


@pytest.mark.parametrize("name", TEMPLATE_FIELDS)
def test_template_formats_with_dummy_values(name):
    template, expected = TEMPLATE_FIELDS[name]
    text = template.format(**{field: f"<{field}>" for field in expected})
    for field in expected:
        assert f"<{field}>" in text


def test_escaped_braces_render_as_single_braces():
    text = CODE_PROMPT.format(question="q", plan="p", history="h")
    assert 'f"{name} has the highest average ({value:,.2f})."' in text
    assert "{{" not in text


def test_code_and_fix_prompts_share_the_rules():
    code = CODE_PROMPT.format(question="q", plan="p", history="h")
    fix = FIX_PROMPT.format(question="q", plan="p", error="e", code="c")
    for rule in ("RESULT", "do NOT import", "Do NOT read or write files", "fig.show()"):
        assert rule in code and rule in fix


def test_values_with_braces_are_not_reformatted():
    code = 'RESULT = df.rename(columns={"a": "b"})'
    text = FIX_PROMPT.format(question="q", plan="p", error="KeyError: 'x'", code=code)
    assert code in text


# --- format_history ----------------------------------------------------------

def test_empty_history():
    assert format_history([]) == NO_HISTORY
    assert format_history(None) == NO_HISTORY


def test_history_shows_question_code_and_result():
    text = format_history([
        {"question": "Revenue by region", "code": "RESULT = df.groupby('region')['revenue'].sum()\n",
         "result_summary": "table, 4 rows"},
    ])
    assert "Q1: Revenue by region" in text
    assert "```python\nRESULT = df.groupby('region')['revenue'].sum()\n```" in text
    assert "Result: table, 4 rows" in text


def test_history_skips_missing_code_and_summary():
    text = format_history([{"question": "Hello?"}, {"question": "Total?", "code": "", "result_summary": None}])
    assert text == "Q1: Hello?\n\nQ2: Total?"


def test_history_keeps_only_the_most_recent_turns():
    history = [{"question": f"question {i}"} for i in range(HISTORY_TURNS + 3)]
    text = format_history(history)
    assert "question 2" not in text
    assert "question 3" in text and f"question {HISTORY_TURNS + 2}" in text
    assert f"Q{HISTORY_TURNS}:" in text and f"Q{HISTORY_TURNS + 1}:" not in text


def test_history_goes_into_the_prompts():
    history = format_history([{"question": "Total revenue?", "code": "RESULT = df['revenue'].sum()"}])
    assert history in PLAN_PROMPT.format(question="now only 2024", history=history)
    assert history in CODE_PROMPT.format(question="now only 2024", plan="p", history=history)
