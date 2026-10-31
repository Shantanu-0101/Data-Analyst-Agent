import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.execution.sandbox import clean_code, run_code


@pytest.fixture
def df():
    return pd.DataFrame({
        "region": ["East", "West", "East", "North"],
        "product": ["A", "B", "A", "C"],
        "revenue": [100.0, 250.5, 75.0, 30.0],
        "quantity": [1, 3, 2, 1],
    })


def ok(code, df):
    out = run_code(code, df)
    assert out["error"] is None, out["error"]
    return out


def failed(code, df, **kwargs):
    out = run_code(code, df, **kwargs)
    assert out["result"] is None and out["result_type"] is None
    assert out["error"]
    return out["error"]


# --- result types ----------------------------------------------------------

def test_dataframe_result(df):
    out = ok("RESULT = df.head(2)", df)
    assert out["result_type"] == "dataframe"
    assert len(out["result"]) == 2


def test_series_becomes_two_column_table(df):
    out = ok("RESULT = df.groupby('region')['revenue'].sum()", df)
    assert out["result_type"] == "dataframe"
    assert list(out["result"].columns) == ["region", "revenue"]


def test_series_named_like_its_index(df):
    out = ok("RESULT = df.groupby('region')['region'].count()", df)
    assert list(out["result"].columns) == ["region", "region (value)"]


def test_unnamed_series(df):
    out = ok("RESULT = pd.Series([1, 2])", df)
    assert list(out["result"].columns) == ["index", "value"]


def test_plotly_result(df):
    out = ok("RESULT = px.bar(df, x='region', y='revenue', title='Revenue')", df)
    assert out["result_type"] == "plotly"
    assert isinstance(out["result"], go.Figure)


def test_numpy_number_becomes_plain_value(df):
    out = ok("RESULT = df['revenue'].sum()", df)
    assert out["result_type"] == "value"
    assert out["result"] == 455.5
    assert not isinstance(out["result"], np.generic)


def test_int_value(df):
    out = ok("RESULT = len(df)", df)
    assert (out["result_type"], out["result"]) == ("value", 4)


def test_text_result(df):
    out = ok("RESULT = f\"Top region is {df.groupby('region')['revenue'].sum().idxmax()}\"", df)
    assert (out["result_type"], out["result"]) == ("text", "Top region is West")


@pytest.mark.parametrize(("code", "text"), [
    ("RESULT = True", "True"),
    ("RESULT = df['revenue'].gt(0).all()", "True"),
    ("RESULT = ['a', 'b']", "['a', 'b']"),
    ("RESULT = None", "None"),
])
def test_other_results_become_text(df, code, text):
    out = ok(code, df)
    assert (out["result_type"], out["result"]) == ("text", text)


# --- rejected code ---------------------------------------------------------

@pytest.mark.parametrize(("code", "message"), [
    ("import os\nRESULT = 1", "line 1: imports are not allowed"),
    ("from os import path\nRESULT = 1", "imports are not allowed"),
    ("RESULT = open('x').read()", "`open` is not allowed"),
    ("RESULT = eval('1 + 1')", "`eval` is not allowed"),
    ("RESULT = getattr(df, 'shape')", "`getattr` is not allowed"),
    ("RESULT = ().__class__.__bases__", "`.__class__`"),
    ("RESULT = __builtins__", "`__builtins__`"),
    ("RESULT = pd.read_csv('sales.csv')", "`.read_csv` is not allowed"),
    ("df.to_excel('out.xlsx')\nRESULT = 1", "`.to_excel` is not allowed"),
    ("df.to_csv('out.csv')\nRESULT = 1", "`.to_csv(...)` with a file target"),
    ("df.to_csv(path_or_buf='out.csv')\nRESULT = 1", "`.to_csv(...)` with a file target"),
    ("RESULT = pd.io.common.os.getcwd()", "`.os` is not allowed"),
    ("fig = px.bar(df, x='region', y='revenue')\nfig.show()\nRESULT = fig", "fig.show() is not needed"),
    ("class A:\n    pass\nRESULT = 1", "class definitions are not supported"),
])
def test_disallowed_code_is_rejected_before_running(df, code, message):
    error = failed(code, df)
    assert error.startswith("The code was rejected before running")
    assert message in error


def test_every_problem_is_listed_in_line_order(df):
    error = failed("RESULT = 1\nimport os\nx = open('a')\nimport sys", df)
    assert error.splitlines()[1:] == [
        "- line 2: imports are not allowed; pd, px and np are already available",
        "- line 3: `open` is not allowed: files cannot be opened; the data is already loaded as df",
        "- line 4: imports are not allowed; pd, px and np are already available",
    ]


def test_string_writers_without_a_target_are_allowed(df):
    out = ok("RESULT = df.to_csv(index=False)", df)
    assert out["result"].startswith("region,product,revenue,quantity")


def test_builtins_outside_the_whitelist_are_missing(df):
    error = failed("RESULT = type(df)", df)
    assert "NameError" in error and "'type'" in error


def test_print_is_ignored(df):
    assert ok("print('hello')\nRESULT = 1", df)["result"] == 1


# --- runtime failures ------------------------------------------------------

def test_missing_result(df):
    assert "did not set RESULT" in failed("x = df.head()", df)


def test_empty_code(df):
    assert "No code was generated" in failed("```python\n```", df)


def test_syntax_error(df):
    error = failed("RESULT = df.groupby('region'", df)
    assert "SyntaxError" in error
    assert '"<generated code>", line 1' in error


def test_wrong_column_gives_short_traceback_with_the_failing_line(df):
    error = failed("x = 1\nRESULT = df['nope'].sum()", df)
    assert '"<generated code>", line 2' in error
    assert "RESULT = df['nope'].sum()" in error  # the source line is quoted
    assert error.splitlines()[-1] == "KeyError: 'nope'"
    assert "During handling" not in error and "direct cause" not in error  # no chained pandas internals
    assert len(error.splitlines()) <= 30
    assert "site-packages" not in error and '"pandas/' in error  # no local paths sent to the LLM
    assert "raise KeyError" in error  # library source lines are still quoted


def test_error_inside_a_function_shows_both_frames(df):
    error = failed("def f(row):\n    return row['nope']\nRESULT = df.apply(f, axis=1)", df)
    assert '"<generated code>", line 3' in error
    assert '"<generated code>", line 2, in f' in error


def test_long_error_text_is_capped(df):
    error = failed("raise ValueError('x' * 10000)", df)
    assert len(error) < 3100
    assert error.endswith("(truncated)")


def test_infinite_loop_times_out_and_its_thread_stops(df):
    started = time.monotonic()
    error = failed("while True:\n    pass", df, timeout=0.5)
    assert "Timed out" in error and "0.5 seconds" in error
    assert time.monotonic() - started < 2
    deadline = time.monotonic() + 2
    while any(t.name == "sandbox" for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(t.name == "sandbox" for t in threading.enumerate())


def test_timeout_is_not_swallowed_by_except_exception(df):
    code = "while True:\n    try:\n        pass\n    except Exception:\n        pass"
    assert "Timed out" in failed(code, df, timeout=0.3)


# --- isolation -------------------------------------------------------------

def test_runs_do_not_share_variables(df):
    ok("leftover = 42\nRESULT = 1", df)
    assert "NameError" in failed("RESULT = leftover", df)


def test_users_dataframe_is_not_modified(df):
    before = df.copy()
    ok("df.pop('revenue')\ndf['new'] = 1\ndf.loc[0, 'region'] = 'X'\nRESULT = df", df)
    pd.testing.assert_frame_equal(df, before)


def test_functions_see_variables_defined_in_the_code(df):
    assert ok("n = 2\ndef double(x):\n    return x * n\nRESULT = double(3)", df)["result"] == 6


# --- cleaning LLM replies --------------------------------------------------

def test_python_fence_is_unwrapped():
    reply = "Here is the code:\n```python\nRESULT = df.head()\n```\nThis shows the first rows."
    assert clean_code(reply) == "RESULT = df.head()"


def test_untagged_and_unclosed_fences():
    assert clean_code("```\nRESULT = 1\n```") == "RESULT = 1"
    assert clean_code("```python\nRESULT = 1\n") == "RESULT = 1"


def test_several_python_blocks_are_joined_and_other_languages_ignored():
    reply = "```python\nx = 1\n```\nthen\n```py\nRESULT = x\n```\n```text\noutput\n```"
    assert clean_code(reply) == "x = 1\n\nRESULT = x"


def test_prose_around_unfenced_code_is_dropped():
    reply = "Sure! Here’s the code:\n\nRESULT = (df\n    .head())\n\n**Explanation:** this shows the first rows."
    assert clean_code(reply) == "RESULT = (df\n    .head())"


def test_valid_code_is_left_alone():
    code = "total = df['revenue'].sum()\nRESULT = f'Total: {total}'"
    assert clean_code(code) == code


def test_real_syntax_error_is_not_hidden_by_dropping_lines():
    code = "x = df.groupby('a'\nRESULT = 1"
    assert clean_code(code) == code


def test_indented_code_is_dedented():
    assert clean_code("    x = 1\n    RESULT = x") == "x = 1\nRESULT = x"
