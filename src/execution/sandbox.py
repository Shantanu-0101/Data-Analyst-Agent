"""Runs LLM-generated pandas/Plotly code in a restricted namespace with a timeout.

This guards against accidental damage (imports, file access, runaway loops, edits to the
user's data), not against code written on purpose to escape: exec() inside the app's own
process is not a security boundary.
"""
import ast
import builtins
import ctypes
import keyword
import linecache
import numbers
import re
import textwrap
import threading
import traceback
from pathlib import PurePath
from typing import Any, Optional, TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import EXEC_TIMEOUT_S

CODE_FILENAME = "<generated code>"  # how the code's own lines are named in tracebacks
MAX_ERROR_LINES = 30   # error text goes into the fix prompt, so keep it short
MAX_ERROR_CHARS = 3000
MAX_PROSE_LINES = 5    # chat text tolerated before or after code that has no ``` fences

SAFE_BUILTIN_NAMES = (
    "len", "range", "sum", "min", "max", "abs", "round", "sorted", "enumerate", "zip",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool", "isinstance", "any",
    "all", "reversed", "map", "filter",
    # so `try/except` in generated code can name what it catches
    "Exception", "ValueError", "KeyError", "TypeError", "IndexError", "ZeroDivisionError",
    "AttributeError",
)
SAFE_BUILTINS = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
# The prompt forbids print(), but code that prints and also sets RESULT is fine: ignore the
# print rather than fail and spend an LLM call on a fix.
SAFE_BUILTINS["print"] = lambda *args, **kwargs: None

_NO_STRING_CODE = "running code from strings is not allowed"
_NO_NAMESPACE = "reading the namespace is not allowed; use the variables directly"
_NO_DYNAMIC_ATTR = "use plain attribute access (obj.name) instead"
BLOCKED_NAMES = {
    "open": "files cannot be opened; the data is already loaded as df",
    "eval": _NO_STRING_CODE,
    "exec": _NO_STRING_CODE,
    "compile": _NO_STRING_CODE,
    "input": "the code cannot ask for input",
    "globals": _NO_NAMESPACE,
    "locals": _NO_NAMESPACE,
    "vars": _NO_NAMESPACE,
    "getattr": _NO_DYNAMIC_ATTR,
    "setattr": _NO_DYNAMIC_ATTR,
    "delattr": _NO_DYNAMIC_ATTR,
    "breakpoint": "the code cannot stop for debugging",
    "exit": "the code cannot end the program",
    "quit": "the code cannot end the program",
}

_NO_FILES = "reading or writing files is not allowed; df is already loaded and the answer goes in RESULT"
_FILE_ATTRS = (
    "to_pickle", "to_parquet", "to_feather", "to_hdf", "to_sql", "to_stata", "to_orc",
    "to_excel", "to_clipboard", "ExcelWriter", "HDFStore", "write_html", "write_image",
    "write_json", "tofile", "fromfile", "save", "savez", "savez_compressed", "savetxt",
    "load", "loadtxt", "genfromtxt", "memmap", "open",
)
# Modules that pandas and numpy import internally, reachable as e.g. pd.io.common.os
_SYSTEM_MODULES = (
    "os", "sys", "subprocess", "shutil", "builtins", "importlib", "pickle", "socket",
    "ctypes", "pathlib",
)
BLOCKED_ATTRS = {
    "show": "fig.show() is not needed; assign the figure to RESULT",
    **dict.fromkeys(_FILE_ATTRS, _NO_FILES),
    **dict.fromkeys(_SYSTEM_MODULES, "reaching system modules is not allowed"),
}
# pandas writers that return a string when called without a target; only a target is blocked
OPTIONAL_WRITERS = {"to_csv", "to_json", "to_html", "to_markdown", "to_latex", "to_xml", "to_string"}
WRITER_TARGET_KWARGS = {"path_or_buf", "buf", "path_or_buffer"}

_FENCE = re.compile(r"^[ \t]*```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:^[ \t]*```|\Z)", re.MULTILINE | re.DOTALL)
_PYTHON_TAGS = {"python", "py", "python3"}
# An optional list number, then a word (maybe ending in punctuation or **) followed by
# another word or the end of the line
_PROSE_LINE = re.compile(r"(?:\d+[.)]\s*)?([A-Za-z][\w'’]*)[!.:,]?[*_`]*(?:[ \t]+[A-Za-z(\"'“]|$)")
_MARKDOWN_CHARS = "*_#>`- \t"

_linecache_lock = threading.Lock()


class ExecResult(TypedDict):
    result: Any
    result_type: Optional[str]  # "dataframe" | "plotly" | "value" | "text"; None on error
    error: Optional[str]


class SandboxTimeout(BaseException):
    """Raised inside a runaway worker thread. A BaseException, so `except Exception` can't swallow it."""


def run_code(code: str, df: pd.DataFrame, timeout: float = EXEC_TIMEOUT_S) -> ExecResult:
    """Run generated code against a copy of df and return what it stored in RESULT.

    Never raises for problems in the code itself: syntax errors, rejected constructs,
    exceptions, timeouts and a missing RESULT all come back as `error` text, written so
    the fix prompt can act on it.
    """
    code = clean_code(code)
    if not code.strip():
        return _failure("No code was generated. Write Python code that assigns the answer to RESULT.")
    try:
        tree = ast.parse(code, filename=CODE_FILENAME)
    except SyntaxError as exc:
        return _failure("".join(traceback.format_exception_only(exc)).rstrip())

    problems = check_code(tree)
    if problems:
        return _failure("The code was rejected before running:\n" + "\n".join(f"- {p}" for p in problems))

    namespace = {"__builtins__": SAFE_BUILTINS, "pd": pd, "px": px, "np": np, "df": df.copy()}
    outcome = {}
    worker = threading.Thread(
        target=_execute, args=(compile(tree, CODE_FILENAME, "exec"), namespace, outcome),
        name="sandbox", daemon=True,
    )
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        _interrupt(worker)
        return _failure(
            f"Timed out: the code ran for more than {timeout:g} seconds. "
            "Use vectorised pandas operations instead of Python loops over rows."
        )

    if "exception" in outcome:
        return _failure(_format_exception(outcome["exception"], code))
    if "RESULT" not in namespace:
        return _failure("The code ran but did not set RESULT. Assign the final answer to a variable named RESULT.")
    return _classify(namespace["RESULT"])


def clean_code(text: str) -> str:
    """Pull the code out of an LLM reply: unwrap ``` fences and drop chat text around it."""
    blocks = _FENCE.findall(text)
    if blocks:
        python = [body for tag, body in blocks if tag.lower() in _PYTHON_TAGS]
        untagged = [body for tag, body in blocks if not tag]
        code = "\n".join(python or untagged or [blocks[0][1]])
    else:
        code = text
    code = textwrap.dedent(code).strip("\n")
    return _strip_prose(code)


def check_code(tree: ast.AST) -> list[str]:
    """Return the rule each disallowed construct breaks, as 'line N: reason', in line order."""
    problems = []
    for node in ast.walk(tree):
        reason = _rule_broken(node)
        if reason:
            problems.append((node.lineno, f"line {node.lineno}: {reason}"))
    return list(dict.fromkeys(text for _, text in sorted(problems)))


def _rule_broken(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "imports are not allowed; pd, px and np are already available"
    if isinstance(node, ast.ClassDef):
        return "class definitions are not supported; use functions or plain code"
    if isinstance(node, ast.Name):
        if node.id.startswith("__"):
            return f"`{node.id}`: names starting with __ are not allowed"
        if node.id in BLOCKED_NAMES:
            return f"`{node.id}` is not allowed: {BLOCKED_NAMES[node.id]}"
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__"):
            return f"`.{node.attr}`: attributes starting with __ are not allowed"
        if node.attr in BLOCKED_ATTRS:
            return f"`.{node.attr}` is not allowed: {BLOCKED_ATTRS[node.attr]}"
        if node.attr.startswith("read_"):
            return f"`.{node.attr}` is not allowed: {_NO_FILES}"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in OPTIONAL_WRITERS
        and (node.args or any(kw.arg is None or kw.arg in WRITER_TARGET_KWARGS for kw in node.keywords))
    ):
        return f"`.{node.func.attr}(...)` with a file target is not allowed; call it with no target to get a string"
    return None


def _execute(compiled, namespace: dict, outcome: dict) -> None:
    try:
        exec(compiled, namespace)
    except BaseException as exc:  # anything, including SystemExit and the injected timeout
        outcome["exception"] = exc


def _interrupt(thread: threading.Thread) -> None:
    """Raise SandboxTimeout inside a runaway worker thread.

    Python cannot kill a thread, so this is the best available: the exception fires at the
    thread's next bytecode, which stops a loop written in the generated code. A single long
    call into C code (a huge merge, say) still runs to the end first, and generated code
    with a bare `except:` around its loop can catch it; in both cases the daemon thread is
    abandoned and its result is never read.
    """
    thread_id = ctypes.c_ulong(thread.ident)
    hits = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SandboxTimeout))
    if hits > 1:  # can't happen for a valid id; undo rather than disturb other threads
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)


def _classify(result) -> ExecResult:
    if isinstance(result, pd.Series):
        result = _series_to_frame(result)
    if isinstance(result, pd.DataFrame):
        return _success(result, "dataframe")
    if isinstance(result, go.Figure):
        return _success(result, "plotly")
    if isinstance(result, (bool, np.bool_)):
        return _success(str(result), "text")
    if isinstance(result, numbers.Real):  # numpy scalars register here too
        return _success(result.item() if isinstance(result, np.generic) else result, "value")
    return _success(result if isinstance(result, str) else str(result), "text")


def _series_to_frame(series: pd.Series) -> pd.DataFrame:
    """Turn a Series into a two-column table: its index, then its values."""
    name = "value" if series.name is None else series.name
    if name in series.index.names:  # e.g. df.groupby("region")["region"].count()
        name = f"{name} (value)"
    try:
        return series.reset_index(name=name)
    except ValueError:  # an index level still clashes with the new column name
        return series.to_frame(name=name)


def _format_exception(exc: BaseException, code: str) -> str:
    """A short traceback: the generated code's own frames, the frame that raised, and the error.

    Frames inside pandas/numpy internals are dropped, and so are chained exceptions,
    since the fix prompt only needs to know which line failed and why.
    """
    tb = exc.__traceback__.tb_next if exc.__traceback__ else None  # skip _execute's own frame
    # Register the code's source so the traceback can quote the failing line.
    with _linecache_lock:
        linecache.cache[CODE_FILENAME] = (len(code), None, code.splitlines(keepends=True), CODE_FILENAME)
        try:
            summary = traceback.TracebackException(type(exc), exc, tb, compact=True)
        finally:
            linecache.cache.pop(CODE_FILENAME, None)
    frames = summary.stack
    kept = [frame for i, frame in enumerate(frames) if frame.filename == CODE_FILENAME or i == len(frames) - 1]
    for frame in kept:  # the error goes to the LLM: don't send local paths (and the username in them)
        parts = PurePath(frame.filename).parts
        if "site-packages" in parts:
            frame.filename = "/".join(parts[parts.index("site-packages") + 1:])
    summary.stack = traceback.StackSummary.from_list(kept)
    lines = "".join(summary.format(chain=False)).rstrip().splitlines()
    if len(lines) > MAX_ERROR_LINES:
        lines = lines[:MAX_ERROR_LINES] + [f"... ({len(lines) - MAX_ERROR_LINES} more lines)"]
    text = "\n".join(lines)
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS].rstrip() + "\n... (truncated)"


def _strip_prose(code: str) -> str:
    """Drop chat lines ("Here's the code:") before or after code that has no fences.

    Only lines that read like English are dropped, and only when that makes the code
    parse; otherwise the code is returned unchanged so its real syntax error is reported.
    """
    if _parses(code):
        return code
    lines = code.splitlines()
    lead = _droppable_run(lines)
    trail = _droppable_run(lines[::-1])
    for i in range(lead + 1):
        for j in range(trail + 1):
            candidate = "\n".join(lines[i:len(lines) - j])
            if candidate.strip() and _parses(candidate):
                return textwrap.dedent(candidate).strip("\n")
    return code


def _droppable_run(lines: list[str]) -> int:
    count = 0
    for line in lines[:MAX_PROSE_LINES]:
        if not _is_prose(line):
            break
        count += 1
    return count


def _is_prose(line: str) -> bool:
    text = line.strip(_MARKDOWN_CHARS)
    if not text:  # blank lines and markdown rules like "---"
        return True
    match = _PROSE_LINE.match(text)
    if not match or keyword.iskeyword(match[1]) or keyword.issoftkeyword(match[1]):
        return False
    return not _parses(line.strip())


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _success(result, result_type: str) -> ExecResult:
    return {"result": result, "result_type": result_type, "error": None}


def _failure(error: str) -> ExecResult:
    return {"result": None, "result_type": None, "error": error}
