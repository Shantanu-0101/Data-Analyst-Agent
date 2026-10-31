"""CSV loading, schema extraction and the schema text sent to the LLM."""
import io
import warnings
from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from src.prompts.templates import SCHEMA_PROMPT

DATE_PARSE_MIN_SHARE = 0.9  # convert a date-named column only if this share of its values parse
SAMPLE_ROWS = 3
MAX_CATEGORY_VALUES = 15    # list a text column's values only if it has this many or fewer
MAX_CELL_CHARS = 50         # long cells and category values are cut to this width
MAX_TABLE_COLS = 30         # columns shown in the sample rows and the numeric stats
MAX_SECTION_CHARS = 2500    # hard cap per prompt section, so wide data keeps the prompt small


class DataLoadError(ValueError):
    """The file could not be read as a CSV with data in it."""


def load_csv(file_or_path) -> pd.DataFrame:
    """Load a CSV from a path or a file-like object (e.g. a Streamlit UploadedFile).

    Tries UTF-8 first and falls back to latin-1. Columns named `date` or `*_date`
    are parsed to datetime when their values allow it.
    """
    data = _read_bytes(file_or_path)
    if not data.strip():
        raise DataLoadError("The file is empty.")
    if b"\x00" in data[:4096]:
        raise DataLoadError("The file looks binary, not a UTF-8 or latin-1 text CSV.")

    try:
        try:
            # utf-8-sig also strips a byte-order mark, which would otherwise end up in the first column name
            df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(data), encoding="latin-1")  # decodes any byte sequence
    except pd.errors.EmptyDataError:
        raise DataLoadError("The file has no columns.") from None
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"Could not parse the file as CSV: {exc}") from None

    if df.empty:
        raise DataLoadError("The file has a header but no data rows.")
    return _parse_date_columns(df)


def extract_schema(df: pd.DataFrame) -> dict:
    """Describe the frame: dtypes, size, sample rows, nulls, date ranges, text values, numeric stats."""
    numeric = df.select_dtypes(include="number")
    return {
        "columns": [(str(col), str(dtype)) for col, dtype in df.dtypes.items()],
        "row_count": len(df),
        "sample": df.head(SAMPLE_ROWS),
        "null_counts": {str(col): int(n) for col, n in df.isna().sum().items() if n},
        "date_ranges": _date_ranges(df),
        "categories": _category_values(df),
        "describe": numeric.describe() if numeric.shape[1] else None,
    }


def format_schema_info(df: pd.DataFrame) -> str:
    """Render the schema as the Schema Prompt text (stored as state["schema_info"])."""
    schema = extract_schema(df)
    sample = schema["sample"].to_string(index=False, max_cols=MAX_TABLE_COLS, max_colwidth=MAX_CELL_CHARS)
    stats = schema["describe"]
    return SCHEMA_PROMPT.format(
        columns_with_dtypes=", ".join(f"{col} ({dtype})" for col, dtype in schema["columns"]),
        row_count=schema["row_count"],
        sample_rows=_cap(sample),
        null_summary=", ".join(f"{col}: {n}" for col, n in schema["null_counts"].items()) or "none",
        date_ranges=", ".join(f"{col}: {lo} to {hi}" for col, (lo, hi) in schema["date_ranges"].items()) or "none",
        category_values=_cap(
            "; ".join(f"{col}: {', '.join(values)}" for col, values in schema["categories"].items()) or "none"
        ),
        describe_output=_cap(stats.round(2).to_string(max_cols=MAX_TABLE_COLS)) if stats is not None else "none",
    )


def dataset_summary(df: pd.DataFrame) -> dict:
    """Row, column and total-null counts for the sidebar."""
    return {"rows": len(df), "columns": df.shape[1], "nulls": int(df.isna().sum().sum())}


def _read_bytes(file_or_path) -> bytes:
    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise DataLoadError(f"Could not open {path}: {exc.strerror or exc}") from None
    # Streamlit's UploadedFile and BytesIO have getvalue(); other file objects only read()
    data = file_or_path.getvalue() if hasattr(file_or_path, "getvalue") else file_or_path.read()
    return data.encode("utf-8") if isinstance(data, str) else data


def _is_date_name(name) -> bool:
    name = str(name).strip().lower()
    return name == "date" or name.endswith("_date")


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        values = df[col]
        # only text columns: to_datetime would read an integer column as nanoseconds since 1970
        if not _is_date_name(col) or not (is_string_dtype(values) or is_object_dtype(values)):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # "could not infer format" noise
                parsed = pd.to_datetime(values, errors="coerce")
        except (ValueError, TypeError):  # e.g. mixed time zones
            continue
        present = values.notna().sum()
        if present and parsed.notna().sum() >= DATE_PARSE_MIN_SHARE * present:
            df[col] = parsed
    return df


def _date_ranges(df: pd.DataFrame) -> dict:
    ranges = {}
    for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        lo, hi = df[col].min(), df[col].max()
        if pd.notna(lo):
            ranges[str(col)] = (_format_timestamp(lo), _format_timestamp(hi))
    return ranges


def _format_timestamp(ts: pd.Timestamp) -> str:
    return str(ts.date()) if ts == ts.normalize() else str(ts)


def _category_values(df: pd.DataFrame) -> dict:
    """Distinct values of low-cardinality text columns, so the LLM filters on real values."""
    categories = {}
    for col in df.columns:
        values = df[col]
        if not (is_string_dtype(values) or is_object_dtype(values) or isinstance(values.dtype, pd.CategoricalDtype)):
            continue
        if values.nunique(dropna=True) <= MAX_CATEGORY_VALUES:
            categories[str(col)] = sorted(str(v)[:MAX_CELL_CHARS] for v in values.dropna().unique())
    return categories


def _cap(text: str, limit: int = MAX_SECTION_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "\n... (truncated)"
