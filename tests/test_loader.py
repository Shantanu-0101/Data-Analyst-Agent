import io

import pandas as pd
import pytest

from src.config import SAMPLE_CSV
from src.data.loader import (
    MAX_SECTION_CHARS,
    DataLoadError,
    dataset_summary,
    extract_schema,
    format_schema_info,
    load_csv,
)


@pytest.fixture(scope="module")
def sales():
    return load_csv(SAMPLE_CSV)


def test_sample_csv_loads_with_parsed_dates(sales):
    assert sales.shape == (1500, 6)
    assert pd.api.types.is_datetime64_any_dtype(sales["date"])


def test_schema_info_has_every_column_and_row_count(sales):
    text = format_schema_info(sales)
    for col in sales.columns:
        assert col in text
    assert "Row count: 1500" in text
    assert "Date ranges: date: 2023-01-01 to 2024-12-31" in text
    assert "region: East, North, South, West" in text


def test_extract_schema_reports_only_nonzero_nulls(sales):
    schema = extract_schema(sales)
    assert schema["null_counts"] == {"quantity": 8, "salesperson": 12}
    assert len(schema["sample"]) == 3
    assert set(schema["describe"].columns) == {"revenue", "quantity"}


def test_dataset_summary(sales):
    assert dataset_summary(sales) == {"rows": 1500, "columns": 6, "nulls": 20}


def test_latin1_fallback():
    df = load_csv(io.BytesIO("name,city\nJosé,Zürich\n".encode("latin-1")))
    assert df.loc[0, "city"] == "Zürich"


def test_utf8_bom_is_stripped_from_first_column():
    df = load_csv(io.BytesIO("﻿date,value\n2024-01-05,1\n".encode("utf-8")))
    assert list(df.columns) == ["date", "value"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_path_as_string(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("x,y\n1,2\n")
    assert load_csv(str(path)).shape == (1, 2)


def test_missing_file():
    with pytest.raises(DataLoadError, match="Could not open"):
        load_csv("does/not/exist.csv")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"   \n\n", "empty"),
        (b"a,b,c\n", "no data rows"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "binary"),
    ],
)
def test_unusable_files_raise_readable_errors(content, message):
    with pytest.raises(DataLoadError, match=message):
        load_csv(io.BytesIO(content))


def test_date_named_column_with_junk_stays_text():
    df = load_csv(io.BytesIO(b"date,v\nsoon,1\nlater,2\n2024-01-01,3\n"))
    assert not pd.api.types.is_datetime64_any_dtype(df["date"])


def test_numeric_date_column_is_not_read_as_epoch():
    df = load_csv(io.BytesIO(b"order_date,v\n20240105,1\n20240106,2\n"))
    assert pd.api.types.is_integer_dtype(df["order_date"])


def test_no_numeric_columns():
    df = load_csv(io.BytesIO(b"a,b\nx,y\n"))
    assert "Numeric stats:\nnone" in format_schema_info(df)


def test_wide_frame_keeps_prompt_small():
    wide = pd.DataFrame({f"col_{i}": ["x" * 500] * 5 for i in range(200)})
    wide["n"] = range(5)
    text = format_schema_info(wide)
    # the column list is always complete; the sample and stats sections are capped
    assert "col_199 (" in text
    assert len(text) < 3 * MAX_SECTION_CHARS + 8000
