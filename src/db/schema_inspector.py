"""
schema_inspector.py

Given a folder of CSVs (or a single CSV), profiles each table:
- columns, inferred types, row counts
- sample rows
- data quality flags: columns that look numeric but are stored as text,
  presence of non-standard missing-value markers (e.g. '\\N'), null rates

This module has NO knowledge of any specific dataset (e.g. F1). It must
work generically on any tabular data handed to it.
"""

import os
import re
import duckdb
import pandas as pd
from dataclasses import dataclass, field


# Common "fake null" markers seen in real-world CSVs beyond pandas' defaults
KNOWN_MISSING_MARKERS = [r"\N", "NULL", "null", "N/A", "n/a", "-", "?", ""]

# Regex to detect values that are numeric-looking but stored as strings,
# e.g. "1:34:50.616" is NOT numeric, but "42" or "3.14" stored as text IS
NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_pct: float
    sample_values: list = field(default_factory=list)
    looks_numeric_but_stored_as_text: bool = False
    missing_markers_found: list = field(default_factory=list)


@dataclass
class TableProfile:
    name: str
    row_count: int
    column_count: int
    columns: list  # list[ColumnProfile]


def _is_text_like(series: pd.Series) -> bool:
    """True for classic 'object' dtype AND the newer pandas string dtypes
    (e.g. pyarrow-backed 'str'/'string' dtype). A plain `dtype == object`
    check misses the newer string dtype, which silently hides real quirks."""
    return (
        series.dtype == object
        or pd.api.types.is_string_dtype(series)
    ) and not pd.api.types.is_numeric_dtype(series)


def _detect_missing_markers(series: pd.Series) -> list:
    """Check a text column for known 'fake null' markers beyond real NaN."""
    if not _is_text_like(series):
        return []
    found = []
    value_set = set(series.dropna().astype(str).unique()[:1000])  # cap for speed
    for marker in KNOWN_MISSING_MARKERS:
        if marker and marker in value_set:
            found.append(marker)
    return found


def _looks_numeric_but_is_text(series: pd.Series) -> bool:
    """Flag columns stored as text whose non-null values are
    actually numeric-looking (e.g. a 'position' column read as strings)."""
    if not _is_text_like(series):
        return False
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    numeric_like = sample.apply(lambda v: bool(NUMERIC_PATTERN.match(v.strip())))
    # If most non-null values are numeric-looking, but pandas stored it as
    # text, that's a strong signal of the F1-style '\N'-in-numeric-column issue
    return numeric_like.mean() > 0.5 and numeric_like.mean() < 1.0


def profile_table(df: pd.DataFrame, table_name: str) -> TableProfile:
    """Profile a single dataframe into a TableProfile."""
    columns = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        missing_markers = _detect_missing_markers(series)
        looks_numeric = _looks_numeric_but_is_text(series)

        columns.append(ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            null_count=null_count,
            null_pct=round(null_count / max(len(series), 1) * 100, 2),
            sample_values=series.dropna().astype(str).unique()[:5].tolist(),
            looks_numeric_but_stored_as_text=looks_numeric,
            missing_markers_found=missing_markers,
        ))

    return TableProfile(
        name=table_name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
    )


def profile_directory(data_dir: str) -> dict:
    """Profile every CSV in a directory. Returns {table_name: TableProfile}."""
    profiles = {}
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    for csv_file in csv_files:
        table_name = csv_file.replace(".csv", "")
        path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(path, low_memory=False)
        profiles[table_name] = profile_table(df, table_name)

    return profiles


def load_into_duckdb(data_dir: str, con: duckdb.DuckDBPyConnection = None) -> duckdb.DuckDBPyConnection:
    """Load every CSV in a directory into a DuckDB connection as tables."""
    if con is None:
        con = duckdb.connect(database=":memory:")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    for csv_file in csv_files:
        table_name = csv_file.replace(".csv", "")
        path = os.path.join(data_dir, csv_file).replace("\\", "/")
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{path}')
        """)
    return con


def summarize_profiles_for_agent(profiles: dict) -> str:
    """Turn table profiles into a compact text summary an LLM can reason over."""
    lines = []
    for name, profile in profiles.items():
        lines.append(f"\n### Table: {name} ({profile.row_count} rows, {profile.column_count} cols)")
        for col in profile.columns:
            flags = []
            if col.looks_numeric_but_stored_as_text:
                flags.append("⚠ numeric-looking but stored as TEXT")
            if col.missing_markers_found:
                flags.append(f"⚠ missing markers found: {col.missing_markers_found}")
            if col.null_pct > 0:
                flags.append(f"{col.null_pct}% null")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  - {col.name} ({col.dtype}){flag_str}: e.g. {col.sample_values}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual test against the F1 dataset
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    profiles = profile_directory(data_dir)
    print(summarize_profiles_for_agent(profiles))
