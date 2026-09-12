from pathlib import Path
from typing import Any

import pandas as pd


def profile_dataframe(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """
    Generate a data-quality profile for a DataFrame.

    This function does NOT modify the data.
    """

    profile: dict[str, Any] = {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "schema": {},
        "nulls": {},
        "duplicates": {},
        "cardinality": {},
        "numeric_distributions": {},
    }

    # -------------------------
    # Schema
    # -------------------------
    for column in df.columns:
        profile["schema"][column] = {
            "dtype": str(df[column].dtype),
            "non_null": int(df[column].notna().sum()),
        }

    # -------------------------
    # Null counts
    # -------------------------
    for column in df.columns:
        profile["nulls"][column] = int(df[column].isna().sum())

    # -------------------------
    # Duplicate records
    # -------------------------
    duplicate_mask = df.duplicated(keep=False)

    profile["duplicates"] = {
        "duplicate_rows": int(duplicate_mask.sum()),
        "duplicate_groups": int(df.duplicated().sum()),
    }

    # -------------------------
    # Cardinality
    # -------------------------
    for column in df.columns:
        profile["cardinality"][column] = int(df[column].nunique(dropna=True))

    # -------------------------
    # Numeric distributions
    # -------------------------
    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()

        if len(series) == 0:
            continue

        profile["numeric_distributions"][column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
        }

    return profile


def profile_csv(file_path: str | Path) -> dict[str, Any]:
    """
    Read a CSV file and generate its profile.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    return profile_dataframe(
        df=df,
        dataset_name=path.name,
    )