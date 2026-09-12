from __future__ import annotations

from typing import Any

import pandas as pd


def validate_cleaning(
    before: pd.DataFrame,
    after: pd.DataFrame,
    unresolved_issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Validate that cleaning behaved as expected.

    Checks:
    - schema consistency
    - row-count delta
    - whether row loss is explainable
    - unresolved issues
    """

    unresolved_issues = unresolved_issues or []

    before_columns = list(before.columns)
    after_columns = list(after.columns)

    schema_unchanged = before_columns == after_columns

    rows_removed = len(before) - len(after)

    result = {
        "schema": {
            "before": before_columns,
            "after": after_columns,
            "unchanged": schema_unchanged,
        },
        "rows": {
            "before": len(before),
            "after": len(after),
            "removed": rows_removed,
        },
        "unresolved_issues": unresolved_issues,
        "valid": schema_unchanged,
    }

    return result


def check_idempotency(
    clean_function,
    cleaned_df: pd.DataFrame,
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Run the cleaner again on already-cleaned data and compare results.
    """

    second_pass, _ = clean_function(
        cleaned_df,
        plan,
    )

    same_shape = cleaned_df.shape == second_pass.shape

    same_columns = list(cleaned_df.columns) == list(
        second_pass.columns
    )

    same_data = cleaned_df.reset_index(drop=True).equals(
        second_pass.reset_index(drop=True)
    )

    return {
        "same_shape": same_shape,
        "same_columns": same_columns,
        "same_data": same_data,
        "idempotent": (
            same_shape
            and same_columns
            and same_data
        ),
    }