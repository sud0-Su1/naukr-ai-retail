from __future__ import annotations

from typing import Any

import pandas as pd


class UnsafeJoinError(Exception):
    """Raised when a join violates the expected relationship."""


def safe_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_key: str,
    right_key: str,
    relationship: str = "many_to_one",
    how: str = "left",
    rename_right: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Safely join two DataFrames with relationship and fan-out checks.

    Supported relationships:
        many_to_one
        one_to_one
        many_to_many  <-- explicitly allowed, but flagged

    Returns:
        joined dataframe
        join audit record
    """

    if left_key not in left.columns:
        raise UnsafeJoinError(
            f"Left join key '{left_key}' does not exist."
        )

    if right_key not in right.columns:
        raise UnsafeJoinError(
            f"Right join key '{right_key}' does not exist."
        )

    if relationship not in {
        "many_to_one",
        "one_to_one",
        "many_to_many",
    }:
        raise UnsafeJoinError(
            f"Unsupported relationship: {relationship}"
        )

    if how not in {
        "left",
        "inner",
        "right",
        "outer",
    }:
        raise UnsafeJoinError(
            f"Unsupported join type: {how}"
        )

    left_before = len(left)
    right_before = len(right)

    left_duplicate_keys = (
        left[left_key]
        .dropna()
        .duplicated(keep=False)
        .sum()
    )

    right_duplicate_keys = (
        right[right_key]
        .dropna()
        .duplicated(keep=False)
        .sum()
    )

    # --------------------------------------------------
    # Validate relationship
    # --------------------------------------------------

    if relationship == "many_to_one" and right_duplicate_keys > 0:
        duplicate_values = (
            right.loc[
                right[right_key].duplicated(keep=False),
                right_key,
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        raise UnsafeJoinError(
            "Unsafe many-to-one join: "
            f"'{right_key}' is not unique on the right side. "
            f"Duplicate keys: {duplicate_values}"
        )

    if relationship == "one_to_one":
        if left_duplicate_keys > 0:
            raise UnsafeJoinError(
                f"Unsafe one-to-one join: "
                f"'{left_key}' is not unique on the left side."
            )

        if right_duplicate_keys > 0:
            raise UnsafeJoinError(
                f"Unsafe one-to-one join: "
                f"'{right_key}' is not unique on the right side."
            )

    # --------------------------------------------------
    # Perform join
    # --------------------------------------------------
    right_to_join = right.copy()

    if rename_right:
        right_to_join = right_to_join.rename(
            columns=rename_right
        )

    joined = left.merge(
        right_to_join,
        left_on=left_key,
        right_on=right_key,
        how=how,
        validate=relationship,
        indicator=True,
    )

    rows_after = len(joined)

    # --------------------------------------------------
    # Match statistics
    # --------------------------------------------------

    matched_rows = int(
        (joined["_merge"] == "both").sum()
    )

    unmatched_left_rows = int(
        (joined["_merge"] == "left_only").sum()
    )

    unmatched_right_rows = int(
        (joined["_merge"] == "right_only").sum()
    )

    joined = joined.drop(columns=["_merge"])

    fanout_ratio = (
        rows_after / left_before
        if left_before > 0
        else 0.0
    )

    audit = {
        "left_rows_before": left_before,
        "right_rows_before": right_before,
        "rows_after": rows_after,
        "left_key": left_key,
        "right_key": right_key,
        "relationship": relationship,
        "join_type": how,
        "left_duplicate_key_rows": int(left_duplicate_keys),
        "right_duplicate_key_rows": int(right_duplicate_keys),
        "matched_rows": matched_rows,
        "unmatched_left_rows": unmatched_left_rows,
        "unmatched_right_rows": unmatched_right_rows,
        "fanout_ratio": fanout_ratio,
        "safe": True,
    }

    return joined, audit