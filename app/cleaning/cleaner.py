from __future__ import annotations

import re
from typing import Any

import pandas as pd


def normalize_numeric_value(value: object) -> float | None:
    """
    Convert values such as:
        1299
        2,499
        ₹1799

    into numeric values.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    text = re.sub(r"[₹,$\s]", "", text)

    try:
        return float(text)
    except ValueError:
        return None


def normalize_date_value(value: object) -> str | None:
    """
    Normalize supported date formats into YYYY-MM-DD.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        parsed = pd.to_datetime(
            text,
            format="%Y-%m-%d",
            errors="raise",
        )
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass

    try:
        parsed = pd.to_datetime(
            text,
            format="%d/%m/%Y",
            errors="raise",
        )
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _copy_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Make a safe copy of the cleaning plan without mutating the caller's
    objects.
    """
    return [
        {
            **step,
            "fields": list(step.get("fields", [])),
            "details": dict(step.get("details", {})),
        }
        for step in plan
    ]


def _mark_unresolved(
    step: dict[str, Any],
    reason: str,
) -> None:
    """
    Mark a planned step as unresolved instead of silently ignoring it.
    """
    step["status"] = "unresolved"
    step["details"] = {
        **step.get("details", {}),
        "reason": reason,
    }


def clean_orders(
    df: pd.DataFrame,
    plan: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Apply only the transformations explicitly represented in the
    cleaning plan.

    The function is deterministic and does not modify the input DataFrame.

    Returns:
        cleaned DataFrame
        updated cleaning plan
    """

    cleaned = df.copy(deep=True)
    updated_plan = _copy_plan(plan)

    for step in updated_plan:
        step_id = step.get("step_id")
        fields = step.get("fields", [])

        # --------------------------------------------------
        # CLEAN-001: Remove exact duplicate rows
        # --------------------------------------------------
        if step_id == "CLEAN-001":
            before_rows = len(cleaned)

            cleaned = cleaned.drop_duplicates()

            after_rows = len(cleaned)

            step["status"] = "applied"
            step["details"] = {
                "rows_before": before_rows,
                "rows_after": after_rows,
                "rows_removed": before_rows - after_rows,
            }

        # --------------------------------------------------
        # CLEAN-002 / numeric formatting
        # --------------------------------------------------
        elif step_id == "CLEAN-002" or step_id.startswith("CLEAN-NUM-"):
            if not fields:
                _mark_unresolved(
                    step,
                    "No target field was provided for numeric normalization.",
                )
                continue

            field = fields[0]

            if field not in cleaned.columns:
                _mark_unresolved(
                    step,
                    f"Target field '{field}' does not exist.",
                )
                continue

            before_nulls = int(cleaned[field].isna().sum())

            cleaned[field] = cleaned[field].map(
                normalize_numeric_value
            )

            after_nulls = int(cleaned[field].isna().sum())

            step["status"] = "applied"
            step["details"] = {
                "target_type": "float",
                "field": field,
                "nulls_before": before_nulls,
                "nulls_after": after_nulls,
            }

        # --------------------------------------------------
        # CLEAN-003 / date formatting
        # --------------------------------------------------
        elif step_id == "CLEAN-003" or step_id.startswith("CLEAN-DATE-"):
            if not fields:
                _mark_unresolved(
                    step,
                    "No target field was provided for date normalization.",
                )
                continue

            field = fields[0]

            if field not in cleaned.columns:
                _mark_unresolved(
                    step,
                    f"Target field '{field}' does not exist.",
                )
                continue

            before_nulls = int(cleaned[field].isna().sum())

            cleaned[field] = cleaned[field].map(
                normalize_date_value
            )

            after_nulls = int(cleaned[field].isna().sum())

            step["status"] = "applied"
            step["details"] = {
                "target_format": "YYYY-MM-DD",
                "field": field,
                "nulls_before": before_nulls,
                "nulls_after": after_nulls,
            }

        # --------------------------------------------------
        # Whitespace normalization
        # --------------------------------------------------
        elif step_id.startswith("CLEAN-WS-"):
            if not fields:
                _mark_unresolved(
                    step,
                    "No target field was provided for whitespace normalization.",
                )
                continue

            field = fields[0]

            if field not in cleaned.columns:
                _mark_unresolved(
                    step,
                    f"Target field '{field}' does not exist.",
                )
                continue

            if not (
                pd.api.types.is_object_dtype(cleaned[field])
                or pd.api.types.is_string_dtype(cleaned[field])
            ):
                _mark_unresolved(
                    step,
                    f"Field '{field}' is not a string-like column.",
                )
                continue

            before = cleaned[field].copy()

            cleaned[field] = cleaned[field].map(
                lambda value: (
                    value.strip()
                    if isinstance(value, str)
                    else value
                )
            )

            changed = int(
                (before.astype("string") != cleaned[field].astype("string"))
                .fillna(False)
                .sum()
            )

            step["status"] = "applied"
            step["details"] = {
                "field": field,
                "values_changed": changed,
            }

        # --------------------------------------------------
        # Case normalization
        # --------------------------------------------------
        elif step_id.startswith("CLEAN-CASE-"):
            if not fields:
                _mark_unresolved(
                    step,
                    "No target field was provided for case normalization.",
                )
                continue

            field = fields[0]

            if field not in cleaned.columns:
                _mark_unresolved(
                    step,
                    f"Target field '{field}' does not exist.",
                )
                continue

            if not (
                pd.api.types.is_object_dtype(cleaned[field])
                or pd.api.types.is_string_dtype(cleaned[field])
            ):
                _mark_unresolved(
                    step,
                    f"Field '{field}' is not a string-like column.",
                )
                continue

            before = cleaned[field].copy()

            cleaned[field] = cleaned[field].map(
                lambda value: (
                    value.lower()
                    if isinstance(value, str)
                    else value
                )
            )

            changed = int(
                (before.astype("string") != cleaned[field].astype("string"))
                .fillna(False)
                .sum()
            )

            step["status"] = "applied"
            step["details"] = {
                "field": field,
                "normalization": "lowercase",
                "values_changed": changed,
            }

        # --------------------------------------------------
        # Unknown plan step
        # --------------------------------------------------
        else:
            _mark_unresolved(
                step,
                f"Cleaning step '{step_id}' is not implemented.",
            )

    return cleaned, updated_plan
