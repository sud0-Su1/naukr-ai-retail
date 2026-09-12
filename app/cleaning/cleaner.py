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

    # Remove currency symbols, commas and spaces.
    text = re.sub(r"[₹,$\s]", "", text)

    try:
        return float(text)
    except ValueError:
        return None


def normalize_date_value(value: object) -> str | None:
    """
    Normalize supported date formats into YYYY-MM-DD.

    Supported examples:
        2025-01-10
        10/02/2025
        03/01/2025
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    # ISO format: YYYY-MM-DD
    try:
        parsed = pd.to_datetime(
            text,
            format="%Y-%m-%d",
            errors="raise",
        )
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass

    # Slash format: DD/MM/YYYY
    try:
        parsed = pd.to_datetime(
            text,
            format="%d/%m/%Y",
            errors="raise",
        )
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def clean_orders(
    df: pd.DataFrame,
    plan: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Apply the approved cleaning plan to an orders DataFrame.

    Returns:
        cleaned DataFrame
        updated cleaning plan
    """

    cleaned = df.copy(deep=True)
    updated_plan = [step.copy() for step in plan]

    # --------------------------------------------------
    # CLEAN-001: Remove exact duplicate rows
    # --------------------------------------------------
    step = next(
        step
        for step in updated_plan
        if step["step_id"] == "CLEAN-001"
    )

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
    # CLEAN-002: Normalize unit_price
    # --------------------------------------------------
    step = next(
        step
        for step in updated_plan
        if step["step_id"] == "CLEAN-002"
    )

    cleaned["unit_price"] = cleaned["unit_price"].map(
        normalize_numeric_value
    )

    step["status"] = "applied"
    step["details"] = {
        "target_type": "float",
    }

    # --------------------------------------------------
    # CLEAN-003: Normalize order_date
    # --------------------------------------------------
    step = next(
        step
        for step in updated_plan
        if step["step_id"] == "CLEAN-003"
    )

    cleaned["order_date"] = cleaned["order_date"].map(
        normalize_date_value
    )

    step["status"] = "applied"
    step["details"] = {
        "target_format": "YYYY-MM-DD",
    }

    return cleaned, updated_plan