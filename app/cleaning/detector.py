from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _looks_numeric(value: object) -> bool:
    """
    Return True if a value can reasonably be interpreted as a number
    after removing common formatting characters.
    """
    if pd.isna(value):
        return False

    text = str(value).strip()

    if not text:
        return False

    cleaned = re.sub(r"[₹,$,\s]", "", text)

    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _looks_date(value: object) -> bool:
    """
    Return True if a value can be parsed as a date.
    """
    if pd.isna(value):
        return False

    text = str(value).strip()

    if not text:
        return False

    parsed = pd.to_datetime(text, errors="coerce")

    return not pd.isna(parsed)


def detect_duplicate_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect completely duplicated rows.
    """

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return []

    return [
        {
            "issue_id": "ISSUE-DUP-001",
            "type": "duplicate_rows",
            "field": None,
            "description": f"Found {duplicate_count} duplicated row(s).",
            "risk": "medium",
            "source": "code",
        }
    ]


def detect_whitespace_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect leading/trailing whitespace in string columns.
    """

    issues = []

    for column in df.select_dtypes(include=["object", "string"]).columns:
        mask = df[column].astype(str).str.strip() != df[column].astype(str)

        if mask.any():
            count = int(mask.sum())

            issues.append(
                {
                    "issue_id": f"ISSUE-WS-{column}",
                    "type": "whitespace",
                    "field": column,
                    "description": (
                        f"Found {count} value(s) with leading or trailing "
                        "whitespace."
                    ),
                    "risk": "low",
                    "source": "code",
                }
            )

    return issues


def detect_case_variants(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect values that differ only by capitalization.
    """

    issues = []

    for column in df.select_dtypes(include=["object", "string"]).columns:
        series = df[column].dropna().astype(str).str.strip()

        normalized = series.str.lower()

        if normalized.nunique() < series.nunique():
            issues.append(
                {
                    "issue_id": f"ISSUE-CASE-{column}",
                    "type": "case_variants",
                    "field": column,
                    "description": (
                        "Multiple values differ only by capitalization."
                    ),
                    "risk": "low",
                    "source": "code",
                }
            )

    return issues


def detect_numeric_formatting(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect string columns that contain values which look numeric
    but have formatting inconsistencies.
    """

    issues = []

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].dropna()

        if values.empty:
            continue

        numeric_like = values.map(_looks_numeric)

        if numeric_like.all() and len(values) > 1:
            raw_values = values.astype(str)

            has_formatting = raw_values.str.contains(
                r"[₹,$]",
                regex=True,
            ).any()

            has_commas = raw_values.str.contains(
                ",",
                regex=False,
            ).any()

            if has_formatting or has_commas:
                issues.append(
                    {
                        "issue_id": f"ISSUE-NUM-{column}",
                        "type": "numeric_formatting",
                        "field": column,
                        "description": (
                            "Column contains numeric values represented "
                            "with formatting such as currency symbols or "
                            "thousands separators."
                        ),
                        "risk": "medium",
                        "source": "code",
                    }
                )

    return issues


def detect_date_formatting(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Detect columns that appear to contain dates represented
    in inconsistent formats.
    """

    issues = []

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].dropna()

        if values.empty:
            continue

        date_like = values.map(_looks_date)

        if not date_like.all():
            continue

        values = values.astype(str).str.strip()

        has_iso = values.str.match(r"^\d{4}-\d{2}-\d{2}$").any()

        has_slash = values.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").any()

        if has_iso and has_slash:
            issues.append(
                {
                    "issue_id": f"ISSUE-DATE-{column}",
                    "type": "date_formatting",
                    "field": column,
                    "description": (
                        "Date values use multiple formats."
                    ),
                    "risk": "medium",
                    "source": "code",
                }
            )

    return issues


def detect_issues(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Run all deterministic data-quality detectors.
    """

    issues: list[dict[str, Any]] = []

    issues.extend(detect_duplicate_rows(df))
    issues.extend(detect_whitespace_issues(df))
    issues.extend(detect_case_variants(df))
    issues.extend(detect_numeric_formatting(df))
    issues.extend(detect_date_formatting(df))

    return issues