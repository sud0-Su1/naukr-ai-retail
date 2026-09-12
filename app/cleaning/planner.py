from __future__ import annotations

from typing import Any


def build_cleaning_plan(
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert detected data-quality issues into a deterministic
    cleaning plan.

    No data is modified here.
    """

    plan: list[dict[str, Any]] = []

    for issue in issues:
        issue_type = issue["type"]
        field = issue.get("field")

        if issue_type == "duplicate_rows":
            plan.append(
                {
                    "step_id": "CLEAN-001",
                    "issue_id": issue["issue_id"],
                    "reason": "Remove exact duplicate records.",
                    "fields": ["order_id"],
                    "risk": "medium",
                    "source": "code",
                    "status": "planned",
                }
            )

        elif issue_type == "numeric_formatting":
            plan.append(
                {
                    "step_id": "CLEAN-002",
                    "issue_id": issue["issue_id"],
                    "reason": (
                        "Normalize numeric values by removing currency "
                        "symbols and thousands separators."
                    ),
                    "fields": [field],
                    "risk": "medium",
                    "source": "code",
                    "status": "planned",
                }
            )

        elif issue_type == "date_formatting":
            plan.append(
                {
                    "step_id": "CLEAN-003",
                    "issue_id": issue["issue_id"],
                    "reason": "Normalize date representations to one format.",
                    "fields": [field],
                    "risk": "medium",
                    "source": "code",
                    "status": "planned",
                }
            )

        elif issue_type == "whitespace":
            plan.append(
                {
                    "step_id": f"CLEAN-WS-{field}",
                    "issue_id": issue["issue_id"],
                    "reason": "Remove leading and trailing whitespace.",
                    "fields": [field],
                    "risk": "low",
                    "source": "code",
                    "status": "planned",
                }
            )

        elif issue_type == "case_variants":
            plan.append(
                {
                    "step_id": f"CLEAN-CASE-{field}",
                    "issue_id": issue["issue_id"],
                    "reason": "Normalize inconsistent capitalization.",
                    "fields": [field],
                    "risk": "low",
                    "source": "code",
                    "status": "planned",
                }
            )

    return plan