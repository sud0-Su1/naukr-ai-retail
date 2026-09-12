from __future__ import annotations

import json


from app.chat.models import QueryPlan


class PlannerError(Exception):
    """Raised when a planner cannot produce a valid plan."""


class MockPlanner:
    """
    Deterministic planner for local development and evaluation.

    This keeps the application usable without an external LLM.
    """

    def plan(self, question: str) -> QueryPlan:
        normalized = question.strip().lower()

        if (
            "highest revenue" in normalized
            and "west" in normalized
            and "category" in normalized
        ):
            return QueryPlan(
                intent="aggregate",
                dataset="canonical_retail",
                filters=[
                    {
                        "field": "store_region",
                        "op": "eq",
                        "value": "West",
                    }
                ],
                group_by=[
                    "category_normalized"
                ],
                metrics=[
                    {
                        "agg": "sum",
                        "field": "revenue",
                        "as": "sales",
                    }
                ],
                sort=[
                    {
                        "field": "sales",
                        "dir": "desc",
                    }
                ],
                limit=10,
            )

        raise PlannerError(
            "Mock planner does not support this question."
        )


def plan_to_json(plan: QueryPlan) -> str:
    """
    Serialize a validated QueryPlan for logging/debugging.
    """
    return json.dumps(
        plan.model_dump(
            by_alias=True
        ),
        indent=2,
        ensure_ascii=False,
    )
