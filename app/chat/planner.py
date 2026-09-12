from __future__ import annotations

import json

from app.chat.models import QueryPlan


class PlannerError(Exception):
    """Raised when the planner cannot produce a supported query plan."""


class MockPlanner:
    """
    Deterministic planner used for tests/evaluation.

    This deliberately supports a small set of known retail questions so
    evaluation does not depend on an external LLM.
    """

    def plan(self, question: str) -> QueryPlan:
        q = question.strip().lower()

        # ---------------------------------------------------------
        # 1. Highest revenue category in the West
        # ---------------------------------------------------------
        if (
            "highest revenue" in q
            and "west" in q
            and "category" in q
        ):
            return QueryPlan(
                intent="top_n",
                dataset="canonical_retail",
                filters=[
                    {
                        "field": "store_region",
                        "op": "eq",
                        "value": "West",
                    }
                ],
                group_by=["category_normalized"],
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

        # ---------------------------------------------------------
        # 2. Revenue by category in the West
        # ---------------------------------------------------------
        if (
            "revenue by category" in q
            and "west" in q
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
                group_by=["category_normalized"],
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
            f"Mock planner does not support this question: {question}"
        )


def plan_to_json(plan: QueryPlan) -> str:
    """Serialize a query plan using its public JSON field aliases."""
    return json.dumps(
        plan.model_dump(by_alias=True),
        indent=2,
    )
