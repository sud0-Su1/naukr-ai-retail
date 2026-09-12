from __future__ import annotations

import json

from app.chat.models import (
    ClarificationRequest,
    QueryPlan,
    RefusalRequest,
)


class PlannerError(Exception):
    pass


class MockPlanner:
    """
    Deterministic planner used by tests/evaluation.

    Supports a small set of known questions and a contextual East follow-up
    so multi-turn behavior can be tested without requiring an LLM.
    """

    def plan(
        self,
        question: str,
        context: dict | None = None,
    ) -> QueryPlan | ClarificationRequest | RefusalRequest:
        normalized = question.strip().lower()

        if "customer lifetime value" in normalized:
            return RefusalRequest(
                message=(
                    "I can't answer that from the available retail data."
                ),
                reason=(
                    "Customer lifetime value is not a supported metric "
                    "in the available dataset."
                ),
            )

        if normalized == "which region generated the highest revenue?":
            return ClarificationRequest(
                question=(
                    "Which region do you mean?"
                ),
                options=[
                    "Store region",
                    "Customer region",
                ],
            )

        if normalized in {
            "what is the top category by revenue in the west?",
            "which category generated the highest revenue in the west?",
        }:
            return self._category_revenue_plan("desc")

        if normalized == "what category has the lowest revenue in the west?":
            return self._category_revenue_plan("asc")

        if normalized == "show revenue by category in the west.":
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

        if normalized == "what about the east?":
            if not context or not context.get("has_previous_turn"):
                raise PlannerError(
                    "The question refers to a previous turn, but no previous "
                    "conversation context is available."
                )

            previous_plan = context.get("previous_plan", {})

            if previous_plan.get("dataset") != "canonical_retail":
                raise PlannerError(
                    "Previous conversation context is not supported."
                )

            return QueryPlan(
                intent="aggregate",
                dataset="canonical_retail",
                filters=[
                    {
                        "field": "store_region",
                        "op": "eq",
                        "value": "East",
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

        raise PlannerError(f"Unsupported question: {question}")

    @staticmethod
    def _category_revenue_plan(
        direction: str,
    ) -> QueryPlan:
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
                    "dir": direction,
                }
            ],
            limit=1,
        )


def plan_to_json(plan: QueryPlan) -> str:
    return json.dumps(
        plan.model_dump(by_alias=True),
        indent=2,
    )
