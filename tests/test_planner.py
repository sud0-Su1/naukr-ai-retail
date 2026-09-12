import pytest

from app.chat.planner import (
    MockPlanner,
    PlannerError,
    plan_to_json,
)


def test_mock_planner_creates_valid_plan():
    planner = MockPlanner()

    plan = planner.plan(
        "Which category generated the highest revenue in the West?"
    )

    assert plan.dataset == "canonical_retail"

    assert plan.group_by == [
        "category_normalized"
    ]

    assert plan.metrics[0].field == "revenue"
    assert plan.metrics[0].as_name == "sales"

    assert plan.filters[0].field == "store_region"
    assert plan.filters[0].value == "West"


@pytest.mark.parametrize(
    ("question", "direction"),
    [
        (
            "What is the top category by revenue in the West?",
            "desc",
        ),
        (
            "Which category generated the highest revenue in the West?",
            "desc",
        ),
        (
            "What category has the lowest revenue in the West?",
            "asc",
        ),
    ],
)
def test_mock_planner_maps_category_revenue_language(
    question,
    direction,
):
    plan = MockPlanner().plan(question)

    assert plan.intent == "top_n"
    assert plan.group_by == ["category_normalized"]
    assert plan.metrics[0].agg == "sum"
    assert plan.metrics[0].field == "revenue"
    assert plan.metrics[0].as_name == "sales"
    assert plan.filters[0].field == "store_region"
    assert plan.filters[0].value == "West"
    assert plan.sort[0].field == "sales"
    assert plan.sort[0].dir == direction
    assert plan.limit == 1


def test_mock_planner_rejects_unknown_question():
    planner = MockPlanner()

    with pytest.raises(PlannerError):
        planner.plan(
            "Tell me something about customers."
        )


def test_plan_to_json():
    planner = MockPlanner()

    plan = planner.plan(
        "Which category generated the highest revenue in the West?"
    )

    serialized = plan_to_json(plan)

    assert '"dataset": "canonical_retail"' in serialized
    assert '"field": "revenue"' in serialized
    assert '"as": "sales"' in serialized
