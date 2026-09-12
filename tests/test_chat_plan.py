import pytest

from app.chat.models import QueryPlan
from app.chat.validator import validate_query_plan


def test_valid_plan():
    plan = QueryPlan(
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

    result = validate_query_plan(plan)

    assert result["valid"] is True


def test_invalid_dataset():
    plan = QueryPlan(
        intent="aggregate",
        dataset="secret_database",
    )

    with pytest.raises(ValueError):
        validate_query_plan(plan)


def test_invalid_field():
    plan = QueryPlan(
        intent="aggregate",
        dataset="canonical_retail",
        group_by=[
            "password"
        ],
    )

    with pytest.raises(ValueError):
        validate_query_plan(plan)


def test_excessive_limit():
    with pytest.raises(Exception):
        QueryPlan(
            intent="aggregate",
            dataset="canonical_retail",
            limit=1000000,
        )


def test_metric_alias_can_be_used_for_sort():
    plan = QueryPlan(
        intent="aggregate",
        dataset="canonical_retail",
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

    result = validate_query_plan(plan)

    assert result["valid"] is True
    assert "sales" in result["metric_aliases"]