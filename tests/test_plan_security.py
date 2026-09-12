import pytest

from app.chat.models import QueryPlan
from app.chat.validator import validate_query_plan


def make_plan(**overrides):
    data = {
        "intent": "aggregate",
        "dataset": "canonical_retail",
        "filters": [],
        "group_by": ["category_normalized"],
        "metrics": [
            {
                "agg": "sum",
                "field": "revenue",
                "as": "sales",
            }
        ],
        "sort": [],
        "limit": 10,
    }

    data.update(overrides)
    return QueryPlan(**data)


def test_rejects_too_many_filters():
    filters = [
        {
            "field": "store_region",
            "op": "eq",
            "value": "West",
        }
        for _ in range(11)
    ]

    plan = make_plan(filters=filters)

    with pytest.raises(ValueError, match="Too many filters"):
        validate_query_plan(plan)


def test_rejects_too_many_group_by_fields():
    plan = make_plan(
        group_by=[
            "category_normalized",
            "brand",
            "store_region",
            "city",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Too many group-by fields",
    ):
        validate_query_plan(plan)


def test_rejects_too_many_metrics():
    metrics = [
        {
            "agg": "sum",
            "field": "revenue",
            "as": f"metric_{i}",
        }
        for i in range(6)
    ]

    plan = make_plan(metrics=metrics)

    with pytest.raises(ValueError, match="Too many metrics"):
        validate_query_plan(plan)


def test_rejects_too_many_sort_fields():
    plan = make_plan(
        sort=[
            {"field": "category_normalized", "dir": "asc"},
            {"field": "brand", "dir": "asc"},
            {"field": "store_region", "dir": "asc"},
            {"field": "city", "dir": "asc"},
        ]
    )

    with pytest.raises(ValueError, match="Too many sort fields"):
        validate_query_plan(plan)


def test_rejects_duplicate_group_by_fields():
    plan = make_plan(
        group_by=[
            "category_normalized",
            "category_normalized",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate group-by fields",
    ):
        validate_query_plan(plan)


def test_rejects_duplicate_metric_aliases():
    plan = make_plan(
        metrics=[
            {
                "agg": "sum",
                "field": "revenue",
                "as": "sales",
            },
            {
                "agg": "avg",
                "field": "revenue",
                "as": "sales",
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate metric aliases",
    ):
        validate_query_plan(plan)


def test_rejects_duplicate_sort_fields():
    plan = make_plan(
        sort=[
            {"field": "category_normalized", "dir": "asc"},
            {"field": "category_normalized", "dir": "desc"},
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate sort fields",
    ):
        validate_query_plan(plan)


def test_allows_reasonable_plan():
    plan = make_plan(
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

    result = validate_query_plan(plan)

    assert result["valid"] is True
    assert result["limit"] == 10
