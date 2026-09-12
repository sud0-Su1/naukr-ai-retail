from pathlib import Path

import pandas as pd
import pytest

from app.chat.executor import execute_query
from app.chat.models import QueryPlan


BASE_DIR = Path(__file__).resolve().parents[1]


def load_canonical() -> pd.DataFrame:
    path = (
        BASE_DIR
        / "data"
        / "cleaned"
        / "canonical_retail.csv"
    )

    return pd.read_csv(path)


def test_total_revenue():
    df = load_canonical()

    plan = QueryPlan(
        intent="aggregate",
        dataset="canonical_retail",
        metrics=[
            {
                "agg": "sum",
                "field": "revenue",
                "as": "total_revenue",
            }
        ],
        limit=10,
    )

    result, trace = execute_query(
        df,
        plan,
    )

    assert result.loc[
        0, "total_revenue"
    ] == pytest.approx(15396.70)

    assert trace["rows_considered"] == 5
    assert trace["result_rows"] == 1


def test_revenue_by_category():
    df = load_canonical()

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

    result, trace = execute_query(
        df,
        plan,
    )

    assert result.iloc[0]["category_normalized"] == (
        "Footwear"
    )

    assert result.iloc[0]["sales"] == pytest.approx(
        7122.15
    )

    assert trace["group_by"] == [
        "category_normalized"
    ]


def test_west_region_category_sales():
    df = load_canonical()

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

    result, trace = execute_query(
        df,
        plan,
    )

    assert len(result) == 1

    assert result.iloc[0]["category_normalized"] == (
        "T-Shirt"
    )

    assert result.iloc[0]["sales"] == pytest.approx(
        3337.20
    )

    assert trace["rows_after_filters"] == 2


def test_limit_is_respected():
    df = load_canonical()

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
        limit=2,
    )

    result, _ = execute_query(
        df,
        plan,
    )

    assert len(result) == 2


def test_unknown_runtime_field_is_rejected():
    df = load_canonical()

    plan = QueryPlan(
        intent="aggregate",
        dataset="canonical_retail",
        group_by=["does_not_exist"],
    )

    with pytest.raises(ValueError):
        execute_query(
            df,
            plan,
        )