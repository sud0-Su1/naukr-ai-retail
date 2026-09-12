from pathlib import Path

import pandas as pd

from app.chat.evidence import build_evidence
from app.chat.executor import execute_query
from app.chat.models import QueryPlan


BASE_DIR = Path(__file__).resolve().parents[1]


def load_canonical() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR
        / "data"
        / "cleaned"
        / "canonical_retail.csv"
    )


def test_evidence_contains_required_traceability():
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

    evidence = build_evidence(
        plan,
        result,
        trace,
    )

    assert evidence["dataset"] == (
        "canonical_retail"
    )

    assert evidence["dataset_version"] == (
        "canonical_retail_v1"
    )

    assert "store_region" in (
        evidence["columns_used"]
    )

    assert "category_normalized" in (
        evidence["columns_used"]
    )

    assert "revenue" in (
        evidence["columns_used"]
    )

    assert evidence["rows_considered"] == 5
    assert evidence["rows_after_filters"] == 2
    assert evidence["result_rows"] == 1

    assert evidence["result_preview"][0][
        "category_normalized"
    ] == "T-Shirt"


def test_evidence_preview_is_bounded():
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
        limit=100,
    )

    result, trace = execute_query(
        df,
        plan,
    )

    evidence = build_evidence(
        plan,
        result,
        trace,
        max_preview_rows=2,
    )

    assert len(
        evidence["result_preview"]
    ) <= 2