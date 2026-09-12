from __future__ import annotations

from typing import Any

import pandas as pd

from app.chat.models import QueryPlan


DATASET_VERSIONS = {
    "canonical_retail": "canonical_retail_v1",
    "orders_clean": "orders_clean_v1",
    "products": "products_v1",
    "customers": "customers_v1",
    "stores": "stores_v1",
    "inventory": "inventory_v1",
}


def build_evidence(
    plan: QueryPlan,
    result: pd.DataFrame,
    execution_trace: dict[str, Any],
    *,
    max_preview_rows: int = 10,
) -> dict[str, Any]:
    """
    Build bounded, machine-readable evidence for a computed answer.

    Source columns are kept separate from generated metric aliases.
    """

    version = DATASET_VERSIONS.get(
        plan.dataset,
        "unknown",
    )

    # Only real source fields belong here.
    columns_used: set[str] = set()

    for condition in plan.filters:
        columns_used.add(condition.field)

    columns_used.update(plan.group_by)

    for metric in plan.metrics:
        columns_used.add(metric.field)

    # Do NOT add sort fields automatically because a sort field
    # may be a generated metric alias such as "sales".

    preview = result.head(
        max_preview_rows
    ).to_dict(orient="records")

    operations: list[str] = []

    if plan.filters:
        operations.append("filter")

    if plan.group_by:
        operations.append(
            f"group_by({', '.join(plan.group_by)})"
        )

    for metric in plan.metrics:
        operations.append(
            f"{metric.agg}({metric.field})"
        )

    if plan.sort:
        operations.append("sort")

    operations.append(
        f"limit({plan.limit})"
    )

    return {
        "dataset": plan.dataset,
        "dataset_version": version,
        "columns_used": sorted(columns_used),
        "filters": [
            {
                "field": condition.field,
                "operator": condition.op,
                "value": condition.value,
            }
            for condition in plan.filters
        ],
        "operations": operations,
        "group_by": plan.group_by,
        "metrics": [
            {
                "aggregation": metric.agg,
                "field": metric.field,
                "alias": metric.as_name,
            }
            for metric in plan.metrics
        ],
        "sort": [
            {
                "field": sort_spec.field,
                "direction": sort_spec.dir,
                "is_metric_alias": (
                    sort_spec.field
                    in {
                        metric.as_name
                        for metric in plan.metrics
                    }
                ),
            }
            for sort_spec in plan.sort
        ],
        "rows_considered": execution_trace[
            "rows_considered"
        ],
        "rows_after_filters": execution_trace[
            "rows_after_filters"
        ],
        "result_rows": len(result),
        "result_preview": preview,
        "execution_trace": execution_trace,
    }
