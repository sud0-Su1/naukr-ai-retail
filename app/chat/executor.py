from __future__ import annotations

from typing import Any

import pandas as pd

from app.chat.models import QueryPlan


ALLOWED_OPERATORS = {
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "contains",
}

ALLOWED_AGGREGATIONS = {
    "sum",
    "avg",
    "min",
    "max",
    "count",
    "nunique",
}


def apply_filter(
    df: pd.DataFrame,
    field: str,
    operator: str,
    value: Any,
) -> pd.Series:
    """Return a boolean mask for one validated filter."""

    series = df[field]

    if operator == "eq":
        return series == value

    if operator == "neq":
        return series != value

    if operator == "gt":
        return series > value

    if operator == "gte":
        return series >= value

    if operator == "lt":
        return series < value

    if operator == "lte":
        return series <= value

    if operator == "in":
        if not isinstance(value, list):
            raise ValueError(
                "'in' operator requires a list value."
            )
        return series.isin(value)

    if operator == "contains":
        return (
            series.astype(str)
            .str.contains(
                str(value),
                case=False,
                na=False,
                regex=False,
            )
        )

    raise ValueError(
        f"Unsupported operator: {operator}"
    )


def apply_filters(
    df: pd.DataFrame,
    filters: list[Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply all filters and return trace information."""

    result = df.copy()
    filter_trace: list[dict[str, Any]] = []

    for condition in filters:
        operator = condition.op

        if operator not in ALLOWED_OPERATORS:
            raise ValueError(
                f"Unsupported operator: {operator}"
            )

        before_count = len(result)

        mask = apply_filter(
            result,
            condition.field,
            operator,
            condition.value,
        )

        result = result.loc[mask].copy()

        filter_trace.append(
            {
                "field": condition.field,
                "operator": operator,
                "value": condition.value,
                "rows_before": before_count,
                "rows_after": len(result),
            }
        )

    return result, filter_trace


def calculate_metric(
    df: pd.DataFrame,
    aggregation: str,
    field: str,
) -> pd.Series:
    """Calculate one supported aggregation."""

    if aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    series = df[field]

    if aggregation == "sum":
        return series.sum()

    if aggregation == "avg":
        return series.mean()

    if aggregation == "min":
        return series.min()

    if aggregation == "max":
        return series.max()

    if aggregation == "count":
        return series.count()

    if aggregation == "nunique":
        return series.nunique()

    raise ValueError(
        f"Unsupported aggregation: {aggregation}"
    )


def execute_query(
    df: pd.DataFrame,
    plan: QueryPlan,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Execute a validated query plan against a canonical dataset.

    Returns:
        result dataframe
        execution trace/evidence
    """

    if plan.limit < 1 or plan.limit > 100:
        raise ValueError(
            "Query limit must be between 1 and 100."
        )

    missing_fields = set()

    for condition in plan.filters:
        if condition.field not in df.columns:
            missing_fields.add(condition.field)

    for field in plan.group_by:
        if field not in df.columns:
            missing_fields.add(field)

    for metric in plan.metrics:
        if metric.field not in df.columns:
            missing_fields.add(metric.field)

    if missing_fields:
        raise ValueError(
            f"Fields not present in dataset: "
            f"{sorted(missing_fields)}"
        )

    rows_before = len(df)

    # --------------------------------------------------
    # Filters
    # --------------------------------------------------

    filtered, filter_trace = apply_filters(
        df,
        plan.filters,
    )

    rows_after_filters = len(filtered)

    # --------------------------------------------------
    # Grouped aggregation
    # --------------------------------------------------

    if plan.group_by:
        if not plan.metrics:
            raise ValueError(
                "Grouped query requires at least one metric."
            )

        grouped = filtered.groupby(
            plan.group_by,
            dropna=False,
        )

        result = grouped.size().reset_index(
            name="row_count"
        )

        for metric in plan.metrics:
            metric_series = grouped[
                metric.field
            ]

            if metric.agg == "sum":
                values = metric_series.sum()
            elif metric.agg == "avg":
                values = metric_series.mean()
            elif metric.agg == "min":
                values = metric_series.min()
            elif metric.agg == "max":
                values = metric_series.max()
            elif metric.agg == "count":
                values = metric_series.count()
            elif metric.agg == "nunique":
                values = metric_series.nunique()
            else:
                raise ValueError(
                    f"Unsupported aggregation: {metric.agg}"
                )

            metric_frame = values.reset_index(
                name=metric.as_name
            )

            result = result.drop(
                columns=["row_count"]
            ).merge(
                metric_frame,
                on=plan.group_by,
                how="left",
            )

    # --------------------------------------------------
    # Ungrouped aggregation
    # --------------------------------------------------

    elif plan.metrics:
        result_data: dict[str, Any] = {}

        for metric in plan.metrics:
            value = calculate_metric(
                filtered,
                metric.agg,
                metric.field,
            )

            if hasattr(value, "item"):
                value = value.item()

            result_data[metric.as_name] = value

        result = pd.DataFrame([result_data])

    # --------------------------------------------------
    # Raw filtered rows
    # --------------------------------------------------

    else:
        result = filtered.copy()

    # --------------------------------------------------
    # Sorting
    # --------------------------------------------------

    for sort_spec in reversed(plan.sort):
        if sort_spec.field not in result.columns:
            raise ValueError(
                f"Cannot sort by '{sort_spec.field}' "
                "because it is not present in the result."
            )

        result = result.sort_values(
            sort_spec.field,
            ascending=sort_spec.dir == "asc",
        )

    # --------------------------------------------------
    # Limit
    # --------------------------------------------------

    result = result.head(plan.limit).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Evidence / execution trace
    # --------------------------------------------------

    trace = {
        "dataset": plan.dataset,
        "rows_considered": rows_before,
        "rows_after_filters": rows_after_filters,
        "filters": [
            {
                "field": condition.field,
                "operator": condition.op,
                "value": condition.value,
            }
            for condition in plan.filters
        ],
        "filter_trace": filter_trace,
        "group_by": plan.group_by,
        "metrics": [
            {
                "aggregation": metric.agg,
                "field": metric.field,
                "as": metric.as_name,
            }
            for metric in plan.metrics
        ],
        "sort": [
            {
                "field": sort_spec.field,
                "direction": sort_spec.dir,
            }
            for sort_spec in plan.sort
        ],
        "limit": plan.limit,
        "result_rows": len(result),
        "result_preview": result.to_dict(
            orient="records"
        ),
    }

    return result, trace