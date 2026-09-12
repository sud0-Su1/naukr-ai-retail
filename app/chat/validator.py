from __future__ import annotations

from typing import Any

from app.chat.models import QueryPlan


ALLOWED_DATASETS = {
    "canonical_retail",
    "orders_clean",
    "products",
    "customers",
    "stores",
    "inventory",
}


ALLOWED_FIELDS = {
    "canonical_retail": {
        "order_id",
        "customer_id",
        "product_id",
        "store_id",
        "order_date",
        "quantity",
        "unit_price",
        "discount",
        "returned",
        "sku",
        "product_name",
        "category_normalized",
        "brand",
        "price",
        "cost",
        "name",
        "customer_region",
        "signup_date",
        "store_name",
        "store_region",
        "city",
        "revenue",
    },
    "orders_clean": {
        "order_id",
        "customer_id",
        "product_id",
        "store_id",
        "order_date",
        "quantity",
        "unit_price",
        "discount",
        "returned",
    },
    "products": {
        "product_id",
        "sku",
        "product_name",
        "category",
        "category_normalized",
        "brand",
        "price",
        "cost",
    },
    "customers": {
        "customer_id",
        "name",
        "customer_region",
        "signup_date",
    },
    "stores": {
        "store_id",
        "store_name",
        "store_region",
        "city",
    },
    "inventory": {
        "inventory_id",
        "product_id",
        "store_id",
        "inventory_date",
        "stock_qty",
    },
}


# Resource limits protect the executor from unnecessarily expensive plans.
MAX_FILTERS = 10
MAX_GROUP_BY = 3
MAX_METRICS = 5
MAX_SORT = 3


def validate_query_plan(
    plan: QueryPlan,
) -> dict[str, Any]:
    """
    Validate a structured query plan against application allow-lists
    and resource/complexity limits.
    """

    if plan.dataset not in ALLOWED_DATASETS:
        raise ValueError(
            f"Dataset '{plan.dataset}' is not allowed."
        )

    # Resource limits.
    if len(plan.filters) > MAX_FILTERS:
        raise ValueError(
            f"Too many filters. Maximum allowed is {MAX_FILTERS}."
        )

    if len(plan.group_by) > MAX_GROUP_BY:
        raise ValueError(
            f"Too many group-by fields. Maximum allowed is {MAX_GROUP_BY}."
        )

    if len(plan.metrics) > MAX_METRICS:
        raise ValueError(
            f"Too many metrics. Maximum allowed is {MAX_METRICS}."
        )

    if len(plan.sort) > MAX_SORT:
        raise ValueError(
            f"Too many sort fields. Maximum allowed is {MAX_SORT}."
        )

    # Duplicate fields can create unnecessary or confusing work.
    if len(plan.group_by) != len(set(plan.group_by)):
        raise ValueError(
            "Duplicate group-by fields are not allowed."
        )

    metric_aliases = [
        metric.as_name
        for metric in plan.metrics
    ]

    if len(metric_aliases) != len(set(metric_aliases)):
        raise ValueError(
            "Duplicate metric aliases are not allowed."
        )

    sort_fields_list = [
        sort_item.field
        for sort_item in plan.sort
    ]

    if len(sort_fields_list) != len(set(sort_fields_list)):
        raise ValueError(
            "Duplicate sort fields are not allowed."
        )

    allowed_fields = ALLOWED_FIELDS[plan.dataset]

    referenced_fields: set[str] = set()

    # Filters must reference real dataset columns.
    for condition in plan.filters:
        referenced_fields.add(condition.field)

    # Group-by fields must reference real dataset columns.
    referenced_fields.update(plan.group_by)

    # Metric input fields must reference real dataset columns.
    metric_aliases_set: set[str] = set()

    for metric in plan.metrics:
        referenced_fields.add(metric.field)
        metric_aliases_set.add(metric.as_name)

    # Sort fields can reference either:
    # 1. a real dataset/result field, or
    # 2. a metric alias such as "sales".
    sort_fields = {
        sort_item.field
        for sort_item in plan.sort
    }

    invalid_dataset_fields = (
        referenced_fields - allowed_fields
    )

    if invalid_dataset_fields:
        raise ValueError(
            "Unknown or disallowed fields: "
            f"{sorted(invalid_dataset_fields)}"
        )

    invalid_sort_fields = (
        sort_fields - allowed_fields - metric_aliases_set
    )

    if invalid_sort_fields:
        raise ValueError(
            "Unknown or disallowed sort fields: "
            f"{sorted(invalid_sort_fields)}"
        )

    if plan.limit < 1 or plan.limit > 100:
        raise ValueError(
            "Query limit must be between 1 and 100."
        )

    return {
        "valid": True,
        "dataset": plan.dataset,
        "validated_fields": sorted(
            referenced_fields
        ),
        "metric_aliases": sorted(
            metric_aliases_set
        ),
        "validated_sort_fields": sorted(
            sort_fields
        ),
        "limit": plan.limit,
        "resource_limits": {
            "max_filters": MAX_FILTERS,
            "max_group_by": MAX_GROUP_BY,
            "max_metrics": MAX_METRICS,
            "max_sort": MAX_SORT,
        },
    }
