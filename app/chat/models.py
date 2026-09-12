from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AllowedOperator = Literal[
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "contains",
]

AllowedAggregation = Literal[
    "sum",
    "avg",
    "min",
    "max",
    "count",
    "nunique",
]

AllowedSortDirection = Literal[
    "asc",
    "desc",
]


class FilterCondition(BaseModel):
    field: str
    op: AllowedOperator
    value: str | int | float | bool | list[str]


class Metric(BaseModel):
    agg: AllowedAggregation
    field: str
    as_name: str = Field(alias="as")


class SortSpec(BaseModel):
    field: str
    dir: AllowedSortDirection


class QueryPlan(BaseModel):
    intent: Literal[
        "filter",
        "aggregate",
        "describe",
        "compare",
        "top_n",
    ]

    dataset: str

    filters: list[FilterCondition] = Field(
        default_factory=list
    )

    group_by: list[str] = Field(
        default_factory=list
    )

    metrics: list[Metric] = Field(
        default_factory=list
    )

    sort: list[SortSpec] = Field(
        default_factory=list
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )
