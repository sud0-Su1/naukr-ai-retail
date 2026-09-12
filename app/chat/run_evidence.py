import json

import pandas as pd

from app.chat.evidence import build_evidence
from app.chat.executor import execute_query
from app.chat.models import QueryPlan
from app.chat.validator import validate_query_plan


DATASET_PATH = (
    "data/cleaned/canonical_retail.csv"
)


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)

    # Example question:
    #
    # "Which category generated the highest
    # revenue in the West?"

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

    # 1. Validate plan.
    validation = validate_query_plan(
        plan
    )

    # 2. Execute deterministic computation.
    result, trace = execute_query(
        df,
        plan,
    )

    # 3. Build evidence.
    evidence = build_evidence(
        plan,
        result,
        trace,
    )

    output = {
        "validation": validation,
        "result": result.to_dict(
            orient="records"
        ),
        "evidence": evidence,
    }

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )