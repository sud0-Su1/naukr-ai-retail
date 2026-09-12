import json

import pandas as pd

from app.chat.evidence import build_evidence
from app.chat.executor import execute_query
from app.chat.planner import MockPlanner
from app.chat.validator import validate_query_plan


DATASET_PATH = "data/cleaned/canonical_retail.csv"


def answer_question(question: str) -> dict:
    df = pd.read_csv(DATASET_PATH)

    planner = MockPlanner()

    # 1. Plan
    plan = planner.plan(question)

    # 2. Validate
    validation = validate_query_plan(plan)

    # 3. Execute
    result, trace = execute_query(
        df,
        plan,
    )

    # 4. Build evidence
    evidence = build_evidence(
        plan,
        result,
        trace,
    )

    return {
        "question": question,
        "plan": plan.model_dump(
            by_alias=True
        ),
        "validation": validation,
        "result": result.to_dict(
            orient="records"
        ),
        "evidence": evidence,
    }


if __name__ == "__main__":
    question = (
        "Which category generated the highest "
        "revenue in the West?"
    )

    response = answer_question(question)

    print(
        json.dumps(
            response,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
