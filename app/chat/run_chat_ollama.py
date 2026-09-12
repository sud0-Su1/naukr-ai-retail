from __future__ import annotations

import json

import pandas as pd

from app.chat.evidence import build_evidence
from app.chat.executor import execute_query
from app.chat.llm_planner import OllamaPlanner
from app.chat.validator import validate_query_plan


DATASET_PATH = (
    "data/cleaned/canonical_retail.csv"
)


def answer_question(question: str) -> dict:
    """
    Full grounded AI flow:

    1. LLM creates structured plan.
    2. Application validates plan.
    3. Deterministic executor computes result.
    4. Evidence is generated.
    """

    df = pd.read_csv(
        DATASET_PATH
    )

    planner = OllamaPlanner()

    # -----------------------------
    # 1. AI planning
    # -----------------------------
    plan = planner.plan(question)

    # -----------------------------
    # 2. Application validation
    # -----------------------------
    validation = validate_query_plan(
        plan
    )

    # -----------------------------
    # 3. Deterministic execution
    # -----------------------------
    result, trace = execute_query(
        df,
        plan,
    )

    # -----------------------------
    # 4. Evidence
    # -----------------------------
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
        "Which category generated "
        "the highest revenue in the West?"
    )

    response = answer_question(
        question
    )

    print(
        json.dumps(
            response,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )