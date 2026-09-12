from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.chat.evidence import build_evidence
from app.chat.executor import execute_query
from app.chat.llm_planner import OllamaPlanner
from app.chat.validator import validate_query_plan


class ChatService:
    """
    Common production chat pipeline.

    The planner can be injected so tests/evaluation can use a deterministic
    planner while still exercising the same validation, execution, and
    evidence pipeline as the application.
    """

    def __init__(
        self,
        planner=None,
        dataset_path: str | Path = "data/cleaned/canonical_retail.csv",
    ):
        self.planner = planner or OllamaPlanner()
        self.dataset_path = Path(dataset_path)

    def answer(self, question: str) -> dict:
        question = question.strip()

        if not question:
            return {
                "status": "error",
                "error_type": "empty_question",
                "message": "Question cannot be empty.",
            }

        # ---------------------------------------------------------
        # 1. Planning
        # ---------------------------------------------------------
        try:
            plan = self.planner.plan(question)
        except Exception as exc:
            return {
                "status": "error",
                "error_type": "planner_failure",
                "message": str(exc),
            }

        # ---------------------------------------------------------
        # 2. Load dataset
        # ---------------------------------------------------------
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as exc:
            return {
                "status": "error",
                "error_type": "dataset_failure",
                "message": str(exc),
            }

        # ---------------------------------------------------------
        # 3. Validate plan
        # ---------------------------------------------------------
        try:
            validation = validate_query_plan(plan)
        except ValueError as exc:
            return {
                "status": "error",
                "error_type": "invalid_plan",
                "message": str(exc),
                "plan": plan.model_dump(),
            }

        # ---------------------------------------------------------
        # 4. Execute validated plan
        # ---------------------------------------------------------
        try:
            result, trace = execute_query(df, plan)
        except Exception as exc:
            return {
                "status": "error",
                "error_type": "execution_failure",
                "message": str(exc),
                "plan": plan.model_dump(),
                "validation": validation,
            }

        # ---------------------------------------------------------
        # 5. Build evidence
        # ---------------------------------------------------------
        evidence = build_evidence(
            plan=plan,
            result=result,
            execution_trace=trace,
        )

        # ---------------------------------------------------------
        # 6. Return grounded result
        # ---------------------------------------------------------
        return {
            "status": "ok",
            "question": question,
            "plan": plan.model_dump(),
            "validation": validation,
            "result": result.to_dict(orient="records"),
            "evidence": evidence,
        }
