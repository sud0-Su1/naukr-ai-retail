from __future__ import annotations

from typing import Any

import pandas as pd

from app.chat.evidence import build_evidence
from app.chat.executor import execute_query
from app.chat.llm_planner import OllamaPlanner, OllamaPlannerError
from app.chat.validator import validate_query_plan


class ChatService:
    def __init__(
        self,
        dataset_path: str = "data/cleaned/canonical_retail.csv",
    ):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.planner = OllamaPlanner()

    def answer(self, question: str) -> dict[str, Any]:
        question = question.strip()

        if not question:
            return {
                "status": "error",
                "error_type": "empty_question",
                "message": "Please provide a question.",
            }

        # -------------------------
        # 1. Planner
        # -------------------------
        try:
            plan = self.planner.plan(question)
        except OllamaPlannerError as exc:
            return {
                "status": "error",
                "error_type": "planner_failure",
                "message": (
                    "I couldn't safely translate that question "
                    "into a supported data query."
                ),
                "details": str(exc),
            }

        # -------------------------
        # 2. Plan validation
        # -------------------------
        try:
            validation = validate_query_plan(plan)
        except ValueError as exc:
            return {
                "status": "error",
                "error_type": "invalid_plan",
                "message": (
                    "The generated query was rejected by "
                    "the safety validator."
                ),
                "details": str(exc),
            }

        # -------------------------
        # 3. Execution
        # -------------------------
        try:
            result, trace = execute_query(
                self.df,
                plan,
            )
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "status": "error",
                "error_type": "execution_failure",
                "message": (
                    "I couldn't execute a safe computation "
                    "for that question."
                ),
                "details": str(exc),
            }

        # -------------------------
        # 4. Evidence
        # -------------------------
        evidence = build_evidence(
            plan,
            result,
            trace,
        )

        return {
            "status": "ok",
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