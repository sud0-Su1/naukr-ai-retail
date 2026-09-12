from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.chat.evidence import build_evidence
from app.chat.executor import execute_query
from app.chat.llm_planner import OllamaPlanner
from app.chat.session import ChatSession
from app.chat.validator import validate_query_plan


class ChatService:
    """
    Common production chat pipeline.

    The planner receives bounded context from the previous turn so that
    follow-up questions can be interpreted as part of the conversation.

    The resulting plan is still validated before execution.
    """

    def __init__(
        self,
        planner=None,
        dataset_path: str | Path = "data/cleaned/canonical_retail.csv",
        session: ChatSession | None = None,
    ):
        self.planner = planner or OllamaPlanner()
        self.dataset_path = Path(dataset_path)
        self.session = session or ChatSession("default-session")

    def _plan(self, question: str):
        """
        Ask the planner to interpret the question using bounded session
        context when the planner supports it.

        Backward compatibility is retained for existing planners whose
        plan() method accepts only the question.
        """
        context = self.session.context()

        try:
            return self.planner.plan(
                question,
                context=context,
            )
        except TypeError:
            # Existing/mock planners may only accept plan(question).
            return self.planner.plan(question)

    def answer(self, question: str) -> dict:
        question = question.strip()

        if not question:
            response = {
                "status": "error",
                "question": question,
                "error_type": "empty_question",
                "message": "Question cannot be empty.",
            }
            self.session.add_turn(question, response)
            return response

        try:
            plan = self._plan(question)
        except Exception as exc:
            response = {
                "status": "error",
                "question": question,
                "error_type": "planner_failure",
                "message": str(exc),
            }
            self.session.add_turn(question, response)
            return response

        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as exc:
            response = {
                "status": "error",
                "question": question,
                "error_type": "dataset_failure",
                "message": str(exc),
            }
            self.session.add_turn(question, response)
            return response

        try:
            validation = validate_query_plan(plan)
        except ValueError as exc:
            response = {
                "status": "error",
                "question": question,
                "error_type": "invalid_plan",
                "message": str(exc),
                "plan": plan.model_dump(by_alias=True),
            }
            self.session.add_turn(question, response)
            return response

        try:
            result, trace = execute_query(df, plan)
        except Exception as exc:
            response = {
                "status": "error",
                "question": question,
                "error_type": "execution_failure",
                "message": str(exc),
                "plan": plan.model_dump(by_alias=True),
                "validation": validation,
            }
            self.session.add_turn(question, response)
            return response

        evidence = build_evidence(
            plan=plan,
            result=result,
            execution_trace=trace,
        )

        response = {
            "status": "ok",
            "question": question,
            "plan": plan.model_dump(by_alias=True),
            "validation": validation,
            "result": result.to_dict(orient="records"),
            "evidence": evidence,
        }

        self.session.add_turn(question, response)

        return response
