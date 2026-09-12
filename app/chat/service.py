from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.chat.evidence import build_evidence
from app.chat.executor import execute_query
from app.chat.llm_planner import OllamaPlanner
from app.chat.planner import PlannerError
from app.chat.session import ChatSession
from app.chat.session_store import ChatSessionStore
from app.chat.validator import validate_query_plan


class ChatService:
    def __init__(
        self,
        planner: Any | None = None,
        dataset_path: str | Path = "data/cleaned/canonical_retail.csv",
        session: ChatSession | None = None,
        session_store: ChatSessionStore | None = None,
        session_id: str = "default-session",
    ):
        self.planner = planner or OllamaPlanner()
        self.dataset_path = Path(dataset_path)

        self.session_store = session_store or ChatSessionStore()

        if session is not None:
            self.session = session
        elif self.session_store.exists(session_id):
            self.session = self.session_store.load(session_id)
        else:
            self.session = ChatSession(session_id)

    def _save_session(self) -> None:
        self.session_store.save(self.session)

    def _plan(self, question: str):
        context = self.session.context()

        try:
            return self.planner.plan(question, context=context)
        except TypeError:
            # Backward compatibility for simple/mock planners
            # that only accept the question argument.
            return self.planner.plan(question)

    def answer(self, question: str) -> dict[str, Any]:
        question = question.strip()

        if not question:
            response = {
                "status": "error",
                "question": question,
                "error_type": "empty_question",
                "message": "Question cannot be empty.",
            }
            self.session.add_turn(question, response)
            self._save_session()
            return response

        try:
            planned = self._plan(question)
        except Exception as exc:
            response = {
                "status": "error",
                "question": question,
                "error_type": "planner_failure",
                "message": str(exc),
            }
            self.session.add_turn(question, response)
            self._save_session()
            return response

        # Clarification is part of the conversation state.
        if hasattr(planned, "type") and planned.type == "clarification":
            response = {
                "status": "clarification",
                "question": question,
                "message": planned.question,
                "options": planned.options,
            }
            self.session.add_turn(question, response)
            self._save_session()
            return response

        # Unsupported requests are explicitly refused.
        if hasattr(planned, "type") and planned.type == "refusal":
            response = {
                "status": "refusal",
                "question": question,
                "message": planned.message,
                "reason": planned.reason,
            }
            self.session.add_turn(question, response)
            self._save_session()
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
            self._save_session()
            return response

        try:
            validation = validate_query_plan(planned)
        except Exception as exc:
            response = {
                "status": "error",
                "question": question,
                "error_type": "invalid_plan",
                "message": str(exc),
            }
            self.session.add_turn(question, response)
            self._save_session()
            return response

        try:
            result, execution_trace = execute_query(
                df,
                planned,
            )
        except Exception as exc:
            response = {
                "status": "error",
                "question": question,
                "error_type": "execution_failure",
                "message": str(exc),
            }
            self.session.add_turn(question, response)
            self._save_session()
            return response

        evidence = build_evidence(
            plan=planned,
            result=result,
            execution_trace=execution_trace,
        )

        response = {
            "status": "ok",
            "question": question,
            "plan": planned.model_dump(by_alias=True),
            "validation": validation,
            "result": result.to_dict(orient="records"),
            "evidence": evidence,
        }

        self.session.add_turn(question, response)
        self._save_session()

        return response
