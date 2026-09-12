from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatSession:
    session_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)

    def add_turn(
        self,
        question: str,
        response: dict[str, Any],
    ) -> None:
        self.turns.append(
            {
                "question": question,
                "response": response,
            }
        )

    @property
    def last_response(self) -> dict[str, Any] | None:
        if not self.turns:
            return None
        return self.turns[-1]["response"]

    @property
    def last_plan(self) -> dict[str, Any] | None:
        response = self.last_response
        if not response:
            return None
        return response.get("plan")

    def context(self) -> dict[str, Any]:
        """
        Return bounded context from the previous turn.

        Only structured information needed for follow-up interpretation is
        exposed. Full previous results are not copied into the planner context.
        """
        plan = self.last_plan

        if not plan:
            return {
                "has_previous_turn": False,
            }

        return {
            "has_previous_turn": True,
            "previous_question": self.turns[-1]["question"],
            "previous_plan": plan,
        }
