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
