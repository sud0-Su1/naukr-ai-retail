from app.chat.llm_planner import OllamaPlanner
from app.chat.models import ClarificationRequest
from app.chat.service import ChatService
from app.chat.session import ChatSession


def test_ollama_planner_returns_clarification_for_ambiguous_region():
    planner = OllamaPlanner()

    response = planner.plan(
        "Which region generated the highest revenue?"
    )

    assert isinstance(response, ClarificationRequest)
    assert response.options == [
        "Store region",
        "Customer region",
    ]


def test_service_stops_before_execution_for_ollama_clarification():
    session = ChatSession("ollama-ambiguity-test")

    service = ChatService(
        planner=OllamaPlanner(),
        session=session,
    )

    response = service.answer(
        "Which region generated the highest revenue?"
    )

    assert response["status"] == "clarification"
    assert "result" not in response
    assert "evidence" not in response
    assert len(session.turns) == 1
