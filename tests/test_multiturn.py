from app.chat.planner import MockPlanner
from app.chat.service import ChatService
from app.chat.session import ChatSession


def test_follow_up_question_uses_previous_context():
    session = ChatSession("multi-turn-test")

    service = ChatService(
        planner=MockPlanner(),
        session=session,
    )

    first = service.answer(
        "Show revenue by category in the West."
    )

    assert first["status"] == "ok"
    assert first["plan"]["filters"][0]["value"] == "West"

    second = service.answer("What about the East?")

    assert second["status"] == "ok"
    assert second["plan"]["dataset"] == "canonical_retail"
    assert second["plan"]["filters"][0]["field"] == "store_region"
    assert second["plan"]["filters"][0]["value"] == "East"

    assert len(session.turns) == 2


def test_follow_up_without_context_is_rejected():
    session = ChatSession("empty-context-test")

    service = ChatService(
        planner=MockPlanner(),
        session=session,
    )

    response = service.answer("What about the East?")

    assert response["status"] == "error"
    assert response["error_type"] == "planner_failure"
