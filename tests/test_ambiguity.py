from app.chat.planner import MockPlanner
from app.chat.service import ChatService
from app.chat.session import ChatSession


def test_ambiguous_region_question_requests_clarification():
    session = ChatSession("ambiguity-test")

    service = ChatService(
        planner=MockPlanner(),
        session=session,
    )

    response = service.answer(
        "Which region generated the highest revenue?"
    )

    assert response["status"] == "clarification"
    assert response["question"] == (
        "Which region generated the highest revenue?"
    )
    assert response["message"] == "Which region do you mean?"
    assert response["options"] == [
        "Store region",
        "Customer region",
    ]

    assert "result" not in response
    assert "evidence" not in response

    assert len(session.turns) == 1
