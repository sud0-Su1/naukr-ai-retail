from app.chat.planner import MockPlanner
from app.chat.service import ChatService
from app.chat.session import ChatSession


def test_unanswerable_question_returns_refusal():
    session = ChatSession("refusal-test")
    service = ChatService(
        planner=MockPlanner(),
        session=session,
    )

    response = service.answer(
        "What is our customer lifetime value?"
    )

    assert response["status"] == "refusal"
    assert "can't answer" in response["message"].lower()
    assert "customer lifetime value" in response["reason"].lower()
    assert "result" not in response
    assert "evidence" not in response
    assert len(session.turns) == 1


def test_refusal_does_not_execute_a_query():
    class ExplodingPlanner(MockPlanner):
        pass

    session = ChatSession("refusal-execution-test")
    service = ChatService(
        planner=ExplodingPlanner(),
        session=session,
    )

    response = service.answer(
        "What is our customer lifetime value?"
    )

    assert response["status"] == "refusal"
    assert "result" not in response
    assert "evidence" not in response
