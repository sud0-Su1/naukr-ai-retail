import json

from app.chat.llm_planner import OllamaPlannerError
from app.chat.planner import MockPlanner
from app.chat.service import ChatService
from app.chat.session import ChatSession


class FailingPlanner:
    def plan(self, question, context=None):
        raise OllamaPlannerError("Ollama request failed: connection refused")


class InvalidResponsePlanner:
    def plan(self, question, context=None):
        raise OllamaPlannerError("Ollama returned invalid JSON.")


class ExplodingDatasetPlanner:
    def plan(self, question, context=None):
        raise OllamaPlannerError("Ollama request failed")


def test_ollama_outage_returns_graceful_error():
    session = ChatSession("ollama-outage-test")

    service = ChatService(
        planner=FailingPlanner(),
        session=session,
    )

    response = service.answer(
        "Show revenue by category in the West."
    )

    assert response["status"] == "error"
    assert response["error_type"] == "planner_failure"
    assert "Ollama request failed" in response["message"]
    assert "result" not in response
    assert "evidence" not in response
    assert len(session.turns) == 1


def test_invalid_json_returns_graceful_error():
    session = ChatSession("invalid-json-test")

    service = ChatService(
        planner=InvalidResponsePlanner(),
        session=session,
    )

    response = service.answer(
        "Show revenue by category in the West."
    )

    assert response["status"] == "error"
    assert response["error_type"] == "planner_failure"
    assert "invalid JSON" in response["message"]
    assert "result" not in response
    assert "evidence" not in response
    assert len(session.turns) == 1


def test_planner_failure_does_not_execute_query():
    class FailingPlanner:
        def plan(self, question, context=None):
            raise OllamaPlannerError(
                "Ollama request failed"
            )

    session = ChatSession("no-execution-after-failure")

    service = ChatService(
        planner=FailingPlanner(),
        session=session,
    )

    response = service.answer(
        "Show revenue by category in the West."
    )

    assert response["status"] == "error"
    assert response["error_type"] == "planner_failure"

    # A planner failure happens before dataset loading,
    # validation, or execution.
    assert "validation" not in response
    assert "result" not in response
    assert "evidence" not in response


def test_real_planner_exception_is_converted_to_structured_error():
    class GenericFailingPlanner:
        def plan(self, question, context=None):
            raise RuntimeError("unexpected planner crash")

    service = ChatService(
        planner=GenericFailingPlanner(),
        session=ChatSession("generic-planner-failure"),
    )

    response = service.answer(
        "Show revenue by category in the West."
    )

    assert response["status"] == "error"
    assert response["error_type"] == "planner_failure"
    assert response["message"] == "unexpected planner crash"


def test_normal_planner_still_works_after_failure_tests():
    service = ChatService(
        planner=MockPlanner(),
        session=ChatSession("normal-after-failure"),
    )

    response = service.answer(
        "Show revenue by category in the West."
    )

    assert response["status"] == "ok"
    assert "result" in response
    assert "evidence" in response
