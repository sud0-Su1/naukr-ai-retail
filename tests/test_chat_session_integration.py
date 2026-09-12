from pathlib import Path

from app.chat.service import ChatService
from app.chat.session_store import ChatSessionStore


class FakePlanner:
    def __init__(self):
        self.calls = []

    def plan(self, question, context=None):
        self.calls.append(
            {
                "question": question,
                "context": context,
            }
        )

        from app.chat.models import QueryPlan

        if question.lower() == "show revenue by category in the west.":
            return QueryPlan(
                intent="aggregate",
                dataset="canonical_retail",
                filters=[
                    {
                        "field": "store_region",
                        "op": "eq",
                        "value": "West",
                    }
                ],
                group_by=["category_normalized"],
                metrics=[
                    {
                        "agg": "sum",
                        "field": "revenue",
                        "as": "sales",
                    }
                ],
                sort=[
                    {
                        "field": "sales",
                        "dir": "desc",
                    }
                ],
                limit=10,
            )

        raise ValueError(f"Unsupported test question: {question}")


def test_chat_service_saves_session(tmp_path):
    store = ChatSessionStore(tmp_path / "sessions")
    planner = FakePlanner()

    service = ChatService(
        planner=planner,
        session_store=store,
        session_id="integration-test",
    )

    response = service.answer(
        "Show revenue by category in the West."
    )

    assert response["status"] == "ok"
    assert store.exists("integration-test")


def test_chat_service_loads_existing_session(tmp_path):
    store = ChatSessionStore(tmp_path / "sessions")

    first_planner = FakePlanner()

    first_service = ChatService(
        planner=first_planner,
        session_store=store,
        session_id="persistent-test",
    )

    first_response = first_service.answer(
        "Show revenue by category in the West."
    )

    assert first_response["status"] == "ok"

    second_planner = FakePlanner()

    second_service = ChatService(
        planner=second_planner,
        session_store=store,
        session_id="persistent-test",
    )

    assert len(second_service.session.turns) == 1

    second_service.answer(
        "Show revenue by category in the West."
    )

    assert second_planner.calls[0]["context"]["has_previous_turn"] is True


def test_different_sessions_are_isolated(tmp_path):
    store = ChatSessionStore(tmp_path / "sessions")

    planner_a = FakePlanner()
    service_a = ChatService(
        planner=planner_a,
        session_store=store,
        session_id="session-a",
    )

    service_a.answer(
        "Show revenue by category in the West."
    )

    planner_b = FakePlanner()
    service_b = ChatService(
        planner=planner_b,
        session_store=store,
        session_id="session-b",
    )

    assert service_b.session.turns == []
    assert planner_b.calls == []


def test_session_persistence_does_not_store_dataset_rows_in_context(
    tmp_path,
):
    store = ChatSessionStore(tmp_path / "sessions")
    planner = FakePlanner()

    service = ChatService(
        planner=planner,
        session_store=store,
        session_id="security-test",
    )

    service.answer(
        "Show revenue by category in the West."
    )

    loaded = store.load("security-test")
    context = loaded.context()

    assert "previous_question" in context
    assert "previous_plan" in context
    assert "previous_result" not in context
    assert "result" not in context
