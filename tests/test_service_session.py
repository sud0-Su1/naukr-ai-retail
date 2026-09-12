from app.chat.models import QueryPlan
from app.chat.service import ChatService
from app.chat.session import ChatSession


class FakePlanner:
    def plan(self, question: str) -> QueryPlan:
        return QueryPlan(
            intent="aggregate",
            dataset="canonical_retail",
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


def test_service_stores_successful_turn():
    session = ChatSession("test-session")

    service = ChatService(
        planner=FakePlanner(),
        session=session,
    )

    response = service.answer("Show revenue by category.")

    assert response["status"] == "ok"
    assert len(session.turns) == 1
    assert session.last_response == response
    assert session.last_plan == response["plan"]


def test_service_stores_multiple_turns():
    session = ChatSession("test-session")

    service = ChatService(
        planner=FakePlanner(),
        session=session,
    )

    first = service.answer("Show revenue by category.")
    second = service.answer("Show revenue by category again.")

    assert first["status"] == "ok"
    assert second["status"] == "ok"

    assert len(session.turns) == 2
    assert session.turns[0]["question"] == "Show revenue by category."
    assert session.turns[1]["question"] == "Show revenue by category again."


def test_service_stores_planner_failure():
    class FailingPlanner:
        def plan(self, question: str):
            raise RuntimeError("planner unavailable")

    session = ChatSession("test-session")

    service = ChatService(
        planner=FailingPlanner(),
        session=session,
    )

    response = service.answer("Show revenue.")

    assert response["status"] == "error"
    assert response["error_type"] == "planner_failure"
    assert len(session.turns) == 1
    assert session.last_response == response
