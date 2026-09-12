from app.chat.llm_planner import OllamaPlannerError
from app.chat.service import ChatService


def test_empty_question():
    service = ChatService()

    result = service.answer("")

    assert result["status"] == "error"
    assert result["error_type"] == "empty_question"


def test_planner_failure(monkeypatch):
    service = ChatService()

    def failing_plan(question):
        raise OllamaPlannerError(
            "simulated Ollama outage"
        )

    monkeypatch.setattr(
        service.planner,
        "plan",
        failing_plan,
    )

    result = service.answer(
        "Which category had the highest revenue?"
    )

    assert result["status"] == "error"
    assert result["error_type"] == "planner_failure"


def test_invalid_plan_is_rejected(monkeypatch):
    service = ChatService()

    class FakePlan:
        def model_dump(self, by_alias=True):
            return {}

    def fake_plan(question):
        from app.chat.models import QueryPlan

        return QueryPlan(
            intent="aggregate",
            dataset="canonical_retail",
            group_by=["not_allowed"],
        )

    monkeypatch.setattr(
        service.planner,
        "plan",
        fake_plan,
    )

    result = service.answer(
        "Tell me something strange"
    )

    assert result["status"] == "error"
    assert result["error_type"] == "invalid_plan"

def test_successful_grounded_answer(monkeypatch):
    service = ChatService()

    from app.chat.models import QueryPlan

    def fake_plan(question):
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
            group_by=[
                "category_normalized"
            ],
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

    monkeypatch.setattr(
        service.planner,
        "plan",
        fake_plan,
    )

    result = service.answer(
        "Which category had the highest revenue in West?"
    )

    assert result["status"] == "ok"
    assert result["result"][0][
        "category_normalized"
    ] == "T-Shirt"  