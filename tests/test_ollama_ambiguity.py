import json

from app.chat.llm_planner import OllamaPlanner
from app.chat.models import ClarificationRequest, QueryPlan
from app.chat.service import ChatService
from app.chat.session import ChatSession


class FakeOllamaResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_planner_maps_top_category_with_mocked_response(
    monkeypatch,
):
    plan_payload = {
        "intent": "top_n",
        "dataset": "canonical_retail",
        "filters": [
            {
                "field": "store_region",
                "op": "eq",
                "value": "West",
            }
        ],
        "group_by": ["category_normalized"],
        "metrics": [
            {
                "agg": "sum",
                "field": "revenue",
                "as": "sales",
            }
        ],
        "sort": [
            {
                "field": "sales",
                "dir": "desc",
            }
        ],
        "limit": 1,
    }

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeOllamaResponse(
            {
                "message": {
                    "content": json.dumps(plan_payload),
                }
            }
        ),
    )

    response = OllamaPlanner().plan(
        "What is the top category by revenue in the West?"
    )

    assert isinstance(response, QueryPlan)
    assert response.group_by == ["category_normalized"]
    assert response.metrics[0].agg == "sum"
    assert response.metrics[0].field == "revenue"
    assert response.filters[0].field == "store_region"
    assert response.sort[0].dir == "desc"
    assert response.limit == 1


def test_ollama_planner_updates_store_region_follow_up_without_llm(
):
    context = {
        "has_previous_turn": True,
        "previous_question": (
            "What is the top category by revenue in the West?"
        ),
        "previous_plan": {
            "intent": "top_n",
            "dataset": "canonical_retail",
            "filters": [
                {
                    "field": "store_region",
                    "op": "eq",
                    "value": "West",
                }
            ],
            "group_by": ["category_normalized"],
            "metrics": [
                {
                    "agg": "sum",
                    "field": "revenue",
                    "as": "sales",
                }
            ],
            "sort": [
                {
                    "field": "sales",
                    "dir": "desc",
                }
            ],
            "limit": 1,
        },
    }

    response = OllamaPlanner().plan(
        "What about the East?",
        context=context,
    )

    assert isinstance(response, QueryPlan)
    assert response.filters[0].field == "store_region"
    assert response.filters[0].value == "East"
    assert response.group_by == ["category_normalized"]
    assert response.metrics[0].field == "revenue"
    assert response.sort[0].dir == "desc"
    assert response.limit == 1


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
