from app.chat.session import ChatSession


def test_session_stores_turn():
    session = ChatSession("test-session")

    response = {
        "status": "ok",
        "plan": {
            "dataset": "canonical_retail",
            "filters": [],
        },
    }

    session.add_turn(
        "Show revenue",
        response,
    )

    assert len(session.turns) == 1
    assert session.last_response == response
    assert session.last_plan == response["plan"]


def test_empty_session_has_no_previous_context():
    session = ChatSession("test-session")

    assert session.last_response is None
    assert session.last_plan is None


def test_session_context_contains_previous_turn():
    session = ChatSession("test-session")

    response = {
        "status": "ok",
        "plan": {
            "dataset": "canonical_retail",
            "filters": [
                {
                    "field": "store_region",
                    "op": "eq",
                    "value": "West",
                }
            ],
        },
    }

    session.add_turn("Show revenue in the West.", response)

    context = session.context()

    assert context["has_previous_turn"] is True
    assert context["previous_question"] == "Show revenue in the West."
    assert context["previous_plan"] == response["plan"]


def test_empty_session_context():
    session = ChatSession("test-session")

    assert session.context() == {
        "has_previous_turn": False,
    }
