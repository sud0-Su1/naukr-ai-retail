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
