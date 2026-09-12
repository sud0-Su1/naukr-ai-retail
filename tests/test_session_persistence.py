from pathlib import Path

import pytest

from app.chat.session import ChatSession
from app.chat.session_store import ChatSessionStore


def test_session_can_be_saved_and_loaded(tmp_path):
    store = ChatSessionStore(tmp_path)

    session = ChatSession("persistent-test")

    session.add_turn(
        "Show revenue by category in the West.",
        {
            "status": "ok",
            "plan": {
                "intent": "aggregate",
                "dataset": "canonical_retail",
                "group_by": ["category_normalized"],
            },
            "result": [
                {
                    "category_normalized": "T-Shirt",
                    "sales": 3337.2,
                }
            ],
        },
    )

    path = store.save(session)

    assert path.exists()
    assert path.name == "persistent-test.json"

    loaded = store.load("persistent-test")

    assert loaded.session_id == "persistent-test"
    assert loaded.turns == session.turns
    assert loaded.last_plan == session.last_plan


def test_multiple_sessions_are_isolated(tmp_path):
    store = ChatSessionStore(tmp_path)

    session_a = ChatSession("session-a")
    session_b = ChatSession("session-b")

    session_a.add_turn(
        "Question A",
        {"status": "ok", "plan": {"dataset": "canonical_retail"}},
    )

    session_b.add_turn(
        "Question B",
        {"status": "ok", "plan": {"dataset": "canonical_retail"}},
    )

    store.save(session_a)
    store.save(session_b)

    loaded_a = store.load("session-a")
    loaded_b = store.load("session-b")

    assert loaded_a.turns[0]["question"] == "Question A"
    assert loaded_b.turns[0]["question"] == "Question B"


def test_missing_session_raises_file_not_found(tmp_path):
    store = ChatSessionStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load("does-not-exist")


def test_session_id_path_traversal_is_rejected(tmp_path):
    store = ChatSessionStore(tmp_path)

    with pytest.raises(ValueError):
        store.save(
            ChatSession("../outside")
        )

    with pytest.raises(ValueError):
        store.load("../outside")


def test_session_exists(tmp_path):
    store = ChatSessionStore(tmp_path)

    assert store.exists("exists-test") is False

    session = ChatSession("exists-test")
    store.save(session)

    assert store.exists("exists-test") is True


def test_session_can_be_deleted(tmp_path):
    store = ChatSessionStore(tmp_path)

    session = ChatSession("delete-test")
    store.save(session)

    assert store.exists("delete-test") is True

    store.delete("delete-test")

    assert store.exists("delete-test") is False


def test_saved_session_is_valid_json(tmp_path):
    store = ChatSessionStore(tmp_path)

    session = ChatSession("json-test")
    session.add_turn(
        "Test question",
        {"status": "error", "message": "test"},
    )

    path = store.save(session)

    text = path.read_text(encoding="utf-8")

    assert '"session_id": "json-test"' in text
    assert '"Test question"' in text
