from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.chat.session import ChatSession


class ChatSessionStore:
    """
    Lightweight JSON-backed persistence for chat sessions.

    Each session is stored independently so that one session cannot
    accidentally overwrite another session's conversation.
    """

    def __init__(
        self,
        storage_dir: str | Path = "artifacts/chat_sessions",
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _path(self, session_id: str) -> Path:
        """
        Return a safe path for a session.

        Session IDs are treated as untrusted input and must not be able
        to escape the configured storage directory.
        """
        safe_id = str(session_id)

        if not safe_id:
            raise ValueError("Session ID cannot be empty.")

        if (
            "/" in safe_id
            or "\\" in safe_id
            or safe_id in {".", ".."}
            or ".." in Path(safe_id).parts
        ):
            raise ValueError("Invalid session ID.")

        path = (self.storage_dir / f"{safe_id}.json").resolve()
        root = self.storage_dir.resolve()

        if root not in path.parents:
            raise ValueError("Invalid session path.")

        return path

    def save(self, session: ChatSession) -> Path:
        """
        Persist a complete chat session to JSON.
        """
        path = self._path(session.session_id)

        payload = {
            "session_id": session.session_id,
            "turns": session.turns,
        }

        temporary_path = path.with_suffix(".tmp")

        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(path)

        return path

    def load(self, session_id: str) -> ChatSession:
        """
        Load a persisted session.

        Raises FileNotFoundError when the session does not exist.
        """
        path = self._path(session_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Chat session '{session_id}' does not exist."
            )

        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        stored_session_id = payload.get("session_id")

        if stored_session_id != session_id:
            raise ValueError(
                "Persisted session ID does not match requested session."
            )

        turns = payload.get("turns", [])

        if not isinstance(turns, list):
            raise ValueError(
                "Persisted session turns must be a list."
            )

        return ChatSession(
            session_id=session_id,
            turns=turns,
        )

    def exists(self, session_id: str) -> bool:
        """
        Return whether a session has been persisted.
        """
        return self._path(session_id).exists()

    def delete(self, session_id: str) -> None:
        """
        Delete a persisted session.
        """
        path = self._path(session_id)

        if path.exists():
            path.unlink()
