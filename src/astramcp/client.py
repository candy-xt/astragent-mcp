"""AstrBot HTTP API client and task persistence (SQLite)."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import httpx


def _extract_text(raw: str) -> str:
    """Try to extract plain text from a possible JSON-wrapped AstrBot response.

    AstrBot's SSE ``complete`` event may return ``data`` as a JSON string
    like ``{"content": "...", "thinking": "..."}``.  If so, extract the
    ``content`` field; otherwise return *raw* as-is.
    """
    if not raw or raw.isspace():
        return raw
    # Quick check: does it look like JSON?
    stripped = raw.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return raw
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return raw
    if isinstance(obj, dict):
        # AstrBot often wraps the answer under "content"
        for key in ("content", "message", "reply", "text", "answer", "data", "result"):
            val = obj.get(key)
            if val and isinstance(val, str):
                # Recursively unwrap if the value is itself JSON-encoded
                if (val.startswith("{") or val.startswith("[")) and '"' in val[:80]:
                    inner = _extract_text(val)
                    if inner != val:
                        return inner
                return val
        # Fallback: concatenate all string values
        parts = [str(v) for v in obj.values() if isinstance(v, str) and v]
        if parts:
            return "\n".join(parts)
    return raw

_TASK_TTL = 3600 * 24 * 7  # keep tasks for 7 days


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class BackgroundTask:
    task_id: str
    session_id: str
    profile: str = ""
    agent: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0


# ---------------------------------------------------------------------------
# SQLite-backed task store
# ---------------------------------------------------------------------------

class TaskStore:
    """Thread-safe SQLite-backed task store."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".astramcp" / "tasks.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id     TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    profile     TEXT NOT NULL DEFAULT '',
                    agent       TEXT NOT NULL DEFAULT '',
                    status      TEXT NOT NULL DEFAULT 'pending',
                    result      TEXT NOT NULL DEFAULT '',
                    error       TEXT NOT NULL DEFAULT '',
                    created_at  REAL NOT NULL,
                    finished_at REAL NOT NULL DEFAULT 0.0
                )
            """)

    def _row_to_task(self, row: sqlite3.Row) -> BackgroundTask:
        return BackgroundTask(
            task_id=row["task_id"],
            session_id=row["session_id"],
            profile=row["profile"],
            agent=row["agent"],
            status=TaskStatus(row["status"]),
            result=row["result"],
            error=row["error"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
        )

    def create(self, session_id: str, profile: str = "", agent: str = "") -> BackgroundTask:
        task = BackgroundTask(
            task_id=str(uuid.uuid4()),
            session_id=session_id,
            profile=profile,
            agent=agent,
            created_at=time.time(),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    task.task_id, task.session_id, task.profile, task.agent,
                    task.status.value, task.result, task.error,
                    task.created_at, task.finished_at,
                ),
            )
        return task

    def get(self, task_id: str) -> BackgroundTask | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._row_to_task(row) if row else None

    def update(self, task_id: str, **kwargs: object) -> BackgroundTask | None:
        if not kwargs:
            return self.get(task_id)
        # Convert TaskStatus enum to string
        if "status" in kwargs and isinstance(kwargs["status"], TaskStatus):
            kwargs["status"] = kwargs["status"].value
        cols = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [task_id]
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE tasks SET {cols} WHERE task_id = ?", vals)
        return self.get(task_id)

    def list_by_profile(self, profile: str) -> list[BackgroundTask]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE profile = ? ORDER BY created_at DESC", (profile,)
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def list_tasks(self, limit: int = 100) -> list[BackgroundTask]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def list_by_group(self, group: str, limit: int = 100) -> list[BackgroundTask]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE profile = ? ORDER BY created_at DESC LIMIT ?",
                (group, limit),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def evict_old(self) -> None:
        """Remove tasks older than TTL that are done/error."""
        cutoff = time.time() - _TASK_TTL
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM tasks WHERE status IN ('done','error') AND finished_at > 0 AND finished_at < ?",
                (cutoff,),
            )


# Global task store (shared across the daemon process)
task_store = TaskStore()


def get_task(task_id: str) -> BackgroundTask | None:
    return task_store.get(task_id)


# ---------------------------------------------------------------------------
# AstrBot API Client
# ---------------------------------------------------------------------------

class AstrBotClient:
    def __init__(self, base_url: str, api_key: str = "", username: str = "astramcp") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.username = username

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_configs(self) -> list[dict]:
        """GET /api/v1/configs."""
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{self.base_url}/api/v1/configs",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json().get("data") or {}
            return data.get("configs") or []

    def list_sessions(self, username: str | None = None) -> list[dict]:
        """GET /api/v1/chat/sessions."""
        params: dict = {"username": username or self.username}
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{self.base_url}/api/v1/chat/sessions",
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json().get("data") or {}
            return data.get("sessions") or []

    async def chat_async(
        self,
        message: str,
        config_id: str | None = None,
        config_name: str | None = None,
        session_id: str | None = None,
        timeout: int = 60,
    ) -> str:
        """POST /api/v1/chat — async SSE stream, returns complete plain-text.

        Supports both ``delta`` streaming chunks and ``complete`` events.
        If the final data is JSON-encoded, ``_extract_text`` is used to
        return plain text.
        """
        payload: dict = {
            "message": message,
            "username": self.username,
            "enable_streaming": True,
        }
        if session_id:
            payload["session_id"] = session_id
        if config_id:
            payload["config_id"] = config_id
        if config_name and not config_id:
            payload["config_name"] = config_name

        accumulated: str = ""
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/v1/chat",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    t = obj.get("type", "")
                    if t == "delta":
                        chunk = obj.get("data", "")
                        if chunk:
                            accumulated += chunk
                    elif t == "complete":
                        accumulated = obj.get("data", "")
                    elif t == "error":
                        raise RuntimeError(
                            obj.get("data") or obj.get("message") or "AstrBot error"
                        )

        return _extract_text(accumulated)

    def chat(
        self,
        message: str,
        config_id: str | None = None,
        config_name: str | None = None,
        session_id: str | None = None,
        timeout: int = 60,
    ) -> str:
        """Sync wrapper — runs chat_async in a fresh event loop."""
        import asyncio
        return asyncio.run(
            self.chat_async(
                message=message,
                config_id=config_id,
                config_name=config_name,
                session_id=session_id,
                timeout=timeout,
            )
        )

    def chat_background(
        self,
        message: str,
        config_id: str | None = None,
        config_name: str | None = None,
        session_id: str | None = None,
        profile: str = "",
        agent: str = "",
        on_done: Callable[[BackgroundTask], None] | None = None,
    ) -> BackgroundTask:
        """Fire-and-forget chat with streaming. Returns a BackgroundTask immediately.

        The task store is updated incrementally as SSE chunks arrive,
        enabling real-time monitoring via the daemon's SSE endpoint.
        """
        task = task_store.create(
            session_id=session_id or f"astramcp_bg_{uuid.uuid4()}",
            profile=profile,
            agent=agent,
        )

        def _run() -> None:
            task_store.update(task.task_id, status=TaskStatus.RUNNING)
            try:
                final_text = self._chat_streaming(
                    message=message,
                    config_id=config_id,
                    config_name=config_name,
                    session_id=task.session_id,
                    task_id=task.task_id,
                    timeout=300,
                )
                task_store.update(
                    task.task_id,
                    result=final_text,
                    status=TaskStatus.DONE,
                    finished_at=time.time(),
                )
            except Exception as e:
                task_store.update(
                    task.task_id,
                    error=str(e),
                    status=TaskStatus.ERROR,
                    finished_at=time.time(),
                )
            finally:
                if on_done:
                    on_done(task_store.get(task.task_id) or task)

        threading.Thread(target=_run, daemon=True).start()
        return task

    def _chat_streaming(
        self,
        message: str,
        config_id: str | None = None,
        config_name: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        timeout: int = 300,
    ) -> str:
        """Consume SSE stream and update task store with partial results."""
        payload: dict = {
            "message": message,
            "username": self.username,
            "enable_streaming": True,
        }
        if session_id:
            payload["session_id"] = session_id
        if config_id:
            payload["config_id"] = config_id
        if config_name and not config_id:
            payload["config_name"] = config_name

        accumulated = ""

        async def _stream() -> str:
            nonlocal accumulated
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/v1/chat",
                    headers=self._headers(),
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        t = obj.get("type", "")
                        if t == "delta":
                            chunk = obj.get("data", "")
                            if chunk:
                                accumulated += chunk
                                if task_id:
                                    task_store.update(task_id, result=accumulated)
                        elif t == "complete":
                            final = obj.get("data", "")
                            accumulated = final
                            if task_id:
                                task_store.update(task_id, result=accumulated)
                        elif t == "error":
                            raise RuntimeError(
                                obj.get("data") or obj.get("message") or "AstrBot error"
                            )
            return _extract_text(accumulated)

        import asyncio
        return asyncio.run(_stream())
