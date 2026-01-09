"""SQLite-backed stores for logs and memory fragments."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..schema.interaction_log import InteractionLog
from ..schema.memory_fragment import MemoryFragment


class SQLiteLogStore:
    """Persist interaction logs to a SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_interaction_logs_session_time
                    ON interaction_logs (session_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_interaction_logs_time
                    ON interaction_logs (timestamp);
                """
            )

    def append(self, *, logs: Sequence[InteractionLog]) -> None:
        if not logs:
            return
        rows = []
        for log in logs:
            payload = log.to_dict()
            rows.append(
                (
                    log.id,
                    log.session_id,
                    log.turn_index,
                    log.timestamp.isoformat(),
                    json.dumps(payload, ensure_ascii=True),
                )
            )
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT OR REPLACE INTO interaction_logs
                        (id, session_id, turn_index, timestamp, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def get(self, *, log_id: str) -> Optional[InteractionLog]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM interaction_logs WHERE id = ?", (log_id,)
            ).fetchone()
        if not row:
            return None
        return InteractionLog.from_dict(json.loads(row["payload"]))

    def query_time_range(
        self, *, session_id: Optional[str], start: datetime, end: datetime
    ) -> Iterable[InteractionLog]:
        params = [start.isoformat(), end.isoformat()]
        query = "SELECT payload FROM interaction_logs WHERE timestamp BETWEEN ? AND ?"
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        for row in rows:
            yield InteractionLog.from_dict(json.loads(row["payload"]))

    def list_recent(
        self,
        *,
        session_id: Optional[str] = None,
        limit: int = 50,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Iterable[InteractionLog]:
        query = "SELECT payload FROM interaction_logs"
        clauses = []
        params = []
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end.isoformat())
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        for row in rows:
            yield InteractionLog.from_dict(json.loads(row["payload"]))

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SQLiteMemoryStore:
    """Persist memory fragments to a SQLite database."""

    def __init__(self, db_path: Path, *, include_embeddings: bool = False) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._include_embeddings = include_embeddings
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_fragments (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_fragments_session_time
                    ON memory_fragments (session_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_memory_fragments_time
                    ON memory_fragments (timestamp);
                """
            )

    def add(self, *, fragments: Sequence[MemoryFragment]) -> None:
        if not fragments:
            return
        rows = []
        for fragment in fragments:
            payload = fragment.to_dict(include_embeddings=self._include_embeddings)
            rows.append(
                (
                    fragment.id,
                    fragment.session_id,
                    fragment.timestamp.isoformat(),
                    json.dumps(payload, ensure_ascii=True),
                )
            )
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT OR REPLACE INTO memory_fragments
                        (id, session_id, timestamp, payload)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )

    def get(self, *, fragment_id: str) -> Optional[MemoryFragment]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM memory_fragments WHERE id = ?", (fragment_id,)
            ).fetchone()
        if not row:
            return None
        return MemoryFragment.from_dict(json.loads(row["payload"]))

    def query_time_range(
        self, *, session_id: Optional[str], start: datetime, end: datetime
    ) -> Iterable[MemoryFragment]:
        params = [start.isoformat(), end.isoformat()]
        query = "SELECT payload FROM memory_fragments WHERE timestamp BETWEEN ? AND ?"
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        for row in rows:
            yield MemoryFragment.from_dict(json.loads(row["payload"]))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
