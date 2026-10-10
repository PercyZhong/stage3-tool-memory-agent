from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, path: Path, namespace: str):
        self.path = path
        self.namespace = namespace
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, content TEXT NOT NULL,
            created_index INTEGER NOT NULL, updated_index INTEGER NOT NULL,
            source_event TEXT NOT NULL, status TEXT NOT NULL, invalid_reason TEXT)"""
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def write(self, content: dict[str, Any], logical_index: int, source_event: str) -> dict[str, Any]:
        memory_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.namespace}:{source_event}:{logical_index}:{json.dumps(content, sort_keys=True)}"))
        self.connection.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, 'active', NULL)",
            (memory_id, self.namespace, json.dumps(content, ensure_ascii=False, sort_keys=True), logical_index, logical_index, source_event),
        )
        self.connection.commit()
        return self.get(memory_id)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = {term.casefold() for term in query.replace("_", " ").split() if term}
        rows = self.connection.execute(
            "SELECT * FROM memories WHERE namespace=? AND status='active'", (self.namespace,)
        ).fetchall()
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for row in rows:
            item = self._row(row)
            text = json.dumps(item["content"], ensure_ascii=False).casefold()
            score = sum(1 for term in terms if term in text)
            ranked.append((score, item["updated_index"], item))
        ranked.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]["memory_id"]))
        return [item | {"rank": index + 1, "score": score} for index, (score, _, item) in enumerate(ranked[:limit]) if score > 0]

    def invalidate(self, query: str, logical_index: int, reason: str) -> list[dict[str, Any]]:
        hits = self.search(query, limit=50)
        for item in hits:
            self.connection.execute(
                "UPDATE memories SET status='invalid', invalid_reason=?, updated_index=? WHERE memory_id=?",
                (reason, logical_index, item["memory_id"]),
            )
        self.connection.commit()
        return [self.get(item["memory_id"]) for item in hits]

    def get(self, memory_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._row(row)

    def snapshot(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM memories WHERE namespace=? ORDER BY created_index, memory_id", (self.namespace,)
        ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "memory_id": row["memory_id"], "namespace": row["namespace"],
            "content": json.loads(row["content"]), "created_index": row["created_index"],
            "updated_index": row["updated_index"], "source_event": row["source_event"],
            "status": row["status"], "invalid_reason": row["invalid_reason"],
        }
