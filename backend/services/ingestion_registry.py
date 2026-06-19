"""Two-phase ingestion visibility registry.

Separate vector and graph stores cannot share a true ACID transaction. This
registry provides the next best prototype behavior: pending writes are hidden
from retrieval, successful writes are committed together, and failed writes are
marked non-visible for cleanup/retry.

The production path uses SQLite instead of a JSON file so concurrent API
workers do not lose updates through read-modify-write races. Tests can still set
``_records`` to a dict for in-memory isolation.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

from config import settings


@dataclass
class IngestionRecord:
    doc_id: str
    content_hash: str
    source: str
    status: str
    updated_at: float


class IngestionRegistry:
    """Persisted registry for ingestion visibility and idempotency."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] | None = None

    def begin(self, doc_id: str, source: str) -> tuple[bool, IngestionRecord]:
        """Start an ingestion attempt.

        Returns (skipped, record). skipped=True means the same content hash was
        already committed and the caller should not rewrite stores.
        """
        content_hash = self.compute_hash(source)
        record = {
            "doc_id": doc_id,
            "content_hash": content_hash,
            "source": os.path.abspath(source),
            "status": "pending",
            "updated_at": time.time(),
        }
        with self._lock:
            if self._records is not None:
                for existing in self._records.values():
                    if existing.get("content_hash") == content_hash and existing.get("status") == "committed":
                        return True, self._record_from_dict(existing)
                self._records[doc_id] = record
                return False, self._record_from_dict(record)

            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """
                    SELECT doc_id, content_hash, source, status, updated_at
                    FROM ingestion_records
                    WHERE content_hash = ? AND status = 'committed'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (content_hash,),
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return True, self._record_from_row(existing)
                conn.execute(
                    """
                    INSERT INTO ingestion_records(doc_id, content_hash, source, status, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        source = excluded.source,
                        status = excluded.status,
                        error = NULL,
                        failed_at = NULL,
                        updated_at = excluded.updated_at
                    """,
                    (record["doc_id"], record["content_hash"], record["source"], record["status"], record["updated_at"]),
                )
                conn.commit()
                return False, self._record_from_dict(record)

    def commit(self, doc_id: str) -> None:
        self._set_status(doc_id, "committed")

    def fail(self, doc_id: str, error: str = "") -> None:
        with self._lock:
            if self._records is not None:
                record = self._records.get(doc_id)
                if record is None:
                    return
                record["status"] = "failed"
                record["error"] = error
                record["failed_at"] = time.time()
                record["retry_count"] = int(record.get("retry_count", 0)) + 1
                record["updated_at"] = time.time()
                return

            now = time.time()
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE ingestion_records
                    SET status = 'failed', error = ?, failed_at = ?,
                        retry_count = retry_count + 1, updated_at = ?
                    WHERE doc_id = ?
                    """,
                    (error, now, now, doc_id),
                )
                conn.commit()

    def is_committed(self, doc_id: str | None) -> bool:
        if not doc_id:
            return True
        with self._lock:
            if self._records is not None:
                record = self._records.get(doc_id)
                return record is None or record.get("status") == "committed"
            with self._connect() as conn:
                row = conn.execute("SELECT status FROM ingestion_records WHERE doc_id = ?", (doc_id,)).fetchone()
                return row is None or row["status"] == "committed"

    def committed_doc_ids(self) -> set[str]:
        with self._lock:
            if self._records is not None:
                return {
                    doc_id for doc_id, record in self._records.items()
                    if record.get("status") == "committed"
                }
            with self._connect() as conn:
                rows = conn.execute("SELECT doc_id FROM ingestion_records WHERE status = 'committed'").fetchall()
                return {str(row["doc_id"]) for row in rows}

    def failed_records(self) -> list[IngestionRecord]:
        with self._lock:
            if self._records is not None:
                return [
                    self._record_from_dict(record)
                    for record in self._records.values()
                    if record.get("status") == "failed"
                ]
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT doc_id, content_hash, source, status, updated_at FROM ingestion_records WHERE status = 'failed'"
                ).fetchall()
                return [self._record_from_row(row) for row in rows]

    def dead_letters(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._records is not None:
                return [
                    self._dead_letter_from_dict(doc_id, record)
                    for doc_id, record in self._records.items()
                    if record.get("status") == "failed"
                ]
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT doc_id, source, content_hash, error, retry_count, failed_at
                    FROM ingestion_records
                    WHERE status = 'failed'
                    ORDER BY failed_at DESC, updated_at DESC
                    """
                ).fetchall()
                return [self._dead_letter_from_row(row) for row in rows]

    def clear_dead_letter(self, doc_id: str) -> bool:
        with self._lock:
            if self._records is not None:
                record = self._records.get(doc_id)
                if record is None or record.get("status") != "failed":
                    return False
                del self._records[doc_id]
                return True
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM ingestion_records WHERE doc_id = ? AND status = 'failed'", (doc_id,))
                conn.commit()
                return cursor.rowcount > 0

    def dead_letter(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            if self._records is not None:
                record = self._records.get(doc_id)
                if record is None or record.get("status") != "failed":
                    return None
                return self._dead_letter_from_dict(doc_id, record)
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT doc_id, source, content_hash, error, retry_count, failed_at
                    FROM ingestion_records
                    WHERE doc_id = ? AND status = 'failed'
                    """,
                    (doc_id,),
                ).fetchone()
                return None if row is None else self._dead_letter_from_row(row)

    @staticmethod
    def compute_hash(path: str) -> str:
        digest = hashlib.sha256()
        if not path or not os.path.exists(path):
            return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _set_status(self, doc_id: str, status: str) -> None:
        with self._lock:
            if self._records is not None:
                record = self._records.get(doc_id)
                if record is None:
                    return
                record["status"] = status
                record["updated_at"] = time.time()
                return
            with self._connect() as conn:
                conn.execute(
                    "UPDATE ingestion_records SET status = ?, updated_at = ? WHERE doc_id = ?",
                    (status, time.time(), doc_id),
                )
                conn.commit()

    def _connect(self) -> sqlite3.Connection:
        path = self._path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_records (
                doc_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'committed', 'failed')),
                error TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                failed_at REAL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_hash_status ON ingestion_records(content_hash, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_records(status)")
        return conn

    @staticmethod
    def _path() -> str:
        return os.path.abspath(os.path.join(settings.upload_dir, "..", "ingestion_registry.sqlite3"))

    @staticmethod
    def _record_from_dict(data: dict[str, Any]) -> IngestionRecord:
        return IngestionRecord(
            doc_id=str(data["doc_id"]),
            content_hash=str(data["content_hash"]),
            source=str(data["source"]),
            status=str(data["status"]),
            updated_at=float(data["updated_at"]),
        )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> IngestionRecord:
        return IngestionRecord(
            doc_id=str(row["doc_id"]),
            content_hash=str(row["content_hash"]),
            source=str(row["source"]),
            status=str(row["status"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _dead_letter_from_dict(doc_id: str, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "source": record.get("source", ""),
            "content_hash": record.get("content_hash", ""),
            "error": record.get("error", ""),
            "retry_count": int(record.get("retry_count", 0)),
            "failed_at": record.get("failed_at"),
        }

    @staticmethod
    def _dead_letter_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "doc_id": str(row["doc_id"]),
            "source": str(row["source"]),
            "content_hash": str(row["content_hash"]),
            "error": "" if row["error"] is None else str(row["error"]),
            "retry_count": int(row["retry_count"]),
            "failed_at": row["failed_at"],
        }


ingestion_registry = IngestionRegistry()
