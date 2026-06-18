"""Two-phase ingestion visibility registry.

Separate vector and graph stores cannot share a true ACID transaction. This
registry provides the next best prototype behavior: pending writes are hidden
from retrieval, successful writes are committed together, and failed writes are
marked non-visible for cleanup/retry.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
    """Small persisted registry for ingestion visibility and idempotency."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] | None = None

    def begin(self, doc_id: str, source: str) -> tuple[bool, IngestionRecord]:
        """Start an ingestion attempt.

        Returns (skipped, record). skipped=True means the same content hash was
        already committed and the caller should not rewrite stores.
        """
        content_hash = self.compute_hash(source)
        with self._lock:
            records = self._load()
            for record in records.values():
                if record.get("content_hash") == content_hash and record.get("status") == "committed":
                    return True, self._record_from_dict(record)

            record = {
                "doc_id": doc_id,
                "content_hash": content_hash,
                "source": os.path.abspath(source),
                "status": "pending",
                "updated_at": time.time(),
            }
            records[doc_id] = record
            self._save(records)
            return False, self._record_from_dict(record)

    def commit(self, doc_id: str) -> None:
        self._set_status(doc_id, "committed")

    def fail(self, doc_id: str, error: str = "") -> None:
        with self._lock:
            records = self._load()
            record = records.get(doc_id)
            if record is None:
                return
            record["status"] = "failed"
            record["error"] = error
            record["failed_at"] = time.time()
            record["retry_count"] = int(record.get("retry_count", 0)) + 1
            record["updated_at"] = time.time()
            self._save(records)

    def is_committed(self, doc_id: str | None) -> bool:
        if not doc_id:
            return True
        with self._lock:
            record = self._load().get(doc_id)
            return record is None or record.get("status") == "committed"

    def committed_doc_ids(self) -> set[str]:
        with self._lock:
            return {
                doc_id for doc_id, record in self._load().items()
                if record.get("status") == "committed"
            }

    def failed_records(self) -> list[IngestionRecord]:
        with self._lock:
            return [
                self._record_from_dict(record)
                for record in self._load().values()
                if record.get("status") == "failed"
            ]

    def dead_letters(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "doc_id": doc_id,
                    "source": record.get("source", ""),
                    "content_hash": record.get("content_hash", ""),
                    "error": record.get("error", ""),
                    "retry_count": int(record.get("retry_count", 0)),
                    "failed_at": record.get("failed_at"),
                }
                for doc_id, record in self._load().items()
                if record.get("status") == "failed"
            ]

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
            records = self._load()
            record = records.get(doc_id)
            if record is None:
                return
            record["status"] = status
            record["updated_at"] = time.time()
            self._save(records)

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._records is not None:
            return self._records
        path = self._path()
        if not os.path.exists(path):
            self._records = {}
            return self._records
        try:
            with open(path, encoding="utf-8") as handle:
                self._records = json.load(handle)
        except (OSError, json.JSONDecodeError):
            self._records = {}
        return self._records

    def _save(self, records: dict[str, dict[str, Any]]) -> None:
        path = self._path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix="ingestion-registry-", suffix=".json", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2, sort_keys=True)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        self._records = records

    @staticmethod
    def _path() -> str:
        return os.path.abspath(os.path.join(settings.upload_dir, "..", "ingestion_registry.json"))

    @staticmethod
    def _record_from_dict(data: dict[str, Any]) -> IngestionRecord:
        return IngestionRecord(
            doc_id=str(data["doc_id"]),
            content_hash=str(data["content_hash"]),
            source=str(data["source"]),
            status=str(data["status"]),
            updated_at=float(data["updated_at"]),
        )


ingestion_registry = IngestionRegistry()
