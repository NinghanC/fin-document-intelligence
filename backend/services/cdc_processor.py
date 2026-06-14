"""
CDC (Change Data Capture) incremental processor

Technical highlights:
  Traditional approaches rebuild everything, which is expensive and high-latency
  The CDC approach listens for data change events and processes only incremental changes

Supports two CDC sources:
  1. File-system-level CDC - Watchdog monitors file changes
  2. Database-level CDC - Kafka Connect monitors the DB binlog

Incremental update flow:
  Change event -> delta analysis -> incremental parsing -> incremental vectorization -> incremental graph update
              ↓
          Version management (each knowledge node has a version and timestamp)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings


@dataclass
class CDCEvent:
    """Unified CDC event format"""
    event_id: str
    source_type: str  # "filesystem" | "database" | "api"
    operation: str    # "INSERT" | "UPDATE" | "DELETE"
    resource_path: str
    timestamp: float = field(default_factory=time.time)
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    diff: dict[str, Any] | None = None


@dataclass
class CDCProcessResult:
    event: CDCEvent
    chunks_affected: int = 0
    entities_affected: int = 0
    processing_time_ms: float = 0
    version: int = 0
    success: bool = True
    error: str = ""


class CDCProcessor:
    """
    CDC incremental processor

    Core design:
      1. Event normalization: convert change events from different sources into the CDCEvent format
      2. Delta calculation: compare before/after and process only actual changes
      3. Incremental processing: only re-parse, vectorize, and graph changed parts
      4. Version tracking: increment the version on each update and support rollback
    """

    def __init__(self) -> None:
        self._version_map: dict[str, int] = {}
        self._event_log: list[CDCEvent] = []
        self._processing_queue: list[CDCEvent] = []

    # ── Event Normalization ──────────────────────────────────

    @staticmethod
    def from_filesystem_event(event_type: str, file_path: str, content_before: str = "", content_after: str = "") -> CDCEvent:
        """Create a CDCEvent from a file system event"""
        op_map = {"created": "INSERT", "modified": "UPDATE", "deleted": "DELETE"}
        return CDCEvent(
            event_id=hashlib.sha256(f"{file_path}:{time.time()}".encode()).hexdigest()[:16],
            source_type="filesystem",
            operation=op_map.get(event_type, "UPDATE"),
            resource_path=file_path,
            before={"content": content_before} if content_before else None,
            after={"content": content_after} if content_after else None,
        )

    @staticmethod
    def from_kafka_message(message: bytes) -> CDCEvent:
        """Create a CDCEvent from a Kafka CDC message (Debezium format)"""
        payload = json.loads(message)
        return CDCEvent(
            event_id=payload.get("id", hashlib.sha256(message).hexdigest()[:16]),
            source_type="database",
            operation=payload.get("op", "UPDATE").upper(),
            resource_path=payload.get("source", {}).get("table", "unknown"),
            before=payload.get("before"),
            after=payload.get("after"),
            timestamp=payload.get("ts_ms", time.time() * 1000) / 1000,
        )

    # ── Diff Computation ─────────────────────────────────────

    @staticmethod
    def compute_diff(before: str, after: str) -> dict[str, Any]:
        """
        Calculate text deltas
        Return statistics and content for added, deleted, and modified lines
        """
        before_lines = before.splitlines() if before else []
        after_lines = after.splitlines() if after else []

        before_set = set(before_lines)
        after_set = set(after_lines)

        added = after_set - before_set
        removed = before_set - after_set

        change_ratio = len(added | removed) / max(len(before_lines) + len(after_lines), 1)

        return {
            "added_lines": list(added),
            "removed_lines": list(removed),
            "added_count": len(added),
            "removed_count": len(removed),
            "change_ratio": round(change_ratio, 4),
            "is_major_change": change_ratio > 0.3,
        }

    # ── Version Management ───────────────────────────────────

    def bump_version(self, resource_path: str) -> int:
        """Increment the resource version number"""
        current = self._version_map.get(resource_path, 0)
        new_version = current + 1
        self._version_map[resource_path] = new_version
        return new_version

    def get_version(self, resource_path: str) -> int:
        return self._version_map.get(resource_path, 0)

    # ── Processing ───────────────────────────────────────────

    async def process_event(self, event: CDCEvent) -> CDCProcessResult:
        """Process a single CDC event"""
        start = time.time()
        result = CDCProcessResult(event=event)

        try:
            version = self.bump_version(event.resource_path)
            result.version = version

            if event.operation == "DELETE":
                result.chunks_affected = -1
                result.entities_affected = -1
            elif event.operation == "INSERT":
                result.chunks_affected = 1
                result.entities_affected = 1
            elif event.operation == "UPDATE":
                if event.before and event.after:
                    diff = self.compute_diff(
                        event.before.get("content", ""),
                        event.after.get("content", ""),
                    )
                    event.diff = diff
                    if diff["is_major_change"]:
                        result.chunks_affected = diff["added_count"]
                    else:
                        result.chunks_affected = max(1, diff["added_count"] // 10)

            self._event_log.append(event)
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.processing_time_ms = (time.time() - start) * 1000
        return result

    async def process_batch(self, events: list[CDCEvent]) -> list[CDCProcessResult]:
        """Process CDC events in a batch"""
        results: list[CDCProcessResult] = []
        for event in events:
            results.append(await self.process_event(event))
        return results

    # ── Kafka Consumer ───────────────────────────────────────

    async def start_kafka_consumer(self, topics: list[str] | None = None) -> None:
        """Start the Kafka CDC consumer loop"""
        from confluent_kafka import Consumer

        if topics is None:
            topics = [settings.kafka_topic_doc_changes]

        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "cdc-processor",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        }
        consumer = Consumer(conf)
        consumer.subscribe(topics)

        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None or msg.error():
                    continue
                event = self.from_kafka_message(msg.value())
                await self.process_event(event)
        finally:
            consumer.close()

    # ── Stats & History ──────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_events_processed": len(self._event_log),
            "tracked_resources": len(self._version_map),
            "queue_size": len(self._processing_queue),
            "versions": dict(self._version_map),
        }

    def get_event_history(self, resource_path: str | None = None, limit: int = 50) -> list[CDCEvent]:
        events = self._event_log
        if resource_path:
            events = [e for e in events if e.resource_path == resource_path]
        return events[-limit:]
