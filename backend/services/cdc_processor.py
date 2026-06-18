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
              v
          Version management (each knowledge node has a version and timestamp)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import structlog

from config import settings

logger = structlog.get_logger("finsight.cdc")


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
    update_result: dict[str, Any] | None = None


class CDCProcessor:
    """
    CDC incremental processor

    Core design:
      1. Event normalization: convert change events from different sources into the CDCEvent format
      2. Delta calculation: compare before/after and process only actual changes
      3. Incremental processing: only re-parse, vectorize, and graph changed parts
      4. Version tracking: increment the version on each update and support rollback
    """

    def __init__(self, update_handler: Callable[[Any], Awaitable[Any]] | None = None) -> None:
        self._version_map: dict[str, int] = {}
        self._event_log: list[CDCEvent] = []
        self._processing_queue: list[CDCEvent] = []
        self._update_handler = update_handler

    def set_update_handler(self, update_handler: Callable[[Any], Awaitable[Any]] | None) -> None:
        self._update_handler = update_handler

    # Event Normalization
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

    # Diff Computation
    @staticmethod
    def compute_diff(before: str, after: str) -> dict[str, Any]:
        """
        Calculate text deltas
        Return statistics and content for added, deleted, and modified lines
        """
        before_lines = before.splitlines() if before else []
        after_lines = after.splitlines() if after else []

        added: list[str] = []
        removed: list[str] = []
        operations: list[dict[str, Any]] = []
        matcher = SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            removed_lines = before_lines[i1:i2]
            added_lines = after_lines[j1:j2]
            removed.extend(removed_lines)
            added.extend(added_lines)
            operations.append({
                "op": tag,
                "before_start": i1,
                "before_end": i2,
                "after_start": j1,
                "after_end": j2,
                "removed": removed_lines,
                "added": added_lines,
            })

        change_ratio = (len(added) + len(removed)) / max(len(before_lines) + len(after_lines), 1)

        return {
            "added_lines": added,
            "removed_lines": removed,
            "added_count": len(added),
            "removed_count": len(removed),
            "operations": operations,
            "change_ratio": round(change_ratio, 4),
            "is_major_change": change_ratio > 0.3,
        }

    # Version Management
    def bump_version(self, resource_path: str) -> int:
        """Increment the resource version number"""
        return self._commit_version(resource_path)

    def get_version(self, resource_path: str) -> int:
        return self._version_map.get(resource_path, 0)

    # Processing
    async def process_event(self, event: CDCEvent) -> CDCProcessResult:
        """Process a single CDC event through the configured update pipeline."""
        start = time.time()
        result = CDCProcessResult(event=event)
        self._processing_queue.append(event)

        try:
            if event.operation == "UPDATE" and event.before and event.after:
                event.diff = self.compute_diff(
                    event.before.get("content", ""),
                    event.after.get("content", ""),
                )

            update_result = await self._apply_update(event)
            result.update_result = self._serialize_update_result(update_result)
            result.chunks_affected = getattr(update_result, "vectors_added", 0) - getattr(
                update_result, "vectors_deleted", 0
            )
            result.entities_affected = getattr(update_result, "entities_added", 0) + getattr(
                update_result, "entities_updated", 0
            )
            if not getattr(update_result, "success", True):
                result.success = False
                result.error = getattr(update_result, "error", "")
            else:
                result.version = self._commit_version(event.resource_path)

            self._event_log.append(event)
        except Exception as exc:
            logger.warning("cdc_event_processing_failed", event_id=event.event_id, error=str(exc))
            result.success = False
            result.error = str(exc)
            self._event_log.append(event)
        finally:
            if event in self._processing_queue:
                self._processing_queue.remove(event)

        result.processing_time_ms = (time.time() - start) * 1000
        return result

    def _commit_version(self, resource_path: str) -> int:
        current = self._version_map.get(resource_path, 0)
        new_version = current + 1
        self._version_map[resource_path] = new_version
        return new_version

    async def process_batch(self, events: list[CDCEvent]) -> list[CDCProcessResult]:
        """Process CDC events in a batch"""
        results: list[CDCProcessResult] = []
        for event in events:
            results.append(await self.process_event(event))
        return results

    # Kafka Consumer
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
                msg = await asyncio.to_thread(consumer.poll, 1.0)
                if msg is None or msg.error():
                    continue
                value = msg.value()
                if value is None:
                    continue
                event = self.from_kafka_message(value)
                await self.process_event(event)
        finally:
            consumer.close()

    async def _apply_update(self, event: CDCEvent) -> Any | None:
        if self._update_handler is None:
            raise RuntimeError("CDC update handler is not configured")
        from agents.knowledge_update_agent import ChangeType, DocumentChange

        operation_map = {
            "INSERT": ChangeType.CREATED,
            "CREATE": ChangeType.CREATED,
            "UPDATE": ChangeType.MODIFIED,
            "MODIFY": ChangeType.MODIFIED,
            "DELETE": ChangeType.DELETED,
        }
        change_type = operation_map.get(event.operation.upper(), ChangeType.MODIFIED)
        change = DocumentChange(file_path=event.resource_path, change_type=change_type)
        return await self._update_handler(change)

    @staticmethod
    def _serialize_update_result(update_result: Any) -> dict[str, Any]:
        change = getattr(update_result, "change", None)
        return {
            "file_path": getattr(change, "file_path", ""),
            "change_type": getattr(getattr(change, "change_type", None), "value", ""),
            "vectors_added": getattr(update_result, "vectors_added", 0),
            "vectors_deleted": getattr(update_result, "vectors_deleted", 0),
            "entities_added": getattr(update_result, "entities_added", 0),
            "entities_updated": getattr(update_result, "entities_updated", 0),
            "relations_added": getattr(update_result, "relations_added", 0),
            "success": getattr(update_result, "success", True),
            "error": getattr(update_result, "error", ""),
        }

    # Stats & History
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
