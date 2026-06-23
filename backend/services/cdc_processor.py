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
    async def start_kafka_consumer(self, topics: list[str] | None = None, max_retries: int = 3) -> None:
        """Start the Kafka CDC consumer loop.

        Offsets are committed manually and only after an event is applied
        successfully. The confluent-kafka default (``enable.auto.commit=True``)
        advances offsets on a timer regardless of whether processing succeeded, so a
        crash or downstream outage between poll and apply silently skips messages
        (data loss). Here a failing event is retried up to ``max_retries`` times
        before being logged as a dropped/dead-letter event and skipped, so a poison
        message cannot block the partition forever while a healthy event is never
        silently lost on restart.
        """
        from confluent_kafka import Consumer, KafkaError, KafkaException

        if topics is None:
            topics = [settings.kafka_topic_doc_changes]

        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "cdc-processor",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
        consumer = Consumer(conf)
        consumer.subscribe(topics)
        retry_counts: dict[tuple[str, int, int], int] = {}

        try:
            while True:
                msg = await asyncio.to_thread(consumer.poll, 1.0)
                if msg is None:
                    continue
                error = msg.error()
                if error is not None:
                    # _PARTITION_EOF is benign (caught up to the high-water mark);
                    # anything else is logged, and fatal errors stop the consumer
                    # rather than spinning invisibly on a broken connection.
                    if error.code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("cdc_kafka_consumer_error", code=error.code(), error=str(error))
                    if error.fatal():
                        raise KafkaException(error)
                    continue
                await self._consume_message(consumer, msg, retry_counts, max_retries)
        finally:
            consumer.close()

    async def _consume_message(
        self,
        consumer: Any,
        msg: Any,
        retry_counts: dict[tuple[str, int, int], int],
        max_retries: int,
    ) -> str:
        """Apply one polled Kafka message and manage its offset commit.

        Returns the action taken ("empty" | "unparseable" | "committed" | "retry" |
        "dropped"), which is the unit the consumer tests assert against.
        """
        from confluent_kafka import TopicPartition

        key = (msg.topic(), msg.partition(), msg.offset())
        value = msg.value()
        if value is None:
            await asyncio.to_thread(consumer.commit, msg, asynchronous=False)
            retry_counts.pop(key, None)
            return "empty"

        try:
            event = self.from_kafka_message(value)
        except Exception as exc:
            # A message that cannot be parsed will never succeed; commit past it so
            # it does not block the partition, but surface it loudly.
            logger.error("cdc_message_unparseable_dropped", offset=msg.offset(), error=str(exc))
            await asyncio.to_thread(consumer.commit, msg, asynchronous=False)
            retry_counts.pop(key, None)
            return "unparseable"

        result = await self.process_event(event)
        if result.success:
            await asyncio.to_thread(consumer.commit, msg, asynchronous=False)
            retry_counts.pop(key, None)
            return "committed"

        attempts = retry_counts.get(key, 0) + 1
        if attempts <= max_retries:
            retry_counts[key] = attempts
            logger.warning(
                "cdc_event_apply_retry",
                event_id=event.event_id,
                attempt=attempts,
                max_retries=max_retries,
                error=result.error,
            )
            # Replay the same offset on the next poll instead of committing.
            await asyncio.to_thread(consumer.seek, TopicPartition(msg.topic(), msg.partition(), msg.offset()))
            await asyncio.sleep(min(0.5 * attempts, 2.0))
            return "retry"

        logger.error(
            "cdc_event_dropped_after_retries",
            event_id=event.event_id,
            attempts=attempts,
            error=result.error,
        )
        await asyncio.to_thread(consumer.commit, msg, asynchronous=False)
        retry_counts.pop(key, None)
        return "dropped"

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
            "C": ChangeType.CREATED,
            "R": ChangeType.CREATED,
            "U": ChangeType.MODIFIED,
            "D": ChangeType.DELETED,
        }
        operation = event.operation.upper()
        if operation not in operation_map:
            raise ValueError(f"Unsupported CDC operation: {event.operation}")
        change_type = operation_map[operation]
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
