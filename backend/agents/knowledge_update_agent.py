"""
Knowledge Update Agent - listens for document changes and incrementally updates the vector store and knowledge graph

Core capabilities:
  1. File system watching (Watchdog) / Kafka CDC consumption
  2. Delta comparison: compare old and new documents and process only changed parts
  3. Incremental vectorization and graph updates
  4. Version management: knowledge nodes include timestamps and version numbers
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import settings


class ChangeType(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class DocumentChange:
    file_path: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_hash: str = ""
    new_hash: str = ""
    diff_chunks: list[str] = field(default_factory=list)


@dataclass
class UpdateResult:
    change: DocumentChange
    vectors_added: int = 0
    vectors_deleted: int = 0
    entities_added: int = 0
    entities_updated: int = 0
    relations_added: int = 0
    success: bool = True
    error: str = ""
    processing_time_ms: float = 0


class KnowledgeUpdateAgent:
    """
    Knowledge Update Agent

    Supports two modes:
      1. File watching mode (Watchdog): monitors local file system changes
      2. CDC mode (Kafka): consumes change events from the message queue

    Workflow:
      detect_change -> diff_analysis -> incremental_parse -> update_vector_store -> update_knowledge_graph -> log
    """

    def __init__(
        self,
        doc_parser: Any = None,
        knowledge_extractor: Any = None,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.doc_parser = doc_parser
        self.knowledge_extractor = knowledge_extractor
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self._file_hashes: dict[str, str] = {}
        self._version_counter: dict[str, int] = {}
        self._observer: Any = None
        self._watch_thread: Any = None

    # public API
    async def process_change(self, change: DocumentChange) -> UpdateResult:
        """Process a single document change"""
        start = time.time()
        result = UpdateResult(change=change)

        try:
            if change.change_type == ChangeType.DELETED:
                await self._handle_delete(change, result)
            elif change.change_type == ChangeType.CREATED:
                await self._handle_create(change, result)
            elif change.change_type == ChangeType.MODIFIED:
                await self._handle_modify(change, result)
        except Exception as e:
            result.success = False
            result.error = str(e)

        result.processing_time_ms = (time.time() - start) * 1000
        return result

    async def process_batch(self, changes: list[DocumentChange]) -> list[UpdateResult]:
        """Process document changes in a batch"""
        results: list[UpdateResult] = []
        for change in changes:
            results.append(await self.process_change(change))
        return results

    def detect_changes(self, file_paths: list[str]) -> list[DocumentChange]:
        """Scan a file list and detect changes"""
        changes: list[DocumentChange] = []
        current_files = set(file_paths)

        for fp in current_files:
            new_hash = self._compute_hash(fp)
            old_hash = self._file_hashes.get(fp, "")

            if not old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.CREATED,
                    new_hash=new_hash,
                ))
            elif new_hash != old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))
            self._file_hashes[fp] = new_hash

        for fp in set(self._file_hashes) - current_files:
            changes.append(DocumentChange(
                file_path=fp,
                change_type=ChangeType.DELETED,
                old_hash=self._file_hashes[fp],
            ))
            del self._file_hashes[fp]

        return changes

    # watchdog mode
    def start_watching(self, directory: str, loop: Any = None) -> Any:
        """Start file system watching (non-blocking, runs in a separate thread)"""
        import asyncio
        import threading

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        agent = self

        def _submit(change: DocumentChange) -> None:
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(agent.process_change(change), loop)
                return
            asyncio.run(agent.process_change(change))

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.CREATED)
                    _submit(change)

            def on_modified(self, event):
                if not event.is_directory:
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.MODIFIED)
                    _submit(change)

            def on_deleted(self, event):
                if not event.is_directory:
                    change = DocumentChange(file_path=event.src_path, change_type=ChangeType.DELETED)
                    _submit(change)

        observer = Observer()
        observer.schedule(_Handler(), directory, recursive=True)
        self._observer = observer

        def _run():
            observer.start()
            try:
                while observer.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._watch_thread = t
        return observer

    def stop_watching(self) -> None:
        """Stop the watchdog observer and join its background thread."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._watch_thread is not None and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=5)
        self._watch_thread = None

    # kafka CDC mode
    async def start_kafka_consumer(self) -> None:
        """Start the Kafka CDC consumer"""
        import json

        from confluent_kafka import Consumer

        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": "knowledge-update-agent",
            "auto.offset.reset": "latest",
        }
        consumer = Consumer(conf)
        consumer.subscribe([settings.kafka_topic_doc_changes])

        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue
                value = msg.value()
                if value is None:
                    continue
                payload = json.loads(value.decode("utf-8"))
                change = DocumentChange(
                    file_path=payload["file_path"],
                    change_type=ChangeType(payload["change_type"]),
                    old_hash=payload.get("old_hash", ""),
                    new_hash=payload.get("new_hash", ""),
                )
                await self.process_change(change)
        finally:
            consumer.close()

    # internal handlers
    async def _handle_create(self, change: DocumentChange, result: UpdateResult) -> None:
        if not self.doc_parser:
            return
        chunks = await self.doc_parser.parse(change.file_path)

        if self.vector_store:
            await self.vector_store.add_chunks(chunks)
            result.vectors_added = len(chunks)

        if self.knowledge_extractor and self.knowledge_graph:
            extractions = await self.knowledge_extractor.extract(chunks)
            for ext in extractions:
                for ent in ext.entities:
                    version = await self._next_version(ent.name)
                    await self.knowledge_graph.upsert_entity(ent, version=version, source=ext.source_chunk_id)
                    result.entities_added += 1
                for rel in ext.relations:
                    await self.knowledge_graph.add_relation(rel, source=ext.source_chunk_id)
                    result.relations_added += 1

    async def _handle_modify(self, change: DocumentChange, result: UpdateResult) -> None:
        doc_id = hashlib.sha256(self._canonical_path(change.file_path).encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted

        await self._handle_create(change, result)

    async def _handle_delete(self, change: DocumentChange, result: UpdateResult) -> None:
        doc_id = hashlib.sha256(self._canonical_path(change.file_path).encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted

        if self.knowledge_graph:
            await self.knowledge_graph.delete_by_source(change.file_path)

    # utilities
    @staticmethod
    def _compute_hash(file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _canonical_path(file_path: str) -> str:
        import os
        return os.path.abspath(file_path)

    async def _next_version(self, entity_name: str) -> int:
        current = self._version_counter.get(entity_name, 0)
        if self.knowledge_graph:
            existing = await self.knowledge_graph.get_entity(entity_name)
            if existing:
                entity_data = existing.get("e", existing)
                current = max(current, int(entity_data.get("version", 0) or 0))
        ver = current + 1
        self._version_counter[entity_name] = ver
        return ver
