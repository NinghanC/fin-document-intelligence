"""Tests for the Kafka CDC consumer commit/retry semantics.

These exercise the per-message handling (`_consume_message`) directly with a fake
consumer, so they run without a broker. The key property under test: offsets are
committed only after an event is applied successfully (no silent data loss), while
poison messages are still skipped so they cannot block a partition forever.
"""

from __future__ import annotations

import pytest

from services.cdc_processor import CDCProcessor


class FakeConsumer:
    def __init__(self) -> None:
        self.committed: list[object] = []
        self.sought: list[object] = []

    def commit(self, message=None, asynchronous=True):
        self.committed.append(message)

    def seek(self, partition):
        self.sought.append(partition)


class FakeMessage:
    def __init__(self, value: bytes | None, topic: str = "doc-changes", partition: int = 0, offset: int = 0) -> None:
        self._value = value
        self._topic = topic
        self._partition = partition
        self._offset = offset

    def value(self):
        return self._value

    def topic(self):
        return self._topic

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def error(self):
        return None


_VALID = b'{"op": "u", "source": {"table": "docs"}}'


@pytest.mark.asyncio
async def test_commits_only_after_successful_apply():
    processor = CDCProcessor()

    async def handler(_change):
        return None  # no `success=False` attribute -> treated as success

    processor.set_update_handler(handler)
    consumer = FakeConsumer()
    msg = FakeMessage(_VALID, offset=5)

    action = await processor._consume_message(consumer, msg, {}, max_retries=3)

    assert action == "committed"
    assert consumer.committed == [msg]
    assert consumer.sought == []


@pytest.mark.asyncio
async def test_failing_event_is_retried_then_dropped_not_silently_committed(monkeypatch):
    monkeypatch.setattr("services.cdc_processor.asyncio.sleep", _no_sleep)
    processor = CDCProcessor()

    async def handler(_change):
        raise RuntimeError("downstream unavailable")

    processor.set_update_handler(handler)
    consumer = FakeConsumer()
    msg = FakeMessage(_VALID, offset=7)
    retry_counts: dict = {}

    for _attempt in range(1, 4):
        action = await processor._consume_message(consumer, msg, retry_counts, max_retries=3)
        assert action == "retry"
    # never committed while still retrying -> message survives a restart
    assert consumer.committed == []
    assert len(consumer.sought) == 3

    # exhausting retries drops the poison event (loudly) and commits past it
    action = await processor._consume_message(consumer, msg, retry_counts, max_retries=3)
    assert action == "dropped"
    assert consumer.committed == [msg]


@pytest.mark.asyncio
async def test_unparseable_message_is_skipped_not_fatal():
    processor = CDCProcessor()
    consumer = FakeConsumer()
    msg = FakeMessage(b"not-json", offset=2)

    action = await processor._consume_message(consumer, msg, {}, max_retries=3)

    assert action == "unparseable"
    assert consumer.committed == [msg]


async def _no_sleep(_seconds):
    return None
