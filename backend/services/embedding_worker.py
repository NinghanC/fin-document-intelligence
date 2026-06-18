"""
Embedding worker subprocess.

This isolates sentence-transformers in a separate process so model-runtime
errors do not crash the main API server.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import queue
from contextlib import suppress
from typing import Any

_MODEL_NAME = os.environ.get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_DEVICE = "cpu"
_SHUTDOWN_TIMEOUT = 30
logger = logging.getLogger("finsight.embedding_worker")


def _worker_process(request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
    """Loads the embedding model once and processes encode requests."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
        model.encode("warmup", show_progress_bar=False)
        response_queue.put({"status": "ready", "model": _MODEL_NAME})
        print(f"[embedding_worker] Model loaded: {_MODEL_NAME}", flush=True)
    except Exception as e:
        response_queue.put({"status": "error", "message": str(e)})
        return

    while True:
        try:
            msg = request_queue.get(timeout=1)
        except queue.Empty:
            continue

        if msg is None:
            break

        msg_id = msg["id"]
        texts = msg["texts"]
        try:
            vectors = model.encode(texts, show_progress_bar=False).tolist()
            response_queue.put({"id": msg_id, "embeddings": vectors})
        except Exception as e:
            response_queue.put({"id": msg_id, "error": str(e)})


class EmbeddingClient:
    """Client that talks to the embedding worker process."""

    def __init__(self):
        self._request_queue: multiprocessing.Queue | None = None
        self._response_queue: multiprocessing.Queue | None = None
        self._process: Any = None
        self._counter = 0

    def start(self):
        if self._process is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_process,
            args=(self._request_queue, self._response_queue),
            daemon=True,
        )
        assert self._process is not None
        self._process.start()
        # Wait for model to load
        try:
            result = self._response_queue.get(timeout=60)
            if result.get("status") == "error":
                raise RuntimeError(f"Worker failed to load model: {result.get('message', '')}")
            if result.get("status") != "ready":
                raise RuntimeError(f"Unexpected embedding worker startup response: {result}")
        except queue.Empty as exc:
            raise RuntimeError("Embedding worker timed out during startup") from exc

    def stop(self):
        if self._process is None:
            return
        assert self._request_queue is not None
        with suppress(Exception):
            self._request_queue.put(None)
        self._process.join(timeout=_SHUTDOWN_TIMEOUT)
        if self._process.is_alive():
            self._process.terminate()
        self._process = None
        self._request_queue = None
        self._response_queue = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("Embedding worker is not running")
        assert self._request_queue is not None
        assert self._response_queue is not None
        self._counter += 1
        msg_id = f"enc_{self._counter}"
        self._request_queue.put({"id": msg_id, "texts": texts})
        try:
            response = self._response_queue.get(timeout=300)
            if response.get("id") != msg_id:
                raise RuntimeError(f"Unexpected embedding worker response id: {response}")
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            return response["embeddings"]
        except queue.Empty as exc:
            raise RuntimeError("Embedding worker timed out") from exc

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, texts)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()


# Singleton
_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient | None:
    global _embedding_client
    if os.environ.get("DISABLE_LOCAL_EMBEDDINGS") == "1":
        return None
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
        try:
            _embedding_client.start()
        except Exception:
            logger.warning("embedding_worker_start_failed", exc_info=True)
            _embedding_client = None
    return _embedding_client
