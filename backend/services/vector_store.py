"""
Vector Store Service - supports ChromaDB / PGVector backends

Responsibilities:
  1. Document chunk vectorization (embedding)
  2. Vector storage and retrieval
  3. Delete by doc_id (supports incremental updates)
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import structlog
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from agents.doc_parser_agent import DocumentChunk
from config import settings
from services.ingestion_registry import ingestion_registry
from utils.model_clients import has_provider_key

logger = structlog.get_logger("finsight.vector_store")


class _SubprocessEmbeddings:
    """Embedding wrapper that delegates to a separate subprocess to avoid
    PyTorch segfaults from crashing the main server process."""

    dimensions = 768

    def __init__(self):
        from services.embedding_worker import get_embedding_client
        self._client = get_embedding_client()
        self._fallback: _HashEmbeddings | None = None
        self._fallback_warned = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            return self._fallback_embeddings().embed_documents(texts)
        return self._client.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._client is None:
            return self._fallback_embeddings().embed_query(text)
        return self._client.encode([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            return self._fallback_embeddings().embed_documents(texts)
        return await self._client.aencode(texts)

    async def aembed_query(self, text: str) -> list[float]:
        if self._client is None:
            return self._fallback_embeddings().embed_query(text)
        result = await self._client.aencode([text])
        return result[0]

    def _fallback_embeddings(self) -> _HashEmbeddings:
        if self._fallback is None:
            if not self._fallback_warned:
                logger.warning("local_embedding_worker_unavailable_using_hash_fallback")
                self._fallback_warned = True
            self._fallback = _HashEmbeddings(dimensions=self.dimensions)
        return self._fallback


class _HashEmbeddings:
    """Deterministic lightweight embeddings for offline demos and tests."""

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        compact = "".join(tokens)
        char_ngrams = [
            compact[index : index + 3]
            for index in range(max(len(compact) - 2, 0))
            if compact[index : index + 3]
        ]
        features = tokens + char_ngrams
        if not features and text:
            features = [text.lower()]
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            for offset in range(0, 16, 4):
                idx = int.from_bytes(digest[offset : offset + 4], "big") % self.dimensions
                sign = 1.0 if digest[(offset + 16) % len(digest)] % 2 == 0 else -1.0
                vector[idx] += sign / 4
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _create_embeddings():
    """Create an embedding instance from configuration, with an offline demo fallback."""
    import os
    provider = settings.embedding_provider.lower()
    if provider == "hash" or not has_provider_key():
        return _HashEmbeddings()
    if provider == "local" and os.environ.get("DISABLE_LOCAL_EMBEDDINGS") != "1":
        return _SubprocessEmbeddings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(settings.openai_api_key),
        base_url=settings.openai_base_url,
    )


class VectorStoreService:
    """Unified vector store interface with switchable ChromaDB / PGVector backends"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        self._embeddings: Any = None
        self._store: Any = None
        self._backend = settings.vector_store_type
        self._pg_engine: Any = None
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def _run_sync(self, fn, *args, **kwargs):
        """Run chromadb operations in thread pool to avoid async segfaults."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    @property
    def embeddings(self):
        if self._embeddings is None:
            # DISABLE_LOCAL_EMBEDDINGS only skips subprocess/local model loading.
            # Hash embeddings keep the public demo searchable without external keys.
            try:
                self._embeddings = _create_embeddings()
            except Exception as exc:
                logger.warning("embedding_provider_failed_using_hash", error=str(exc))
                self._embeddings = _HashEmbeddings()
        return self._embeddings

    @property
    def embeddings_available(self) -> bool:
        if self._embeddings is not None:
            return True
        # Try loading; if it fails, stay disabled
        try:
            return self.embeddings is not None
        except Exception as exc:
            logger.warning("embeddings_availability_check_failed", error=str(exc))
            return False

    # initialization
    async def init(self) -> None:
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()

    async def _init_chroma(self) -> None:
        def _init():
            import os

            import chromadb

            if settings.chroma_mode == "http":
                client = chromadb.HttpClient(
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                )
            else:
                persist_dir = os.path.join(settings.upload_dir, "..", "chroma_data")
                os.makedirs(persist_dir, exist_ok=True)
                client = chromadb.PersistentClient(path=persist_dir)

            return client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        self._store = await self._run_sync(_init)

    async def _init_pgvector(self) -> None:
        from langchain_community.vectorstores import PGVector
        self._store = PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            use_jsonb=True,
            create_extension=True,
        )

    # CRUD
    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Vectorize and store document chunks."""
        if not chunks or self._store is None or not self.embeddings_available:
            return 0

        documents = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                **chunk.metadata,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "doc_type": chunk.doc_type.value,
                "source": chunk.metadata.get("source", ""),
            }
            for chunk in chunks
        ]

        try:
            if self._backend == "chroma":
                embeddings = await self._run_sync(self.embeddings.embed_documents, documents)
                await self._run_sync(
                    self._store.add,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings,
                )
                return len(chunks)

            if self._backend == "pgvector":
                await self._run_sync(
                    self._store.add_texts,
                    documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                return len(chunks)
        except Exception as exc:
            logger.warning("vector_add_chunks_failed", backend=self._backend, error=str(exc))
            return 0

        return 0

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Semantic search over the configured vector backend."""
        if self._store is None or not self.embeddings_available:
            return []

        if self._backend == "chroma":
            query_embedding = await self._run_sync(self.embeddings.embed_query, query)
            result = await self._run_sync(
                self._store.query,
                query_embeddings=[query_embedding],
                n_results=max(top_k * 8, top_k),
                include=["documents", "metadatas", "distances"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            candidates = [
                self._score_result(
                    query,
                    doc,
                    metadata,
                    max(0.0, 1.0 - float(distance)),
                )
                for doc, metadata, distance in zip(documents, metadatas, distances, strict=False)
                if ingestion_registry.is_committed(str(metadata.get("doc_id", "")))
            ]
            candidates.extend(await self._chroma_lexical_candidates(query, top_k=max(top_k * 4, top_k)))
            return self._merge_ranked_candidates(candidates, top_k)

        results = await self._run_sync(self._store.similarity_search_with_score, query, k=top_k)
        candidates = [
            self._score_result(query, doc.page_content, doc.metadata, score)
            for doc, score in results
            if ingestion_registry.is_committed(str(doc.metadata.get("doc_id", "")))
        ]
        return self._merge_ranked_candidates(candidates, top_k)

    async def _chroma_lexical_candidates(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """Scan Chroma documents for exact keyword matches.

        This keeps offline/hash-embedding demos usable for financial metric
        questions where exact terms matter more than approximate semantics.
        """
        scan_limit = max(settings.chroma_lexical_scan_limit, 0)
        if scan_limit == 0:
            return []
        try:
            result = await self._run_sync(self._store.get, include=["documents", "metadatas"], limit=scan_limit)
        except Exception as exc:
            logger.warning("chroma_lexical_candidate_scan_failed", error=str(exc))
            return []
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        candidates = [
            self._score_result(query, doc, metadata, vector_score=0.0)
            for doc, metadata in zip(documents, metadatas, strict=False)
            if ingestion_registry.is_committed(str(metadata.get("doc_id", "")))
            and self._lexical_score(query, doc) > 0
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _merge_ranked_candidates(candidates: list[tuple[dict, float]], top_k: int) -> list[tuple[dict, float]]:
        merged: dict[str, tuple[dict, float]] = {}
        for doc, score in candidates:
            key = str(doc["metadata"].get("chunk_id") or doc.get("content", "")[:120])
            if key not in merged or score > merged[key][1]:
                merged[key] = (doc, score)
        ranked = list(merged.values())
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    @classmethod
    def _score_result(
        cls,
        query: str,
        content: str,
        metadata: dict[str, Any],
        vector_score: float,
    ) -> tuple[dict, float]:
        content_lexical_score = cls._lexical_score(query, content)
        metadata_score = cls._metadata_score(query, metadata)
        score = min(
            1.0,
            max(
                (vector_score * 0.55) + (content_lexical_score * 0.35) + (metadata_score * 0.10),
                content_lexical_score * 0.9,
                metadata_score * 0.25,
            ),
        )
        return (
            {
                "content": content,
                "source": metadata.get("source", ""),
                "metadata": {
                    **metadata,
                    "vector_score": vector_score,
                    "lexical_score": content_lexical_score,
                    "metadata_score": metadata_score,
                },
            },
            round(score, 6),
        )

    @classmethod
    def _metadata_score(cls, query: str, metadata: dict[str, Any]) -> float:
        metadata_text = " ".join(
            str(metadata.get(key, ""))
            for key in ("source", "doc_id", "doc_type")
        )
        return cls._lexical_score(query, metadata_text)

    @classmethod
    def _lexical_score(cls, query: str, content: str) -> float:
        query_tokens = cls._query_tokens(query)
        if not query_tokens:
            return 0.0
        content_tokens = set(re.findall(r"[a-zA-Z0-9]+", content.lower()))
        return len(query_tokens & content_tokens) / len(query_tokens)

    @staticmethod
    def _query_tokens(query: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
            if len(token) >= 3 and token not in {"and", "for", "the", "their", "what", "which", "did"}
        }

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all related vectors by doc_id"""
        if self._store is None:
            return 0
        if self._backend == "chroma":
            existing = await self._run_sync(self._store.get, where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                await self._run_sync(self._store.delete, ids=ids)
            return len(ids)
        if self._backend == "pgvector":
            return await self._delete_pgvector_by_doc_id(doc_id)
        return 0

    async def get_stats(self) -> dict:
        """Get vector store statistics from the active backend."""
        if self._backend == "chroma":
            if self._store is None:
                return {"backend": "chroma", "total_vectors": 0, "collection": self.COLLECTION_NAME}
            try:
                total_vectors = await self._run_sync(self._store.count)
            except Exception as exc:
                logger.warning("vector_stats_failed", backend="chroma", error=str(exc))
                total_vectors = 0
            return {"backend": "chroma", "total_vectors": total_vectors, "collection": self.COLLECTION_NAME}
        return {
            "backend": "pgvector",
            "collection": self.COLLECTION_NAME,
            "total_vectors": await self._count_pgvector_vectors(),
        }

    async def _delete_pgvector_by_doc_id(self, doc_id: str) -> int:
        """Delete PGVector rows whose metadata belongs to the given document."""
        def _delete() -> int:
            from sqlalchemy import text

            with self._pgvector_engine().begin() as conn:
                result = conn.execute(
                    text(
                        """
                        DELETE FROM langchain_pg_embedding AS e
                        USING langchain_pg_collection AS c
                        WHERE e.collection_id = c.uuid
                          AND c.name = :collection
                          AND e.cmetadata ->> 'doc_id' = :doc_id
                        """
                    ),
                    {"collection": self.COLLECTION_NAME, "doc_id": doc_id},
                )
                return result.rowcount or 0

        try:
            return await self._run_sync(_delete)
        except Exception as exc:
            logger.warning("pgvector_delete_by_doc_id_failed", doc_id=doc_id, error=str(exc))
            return 0

    async def _count_pgvector_vectors(self) -> int:
        """Count vectors stored in the configured PGVector collection."""
        def _count() -> int:
            from sqlalchemy import text

            with self._pgvector_engine().begin() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) AS cnt
                        FROM langchain_pg_embedding AS e
                        JOIN langchain_pg_collection AS c
                          ON e.collection_id = c.uuid
                        WHERE c.name = :collection
                        """
                    ),
                    {"collection": self.COLLECTION_NAME},
                )
                return int(result.scalar() or 0)

        try:
            return await self._run_sync(_count)
        except Exception as exc:
            logger.warning("pgvector_count_failed", error=str(exc))
            return 0

    def _pgvector_engine(self) -> Any:
        if self._pg_engine is None:
            from sqlalchemy import create_engine

            self._pg_engine = create_engine(settings.pgvector_dsn, pool_pre_ping=True)
        return self._pg_engine
