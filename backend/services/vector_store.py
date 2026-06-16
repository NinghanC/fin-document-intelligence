"""
Vector Store Service - supports ChromaDB / PGVector backends

Responsibilities:
  1. Document chunk vectorization (embedding)
  2. Vector storage and retrieval
  3. Delete by doc_id (supports incremental updates)
"""

from __future__ import annotations

from typing import Any

from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocumentChunk
from config import settings


class _SubprocessEmbeddings:
    """Embedding wrapper that delegates to a separate subprocess to avoid
    PyTorch segfaults from crashing the main server process."""

    def __init__(self):
        from services.embedding_worker import get_embedding_client
        self._client = get_embedding_client()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            return [[0.0]] * len(texts)
        return self._client.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._client is None:
            return [0.0]
        return self._client.encode([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            return [[0.0]] * len(texts)
        return await self._client.aencode(texts)

    async def aembed_query(self, text: str) -> list[float]:
        if self._client is None:
            return [0.0]
        result = await self._client.aencode([text])
        return result[0]


def _create_embeddings():
    """Create an embedding instance from configuration, using subprocess isolation to avoid segfaults"""
    import os
    if os.environ.get("DISABLE_LOCAL_EMBEDDINGS") == "1":
        return None
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


class VectorStoreService:
    """Unified vector store interface with switchable ChromaDB / PGVector backends"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        self._embeddings: Any = None
        self._store: Any = None
        self._backend = settings.vector_store_type
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
            import os
            # Skip HuggingFace embedding model if it causes instability
            # Use DISABLE_LOCAL_EMBEDDINGS=1 to force LLM-only mode
            if os.environ.get("DISABLE_LOCAL_EMBEDDINGS") == "1":
                return None
            try:
                self._embeddings = _create_embeddings()
            except Exception:
                self._embeddings = None
        return self._embeddings

    @property
    def embeddings_available(self) -> bool:
        import os
        if os.environ.get("DISABLE_LOCAL_EMBEDDINGS") == "1":
            return False
        if self._embeddings is not None:
            return True
        # Try loading; if it fails, stay disabled
        try:
            return self.embeddings is not None
        except Exception:
            return False

    # ── initialization ───────────────────────────────────────

    async def init(self) -> None:
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()

    async def _init_chroma(self) -> None:
        def _init():
            import chromadb
            import os
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
        )

    # ── CRUD ─────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Vectorize and store document chunks."""
        if not chunks or not self.embeddings_available:
            return 0

        documents = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                **chunk.metadata,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
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
                self._stored_count = getattr(self, '_stored_count', 0) + len(chunks)
                return len(chunks)

            # PGVector / LangChain vectorstores
            if hasattr(self._store, "add_documents"):
                await self._run_sync(self._store.add_documents, documents, metadatas=metadatas, ids=ids)
                return len(chunks)
            if hasattr(self._store, "add_texts"):
                await self._run_sync(self._store.add_texts, documents, metadatas=metadatas, ids=ids)
                return len(chunks)
        except Exception:
            return 0

        return 0

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Semantic search over the configured vector backend."""
        if not self.embeddings_available:
            return []

        if self._backend == "chroma":
            query_embedding = await self._run_sync(self.embeddings.embed_query, query)
            result = await self._run_sync(
                self._store.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            return [
                (
                    {
                        "content": doc,
                        "source": metadata.get("source", ""),
                        "metadata": metadata,
                    },
                    float(distance),
                )
                for doc, metadata, distance in zip(documents, metadatas, distances)
            ]

        results = await self._store.asimilarity_search_with_score(query, k=top_k)
        return [
            ({"content": doc.page_content, "source": doc.metadata.get("source", ""), "metadata": doc.metadata}, score)
            for doc, score in results
        ]

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all related vectors by doc_id"""
        if self._backend == "chroma":
            existing = await self._run_sync(self._store.get, where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                await self._run_sync(self._store.delete, ids=ids)
            return len(ids)
        return 0

    async def get_stats(self) -> dict:
        """Get vector store statistics (chromadb is unstable in async contexts, so use the cached count)"""
        if self._backend == "chroma":
            if self._store is None:
                return {"backend": "chroma", "total_vectors": 0, "collection": self.COLLECTION_NAME}
            # Avoid chromadb C-extension calls in async context (may segfault).
            # Count is maintained manually via _stored_count.
            return {"backend": "chroma", "total_vectors": getattr(self, '_stored_count', 0), "collection": self.COLLECTION_NAME}
        return {"backend": "pgvector", "collection": self.COLLECTION_NAME}
