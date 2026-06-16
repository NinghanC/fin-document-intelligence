import pytest
import sys
import types
from langchain_core.documents import Document

from agents.doc_parser_agent import DocType, DocumentChunk
import services.vector_store as vector_store_module
from services.vector_store import VectorStoreService


class DummyEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]


class DummyChromaStore:
    def __init__(self) -> None:
        self.added: list[dict] = []

    def add(self, *, documents: list[str], metadatas: list[dict], ids: list[str], embeddings: list[list[float]]) -> None:
        self.added.append({
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
            "embeddings": embeddings,
        })

    def query(self, query_embeddings: list[list[float]], n_results: int, include: list[str]) -> dict:
        if not self.added:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        documents = self.added[-1]["documents"]
        metadatas = self.added[-1]["metadatas"]
        distances = [0.123 for _ in documents]
        return {"documents": [documents], "metadatas": [metadatas], "distances": [distances]}


@pytest.mark.asyncio
async def test_vector_store_add_and_search(monkeypatch):
    service = VectorStoreService()
    service._backend = "chroma"
    service._store = DummyChromaStore()
    monkeypatch.setattr(service, "_embeddings", DummyEmbeddings())

    chunks = [
        DocumentChunk(
            content="Hello world",
            doc_id="doc-1",
            chunk_index=0,
            doc_type=DocType.TEXT,
            metadata={"source": "test"},
        ),
        DocumentChunk(
            content="Goodbye world",
            doc_id="doc-1",
            chunk_index=1,
            doc_type=DocType.TEXT,
            metadata={"source": "test"},
        ),
    ]

    count = await service.add_chunks(chunks)
    assert count == 2
    assert service._stored_count == 2

    results = await service.search("Hello")
    assert len(results) == 2
    assert results[0][0]["content"] == "Hello world"
    assert results[0][0]["source"] == "test"
    assert results[0][1] == pytest.approx(0.877)
    assert results[0][0]["metadata"]["doc_id"] == "doc-1"
    assert results[1][0]["metadata"]["chunk_id"] == "doc-1#chunk-1"


class DummyChromaStoreWithDelete(DummyChromaStore):
    def delete(self, ids: list[str]) -> None:
        self.deleted = ids

    def get(self, where: dict[str, str], include: list[str]) -> dict:
        if where.get("doc_id") == "doc-1":
            return {"ids": ["doc-1#chunk-0", "doc-1#chunk-1"]}
        return {"ids": []}


@pytest.mark.asyncio
async def test_vector_store_delete_by_doc_id(monkeypatch):
    service = VectorStoreService()
    service._backend = "chroma"
    service._store = DummyChromaStoreWithDelete()
    service._stored_count = 2
    monkeypatch.setattr(service, "_embeddings", DummyEmbeddings())

    deleted = await service.delete_by_doc_id("doc-1")
    assert deleted == 2
    assert service._store.deleted == ["doc-1#chunk-0", "doc-1#chunk-1"]
    assert service._stored_count == 0


@pytest.mark.asyncio
async def test_chroma_http_mode_uses_configured_host_and_port(monkeypatch):
    calls: dict[str, object] = {}

    class DummyClient:
        def __init__(self, host: str, port: int) -> None:
            calls["host"] = host
            calls["port"] = port

        def get_or_create_collection(self, name: str, metadata: dict):
            calls["collection"] = name
            calls["metadata"] = metadata
            return "collection"

    fake_chromadb = types.SimpleNamespace(HttpClient=DummyClient)
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setattr(vector_store_module.settings, "chroma_mode", "http")
    monkeypatch.setattr(vector_store_module.settings, "chroma_host", "chromadb")
    monkeypatch.setattr(vector_store_module.settings, "chroma_port", 8000)

    service = VectorStoreService()
    await service._init_chroma()

    assert service._store == "collection"
    assert calls["host"] == "chromadb"
    assert calls["port"] == 8000
    assert calls["collection"] == service.COLLECTION_NAME


class DummyPGVectorStore:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self.ids: list[str] = []

    def add_texts(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> list[str]:
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        return ids

    def similarity_search_with_score(self, query: str, k: int = 4):
        return [
            (
                Document(
                    page_content=text,
                    metadata=metadata,
                ),
                0.42,
            )
            for text, metadata in zip(self.texts[:k], self.metadatas[:k])
        ]


@pytest.mark.asyncio
async def test_pgvector_add_and_search(monkeypatch):
    service = VectorStoreService()
    service._backend = "pgvector"
    service._store = DummyPGVectorStore()
    monkeypatch.setattr(service, "_embeddings", DummyEmbeddings())

    chunks = [
        DocumentChunk(
            content="Global Income Fund mentions liquidity constraints.",
            doc_id="fund-report",
            chunk_index=0,
            doc_type=DocType.PDF,
            metadata={"source": "Q4_global_income_fund_risk_report.pdf"},
        )
    ]

    count = await service.add_chunks(chunks)
    assert count == 1
    assert service._store.ids == ["fund-report#chunk-0"]
    assert service._store.metadatas[0]["doc_id"] == "fund-report"
    assert service._store.metadatas[0]["chunk_id"] == "fund-report#chunk-0"

    results = await service.search("liquidity constraints")
    assert len(results) == 1
    assert results[0][0]["content"] == "Global Income Fund mentions liquidity constraints."
    assert results[0][0]["source"] == "Q4_global_income_fund_risk_report.pdf"
    assert results[0][1] == pytest.approx(0.42)
