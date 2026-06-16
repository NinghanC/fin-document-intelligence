import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
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
    assert results[0][1] == pytest.approx(0.123)
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
    monkeypatch.setattr(service, "_embeddings", DummyEmbeddings())

    deleted = await service.delete_by_doc_id("doc-1")
    assert deleted == 2
    assert service._store.deleted == ["doc-1#chunk-0", "doc-1#chunk-1"]
