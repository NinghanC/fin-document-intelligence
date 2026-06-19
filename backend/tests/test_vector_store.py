import sys
import types

import pytest
from langchain_core.documents import Document

import services.vector_store as vector_store_module
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

    def get(self, include: list[str], limit: int | None = None) -> dict:
        if not self.added:
            return {"documents": [], "metadatas": []}
        return {
            "documents": self.added[-1]["documents"][:limit],
            "metadatas": self.added[-1]["metadatas"][:limit],
        }

    def count(self) -> int:
        if not self.added:
            return 0
        return len(self.added[-1]["ids"])


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

    results = await service.search("Hello")
    assert len(results) == 2
    assert results[0][0]["content"] == "Hello world"
    assert results[0][0]["source"] == "test"
    assert results[0][1] == pytest.approx(0.9)
    assert results[0][0]["metadata"]["doc_id"] == "doc-1"
    assert results[0][0]["metadata"]["lexical_score"] == 1.0
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


@pytest.mark.asyncio
async def test_chroma_stats_reads_collection_count(monkeypatch):
    service = VectorStoreService()
    service._backend = "chroma"
    service._store = DummyChromaStore()
    monkeypatch.setattr(service, "_embeddings", DummyEmbeddings())

    chunks = [
        DocumentChunk("A", "doc", 0, DocType.TEXT, {}),
        DocumentChunk("B", "doc", 1, DocType.TEXT, {}),
    ]
    await service.add_chunks(chunks)

    stats = await service.get_stats()

    assert stats["total_vectors"] == 2


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
            for text, metadata in zip(self.texts[:k], self.metadatas[:k], strict=False)
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
    assert results[0][1] == pytest.approx(0.9)


def test_lexical_boost_can_promote_exact_financial_term_match():
    exact = VectorStoreService._score_result(
        "liquidity coverage ratio",
        "The firm reported a liquidity coverage ratio of 113% for 2023.",
        {"source": "jpmorgan.pdf"},
        vector_score=0.2,
    )
    fuzzy = VectorStoreService._score_result(
        "liquidity coverage ratio",
        "The filing discusses liquidity risk management and cash needs.",
        {"source": "jpmorgan.pdf"},
        vector_score=0.6,
    )

    assert exact[1] > fuzzy[1]
    assert exact[0]["metadata"]["lexical_score"] == 1.0



def test_lexical_scoring_uses_source_filename_for_entity_queries():
    microsoft = VectorStoreService._score_result(
        "Microsoft reported revenue segments fiscal 2023",
        "Reported revenue by business segment is shown below.",
        {"source": "microsoft_2023_10k.pdf"},
        vector_score=0.2,
    )
    jpmorgan = VectorStoreService._score_result(
        "Microsoft reported revenue segments fiscal 2023",
        "Reported revenue by business segment is shown below.",
        {"source": "jpmorgan_2023_annual_report.pdf"},
        vector_score=0.2,
    )

    assert microsoft[0]["metadata"]["lexical_score"] == jpmorgan[0]["metadata"]["lexical_score"]
    assert microsoft[0]["metadata"]["metadata_score"] > jpmorgan[0]["metadata"]["metadata_score"]
def test_hash_embeddings_are_not_degenerate_for_short_texts():
    embeddings = vector_store_module._HashEmbeddings(dimensions=64)
    exact = embeddings.embed_query("AI")
    same = embeddings.embed_query("AI")
    different = embeddings.embed_query("duration")

    exact_similarity = sum(a * b for a, b in zip(exact, same, strict=False))
    different_similarity = sum(a * b for a, b in zip(exact, different, strict=False))

    assert any(value != 0.0 for value in exact)
    assert exact_similarity > different_similarity


@pytest.mark.asyncio
async def test_chroma_search_adds_lexical_candidates_outside_vector_top_k(monkeypatch):
    class QueryOnlyReturnsWeakCandidate(DummyChromaStore):
        def query(self, query_embeddings: list[list[float]], n_results: int, include: list[str]) -> dict:
            return {
                "documents": [["Generic liquidity risk overview."]],
                "metadatas": [[self.added[-1]["metadatas"][0]]],
                "distances": [[0.01]],
            }

    service = VectorStoreService()
    service._backend = "chroma"
    service._store = QueryOnlyReturnsWeakCandidate()
    monkeypatch.setattr(service, "_embeddings", DummyEmbeddings())

    chunks = [
        DocumentChunk("Generic liquidity risk overview.", "doc-1", 0, DocType.PDF, {"source": "report.pdf"}),
        DocumentChunk("Liquidity coverage ratio was 113% in 2023.", "doc-1", 1, DocType.PDF, {"source": "report.pdf"}),
    ]
    await service.add_chunks(chunks)

    results = await service.search("liquidity coverage ratio", top_k=1)

    assert results[0][0]["content"] == "Liquidity coverage ratio was 113% in 2023."


@pytest.mark.asyncio
async def test_chroma_lexical_scan_uses_configured_limit(monkeypatch):
    class StoreWithLimit(DummyChromaStore):
        def __init__(self):
            super().__init__()
            self.limit = None

        def get(self, include: list[str], limit: int | None = None) -> dict:
            self.limit = limit
            return super().get(include, limit)

    service = VectorStoreService()
    service._backend = "chroma"
    service._store = StoreWithLimit()
    monkeypatch.setattr(service, "_embeddings", DummyEmbeddings())
    monkeypatch.setattr(vector_store_module.settings, "chroma_lexical_scan_limit", 1)

    chunks = [
        DocumentChunk("Liquidity coverage ratio was 113% in 2023.", "doc-1", 0, DocType.PDF, {"source": "report.pdf"}),
        DocumentChunk("Another liquidity coverage ratio row.", "doc-1", 1, DocType.PDF, {"source": "report.pdf"}),
    ]
    await service.add_chunks(chunks)
    await service._chroma_lexical_candidates("liquidity coverage ratio", top_k=10)

    assert service._store.limit == 1
