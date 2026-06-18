"""Optional Docker-backed infrastructure smoke tests.

These tests are skipped unless RUN_LIVE_INFRA_TESTS=1. They validate that the
real Neo4j, ChromaDB, and PGVector paths work, instead of only exercising
in-memory fallbacks and test doubles.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, Relation
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService, _HashEmbeddings

pytestmark = pytest.mark.live_infra


def _require_live_infra() -> None:
    if os.getenv("RUN_LIVE_INFRA_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_INFRA_TESTS=1 to run Docker-backed infrastructure smoke tests")


@pytest.mark.asyncio
async def test_live_neo4j_writes_and_reads_relationship(monkeypatch) -> None:
    _require_live_infra()
    suffix = uuid4().hex[:8]
    source = f"live-infra-neo4j-{suffix}"
    fund = f"Live Infra Fund {suffix}"
    company = f"Live Infra Company {suffix}"

    graph = KnowledgeGraphService()
    await graph.init()
    try:
        await graph.upsert_entity(Entity(fund, "Fund"), source=source)
        await graph.upsert_entity(Entity(company, "Company"), source=source)
        await graph.add_relation(Relation(fund, "holds", company, confidence=1.0), source=source)

        neighbors = await graph.get_neighbors(fund, hops=1)

        assert any(record.get("target") == company for record in neighbors)
        assert any("HOLDS" in record.get("relations", []) for record in neighbors)
    finally:
        await graph.delete_by_source(source)
        await graph.close()


@pytest.mark.asyncio
async def test_live_chromadb_http_adds_and_searches(monkeypatch) -> None:
    _require_live_infra()
    suffix = uuid4().hex[:8]
    doc_id = f"live-chroma-{suffix}"
    source = f"{doc_id}.txt"

    monkeypatch.setattr("services.vector_store.settings.vector_store_type", "chroma")
    monkeypatch.setattr("services.vector_store.settings.chroma_mode", "http")
    monkeypatch.setattr("services.vector_store.settings.chroma_host", "localhost")
    monkeypatch.setattr("services.vector_store.settings.chroma_port", 8000)

    service = VectorStoreService()
    service._embeddings = _HashEmbeddings()
    await service.init()
    try:
        chunk = DocumentChunk(
            content=f"Live Chroma liquidity coverage ratio smoke test {suffix}",
            doc_id=doc_id,
            chunk_index=0,
            doc_type=DocType.TEXT,
            metadata={"source": source},
        )
        assert await service.add_chunks([chunk]) == 1

        results = await service.search("liquidity coverage ratio", top_k=3)

        assert any(result[0]["metadata"].get("doc_id") == doc_id for result in results)
    finally:
        await service.delete_by_doc_id(doc_id)


@pytest.mark.asyncio
async def test_live_pgvector_adds_and_searches(monkeypatch) -> None:
    _require_live_infra()
    suffix = uuid4().hex[:8]
    doc_id = f"live-pgvector-{suffix}"
    source = f"{doc_id}.txt"

    monkeypatch.setattr("services.vector_store.settings.vector_store_type", "pgvector")
    monkeypatch.setattr("services.vector_store.settings.pgvector_dsn", "postgresql://postgres:postgres@localhost:5432/knowledge")

    service = VectorStoreService()
    service._embeddings = _HashEmbeddings()
    await service.init()
    try:
        chunk = DocumentChunk(
            content=f"Live PGVector duration risk smoke test {suffix}",
            doc_id=doc_id,
            chunk_index=0,
            doc_type=DocType.TEXT,
            metadata={"source": source},
        )
        assert await service.add_chunks([chunk]) == 1

        results = await service.search("duration risk", top_k=3)

        assert any(result[0]["metadata"].get("doc_id") == doc_id for result in results)
    finally:
        await service.delete_by_doc_id(doc_id)
