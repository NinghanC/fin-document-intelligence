import os

import pytest

from agents.doc_parser_agent import DocParserAgent, DocType, DocumentChunk
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.qa_agent import QAAgent
from services.cdc_processor import CDCProcessor
from services.embedding_worker import get_embedding_client
from services.graph_rag import GraphRAGPipeline
from services.knowledge_graph import KnowledgeGraphService
from services.multimodal import MultimodalService
from services.vector_store import _HashEmbeddings


class FakeVectorStore:
    async def search(self, query: str, top_k: int = 5):
        return [
            (
                {
                    "content": "Global Income Fund reduced duration risk and increased its liquidity buffer.",
                    "source": "Q4_global_income_fund_risk_report.pdf",
                    "metadata": {"source": "Q4_global_income_fund_risk_report.pdf", "doc_id": "fund-report"},
                },
                0.91,
            )
        ]


class FakeKnowledgeGraph:
    async def get_neighbors(self, entity_name: str, hops: int = 2):
        if entity_name != "Global Income Fund":
            return []
        return [
            {
                "source": "Global Income Fund",
                "relations": ["related_to"],
                "target": "duration risk",
                "target_type": "Concept",
                "target_desc": "Interest-rate sensitivity monitored by the risk committee.",
            }
        ]

    async def execute_cypher(self, cypher: str, params: dict | None = None):
        return []


@pytest.mark.asyncio
async def test_hash_embeddings_make_vector_demo_searchable():
    embeddings = _HashEmbeddings(dimensions=32)
    docs = embeddings.embed_documents(["duration risk liquidity buffer"])
    query = embeddings.embed_query("duration risk")

    assert len(docs) == 1
    assert len(docs[0]) == 32
    assert len(query) == 32
    assert any(value != 0 for value in query)


@pytest.mark.asyncio
async def test_qa_uses_graphrag_and_returns_confidence_and_sources():
    agent = QAAgent(vector_store=FakeVectorStore(), knowledge_graph=FakeKnowledgeGraph())

    result = await agent.answer("What is the duration risk for Global Income Fund?")

    assert result.confidence > 0
    assert result.contexts
    assert any(ctx.retrieval_type == "vector" for ctx in result.contexts)
    assert any(ctx.retrieval_type == "graph" for ctx in result.contexts)
    assert any(ctx.source == "Q4_global_income_fund_risk_report.pdf" for ctx in result.contexts)
    assert "retrieved" in result.answer.lower() or "source" in result.answer.lower()


@pytest.mark.asyncio
async def test_graphrag_pipeline_returns_vector_and_graph_contexts():
    pipeline = GraphRAGPipeline(vector_store=FakeVectorStore(), knowledge_graph=FakeKnowledgeGraph())

    contexts = await pipeline.retrieve("Global Income Fund duration risk", top_k=5)

    assert contexts
    assert {ctx.source_type for ctx in contexts} >= {"vector", "subgraph"}


@pytest.mark.asyncio
async def test_memory_knowledge_graph_supports_neighbors_and_stats():
    graph = KnowledgeGraphService()
    extractor = KnowledgeExtractAgent()
    extraction = await extractor.extract_single(
        "Global Income Fund has duration risk and a liquidity buffer.",
        chunk_id="fund-report#chunk-0",
    )

    for entity in extraction.entities:
        await graph.upsert_entity(entity, source=extraction.source_chunk_id)
    for relation in extraction.relations:
        await graph.add_relation(relation, source=extraction.source_chunk_id)

    neighbors = await graph.get_neighbors("Global Income Fund")
    stats = await graph.get_stats()

    assert stats["backend"] == "memory"
    assert stats["total_entities"] >= 2
    assert stats["total_relations"] >= 1
    assert any(row["target"] == "duration risk" for row in neighbors)


@pytest.mark.asyncio
async def test_cdc_processor_tracks_versions_and_diff():
    processor = CDCProcessor()
    event = processor.from_filesystem_event(
        "modified",
        "fund-report.txt",
        content_before="duration risk\nliquidity buffer",
        content_after="duration risk\nliquidity buffer\ncredit spread",
    )

    result = await processor.process_event(event)

    assert result.success
    assert result.version == 1
    assert event.diff["added_count"] == 1
    assert processor.get_stats()["total_events_processed"] == 1


@pytest.mark.asyncio
async def test_doc_parser_text_and_extractor_demo_model(tmp_path):
    report = tmp_path / "fund-report.txt"
    report.write_text("Global Income Fund monitors duration risk and liquidity buffer.", encoding="utf-8")

    chunks = await DocParserAgent().parse(str(report))
    extraction = await KnowledgeExtractAgent().extract(chunks)

    assert chunks[0].doc_type == DocType.TEXT
    assert any(entity.name == "Global Income Fund" for item in extraction for entity in item.entities)


def test_multimodal_weighted_rerank_uses_modality_weights():
    service = MultimodalService()
    text_chunk = DocumentChunk("text result", "doc", 0, DocType.TEXT, {})
    image_chunk = DocumentChunk("image result", "doc", 1, DocType.IMAGE, {})

    results = service.weighted_rerank([(image_chunk, 1.0), (text_chunk, 0.9)])

    assert results[0].modality == "text"
    assert results[0].score == pytest.approx(0.9)
    assert results[1].modality == "image"
    assert results[1].score == pytest.approx(0.85)


def test_embedding_worker_disabled_path(monkeypatch):
    monkeypatch.setenv("DISABLE_LOCAL_EMBEDDINGS", "1")
    assert get_embedding_client() is None
    os.environ.pop("DISABLE_LOCAL_EMBEDDINGS", None)


def test_doc_id_uses_canonical_path(tmp_path, monkeypatch):
    nested = tmp_path / "uploads"
    nested.mkdir()
    report = nested / "fund-report.txt"
    report.write_text("Global Income Fund", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    relative_id = DocParserAgent._make_doc_id("uploads/fund-report.txt")
    absolute_id = DocParserAgent._make_doc_id(str(report))

    assert relative_id == absolute_id
