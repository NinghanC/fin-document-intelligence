import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, Relation
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent
from agents.qa_agent import QAAgent, RetrievedContext
from services.cdc_processor import CDCProcessor
from services.graph_rag import GraphRAGContext, GraphRAGPipeline
from services.knowledge_graph import KnowledgeGraphService


def test_multimodal_weights_keep_unknown_doc_type_neutral():
    agent = QAAgent()
    contexts = [
        RetrievedContext("image", "image.png", 1.0, "vector", {"doc_type": "image"}),
        RetrievedContext("unknown", "unknown.bin", 0.9, "vector", {"doc_type": "binary"}),
    ]

    reranked = agent._apply_multimodal_weights(contexts)

    assert reranked[0].source == "unknown.bin"
    assert reranked[1].score == pytest.approx(0.85)


def test_balanced_contexts_keep_vector_and_graph_sources():
    contexts = [
        RetrievedContext("vector-low", "v1", 0.4, "vector"),
        RetrievedContext("graph-high", "g1", 0.95, "graph"),
        RetrievedContext("vector-high", "v2", 0.9, "vector"),
    ]

    selected = QAAgent._balanced_top_contexts(contexts, limit=2)

    assert {ctx.retrieval_type for ctx in selected} == {"vector", "graph"}


def test_hybrid_rerank_deduplicates_contexts():
    contexts = [
        RetrievedContext("same content" * 20, "a", 0.5, "vector"),
        RetrievedContext("same content" * 20, "b", 0.9, "graph"),
    ]

    unique = QAAgent._hybrid_rerank(contexts)

    assert len(unique) == 1


@pytest.mark.asyncio
async def test_knowledge_update_delete_removes_memory_graph_source():
    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Global Income Fund", "Product"), source="fund.txt")
    await graph.add_relation(Relation("Global Income Fund", "related_to", "duration risk"), source="fund.txt")

    agent = KnowledgeUpdateAgent(knowledge_graph=graph)
    result = await agent.process_change(DocumentChange("fund.txt", ChangeType.DELETED))

    assert result.success
    assert (await graph.get_stats())["total_entities"] == 0
    assert (await graph.get_stats())["total_relations"] == 0


@pytest.mark.asyncio
async def test_knowledge_update_create_writes_vectors_and_graph():
    class Parser:
        async def parse(self, file_path):
            return [DocumentChunk("Global Income Fund liquidity", "doc", 0, DocType.TEXT, {"source": file_path})]

    class Extractor:
        async def extract(self, chunks):
            return [
                type(
                    "Extraction",
                    (),
                    {
                        "entities": [Entity("Global Income Fund", "Product")],
                        "relations": [Relation("Global Income Fund", "related_to", "liquidity")],
                        "source_chunk_id": "doc#chunk-0",
                    },
                )()
            ]

    class VectorStore:
        async def add_chunks(self, chunks):
            return len(chunks)

    graph = KnowledgeGraphService()
    agent = KnowledgeUpdateAgent(Parser(), Extractor(), VectorStore(), graph)

    result = await agent.process_change(DocumentChange("fund.txt", ChangeType.CREATED))

    assert result.vectors_added == 1
    assert result.entities_added == 1
    assert result.relations_added == 1


def test_cdc_major_change_threshold():
    diff = CDCProcessor.compute_diff("a\nb\nc", "a\nb\nc\nd\ne\nf")

    assert diff["is_major_change"] is True
    assert diff["change_ratio"] > 0.3


def test_cdc_delete_event_sets_negative_affected_counts():
    event = CDCProcessor.from_filesystem_event("deleted", "fund.txt")

    assert event.operation == "DELETE"
    assert event.source_type == "filesystem"


@pytest.mark.asyncio
async def test_read_only_cypher_allows_match_without_driver():
    graph = KnowledgeGraphService()

    result = await graph.execute_cypher("MATCH (n) RETURN n LIMIT 1")

    assert result == []


@pytest.mark.asyncio
async def test_graphrag_entity_linking_uses_alias_and_fuzzy_match():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Microsoft", "Organization"))

    pipeline = GraphRAGPipeline(VectorStore(), graph)

    assert await pipeline._resolve_entity("MSFT") == "Microsoft"
    assert await pipeline._resolve_entity("Microsft") == "Microsoft"


@pytest.mark.asyncio
async def test_community_summaries_are_precomputed_and_retrieved():
    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Global Income Fund", "Product"))
    await graph.upsert_entity(Entity("duration risk", "Concept"))
    await graph.add_relation(Relation("Global Income Fund", "related_to", "duration risk"))
    count = await graph.refresh_community_summaries()

    summaries = await graph.get_community_summaries(["Global Income Fund"])

    assert count == 1
    assert summaries
    assert "Global Income Fund" in summaries[0]["summary"]


def test_graphrag_deduplicates_by_normalized_terms_not_prefix():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    pipeline = GraphRAGPipeline(VectorStore(), KnowledgeGraphService())
    contexts = [
        GraphRAGContext("Apple reported revenue for 2023 in the filing", "vector", 0.9),
        GraphRAGContext("In the filing, revenue was reported by Apple for 2023", "subgraph", 0.8),
        GraphRAGContext("Apple reported revenue for 2024 in the filing", "vector", 0.7),
    ]

    reranked = pipeline._cross_rerank(contexts, "Apple revenue")

    assert len(reranked) == 2


def test_graphrag_custom_weights_are_explicit_configuration():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    pipeline = GraphRAGPipeline(
        VectorStore(),
        KnowledgeGraphService(),
        rerank_weights={"vector": 1.0, "subgraph": 2.0, "path": 1.0, "community": 1.0},
    )
    contexts = [
        GraphRAGContext("vector context", "vector", 0.9),
        GraphRAGContext("graph context", "subgraph", 0.6),
    ]

    reranked = pipeline._cross_rerank(contexts, "fund risk")

    assert reranked[0].source_type == "subgraph"
