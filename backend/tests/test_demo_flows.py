import os

import pytest

from agents.doc_parser_agent import DocParserAgent, DocType
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.qa_agent import QAAgent
from services.cdc_processor import CDCProcessor
from services.embedding_worker import get_embedding_client
from services.graph_rag import GraphRAGPipeline
from services.knowledge_graph import KnowledgeGraphService
from services.multimodal import MultimodalService
from services.vector_store import _HashEmbeddings
from utils.model_clients import DemoChatModel


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
    def __init__(self):
        self.entities = {"Global Income Fund": {"name": "Global Income Fund"}}

    async def get_entity(self, name: str):
        return self.entities.get(name)

    async def search_entities(self, keyword: str, limit: int = 20):
        if keyword.lower() in "global income fund":
            return [{"name": "Global Income Fund"}]
        return []

    async def get_all_entity_names(self, limit: int = 1000):
        return list(self.entities)

    async def find_entity_alias(self, alias: str):
        return None

    async def find_entity_normalized(self, name: str):
        if name.lower() == "global income fund":
            return self.entities["Global Income Fund"]
        return None

    async def find_entities_by_name_similarity(self, mention: str, threshold: float = 0.8, limit: int = 3):
        return []

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

    async def get_community_summaries(self, entities: list[str], limit: int = 3):
        if "Global Income Fund" not in entities:
            return []
        return [
            {
                "community_id": "fund-risk",
                "members": ["Global Income Fund", "duration risk"],
                "summary": "Global Income Fund is linked to duration risk and liquidity buffer controls.",
            }
        ]


@pytest.mark.asyncio
async def test_hash_embeddings_make_vector_demo_searchable():
    embeddings = _HashEmbeddings(dimensions=32)
    docs = embeddings.embed_documents(["duration risk liquidity buffer"])
    query = embeddings.embed_query("duration risk")

    assert len(docs) == 1
    assert len(docs[0]) == 32
    assert len(query) == 32
    assert any(value != 0 for value in query)


def test_demo_model_answer_selects_relevant_metric_line():
    context = """\
[Source 1: northstar_bank.pdf | Type: vector | Score: 0.95]
Liquidity risk management overview
Liquidity coverage ratio was 113% for Northstar Bank in 2023.
"""
    answer = DemoChatModel._answer(
        f"Context information:\n{context}\n\nUser question: What liquidity coverage ratio did Northstar Bank report for 2023?"
    )

    assert "Liquidity coverage ratio" in answer
    assert "113" in answer


def test_demo_model_answer_focuses_long_table_excerpt_on_query_terms():
    context = """\
[Source 1: northstar_bank.pdf | Type: vector | Score: 0.95]
Selected ratios and metrics Return on common equity 17 % Return on tangible common equity 21 % Return on assets 1.30 Overhead ratio 55 Loans-to-deposits ratio 55 Firm Liquidity coverage ratio (LCR) average 113 112 111.
"""
    answer = DemoChatModel._answer(
        f"Context information:\n{context}\n\nUser question: What liquidity coverage ratio did Northstar Bank report for 2023?"
    )

    assert "Liquidity coverage ratio" in answer
    assert "113" in answer


def test_demo_model_says_insufficient_when_metric_context_is_truncated():
    context = """\
[Source 1: northstar_bank.pdf | Type: vector | Score: 0.95]
Selected ratios and metrics Return on common equity 17 % Return on assets 1.30 Overhead ratio 55 59 59 Lo.
"""
    answer = DemoChatModel._answer(
        f"Context information:\n{context}\n\nUser question: What liquidity coverage ratio did Northstar Bank report for 2023?"
    )

    assert "insufficient" in answer.lower()


def test_demo_model_answer_prefers_revenue_window_over_generic_risk_line():
    context = """\
[Source 1: cloudware.pdf | Type: vector | Score: 0.95]
Refer to Risk Factors in our fiscal year 2023 Form 10-K for a discussion of these factors.
Fiscal Year 2023 Compared with Fiscal Year 2022
Revenue increased $13.6 billion or 7% driven by growth in Intelligent Cloud and Productivity and Business Processes.
"""
    answer = DemoChatModel._answer(
        f"Context information:\n{context}\n\nUser question: What did Cloudware identify as major revenue sources in fiscal 2023?"
    )

    assert "Revenue increased $13.6 billion or 7%" in answer
    assert "Intelligent Cloud and Productivity and Business Processes" in answer


def test_demo_model_surfaces_segment_revenue_table_evidence():
    context = """\
[Source 1: cloudware.pdf | Type: vector | Score: 0.95]
Revenue
Productivity and Business Processes  $ 69,274   $ 63,364   $ 53,915
Intelligent Cloud   87,907    74,965    59,728
More Personal Computing   54,734    59,941    54,445
Total  $ 211,915   $ 198,270   $ 168,088
"""
    answer = DemoChatModel._answer(
        f"Context information:\n{context}\n\nUser question: What were Cloudware's reported revenue segments in fiscal 2023?"
    )

    assert "Productivity and Business Processes" in answer
    assert "Intelligent Cloud" in answer


def test_demo_model_surfaces_deferred_revenue_evidence():
    context = """\
[Source 1: apex_devices.pdf | Type: vector | Score: 0.95]
Total net sales include $8.2 billion of revenue recognized in 2023 that was included in deferred revenue as of September 24, 2022.
"""
    answer = DemoChatModel._answer(
        "Context information:\n"
        f"{context}\n\n"
        "User question: How much revenue did Apex Devices recognize in 2023 that was included in deferred revenue as of September 24, 2022?"
    )

    assert "$8.2 billion of revenue recognized in 2023" in answer


def test_demo_model_prefers_deferred_revenue_over_distractors():
    context = """\
[Source 1: distractor.pdf | Type: vector | Score: 1.00]
Our cloud revenue was $111.6 billion in fiscal year 2023.

[Source 2: apex_devices.pdf | Type: vector | Score: 0.95]
Total net sales include $8.2 billion of revenue recognized in 2023 that was included in deferred revenue as of September 24, 2022, $7.5 billion of revenue recognized in 2022.
"""
    answer = DemoChatModel._answer(
        "Context information:\n"
        f"{context}\n\n"
        "User question: How much revenue did Apex Devices recognize in 2023 that was included in deferred revenue as of September 24, 2022?"
    )

    assert "$8.2 billion of revenue recognized in 2023" in answer


def test_demo_model_classifies_how_much_as_factoid():
    assert DemoChatModel._classify_intent("How much revenue did Apex Devices recognize?") == "factoid"


@pytest.mark.asyncio
async def test_qa_uses_graphrag_and_returns_retrieval_quality_and_sources():
    agent = QAAgent(vector_store=FakeVectorStore(), knowledge_graph=FakeKnowledgeGraph())

    result = await agent.answer("What is the duration risk for Global Income Fund?")

    assert result.retrieval_quality > 0
    assert result.confidence == result.retrieval_quality
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
    assert {ctx.source_type for ctx in contexts} >= {"vector", "subgraph", "community"}


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
    async def handler(change):
        return type(
            "Result",
            (),
            {
                "change": change,
                "vectors_added": 1,
                "vectors_deleted": 0,
                "entities_added": 1,
                "entities_updated": 0,
                "relations_added": 0,
                "success": True,
                "error": "",
            },
        )()

    processor = CDCProcessor(update_handler=handler)
    event = processor.from_filesystem_event(
        "modified",
        "fund-report.txt",
        content_before="duration risk\nliquidity buffer",
        content_after="duration risk\nliquidity buffer\ncredit spread",
    )

    result = await processor.process_event(event)

    assert result.success
    assert result.version == 1
    assert result.chunks_affected == 1
    assert result.entities_affected == 1
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


@pytest.mark.asyncio
async def test_multimodal_service_reasons_over_serialized_table():
    service = MultimodalService()
    context = type(
        "Context",
        (),
        {
            "content": (
                "Headers: Fund | Sector | Exposure\n"
                "Fund: Global Income Fund | Sector: Technology | Exposure: 42%\n"
                "Fund: Credit Fund | Sector: Financials | Exposure: 18%"
            ),
            "source": "exposures.csv",
            "score": 0.6,
            "metadata": {"doc_type": "table", "source": "exposures.csv"},
        },
    )()

    results = await service.reason_over_contexts("What technology exposure does Global Income Fund have?", [context])

    assert results
    assert results[0].modality == "table"
    assert results[0].metadata["reasoning_mode"] == "structured_table_reasoning"
    assert "Technology" in results[0].content
    assert "42%" in results[0].content


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

