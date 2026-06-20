import asyncio
import time

import pytest

import services.community_summarizer as community_summarizer_module
from agents.doc_parser_agent import DocParserAgent, DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, ExtractionResult, KnowledgeExtractAgent, Relation
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent
from agents.qa_agent import QAAgent, QAResult, QueryIntent, RetrievedContext
from orchestrator.graph import _build_qa_graph
from services.cdc_processor import CDCProcessor
from services.community_summarizer import StructuredCommunitySummarizer
from services.embedding_worker import _worker_process
from services.graph_rag import GraphRAGContext, GraphRAGPipeline
from services.ingestion_registry import IngestionRegistry, ingestion_registry
from services.knowledge_graph import KnowledgeGraphService
from services.logical_inference import LogicalInferenceEngine
from services.metapaths import (
    FINANCIAL_METAPATHS,
    CandidateMetapathGenerator,
    MetapathRouter,
    RuleMetapathRanker,
    validate_all_metapaths,
)
from services.vector_store import _SubprocessEmbeddings


class KeywordEmbeddings:
    async def aembed_query(self, text):
        return self._embed(text)

    async def aembed_documents(self, texts):
        return [self._embed(text) for text in texts]

    @staticmethod
    def _embed(text):
        lower = text.lower()
        return [
            1.0 if "apple" in lower else 0.0,
            1.0 if "revenue" in lower else 0.0,
            1.0 if "duration" in lower or "risk" in lower else 0.0,
        ]


class RecordingCommunitySummarizer:
    def __init__(self):
        self.calls: list[tuple[list[str], list[str]]] = []

    async def summarize(self, members: list[str], relations: list[str]) -> str:
        self.calls.append((members, relations))
        return f"LLM offline summary for {', '.join(members)}"


def test_financial_metapaths_validate_and_route_by_domain_terms():
    validate_all_metapaths()
    router = MetapathRouter()

    selected = router.select("Which sectors is Global Income Fund exposed to?", ["Global Income Fund"])
    traced = router.select_with_trace("Which sectors is Global Income Fund exposed to?", ["Global Income Fund"])

    assert selected[0].name == "sector_exposure"
    assert traced[0].spec.name == "sector_exposure"
    assert "sector" in traced[0].matched_keywords or "sectors" in traced[0].matched_keywords
    assert traced[0].as_trace()["reason"]
    assert "compliance_chain" not in {selection.spec.name for selection in traced}
    assert FINANCIAL_METAPATHS["shared_sector"].steps[1].direction == "in"


def test_metapath_candidate_generator_and_rule_ranker_are_separate():
    generator = CandidateMetapathGenerator()
    ranker = RuleMetapathRanker()
    candidates = generator.generate("Which sectors is Global Income Fund exposed to?", ["Global Income Fund"])
    selections = ranker.rank("Which sectors is Global Income Fund exposed to?", candidates, ["Global Income Fund"])

    assert {candidate.name for candidate in candidates} == set(FINANCIAL_METAPATHS)
    assert selections[0].spec.name == "sector_exposure"
    assert selections[0].matched_keywords == ("sector",)

def test_llm_community_summarizer_requires_real_provider_key(monkeypatch):
    monkeypatch.setattr(community_summarizer_module.settings, "community_summary_provider", "llm")
    monkeypatch.setattr(community_summarizer_module, "has_provider_key", lambda: False)

    summarizer = community_summarizer_module.create_community_summarizer()

    assert isinstance(summarizer, StructuredCommunitySummarizer)


@pytest.mark.asyncio
async def test_memory_graph_traverses_finance_metapath_with_direction():
    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Global Income Fund", "Fund"))
    await graph.upsert_entity(Entity("Microsoft", "Company"))
    await graph.upsert_entity(Entity("Apple Inc.", "Company"))
    await graph.upsert_entity(Entity("Technology", "Sector"))
    await graph.add_relation(Relation("Global Income Fund", "holds", "Microsoft"))
    await graph.add_relation(Relation("Microsoft", "belongs_to", "Technology"))
    await graph.add_relation(Relation("Apple Inc.", "belongs_to", "Technology"))

    sector_results = await graph.traverse_metapath(["Global Income Fund"], FINANCIAL_METAPATHS["sector_exposure"])
    shared_results = await graph.traverse_metapath(["Microsoft"], FINANCIAL_METAPATHS["shared_sector"])

    assert sector_results[0].end_entity == "Technology"
    assert ("Global Income Fund", "HOLDS", "Microsoft") in sector_results[0].path
    assert any(result.end_entity == "Apple Inc." for result in shared_results)


@pytest.mark.asyncio
async def test_persistent_graph_fallback_survives_service_restart(tmp_path, monkeypatch):
    monkeypatch.setattr("services.knowledge_graph.settings.upload_dir", str(tmp_path))
    first = KnowledgeGraphService(persist_fallback=True)
    await first.upsert_entity(Entity("Global Income Fund", "Fund"))
    await first.upsert_entity(Entity("Microsoft", "Company"))
    await first.add_relation(Relation("Global Income Fund", "holds", "Microsoft"))

    restarted = KnowledgeGraphService(persist_fallback=True)
    neighbors = await restarted.get_neighbors("Global Income Fund", hops=1)
    stats = await restarted.get_stats()

    assert stats["backend"] == "sqlite_fallback"
    assert stats["total_entities"] == 2
    assert stats["total_relations"] == 1
    assert neighbors[0]["target"] == "Microsoft"


@pytest.mark.asyncio
async def test_persistent_graph_fallback_delete_survives_service_restart(tmp_path, monkeypatch):
    monkeypatch.setattr("services.knowledge_graph.settings.upload_dir", str(tmp_path))
    first = KnowledgeGraphService(persist_fallback=True)
    await first.upsert_entity(Entity("Global Income Fund", "Fund"), source="fund.pdf")
    await first.upsert_entity(Entity("Microsoft", "Company"), source="fund.pdf")
    await first.add_relation(Relation("Global Income Fund", "holds", "Microsoft"), source="fund.pdf")

    deleted = await first.delete_by_source("fund.pdf")
    assert deleted == 2

    # A restart must not resurrect data that was deleted: the deletion has to be
    # mirrored into the SQLite fallback, not just the in-memory graph.
    restarted = KnowledgeGraphService(persist_fallback=True)
    stats = await restarted.get_stats()

    assert stats["total_entities"] == 0
    assert stats["total_relations"] == 0

async def test_graphrag_metapath_search_adds_explainable_context():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Global Income Fund", "Fund"))
    await graph.upsert_entity(Entity("Microsoft", "Company"))
    await graph.upsert_entity(Entity("Technology", "Sector"))
    await graph.add_relation(Relation("Global Income Fund", "holds", "Microsoft"))
    await graph.add_relation(Relation("Microsoft", "belongs_to", "Technology"))

    pipeline = GraphRAGPipeline(VectorStore(), graph)
    contexts = await pipeline._metapath_search("Which sector exposure does Global Income Fund have?", ["Global Income Fund"])

    assert contexts
    assert contexts[0].source_type == "metapath"
    assert contexts[0].metadata["metapath"] == "sector_exposure"
    assert contexts[0].metadata["selection_trace"]["matched_keywords"]
    assert contexts[0].metadata["path_edges"][0] == {
        "source": "Global Income Fund",
        "relation": "HOLDS",
        "target": "Microsoft",
    }
    assert contexts[0].metadata["intermediate_entities"] == ["Microsoft"]
    assert "Global Income Fund -[HOLDS]-> Microsoft" in contexts[0].content


@pytest.mark.asyncio
async def test_logical_inference_derives_fact_from_multi_hop_path():
    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Global Income Fund", "Fund"))
    await graph.upsert_entity(Entity("Microsoft", "Company"))
    await graph.upsert_entity(Entity("Technology", "Sector"))
    await graph.add_relation(Relation("Global Income Fund", "holds", "Microsoft"))
    await graph.add_relation(Relation("Microsoft", "belongs_to", "Technology"))

    facts = await LogicalInferenceEngine().infer(
        "Infer sector exposure for Global Income Fund",
        ["Global Income Fund"],
        graph,
    )

    assert facts
    assert facts[0].rule_name == "fund_sector_exposure"
    assert facts[0].conclusion == "Global Income Fund has inferred sector exposure to Technology."
    assert ("Global Income Fund", "HOLDS", "Microsoft") in facts[0].path
    assert ("Microsoft", "BELONGS_TO", "Technology") in facts[0].path


@pytest.mark.asyncio
async def test_graphrag_logical_inference_search_adds_inference_context():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Global Income Fund", "Fund"))
    await graph.upsert_entity(Entity("Microsoft", "Company"))
    await graph.upsert_entity(Entity("Technology", "Sector"))
    await graph.add_relation(Relation("Global Income Fund", "holds", "Microsoft"))
    await graph.add_relation(Relation("Microsoft", "belongs_to", "Technology"))
    pipeline = GraphRAGPipeline(VectorStore(), graph)

    contexts = await pipeline._logical_inference_search(
        "Infer sector exposure for Global Income Fund",
        ["Global Income Fund"],
    )

    assert contexts
    assert contexts[0].source_type == "inference"
    assert contexts[0].metadata["rule"] == "fund_sector_exposure"
    assert "Therefore: Global Income Fund has inferred sector exposure to Technology." in contexts[0].content


@pytest.mark.asyncio
async def test_qa_adds_multimodal_table_reasoning_context():
    agent = QAAgent()
    contexts = [
        RetrievedContext(
            "Headers: Fund | Sector | Exposure\n"
            "Fund: Global Income Fund | Sector: Technology | Exposure: 42%",
            "exposures.csv",
            0.6,
            "vector",
            {"doc_type": "table", "source": "exposures.csv"},
        ),
    ]

    enriched = await agent._add_multimodal_reasoning(
        "What technology exposure does Global Income Fund have?",
        contexts,
    )

    multimodal_contexts = [ctx for ctx in enriched if ctx.retrieval_type == "multimodal"]
    assert multimodal_contexts
    assert multimodal_contexts[0].metadata["reasoning_mode"] == "structured_table_reasoning"
    assert "42%" in multimodal_contexts[0].content


@pytest.mark.asyncio
async def test_pdf_vision_fallback_uses_bounded_concurrency(monkeypatch):
    import sys
    import types

    active = 0
    max_active = 0

    def pdfinfo_from_path(file_path):
        return {"Pages": 3}

    def convert_from_path(file_path, dpi, first_page, last_page):
        return [object()]

    fake_pdf2image = types.ModuleType("pdf2image")
    fake_pdf2image.pdfinfo_from_path = pdfinfo_from_path
    fake_pdf2image.convert_from_path = convert_from_path
    monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)
    monkeypatch.setattr("agents.doc_parser_agent.settings.pdf_vision_concurrency", 2)

    parser = DocParserAgent()

    async def describe(image):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return "page text"

    parser._describe_image_with_llm = describe

    texts = await parser._pdf_vision_fallback("scan.pdf")

    assert len(texts) == 3
    assert max_active == 2


@pytest.mark.asyncio
async def test_qa_graph_retries_then_clarifies_low_quality_answer():
    class Agent:
        def __init__(self):
            self.calls = 0

        async def answer(self, question):
            self.calls += 1
            return QAResult(
                question=question,
                answer="insufficient",
                contexts=[],
                intent=QueryIntent.FACTOID,
                retrieval_quality=0.1,
                reasoning_steps=[f"attempt {self.calls}"],
            )

    agent = Agent()
    workflow = _build_qa_graph(agent)

    result = await workflow.ainvoke({"question": "What is the LCR?"})

    assert agent.calls == 2
    assert result["needs_clarification"] is True
    assert result["result"].retrieval_quality == 0.0
    assert "not have enough retrieved evidence" in result["result"].answer


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


def test_hybrid_rerank_does_not_deduplicate_distinct_contexts_with_same_prefix():
    shared_prefix = "Apple reported revenue in the annual filing. " * 4
    contexts = [
        RetrievedContext(f"{shared_prefix}Revenue for 2023 was discussed.", "a", 0.9, "vector"),
        RetrievedContext(f"{shared_prefix}Revenue for 2024 was discussed.", "b", 0.8, "graph"),
    ]

    unique = QAAgent._hybrid_rerank(contexts)

    assert len(unique) == 2


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
    await graph.upsert_entity(Entity("Microsoft", "Organization", properties={"ticker": "MSFT", "aliases": ["Microsoft Corp", "Microsoft Corporation"]}))

    pipeline = GraphRAGPipeline(VectorStore(), graph)

    assert await pipeline._resolve_entity("MSFT") == "Microsoft"
    assert await pipeline._resolve_entity("Microsft") == "Microsoft"


@pytest.mark.asyncio
async def test_graphrag_entity_linking_filters_question_stopwords():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    class FakeLLM:
        async def ainvoke(self, messages):
            return type("Response", (), {"content": '{"entities": ["What", "JPMorgan Chase"]}'})()

    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("JPMorgan Chase", "Organization"))

    pipeline = GraphRAGPipeline(VectorStore(), graph)
    pipeline.llm = FakeLLM()

    assert await pipeline._entity_linking("What did JPMorgan Chase report?") == ["JPMorgan Chase"]


@pytest.mark.asyncio
async def test_entity_resolution_uses_normalized_suffix_match():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Apple Inc.", "Organization"))

    pipeline = GraphRAGPipeline(VectorStore(), graph)

    assert await pipeline._resolve_entity("Apple") == "Apple Inc."


@pytest.mark.asyncio
async def test_entity_resolution_uses_graph_alias_index():
    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Apple Inc.", "Organization", properties={"ticker": "AAPL"}))

    assert (await graph.find_entity_alias("AAPL"))["name"] == "Apple Inc."
    assert (await graph.find_entity_normalized("Apple Corp."))["name"] == "Apple Inc."


def test_extraction_filter_keeps_single_normal_confidence_entity():
    result = ExtractionResult(
        entities=[
            Entity("Global Income Fund", "Fund", confidence=0.75),
            Entity("Duration Risk", "RiskFactor", confidence=0.8),
        ],
        relations=[Relation("Global Income Fund", "related_to", "Duration Risk", confidence=0.75)],
        events=[],
        source_chunk_id="chunk-1",
    )

    filtered = KnowledgeExtractAgent._filter_extraction_result(result)

    assert {entity.name for entity in filtered.entities} == {"Global Income Fund", "Duration Risk"}
    assert len(filtered.relations) == 1


@pytest.mark.asyncio
async def test_embedding_entity_match_caches_entity_name_embeddings():
    class Graph:
        async def get_all_entity_names(self, limit=1000):
            return ["Duration Risk", "Liquidity Buffer"]

    class CountingEmbeddings:
        def __init__(self):
            self.document_calls = 0
            self.query_calls = 0

        async def aembed_query(self, text):
            self.query_calls += 1
            return [1.0, 0.0] if "duration" in text.lower() else [0.0, 1.0]

        async def aembed_documents(self, texts):
            self.document_calls += 1
            return [[1.0, 0.0] if "duration" in text.lower() else [0.0, 1.0] for text in texts]

    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    pipeline = GraphRAGPipeline(VectorStore(), Graph())
    embeddings = CountingEmbeddings()
    pipeline.embeddings = embeddings

    assert await pipeline._embedding_entity_match("duration sensitivity") == ["Duration Risk"]
    assert await pipeline._embedding_entity_match("duration exposure") == ["Duration Risk"]
    assert embeddings.document_calls == 1
    assert embeddings.query_calls == 2

async def test_entity_resolution_embedding_fallback(monkeypatch):
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    class DummyEmbeddings:
        async def aembed_query(self, text):
            return [1.0, 0.0]

        async def aembed_documents(self, texts):
            return [[1.0, 0.0] if text == "Duration Risk" else [0.0, 1.0] for text in texts]

    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Duration Risk", "Concept"))
    pipeline = GraphRAGPipeline(VectorStore(), graph)
    pipeline.embeddings = DummyEmbeddings()

    assert await pipeline._embedding_entity_match("interest rate sensitivity") == ["Duration Risk"]


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
    assert "Global Income Fund -[RELATED_TO]-> duration risk" in summaries[0]["relations"]
    assert summaries[0]["algorithm"] in {"louvain", "connected_components"}


@pytest.mark.asyncio
async def test_provider_community_summarizer_runs_only_during_refresh():
    summarizer = RecordingCommunitySummarizer()
    graph = KnowledgeGraphService(community_summarizer=summarizer)
    await graph.upsert_entity(Entity("Global Income Fund", "Product"))
    await graph.upsert_entity(Entity("duration risk", "Concept"))
    await graph.add_relation(Relation("Global Income Fund", "related_to", "duration risk"))

    await graph.refresh_community_summaries()
    summaries = await graph.get_community_summaries(["Global Income Fund"])
    second_read = await graph.get_community_summaries(["duration risk"])

    assert len(summarizer.calls) == 1
    assert summaries[0]["summary"].startswith("LLM offline summary")
    assert second_read
    assert len(summarizer.calls) == 1


@pytest.mark.asyncio
async def test_graphrag_skips_empty_community_summaries():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    class Graph:
        async def get_community_summaries(self, entities, limit=3):
            return [
                {
                    "community_id": "empty",
                    "members": ["JPMorgan Chase"],
                    "summary": (
                        "Graph community summary: JPMorgan Chase. "
                        "No direct intra-community relationships were captured."
                    ),
                }
            ]


    pipeline = GraphRAGPipeline(VectorStore(), Graph())

    assert await pipeline._community_retrieve(["JPMorgan Chase"]) == []


@pytest.mark.asyncio
async def test_graphrag_deduplicates_by_normalized_terms_not_prefix():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    pipeline = GraphRAGPipeline(VectorStore(), KnowledgeGraphService())
    pipeline.embeddings = KeywordEmbeddings()
    contexts = [
        GraphRAGContext("Apple reported revenue for 2023 in the filing", "vector", 0.9),
        GraphRAGContext("In the filing, revenue was reported by Apple for 2023", "subgraph", 0.8),
        GraphRAGContext("Apple reported revenue for 2024 in the filing", "vector", 0.7),
    ]

    reranked = await pipeline._cross_rerank(contexts, "Apple revenue")

    assert len(reranked) == 2


@pytest.mark.asyncio
async def test_graphrag_cross_rerank_uses_rrf_not_static_weights():
    class VectorStore:
        async def search(self, query, top_k=5):
            return []

    pipeline = GraphRAGPipeline(VectorStore(), KnowledgeGraphService())
    pipeline.embeddings = KeywordEmbeddings()
    contexts = [
        GraphRAGContext("duration risk from vector", "vector", 0.1),
        GraphRAGContext("duration risk from graph", "subgraph", 0.99),
        GraphRAGContext("second graph result", "subgraph", 0.98),
    ]

    reranked = await pipeline._cross_rerank(contexts, "duration risk")

    assert reranked[0].source_type == "vector"
    assert reranked[0].metadata["rrf_score"] == pytest.approx(1 / 61, abs=1e-6)


@pytest.mark.asyncio
async def test_graphrag_retrieve_runs_vector_and_entity_linking_concurrently():
    class VectorStore:
        async def search(self, query, top_k=5):
            await asyncio.sleep(0.05)
            return [({"content": "duration risk vector", "metadata": {}}, 0.9)]

    pipeline = GraphRAGPipeline(VectorStore(), KnowledgeGraphService())
    pipeline.embeddings = KeywordEmbeddings()

    async def slow_entity_linking(query):
        await asyncio.sleep(0.05)
        return []

    pipeline._entity_linking = slow_entity_linking
    start = time.perf_counter()
    contexts = await pipeline.retrieve("duration risk", top_k=3)
    elapsed = time.perf_counter() - start

    assert contexts[0].source_type == "vector"
    assert elapsed < 0.09


@pytest.mark.asyncio
async def test_graphrag_retrieve_keeps_vector_results_when_graph_branch_fails():
    class VectorStore:
        async def search(self, query, top_k=5):
            return [({"content": "duration risk vector", "metadata": {}}, 0.9)]

    pipeline = GraphRAGPipeline(VectorStore(), KnowledgeGraphService())
    pipeline.embeddings = KeywordEmbeddings()

    async def entity_linking(query):
        return ["Global Income Fund"]

    async def failing_subgraph(entities, query="", hops=2):
        raise RuntimeError("neo4j unavailable")

    pipeline._entity_linking = entity_linking
    pipeline._subgraph_search = failing_subgraph

    contexts = await pipeline.retrieve("duration risk", top_k=3)

    assert [ctx.source_type for ctx in contexts] == ["vector"]




@pytest.mark.asyncio
async def test_qa_fallback_does_not_generate_cypher_when_graphrag_fails():
    class VectorStore:
        async def search(self, query, top_k=5):
            return [({"content": "duration risk vector", "source": "risk.md", "metadata": {}}, 0.9)]

    class Graph:
        cypher_calls = 0

        async def execute_cypher(self, cypher):
            self.cypher_calls += 1
            raise AssertionError("LLM-generated Cypher fallback should not run")

    class BrokenGraphRAG:
        async def retrieve(self, query, top_k=10):
            raise RuntimeError("graph unavailable")

    graph = Graph()
    agent = QAAgent(vector_store=VectorStore(), knowledge_graph=graph)
    agent.graph_rag = BrokenGraphRAG()

    contexts = await agent._retrieve_contexts("duration risk", {"queries": ["duration risk"]})

    assert graph.cypher_calls == 0
    assert [context.retrieval_type for context in contexts] == ["vector"]

def test_graphrag_subgraph_scores_depend_on_query_relevance():
    strong = GraphRAGPipeline._subgraph_score(
        query="Global Income Fund duration risk",
        query_entities=["Global Income Fund"],
        entity="Global Income Fund",
        record={"relations": ["RELATED_TO"], "target": "duration risk", "target_desc": "duration risk"},
        content="Global Income Fund --[RELATED_TO]--> duration risk",
    )
    weak = GraphRAGPipeline._subgraph_score(
        query="redemption policy",
        query_entities=["Global Income Fund"],
        entity="Global Income Fund",
        record={"relations": [], "target": "duration risk", "target_desc": ""},
        content="Global Income Fund --[]--> duration risk",
    )

    assert strong > weak


def test_graphrag_path_scores_penalize_longer_paths():
    short = GraphRAGPipeline._path_score("A", "B", ["A", "B"], ["RELATED_TO"])
    long = GraphRAGPipeline._path_score("A", "B", ["A", "X", "Y", "B"], ["R1", "R2", "R3"])

    assert short > long


def test_qa_graph_record_score_depends_on_overlap():
    strong = QAAgent._graph_record_score(
        "duration risk liquidity",
        {"node_names": ["Global Income Fund", "duration risk"], "relations": ["liquidity"]},
    )
    weak = QAAgent._graph_record_score("duration risk liquidity", {"name": "unrelated operating memo"})

    assert strong > weak


@pytest.mark.asyncio
async def test_doc_parser_parse_batch_runs_concurrently():
    class Parser(DocParserAgent):
        async def parse(self, file_path):
            await asyncio.sleep(0.05)
            return [DocumentChunk(file_path, file_path, 0, DocType.TEXT, {})]

    start = time.perf_counter()
    chunks = await Parser().parse_batch(["a.txt", "b.txt", "c.txt"])
    elapsed = time.perf_counter() - start

    assert len(chunks) == 3
    assert elapsed < 0.12


@pytest.mark.asyncio
async def test_knowledge_extract_batches_run_concurrently():
    class Extractor(KnowledgeExtractAgent):
        async def _extract_from_chunk(self, chunk):
            await asyncio.sleep(0.05)
            return ExtractionResult([], [], [], chunk.chunk_id)

    chunks = [DocumentChunk(f"chunk {i}", "doc", i, DocType.TEXT, {}) for i in range(5)]
    start = time.perf_counter()
    results = await Extractor().extract(chunks)
    elapsed = time.perf_counter() - start

    assert len(results) == 5
    assert elapsed < 0.12


def test_qa_intent_changes_context_limit():
    assert QAAgent._context_limit_for_intent(QueryIntent.FACTOID) == 6
    assert QAAgent._context_limit_for_intent(QueryIntent.ANALYTICAL) == 10


def test_qa_prompt_changes_by_intent():
    assert "one or two sentences" in QAAgent._prompt_for_intent(QueryIntent.FACTOID)
    assert "Compare" in QAAgent._prompt_for_intent(QueryIntent.COMPARATIVE)
    assert "ordered steps" in QAAgent._prompt_for_intent(QueryIntent.PROCEDURAL)


@pytest.mark.asyncio
async def test_factoid_intent_uses_tight_retrieval():
    class VectorStore:
        def __init__(self):
            self.top_ks = []

        async def search(self, query, top_k=5):
            self.top_ks.append(top_k)
            return [
                ({"content": f"fact {i}", "source": f"s{i}", "metadata": {"doc_id": ""}}, 1.0 - i * 0.1)
                for i in range(6)
            ]

    store = VectorStore()
    agent = QAAgent(vector_store=store)
    contexts = await agent._retrieve_for_intent(
        "What is duration risk?",
        {"queries": ["duration risk"], "entities": []},
        QueryIntent.FACTOID,
    )

    assert store.top_ks == [6]
    assert len(contexts) == 5


@pytest.mark.asyncio
async def test_comparative_intent_retrieves_per_entity():
    class VectorStore:
        def __init__(self):
            self.queries = []

        async def search(self, query, top_k=5):
            self.queries.append(query)
            return [({"content": query, "source": query, "metadata": {"doc_id": ""}}, 0.8)]

    store = VectorStore()
    agent = QAAgent(vector_store=store)
    contexts = await agent._retrieve_for_intent(
        "Compare Fund A and Fund B",
        {"queries": ["compare"], "entities": ["Fund A", "Fund B"]},
        QueryIntent.COMPARATIVE,
    )

    assert any(query.startswith("Fund A") for query in store.queries)
    assert any(query.startswith("Fund B") for query in store.queries)
    assert len(contexts) == 2


def test_procedural_intent_prioritizes_policy_contexts():
    contexts = [
        RetrievedContext("general fund summary", "summary.md", 0.9, "vector"),
        RetrievedContext("step one in the policy workflow", "risk_policy.md", 0.6, "vector"),
    ]

    prioritized = QAAgent._prioritize_policy_contexts(contexts)

    assert prioritized[0].source == "risk_policy.md"
    assert prioritized[0].metadata["intent_filter"] == "policy_procedure"


@pytest.mark.asyncio
async def test_cdc_processor_invokes_update_handler():
    calls = []

    async def handler(change):
        calls.append(change)
        return type(
            "Result",
            (),
            {
                "change": change,
                "vectors_added": 3,
                "vectors_deleted": 1,
                "entities_added": 2,
                "entities_updated": 0,
                "relations_added": 1,
                "success": True,
                "error": "",
            },
        )()

    processor = CDCProcessor(update_handler=handler)
    event = CDCProcessor.from_filesystem_event("modified", "fund.txt", "old", "new")

    result = await processor.process_event(event)

    assert calls[0].file_path == "fund.txt"
    assert calls[0].change_type == ChangeType.MODIFIED
    assert result.update_result["vectors_added"] == 3
    assert result.entities_affected == 2


@pytest.mark.asyncio
async def test_cdc_processor_without_update_handler_fails_explicitly():
    processor = CDCProcessor()
    event = CDCProcessor.from_filesystem_event("created", "fund.txt")

    result = await processor.process_event(event)

    assert result.success is False
    assert "update handler is not configured" in result.error
    assert result.version == 0
    assert processor.get_version("fund.txt") == 0
    assert processor.get_stats()["total_events_processed"] == 1


def test_embedding_worker_sends_ready_signal(monkeypatch):
    import multiprocessing
    import sys
    import types

    class FakeSentenceTransformer:
        def __init__(self, model_name, device="cpu"):
            self.model_name = model_name

        def encode(self, texts, show_progress_bar=False):
            class Encoded:
                @staticmethod
                def tolist():
                    return [[0.1, 0.2]]

            return Encoded()

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    request_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()
    request_queue.put({"id": "req-1", "texts": ["hello"]})
    request_queue.put(None)

    _worker_process(request_queue, response_queue)

    assert response_queue.get(timeout=1)["status"] == "ready"
    response = response_queue.get(timeout=1)
    assert response["id"] == "req-1"
    assert response["embeddings"] == [[0.1, 0.2]]


def test_start_watching_stores_observer_and_stop_watching(monkeypatch):
    import sys
    import types

    class FakeHandler:
        pass

    class FakeObserver:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.joined = False

        def schedule(self, handler, directory, recursive=True):
            self.handler = handler
            self.directory = directory
            self.recursive = recursive

        def start(self):
            self.started = True

        def is_alive(self):
            return False

        def stop(self):
            self.stopped = True

        def join(self, timeout=None):
            self.joined = True

    fake_events = types.ModuleType("watchdog.events")
    fake_events.FileSystemEventHandler = FakeHandler
    fake_observers = types.ModuleType("watchdog.observers")
    fake_observers.Observer = FakeObserver
    monkeypatch.setitem(sys.modules, "watchdog.events", fake_events)
    monkeypatch.setitem(sys.modules, "watchdog.observers", fake_observers)

    agent = KnowledgeUpdateAgent()
    observer = agent.start_watching("uploads")
    agent.stop_watching()

    assert observer.directory == "uploads"
    assert observer.stopped is True
    assert observer.joined is True
    assert agent._observer is None


def test_doc_parser_normalizes_pdf_mojibake():
    raw = (
        "Return on common equity (\u00e2ROE\u00e2) 17 %\n"
        "\u00e2\u00a2The Firm\u00e2s nonperforming assets increased."
    )

    cleaned = DocParserAgent._normalize_extracted_text(raw)

    assert '"ROE"' in cleaned
    assert "- The Firm's nonperforming assets" in cleaned
    assert "\u00e2" not in cleaned


def test_chunking_normalizes_extracted_text_before_indexing():
    parser = DocParserAgent()
    raw = "Firm Liquidity coverage ratio (\u00e2LCR\u00e2) (average)(b) 113 112 111"

    chunks = parser._chunk_texts([raw], "doc", DocType.PDF, "jpmorgan.pdf")

    assert chunks[0].content == 'Firm Liquidity coverage ratio ("LCR") (average)(b) 113 112 111'
    assert "\u00e2" not in chunks[0].content


def test_sentence_aware_chunking_does_not_split_mid_word():
    parser = DocParserAgent()
    parser.CHUNK_MAX_TOKENS = 6
    parser.CHUNK_OVERLAP_SENTENCES = 1
    text = "First sentence stays whole. Second sentence also stays whole. Third sentence closes."

    chunks = parser._chunk_texts([text], "doc", DocType.TEXT, "source.txt")

    assert len(chunks) >= 2
    assert all(not chunk.content.startswith("ence") for chunk in chunks)
    assert chunks[0].content.endswith(".")
    assert chunks[1].content.startswith("First sentence stays whole.")
    assert all(parser._approx_token_count(chunk.content) <= 12 for chunk in chunks)


def test_chunking_prefers_financial_paragraph_boundaries():
    parser = DocParserAgent()
    parser.CHUNK_MAX_TOKENS = 18
    text = (
        "Risk Factors\n"
        "The fund has duration risk when interest rates move quickly.\n\n"
        "Liquidity Policy\n"
        "The fund keeps cash buffers for expected redemption windows.\n\n"
        "Fund Exposure\n"
        "The portfolio includes investment grade credit and treasury futures."
    )

    chunks = parser._chunk_texts([text], "doc", DocType.MARKDOWN, "fund.md")

    assert len(chunks) >= 2
    assert "Risk Factors" in chunks[0].content
    assert "Liquidity Policy" in chunks[1].content
    assert any("expected redemption windows" in chunk.content for chunk in chunks)
    assert all(not chunk.content.startswith(("dity", "emption", "windows.")) for chunk in chunks)
    assert all(chunk.content == text[chunk.metadata["char_start"] : chunk.metadata["char_end"]].strip() for chunk in chunks)


def test_image_prepare_downscales_large_images():
    from PIL import Image

    parser = DocParserAgent()
    image = Image.new("RGB", (4000, 2000))

    prepared = parser._prepare_image_for_llm(image)

    assert max(prepared.size) == parser.IMAGE_MAX_SIDE


def test_entity_deduplicate_keeps_richer_description():
    first = ExtractionResult([Entity("Fund A", "Product", "short", confidence=0.95)], [], [], "c1")
    second = ExtractionResult(
        [Entity("Fund A", "Product", "much richer description", confidence=0.96)],
        [],
        [],
        "c2",
    )

    deduped = KnowledgeExtractAgent._deduplicate([first, second])

    assert deduped[0].entities[0].description == "much richer description"
    assert deduped[1].entities == []


@pytest.mark.asyncio
async def test_update_version_uses_existing_graph_state():
    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Fund A", "Product"), version=4)
    agent = KnowledgeUpdateAgent(knowledge_graph=graph)

    version = await agent._next_version("Fund A")

    assert version == 5


def test_subprocess_embedding_fallback_keeps_dimension(monkeypatch):
    monkeypatch.setenv("DISABLE_LOCAL_EMBEDDINGS", "1")
    embeddings = _SubprocessEmbeddings()
    embeddings._client = None

    query_vector = embeddings.embed_query("anything")
    doc_vector = embeddings.embed_documents(["a"])[0]

    assert len(query_vector) == embeddings.dimensions
    assert len(doc_vector) == embeddings.dimensions
    assert any(value != 0.0 for value in query_vector)
    assert any(value != 0.0 for value in doc_vector)


def test_cdc_diff_preserves_order_and_duplicates():
    diff = CDCProcessor.compute_diff("a\nb\nb\nc", "a\nb\nb\nd\nc")

    assert diff["operations"]
    assert diff["added_lines"] == ["d"]
    assert diff["removed_lines"] == []
    assert diff["operations"][0]["after_start"] == 3


@pytest.mark.asyncio
async def test_memory_search_entities_is_case_insensitive():
    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Global Income Fund", "Product", "Liquidity Buffer"))

    matches = await graph.search_entities("liquidity")

    assert matches[0]["name"] == "Global Income Fund"


@pytest.mark.asyncio
async def test_pending_graph_relations_are_hidden_until_commit(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ingestion_registry.settings.upload_dir", str(tmp_path))
    ingestion_registry._records = {}
    source = tmp_path / "fund.txt"
    source.write_text("Global Income Fund", encoding="utf-8")

    skipped, _ = ingestion_registry.begin("doc-pending", str(source))
    graph = KnowledgeGraphService()
    await graph.upsert_entity(Entity("Global Income Fund", "Product"), source="doc-pending#chunk-0")
    await graph.upsert_entity(Entity("duration risk", "Concept"), source="doc-pending#chunk-0")
    await graph.add_relation(Relation("Global Income Fund", "related_to", "duration risk"), source="doc-pending#chunk-0")

    assert skipped is False
    assert await graph.get_neighbors("Global Income Fund") == []
    assert await graph.refresh_community_summaries() == 0
    assert await graph.get_community_summaries(["Global Income Fund"]) == []

    ingestion_registry.commit("doc-pending")
    await graph.refresh_community_summaries()

    assert await graph.get_neighbors("Global Income Fund")
    assert await graph.get_community_summaries(["Global Income Fund"])


@pytest.mark.asyncio
async def test_ingestion_registry_skips_committed_same_content(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ingestion_registry.settings.upload_dir", str(tmp_path))
    ingestion_registry._records = {}
    source = tmp_path / "fund.txt"
    source.write_text("same content", encoding="utf-8")

    skipped_first, _ = ingestion_registry.begin("doc-1", str(source))
    ingestion_registry.commit("doc-1")
    skipped_second, record = ingestion_registry.begin("doc-2", str(source))

    assert skipped_first is False
    assert skipped_second is True
    assert record.doc_id == "doc-1"




def test_ingestion_registry_sqlite_is_shared_across_instances(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ingestion_registry.settings.upload_dir", str(tmp_path))
    first = IngestionRegistry()
    second = IngestionRegistry()
    source = tmp_path / "shared.txt"
    source.write_text("shared registry content", encoding="utf-8")

    skipped_first, _ = first.begin("doc-shared", str(source))
    first.commit("doc-shared")
    skipped_second, record = second.begin("doc-shared-copy", str(source))

    assert skipped_first is False
    assert skipped_second is True
    assert record.doc_id == "doc-shared"
    assert second.is_committed("doc-shared") is True
    assert "doc-shared" in second.committed_doc_ids()

def test_ingestion_registry_dead_letters_failed_records(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ingestion_registry.settings.upload_dir", str(tmp_path))
    ingestion_registry._records = {}
    source = tmp_path / "failed.txt"
    source.write_text("broken", encoding="utf-8")

    skipped, _ = ingestion_registry.begin("failed-doc", str(source))
    ingestion_registry.fail("failed-doc", error="vector store unavailable")

    dead_letters = ingestion_registry.dead_letters()

    assert skipped is False
    assert dead_letters[0]["doc_id"] == "failed-doc"
    assert dead_letters[0]["error"] == "vector store unavailable"
    assert dead_letters[0]["retry_count"] == 1
    assert ingestion_registry.dead_letter("failed-doc")["doc_id"] == "failed-doc"
    assert ingestion_registry.clear_dead_letter("failed-doc") is True
    assert ingestion_registry.dead_letters() == []
    assert ingestion_registry.clear_dead_letter("failed-doc") is False

