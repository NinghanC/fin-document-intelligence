"""
LangGraph orchestration engine - hybrid orchestration for 4 agents

Orchestration patterns:
  1. Document ingestion flow: DocParser -> KnowledgeExtract -> (VectorStore + KnowledgeGraph)
  2. QA flow: Query -> QA Agent -> (VectorRetrieval || GraphRetrieval) -> Answer
  3. Incremental update flow: CDC Event -> UpdateAgent -> (Diff -> Parse -> Store)

Uses LangGraph StateGraph for directed graph orchestration with conditional routing and parallel branches
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.doc_parser_agent import DocParserAgent, DocumentChunk
from agents.knowledge_extract_agent import ExtractionResult, KnowledgeExtractAgent
from agents.knowledge_update_agent import (
    ChangeType,
    DocumentChange,
    KnowledgeUpdateAgent,
    UpdateResult,
)
from agents.qa_agent import QAAgent, QAResult
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService


class WorkflowType(str, Enum):
    INGEST = "ingest"
    QA = "qa"
    UPDATE = "update"


# State Schemas
class IngestState(dict):
    """Document Ingestionflow state"""
    file_paths: list[str]
    chunks: list[DocumentChunk]
    extractions: list[ExtractionResult]
    vectors_stored: int
    entities_stored: int
    messages: Annotated[list, add_messages]


class QAState(dict):
    """QA flow state"""
    question: str
    result: QAResult | None
    messages: Annotated[list, add_messages]


class UpdateState(dict):
    """Incremental update flow state"""
    changes: list[DocumentChange]
    results: list[UpdateResult]
    messages: Annotated[list, add_messages]


# Workflow Builder
def build_knowledge_graph_workflow(
    vector_store: VectorStoreService | None = None,
    knowledge_graph: KnowledgeGraphService | None = None,
) -> dict[str, Any]:
    """
    Build three orchestration pipelines and return {"ingest": graph, "qa": graph, "update": graph}
    """
    doc_parser = DocParserAgent()
    extractor = KnowledgeExtractAgent()
    qa_agent = QAAgent(vector_store=vector_store, knowledge_graph=knowledge_graph)
    update_agent = KnowledgeUpdateAgent(
        doc_parser=doc_parser,
        knowledge_extractor=extractor,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )

    return {
        "ingest": _build_ingest_graph(doc_parser, extractor, vector_store, knowledge_graph),
        "qa": _build_qa_graph(qa_agent),
        "update": _build_update_graph(update_agent),
    }


# Ingest Pipeline
def _build_ingest_graph(
    doc_parser: DocParserAgent,
    extractor: KnowledgeExtractAgent,
    vector_store: VectorStoreService | None,
    knowledge_graph: KnowledgeGraphService | None,
) -> StateGraph:

    async def parse_documents(state: dict) -> dict:
        file_paths = state.get("file_paths", [])
        chunks = await doc_parser.parse_batch(file_paths)
        return {**state, "chunks": chunks}

    async def extract_knowledge(state: dict) -> dict:
        chunks = state.get("chunks", [])
        extractions = await extractor.extract(chunks)
        return {**state, "extractions": extractions}

    async def store_vectors(state: dict) -> dict:
        chunks = state.get("chunks", [])
        count = 0
        if vector_store and chunks:
            try:
                if vector_store.embeddings_available:
                    count = await vector_store.add_chunks(chunks)
            except Exception:
                pass
        return {**state, "vectors_stored": count}

    async def store_graph(state: dict) -> dict:
        extractions = state.get("extractions", [])
        entity_count = 0
        if knowledge_graph:
            try:
                for ext in extractions:
                    for ent in ext.entities:
                        await knowledge_graph.upsert_entity(ent, source=ext.source_chunk_id)
                        entity_count += 1
                    for rel in ext.relations:
                        await knowledge_graph.add_relation(rel, source=ext.source_chunk_id)
            except Exception:
                pass
        return {**state, "entities_stored": entity_count}

    graph = StateGraph(dict)
    graph.add_node("parse", parse_documents)
    graph.add_node("extract", extract_knowledge)
    graph.add_node("store_vectors", store_vectors)
    graph.add_node("store_graph", store_graph)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "store_vectors")
    graph.add_edge("store_vectors", "store_graph")
    graph.add_edge("store_graph", END)

    return graph.compile()


# QA Pipeline
def _build_qa_graph(qa_agent: QAAgent) -> StateGraph:

    async def process_question(state: dict) -> dict:
        question = state.get("question", "")
        result = await qa_agent.answer(question)
        return {**state, "result": result}

    graph = StateGraph(dict)
    graph.add_node("answer", process_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)

    return graph.compile()


# Update Pipeline
def _build_update_graph(update_agent: KnowledgeUpdateAgent) -> StateGraph:

    async def process_updates(state: dict) -> dict:
        changes = state.get("changes", [])
        results = await update_agent.process_batch(changes)
        return {**state, "results": results}

    def should_continue(state: dict) -> str:
        results = state.get("results", [])
        failed = [r for r in results if not r.success]
        if failed:
            return "retry"
        return "done"

    async def retry_failed(state: dict) -> dict:
        results = state.get("results", [])
        failed_changes = [r.change for r in results if not r.success]
        retried = await update_agent.process_batch(failed_changes)
        all_results = [r for r in results if r.success] + retried
        return {**state, "results": all_results}

    graph = StateGraph(dict)
    graph.add_node("process", process_updates)
    graph.add_node("retry", retry_failed)

    graph.set_entry_point("process")
    graph.add_conditional_edges("process", should_continue, {"retry": "retry", "done": END})
    graph.add_edge("retry", END)

    return graph.compile()
