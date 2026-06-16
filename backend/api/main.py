"""
FastAPI entry point - enterprise knowledge management REST API

Provides three endpoint groups:
  1. /api/ingest   - document upload and ingestion
  2. /api/qa       - intelligent QA
  3. /api/admin    - administration (statistics and update triggers)
"""

from __future__ import annotations

import os
import shutil
import asyncio
from contextlib import asynccontextmanager
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent
from agents.qa_agent import QAAgent
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.cdc_processor import CDCEvent, CDCProcessor
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService

vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
cdc_processor = CDCProcessor()
workflows: dict[str, Any] = {}
background_tasks: list[asyncio.Task] = []


async def _init_with_retry(name: str, init_fn: Callable[[], Awaitable[None]], attempts: int = 10, delay: float = 2.0) -> bool:
    """Initialize a service that may start slightly after the API container."""
    for attempt in range(1, attempts + 1):
        try:
            await init_fn()
            return True
        except Exception:
            if attempt == attempts:
                return False
            await asyncio.sleep(delay)
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)
    await _init_with_retry("vector_store", vector_store.init)
    await _init_with_retry("knowledge_graph", knowledge_graph.init)
    workflows.update(
        build_knowledge_graph_workflow(vector_store=vector_store, knowledge_graph=knowledge_graph)
    )
    if settings.enable_cdc_consumer:
        background_tasks.append(asyncio.create_task(cdc_processor.start_kafka_consumer()))
    yield
    for task in background_tasks:
        task.cancel()
    await knowledge_graph.close()


app = FastAPI(
    title="FinSight Assistant",
    description="Financial document intelligence API with hybrid RAG, knowledge graphs, and incremental updates",
    version="1.0.0",
    lifespan=lifespan,
)

# Static Files & Frontend
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_dir, "index.html"))


# Request / Response Models
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]
    cdc: dict[str, Any] = {}


class UpdateRequest(BaseModel):
    file_path: str
    change_type: str = "modified"


class UpdateResponse(BaseModel):
    file_path: str
    vectors_added: int
    vectors_deleted: int
    entities_added: int
    relations_added: int
    success: bool
    processing_time_ms: float


class CDCEventRequest(BaseModel):
    operation: str = "UPDATE"
    resource_path: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


# Ingest Endpoints
@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["Document Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and parse a document, then automatically ingest it into the vector store and knowledge graph"""
    save_path = os.path.join(settings.upload_dir, file.filename or "unknown")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")

    result = await ingest_wf.ainvoke({"file_paths": [save_path]})
    chunks = result.get("chunks", [])
    extractions = result.get("extractions", [])
    total_entities = sum(len(e.entities) for e in extractions)
    total_relations = sum(len(e.relations) for e in extractions)

    return IngestResponse(
        file_name=file.filename or "unknown",
        chunks_count=len(chunks),
        entities_count=total_entities,
        relations_count=total_relations,
        status="success",
    )


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["Document Ingestion"])
async def upload_batch(files: list[UploadFile] = File(...)):
    """Upload documents in a batch"""
    results = []
    for file in files:
        resp = await upload_document(file)
        results.append(resp)
    return results


# QA Endpoints
@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["intelligent QA"])
async def ask_question(req: QuestionRequest):
    """Intelligent QA - hybrid retrieval + knowledge graph reasoning"""
    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    result = await qa_wf.ainvoke({"question": req.question})
    qa_result = result.get("result")
    if not qa_result:
        raise HTTPException(status_code=500, detail="QA failed")

    return QuestionResponse(
        question=qa_result.question,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        sources=[
            {"content": c.content[:200], "source": c.source, "score": c.score, "type": c.retrieval_type}
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
    )


# Admin Endpoints
@app.get("/api/admin/stats", response_model=StatsResponse, tags=["System Administration"])
async def get_stats():
    """Get system statistics"""
    try:
        vs_stats = await vector_store.get_stats()
    except Exception:
        vs_stats = {"backend": "chroma", "total_vectors": 0}
    try:
        kg_stats = await knowledge_graph.get_stats()
    except Exception:
        kg_stats = {"total_entities": 0, "total_relations": 0}
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats, cdc=cdc_processor.get_stats())


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["System Administration"])
async def trigger_update(req: UpdateRequest):
    """Manually trigger a knowledge update"""
    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="Update workflow not initialized")

    change = DocumentChange(
        file_path=req.file_path,
        change_type=ChangeType(req.change_type),
    )
    result = await update_wf.ainvoke({"changes": [change]})
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="Update failed")

    r = results[0]
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


@app.post("/api/admin/cdc/events", tags=["System Administration"])
async def process_cdc_event(req: CDCEventRequest):
    """Process a normalized CDC event through the incremental CDC processor."""
    event = CDCEvent(
        event_id=f"api-{req.resource_path}-{req.operation}",
        source_type="api",
        operation=req.operation.upper(),
        resource_path=req.resource_path,
        before=req.before,
        after=req.after,
    )
    result = await cdc_processor.process_event(event)
    return {
        "success": result.success,
        "event_id": result.event.event_id,
        "version": result.version,
        "chunks_affected": result.chunks_affected,
        "entities_affected": result.entities_affected,
        "processing_time_ms": result.processing_time_ms,
        "diff": result.event.diff,
        "error": result.error,
    }


@app.get("/api/health", tags=["System Administration"])
async def health():
    return {"status": "ok", "name": "FinSight Assistant"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
