"""
FastAPI entry point - enterprise knowledge management REST API

Provides three endpoint groups:
  1. /api/ingest   - document upload and ingestion
  2. /api/qa       - intelligent QA
  3. /api/admin    - administration (statistics and update triggers)
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.knowledge_update_agent import ChangeType, DocumentChange
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.api_state import APIStateStore
from services.cdc_processor import CDCEvent, CDCProcessor
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService

vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
cdc_processor = CDCProcessor()
workflows: dict[str, Any] = {}
background_tasks: list[asyncio.Task] = []
_rate_limit_buckets: dict[str, list[float]] = {}
_request_metrics: dict[str, Any] = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "by_path": {},
}
api_state_store = APIStateStore(
    backend=settings.api_state_backend,
    dsn=settings.api_state_dsn,
    rate_limit_buckets=_rate_limit_buckets,
    request_metrics=_request_metrics,
)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger("finsight.api")

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".csv", ".xlsx", ".xls", ".txt", ".md"}


async def _init_with_retry(init_fn: Callable[[], Awaitable[None]], attempts: int = 10, delay: float = 2.0) -> bool:
    """Initialize a service that may start slightly after the API container."""
    for attempt in range(1, attempts + 1):
        try:
            await init_fn()
            return True
        except Exception as exc:
            logger.warning("service_init_retry_failed", attempt=attempt, error=str(exc))
            if attempt == attempts:
                return False
            await asyncio.sleep(delay)
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)
    await api_state_store.init()
    await _init_with_retry(vector_store.init)
    await _init_with_retry(knowledge_graph.init)
    if not workflows:
        workflows.update(
            build_knowledge_graph_workflow(vector_store=vector_store, knowledge_graph=knowledge_graph)
        )
    update_wf = workflows.get("update")
    if update_wf:
        async def _process_cdc_change(change: DocumentChange) -> Any:
            result = await update_wf.ainvoke({"changes": [change]})
            results = result.get("results", [])
            return results[0] if results else None

        cdc_processor.set_update_handler(_process_cdc_change)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception("request_failed", duration_ms=round(duration_ms, 2), error=str(exc))
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    await api_state_store.record_request_metric(request.url.path, duration_ms)
    logger.info("request_completed", status_code=response.status_code, duration_ms=round(duration_ms, 2))
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/static"):
        return await call_next(request)

    client = request.client.host if request.client else "unknown"
    allowed = await api_state_store.allow_request(
        client,
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    )
    if not allowed:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    return await call_next(request)


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not settings.auth_enabled:
        return
    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _record_request_metric(path: str, duration_ms: float) -> None:
    api_state_store._record_request_metric_memory(path, duration_ms)


def _request_stats() -> dict[str, Any]:
    return api_state_store._request_stats_memory()


def _source_excerpt(content: str, question: str, max_chars: int = 360) -> str:
    """Return a source preview centered on question-specific terms."""
    if len(content) <= max_chars:
        return content
    tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", question.lower())
        if len(token) >= 3 and token not in {"and", "did", "for", "the", "their", "what", "which"}
    }
    broad_tokens = {"2021", "2022", "2023", "annual", "chase", "fiscal", "jpmorgan", "microsoft", "report", "year"}
    generic_tokens = {"major", "ratio", "source", "sources"}
    focused_tokens = (tokens - broad_tokens - generic_tokens) or (tokens - broad_tokens) or tokens
    lowered = content.lower()
    positions = [lowered.find(token) for token in focused_tokens if lowered.find(token) >= 0]
    if not positions:
        return content[:max_chars]
    start = max(min(positions) - 80, 0)
    end = min(start + max_chars, len(content))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    return f"{prefix}{content[start:end].strip()}{suffix}"


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
    retrieval_quality: float
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
    cdc: dict[str, Any] = Field(default_factory=dict)
    api: dict[str, Any] = Field(default_factory=dict)


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


def _validate_upload_content(filename: str, content: bytes) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'none'}")

    if ext == ".pdf" and not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF signature")
    if ext == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=400, detail="Invalid PNG signature")
    if ext in {".jpg", ".jpeg"} and not content.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=400, detail="Invalid JPEG signature")
    if ext == ".xlsx" and not content.startswith(b"PK"):
        raise HTTPException(status_code=400, detail="Invalid XLSX signature")
    if ext == ".xls" and not content.startswith(b"\xd0\xcf\x11\xe0"):
        raise HTTPException(status_code=400, detail="Invalid XLS signature")
    if ext in {".txt", ".md", ".csv"}:
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Text uploads must be UTF-8") from exc


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    safe_name = Path(file.filename or "unknown").name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_size_mb}MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    _validate_upload_content(safe_name, content)

    os.makedirs(settings.upload_dir, exist_ok=True)
    save_path = os.path.abspath(os.path.join(settings.upload_dir, safe_name))
    upload_root = os.path.abspath(settings.upload_dir)
    if not save_path.startswith(upload_root + os.sep):
        raise HTTPException(status_code=400, detail="Invalid upload path")

    with open(save_path, "wb") as f:
        f.write(content)
    return safe_name, save_path


# Ingest Endpoints
@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["Document Ingestion"], dependencies=[Depends(require_api_key)])
async def upload_document(file: UploadFile = File(...)):
    """Upload and parse a document, then automatically ingest it into the vector store and knowledge graph"""
    safe_name, save_path = await _save_upload(file)

    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")

    result = await ingest_wf.ainvoke({"file_paths": [save_path]})
    chunks = result.get("chunks", [])
    extractions = result.get("extractions", [])
    total_entities = sum(len(e.entities) for e in extractions)
    total_relations = sum(len(e.relations) for e in extractions)

    return IngestResponse(
        file_name=safe_name,
        chunks_count=len(chunks),
        entities_count=total_entities,
        relations_count=total_relations,
        status="success",
    )


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["Document Ingestion"], dependencies=[Depends(require_api_key)])
async def upload_batch(files: list[UploadFile] = File(...)):
    """Upload documents in a bounded-concurrency batch."""
    semaphore = asyncio.Semaphore(max(settings.batch_upload_concurrency, 1))

    async def _upload_one(file: UploadFile) -> IngestResponse:
        async with semaphore:
            return await upload_document(file)

    return await asyncio.gather(*(_upload_one(file) for file in files))


# QA Endpoints
@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["intelligent QA"], dependencies=[Depends(require_api_key)])
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
        retrieval_quality=qa_result.retrieval_quality,
        intent=qa_result.intent.value,
        sources=[
            {
                "content": _source_excerpt(c.content, req.question),
                "source": c.source,
                "score": c.score,
                "type": c.retrieval_type,
            }
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
    )


# Admin Endpoints
@app.get("/api/admin/stats", response_model=StatsResponse, tags=["System Administration"], dependencies=[Depends(require_api_key)])
async def get_stats():
    """Get system statistics"""
    try:
        vs_stats = await vector_store.get_stats()
    except Exception as exc:
        logger.warning("vector_store_stats_failed", error=str(exc))
        vs_stats = {"backend": "chroma", "total_vectors": 0}
    try:
        kg_stats = await knowledge_graph.get_stats()
    except Exception as exc:
        logger.warning("knowledge_graph_stats_failed", error=str(exc))
        kg_stats = {"total_entities": 0, "total_relations": 0}
    return StatsResponse(
        vector_store=vs_stats,
        knowledge_graph=kg_stats,
        cdc=cdc_processor.get_stats(),
        api=await api_state_store.get_request_stats(),
    )


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["System Administration"], dependencies=[Depends(require_api_key)])
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


@app.post("/api/admin/cdc/events", tags=["System Administration"], dependencies=[Depends(require_api_key)])
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
        "update_result": result.update_result,
        "error": result.error,
    }


@app.get("/api/health", tags=["System Administration"])
async def health():
    services: dict[str, Any] = {}
    status = "ok"
    try:
        services["vector_store"] = await vector_store.get_stats()
    except Exception as exc:
        logger.warning("health_vector_store_failed", error=str(exc))
        services["vector_store"] = {"status": "error", "error": str(exc)}
        status = "degraded"
    try:
        services["knowledge_graph"] = await knowledge_graph.get_stats()
    except Exception as exc:
        logger.warning("health_knowledge_graph_failed", error=str(exc))
        services["knowledge_graph"] = {"status": "error", "error": str(exc)}
        status = "degraded"
    services["cdc"] = cdc_processor.get_stats()
    return {"status": status, "name": "FinSight Assistant", "services": services}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
