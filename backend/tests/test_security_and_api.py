import asyncio
import io
import time

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import api.main as api_main
from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from config.settings import Settings
from orchestrator.graph import _build_ingest_graph
from services.ingestion_registry import ingestion_registry
from services.knowledge_graph import KnowledgeGraphService


def make_upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content))


@pytest.mark.asyncio
async def test_upload_strips_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    safe_name, save_path = await api_main._save_upload(make_upload("../evil.txt", b"safe text"))

    assert safe_name == "evil.txt"
    assert save_path == str(tmp_path / "evil.txt")
    assert (tmp_path / "evil.txt").read_text(encoding="utf-8") == "safe text"


@pytest.mark.asyncio
async def test_upload_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(api_main.settings, "max_upload_size_mb", 1)

    with pytest.raises(HTTPException) as exc:
        await api_main._save_upload(make_upload("large.txt", b"a" * (1024 * 1024 + 1)))

    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_upload_rejects_invalid_magic_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        await api_main._save_upload(make_upload("fake.pdf", b"not a pdf"))

    assert exc.value.status_code == 400
    assert "PDF" in exc.value.detail


@pytest.mark.asyncio
async def test_api_key_dependency(monkeypatch):
    monkeypatch.setattr(api_main.settings, "auth_enabled", True)
    monkeypatch.setattr(api_main.settings, "api_key", "secret")

    with pytest.raises(HTTPException) as exc:
        await api_main.require_api_key(None)

    assert exc.value.status_code == 401
    await api_main.require_api_key("secret")


@pytest.mark.asyncio
async def test_api_key_dependency_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(api_main.settings, "auth_enabled", False)
    await api_main.require_api_key(None)


@pytest.mark.asyncio
async def test_read_only_cypher_rejects_writes():
    graph = KnowledgeGraphService()

    with pytest.raises(ValueError):
        await graph.execute_cypher("MATCH (n) DELETE n")


def test_relation_type_sanitization_blocks_injection():
    assert KnowledgeGraphService._sanitize_rel_type("related_to") == "RELATED_TO"
    assert KnowledgeGraphService._sanitize_rel_type("holds") == "HOLDS"
    assert KnowledgeGraphService._sanitize_rel_type("REL] DELETE n //") == "RELATED_TO"
    assert KnowledgeGraphService._sanitize_rel_type("looks_safe_but_unknown") == "RELATED_TO"


def test_settings_enable_auth_by_default():
    settings = Settings(_env_file=None)

    assert settings.auth_enabled is True
    assert settings.api_key == ""


@pytest.mark.asyncio
async def test_enabled_auth_without_configured_key_rejects_requests(monkeypatch):
    monkeypatch.setattr(api_main.settings, "auth_enabled", True)
    monkeypatch.setattr(api_main.settings, "api_key", "")

    with pytest.raises(HTTPException) as exc:
        await api_main.require_api_key("anything")

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_parallel_ingest_store_node_runs_vector_and_graph_work(tmp_path, monkeypatch):
    source = tmp_path / "fund.txt"
    source.write_text("Global Income Fund duration risk", encoding="utf-8")
    monkeypatch.setattr("services.ingestion_registry.settings.upload_dir", str(tmp_path))
    ingestion_registry._records = {}

    class Parser:
        async def parse_batch(self, file_paths):
            return [
                DocumentChunk(
                    content="Global Income Fund duration risk",
                    doc_id="doc",
                    chunk_index=0,
                    doc_type=DocType.TEXT,
                    metadata={"source": "fund.txt"},
                )
            ]

    class Extractor:
        async def extract(self, chunks):
            return [
                ExtractionResult(
                    entities=[Entity(name="Global Income Fund", type="Product")],
                    relations=[Relation(head="Global Income Fund", relation="related_to", tail="duration risk")],
                    events=[],
                    source_chunk_id="doc#chunk-0",
                )
            ]

    class VectorStore:
        @property
        def embeddings_available(self):
            return True

        async def add_chunks(self, chunks):
            await asyncio_sleep()
            return len(chunks)

    class Graph:
        def __init__(self):
            self.entities = 0
            self.relations = 0

        async def upsert_entity(self, entity, source=""):
            await asyncio_sleep()
            self.entities += 1

        async def add_relation(self, relation, source=""):
            self.relations += 1

    async def asyncio_sleep():
        import asyncio
        await asyncio.sleep(0.05)

    graph_service = Graph()
    workflow = _build_ingest_graph(Parser(), Extractor(), VectorStore(), graph_service)

    start = time.perf_counter()
    result = await workflow.ainvoke({"file_paths": [str(source)]})
    elapsed = time.perf_counter() - start

    assert result["vectors_stored"] == 1
    assert result["entities_stored"] == 1
    assert graph_service.relations == 1
    assert elapsed < 0.15


def test_upload_endpoint_returns_ingest_response(tmp_path, monkeypatch):
    class Workflow:
        async def ainvoke(self, state):
            return {
                "chunks": [
                    DocumentChunk(
                        content="Global Income Fund",
                        doc_id="doc",
                        chunk_index=0,
                        doc_type=DocType.TEXT,
                        metadata={"source": state["file_paths"][0]},
                    )
                ],
                "extractions": [
                    ExtractionResult(
                        entities=[Entity(name="Global Income Fund", type="Product")],
                        relations=[],
                        events=[],
                        source_chunk_id="doc#chunk-0",
                    )
                ],
            }

    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(api_main.settings, "auth_enabled", False)
    async def skip_init(init_fn, attempts=10, delay=2.0):
        return True

    monkeypatch.setattr(api_main, "_init_with_retry", skip_init)
    api_main._rate_limit_buckets.clear()
    api_main.workflows["ingest"] = Workflow()

    with TestClient(api_main.app) as client:
        response = client.post(
            "/api/ingest/upload",
            files={"file": ("../fund.txt", b"Global Income Fund", "text/plain")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["file_name"] == "fund.txt"
    assert body["chunks_count"] == 1
    assert body["entities_count"] == 1
    assert (tmp_path / "fund.txt").exists()


@pytest.mark.asyncio
async def test_upload_batch_uses_bounded_concurrency(monkeypatch):
    calls = 0
    max_active = 0
    active = 0

    async def fake_upload(file):
        nonlocal calls, max_active, active
        calls += 1
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return api_main.IngestResponse(
            file_name=file.filename,
            chunks_count=1,
            entities_count=0,
            relations_count=0,
            status="success",
        )

    monkeypatch.setattr(api_main.settings, "batch_upload_concurrency", 2)
    monkeypatch.setattr(api_main, "upload_document", fake_upload)

    files = [make_upload(f"fund-{idx}.txt", b"Global Income Fund") for idx in range(4)]
    start = time.perf_counter()
    results = await api_main.upload_batch(files)
    elapsed = time.perf_counter() - start

    assert calls == 4
    assert len(results) == 4
    assert max_active == 2
    assert elapsed < 0.16

@pytest.mark.asyncio
async def test_upload_batch_returns_per_file_failure_for_http_errors(monkeypatch):
    async def fake_upload(file):
        if file.filename == "bad.pdf":
            raise HTTPException(status_code=400, detail="Invalid PDF signature")
        return api_main.IngestResponse(
            file_name=file.filename,
            chunks_count=1,
            entities_count=0,
            relations_count=0,
            status="success",
        )

    monkeypatch.setattr(api_main.settings, "batch_upload_concurrency", 2)
    monkeypatch.setattr(api_main, "upload_document", fake_upload)

    results = await api_main.upload_batch([
        make_upload("good.txt", b"Global Income Fund"),
        make_upload("bad.pdf", b"not a pdf"),
    ])

    assert [result.status for result in results] == ["success", "failed"]
    assert results[1].file_name == "bad.pdf"
    assert results[1].error == "Invalid PDF signature"


@pytest.mark.asyncio
async def test_upload_batch_returns_generic_failure_for_unexpected_errors(monkeypatch):
    async def fake_upload(file):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(api_main.settings, "batch_upload_concurrency", 1)
    monkeypatch.setattr(api_main, "upload_document", fake_upload)

    results = await api_main.upload_batch([make_upload("fund.txt", b"Global Income Fund")])

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].file_name == "fund.txt"
    assert results[0].error == "Ingestion failed"
