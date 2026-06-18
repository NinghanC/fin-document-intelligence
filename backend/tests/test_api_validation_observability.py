import io

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import api.main as api_main
from services.api_state import APIStateStore


def make_upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content))


@pytest.fixture(autouse=True)
def reset_api_state(monkeypatch):
    monkeypatch.setattr(api_main.settings, "auth_enabled", False)
    monkeypatch.setattr(api_main.settings, "rate_limit_requests", 60)
    monkeypatch.setattr(api_main.settings, "rate_limit_window_seconds", 60)
    api_main._rate_limit_buckets.clear()
    api_main._request_metrics["total_requests"] = 0
    api_main._request_metrics["total_latency_ms"] = 0.0
    api_main._request_metrics["by_path"] = {}


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        await api_main._save_upload(make_upload("fund.exe", b"not allowed"))

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        await api_main._save_upload(make_upload("empty.txt", b""))

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_non_utf8_text(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        await api_main._save_upload(make_upload("fund.txt", b"\xff\xfe\x00"))

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_accepts_png_magic_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    safe_name, _ = await api_main._save_upload(make_upload("chart.png", b"\x89PNG\r\n\x1a\npayload"))

    assert safe_name == "chart.png"


@pytest.mark.asyncio
async def test_upload_rejects_invalid_jpeg_magic_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        await api_main._save_upload(make_upload("chart.jpg", b"bad jpeg"))

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_accepts_xlsx_zip_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    safe_name, _ = await api_main._save_upload(make_upload("holdings.xlsx", b"PK\x03\x04payload"))

    assert safe_name == "holdings.xlsx"


@pytest.mark.asyncio
async def test_upload_rejects_invalid_xls_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.settings, "upload_dir", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        await api_main._save_upload(make_upload("holdings.xls", b"bad xls"))

    assert exc.value.status_code == 400


def test_request_metric_recorder_tracks_average_latency():
    api_main._record_request_metric("/api/qa/ask", 10.0)
    api_main._record_request_metric("/api/qa/ask", 30.0)

    stats = api_main._request_stats()

    assert stats["total_requests"] == 2
    assert stats["avg_latency_ms"] == 20.0
    assert stats["by_path"]["/api/qa/ask"]["avg_latency_ms"] == 20.0


def test_source_excerpt_centers_on_financial_metric_terms():
    content = (
        "Selected ratios and metrics Return on common equity 17 % Return on tangible common equity 21 % "
        "Return on assets 1.30 Overhead ratio 55 Loans-to-deposits ratio 55 Firm "
        "Liquidity coverage ratio (average) 113 112 111 JPMorgan Chase Bank liquidity coverage ratio 148."
    )

    excerpt = api_main._source_excerpt(
        content,
        "What liquidity coverage ratio did JPMorgan Chase report for 2023?",
        max_chars=120,
    )

    assert "Liquidity coverage ratio" in excerpt
    assert "113" in excerpt
    assert excerpt.startswith("...")


def test_health_response_includes_request_id_header(monkeypatch):
    async def skip_init(init_fn, attempts=10, delay=2.0):
        return True

    monkeypatch.setattr(api_main, "_init_with_retry", skip_init)

    with TestClient(api_main.app) as client:
        response = client.get("/api/health", headers={"X-Request-ID": "req-test"})

    assert response.headers["X-Request-ID"] == "req-test"
    assert response.json()["name"] == "FinSight Assistant"


def test_rate_limit_returns_429(monkeypatch):
    async def skip_init(init_fn, attempts=10, delay=2.0):
        return True

    monkeypatch.setattr(api_main, "_init_with_retry", skip_init)
    monkeypatch.setattr(api_main.settings, "rate_limit_requests", 1)

    with TestClient(api_main.app) as client:
        first = client.get("/api/health")
        second = client.get("/api/health")

    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.asyncio
async def test_api_state_store_memory_tracks_rate_limits_and_metrics():
    buckets: dict[str, list[float]] = {}
    metrics = {"total_requests": 0, "total_latency_ms": 0.0, "by_path": {}}
    store = APIStateStore(
        backend="memory",
        dsn="",
        rate_limit_buckets=buckets,
        request_metrics=metrics,
    )

    assert await store.allow_request("client", limit=1, window_seconds=60) is True
    assert await store.allow_request("client", limit=1, window_seconds=60) is False

    await store.record_request_metric("/api/qa/ask", 25.0)
    stats = await store.get_request_stats()

    assert stats["total_requests"] == 1
    assert stats["by_path"]["/api/qa/ask"]["avg_latency_ms"] == 25.0


def test_admin_stats_includes_api_metrics(monkeypatch):
    async def skip_init(init_fn, attempts=10, delay=2.0):
        return True

    monkeypatch.setattr(api_main, "_init_with_retry", skip_init)

    with TestClient(api_main.app) as client:
        client.get("/api/health")
        response = client.get("/api/admin/stats")

    assert response.status_code == 200
    assert "api" in response.json()
    assert "ingestion" in response.json()
    assert "dead_letters" in response.json()["ingestion"]
    assert response.json()["api"]["total_requests"] >= 1
