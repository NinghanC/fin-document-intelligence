"""Request metrics and rate-limit state with optional PostgreSQL persistence."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

logger = structlog.get_logger("finsight.api_state")


class APIStateStore:
    def __init__(
        self,
        *,
        backend: str,
        dsn: str,
        rate_limit_buckets: dict[str, list[float]],
        request_metrics: dict[str, Any],
    ) -> None:
        self.backend = backend.lower()
        self.dsn = dsn
        self._rate_limit_buckets = rate_limit_buckets
        self._request_metrics = request_metrics
        self._engine: Any = None

    async def init(self) -> None:
        if self.backend != "postgres":
            return
        try:
            await asyncio.to_thread(self._init_postgres)
        except Exception as exc:
            logger.warning("api_state_postgres_init_failed_using_memory", error=str(exc))
            self._engine = None

    async def close(self) -> None:
        if self._engine is not None:
            await asyncio.to_thread(self._engine.dispose)
            self._engine = None

    async def allow_request(self, client: str, limit: int, window_seconds: int) -> bool:
        if self.backend == "postgres" and self._engine is not None:
            try:
                return await asyncio.to_thread(self._allow_request_postgres, client, limit, window_seconds)
            except Exception as exc:
                logger.warning("api_state_postgres_rate_limit_failed_using_memory", error=str(exc))
        return self._allow_request_memory(client, limit, window_seconds)

    async def record_request_metric(self, path: str, duration_ms: float) -> None:
        self._record_request_metric_memory(path, duration_ms)
        if self.backend == "postgres" and self._engine is not None:
            try:
                await asyncio.to_thread(self._record_request_metric_postgres, path, duration_ms)
            except Exception as exc:
                logger.warning("api_state_postgres_metric_write_failed", error=str(exc))

    async def get_request_stats(self) -> dict[str, Any]:
        if self.backend == "postgres" and self._engine is not None:
            try:
                return await asyncio.to_thread(self._get_request_stats_postgres)
            except Exception as exc:
                logger.warning("api_state_postgres_stats_failed_using_memory", error=str(exc))
        return self._request_stats_memory()

    def _allow_request_memory(self, client: str, limit: int, window_seconds: int) -> bool:
        now = time.time()
        window_start = now - window_seconds
        bucket = [ts for ts in self._rate_limit_buckets.get(client, []) if ts >= window_start]
        if len(bucket) >= limit:
            self._rate_limit_buckets[client] = bucket
            return False
        bucket.append(now)
        self._rate_limit_buckets[client] = bucket
        return True

    def _record_request_metric_memory(self, path: str, duration_ms: float) -> None:
        self._request_metrics["total_requests"] += 1
        self._request_metrics["total_latency_ms"] += duration_ms
        by_path = self._request_metrics["by_path"]
        path_stats = by_path.setdefault(path, {"count": 0, "total_latency_ms": 0.0, "last_latency_ms": 0.0})
        path_stats["count"] += 1
        path_stats["total_latency_ms"] += duration_ms
        path_stats["last_latency_ms"] = round(duration_ms, 2)

    def _request_stats_memory(self) -> dict[str, Any]:
        total = self._request_metrics["total_requests"]
        by_path = {
            path: {
                "count": stats["count"],
                "avg_latency_ms": round(stats["total_latency_ms"] / max(stats["count"], 1), 2),
                "last_latency_ms": stats["last_latency_ms"],
            }
            for path, stats in self._request_metrics["by_path"].items()
        }
        return {
            "total_requests": total,
            "avg_latency_ms": round(self._request_metrics["total_latency_ms"] / max(total, 1), 2),
            "by_path": by_path,
        }

    def _init_postgres(self) -> None:
        from sqlalchemy import create_engine, text

        self._engine = create_engine(self.dsn, pool_pre_ping=True)
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS api_rate_limit_events (
                    client TEXT NOT NULL,
                    ts DOUBLE PRECISION NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_api_rate_limit_events_client_ts
                ON api_rate_limit_events (client, ts)
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS api_request_metrics (
                    path TEXT PRIMARY KEY,
                    count BIGINT NOT NULL,
                    total_latency_ms DOUBLE PRECISION NOT NULL,
                    last_latency_ms DOUBLE PRECISION NOT NULL
                )
            """))

    def _allow_request_postgres(self, client: str, limit: int, window_seconds: int) -> bool:
        from sqlalchemy import text

        now = time.time()
        window_start = now - window_seconds
        with self._engine.begin() as conn:
            conn.execute(text("DELETE FROM api_rate_limit_events WHERE ts < :window_start"), {"window_start": window_start})
            count = int(conn.execute(
                text("SELECT COUNT(*) FROM api_rate_limit_events WHERE client = :client AND ts >= :window_start"),
                {"client": client, "window_start": window_start},
            ).scalar() or 0)
            if count >= limit:
                return False
            conn.execute(
                text("INSERT INTO api_rate_limit_events (client, ts) VALUES (:client, :ts)"),
                {"client": client, "ts": now},
            )
            return True

    def _record_request_metric_postgres(self, path: str, duration_ms: float) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO api_request_metrics (path, count, total_latency_ms, last_latency_ms)
                    VALUES (:path, 1, :duration_ms, :duration_ms)
                    ON CONFLICT (path) DO UPDATE SET
                        count = api_request_metrics.count + 1,
                        total_latency_ms = api_request_metrics.total_latency_ms + EXCLUDED.total_latency_ms,
                        last_latency_ms = EXCLUDED.last_latency_ms
                """),
                {"path": path, "duration_ms": duration_ms},
            )

    def _get_request_stats_postgres(self) -> dict[str, Any]:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT path, count, total_latency_ms, last_latency_ms
                FROM api_request_metrics
            """)).mappings().all()
        total = sum(int(row["count"]) for row in rows)
        total_latency = sum(float(row["total_latency_ms"]) for row in rows)
        by_path = {
            str(row["path"]): {
                "count": int(row["count"]),
                "avg_latency_ms": round(float(row["total_latency_ms"]) / max(int(row["count"]), 1), 2),
                "last_latency_ms": round(float(row["last_latency_ms"]), 2),
            }
            for row in rows
        }
        return {
            "total_requests": total,
            "avg_latency_ms": round(total_latency / max(total, 1), 2),
            "by_path": by_path,
        }