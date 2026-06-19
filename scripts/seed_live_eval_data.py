"""Seed provider-backed live-eval data into the running local stack.

This script is intentionally separate from CI. It prepares the public demo graph
and table fixture used by bench/run_live_eval.py against Docker Compose.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bootstrap_helpers import load_env_file  # noqa: E402

BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _load_env(path: Path) -> None:
    load_env_file(path)


async def _seed_graph(dataset_path: Path, neo4j_uri: str) -> dict[str, int]:
    os.environ["NEO4J_URI"] = neo4j_uri

    from agents.knowledge_extract_agent import Entity, Relation
    from services.knowledge_graph import KnowledgeGraphService

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    graph = KnowledgeGraphService()
    await graph.init()
    try:
        for item in dataset.get("entities", []):
            await graph.upsert_entity(
                Entity(
                    name=item["name"],
                    type=item.get("type", "Concept"),
                    description=f"Live-eval seed entity: {item['name']}",
                    confidence=1.0,
                ),
                source=str(dataset_path),
            )
        for item in dataset.get("relations", []):
            await graph.add_relation(
                Relation(
                    head=item["head"],
                    relation=item["relation"],
                    tail=item["tail"],
                    confidence=1.0,
                ),
                source=str(dataset_path),
            )
        communities = await graph.refresh_community_summaries()
        return {
            "entities": len(dataset.get("entities", [])),
            "relations": len(dataset.get("relations", [])),
            "communities": communities,
        }
    finally:
        await graph.close()


async def _upload_fixture(base_url: str, api_key: str, csv_path: Path) -> dict[str, Any]:
    headers = {"X-API-Key": api_key} if api_key else {}
    with csv_path.open("rb") as handle:
        files = {"file": (csv_path.name, handle, "text/csv")}
        async with httpx.AsyncClient(base_url=base_url, timeout=120, headers=headers) as client:
            response = await client.post("/api/ingest/upload", files=files)
            response.raise_for_status()
            return response.json()


async def _main_async(args: argparse.Namespace) -> None:
    _load_env(Path(args.env_file))
    graph_result = await _seed_graph(Path(args.dataset), args.neo4j_uri)
    upload_result = await _upload_fixture(args.base_url, args.api_key, Path(args.csv))
    print(json.dumps({"graph": graph_result, "upload": upload_result}, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(BACKEND / ".env"))
    parser.add_argument("--dataset", default=str(ROOT / "bench" / "metapath_dataset.json"))
    parser.add_argument("--csv", default=str(ROOT / "bench" / "live_eval" / "fund_exposures.csv"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "dev-local-secret"))
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:7687")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()