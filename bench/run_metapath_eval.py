"""Evaluate finance metapath routing and traversal without Docker/API.

This benchmark is intentionally deterministic. It measures whether the
domain-rule router selects the expected metapath and whether typed graph
traversal reaches the expected end entities.

Run:
    PYTHONPATH=backend python bench/run_metapath_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agents.knowledge_extract_agent import Entity, Relation
from services.knowledge_graph import KnowledgeGraphService
from services.metapaths import FINANCIAL_METAPATHS, MetapathRouter


def _load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


async def _build_graph(dataset: dict[str, Any]) -> KnowledgeGraphService:
    graph = KnowledgeGraphService()
    for item in dataset["entities"]:
        await graph.upsert_entity(Entity(name=item["name"], type=item["type"]))
    for item in dataset["relations"]:
        await graph.add_relation(Relation(
            head=item["head"],
            relation=item["relation"],
            tail=item["tail"],
            confidence=float(item.get("confidence", 1.0)),
        ))
    return graph


async def _evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    graph = await _build_graph(dataset)
    router = MetapathRouter()
    rows: list[dict[str, Any]] = []

    for item in dataset["questions"]:
        selected = router.select(item["question"], item["start_entities"], limit=3)
        selected_names = [spec.name for spec in selected]
        expected_metapath = item["expected_metapath"]
        router_hit = expected_metapath in selected_names

        spec = FINANCIAL_METAPATHS[expected_metapath]
        path_results = await graph.traverse_metapath(item["start_entities"], spec)
        reached = sorted({result.end_entity for result in path_results})
        expected_end_entities = set(item["expected_end_entities"])
        reached_expected = expected_end_entities & set(reached)
        recall = len(reached_expected) / max(len(expected_end_entities), 1)

        rows.append({
            "question": item["question"],
            "expected_metapath": expected_metapath,
            "selected_metapaths": selected_names,
            "router_hit": router_hit,
            "expected_end_entities": sorted(expected_end_entities),
            "reached_end_entities": reached,
            "path_recall": round(recall, 3),
            "path_hit": recall == 1.0,
        })

    total = len(rows)
    router_hits = sum(1 for row in rows if row["router_hit"])
    path_hits = sum(1 for row in rows if row["path_hit"])
    average_recall = sum(float(row["path_recall"]) for row in rows) / max(total, 1)
    return {
        "dataset": "finance_metapath_synthetic",
        "total": total,
        "router_hits": router_hits,
        "router_hit_rate": round(router_hits / max(total, 1), 3),
        "path_hits": path_hits,
        "path_hit_rate": round(path_hits / max(total, 1), 3),
        "average_path_recall": round(average_recall, 3),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path(__file__).with_name("metapath_dataset.json")))
    parser.add_argument("--show-rows", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(_evaluate(_load_dataset(Path(args.dataset))))
    if not args.show_rows:
        result = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
