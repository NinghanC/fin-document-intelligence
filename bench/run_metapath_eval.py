"""Evaluate finance metapath routing and traversal without Docker/API.

This benchmark is intentionally deterministic. It reports two separate path
metrics: routed traversal for end-to-end router quality, and oracle traversal
for validating graph data/traversal when the expected metapath is supplied.

Run:
    python bench/run_metapath_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from agents.knowledge_extract_agent import Entity, Relation  # noqa: E402
from services.knowledge_graph import KnowledgeGraphService  # noqa: E402
from services.metapaths import FINANCIAL_METAPATHS, MetapathRouter  # noqa: E402


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


async def _traverse_selected(
    graph: KnowledgeGraphService,
    start_entities: list[str],
    metapath_names: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    reached_by_metapath: dict[str, list[str]] = {}
    reached: set[str] = set()
    for name in metapath_names:
        paths = await graph.traverse_metapath(start_entities, FINANCIAL_METAPATHS[name])
        end_entities = sorted({result.end_entity for result in paths})
        reached_by_metapath[name] = end_entities
        reached.update(end_entities)
    return sorted(reached), reached_by_metapath


async def _evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    graph = await _build_graph(dataset)
    router = MetapathRouter()
    rows: list[dict[str, Any]] = []

    for item in dataset["questions"]:
        selections = router.select_with_trace(item["question"], item["start_entities"], limit=3)
        selected_names = [selection.spec.name for selection in selections]
        expected_metapath = item["expected_metapath"]
        router_hit = expected_metapath in selected_names
        router_top1_hit = bool(selected_names) and selected_names[0] == expected_metapath
        router_precision = (1.0 / len(selected_names)) if router_hit and selected_names else 0.0
        extra_metapaths = [name for name in selected_names if name != expected_metapath]

        oracle_paths = await graph.traverse_metapath(item["start_entities"], FINANCIAL_METAPATHS[expected_metapath])
        oracle_reached = sorted({result.end_entity for result in oracle_paths})
        routed_reached, reached_by_selected = await _traverse_selected(graph, item["start_entities"], selected_names)
        expected_end_entities = set(item["expected_end_entities"])
        oracle_recall = len(expected_end_entities & set(oracle_reached)) / max(len(expected_end_entities), 1)
        routed_recall = len(expected_end_entities & set(routed_reached)) / max(len(expected_end_entities), 1)

        rows.append({
            "question": item["question"],
            "expected_metapath": expected_metapath,
            "selected_metapaths": selected_names,
            "router_trace": [selection.as_trace() for selection in selections],
            "router_hit": router_hit,
            "router_top1_hit": router_top1_hit,
            "router_precision": round(router_precision, 3),
            "extra_metapaths": extra_metapaths,
            "expected_end_entities": sorted(expected_end_entities),
            "routed_reached_end_entities": routed_reached,
            "oracle_reached_end_entities": oracle_reached,
            "reached_by_selected_metapath": reached_by_selected,
            "routed_path_recall": round(routed_recall, 3),
            "oracle_path_recall": round(oracle_recall, 3),
            "routed_path_hit": routed_recall == 1.0,
            "oracle_path_hit": oracle_recall == 1.0,
        })

    total = len(rows)
    router_hits = sum(1 for row in rows if row["router_hit"])
    router_top1_hits = sum(1 for row in rows if row["router_top1_hit"])
    average_router_precision = sum(float(row["router_precision"]) for row in rows) / max(total, 1)
    average_selected_metapaths = sum(len(row["selected_metapaths"]) for row in rows) / max(total, 1)
    routed_path_hits = sum(1 for row in rows if row["routed_path_hit"])
    oracle_path_hits = sum(1 for row in rows if row["oracle_path_hit"])
    average_routed_recall = sum(float(row["routed_path_recall"]) for row in rows) / max(total, 1)
    average_oracle_recall = sum(float(row["oracle_path_recall"]) for row in rows) / max(total, 1)
    return {
        "dataset": "finance_metapath_synthetic",
        "total": total,
        "router_hits": router_hits,
        "router_hit_rate": round(router_hits / max(total, 1), 3),
        "router_top1_hit_rate": round(router_top1_hits / max(total, 1), 3),
        "average_router_precision": round(average_router_precision, 3),
        "average_selected_metapaths": round(average_selected_metapaths, 3),
        "routed_path_hits": routed_path_hits,
        "routed_path_hit_rate": round(routed_path_hits / max(total, 1), 3),
        "average_routed_path_recall": round(average_routed_recall, 3),
        "oracle_path_hits": oracle_path_hits,
        "oracle_path_hit_rate": round(oracle_path_hits / max(total, 1), 3),
        "average_oracle_path_recall": round(average_oracle_recall, 3),
        "path_hit_rate": round(routed_path_hits / max(total, 1), 3),
        "average_path_recall": round(average_routed_recall, 3),
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
