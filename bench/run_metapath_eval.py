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
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.metapaths import FINANCIAL_METAPATHS, MetapathResult, MetapathRouter, MetapathSpec  # noqa: E402


def _load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_questions(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


class _EvalGraph:
    def __init__(self) -> None:
        self.entities: dict[str, str] = {}
        self.relations: list[tuple[str, str, str]] = []

    async def upsert_entity(self, name: str, entity_type: str) -> None:
        self.entities[name] = entity_type

    async def add_relation(self, head: str, relation: str, tail: str) -> None:
        self.relations.append((head, relation.lower(), tail))

    async def traverse_metapath(
        self,
        start_entities: list[str],
        metapath: MetapathSpec,
        limit: int = 20,
    ) -> list[MetapathResult]:
        results: list[MetapathResult] = []
        for start in start_entities:
            states: list[tuple[str, list[tuple[str, str, str]]]] = [(start, [])]
            for step in metapath.steps:
                next_states: list[tuple[str, list[tuple[str, str, str]]]] = []
                for current, path in states:
                    if self.entities.get(current) != step.from_type:
                        continue
                    for head, relation, tail in self.relations:
                        if relation != step.relation:
                            continue
                        if step.direction == "out" and head == current and self.entities.get(tail) == step.to_type:
                            next_states.append((tail, [*path, (head, relation.upper(), tail)]))
                        elif step.direction == "in" and tail == current and self.entities.get(head) == step.to_type:
                            next_states.append((head, [*path, (tail, relation.upper(), head)]))
                states = next_states
                if not states:
                    break
            for end, path in states:
                results.append(MetapathResult(
                    metapath_name=metapath.name,
                    start_entity=start,
                    end_entity=end,
                    path=tuple(path),
                    evidence=" -> ".join(f"{head}-[{rel}]->{tail}" for head, rel, tail in path),
                    score=1.0,
                ))
                if len(results) >= limit:
                    return results
        return results


async def _build_graph(dataset: dict[str, Any] | list[dict[str, str]]) -> _EvalGraph:
    graph = _EvalGraph()
    if isinstance(dataset, list):
        managers = sorted({row["manager"] for row in dataset})
        companies = sorted({row["company"] for row in dataset})
        sectors = sorted({row["sector"] for row in dataset})
        regions = sorted({row["region"] for row in dataset})

        for manager in managers:
            await graph.upsert_entity(manager, "Fund")
        for company in companies:
            await graph.upsert_entity(company, "Company")
        for sector in sectors:
            await graph.upsert_entity(sector, "Sector")
        for region in regions:
            await graph.upsert_entity(region, "Region")

        seen_relations: set[tuple[str, str, str]] = set()
        for row in dataset:
            for head, relation, tail in (
                (row["manager"], "holds", row["company"]),
                (row["company"], "belongs_to", row["sector"]),
                (row["company"], "located_in", row["region"]),
            ):
                key = (head, relation, tail)
                if key not in seen_relations:
                    seen_relations.add(key)
                    await graph.add_relation(head, relation, tail)
        return graph

    for item in dataset["entities"]:
        await graph.upsert_entity(item["name"], item["type"])
    for item in dataset["relations"]:
        await graph.add_relation(item["head"], item["relation"], item["tail"])
    return graph


async def _traverse_selected(
    graph: _EvalGraph,
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


async def _evaluate_questions(
    graph_data: dict[str, Any] | list[dict[str, str]],
    questions: list[dict[str, Any]],
    result_prefix: dict[str, Any],
) -> dict[str, Any]:
    graph = await _build_graph(graph_data)
    router = MetapathRouter()
    rows: list[dict[str, Any]] = []

    for item in questions:
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
        **result_prefix,
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


async def _evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    return await _evaluate_questions(
        dataset,
        dataset["questions"],
        {"dataset": "finance_metapath_synthetic", "total": len(dataset["questions"])},
    )


async def _evaluate_real_holdings(rows: list[dict[str, str]], questions: list[dict[str, Any]]) -> dict[str, Any]:
    return await _evaluate_questions(
        rows,
        questions,
        {"dataset": "real_13f_style_holdings_sample", "holdings_rows": len(rows), "questions": len(questions)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).with_name("real_holdings")
    parser.add_argument("--kind", choices=["synthetic", "real-holdings"], default="synthetic")
    parser.add_argument("--dataset", default=str(Path(__file__).with_name("metapath_dataset.json")))
    parser.add_argument("--holdings", default=str(root / "holdings_sample.csv"))
    parser.add_argument("--questions", default=str(root / "questions.json"))
    parser.add_argument("--show-rows", action="store_true")
    args = parser.parse_args()

    if args.kind == "real-holdings":
        result = asyncio.run(_evaluate_real_holdings(_read_csv(Path(args.holdings)), _read_questions(Path(args.questions))))
    else:
        result = asyncio.run(_evaluate(_load_dataset(Path(args.dataset))))
    if not args.show_rows:
        result = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
