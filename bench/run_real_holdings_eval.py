"""Evaluate metapath retrieval on a small public 13F-style holdings sample.

The CSV is committed so this benchmark is deterministic and does not require
network access. It complements the synthetic metapath benchmark by using a data
shape closer to public holdings disclosures.

Run:
    python bench/run_real_holdings_eval.py
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

from agents.knowledge_extract_agent import Entity, Relation  # noqa: E402
from services.knowledge_graph import KnowledgeGraphService  # noqa: E402
from services.metapaths import FINANCIAL_METAPATHS, MetapathRouter  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_questions(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


async def _build_graph(rows: list[dict[str, str]]) -> KnowledgeGraphService:
    graph = KnowledgeGraphService()
    managers = sorted({row["manager"] for row in rows})
    companies = sorted({row["company"] for row in rows})
    sectors = sorted({row["sector"] for row in rows})
    regions = sorted({row["region"] for row in rows})

    for manager in managers:
        await graph.upsert_entity(Entity(name=manager, type="Fund", description="Public 13F-style portfolio sample"))
    for company in companies:
        await graph.upsert_entity(Entity(name=company, type="Company"))
    for sector in sectors:
        await graph.upsert_entity(Entity(name=sector, type="Sector"))
    for region in regions:
        await graph.upsert_entity(Entity(name=region, type="Region"))

    seen_relations: set[tuple[str, str, str]] = set()
    for row in rows:
        relation_specs = [
            (row["manager"], "holds", row["company"]),
            (row["company"], "belongs_to", row["sector"]),
            (row["company"], "located_in", row["region"]),
        ]
        for head, relation, tail in relation_specs:
            key = (head, relation, tail)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            await graph.add_relation(Relation(head=head, relation=relation, tail=tail, confidence=1.0))

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


async def _evaluate(rows: list[dict[str, str]], questions: list[dict[str, Any]]) -> dict[str, Any]:
    graph = await _build_graph(rows)
    router = MetapathRouter()
    result_rows: list[dict[str, Any]] = []

    for question in questions:
        selections = router.select_with_trace(question["question"], question["start_entities"], limit=3)
        selected_names = [selection.spec.name for selection in selections]
        expected_metapath = question["expected_metapath"]
        router_hit = expected_metapath in selected_names
        router_top1_hit = bool(selected_names) and selected_names[0] == expected_metapath
        router_precision = (1.0 / len(selected_names)) if router_hit and selected_names else 0.0
        extra_metapaths = [name for name in selected_names if name != expected_metapath]
        oracle_paths = await graph.traverse_metapath(question["start_entities"], FINANCIAL_METAPATHS[expected_metapath])
        oracle_reached = sorted({result.end_entity for result in oracle_paths})
        routed_reached, reached_by_selected = await _traverse_selected(graph, question["start_entities"], selected_names)
        expected = set(question["expected_end_entities"])
        oracle_recall = len(expected & set(oracle_reached)) / max(len(expected), 1)
        routed_recall = len(expected & set(routed_reached)) / max(len(expected), 1)
        result_rows.append({
            "question": question["question"],
            "expected_metapath": expected_metapath,
            "selected_metapaths": selected_names,
            "router_trace": [selection.as_trace() for selection in selections],
            "router_hit": router_hit,
            "router_top1_hit": router_top1_hit,
            "router_precision": round(router_precision, 3),
            "extra_metapaths": extra_metapaths,
            "expected_end_entities": sorted(expected),
            "routed_reached_end_entities": routed_reached,
            "oracle_reached_end_entities": oracle_reached,
            "reached_by_selected_metapath": reached_by_selected,
            "routed_path_recall": round(routed_recall, 3),
            "oracle_path_recall": round(oracle_recall, 3),
            "routed_path_hit": routed_recall == 1.0,
            "oracle_path_hit": oracle_recall == 1.0,
        })

    total = len(result_rows)
    router_hits = sum(1 for row in result_rows if row["router_hit"])
    router_top1_hits = sum(1 for row in result_rows if row["router_top1_hit"])
    average_router_precision = sum(float(row["router_precision"]) for row in result_rows) / max(total, 1)
    average_selected_metapaths = sum(len(row["selected_metapaths"]) for row in result_rows) / max(total, 1)
    routed_path_hits = sum(1 for row in result_rows if row["routed_path_hit"])
    oracle_path_hits = sum(1 for row in result_rows if row["oracle_path_hit"])
    average_routed_recall = sum(float(row["routed_path_recall"]) for row in result_rows) / max(total, 1)
    average_oracle_recall = sum(float(row["oracle_path_recall"]) for row in result_rows) / max(total, 1)
    return {
        "dataset": "real_13f_style_holdings_sample",
        "holdings_rows": len(rows),
        "questions": total,
        "router_hit_rate": round(router_hits / max(total, 1), 3),
        "router_top1_hit_rate": round(router_top1_hits / max(total, 1), 3),
        "average_router_precision": round(average_router_precision, 3),
        "average_selected_metapaths": round(average_selected_metapaths, 3),
        "routed_path_hit_rate": round(routed_path_hits / max(total, 1), 3),
        "average_routed_path_recall": round(average_routed_recall, 3),
        "oracle_path_hit_rate": round(oracle_path_hits / max(total, 1), 3),
        "average_oracle_path_recall": round(average_oracle_recall, 3),
        "path_hit_rate": round(routed_path_hits / max(total, 1), 3),
        "average_path_recall": round(average_routed_recall, 3),
        "rows": result_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).with_name("real_holdings")
    parser.add_argument("--holdings", default=str(root / "holdings_sample.csv"))
    parser.add_argument("--questions", default=str(root / "questions.json"))
    parser.add_argument("--show-rows", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(_evaluate(_read_csv(Path(args.holdings)), _read_questions(Path(args.questions))))
    if not args.show_rows:
        result = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
