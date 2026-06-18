"""Evaluate metapath retrieval on a small public 13F-style holdings sample.

The CSV is committed so this benchmark is deterministic and does not require
network access. It complements the synthetic metapath benchmark by using a data
shape closer to public holdings disclosures.

Run:
    PYTHONPATH=backend python bench/run_real_holdings_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from agents.knowledge_extract_agent import Entity, Relation
from services.knowledge_graph import KnowledgeGraphService
from services.metapaths import FINANCIAL_METAPATHS, MetapathRouter


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


async def _evaluate(rows: list[dict[str, str]], questions: list[dict[str, Any]]) -> dict[str, Any]:
    graph = await _build_graph(rows)
    router = MetapathRouter()
    result_rows: list[dict[str, Any]] = []

    for question in questions:
        selected = router.select(question["question"], question["start_entities"], limit=3)
        selected_names = [spec.name for spec in selected]
        expected_metapath = question["expected_metapath"]
        spec = FINANCIAL_METAPATHS[expected_metapath]
        paths = await graph.traverse_metapath(question["start_entities"], spec)
        reached = sorted({result.end_entity for result in paths})
        expected = set(question["expected_end_entities"])
        recall = len(expected & set(reached)) / max(len(expected), 1)
        result_rows.append({
            "question": question["question"],
            "expected_metapath": expected_metapath,
            "selected_metapaths": selected_names,
            "router_hit": expected_metapath in selected_names,
            "expected_end_entities": sorted(expected),
            "reached_end_entities": reached,
            "path_recall": round(recall, 3),
            "path_hit": recall == 1.0,
        })

    total = len(result_rows)
    router_hits = sum(1 for row in result_rows if row["router_hit"])
    path_hits = sum(1 for row in result_rows if row["path_hit"])
    average_recall = sum(float(row["path_recall"]) for row in result_rows) / max(total, 1)
    return {
        "dataset": "real_13f_style_holdings_sample",
        "holdings_rows": len(rows),
        "questions": total,
        "router_hit_rate": round(router_hits / max(total, 1), 3),
        "path_hit_rate": round(path_hits / max(total, 1), 3),
        "average_path_recall": round(average_recall, 3),
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
