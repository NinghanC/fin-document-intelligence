"""Export HAN-ready graph artifacts from the metapath benchmarks.

This does not train HAN. It prepares stable IDs and metapath-specific path
instances so a future neural attention model can consume the same graph/query
labels without coupling to the runtime service objects.

Run:
    python bench/export_han_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.metapaths import FINANCIAL_METAPATHS, MetapathSpec  # noqa: E402

EntityMap = dict[str, str]
RelationRow = dict[str, str]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _add_entity(entities: EntityMap, name: str, entity_type: str) -> None:
    if name:
        entities[name] = entity_type


def build_graph_inputs(
    synthetic_dataset: dict[str, Any],
    real_holdings_rows: list[dict[str, str]],
) -> tuple[EntityMap, list[RelationRow]]:
    entities: EntityMap = {}
    relations: list[RelationRow] = []

    for item in synthetic_dataset.get("entities", []):
        _add_entity(entities, str(item["name"]), str(item["type"]))
    for item in synthetic_dataset.get("relations", []):
        relations.append({
            "head": str(item["head"]),
            "relation": str(item["relation"]),
            "tail": str(item["tail"]),
            "dataset": "synthetic_finance_graph",
        })

    seen_real_relations: set[tuple[str, str, str]] = set()
    for row in real_holdings_rows:
        manager = row.get("manager", "")
        company = row.get("company", "")
        sector = row.get("sector", "")
        region = row.get("region", "")
        _add_entity(entities, manager, "Fund")
        _add_entity(entities, company, "Company")
        _add_entity(entities, sector, "Sector")
        _add_entity(entities, region, "Region")
        for head, relation, tail in (
            (manager, "holds", company),
            (company, "belongs_to", sector),
            (company, "located_in", region),
        ):
            key = (head, relation, tail)
            if head and tail and key not in seen_real_relations:
                seen_real_relations.add(key)
                relations.append({
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "dataset": "real_13f_style_holdings",
                })

    return entities, relations


def build_id_maps(entities: EntityMap, relations: list[RelationRow]) -> tuple[dict[str, int], dict[str, int]]:
    entity_ids = {name: index for index, name in enumerate(sorted(entities))}
    relation_types = sorted({relation["relation"] for relation in relations})
    relation_type_ids = {relation_type: index for index, relation_type in enumerate(relation_types)}
    return entity_ids, relation_type_ids


def _entity_records(entities: EntityMap, entity_ids: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"id": entity_ids[name], "name": name, "type": entities[name]}
        for name in sorted(entities)
    ]


def _relation_records(
    relations: list[RelationRow],
    entity_ids: dict[str, int],
    relation_type_ids: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "id": index,
            "head_id": entity_ids[row["head"]],
            "tail_id": entity_ids[row["tail"]],
            "head": row["head"],
            "tail": row["tail"],
            "relation": row["relation"],
            "relation_type_id": relation_type_ids[row["relation"]],
            "dataset": row["dataset"],
        }
        for index, row in enumerate(relations)
        if row["head"] in entity_ids and row["tail"] in entity_ids
    ]


def _metapath_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (name, spec) in enumerate(FINANCIAL_METAPATHS.items()):
        records.append({
            "id": index,
            "name": name,
            "description": spec.description,
            "steps": [
                {
                    "from_type": step.from_type,
                    "relation": step.relation,
                    "direction": step.direction,
                    "to_type": step.to_type,
                }
                for step in spec.steps
            ],
        })
    return records


def _relation_matches(
    row: RelationRow,
    current_name: str,
    step_relation: str,
    direction: str,
) -> tuple[str, str] | None:
    if row["relation"] != step_relation:
        return None
    if direction == "out" and row["head"] == current_name:
        return row["head"], row["tail"]
    if direction == "in" and row["tail"] == current_name:
        return row["tail"], row["head"]
    return None


def traverse_metapath_instances(
    spec: MetapathSpec,
    entities: EntityMap,
    relations: list[RelationRow],
    entity_ids: dict[str, int],
    relation_type_ids: dict[str, int],
    limit: int = 1000,
) -> list[dict[str, Any]]:
    start_names = sorted(name for name, entity_type in entities.items() if entity_type == spec.steps[0].from_type)
    instances: list[dict[str, Any]] = []
    for start_name in start_names:
        states: list[tuple[str, list[str], list[dict[str, Any]]]] = [(start_name, [start_name], [])]
        for step in spec.steps:
            next_states: list[tuple[str, list[str], list[dict[str, Any]]]] = []
            for current_name, node_path, edge_path in states:
                if entities.get(current_name) != step.from_type:
                    continue
                for row in relations:
                    matched = _relation_matches(row, current_name, step.relation, step.direction)
                    if not matched:
                        continue
                    source_name, target_name = matched
                    if entities.get(target_name) != step.to_type:
                        continue
                    next_states.append((
                        target_name,
                        [*node_path, target_name],
                        [
                            *edge_path,
                            {
                                "source_id": entity_ids[source_name],
                                "target_id": entity_ids[target_name],
                                "source": source_name,
                                "target": target_name,
                                "relation": step.relation,
                                "relation_type_id": relation_type_ids[step.relation],
                            },
                        ],
                    ))
            states = next_states
            if not states:
                break
        for end_name, node_path, edge_path in states:
            instances.append({
                "start_entity_id": entity_ids[start_name],
                "end_entity_id": entity_ids[end_name],
                "node_ids": [entity_ids[name] for name in node_path],
                "nodes": node_path,
                "edges": edge_path,
            })
            if len(instances) >= limit:
                return instances
    return instances


def _question_rows(synthetic_dataset: dict[str, Any], real_questions: list[dict[str, Any]]) -> list[tuple[str, int, dict[str, Any]]]:
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for index, question in enumerate(synthetic_dataset.get("questions", [])):
        rows.append(("synthetic_finance_graph", index, question))
    for index, question in enumerate(real_questions):
        rows.append(("real_13f_style_holdings", index, question))
    return rows


def build_query_labels(
    synthetic_dataset: dict[str, Any],
    real_questions: list[dict[str, Any]],
    entity_ids: dict[str, int],
    metapath_ids: dict[str, int],
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for dataset, index, question in _question_rows(synthetic_dataset, real_questions):
        start_entities = [str(entity) for entity in question.get("start_entities", [])]
        labels.append({
            "query_id": f"{dataset}:{index:03d}",
            "dataset": dataset,
            "query": question["question"],
            "start_entity_ids": [entity_ids[name] for name in start_entities if name in entity_ids],
            "start_entities": start_entities,
            "positive_metapath_id": metapath_ids[str(question["expected_metapath"])],
            "positive_metapath": str(question["expected_metapath"]),
            "expected_end_entity_ids": [
                entity_ids[name] for name in question.get("expected_end_entities", []) if name in entity_ids
            ],
            "expected_end_entities": question.get("expected_end_entities", []),
        })
    return labels


def build_han_artifacts(
    synthetic_dataset: dict[str, Any],
    real_holdings_rows: list[dict[str, str]],
    real_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    entities, relations = build_graph_inputs(synthetic_dataset, real_holdings_rows)
    entity_ids, relation_type_ids = build_id_maps(entities, relations)
    relation_records = _relation_records(relations, entity_ids, relation_type_ids)
    metapaths = _metapath_records()
    metapath_ids = {record["name"]: record["id"] for record in metapaths}
    adjacency_by_metapath = {
        name: traverse_metapath_instances(spec, entities, relations, entity_ids, relation_type_ids)
        for name, spec in FINANCIAL_METAPATHS.items()
    }
    query_labels = build_query_labels(synthetic_dataset, real_questions, entity_ids, metapath_ids)
    return {
        "manifest": {
            "format": "han_ready_metapath_graph_v1",
            "entity_count": len(entity_ids),
            "relation_count": len(relation_records),
            "relation_type_count": len(relation_type_ids),
            "metapath_count": len(metapaths),
            "query_count": len(query_labels),
        },
        "entities": _entity_records(entities, entity_ids),
        "relations": relation_records,
        "relation_types": [
            {"id": relation_type_ids[name], "name": name}
            for name in sorted(relation_type_ids)
        ],
        "metapaths": metapaths,
        "query_labels": query_labels,
        "adjacency_by_metapath": adjacency_by_metapath,
    }


def write_han_artifacts(artifacts: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(artifacts["manifest"], indent=2), encoding="utf-8")
    for name in ("entities", "relations", "relation_types", "metapaths"):
        (output_dir / f"{name}.json").write_text(json.dumps(artifacts[name], indent=2), encoding="utf-8")
    labels = [json.dumps(row, sort_keys=True) for row in artifacts["query_labels"]]
    (output_dir / "query_metapath_labels.jsonl").write_text("\n".join(labels) + "\n", encoding="utf-8")
    adjacency_dir = output_dir / "adjacency_by_metapath"
    adjacency_dir.mkdir(exist_ok=True)
    for name, instances in artifacts["adjacency_by_metapath"].items():
        (adjacency_dir / f"{name}.json").write_text(json.dumps(instances, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=str(Path(__file__).with_name("metapath_dataset.json")))
    parser.add_argument("--real-holdings", default=str(Path(__file__).with_name("real_holdings") / "holdings_sample.csv"))
    parser.add_argument("--real-holdings-questions", default=str(Path(__file__).with_name("real_holdings") / "questions.json"))
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("han_data")))
    args = parser.parse_args()

    artifacts = build_han_artifacts(
        synthetic_dataset=_read_json(Path(args.synthetic)),
        real_holdings_rows=_read_csv(Path(args.real_holdings)),
        real_questions=_read_json(Path(args.real_holdings_questions)),
    )
    write_han_artifacts(artifacts, Path(args.output_dir))
    print(json.dumps({"output_dir": str(Path(args.output_dir)), **artifacts["manifest"]}, indent=2))


if __name__ == "__main__":
    main()
