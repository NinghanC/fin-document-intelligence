"""Export metapath router training data for learned rankers and HAN prep.

The exporter turns labeled benchmark questions into pairwise query-metapath
examples. Each row is one candidate metapath for one query, with the benchmark
metapath marked as label=1 and all other configured metapaths marked label=0.

Run:
    python bench/export_metapath_training_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.metapaths import FINANCIAL_METAPATHS, MetapathRouter, MetapathSelection, MetapathSpec  # noqa: E402

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _candidate_steps(spec: MetapathSpec) -> list[dict[str, str]]:
    return [
        {
            "from_type": step.from_type,
            "relation": step.relation,
            "direction": step.direction,
            "to_type": step.to_type,
        }
        for step in spec.steps
    ]


def _selection_by_name(selections: list[MetapathSelection]) -> dict[str, MetapathSelection]:
    return {selection.spec.name: selection for selection in selections}


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_entity_type_lookup(
    synthetic_dataset: dict[str, Any],
    real_holdings_rows: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    entity_types = {
        str(entity["name"]): str(entity["type"])
        for entity in synthetic_dataset.get("entities", [])
        if entity.get("name") and entity.get("type")
    }
    for row in real_holdings_rows or []:
        if row.get("manager"):
            entity_types[row["manager"]] = "Fund"
        if row.get("company"):
            entity_types[row["company"]] = "Company"
        if row.get("sector"):
            entity_types[row["sector"]] = "Sector"
        if row.get("region"):
            entity_types[row["region"]] = "Region"
    return entity_types


def _features_for_candidate(
    question: dict[str, Any],
    spec: MetapathSpec,
    selection: MetapathSelection | None,
    router_rank: int | None,
    entity_types: dict[str, str],
) -> dict[str, int | float | bool | str | list[str]]:
    query = str(question["question"])
    query_tokens = _tokenize(query)
    start_entities = [str(entity) for entity in question.get("start_entities", [])]
    start_types = sorted({entity_types[entity] for entity in start_entities if entity in entity_types})
    first_step_type = spec.steps[0].from_type
    matched_keyword_count = len(selection.matched_keywords) if selection else 0
    candidate_keyword_count = len(spec.keywords)
    keyword_coverage = matched_keyword_count / max(candidate_keyword_count, 1)
    return {
        "query_token_count": len(query_tokens),
        "unique_query_token_count": len(set(query_tokens)),
        "start_entity_count": len(start_entities),
        "known_start_entity_type_count": len(start_types),
        "start_entity_types": start_types,
        "candidate_first_entity_type": first_step_type,
        "candidate_last_entity_type": spec.steps[-1].to_type,
        "first_step_matches_start_type": first_step_type in start_types,
        "candidate_path_length": len(spec.steps),
        "candidate_keyword_count": candidate_keyword_count,
        "matched_keyword_count": matched_keyword_count,
        "keyword_coverage": round(keyword_coverage, 4),
        "router_rank": router_rank or 0,
        "router_selected": selection is not None,
        "router_score": selection.score if selection else 0,
        "selection_fallback": selection.fallback if selection else False,
    }


def _records_for_question(
    dataset_name: str,
    query_index: int,
    question: dict[str, Any],
    router: MetapathRouter,
    entity_types: dict[str, str],
) -> list[dict[str, Any]]:
    selections = router.select_with_trace(question["question"], question.get("start_entities", []), limit=3)
    selected_names = [selection.spec.name for selection in selections]
    selection_map = _selection_by_name(selections)
    expected_metapath = str(question["expected_metapath"])
    query_id = f"{dataset_name}:{query_index:03d}"

    records: list[dict[str, Any]] = []
    for candidate_name, spec in FINANCIAL_METAPATHS.items():
        selection = selection_map.get(candidate_name)
        router_rank = selected_names.index(candidate_name) + 1 if candidate_name in selected_names else None
        features = _features_for_candidate(question, spec, selection, router_rank, entity_types)
        records.append({
            "dataset": dataset_name,
            "query_id": query_id,
            "query": question["question"],
            "start_entities": question.get("start_entities", []),
            "expected_metapath": expected_metapath,
            "expected_end_entities": question.get("expected_end_entities", []),
            "candidate_metapath": candidate_name,
            "candidate_description": spec.description,
            "candidate_steps": _candidate_steps(spec),
            "candidate_path_length": len(spec.steps),
            "label": 1 if candidate_name == expected_metapath else 0,
            "router_selected": selection is not None,
            "router_rank": router_rank,
            "router_score": selection.score if selection else 0,
            "matched_keywords": list(selection.matched_keywords) if selection else [],
            "selection_reason": selection.reason if selection else "not selected by rule router",
            "selection_fallback": selection.fallback if selection else False,
            "features": features,
        })
    return records


def build_training_records(
    synthetic_dataset: dict[str, Any],
    real_holdings_questions: list[dict[str, Any]],
    real_holdings_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    router = MetapathRouter()
    entity_types = build_entity_type_lookup(synthetic_dataset, real_holdings_rows)
    records: list[dict[str, Any]] = []
    for index, question in enumerate(synthetic_dataset["questions"]):
        records.extend(_records_for_question("synthetic_finance_graph", index, question, router, entity_types))
    for index, question in enumerate(real_holdings_questions):
        records.extend(_records_for_question("real_13f_style_holdings", index, question, router, entity_types))
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    query_ids = {record["query_id"] for record in records}
    positives = [record for record in records if record["label"] == 1]
    selected = [record for record in records if record["router_selected"]]
    top1_hits = [record for record in positives if record["router_rank"] == 1]
    type_compatible_positives = [
        record for record in positives
        if record.get("features", {}).get("first_step_matches_start_type") is True
    ]
    datasets = Counter(str(record["dataset"]) for record in records)
    return {
        "records": len(records),
        "queries": len(query_ids),
        "candidate_metapaths": len(FINANCIAL_METAPATHS),
        "positive_records": len(positives),
        "negative_records": len(records) - len(positives),
        "router_selected_records": len(selected),
        "positive_top1_rate": round(len(top1_hits) / max(len(positives), 1), 3),
        "positive_type_compatibility_rate": round(len(type_compatible_positives) / max(len(positives), 1), 3),
        "datasets": dict(sorted(datasets.items())),
    }


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True) for record in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=str(Path(__file__).with_name("metapath_dataset.json")))
    parser.add_argument("--real-holdings-questions", default=str(Path(__file__).with_name("real_holdings") / "questions.json"))
    parser.add_argument("--real-holdings", default=str(Path(__file__).with_name("real_holdings") / "holdings_sample.csv"))
    parser.add_argument("--output", default=str(Path(__file__).with_name("han_data") / "metapath_training.jsonl"))
    args = parser.parse_args()

    records = build_training_records(
        synthetic_dataset=_read_json(Path(args.synthetic)),
        real_holdings_questions=_read_json(Path(args.real_holdings_questions)),
        real_holdings_rows=_read_csv(Path(args.real_holdings)),
    )
    write_jsonl(records, Path(args.output))
    print(json.dumps({"output": str(Path(args.output)), **summarize(records)}, indent=2))


if __name__ == "__main__":
    main()
