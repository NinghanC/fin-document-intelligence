"""Train an offline HAN-style metapath attention baseline.

This is a small dependency-free prototype that consumes the HAN-ready artifacts
and pairwise query-metapath training rows. It is not a production neural HAN.
It adds graph-path instance features to the existing query/metapath features so
we can measure whether metapath-aware graph signals improve ranking offline.

Run:
    python bench/train_han_attention.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import train_metapath_ranker as ranker  # noqa: E402

Record = dict[str, Any]
Weights = dict[str, float]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_han_context(han_dir: Path) -> dict[str, Any]:
    entities = _read_json(han_dir / "entities.json")
    metapaths = _read_json(han_dir / "metapaths.json")
    labels = [
        json.loads(line)
        for line in (han_dir / "query_metapath_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for path in (han_dir / "adjacency_by_metapath").glob("*.json"):
        adjacency[path.stem] = _read_json(path)
    return {
        "entities": {entity["id"]: entity for entity in entities},
        "metapaths": {metapath["name"]: metapath for metapath in metapaths},
        "labels": {label["query_id"]: label for label in labels},
        "adjacency": adjacency,
    }


def _path_instances_for_record(record: Record, han_context: dict[str, Any]) -> list[dict[str, Any]]:
    label = han_context["labels"].get(record["query_id"], {})
    start_ids = set(label.get("start_entity_ids", []))
    candidate = str(record["candidate_metapath"])
    return [
        instance for instance in han_context["adjacency"].get(candidate, [])
        if instance.get("start_entity_id") in start_ids
    ]


def han_features(record: Record, han_context: dict[str, Any]) -> dict[str, float]:
    instances = _path_instances_for_record(record, han_context)
    end_ids = {instance.get("end_entity_id") for instance in instances}
    path_count = len(instances)
    reachable_end_count = len(end_ids)
    base = ranker.vectorize(record)
    base.update({
        "han_path_instance_count": float(path_count),
        "han_log_path_instance_count": math.log1p(path_count),
        "han_reachable_end_count": float(reachable_end_count),
        "han_has_reachable_path": 1.0 if path_count else 0.0,
    })
    return base


def dot(weights: Weights, vector: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in vector.items())


def _add_scaled(weights: Weights, vector: dict[str, float], scale: float) -> None:
    for name, value in vector.items():
        weights[name] = weights.get(name, 0.0) + scale * value


def train_attention_ranker(
    records: list[Record],
    han_context: dict[str, Any],
    epochs: int = 80,
    learning_rate: float = 0.02,
    margin: float = 1.0,
) -> Weights:
    weights: Weights = {}
    groups = ranker.group_by_query(records)
    for _ in range(epochs):
        for group in groups.values():
            positives = [record for record in group if int(record["label"]) == 1]
            negatives = [record for record in group if int(record["label"]) == 0]
            if not positives or not negatives:
                continue
            positive = positives[0]
            positive_vector = han_features(positive, han_context)
            positive_score = dot(weights, positive_vector)
            for negative in negatives:
                negative_vector = han_features(negative, han_context)
                if positive_score <= dot(weights, negative_vector) + margin:
                    _add_scaled(weights, positive_vector, learning_rate)
                    _add_scaled(weights, negative_vector, -learning_rate)
                    positive_score = dot(weights, positive_vector)
    return weights


def attention_score(weights: Weights, record: Record, han_context: dict[str, Any]) -> float:
    return dot(weights, han_features(record, han_context))


def evaluate_attention_ranker(records: list[Record], weights: Weights, han_context: dict[str, Any]) -> dict[str, Any]:
    groups = ranker.group_by_query(records)
    attention_top1 = 0
    rule_top1 = 0
    attention_reciprocal_rank = 0.0
    rule_reciprocal_rank = 0.0
    rows: list[dict[str, Any]] = []

    for query_id, group in sorted(groups.items()):
        attention_sorted = sorted(
            group,
            key=lambda record: (attention_score(weights, record, han_context), str(record["candidate_metapath"])),
            reverse=True,
        )
        rule_sorted = sorted(group, key=ranker.rule_score, reverse=True)
        attention_rank = next(
            index for index, record in enumerate(attention_sorted, start=1) if int(record["label"]) == 1
        )
        rule_rank = next(index for index, record in enumerate(rule_sorted, start=1) if int(record["label"]) == 1)
        attention_top1 += int(attention_rank == 1)
        rule_top1 += int(rule_rank == 1)
        attention_reciprocal_rank += 1.0 / attention_rank
        rule_reciprocal_rank += 1.0 / rule_rank
        positive = next(record for record in group if int(record["label"]) == 1)
        rows.append({
            "query_id": query_id,
            "query": positive["query"],
            "expected_metapath": positive["expected_metapath"],
            "attention_top1": attention_sorted[0]["candidate_metapath"],
            "rule_top1": rule_sorted[0]["candidate_metapath"],
            "attention_positive_rank": attention_rank,
            "rule_positive_rank": rule_rank,
        })

    total = max(len(groups), 1)
    return {
        "queries": len(groups),
        "attention_top1_hit_rate": round(attention_top1 / total, 3),
        "rule_top1_hit_rate": round(rule_top1 / total, 3),
        "attention_mrr": round(attention_reciprocal_rank / total, 3),
        "rule_mrr": round(rule_reciprocal_rank / total, 3),
        "rows": rows,
    }


def top_weights(weights: Weights, limit: int = 12) -> list[dict[str, float | str]]:
    ranked = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "weight": round(value, 4)} for name, value in ranked[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=str(BENCH / "han_data" / "metapath_training.jsonl"))
    parser.add_argument("--han-dir", default=str(BENCH / "han_data"))
    parser.add_argument("--train-dataset", default="synthetic_finance_graph")
    parser.add_argument("--eval-dataset", default="real_13f_style_holdings")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--show-rows", action="store_true")
    args = parser.parse_args()

    records = ranker.load_jsonl(Path(args.training))
    train_records, eval_records = ranker.split_records(records, args.train_dataset, args.eval_dataset)
    han_context = load_han_context(Path(args.han_dir))
    weights = train_attention_ranker(
        train_records,
        han_context,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    result = {
        "model": "dependency_free_han_style_attention_baseline",
        "train_dataset": args.train_dataset,
        "eval_dataset": args.eval_dataset,
        "train_queries": len(ranker.group_by_query(train_records)),
        "eval_queries": len(ranker.group_by_query(eval_records)),
        **evaluate_attention_ranker(eval_records, weights, han_context),
        "top_weights": top_weights(weights),
    }
    if not args.show_rows:
        result.pop("rows", None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
