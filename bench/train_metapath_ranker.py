"""Train a lightweight metapath ranker baseline before adding HAN.

This is intentionally dependency-free. It trains a pairwise linear ranker over
the exported query-metapath feature rows and compares it with the rule router.
The goal is not to ship this model in production; it is a measurable baseline
for deciding whether a learned router or HAN attention layer is worthwhile.

Run:
    python bench/train_metapath_ranker.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

Record = dict[str, Any]
Weights = dict[str, float]

NUMERIC_FEATURES = (
    "query_token_count",
    "unique_query_token_count",
    "start_entity_count",
    "known_start_entity_type_count",
    "first_step_matches_start_type",
    "candidate_path_length",
    "candidate_keyword_count",
    "matched_keyword_count",
    "keyword_coverage",
    "router_reciprocal_rank",
    "router_selected",
    "router_score",
    "selection_fallback",
)


def load_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_by_query(records: list[Record]) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[str(record["query_id"])].append(record)
    return dict(groups)


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def vectorize(record: Record) -> dict[str, float]:
    features = record.get("features", {})
    vector: dict[str, float] = {"bias": 1.0}
    for name in NUMERIC_FEATURES:
        vector[name] = _as_float(features.get(name))

    candidate = str(record["candidate_metapath"])
    vector[f"candidate={candidate}"] = 1.0
    vector[f"first_type={features.get('candidate_first_entity_type', '')}"] = 1.0
    vector[f"last_type={features.get('candidate_last_entity_type', '')}"] = 1.0
    for entity_type in features.get("start_entity_types", []):
        vector[f"start_type={entity_type}"] = 1.0
    return vector


def dot(weights: Weights, vector: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in vector.items())


def _add_scaled(weights: Weights, vector: dict[str, float], scale: float) -> None:
    for name, value in vector.items():
        weights[name] = weights.get(name, 0.0) + scale * value


def train_pairwise_ranker(
    records: list[Record],
    epochs: int = 80,
    learning_rate: float = 0.03,
    margin: float = 1.0,
) -> Weights:
    weights: Weights = {}
    groups = group_by_query(records)
    for _ in range(epochs):
        for group in groups.values():
            positives = [record for record in group if int(record["label"]) == 1]
            negatives = [record for record in group if int(record["label"]) == 0]
            if not positives or not negatives:
                continue
            positive = positives[0]
            positive_vector = vectorize(positive)
            positive_score = dot(weights, positive_vector)
            for negative in negatives:
                negative_vector = vectorize(negative)
                if positive_score <= dot(weights, negative_vector) + margin:
                    _add_scaled(weights, positive_vector, learning_rate)
                    _add_scaled(weights, negative_vector, -learning_rate)
                    positive_score = dot(weights, positive_vector)
    return weights


def learned_score(weights: Weights, record: Record) -> float:
    return dot(weights, vectorize(record))


def rule_score(record: Record) -> tuple[int, float, str]:
    rank = record.get("router_rank")
    if isinstance(rank, int) and rank > 0:
        return (1, -float(rank), str(record["candidate_metapath"]))
    return (0, 0.0, str(record["candidate_metapath"]))


def evaluate_ranker(records: list[Record], weights: Weights) -> dict[str, Any]:
    groups = group_by_query(records)
    learned_top1 = 0
    rule_top1 = 0
    learned_reciprocal_rank = 0.0
    rule_reciprocal_rank = 0.0
    rows: list[dict[str, Any]] = []

    for query_id, group in sorted(groups.items()):
        learned_sorted = sorted(
            group,
            key=lambda record: (learned_score(weights, record), str(record["candidate_metapath"])),
            reverse=True,
        )
        rule_sorted = sorted(group, key=rule_score, reverse=True)
        learned_rank = next(index for index, record in enumerate(learned_sorted, start=1) if int(record["label"]) == 1)
        rule_rank = next(index for index, record in enumerate(rule_sorted, start=1) if int(record["label"]) == 1)
        learned_top1 += int(learned_rank == 1)
        rule_top1 += int(rule_rank == 1)
        learned_reciprocal_rank += 1.0 / learned_rank
        rule_reciprocal_rank += 1.0 / rule_rank
        positive = next(record for record in group if int(record["label"]) == 1)
        rows.append({
            "query_id": query_id,
            "query": positive["query"],
            "expected_metapath": positive["expected_metapath"],
            "learned_top1": learned_sorted[0]["candidate_metapath"],
            "rule_top1": rule_sorted[0]["candidate_metapath"],
            "learned_positive_rank": learned_rank,
            "rule_positive_rank": rule_rank,
        })

    total = max(len(groups), 1)
    return {
        "queries": len(groups),
        "learned_top1_hit_rate": round(learned_top1 / total, 3),
        "rule_top1_hit_rate": round(rule_top1 / total, 3),
        "learned_mrr": round(learned_reciprocal_rank / total, 3),
        "rule_mrr": round(rule_reciprocal_rank / total, 3),
        "rows": rows,
    }


def split_records(records: list[Record], train_dataset: str, eval_dataset: str) -> tuple[list[Record], list[Record]]:
    train = [record for record in records if record["dataset"] == train_dataset]
    evaluate = [record for record in records if record["dataset"] == eval_dataset]
    if not train or not evaluate:
        raise ValueError("Both train and eval splits must contain records")
    return train, evaluate


def top_weights(weights: Weights, limit: int = 10) -> list[dict[str, float | str]]:
    ranked = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "weight": round(value, 4)} for name, value in ranked[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).with_name("han_data") / "metapath_training.jsonl"))
    parser.add_argument("--train-dataset", default="synthetic_finance_graph")
    parser.add_argument("--eval-dataset", default="real_13f_style_holdings")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--show-rows", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    train_records, eval_records = split_records(records, args.train_dataset, args.eval_dataset)
    weights = train_pairwise_ranker(train_records, epochs=args.epochs, learning_rate=args.learning_rate)
    result = {
        "train_dataset": args.train_dataset,
        "eval_dataset": args.eval_dataset,
        "train_queries": len(group_by_query(train_records)),
        "eval_queries": len(group_by_query(eval_records)),
        **evaluate_ranker(eval_records, weights),
        "top_weights": top_weights(weights),
    }
    if not args.show_rows:
        result.pop("rows", None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
