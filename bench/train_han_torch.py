"""Train a small PyTorch HAN-style metapath attention ranker offline.

This consumes HAN-ready artifacts and uses a PyTorch model with learned
metapath embeddings and an attention gate between
query/metapath features and metapath identity. It is intentionally kept in
bench/ and is not wired into the runtime retrieval path.

Run:
    python bench/train_han_torch.py
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

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise SystemExit("PyTorch is required. Install optional bench deps: pip install -r bench/requirements-han.txt") from exc

Record = dict[str, Any]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_han_context(han_dir: Path) -> dict[str, Any]:
    labels = [
        json.loads(line)
        for line in (han_dir / "query_metapath_labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adjacency = {path.stem: _read_json(path) for path in (han_dir / "adjacency_by_metapath").glob("*.json")}
    return {
        "entities": {entity["id"]: entity for entity in _read_json(han_dir / "entities.json")},
        "metapaths": {metapath["name"]: metapath for metapath in _read_json(han_dir / "metapaths.json")},
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
    path_count = len(instances)
    base = ranker.vectorize(record)
    base.update({
        "han_path_instance_count": float(path_count),
        "han_log_path_instance_count": math.log1p(path_count),
        "han_reachable_end_count": float(len({instance.get("end_entity_id") for instance in instances})),
        "han_has_reachable_path": 1.0 if path_count else 0.0,
    })
    return base


def _feature_vector(record: Record, han_context: dict[str, Any], feature_names: list[str]) -> list[float]:
    values = han_features(record, han_context)
    return [float(values.get(name, 0.0)) for name in feature_names]


def build_feature_names(records: list[Record], han_context: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for record in records:
        names.update(han_features(record, han_context))
    return sorted(names)


def _metapath_ids(records: list[Record]) -> dict[str, int]:
    names = sorted({str(record["candidate_metapath"]) for record in records})
    return {name: index for index, name in enumerate(names)}


class TorchHANRanker(nn.Module):
    """Small HAN-style scorer with learned metapath attention."""

    def __init__(self, feature_dim: int, metapath_count: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim)
        self.metapath_embedding = nn.Embedding(metapath_count, hidden_dim)
        self.attention_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, metapath_ids: torch.Tensor) -> torch.Tensor:
        feature_state = torch.tanh(self.feature_encoder(features))
        metapath_state = self.metapath_embedding(metapath_ids)
        gate = torch.sigmoid(self.attention_gate(torch.cat([feature_state, metapath_state], dim=-1)))
        attended = gate * feature_state + (1.0 - gate) * metapath_state
        return self.scorer(torch.tanh(attended)).squeeze(-1)


def _tensor_for(record: Record, han_context: dict[str, Any], feature_names: list[str]) -> torch.Tensor:
    return torch.tensor(_feature_vector(record, han_context, feature_names), dtype=torch.float32)


def train_torch_han_ranker(
    records: list[Record],
    han_context: dict[str, Any],
    epochs: int = 120,
    learning_rate: float = 0.01,
    hidden_dim: int = 24,
    seed: int = 7,
) -> tuple[TorchHANRanker, list[str], dict[str, int]]:
    torch.manual_seed(seed)
    feature_names = build_feature_names(records, han_context)
    metapath_ids = _metapath_ids(records)
    model = TorchHANRanker(len(feature_names), len(metapath_ids), hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MarginRankingLoss(margin=1.0)
    groups = ranker.group_by_query(records)

    for _ in range(epochs):
        for group in groups.values():
            positives = [record for record in group if int(record["label"]) == 1]
            negatives = [record for record in group if int(record["label"]) == 0]
            if not positives or not negatives:
                continue
            positive = positives[0]
            positive_features = _tensor_for(positive, han_context, feature_names).unsqueeze(0)
            positive_metapath = torch.tensor([metapath_ids[str(positive["candidate_metapath"])]], dtype=torch.long)
            for negative in negatives:
                negative_features = _tensor_for(negative, han_context, feature_names).unsqueeze(0)
                negative_metapath = torch.tensor([metapath_ids[str(negative["candidate_metapath"])]], dtype=torch.long)
                positive_score = model(positive_features, positive_metapath)
                negative_score = model(negative_features, negative_metapath)
                target = torch.ones_like(positive_score)
                loss = loss_fn(positive_score, negative_score, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return model, feature_names, metapath_ids


def score_record(
    model: TorchHANRanker,
    record: Record,
    han_context: dict[str, Any],
    feature_names: list[str],
    metapath_ids: dict[str, int],
) -> float:
    model.eval()
    with torch.no_grad():
        features = _tensor_for(record, han_context, feature_names).unsqueeze(0)
        metapath_id = torch.tensor([metapath_ids[str(record["candidate_metapath"])]], dtype=torch.long)
        return float(model(features, metapath_id).item())


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a binomial proportion (top-1 hit rate).

    The eval set here is small (~12 held-out query groups), so a bare point
    estimate is misleading: a top-1 hit rate of 1.0 on 12 queries is weak
    evidence, not proof. A bootstrap is useless on a perfect result (every
    resample is all-correct -> [1.0, 1.0]), but the Wilson interval stays honest:
    12/12 yields roughly [0.76, 1.0], exposing the small-sample uncertainty.
    """
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * total)) / total) / denom
    return [round(max(0.0, center - margin), 3), round(min(1.0, center + margin), 3)]


def evaluate_torch_han_ranker(
    records: list[Record],
    model: TorchHANRanker,
    han_context: dict[str, Any],
    feature_names: list[str],
    metapath_ids: dict[str, int],
) -> dict[str, Any]:
    groups = ranker.group_by_query(records)
    torch_top1 = 0
    rule_top1 = 0
    torch_mrr = 0.0
    rule_mrr = 0.0
    rows: list[dict[str, Any]] = []
    for query_id, group in sorted(groups.items()):
        torch_sorted = sorted(
            group,
            key=lambda record: (
                score_record(model, record, han_context, feature_names, metapath_ids),
                str(record["candidate_metapath"]),
            ),
            reverse=True,
        )
        rule_sorted = sorted(group, key=ranker.rule_score, reverse=True)
        torch_rank = next(index for index, record in enumerate(torch_sorted, start=1) if int(record["label"]) == 1)
        rule_rank = next(index for index, record in enumerate(rule_sorted, start=1) if int(record["label"]) == 1)
        torch_top1 += int(torch_rank == 1)
        rule_top1 += int(rule_rank == 1)
        torch_mrr += 1.0 / torch_rank
        rule_mrr += 1.0 / rule_rank
        positive = next(record for record in group if int(record["label"]) == 1)
        rows.append(
            {
                "query_id": query_id,
                "query": positive["query"],
                "expected_metapath": positive["expected_metapath"],
                "torch_top1": torch_sorted[0]["candidate_metapath"],
                "rule_top1": rule_sorted[0]["candidate_metapath"],
                "torch_positive_rank": torch_rank,
                "rule_positive_rank": rule_rank,
            }
        )
    total = max(len(groups), 1)
    return {
        "queries": len(groups),
        "torch_top1_hit_rate": round(torch_top1 / total, 3),
        "torch_top1_ci95": wilson_interval(torch_top1, total),
        "rule_top1_hit_rate": round(rule_top1 / total, 3),
        "rule_top1_ci95": wilson_interval(rule_top1, total),
        "torch_mrr": round(torch_mrr / total, 3),
        "rule_mrr": round(rule_mrr / total, 3),
        "small_sample_warning": (
            f"only {len(groups)} held-out query groups; treat point estimates as "
            "directional, not significant"
            if len(groups) < 30
            else None
        ),
        "rows": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_model_artifact(
    path: Path,
    model: TorchHANRanker,
    feature_names: list[str],
    metapath_ids: dict[str, int],
    result: dict[str, Any],
    hidden_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "pytorch_han_style_metapath_attention",
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "metapath_ids": metapath_ids,
            "hidden_dim": hidden_dim,
            "evaluation": result,
        },
        path,
    )


def attention_summary(model: TorchHANRanker) -> dict[str, float]:
    gate_weight = model.attention_gate.weight.detach().abs().mean().item()
    metapath_embedding_norm = model.metapath_embedding.weight.detach().norm(dim=1).mean().item()
    return {
        "mean_abs_attention_gate_weight": round(gate_weight, 4),
        "mean_metapath_embedding_norm": round(metapath_embedding_norm, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=str(BENCH / "han_data" / "metapath_training.jsonl"))
    parser.add_argument("--han-dir", default=str(BENCH / "han_data"))
    parser.add_argument("--train-dataset", default="synthetic_finance_graph")
    parser.add_argument("--eval-dataset", default="real_13f_style_holdings")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--show-rows", action="store_true")
    parser.add_argument("--output", help="Optional path for the evaluation JSON report.")
    parser.add_argument("--model-output", help="Optional path for the trained torch model artifact.")
    args = parser.parse_args()

    records = ranker.load_jsonl(Path(args.training))
    train_records, eval_records = ranker.split_records(records, args.train_dataset, args.eval_dataset)
    han_context = load_han_context(Path(args.han_dir))
    model, feature_names, metapath_ids = train_torch_han_ranker(
        train_records,
        han_context,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
    )
    result = {
        "model": "pytorch_han_style_metapath_attention",
        "train_dataset": args.train_dataset,
        "eval_dataset": args.eval_dataset,
        "train_queries": len(ranker.group_by_query(train_records)),
        "eval_queries": len(ranker.group_by_query(eval_records)),
        **evaluate_torch_han_ranker(eval_records, model, han_context, feature_names, metapath_ids),
        "feature_count": len(feature_names),
        "attention_summary": attention_summary(model),
    }
    if not args.show_rows:
        result.pop("rows", None)
    if args.output:
        write_json(Path(args.output), result)
    if args.model_output:
        save_model_artifact(Path(args.model_output), model, feature_names, metapath_ids, result, args.hidden_dim)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
