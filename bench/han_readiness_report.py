"""Report whether the metapath pipeline is ready for a HAN implementation.

The report is intentionally conservative. It checks that data exports exist,
that train/eval splits are present, that a lightweight learned ranker improves
or matches the rule router, and that the labeled query set is large enough to
justify a neural model.

Run:
    python bench/han_readiness_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import train_metapath_ranker as ranker  # noqa: E402

REQUIRED_HAN_FILES = (
    "manifest.json",
    "entities.json",
    "relations.json",
    "relation_types.json",
    "metapaths.json",
    "query_metapath_labels.jsonl",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_count(records: list[ranker.Record]) -> int:
    return len(ranker.group_by_query(records))


def _positive_metapaths(records: list[ranker.Record]) -> set[str]:
    return {str(record["candidate_metapath"]) for record in records if int(record["label"]) == 1}


def _han_files_present(han_dir: Path) -> tuple[bool, list[str]]:
    missing = [name for name in REQUIRED_HAN_FILES if not (han_dir / name).exists()]
    adjacency_dir = han_dir / "adjacency_by_metapath"
    if not adjacency_dir.exists():
        missing.append("adjacency_by_metapath/")
    elif not any(adjacency_dir.glob("*.json")):
        missing.append("adjacency_by_metapath/*.json")
    return not missing, missing


def build_report(
    training_path: Path,
    han_dir: Path,
    train_dataset: str = "synthetic_finance_graph",
    eval_dataset: str = "real_13f_style_holdings",
    min_queries: int = 50,
    min_eval_queries: int = 10,
    min_positive_metapaths: int = 6,
) -> dict[str, Any]:
    records = ranker.load_jsonl(training_path)
    train_records, eval_records = ranker.split_records(records, train_dataset, eval_dataset)
    weights = ranker.train_pairwise_ranker(train_records)
    ranker_eval = ranker.evaluate_ranker(eval_records, weights)
    han_files_ok, missing_han_files = _han_files_present(han_dir)
    manifest = _load_json(han_dir / "manifest.json") if (han_dir / "manifest.json").exists() else {}

    gates = {
        "training_records_exist": len(records) > 0,
        "minimum_labeled_queries": _query_count(records) >= min_queries,
        "minimum_eval_queries": _query_count(eval_records) >= min_eval_queries,
        "metapath_label_coverage": len(_positive_metapaths(records)) >= min_positive_metapaths,
        "han_artifacts_exist": han_files_ok,
        "han_manifest_matches_queries": manifest.get("query_count") == _query_count(records),
        "learned_ranker_not_worse_than_rule": (
            ranker_eval["learned_top1_hit_rate"] >= ranker_eval["rule_top1_hit_rate"]
            and ranker_eval["learned_mrr"] >= ranker_eval["rule_mrr"]
        ),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "ready_for_han": not blockers,
        "gates": gates,
        "blockers": blockers,
        "thresholds": {
            "min_queries": min_queries,
            "min_eval_queries": min_eval_queries,
            "min_positive_metapaths": min_positive_metapaths,
        },
        "data": {
            "records": len(records),
            "queries": _query_count(records),
            "train_queries": _query_count(train_records),
            "eval_queries": _query_count(eval_records),
            "positive_metapaths": sorted(_positive_metapaths(records)),
            "missing_han_files": missing_han_files,
            "han_manifest": manifest,
        },
        "ranker_eval": {
            key: value for key, value in ranker_eval.items() if key != "rows"
        },
        "recommendation": _recommendation(blockers),
    }


def _recommendation(blockers: list[str]) -> str:
    if not blockers:
        return "HAN prerequisites are satisfied; next step is a small offline HAN prototype with held-out evaluation."
    if "minimum_labeled_queries" in blockers or "minimum_eval_queries" in blockers:
        return "Add more labeled finance query-metapath examples before implementing HAN."
    if "han_artifacts_exist" in blockers:
        return "Run bench/export_han_data.py to refresh HAN-ready graph artifacts."
    if "learned_ranker_not_worse_than_rule" in blockers:
        return "Improve features or labels before replacing the transparent rule ranker."
    return "Resolve failed gates before implementing HAN."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=str(BENCH / "han_data" / "metapath_training.jsonl"))
    parser.add_argument("--han-dir", default=str(BENCH / "han_data"))
    parser.add_argument("--min-queries", type=int, default=50)
    parser.add_argument("--min-eval-queries", type=int, default=10)
    parser.add_argument("--min-positive-metapaths", type=int, default=6)
    args = parser.parse_args()

    report = build_report(
        training_path=Path(args.training),
        han_dir=Path(args.han_dir),
        min_queries=args.min_queries,
        min_eval_queries=args.min_eval_queries,
        min_positive_metapaths=args.min_positive_metapaths,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
