"""Retrieval evaluation harness for the public prototype.

Run while the API is available:
    python bench/run_eval.py --base-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import httpx

GRID_VALUES = [0.8, 1.0, 1.2, 1.4]


def _retrieval_hit(item: dict[str, Any], response: dict[str, Any]) -> bool:
    source_blob = " ".join(str(source) for source in response.get("sources", []))
    terms = item.get("expected_terms", [])
    return item["expected_source"] in source_blob and all(term.lower() in source_blob.lower() for term in terms)


def _answer_hit(item: dict[str, Any], response: dict[str, Any]) -> bool:
    answer_blob = str(response.get("answer", ""))
    return all(term.lower() in answer_blob.lower() for term in item.get("expected_answer_terms", []))


def _source_hit(item: dict[str, Any], sources: list[dict[str, Any]]) -> bool:
    source_blob = " ".join(str(source) for source in sources)
    terms = item.get("expected_terms", [])
    return item["expected_source"] in source_blob and all(term.lower() in source_blob.lower() for term in terms)


def _weighted_sources(sources: list[dict[str, Any]], weights: dict[str, float], top_k: int) -> list[dict[str, Any]]:
    return sorted(
        sources,
        key=lambda source: float(source.get("score", 0.0)) * weights.get(str(source.get("type", "")), 1.0),
        reverse=True,
    )[:top_k]


def _weight_grid() -> list[dict[str, float]]:
    return [
        {"vector": vector_weight, "graph": graph_weight}
        for vector_weight, graph_weight in itertools.product(GRID_VALUES, repeat=2)
    ]


def _evaluate_rrf(
    questions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    include_answer_smoke: bool = False,
) -> dict[str, Any]:
    retrieval_hits = sum(int(_retrieval_hit(item, response)) for item, response in zip(questions, responses, strict=False))
    result = {
        "mode": "rrf",
        "fusion": "reciprocal_rank_fusion",
        "primary_metric": "retrieval_hit_rate",
        "note": "Retrieval hit rate checks expected source and evidence terms in returned sources only. Generated answer text is not scored by default.",
        "total": len(questions),
        "expected_source_hits": retrieval_hits,
        "hit_rate": round(retrieval_hits / max(len(questions), 1), 3),
    }
    if include_answer_smoke:
        answer_hits = sum(int(_answer_hit(item, response)) for item, response in zip(questions, responses, strict=False))
        result["answer_smoke_hits"] = answer_hits
        result["answer_smoke_hit_rate"] = round(answer_hits / max(len(questions), 1), 3)
        result["answer_smoke_note"] = "Optional smoke check only; do not present as retrieval or answer-quality evaluation."
    return result


def _evaluate_weighted_grid(
    questions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for weights in _weight_grid():
        hits = 0
        for item, response in zip(questions, responses, strict=False):
            hits += int(_source_hit(item, _weighted_sources(response.get("sources", []), weights, top_k)))
        candidates.append({
            "weights": weights,
            "expected_source_hits": hits,
            "hit_rate": round(hits / max(len(questions), 1), 3),
        })
    candidates.sort(key=lambda item: (item["hit_rate"], item["expected_source_hits"]), reverse=True)
    return {
        "mode": "weighted-grid",
        "note": "Experimental post-fusion branch boost over API-returned sources; use a candidate-level export before adopting weights in production.",
        "source_types": ["vector", "graph"],
        "top_k": top_k,
        "grid_size": len(candidates),
        "best": candidates[0] if candidates else {},
        "all_results": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--questions", default=str(Path(__file__).with_name("questions.json")))
    parser.add_argument("--mode", choices=["rrf", "weighted-grid", "both"], default="rrf")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-answer-smoke", action="store_true")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    with httpx.Client(base_url=args.base_url, timeout=30, headers=headers) as client:
        responses: list[dict[str, Any]] = []
        try:
            for item in questions:
                response = client.post("/api/qa/ask", json={"question": item["question"]})
                response.raise_for_status()
                responses.append(response.json())
        except httpx.HTTPError as exc:
            print(
                f"Evaluation API request failed: {exc}. "
                f"Start the API first or pass --base-url. Current base URL: {args.base_url}",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc

    if args.mode == "rrf":
        result = _evaluate_rrf(questions, responses, include_answer_smoke=args.include_answer_smoke)
    elif args.mode == "weighted-grid":
        result = _evaluate_weighted_grid(questions, responses, top_k=args.top_k)
    else:
        result = {
            "mode": "both",
            "rrf": _evaluate_rrf(questions, responses, include_answer_smoke=args.include_answer_smoke),
            "weighted_grid": _evaluate_weighted_grid(questions, responses, top_k=args.top_k),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
