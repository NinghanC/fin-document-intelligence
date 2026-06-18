"""Evaluate GraphRAG retrieval against a small labeled set.

This script expects the API to be running and the benchmark documents to be
ingested. The production path uses RRF. The weighted-grid mode is an experiment
over returned source branches, useful for deciding whether a learned or tuned
branch boost is worth implementing later.
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


def _hit(item: dict[str, Any], response: dict[str, Any]) -> bool:
    source_blob = " ".join(str(source) for source in response.get("sources", []))
    answer_blob = str(response.get("answer", ""))
    terms = item.get("expected_terms", [])
    return item["expected_source"] in source_blob and all(
        term.lower() in (answer_blob + source_blob).lower()
        for term in terms
    )


def _source_hit(item: dict[str, Any], sources: list[dict[str, Any]], answer: str = "") -> bool:
    source_blob = " ".join(str(source) for source in sources)
    terms = item.get("expected_terms", [])
    return item["expected_source"] in source_blob and all(
        term.lower() in (answer + source_blob).lower()
        for term in terms
    )


def _weighted_sources(
    sources: list[dict[str, Any]],
    weights: dict[str, float],
    top_k: int,
) -> list[dict[str, Any]]:
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


def _evaluate_rrf(questions: list[dict[str, Any]], responses: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(int(_hit(item, response)) for item, response in zip(questions, responses, strict=False))
    return {
        "mode": "rrf",
        "fusion": "reciprocal_rank_fusion",
        "total": len(questions),
        "expected_source_hits": hits,
        "hit_rate": round(hits / max(len(questions), 1), 3),
    }


def _evaluate_weighted_grid(
    questions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for weights in _weight_grid():
        hits = 0
        for item, response in zip(questions, responses, strict=False):
            reranked = _weighted_sources(response.get("sources", []), weights, top_k)
            hits += int(_source_hit(item, reranked, str(response.get("answer", ""))))
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
                f"GraphRAG evaluation request failed: {exc}. "
                f"Start the API first or pass --base-url. Current base URL: {args.base_url}",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc

    results: dict[str, Any]
    if args.mode == "rrf":
        results = _evaluate_rrf(questions, responses)
    elif args.mode == "weighted-grid":
        results = _evaluate_weighted_grid(questions, responses, top_k=args.top_k)
    else:
        results = {
            "mode": "both",
            "rrf": _evaluate_rrf(questions, responses),
            "weighted_grid": _evaluate_weighted_grid(questions, responses, top_k=args.top_k),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
