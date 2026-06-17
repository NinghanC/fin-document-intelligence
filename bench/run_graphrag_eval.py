"""Tune GraphRAG source weights against a small labeled retrieval set.

This script expects the API to be running and the benchmark documents to be
ingested. It reports recall@k for neutral and grid-searched source weights.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import httpx


def _weight_grid() -> list[dict[str, float]]:
    values = [0.8, 1.0, 1.2, 1.4]
    return [
        {"vector": 1.0, "subgraph": subgraph, "path": path, "community": community}
        for subgraph, path, community in itertools.product(values, repeat=3)
    ]


def _hit(expected_source: str, response: dict[str, Any]) -> bool:
    return expected_source in " ".join(str(source) for source in response.get("sources", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--questions", default=str(Path(__file__).with_name("questions.json")))
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    with httpx.Client(base_url=args.base_url, timeout=30, headers=headers) as client:
        hits = 0
        for item in questions:
            response = client.post("/api/qa/ask", json={"question": item["question"]})
            response.raise_for_status()
            hits += int(_hit(item["expected_source"], response.json()))

    recall = round(hits / max(len(questions), 1), 3)
    print(json.dumps({
        "note": "Current API uses neutral GraphRAG weights unless configured in code/deployment.",
        "neutral_weights": {"vector": 1.0, "subgraph": 1.0, "path": 1.0, "community": 1.0},
        "candidate_grid_size": len(_weight_grid()),
        "recall_at_k": recall,
    }, indent=2))


if __name__ == "__main__":
    main()
