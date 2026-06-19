"""Provider-backed live evaluation for answer quality and grounding.

This runner calls a running API. It is intentionally outside default CI because
it needs ingested documents plus a real model provider behind the API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

INSUFFICIENT_TERMS = ("insufficient", "not enough", "not available", "cannot determine", "no evidence")


def _blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def _contains_all(blob: str, terms: list[str]) -> bool:
    return all(term.lower() in blob for term in terms)


def _source_hit(item: dict[str, Any], response: dict[str, Any]) -> bool:
    expected_sources = item.get("expected_sources", [])
    if not expected_sources:
        return True
    source_blob = _blob(response.get("sources", []))
    return any(source.lower() in source_blob for source in expected_sources)


def _evidence_hit(item: dict[str, Any], response: dict[str, Any]) -> bool:
    terms = item.get("expected_evidence_terms", [])
    if not terms:
        return True
    return _contains_all(_blob(response.get("sources", [])), terms)


def _answer_point_hit(item: dict[str, Any], response: dict[str, Any]) -> bool:
    points = item.get("expected_answer_points", [])
    if not points:
        return True
    return _contains_all(str(response.get("answer", "")).lower(), points)


def _insufficient_hit(item: dict[str, Any], response: dict[str, Any]) -> bool:
    if item.get("answer_type") != "insufficient":
        return True
    answer = str(response.get("answer", "")).lower()
    return any(term in answer for term in INSUFFICIENT_TERMS)


def _score_item(item: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    source_hit = _source_hit(item, response)
    evidence_hit = _evidence_hit(item, response)
    answer_point_hit = _answer_point_hit(item, response)
    insufficient_hit = _insufficient_hit(item, response)
    passed = source_hit and evidence_hit and answer_point_hit and insufficient_hit
    return {
        "id": item.get("id", item.get("question", "")),
        "answer_type": item.get("answer_type", "unknown"),
        "question": item.get("question", ""),
        "source_hit": source_hit,
        "evidence_hit": evidence_hit,
        "answer_point_hit": answer_point_hit,
        "insufficient_hit": insufficient_hit,
        "passed": passed,
        "retrieval_quality": response.get("retrieval_quality"),
        "intent": response.get("intent"),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(int(bool(row[key])) for row in rows) / max(len(rows), 1), 3)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["answer_type"])].append(row)

    return {
        "total": len(rows),
        "pass_rate": _rate(rows, "passed"),
        "source_hit_rate": _rate(rows, "source_hit"),
        "evidence_hit_rate": _rate(rows, "evidence_hit"),
        "answer_point_hit_rate": _rate(rows, "answer_point_hit"),
        "insufficient_hit_rate": _rate(rows, "insufficient_hit"),
        "by_answer_type": {
            answer_type: {
                "total": len(type_rows),
                "pass_rate": _rate(type_rows, "passed"),
                "source_hit_rate": _rate(type_rows, "source_hit"),
                "evidence_hit_rate": _rate(type_rows, "evidence_hit"),
                "answer_point_hit_rate": _rate(type_rows, "answer_point_hit"),
            }
            for answer_type, type_rows in sorted(by_type.items())
        },
    }


def evaluate(
    questions: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [_score_item(item, response) for item, response in zip(questions, responses, strict=False)]
    return {
        "evaluation": "live_provider_answer_grounding",
        "note": "Requires a real provider-backed API. Scores answer points and source grounding; deterministic demo-model results should not be reported as live eval.",
        "summary": _summarize(rows),
        "items": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "live_eval" / "questions.json"))
    parser.add_argument("--allow-demo", action="store_true", help="Allow running without local provider env vars.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.allow_demo and not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set locally. If the target API is already provider-backed, rerun with --allow-demo.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    responses: list[dict[str, Any]] = []

    with httpx.Client(base_url=args.base_url, timeout=60, headers=headers) as client:
        try:
            for item in questions:
                response = client.post("/api/qa/ask", json={"question": item["question"]})
                response.raise_for_status()
                responses.append(response.json())
        except httpx.HTTPError as exc:
            print(
                f"Live evaluation request failed: {exc}. Start the API, ingest the eval documents, and pass --base-url.",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc

    result = evaluate(questions, responses)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
