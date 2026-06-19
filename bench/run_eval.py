"""Small retrieval evaluation harness for the public prototype.

Run while the API is available:
    python bench/run_eval.py --base-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx


def _matches(item: dict[str, Any], response: dict[str, Any]) -> bool:
    sources = response.get("sources", [])
    source_blob = " ".join(str(source) for source in sources)
    expected_source = item["expected_source"]
    terms = item.get("expected_terms", [])
    return expected_source in source_blob and all(term.lower() in source_blob.lower() for term in terms)


def _answer_matches(item: dict[str, Any], response: dict[str, Any]) -> bool:
    answer = response.get("answer", "")
    return all(term.lower() in answer.lower() for term in item.get("expected_answer_terms", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--questions", default=str(Path(__file__).with_name("questions.json")))
    parser.add_argument("--include-answer-smoke", action="store_true")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    passed = 0
    answer_passed = 0

    with httpx.Client(base_url=args.base_url, timeout=30, headers=headers) as client:
        try:
            for item in questions:
                response = client.post("/api/qa/ask", json={"question": item["question"]})
                response.raise_for_status()
                data = response.json()
                ok = _matches(item, data)
                answer_ok = _answer_matches(item, data) if args.include_answer_smoke else False
                passed += int(ok)
                answer_passed += int(answer_ok)
                answer_status = f" | {'PASS' if answer_ok else 'FAIL'} answer-smoke" if args.include_answer_smoke else ""
                print(f"{'PASS' if ok else 'FAIL'} retrieval{answer_status} | {item['question']}")
        except httpx.HTTPError as exc:
            print(
                f"Evaluation API request failed: {exc}. "
                f"Start the API first or pass --base-url. Current base URL: {args.base_url}",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc

    total = len(questions)
    result = {
        "primary_metric": "retrieval_hit_rate",
        "note": "Generated answer text is not scored by default; use --include-answer-smoke for an optional smoke check.",
        "total": total,
        "passed": passed,
        "hit_rate": round(passed / max(total, 1), 3),
    }
    if args.include_answer_smoke:
        result["answer_smoke_passed"] = answer_passed
        result["answer_smoke_hit_rate"] = round(answer_passed / max(total, 1), 3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()