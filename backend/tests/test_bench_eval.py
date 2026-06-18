import importlib.util
import json
from pathlib import Path


def load_graphrag_eval_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "run_graphrag_eval.py"
    spec = importlib.util.spec_from_file_location("run_graphrag_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_weighted_grid_reports_best_branch_boost():
    module = load_graphrag_eval_module()
    questions = [
        {
            "question": "Which fund mentions liquidity?",
            "expected_source": "graph-source.md",
            "expected_terms": ["liquidity"],
        }
    ]
    responses = [
        {
            "answer": "liquidity",
            "sources": [
                {"source": "vector-source.md", "score": 0.9, "type": "vector"},
                {"source": "graph-source.md", "score": 0.7, "type": "graph"},
            ],
        }
    ]

    result = module._evaluate_weighted_grid(questions, responses, top_k=1)

    assert result["mode"] == "weighted-grid"
    assert result["best"]["weights"]["graph"] > result["best"]["weights"]["vector"]
    assert result["best"]["hit_rate"] == 1.0


def test_public_demo_questions_are_labeled():
    questions_path = Path(__file__).resolve().parents[2] / "bench" / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    assert len(questions) >= 4
    expected_sources = {item["expected_source"] for item in questions}
    assert {
        "jpmorgan_2023_annual_report.pdf",
        "microsoft_2023_10k.pdf",
        "apple_2023_10k.pdf",
    } <= expected_sources
    for item in questions:
        assert item["question"]
        assert item["expected_source"]
        assert item["expected_terms"]
