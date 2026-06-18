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


def load_metapath_eval_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "run_metapath_eval.py"
    spec = importlib.util.spec_from_file_location("run_metapath_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_real_holdings_eval_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "run_real_holdings_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_holdings_eval", module_path)
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
                {"source": "vector-source.md", "snippet": "general fund overview", "score": 0.9, "type": "vector"},
                {"source": "graph-source.md", "snippet": "liquidity controls", "score": 0.7, "type": "graph"},
            ],
        }
    ]

    result = module._evaluate_weighted_grid(questions, responses, top_k=1)

    assert result["mode"] == "weighted-grid"
    assert result["best"]["weights"]["graph"] > result["best"]["weights"]["vector"]
    assert result["best"]["hit_rate"] == 1.0


def test_graphrag_eval_primary_metric_ignores_answer_text():
    module = load_graphrag_eval_module()
    questions = [
        {
            "question": "What liquidity coverage ratio did JPMorgan report?",
            "expected_source": "jpmorgan.pdf",
            "expected_terms": ["liquidity coverage ratio", "113"],
            "expected_answer_terms": ["113%"],
        }
    ]
    responses = [
        {
            "answer": "wrong answer from a demo model",
            "sources": [
                {
                    "source": "jpmorgan.pdf",
                    "snippet": "Liquidity coverage ratio average was 113 for 2023.",
                    "type": "vector",
                    "score": 0.9,
                }
            ],
        }
    ]

    result = module._evaluate_rrf(questions, responses)

    assert result["primary_metric"] == "retrieval_hit_rate"
    assert result["hit_rate"] == 1.0
    assert result["answer_hit_rate"] == 0.0


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


def test_metapath_eval_dataset_reaches_expected_paths():
    module = load_metapath_eval_module()
    dataset_path = Path(__file__).resolve().parents[2] / "bench" / "metapath_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    result = module.asyncio.run(module._evaluate(dataset))

    assert result["total"] >= 16
    assert result["router_hit_rate"] == 1.0
    assert result["path_hit_rate"] == 1.0
    assert result["average_path_recall"] == 1.0


def test_real_holdings_eval_dataset_reaches_expected_paths():
    module = load_real_holdings_eval_module()
    root = Path(__file__).resolve().parents[2] / "bench" / "real_holdings"

    rows = module._read_csv(root / "holdings_sample.csv")
    questions = module._read_questions(root / "questions.json")
    result = module.asyncio.run(module._evaluate(rows, questions))

    assert result["holdings_rows"] >= 20
    assert result["questions"] >= 8
    assert result["router_hit_rate"] == 1.0
    assert result["path_hit_rate"] == 1.0
    assert result["average_path_recall"] == 1.0
