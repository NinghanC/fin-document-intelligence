import importlib.util
import json
from pathlib import Path

import pytest


def load_eval_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval", module_path)
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


def load_metapath_training_export_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "export_metapath_training_data.py"
    spec = importlib.util.spec_from_file_location("export_metapath_training_data", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def load_han_torch_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "train_han_torch.py"
    spec = importlib.util.spec_from_file_location("train_han_torch", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
def load_han_readiness_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "han_readiness_report.py"
    spec = importlib.util.spec_from_file_location("han_readiness_report", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def load_han_export_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "export_han_data.py"
    spec = importlib.util.spec_from_file_location("export_han_data", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def load_metapath_ranker_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "train_metapath_ranker.py"
    spec = importlib.util.spec_from_file_location("train_metapath_ranker", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def load_live_eval_module():
    module_path = Path(__file__).resolve().parents[2] / "bench" / "run_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_live_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
def test_weighted_grid_reports_best_branch_boost():
    module = load_eval_module()
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


def test_graphrag_eval_default_output_excludes_answer_text_scoring():
    module = load_eval_module()
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
    assert "answer_hit_rate" not in result
    assert "answer_smoke_hit_rate" not in result


def test_graphrag_eval_answer_smoke_is_explicit_opt_in():
    module = load_eval_module()
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
            "answer": "113%",
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

    result = module._evaluate_rrf(questions, responses, include_answer_smoke=True)

    assert result["primary_metric"] == "retrieval_hit_rate"
    assert result["answer_smoke_hit_rate"] == 1.0


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
        assert "expected_answer_terms" not in item


def test_metapath_eval_dataset_reaches_expected_paths():
    module = load_metapath_eval_module()
    dataset_path = Path(__file__).resolve().parents[2] / "bench" / "metapath_dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    result = module.asyncio.run(module._evaluate(dataset))

    assert result["total"] >= 16
    assert result["router_hit_rate"] == 1.0
    assert 0 < result["router_top1_hit_rate"] <= result["router_hit_rate"]
    assert 0 < result["average_router_precision"] <= 1.0
    assert result["average_selected_metapaths"] >= 1.0
    assert result["routed_path_hit_rate"] == 1.0
    assert result["oracle_path_hit_rate"] == 1.0
    assert result["average_routed_path_recall"] == 1.0
    assert result["average_oracle_path_recall"] == 1.0
    assert result["path_hit_rate"] == result["routed_path_hit_rate"]
    assert result["average_path_recall"] == result["average_routed_path_recall"]
    assert all(row["router_trace"] for row in result["rows"])


def test_real_holdings_eval_dataset_reaches_expected_paths():
    module = load_metapath_eval_module()
    root = Path(__file__).resolve().parents[2] / "bench" / "real_holdings"

    rows = module._read_csv(root / "holdings_sample.csv")
    questions = module._read_questions(root / "questions.json")
    result = module.asyncio.run(module._evaluate_real_holdings(rows, questions))

    assert result["holdings_rows"] >= 20
    assert result["questions"] >= 8
    assert result["router_hit_rate"] == 1.0
    assert 0 < result["router_top1_hit_rate"] <= result["router_hit_rate"]
    assert 0 < result["average_router_precision"] <= 1.0
    assert result["average_selected_metapaths"] >= 1.0
    assert result["routed_path_hit_rate"] == 1.0
    assert result["oracle_path_hit_rate"] == 1.0
    assert result["average_routed_path_recall"] == 1.0
    assert result["average_oracle_path_recall"] == 1.0
    assert result["path_hit_rate"] == result["routed_path_hit_rate"]
    assert result["average_path_recall"] == result["average_routed_path_recall"]
    assert all(row["router_trace"] for row in result["rows"])


def test_metapath_training_export_builds_pairwise_records(tmp_path):
    module = load_metapath_training_export_module()
    root = Path(__file__).resolve().parents[2]
    synthetic = json.loads((root / "bench" / "metapath_dataset.json").read_text(encoding="utf-8"))
    real_questions = json.loads((root / "bench" / "real_holdings" / "questions.json").read_text(encoding="utf-8"))

    records = module.build_training_records(synthetic, real_questions)
    summary = module.summarize(records)
    output = tmp_path / "metapath_training.jsonl"
    module.write_jsonl(records, output)
    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    expected_queries = len(synthetic["questions"]) + len(real_questions)
    assert summary["queries"] == expected_queries
    assert summary["records"] == expected_queries * summary["candidate_metapaths"]
    assert summary["positive_records"] == expected_queries
    assert summary["negative_records"] == summary["records"] - expected_queries
    assert 0 < summary["positive_top1_rate"] <= 1.0
    assert exported == records
    first = exported[0]
    assert {
        "query_id",
        "query",
        "candidate_metapath",
        "candidate_steps",
        "label",
        "router_selected",
        "router_score",
        "matched_keywords",
    } <= set(first)
    assert {record["label"] for record in exported} == {0, 1}

def test_han_export_builds_graph_artifacts(tmp_path):
    module = load_han_export_module()
    root = Path(__file__).resolve().parents[2]
    synthetic = json.loads((root / "bench" / "metapath_dataset.json").read_text(encoding="utf-8"))
    real_rows = module._read_csv(root / "bench" / "real_holdings" / "holdings_sample.csv")
    real_questions = json.loads((root / "bench" / "real_holdings" / "questions.json").read_text(encoding="utf-8"))

    artifacts = module.build_han_artifacts(synthetic, real_rows, real_questions)
    module.write_han_artifacts(artifacts, tmp_path)
    labels = [
        json.loads(line)
        for line in (tmp_path / "query_metapath_labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert artifacts["manifest"]["format"] == "han_ready_metapath_graph_v1"
    assert artifacts["manifest"]["query_count"] == len(synthetic["questions"]) + len(real_questions)
    assert artifacts["manifest"]["metapath_count"] == 8
    assert artifacts["entities"]
    assert artifacts["relations"]
    assert len(labels) == artifacts["manifest"]["query_count"]
    assert {"query_id", "positive_metapath_id", "start_entity_ids"} <= set(labels[0])
    assert (tmp_path / "adjacency_by_metapath" / "sector_exposure.json").exists()
    assert artifacts["adjacency_by_metapath"]["sector_exposure"]
    assert artifacts["adjacency_by_metapath"]["shared_sector"]

def test_torch_han_ranker_uses_optional_pytorch_and_graph_artifacts(tmp_path):
    pytest.importorskip("torch")
    export_module = load_metapath_training_export_module()
    han_export_module = load_han_export_module()
    torch_module = load_han_torch_module()
    root = Path(__file__).resolve().parents[2]
    synthetic = json.loads((root / "bench" / "metapath_dataset.json").read_text(encoding="utf-8"))
    real_questions = json.loads((root / "bench" / "real_holdings" / "questions.json").read_text(encoding="utf-8"))
    real_rows = export_module._read_csv(root / "bench" / "real_holdings" / "holdings_sample.csv")

    records = export_module.build_training_records(synthetic, real_questions, real_rows)
    training_path = tmp_path / "metapath_training.jsonl"
    export_module.write_jsonl(records, training_path)
    artifacts = han_export_module.build_han_artifacts(synthetic, real_rows, real_questions)
    han_export_module.write_han_artifacts(artifacts, tmp_path)

    train_records, eval_records = torch_module.ranker.split_records(
        torch_module.ranker.load_jsonl(training_path),
        "synthetic_finance_graph",
        "real_13f_style_holdings",
    )
    han_context = torch_module.load_han_context(tmp_path)
    model, feature_names, metapath_ids = torch_module.train_torch_han_ranker(
        train_records,
        han_context,
        epochs=60,
        learning_rate=0.01,
        hidden_dim=16,
    )
    result = torch_module.evaluate_torch_han_ranker(eval_records, model, han_context, feature_names, metapath_ids)
    attention = torch_module.attention_summary(model)

    assert result["queries"] == len(real_questions)
    assert result["torch_top1_hit_rate"] >= result["rule_top1_hit_rate"]
    assert result["torch_mrr"] >= result["rule_mrr"]
    assert attention["mean_abs_attention_gate_weight"] > 0
    assert attention["mean_metapath_embedding_norm"] > 0

    report_path = tmp_path / "torch_han_eval.json"
    model_path = tmp_path / "torch_han_model.pt"
    torch_module.write_json(report_path, result)
    torch_module.save_model_artifact(model_path, model, feature_names, metapath_ids, result, hidden_dim=16)

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["torch_top1_hit_rate"] == result["torch_top1_hit_rate"]
    assert model_path.exists()

def test_han_readiness_report_blocks_small_labeled_sets(tmp_path):
    export_module = load_metapath_training_export_module()
    han_module = load_han_export_module()
    readiness_module = load_han_readiness_module()
    root = Path(__file__).resolve().parents[2]
    synthetic = json.loads((root / "bench" / "metapath_dataset.json").read_text(encoding="utf-8"))
    real_questions = json.loads((root / "bench" / "real_holdings" / "questions.json").read_text(encoding="utf-8"))
    real_rows = export_module._read_csv(root / "bench" / "real_holdings" / "holdings_sample.csv")

    records = export_module.build_training_records(synthetic, real_questions, real_rows)
    training_path = tmp_path / "metapath_training.jsonl"
    export_module.write_jsonl(records, training_path)
    artifacts = han_module.build_han_artifacts(synthetic, real_rows, real_questions)
    han_module.write_han_artifacts(artifacts, tmp_path)

    conservative = readiness_module.build_report(training_path, tmp_path, min_queries=80, min_eval_queries=20)
    relaxed = readiness_module.build_report(training_path, tmp_path, min_queries=50, min_eval_queries=10)

    assert conservative["ready_for_han"] is False
    assert "minimum_labeled_queries" in conservative["blockers"]
    assert "minimum_eval_queries" in conservative["blockers"]
    assert conservative["gates"]["learned_ranker_not_worse_than_rule"] is True
    assert relaxed["ready_for_han"] is True

def test_metapath_ranker_baseline_beats_or_matches_rule_router():
    export_module = load_metapath_training_export_module()
    ranker_module = load_metapath_ranker_module()
    root = Path(__file__).resolve().parents[2]
    synthetic = json.loads((root / "bench" / "metapath_dataset.json").read_text(encoding="utf-8"))
    real_questions = json.loads((root / "bench" / "real_holdings" / "questions.json").read_text(encoding="utf-8"))
    real_rows = export_module._read_csv(root / "bench" / "real_holdings" / "holdings_sample.csv")
    records = export_module.build_training_records(synthetic, real_questions, real_rows)

    train_records, eval_records = ranker_module.split_records(
        records,
        "synthetic_finance_graph",
        "real_13f_style_holdings",
    )
    weights = ranker_module.train_pairwise_ranker(train_records, epochs=40, learning_rate=0.03)
    result = ranker_module.evaluate_ranker(eval_records, weights)

    assert result["queries"] == len(real_questions)
    assert result["learned_top1_hit_rate"] >= result["rule_top1_hit_rate"]
    assert result["learned_mrr"] >= result["rule_mrr"]
    assert ranker_module.top_weights(weights)

def test_live_eval_questions_exclude_graph_inference_scope():
    questions_path = Path(__file__).resolve().parents[2] / "bench" / "live_eval" / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    assert questions
    assert all(item["answer_type"] != "graph_inference" for item in questions)
    assert {"factoid", "table", "insufficient"} <= {item["answer_type"] for item in questions}
def test_live_eval_scores_grounded_answers_and_insufficient_cases():
    module = load_live_eval_module()
    questions = [
        {
            "id": "factoid",
            "answer_type": "factoid",
            "question": "What liquidity coverage ratio did JPMorgan Chase report for 2023?",
            "expected_sources": ["jpmorgan_2023_annual_report.pdf"],
            "expected_evidence_terms": ["liquidity coverage ratio", "113"],
            "expected_answer_points": ["113"],
        },
        {
            "id": "insufficient",
            "answer_type": "insufficient",
            "question": "What is the private internal trading limit?",
            "expected_sources": [],
            "expected_evidence_terms": [],
            "expected_answer_points": [],
        },
    ]
    responses = [
        {
            "answer": "JPMorgan Chase reported an average liquidity coverage ratio of 113% for 2023.",
            "sources": [
                {
                    "source": "jpmorgan_2023_annual_report.pdf",
                    "snippet": "Liquidity coverage ratio (average) 2023 113.",
                }
            ],
            "retrieval_quality": 0.9,
            "intent": "factoid",
        },
        {
            "answer": "The retrieved context is insufficient to determine that private internal limit.",
            "sources": [],
            "retrieval_quality": 0.0,
            "intent": "factoid",
        },
    ]

    result = module.evaluate(questions, responses)

    assert result["evaluation"] == "live_provider_answer_grounding"
    assert result["summary"]["pass_rate"] == 1.0
    assert result["summary"]["source_hit_rate"] == 1.0
    assert result["summary"]["evidence_hit_rate"] == 1.0
    assert result["summary"]["answer_point_hit_rate"] == 1.0
    assert result["summary"]["insufficient_hit_rate"] == 1.0
    assert result["summary"]["by_answer_type"]["factoid"]["pass_rate"] == 1.0
    assert result["summary"]["by_answer_type"]["insufficient"]["pass_rate"] == 1.0

def test_live_eval_accepts_bedrock_env_file(tmp_path, monkeypatch):
    module = load_live_eval_module()
    env_file = tmp_path / "bedrock.env"
    env_file.write_text(
        "MODEL_PROVIDER=bedrock\nAWS_REGION=us-east-1\nBEDROCK_MODEL_ID=amazon.nova-lite-v1:0\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    module._load_env_file(str(env_file))

    assert module._provider_configured() is True


def test_live_eval_insufficient_accepts_context_does_not_contain_phrase():
    module = load_live_eval_module()
    result = module.evaluate(
        [
            {
                "id": "insufficient",
                "answer_type": "insufficient",
                "question": "What is the private internal stress-loss limit?",
                "expected_sources": [],
                "expected_evidence_terms": [],
                "expected_answer_points": [],
            }
        ],
        [
            {
                "answer": "The provided context does not contain that private internal limit.",
                "sources": [],
                "retrieval_quality": 0.0,
                "intent": "factoid",
            }
        ],
    )

    assert result["summary"]["pass_rate"] == 1.0
    assert result["items"][0]["insufficient_hit"] is True
