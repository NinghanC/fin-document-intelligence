from __future__ import annotations

import importlib.util
from pathlib import Path


def load_check_bedrock_config_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "check_bedrock_config.py"
    spec = importlib.util.spec_from_file_location("check_bedrock_config", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bedrock_config_dry_run_passes_with_required_values():
    module = load_check_bedrock_config_module()

    result = module.evaluate_config(
        {
            "MODEL_PROVIDER": "bedrock",
            "AWS_REGION": "us-east-1",
            "BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "AWS_PROFILE": "default",
        },
        boto3_available=True,
        credentials_available=True,
    )

    assert result["passed"] is True


def test_bedrock_config_dry_run_rejects_placeholders_and_missing_credentials():
    module = load_check_bedrock_config_module()

    result = module.evaluate_config(
        {
            "MODEL_PROVIDER": "openai_compatible",
            "AWS_REGION": "",
            "BEDROCK_MODEL_ID": "gpt-4o",
        },
        boto3_available=False,
        credentials_available=False,
    )

    failed_checks = {check["name"] for check in result["checks"] if not check["passed"]}

    assert result["passed"] is False
    assert {
        "model_provider_is_bedrock",
        "boto3_installed",
        "aws_region_configured",
        "bedrock_model_id_configured",
        "aws_credentials_visible",
    } <= failed_checks
