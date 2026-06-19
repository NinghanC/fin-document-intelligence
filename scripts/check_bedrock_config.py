"""Dry-run AWS Bedrock configuration checks.

This script intentionally does not call Bedrock. It validates local
configuration, dependency availability, and AWS credential visibility so a
developer can catch setup mistakes before running live provider tests.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PLACEHOLDER_MODEL_IDS = {"", "gpt-4o", "your-model", "your-bedrock-model-id"}


def _load_env(env_file: Path) -> dict[str, str]:
    values = {key: str(value) for key, value in dotenv_values(env_file).items() if value is not None}
    merged = {**values, **os.environ}
    return {key: str(value) for key, value in merged.items()}


def evaluate_config(
    env: dict[str, str],
    *,
    boto3_available: bool,
    credentials_available: bool,
) -> dict[str, Any]:
    provider = env.get("MODEL_PROVIDER", "openai_compatible").strip().lower()
    region = env.get("AWS_REGION", "").strip()
    profile = env.get("AWS_PROFILE", "").strip()
    model_id = env.get("BEDROCK_MODEL_ID", "").strip()

    checks = [
        {
            "name": "model_provider_is_bedrock",
            "passed": provider == "bedrock",
            "detail": f"MODEL_PROVIDER={provider or '<empty>'}",
        },
        {
            "name": "boto3_installed",
            "passed": boto3_available,
            "detail": "boto3 import is available" if boto3_available else "Install backend requirements to add boto3",
        },
        {
            "name": "aws_region_configured",
            "passed": bool(region),
            "detail": f"AWS_REGION={region or '<empty>'}",
        },
        {
            "name": "bedrock_model_id_configured",
            "passed": bool(model_id) and model_id not in PLACEHOLDER_MODEL_IDS,
            "detail": f"BEDROCK_MODEL_ID={model_id or '<empty>'}",
        },
        {
            "name": "aws_credentials_visible",
            "passed": credentials_available,
            "detail": f"AWS_PROFILE={profile or '<default credential chain>'}",
        },
    ]
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }


def _credentials_available(env: dict[str, str]) -> bool:
    try:
        import boto3
    except ImportError:
        return False

    session_kwargs: dict[str, str] = {}
    profile = env.get("AWS_PROFILE", "").strip()
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    return session.get_credentials() is not None


def _boto3_available() -> bool:
    try:
        import boto3  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="backend/.env")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    args = parser.parse_args()

    env_file = Path(args.env_file)
    env = _load_env(env_file) if env_file.exists() else dict(os.environ)
    result = evaluate_config(
        env,
        boto3_available=_boto3_available(),
        credentials_available=_credentials_available(env),
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"Bedrock config dry-run: {status}")
        for check in result["checks"]:
            marker = "ok" if check["passed"] else "missing"
            print(f"- {marker}: {check['name']} ({check['detail']})")

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
