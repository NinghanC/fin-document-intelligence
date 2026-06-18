"""Optional live LLM validation.

These tests are skipped unless RUN_LIVE_LLM_TESTS=1 and a real provider key is
configured. They are not part of the deterministic CI suite; their purpose is
to catch drift between the regex demo model and a real chat model.
"""

from __future__ import annotations

import json
import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agents.knowledge_extract_agent import EXTRACTION_SYSTEM_PROMPT, KnowledgeExtractAgent
from agents.qa_agent import INTENT_PROMPT
from utils.model_clients import create_chat_model, has_provider_key

pytestmark = pytest.mark.live_llm


def _require_live_llm() -> None:
    if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_LLM_TESTS=1 to run provider-backed LLM smoke tests")
    if not has_provider_key():
        pytest.skip("OPENAI_API_KEY is not configured with a real provider key")


def _json_from_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(cleaned)


@pytest.mark.asyncio
async def test_live_llm_classifies_factoid_intent() -> None:
    _require_live_llm()
    llm = create_chat_model()

    response = await llm.ainvoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content="What liquidity coverage ratio did JPMorgan Chase report for 2023?"),
    ])

    assert str(response.content).strip().lower() == "factoid"


@pytest.mark.asyncio
async def test_live_llm_extracts_finance_entities_and_relations_as_json() -> None:
    _require_live_llm()
    llm = create_chat_model()
    passage = (
        "Global Income Fund holds Microsoft and JPMorgan Chase. "
        "Microsoft belongs to the Technology sector. "
        "JPMorgan Chase belongs to the Financials sector and is subject to Basel III."
    )

    response = await llm.ainvoke([
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract knowledge from the following text:\n\n{passage}"),
    ])
    data = _json_from_response(str(response.content))

    parsed = KnowledgeExtractAgent()._parse_response(json.dumps(data), "live-smoke")
    entity_names = {str(entity.get("name", "")) for entity in data.get("entities", [])}
    relation_names = {str(relation.get("relation", "")) for relation in data.get("relations", [])}

    assert {"Global Income Fund", "Microsoft", "JPMorgan Chase"} <= entity_names
    assert {"holds", "belongs_to"} & relation_names
    assert parsed.source_chunk_id == "live-smoke"


@pytest.mark.asyncio
async def test_live_llm_grounded_answer_uses_supplied_context() -> None:
    _require_live_llm()
    llm = create_chat_model()
    context = (
        "Context information:\n"
        "[Source 1: jpmorgan_2023_annual_report.pdf | Type: vector | Score: 0.95]\n"
        "Liquidity coverage ratio (average): 2023 113, 2022 112, 2021 111.\n\n"
        "User question: What liquidity coverage ratio did JPMorgan Chase report for 2023?\n"
        "Answer with the value and cite the source name."
    )

    response = await llm.ainvoke([HumanMessage(content=context)])
    answer = str(response.content)

    assert "113" in answer
    assert "jpmorgan_2023_annual_report.pdf" in answer.lower()
