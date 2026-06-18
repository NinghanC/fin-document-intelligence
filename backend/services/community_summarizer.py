"""Offline community summarizers for precomputed graph summaries."""

from __future__ import annotations

import re
from typing import Protocol

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from utils.model_clients import create_chat_model, has_provider_key

logger = structlog.get_logger("finsight.community_summarizer")


class CommunitySummarizer(Protocol):
    async def summarize(self, members: list[str], relations: list[str]) -> str:
        """Return a summary for one detected graph community."""


class StructuredCommunitySummarizer:
    """Deterministic offline summarizer used for CI and local demos."""

    async def summarize(self, members: list[str], relations: list[str]) -> str:
        return self.format(members, relations)

    @staticmethod
    def format(members: list[str], relations: list[str]) -> str:
        member_text = ", ".join(members[:8])
        if not relations:
            return (
                "Graph community summary: "
                f"{member_text}. No direct intra-community relationships were captured."
            )

        relation_types = []
        for relation in relations:
            match = re.search(r"-\[(?P<type>[A-Z_]+)\]->", relation)
            if match:
                relation_types.append(match.group("type").replace("_", " ").lower())
        themes = ", ".join(sorted(set(relation_types))[:5]) or "captured relationships"
        relation_text = "; ".join(relations[:8])
        return (
            "Graph community summary: "
            f"{member_text}. Main relationship themes: {themes}. "
            f"Representative evidence: {relation_text}."
        )


class LLMCommunitySummarizer:
    """Provider-backed offline summarizer for ingestion/update time."""

    SYSTEM_PROMPT = """\
You summarize detected communities from a financial knowledge graph.

Rules:
- Use only the provided entities and relationships.
- Do not invent facts, metrics, risks, or relationships.
- Be concise and source-grounded.
- Mention relationship themes and why the cluster may matter for financial analysis.
"""

    def __init__(self) -> None:
        self.llm = create_chat_model()

    async def summarize(self, members: list[str], relations: list[str]) -> str:
        relation_text = "\n".join(f"- {relation}" for relation in relations[:50]) or "- No direct relationships"
        member_text = "\n".join(f"- {member}" for member in members[:50])
        response = await self.llm.ainvoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"Entities:\n{member_text}\n\nRelationships:\n{relation_text}"),
        ])
        return str(response.content).strip()


def create_community_summarizer() -> CommunitySummarizer:
    provider = settings.community_summary_provider.strip().lower()
    if provider == "llm":
        if has_provider_key():
            return LLMCommunitySummarizer()
        logger.warning("community_summary_llm_requested_without_provider_key_using_structured")
    return StructuredCommunitySummarizer()
