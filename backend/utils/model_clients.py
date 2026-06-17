"""Model client helpers used by the public demo and provider-backed deployments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import settings

PLACEHOLDER_KEYS = {
    "",
    "your-key",
    "your-provider-api-key",
    "your-azure-openai-key",
    "your-aws-or-gateway-key",
    "your-databricks-token",
}


def has_provider_key() -> bool:
    return settings.openai_api_key.strip() not in PLACEHOLDER_KEYS


def create_chat_model() -> Any:
    """Return a provider-backed chat model, or a deterministic demo model."""
    if has_provider_key():
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=SecretStr(settings.openai_api_key),
            base_url=settings.openai_base_url,
            temperature=0,
        )
    return DemoChatModel()


@dataclass
class DemoChatModel:
    """Small deterministic model for offline demo and automated tests."""

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        system = self._content(messages[0]) if messages else ""
        user = self._content(messages[-1]) if messages else ""

        if "query intent classifier" in system:
            return AIMessage(content=self._classify_intent(user))
        if "query rewriting expert" in system:
            return AIMessage(content=json.dumps(self._rewrite(user)))
        if "Cypher query generation" in system:
            return AIMessage(content=json.dumps({"queries": []}))
        if "Extract all possible entity names" in system:
            return AIMessage(content=json.dumps({"entities": self._entities(user)}))
        if "knowledge extraction engine" in system:
            return AIMessage(content=json.dumps(self._extract(user)))
        if "knowledge graph analysis expert" in system:
            return AIMessage(content=self._summarize(user))
        if "Describe the image" in user or "Describe all content" in user:
            return AIMessage(content="Image content was captured for document intelligence processing.")
        return AIMessage(content=self._answer(user))

    @staticmethod
    def _content(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            return " ".join(str(item) for item in content)
        return str(content)

    @staticmethod
    def _entities(text: str) -> list[str]:
        candidates = [
            "Global Income Fund",
            "duration risk",
            "interest rates",
            "credit spread",
            "liquidity buffer",
            "portfolio manager",
            "risk committee",
        ]
        lowered = text.lower()
        found = [name for name in candidates if name.lower() in lowered]
        if found:
            return found
        words = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b", text)
        return list(dict.fromkeys(words))[:5]

    def _rewrite(self, question: str) -> dict[str, Any]:
        entities = self._entities(question)
        keywords = re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())[:8]
        return {
            "queries": [question, " ".join(entities + keywords).strip() or question],
            "entities": entities,
            "keywords": keywords,
        }

    @staticmethod
    def _classify_intent(question: str) -> str:
        lowered = question.lower()
        if any(word in lowered for word in ("compare", "versus", "difference")):
            return "comparative"
        if any(word in lowered for word in ("why", "impact", "risk", "explain")):
            return "analytical"
        if any(word in lowered for word in ("how", "steps", "process")):
            return "procedural"
        return "factoid"

    def _extract(self, user: str) -> dict[str, Any]:
        text = user.split("\n\n", 1)[-1]
        entity_names = self._entities(text)
        entities = [
            {
                "name": name,
                "type": "Product" if "Fund" in name else "Concept",
                "description": f"Entity mentioned in the ingested financial document: {name}",
            }
            for name in entity_names
        ]
        relations = []
        if "Global Income Fund" in entity_names:
            for tail in entity_names:
                if tail != "Global Income Fund":
                    relations.append({
                        "head": "Global Income Fund",
                        "relation": "related_to",
                        "tail": tail,
                        "confidence": 0.82,
                    })
        return {"entities": entities, "relations": relations, "events": []}

    @staticmethod
    def _summarize(user: str) -> str:
        body = user.split("Subgraph information:", 1)[-1].strip()
        digest = sha256(body.encode("utf-8")).hexdigest()[:8]
        return f"Community summary ({digest}): related fund-risk entities form a connected reasoning context."

    @staticmethod
    def _answer(user: str) -> str:
        if "Context information:" in user:
            context = user.split("Context information:", 1)[1].split("User question:", 1)[0].strip()
            first_line = next((line for line in context.splitlines() if line and not line.startswith("[Source")), "")
            return (
                "Based on the retrieved fund-document context, "
                f"{first_line or 'the available evidence supports the answer'}. "
                "[Source: retrieved context]"
            )
        return "The demo model needs retrieved document context to provide a grounded answer."
