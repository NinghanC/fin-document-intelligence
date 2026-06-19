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
    """Return a provider-backed chat model, or the offline demo model.

    The demo model exists only to keep local demos and deterministic CI usable
    without provider credentials. It is not a substitute for live LLM
    validation; see the optional live_llm tests.
    """
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
    """Small deterministic model for offline demos and automated tests.

    This class intentionally uses simple rules and fixture-friendly formatting.
    Provider-backed behavior must be validated with live_llm smoke tests.
    """

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
        if "how much" not in lowered and any(word in lowered for word in ("how", "steps", "process")):
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
            question = user.split("User question:", 1)[-1]

            evidence = DemoChatModel._best_evidence_line(context, question)
            if not DemoChatModel._evidence_supports_question(evidence, question):
                return (
                    "The retrieved context is truncated or insufficient to answer this question reliably. "
                    "[Source: retrieved context]"
                )

            return (
                "Based on the retrieved context, "
                f"{evidence or 'the available evidence supports the answer'}. "
                "[Source: retrieved context]"
            )
        return "The demo model needs retrieved document context to provide a grounded answer."

    @staticmethod
    def _best_evidence_line(context: str, question: str) -> str:
        query_tokens = {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", question.lower())
            if len(token) >= 3 and token not in {"and", "did", "for", "the", "their", "what", "which"}
        }
        lines = [
            line.strip()
            for line in re.split(r"[\n;]+", context)
            if line.strip() and not line.startswith("[Source")
        ]
        if not lines:
            return ""

        windows = []
        for i, line in enumerate(lines):
            windows.append(line)
            if i + 1 < len(lines):
                windows.append(f"{line} {lines[i + 1]}")
            if i + 2 < len(lines):
                windows.append(f"{line} {lines[i + 1]} {lines[i + 2]}")

        def score(text: str) -> tuple[float, int]:
            tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
            overlap = len(query_tokens & tokens)
            numeric_bonus = 1 if re.search(r"\d", text) else 0
            risk_penalty = 1 if "risk factor" in text.lower() and "risk" not in query_tokens else 0
            phrase_bonus = 0.5 if any(phrase in text.lower() for phrase in ("driven by", "included in", "compared with")) else 0.0
            return (overlap + numeric_bonus * 0.5 + phrase_bonus - risk_penalty, len(text))

        best = max(windows, key=score)
        return DemoChatModel._focused_excerpt(best, query_tokens)

    @staticmethod
    def _focused_excerpt(text: str, query_tokens: set[str], max_chars: int = 240) -> str:
        if len(text) <= max_chars:
            return text
        lowered = text.lower()
        broad_tokens = {"2021", "2022", "2023", "annual", "fiscal", "report", "reported", "year"}
        focused_tokens = query_tokens - broad_tokens
        generic_metric_tokens = {"major", "ratio", "source", "sources"}
        preferred_tokens = focused_tokens - generic_metric_tokens
        positions = [lowered.find(token) for token in preferred_tokens if lowered.find(token) >= 0]
        if not positions:
            positions = [lowered.find(token) for token in focused_tokens if lowered.find(token) >= 0]
        if not positions:
            positions = [lowered.find(token) for token in query_tokens if lowered.find(token) >= 0]
        if not positions:
            return text[:max_chars]
        start = max(min(positions) - 60, 0)
        end = min(start + max_chars, len(text))
        return text[start:end].strip()

    @staticmethod
    def _evidence_supports_question(evidence: str, question: str) -> bool:
        if not evidence:
            return False
        question_lower = question.lower()
        evidence_lower = evidence.lower()
        metric_terms = {
            token
            for token in re.findall(r"[a-zA-Z]+", question_lower)
            if token
            in {
                "assets",
                "coverage",
                "income",
                "liquidity",
                "margin",
                "ratio",
                "revenue",
                "sales",
                "yield",
            }
        }
        if metric_terms:
            has_specific_metric_term = bool((metric_terms - {"ratio"}) & set(re.findall(r"[a-zA-Z]+", evidence_lower)))
            return has_specific_metric_term and bool(re.search(r"\d", evidence))
        return True




