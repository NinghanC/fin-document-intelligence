"""Model client helpers used by the public demo and provider-backed deployments."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import structlog
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import settings

logger = structlog.get_logger("finsight.model_clients")

PLACEHOLDER_KEYS = {
    "",
    "your-key",
    "your-provider-api-key",
    "your-azure-openai-key",
    "your-aws-or-gateway-key",
    "your-databricks-token",
}


def has_provider_key() -> bool:
    provider = settings.model_provider.strip().lower()
    if provider == "bedrock":
        return bool(settings.bedrock_model_id.strip())
    return settings.openai_api_key.strip() not in PLACEHOLDER_KEYS


def create_chat_model() -> Any:
    """Return a resilient provider-backed chat model, or the offline demo model."""
    fallback = DemoChatModel()
    provider = settings.model_provider.strip().lower()
    if provider == "bedrock" and has_provider_key():
        return ResilientChatModel(BedrockChatModel(), fallback=fallback)
    if has_provider_key():
        provider_model = ChatOpenAI(
            model=settings.openai_model,
            api_key=SecretStr(settings.openai_api_key),
            base_url=settings.openai_base_url,
            temperature=0,
            timeout=settings.model_call_timeout_seconds,
        )
        return ResilientChatModel(provider_model, fallback=fallback)
    return fallback




@dataclass
class ResilientChatModel:
    """Adds timeout, retry, and optional demo fallback around chat models."""

    primary: Any
    fallback: Any | None = None

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        attempts = max(int(settings.model_call_max_retries), 0) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.wait_for(
                    self.primary.ainvoke(messages),
                    timeout=settings.model_call_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "model_call_failed",
                    attempt=attempt,
                    attempts=attempts,
                    provider=settings.model_provider,
                    error=str(exc),
                )
                if attempt < attempts:
                    await asyncio.sleep(min(0.25 * attempt, 1.0))

        if settings.model_call_fallback_to_demo and self.fallback is not None:
            logger.warning("model_call_falling_back_to_demo", provider=settings.model_provider)
            return await self.fallback.ainvoke(messages)
        assert last_error is not None
        raise last_error

@dataclass
class BedrockChatModel:
    """Async wrapper around Amazon Bedrock Runtime Converse API."""

    model_id: str | None = None

    def __post_init__(self) -> None:
        self.model_id = self.model_id or settings.bedrock_model_id
        if not self.model_id:
            raise RuntimeError("BEDROCK_MODEL_ID must be configured for MODEL_PROVIDER=bedrock")
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is missing
            raise RuntimeError("boto3 is required for MODEL_PROVIDER=bedrock") from exc

        session_kwargs: dict[str, str] = {}
        if settings.aws_profile.strip():
            session_kwargs["profile_name"] = settings.aws_profile.strip()
        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime", region_name=settings.aws_region)

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        response = await asyncio.to_thread(self._converse, messages)
        return AIMessage(content=self._extract_text(response))

    def _converse(self, messages: list[Any]) -> dict[str, Any]:
        system_blocks, conversation = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": conversation,
            "inferenceConfig": {
                "temperature": 0,
                "maxTokens": settings.bedrock_max_tokens,
            },
        }
        if system_blocks:
            payload["system"] = system_blocks
        return self.client.converse(**payload)

    @classmethod
    def _convert_messages(cls, messages: list[Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        system_blocks: list[dict[str, str]] = []
        conversation: list[dict[str, Any]] = []
        for message in messages:
            message_type = getattr(message, "type", "human")
            content = getattr(message, "content", message)
            if message_type == "system":
                system_blocks.extend(cls._text_blocks(content))
                continue
            role = "assistant" if message_type in {"ai", "assistant"} else "user"
            blocks = cls._content_blocks(content)
            if blocks:
                conversation.append({"role": role, "content": blocks})
        if not conversation:
            conversation.append({"role": "user", "content": [{"text": ""}]})
        return system_blocks, conversation

    @classmethod
    def _text_blocks(cls, content: Any) -> list[dict[str, str]]:
        text = cls._string_content(content).strip()
        return [{"text": text}] if text else []

    @classmethod
    def _content_blocks(cls, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    blocks.append({"text": str(item.get("text", ""))})
                    continue
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_block = cls._image_block(item)
                    if image_block:
                        blocks.append(image_block)
                    continue
                blocks.append({"text": cls._string_content(item)})
            return blocks
        return [{"text": cls._string_content(content)}]

    @staticmethod
    def _image_block(item: dict[str, Any]) -> dict[str, Any] | None:
        image_url = item.get("image_url", {})
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str) or not url.startswith("data:image/"):
            return None
        header, encoded = url.split(",", 1)
        image_format = header.split("/", 1)[1].split(";", 1)[0].lower()
        if image_format == "jpg":
            image_format = "jpeg"
        return {
            "image": {
                "format": image_format,
                "source": {"bytes": base64.b64decode(encoded)},
            }
        }

    @staticmethod
    def _string_content(content: Any) -> str:
        if isinstance(content, list):
            return " ".join(BedrockChatModel._string_content(item) for item in content)
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        parts = [str(block.get("text", "")) for block in blocks if isinstance(block, dict) and "text" in block]
        return "\n".join(part for part in parts if part).strip()

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




