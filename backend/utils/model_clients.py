"""Model client helpers used by the public demo and provider-backed deployments."""

from __future__ import annotations

import asyncio
import base64
import json
import os
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
        profile_name = settings.aws_profile.strip()
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        else:
            os.environ.pop("AWS_PROFILE", None)
            os.environ.pop("AWS_DEFAULT_PROFILE", None)
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
    """Generic offline stub for local wiring tests only.

    It deliberately avoids benchmark-specific financial knowledge, answer
    extraction rules, and synthetic relationship generation. Configure a real
    provider for any answer-quality or extraction-quality validation.
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
            return AIMessage(content="Offline demo model cannot perform real vision reasoning.")
        return AIMessage(content=self._answer(user))

    @staticmethod
    def _content(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            return " ".join(str(item) for item in content)
        return str(content)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.strip(".,:;!?()[]{}<>\\\"'") for token in text.split() if token.strip(".,:;!?()[]{}<>\\\"'")]

    @classmethod
    def _entities(cls, text: str) -> list[str]:
        entities: list[str] = []
        current: list[str] = []
        for token in cls._tokens(text):
            if token[:1].isupper() and any(char.isalpha() for char in token):
                current.append(token)
                continue
            if current:
                entities.append(" ".join(current))
                current = []
        if current:
            entities.append(" ".join(current))
        return list(dict.fromkeys(entities))[:5]

    def _rewrite(self, question: str) -> dict[str, Any]:
        entities = self._entities(question)
        keywords = [token.lower() for token in self._tokens(question) if len(token) >= 4 and token.isalpha()][:8]
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
        if any(word in lowered for word in ("why", "impact", "analyze", "explain")):
            return "analytical"
        if "how much" not in lowered and any(word in lowered for word in ("how", "steps", "process")):
            return "procedural"
        return "factoid"

    def _extract(self, user: str) -> dict[str, Any]:
        text = user.split("\n\n", 1)[-1]
        entities = [
            {
                "name": name,
                "type": "Concept",
                "description": f"Entity mention detected by offline demo parser: {name}",
                "confidence": 0.5,
            }
            for name in self._entities(text)
        ]
        return {"entities": entities, "relations": [], "events": []}

    @staticmethod
    def _summarize(user: str) -> str:
        body = user.split("Subgraph information:", 1)[-1].strip()
        digest = sha256(body.encode("utf-8")).hexdigest()[:8]
        return f"Offline structured summary ({digest}). Configure a provider for natural-language community summaries."

    @staticmethod
    def _answer(user: str) -> str:
        if "Context information:" not in user:
            return "The offline demo model cannot answer without retrieved context. Configure a provider for real QA."
        context = user.split("Context information:", 1)[1].split("User question:", 1)[0].strip()
        lines = [line.strip() for line in context.splitlines() if line.strip() and not line.startswith("[Source")]
        if not lines:
            return "The retrieved context is insufficient to answer this question. [Source: retrieved context]"
        excerpt = lines[0][:240]
        return f"Offline demo excerpt: {excerpt}. [Source: retrieved context]"
