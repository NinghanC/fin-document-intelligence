"""Query-time multimodal reasoning over image and table evidence."""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from agents.doc_parser_agent import DocType, DocumentChunk
from services.vector_store import _create_embeddings
from utils.model_clients import create_chat_model, has_provider_key


class RetrievalContextLike(Protocol):
    content: str
    source: str
    score: float
    metadata: dict[str, Any]


@dataclass
class MultimodalSearchResult:
    content: str
    modality: str
    score: float
    metadata: dict[str, Any]


class MultimodalService:
    """
    Query-time multimodal reasoning helper.

    Text chunks still use the normal vector path. Table and image chunks get an
    additional reasoning pass:
    - tables are parsed into headers/rows and matched against the question
    - images can be re-opened and sent to a vision-capable provider model when
      a real provider key is configured
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self) -> None:
        self.embeddings = _create_embeddings()
        self.llm = create_chat_model()

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        texts = [c.content for c in chunks]
        return await self.embeddings.aembed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        return await self.embeddings.aembed_query(query)

    async def reason_over_contexts(
        self,
        question: str,
        contexts: Sequence[RetrievalContextLike],
    ) -> list[MultimodalSearchResult]:
        results: list[MultimodalSearchResult] = []
        for context in contexts:
            doc_type = str(context.metadata.get("doc_type", "")).lower()
            if doc_type == DocType.TABLE.value:
                table_result = self._reason_over_table(question, context.content, context.metadata, context.score)
                if table_result:
                    results.append(table_result)
            elif doc_type == DocType.IMAGE.value:
                results.append(await self._reason_over_image(question, context))
        return sorted(results, key=lambda result: result.score, reverse=True)

    def _reason_over_table(
        self,
        question: str,
        content: str,
        metadata: dict[str, Any],
        base_score: float,
    ) -> MultimodalSearchResult | None:
        table = self._parse_serialized_table(content)
        if not table["headers"] or not table["rows"]:
            return None

        query_tokens = self._tokens(question)
        headers = table["headers"]
        rows = table["rows"]
        matched_headers = [header for header in headers if self._tokens(header) & query_tokens]
        matched_rows = [
            row for row in rows
            if self._tokens(" ".join(row.values())) & query_tokens
        ][:5]
        numeric_facts = self._numeric_facts(rows, matched_headers or headers)
        if not matched_headers and not matched_rows and not numeric_facts:
            return None

        row_preview = "; ".join(
            ", ".join(f"{key}={value}" for key, value in row.items() if value)
            for row in matched_rows[:3]
        )
        content_parts = [
            "Table reasoning evidence:",
            f"headers={', '.join(headers)}",
        ]
        if matched_headers:
            content_parts.append(f"matched_columns={', '.join(matched_headers)}")
        if row_preview:
            content_parts.append(f"matched_rows={row_preview}")
        if numeric_facts:
            content_parts.append(f"numeric_values={'; '.join(numeric_facts[:8])}")

        structural_score = 0.5
        structural_score += min(len(matched_headers) * 0.1, 0.2)
        structural_score += min(len(matched_rows) * 0.05, 0.2)
        structural_score += 0.1 if numeric_facts else 0.0
        return MultimodalSearchResult(
            content=" ".join(content_parts),
            modality=DocType.TABLE.value,
            score=round(min(max(base_score, structural_score), 1.0), 4),
            metadata={**metadata, "reasoning_mode": "structured_table_reasoning"},
        )

    async def _reason_over_image(
        self,
        question: str,
        context: RetrievalContextLike,
    ) -> MultimodalSearchResult:
        source = str(context.metadata.get("source") or context.source)
        if has_provider_key() and self._is_local_image(source):
            answer = await self._ask_vision_model(question, source)
            return MultimodalSearchResult(
                content=f"Visual reasoning evidence: {answer}",
                modality=DocType.IMAGE.value,
                score=min(max(context.score, 0.75), 1.0),
                metadata={**context.metadata, "reasoning_mode": "provider_vision_qa"},
            )

        overlap = len(self._tokens(question) & self._tokens(context.content))
        return MultimodalSearchResult(
            content=(
                "Visual reasoning evidence from parser description: "
                f"{context.content}"
            ),
            modality=DocType.IMAGE.value,
            score=round(min(max(context.score, 0.45 + overlap * 0.05), 1.0), 4),
            metadata={**context.metadata, "reasoning_mode": "vision_description_reasoning"},
        )

    async def _ask_vision_model(self, question: str, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("ascii")
        messages = [
            SystemMessage(content=(
                "You answer questions about financial document images. "
                "Use only visible content in the image. If the image is insufficient, say so."
            )),
            HumanMessage(content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]),
        ]
        response = await self.llm.ainvoke(messages)
        return str(response.content).strip()

    @classmethod
    def _parse_serialized_table(cls, content: str) -> dict[str, Any]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        headers: list[str] = []
        rows: list[dict[str, str]] = []
        for line in lines:
            if line.lower().startswith("headers:"):
                headers = [item.strip() for item in line.split(":", 1)[1].split("|")]
                continue
            if ":" in line and "|" in line:
                values: dict[str, str] = {}
                for cell in line.split("|"):
                    if ":" not in cell:
                        continue
                    key, value = cell.split(":", 1)
                    values[key.strip()] = value.strip()
                if values:
                    rows.append(values)
        if not headers and rows:
            headers = list(rows[0])
        return {"headers": headers, "rows": rows}

    @staticmethod
    def _numeric_facts(rows: list[dict[str, str]], candidate_headers: list[str]) -> list[str]:
        facts: list[str] = []
        for row in rows[:10]:
            label = next((value for key, value in row.items() if key not in candidate_headers and value), "")
            for header in candidate_headers:
                value = row.get(header, "")
                if re.search(r"\d", value):
                    facts.append(f"{label} {header}={value}".strip())
        return facts

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) >= 3}

    @classmethod
    def _is_local_image(cls, path: str) -> bool:
        return os.path.exists(path) and os.path.splitext(path)[1].lower() in cls.IMAGE_EXTENSIONS
