"""
Multimodal Service - unified embedding and retrieval for different data modalities

Responsibilities:
  1. Text embedding
  2. Image embedding (describe with LLM vision, then embed)
  3. Table embedding (structured data -> natural language -> embedding)
  4. Weighted score fusion for cross-modal retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.doc_parser_agent import DocType, DocumentChunk
from services.vector_store import _create_embeddings


@dataclass
class MultimodalSearchResult:
    content: str
    modality: str
    score: float
    metadata: dict[str, Any]


class MultimodalService:
    """
    Multimodal processing service

    Strategy: convert each modality to text first, then embed uniformly
    Apply different weights to modalities during retrieval based on query match quality
    """

    MODALITY_WEIGHTS: dict[str, float] = {
        DocType.TEXT.value: 1.0,
        DocType.MARKDOWN.value: 1.0,
        DocType.PDF.value: 0.95,
        DocType.TABLE.value: 0.9,
        DocType.IMAGE.value: 0.85,
    }

    def __init__(self) -> None:
        self.embeddings = _create_embeddings()

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Embed document chunks in a batch"""
        texts = [c.content for c in chunks]
        return await self.embeddings.aembed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        """Embed query text"""
        return await self.embeddings.aembed_query(query)

    def weighted_rerank(
        self,
        results: list[tuple[DocumentChunk, float]],
    ) -> list[MultimodalSearchResult]:
        """
        Cross-modal weighted reranking
        Apply different weights to retrieval results from each modality, then sort them together
        """
        reranked: list[MultimodalSearchResult] = []
        for chunk, score in results:
            weight = self.MODALITY_WEIGHTS.get(chunk.doc_type.value, 1.0)
            reranked.append(MultimodalSearchResult(
                content=chunk.content,
                modality=chunk.doc_type.value,
                score=score * weight,
                metadata=chunk.metadata,
            ))
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
