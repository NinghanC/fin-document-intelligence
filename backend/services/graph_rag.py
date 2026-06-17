"""
GraphRAG hybrid retrieval pipeline - vector retrieval + graph traversal + reranking

This is one of the core technical highlights of the project:
  Traditional RAG only performs vector retrieval and loses structured relationships between entities
  GraphRAG combines the knowledge graph with vector retrieval to enable multi-hop reasoning

Workflow:
  Query -> [vector retrieval branch] -> merge -> cross-rerank -> Top-K
         [graph retrieval branch]

Graph retrieval strategy:
  1. Entity linking: identify entities from the query and locate them in the graph
  2. Subgraph recall: traverse N hops from located entities
  3. Path reasoning: find shortest paths between entities and provide reasoning chains
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService
from utils.model_clients import create_chat_model


@dataclass
class GraphRAGContext:
    content: str
    source_type: str  # "vector" | "subgraph" | "path" | "community"
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


ENTITY_LINKING_PROMPT = """\
Extract all possible entity names from the following question, such as people, organizations, technologies, products, and concepts.
Return JSON: {"entities": ["entity_1", "entities2"]}
Return only JSON.
"""

DEFAULT_RERANK_WEIGHTS = {
    # Neutral defaults. Use bench/run_graphrag_eval.py to tune these on a
    # labeled retrieval set before claiming one source type should dominate.
    "vector": 1.0,
    "subgraph": 1.0,
    "path": 1.0,
    "community": 1.0,
}

DEFAULT_ALIAS_TABLE = {
    "msft": "Microsoft",
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "aapl": "Apple Inc",
    "apple": "Apple Inc",
}


class GraphRAGPipeline:
    """
    GraphRAG hybrid retrieval pipeline

    Combines three retrieval strategies:
      1. Vector semantic retrieval - captures semantically similar content
      2. Graph subgraph retrieval - structured reasoning through entity relationships
      3. Community summary retrieval - summarizes subgraphs to provide a high-level overview
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        knowledge_graph: KnowledgeGraphService,
        rerank_weights: dict[str, float] | None = None,
        alias_table: dict[str, str] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = create_chat_model()
        self.rerank_weights = rerank_weights or DEFAULT_RERANK_WEIGHTS
        self.alias_table = {**DEFAULT_ALIAS_TABLE, **(alias_table or {})}

    async def retrieve(self, query: str, top_k: int = 10) -> list[GraphRAGContext]:
        """
        Hybrid retrieval entry point
        Run vector retrieval and graph retrieval in parallel, then cross-rerank
        """
        vector_task = asyncio.create_task(self._vector_search(query, top_k=top_k))
        entity_task = asyncio.create_task(self._entity_linking(query))

        entities = await entity_task
        vector_results, subgraph_results, path_results, community_results = await asyncio.gather(
            vector_task,
            self._subgraph_search(entities, query=query),
            self._path_search(entities),
            self._community_retrieve(entities),
        )

        all_results = vector_results + subgraph_results + path_results + community_results

        reranked = self._cross_rerank(all_results, query)
        return reranked[:top_k]

    # Step 1: Vector retrieval
    async def _vector_search(self, query: str, top_k: int = 5) -> list[GraphRAGContext]:
        results = await self.vector_store.search(query, top_k=top_k)
        return [
            GraphRAGContext(
                content=doc["content"],
                source_type="vector",
                score=score,
                metadata=doc.get("metadata", {}),
            )
            for doc, score in results
        ]

    # Step 2: Entity linking
    async def _entity_linking(self, query: str) -> list[str]:
        messages = [
            SystemMessage(content=ENTITY_LINKING_PROMPT),
            HumanMessage(content=query),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            mentions = [str(entity) for entity in data.get("entities", []) if entity]
        except (json.JSONDecodeError, IndexError):
            mentions = []

        linked: list[str] = []
        for mention in mentions:
            entity = await self._resolve_entity(mention)
            if entity and entity not in linked:
                linked.append(entity)
        return linked

    async def _resolve_entity(self, mention: str) -> str | None:
        normalized = self._normalize_entity_name(mention)
        if not normalized:
            return None

        if normalized in self.alias_table:
            return self.alias_table[normalized]

        exact = await self.knowledge_graph.get_entity(mention)
        if exact:
            return mention

        candidates = await self.knowledge_graph.search_entities(mention, limit=5)
        if candidates:
            return candidates[0].get("name")

        all_names = await self.knowledge_graph.get_all_entity_names()
        best_name = None
        best_score = 0.0
        for name in all_names:
            score = SequenceMatcher(None, normalized, self._normalize_entity_name(name)).ratio()
            if score > best_score:
                best_name = name
                best_score = score
        return best_name if best_score >= 0.82 else None

    # Step 3: Subgraph retrieval
    async def _subgraph_search(self, entities: list[str], query: str = "", hops: int = 2) -> list[GraphRAGContext]:
        contexts: list[GraphRAGContext] = []
        for entity_name in entities:
            neighbors = await self.knowledge_graph.get_neighbors(entity_name, hops=hops)
            for record in neighbors:
                content = (
                    f"{record.get('source', '')} "
                    f"--[{', '.join(record.get('relations', []))}]--> "
                    f"{record.get('target', '')} "
                    f"({record.get('target_type', '')}): "
                    f"{record.get('target_desc', '')}"
                )
                contexts.append(GraphRAGContext(
                    content=content,
                    source_type="subgraph",
                    score=self._subgraph_score(
                        query=query or " ".join(entities),
                        query_entities=entities,
                        entity=entity_name,
                        record=record,
                        content=content,
                    ),
                    metadata={"entity": entity_name, "hops": hops, "score_method": "lexical_entity_relation"},
                ))
        return contexts

    # Step 4: Path retrieval
    async def _path_search(self, entities: list[str]) -> list[GraphRAGContext]:
        """Find shortest paths between entity pairs and provide reasoning chains"""
        if len(entities) < 2:
            return []

        contexts: list[GraphRAGContext] = []
        for i in range(len(entities)):
            for j in range(i + 1, min(i + 3, len(entities))):
                cypher = """
                MATCH path = shortestPath(
                    (a:Entity {name: $name_a})-[*..5]-(b:Entity {name: $name_b})
                )
                RETURN
                    [n IN nodes(path) | n.name] AS node_names,
                    [r IN relationships(path) | type(r)] AS rel_types
                LIMIT 3
                """
                try:
                    records = await self.knowledge_graph.execute_cypher(
                        cypher, {"name_a": entities[i], "name_b": entities[j]}
                    )
                    for rec in records:
                        nodes = rec.get("node_names", [])
                        rels = rec.get("rel_types", [])
                        path_str = ""
                        for k, node in enumerate(nodes):
                            path_str += node
                            if k < len(rels):
                                path_str += f" --[{rels[k]}]--> "
                        content = f"Reasoning path: {path_str}"
                        contexts.append(GraphRAGContext(
                            content=content,
                            source_type="path",
                            score=self._path_score(entities[i], entities[j], nodes, rels),
                            metadata={"from": entities[i], "to": entities[j], "score_method": "entity_coverage_path_length"},
                        ))
                except Exception:
                    continue
        return contexts

    # Step 5: Community summary
    async def _community_retrieve(self, entities: list[str]) -> list[GraphRAGContext]:
        """Retrieve precomputed community summaries instead of generating them per query."""
        if not entities:
            return []
        summaries = await self.knowledge_graph.get_community_summaries(entities)
        return [
            GraphRAGContext(
                content=summary.get("summary", ""),
                source_type="community",
                score=self._community_score(entities, summary),
                metadata={
                    "community_id": summary.get("community_id", ""),
                    "members": summary.get("members", []),
                    "score_method": "member_overlap_lexical",
                },
            )
            for summary in summaries
            if summary.get("summary")
        ]

    # Step 6: cross-rerank
    def _cross_rerank(self, contexts: list[GraphRAGContext], query: str) -> list[GraphRAGContext]:
        """
        Cross-reranking strategy.

        Default weights are neutral. Non-uniform weights should come from an
        evaluation run, not intuition.
        """
        for ctx in contexts:
            ctx.score *= self.rerank_weights.get(ctx.source_type, 1.0)
            ctx.metadata["dedup_key"] = self._dedup_key(ctx.content)

        seen: set[str] = set()
        unique: list[GraphRAGContext] = []
        for ctx in contexts:
            key = ctx.metadata["dedup_key"]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        unique.sort(key=lambda c: c.score, reverse=True)
        return unique

    @staticmethod
    def _normalize_entity_name(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", value.lower())).strip()

    @classmethod
    def _token_set(cls, text: str) -> set[str]:
        return {token for token in re.findall(r"\w{3,}", cls._normalize_entity_name(text))}

    @classmethod
    def _lexical_similarity(cls, query: str, content: str) -> float:
        query_tokens = cls._token_set(query)
        if not query_tokens:
            return 0.0
        content_tokens = cls._token_set(content)
        return len(query_tokens & content_tokens) / len(query_tokens)

    @classmethod
    def _subgraph_score(
        cls,
        query: str,
        query_entities: list[str],
        entity: str,
        record: dict[str, Any],
        content: str,
    ) -> float:
        entity_score = 1.0 if entity in query_entities else 0.5
        target = str(record.get("target", ""))
        target_score = 0.2 if target and target in query_entities else 0.0
        relation_count = len(record.get("relations", []))
        relation_score = min(relation_count / 3, 1.0)
        lexical_score = cls._lexical_similarity(query, content)
        score = (entity_score * 0.4) + (target_score * 0.2) + (relation_score * 0.2) + (lexical_score * 0.2)
        return round(min(score, 1.0), 4)

    @staticmethod
    def _path_score(start: str, end: str, nodes: list[str], rels: list[str]) -> float:
        if not nodes:
            return 0.0
        coverage = (int(start in nodes) + int(end in nodes)) / 2
        path_length_penalty = 1 / max(len(rels), 1)
        score = (coverage * 0.7) + (path_length_penalty * 0.3)
        return round(min(score, 1.0), 4)

    @classmethod
    def _community_score(cls, entities: list[str], summary: dict[str, Any]) -> float:
        members = {str(member) for member in summary.get("members", [])}
        if not entities:
            return 0.0
        member_overlap = len(set(entities) & members) / len(set(entities))
        lexical_score = cls._lexical_similarity(" ".join(entities), str(summary.get("summary", "")))
        score = (member_overlap * 0.7) + (lexical_score * 0.3)
        return round(min(score, 1.0), 4)

    @staticmethod
    def _dedup_key(content: str) -> str:
        words = sorted(set(re.findall(r"\w{4,}", content.lower())))
        return hashlib.md5(" ".join(words[:30]).encode(), usedforsecurity=False).hexdigest()
