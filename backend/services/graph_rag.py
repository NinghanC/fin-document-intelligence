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
from typing import Any, TypeGuard

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from services.knowledge_graph import KnowledgeGraphService
from services.metapaths import MetapathRouter, MetapathSpec
from services.vector_store import VectorStoreService, _create_embeddings
from utils.model_clients import create_chat_model

logger = structlog.get_logger("finsight.graphrag")


@dataclass
class GraphRAGContext:
    content: str
    source_type: str  # "vector" | "subgraph" | "path" | "community" | "metapath"
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


ENTITY_LINKING_PROMPT = """\
Extract all possible entity names from the following question, such as people, organizations, technologies, products, and concepts.
Return JSON: {"entities": ["entity_1", "entities2"]}
Return only JSON.
"""

DEFAULT_ALIAS_TABLE = {
    "msft": "Microsoft",
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "aapl": "Apple Inc",
    "apple": "Apple Inc",
}

ENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "of",
    "report",
    "reported",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
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
        alias_table: dict[str, str] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = create_chat_model()
        self.embeddings = _create_embeddings()
        self.alias_table = {**DEFAULT_ALIAS_TABLE, **(alias_table or {})}
        self.metapath_router = MetapathRouter()

    async def retrieve(self, query: str, top_k: int = 10) -> list[GraphRAGContext]:
        """
        Hybrid retrieval entry point
        Run independent retrieval branches in parallel, then cross-rerank.
        Branch failures degrade to the remaining retrieval sources.
        """
        vector_result, entity_result = await asyncio.gather(
            self._vector_search(query, top_k=top_k),
            self._entity_linking(query),
            return_exceptions=True,
        )

        all_results: list[GraphRAGContext] = []
        if self._is_context_list(vector_result):
            all_results.extend(vector_result)

        if isinstance(entity_result, BaseException):
            return (await self._cross_rerank(all_results, query))[:top_k]
        entities = entity_result

        branch_results = await asyncio.gather(
            self._subgraph_search(entities, query=query),
            self._path_search(entities),
            self._community_retrieve(entities),
            self._metapath_search(query, entities),
            return_exceptions=True,
        )
        for branch_result in branch_results:
            if self._is_context_list(branch_result):
                all_results.extend(branch_result)

        reranked = await self._cross_rerank(all_results, query)
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
            mentions = [
                str(entity)
                for entity in data.get("entities", [])
                if entity and self._normalize_entity_name(str(entity)) not in ENTITY_STOPWORDS
            ]
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
            alias_target = self.alias_table[normalized]
            canonical = await self._canonicalize_alias_target(alias_target)
            return canonical or alias_target

        exact = await self.knowledge_graph.get_entity(mention)
        if exact:
            return self._entity_name(exact) or mention

        find_alias = getattr(self.knowledge_graph, "find_entity_alias", None)
        if callable(find_alias):
            alias = await find_alias(mention)
            if alias:
                return self._entity_name(alias)

        find_normalized = getattr(self.knowledge_graph, "find_entity_normalized", None)
        if callable(find_normalized):
            normalized_match = await find_normalized(mention)
            if normalized_match:
                return self._entity_name(normalized_match)

        candidates = await self.knowledge_graph.search_entities(mention, limit=5)
        if candidates:
            return candidates[0].get("name")

        find_similar = getattr(self.knowledge_graph, "find_entities_by_name_similarity", None)
        fuzzy_candidates = await find_similar(mention, threshold=0.82, limit=3) if callable(find_similar) else []
        if fuzzy_candidates:
            return fuzzy_candidates[0].get("name")

        embedding_candidates = await self._embedding_entity_match(mention, threshold=0.8)
        if embedding_candidates:
            return embedding_candidates[0]
        return None

    async def _canonicalize_alias_target(self, alias_target: str) -> str | None:
        exact = await self.knowledge_graph.get_entity(alias_target)
        if exact:
            return self._entity_name(exact) or alias_target
        find_normalized = getattr(self.knowledge_graph, "find_entity_normalized", None)
        if callable(find_normalized):
            normalized = await find_normalized(alias_target)
            if normalized:
                return self._entity_name(normalized)
        return None

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
                except Exception as exc:
                    logger.warning(
                        "path_search_failed",
                        source=entities[i],
                        target=entities[j],
                        error=str(exc),
                    )
                    continue
        return contexts

    # Step 5: Community summary
    async def _community_retrieve(self, entities: list[str]) -> list[GraphRAGContext]:
        """Retrieve precomputed community summaries instead of generating them per query."""
        if not entities:
            return []
        summaries = await self.knowledge_graph.get_community_summaries(entities)
        contexts: list[GraphRAGContext] = []
        for summary in summaries:
            content = str(summary.get("summary", ""))
            if not content or "No direct relationships captured" in content:
                continue
            contexts.append(GraphRAGContext(
                content=content,
                source_type="community",
                score=self._community_score(entities, summary),
                metadata={
                    "community_id": summary.get("community_id", ""),
                    "members": summary.get("members", []),
                    "score_method": "member_overlap_lexical",
                },
            ))
        return contexts

    # Step 6: Finance metapath retrieval
    async def _metapath_search(self, query: str, entities: list[str]) -> list[GraphRAGContext]:
        traverse_metapath = getattr(self.knowledge_graph, "traverse_metapath", None)
        if not callable(traverse_metapath):
            return []
        selected_metapaths = self.metapath_router.select(query, entities)
        if not selected_metapaths:
            return []

        contexts: list[GraphRAGContext] = []
        traversal_results = await asyncio.gather(
            *(
                self.knowledge_graph.traverse_metapath(
                    start_entities=entities,
                    metapath=metapath,
                    limit=10,
                )
                for metapath in selected_metapaths
            ),
            return_exceptions=True,
        )
        for metapath, result in zip(selected_metapaths, traversal_results, strict=False):
            if isinstance(result, BaseException):
                logger.warning("metapath_search_failed", metapath=metapath.name, error=str(result))
                continue
            for path_result in result:
                content = (
                    f"Metapath {path_result.metapath_name}: {path_result.evidence}. "
                    f"This supports {metapath.description}"
                )
                contexts.append(GraphRAGContext(
                    content=content,
                    source_type="metapath",
                    score=self._metapath_score(query, metapath, path_result.evidence, path_result.score),
                    metadata={
                        "metapath": path_result.metapath_name,
                        "path": list(path_result.path),
                        "start_entity": path_result.start_entity,
                        "end_entity": path_result.end_entity,
                        "description": metapath.description,
                        "score_method": "metapath_coverage_lexical",
                    },
                ))
        return contexts

    @staticmethod
    def _is_context_list(value: object) -> TypeGuard[list[GraphRAGContext]]:
        return isinstance(value, list)

    # Step 7: cross-rerank
    async def _cross_rerank(self, contexts: list[GraphRAGContext], query: str) -> list[GraphRAGContext]:
        """
        Cross-rerank with query relevance inside each branch and RRF across branches.

        RRF uses rank positions instead of assuming vector, path, subgraph, and
        community scores are directly comparable.
        """
        await self._score_by_query_embedding(query, contexts)
        for ctx in contexts:
            ctx.metadata["dedup_key"] = self._dedup_key(ctx.content)

        seen: set[str] = set()
        unique: list[GraphRAGContext] = []
        for ctx in contexts:
            key = ctx.metadata["dedup_key"]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        return self._reciprocal_rank_fusion(unique)

    async def _score_by_query_embedding(self, query: str, contexts: list[GraphRAGContext]) -> None:
        if not contexts:
            return
        query_vector = await self.embeddings.aembed_query(query)
        content_vectors = await self.embeddings.aembed_documents([ctx.content for ctx in contexts])
        for ctx, vector in zip(contexts, content_vectors, strict=False):
            base_score = ctx.score
            semantic_score = max(0.0, min(self._cosine_similarity(query_vector, vector), 1.0))
            ctx.score = round(semantic_score, 4)
            ctx.metadata["base_score"] = base_score
            ctx.metadata["score_method"] = "embedding_similarity"

    @staticmethod
    def _reciprocal_rank_fusion(contexts: list[GraphRAGContext], rrf_k: int = 60) -> list[GraphRAGContext]:
        fused: dict[str, tuple[GraphRAGContext, float, list[str]]] = {}
        for source_type in ("vector", "subgraph", "path", "community", "metapath"):
            branch = sorted(
                [ctx for ctx in contexts if ctx.source_type == source_type],
                key=lambda item: item.score,
                reverse=True,
            )
            for rank, ctx in enumerate(branch, start=1):
                key = str(ctx.metadata.get("dedup_key", ctx.content))
                contribution = 1 / (rrf_k + rank)
                if key in fused:
                    existing, score, sources = fused[key]
                    if ctx.score > existing.score:
                        existing = ctx
                    fused[key] = (existing, score + contribution, [*sources, source_type])
                else:
                    fused[key] = (ctx, contribution, [source_type])

        best = max((score for _, score, _ in fused.values()), default=1.0)
        ranked: list[GraphRAGContext] = []
        for ctx, score, sources in fused.values():
            ctx.metadata["rrf_score"] = round(score, 6)
            ctx.metadata["rrf_sources"] = sorted(set(sources))
            ctx.score = round(score / best, 4)
            ranked.append(ctx)

        ranked.sort(
            key=lambda item: (float(item.metadata.get("rrf_score", 0.0)), item.score),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _normalize_entity_name(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", value.lower())).strip()

    @staticmethod
    def _entity_name(record: dict[str, Any]) -> str | None:
        if "name" in record:
            return str(record["name"])
        entity = record.get("e")
        if isinstance(entity, dict):
            return entity.get("name")
        if entity is None:
            return None
        get_value = getattr(entity, "get", None)
        name = get_value("name", None) if callable(get_value) else None
        return str(name) if name else None

    async def _embedding_entity_match(self, mention: str, threshold: float = 0.8, limit: int = 3) -> list[str]:
        names = await self.knowledge_graph.get_all_entity_names()
        if not names:
            return []
        mention_vector = await self.embeddings.aembed_query(mention)
        name_vectors = await self.embeddings.aembed_documents(names)
        scored = [
            (self._cosine_similarity(mention_vector, vector), name)
            for name, vector in zip(names, name_vectors, strict=False)
        ]
        scored = [(score, name) for score, name in scored if score >= threshold]
        scored.sort(reverse=True)
        return [name for _, name in scored[:limit]]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

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

    @classmethod
    def _metapath_score(cls, query: str, metapath: MetapathSpec, evidence: str, path_score: float) -> float:
        lexical_score = cls._lexical_similarity(query, f"{metapath.name} {metapath.description} {evidence}")
        score = (path_score * 0.6) + (lexical_score * 0.4)
        return round(min(score, 1.0), 4)

    @staticmethod
    def _dedup_key(content: str) -> str:
        words = sorted(set(re.findall(r"\w{4,}", content.lower())))
        return hashlib.md5(" ".join(words[:30]).encode(), usedforsecurity=False).hexdigest()
