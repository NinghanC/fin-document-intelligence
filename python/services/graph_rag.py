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

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService


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

COMMUNITY_SUMMARY_PROMPT = """\
You are a knowledge graph analysis expert. Generate a structured summary from the following subgraph information.
Requirements:
1. Summarize the core entities and relationships in the subgraph
2. Highlight key connections between entities
3. Identify any valuable reasoning chains
"""


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
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    async def retrieve(self, query: str, top_k: int = 10) -> list[GraphRAGContext]:
        """
        Hybrid retrieval entry point
        Run vector retrieval and graph retrieval in parallel, then cross-rerank
        """
        vector_results = await self._vector_search(query, top_k=top_k)
        entities = await self._entity_linking(query)
        subgraph_results = await self._subgraph_search(entities)
        path_results = await self._path_search(entities)

        all_results = vector_results + subgraph_results + path_results

        if subgraph_results:
            community_ctx = await self._community_summary(subgraph_results)
            all_results.append(community_ctx)

        reranked = self._cross_rerank(all_results, query)
        return reranked[:top_k]

    # ── Step 1: Vector retrieval ─────────────────────────────────────

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

    # ── Step 2: Entity linking ─────────────────────────────────────

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
            return data.get("entities", [])
        except (json.JSONDecodeError, IndexError):
            return []

    # ── Step 3: Subgraph retrieval ─────────────────────────────────────

    async def _subgraph_search(self, entities: list[str], hops: int = 2) -> list[GraphRAGContext]:
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
                    score=0.75,
                    metadata={"entity": entity_name, "hops": hops},
                ))
        return contexts

    # ── Step 4: Path retrieval ─────────────────────────────────────

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
                        contexts.append(GraphRAGContext(
                            content=f"Reasoning path: {path_str}",
                            source_type="path",
                            score=0.85,
                            metadata={"from": entities[i], "to": entities[j]},
                        ))
                except Exception:
                    continue
        return contexts

    # ── Step 5: Community summary ─────────────────────────────────────

    async def _community_summary(self, subgraph_results: list[GraphRAGContext]) -> GraphRAGContext:
        """Summarize retrieved subgraph information"""
        subgraph_text = "\n".join(r.content for r in subgraph_results[:20])
        messages = [
            SystemMessage(content=COMMUNITY_SUMMARY_PROMPT),
            HumanMessage(content=f"Subgraph information:\n{subgraph_text}"),
        ]
        resp = await self.llm.ainvoke(messages)
        return GraphRAGContext(
            content=resp.content,
            source_type="community",
            score=0.9,
            metadata={"type": "community_summary"},
        )

    # ── Step 6: cross-rerank ───────────────────────────────────

    @staticmethod
    def _cross_rerank(contexts: list[GraphRAGContext], query: str) -> list[GraphRAGContext]:
        """
        Cross-reranking strategy:
          - Vector retrieval: base score × 1.0
          - Subgraph retrieval: base score × 1.15 (structured information is more precise)
          - Path retrieval: base score × 1.25 (reasoning chains are most valuable)
          - Community summary: base score × 1.1  (high-level overview)
        """
        weight_map = {"vector": 1.0, "subgraph": 1.15, "path": 1.25, "community": 1.1}
        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.source_type, 1.0)

        seen: set[str] = set()
        unique: list[GraphRAGContext] = []
        for ctx in contexts:
            key = ctx.content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        unique.sort(key=lambda c: c.score, reverse=True)
        return unique
