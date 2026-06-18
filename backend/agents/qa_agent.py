"""
QA Agent - hybrid retrieval (Vector + Graph), multi-hop reasoning, and answer generation

Core capabilities:
  1. Intent recognition and query rewriting
  2. Vector retrieval (semantic similarity)
  3. Graph retrieval (Cypher queries / subgraph traversal)
  4. Hybrid ranking and reranking
  5. Answer generation from retrieved results with source citations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from services.graph_rag import GraphRAGPipeline
from services.multimodal import MultimodalService
from utils.model_clients import create_chat_model

logger = structlog.get_logger("finsight.qa")


class QueryIntent(str, Enum):
    FACTOID = "factoid"           # fact-based question
    ANALYTICAL = "analytical"     # analytical question
    COMPARATIVE = "comparative"   # comparative question
    PROCEDURAL = "procedural"     # procedural question
    EXPLORATORY = "exploratory"   # exploratory question


@dataclass
class RetrievedContext:
    content: str
    source: str
    score: float
    retrieval_type: str  # "vector" | "graph" | "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    retrieval_quality: float
    reasoning_steps: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Backward-compatible alias for older clients."""
        return self.retrieval_quality


INTENT_PROMPT = """\
You are a query intent classifier. Based on the user question, return the intent category, and return only the category name:
- factoid: fact-based (who/what/where/when)
- analytical: analytical (why/how to understand)
- comparative: comparative (differences between A and B)
- procedural: procedural (how to do it / steps)
- exploratory: exploratory (what options / overview)
"""

QUERY_REWRITE_PROMPT = """\
You are a query rewriting expert. Rewrite the user question into a form better suited for retrieval.
Requirements:
1. Extract core entities and keywords
2. Generate 1-3 retrieval queries
3. Return JSON: {"queries": ["query_1", "query_2"], "entities": ["entity_1"], "keywords": ["keyword_1"]}
"""

CYPHER_GENERATION_PROMPT = """\
You are a Neo4j Cypher query generation expert. Generate Cypher queries from the user question and extracted entities.

Knowledge graph schema:
- Node labels: Person, Organization, Technology, Product, Concept, Location
- Relationship types: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- Node properties: name, type, description, created_at, version

Generate 1-2 Cypher queries and return JSON: {"queries": ["MATCH ...", "MATCH ..."]}
Return only JSON without any other text.
"""

ANSWER_PROMPT = """\
You are a professional enterprise knowledge QA assistant. Answer the user question using the retrieved context.

Requirements:
1. The answer must be based on the provided context; do not fabricate information
2. If the context is insufficient, tell the user clearly
3. Cite information sources, such as [Source: xxx]
4. If multiple sources are involved, synthesize them before giving a conclusion
5. Keep the response professional, accurate, and concise
"""

FACTOID_PROMPT = ANSWER_PROMPT + "\nAnswer in one or two sentences. Cite the strongest source."

COMPARATIVE_PROMPT = ANSWER_PROMPT + "\nCompare the requested entities or options explicitly. Use a compact table or grouped bullets when helpful."

ANALYTICAL_PROMPT = ANSWER_PROMPT + "\nProvide an evidence-backed analysis. Explain the key drivers, caveats, and source support."

PROCEDURAL_PROMPT = ANSWER_PROMPT + "\nReturn clear ordered steps. Cite the source for each material step when available."

EXPLORATORY_PROMPT = ANSWER_PROMPT + "\nGive a structured overview of options, themes, and open questions supported by the sources."


class QAAgent:
    """
    QA Agent

    Workflow:
      query -> intent_classify -> rewrite -> parallel_retrieve -> rerank -> generate_answer
    """

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.llm = create_chat_model()
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.multimodal = MultimodalService()
        self.graph_rag = (
            GraphRAGPipeline(vector_store, knowledge_graph)
            if vector_store is not None and knowledge_graph is not None
            else None
        )

    # public API
    async def answer(self, question: str) -> QAResult:
        """Complete QA flow"""
        intent = await self._classify_intent(question)
        rewritten = await self._rewrite_query(question)

        top_contexts = await self._retrieve_for_intent(question, rewritten, intent)

        answer_text, reasoning = await self._generate_answer(question, top_contexts, intent)

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=top_contexts,
            intent=intent,
            retrieval_quality=self._calc_retrieval_quality(top_contexts),
            reasoning_steps=reasoning,
        )

    async def _retrieve_for_intent(self, question: str, rewritten: dict, intent: QueryIntent) -> list[RetrievedContext]:
        if intent == QueryIntent.FACTOID:
            contexts = await self._retrieve_contexts(question, rewritten, top_k=6)
            return self._balanced_top_contexts(contexts, limit=3)

        if intent == QueryIntent.COMPARATIVE:
            contexts = await self._retrieve_per_entity(question, rewritten, per_entity_k=3)
            if not contexts:
                contexts = await self._retrieve_contexts(question, rewritten, top_k=12)
            return self._balanced_top_contexts(contexts, limit=10)

        if intent == QueryIntent.PROCEDURAL:
            contexts = await self._retrieve_contexts(question, rewritten, top_k=10)
            policy_contexts = self._prioritize_policy_contexts(contexts)
            return self._balanced_top_contexts(policy_contexts, limit=8)

        if intent == QueryIntent.ANALYTICAL:
            contexts = await self._retrieve_contexts(question, rewritten, top_k=14)
            return self._balanced_top_contexts(contexts, limit=10)

        contexts = await self._retrieve_contexts(question, rewritten, top_k=12)
        return self._balanced_top_contexts(contexts, limit=10)

    async def _retrieve_contexts(self, question: str, rewritten: dict, top_k: int = 20) -> list[RetrievedContext]:
        """Use the GraphRAG service when available, with the original hybrid path as fallback."""
        if self.graph_rag is not None:
            try:
                graph_rag_contexts = await self.graph_rag.retrieve(question, top_k=top_k)
                contexts = [
                    RetrievedContext(
                        content=ctx.content,
                        source=ctx.metadata.get("source", ctx.source_type),
                        score=ctx.score,
                        retrieval_type="vector" if ctx.source_type == "vector" else "graph",
                        metadata={"source_type": ctx.source_type, **ctx.metadata},
                    )
                    for ctx in graph_rag_contexts
                ]
                return self._apply_multimodal_weights(contexts)
            except Exception as exc:
                logger.warning("graphrag_retrieve_failed_using_fallback", error=str(exc))
                pass

        vector_contexts = await self._vector_retrieve(rewritten, top_k=top_k)
        graph_contexts = await self._graph_retrieve(question, rewritten)
        return self._apply_multimodal_weights(self._hybrid_rerank(vector_contexts + graph_contexts))

    async def _retrieve_per_entity(self, question: str, rewritten: dict, per_entity_k: int = 3) -> list[RetrievedContext]:
        entities = [str(entity) for entity in rewritten.get("entities", []) if entity]
        if not entities:
            return []

        contexts: list[RetrievedContext] = []
        for entity in entities:
            entity_rewrite = {
                **rewritten,
                "queries": [f"{entity} {question}"],
                "entities": [entity],
            }
            contexts.extend(await self._retrieve_contexts(question, entity_rewrite, top_k=per_entity_k))
        return self._apply_multimodal_weights(self._hybrid_rerank(contexts))

    @staticmethod
    def _prioritize_policy_contexts(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        policy_terms = ("policy", "procedure", "process", "playbook", "control", "step", "workflow")
        prioritized: list[RetrievedContext] = []
        fallback: list[RetrievedContext] = []
        for context in contexts:
            haystack = " ".join([
                context.content,
                context.source,
                str(context.metadata.get("doc_type", "")),
                str(context.metadata.get("source_type", "")),
            ]).lower()
            if any(term in haystack for term in policy_terms):
                context.score = min(context.score + 0.15, 1.0)
                context.metadata["intent_filter"] = "policy_procedure"
                prioritized.append(context)
            else:
                fallback.append(context)
        return sorted(prioritized, key=lambda ctx: ctx.score, reverse=True) + sorted(
            fallback, key=lambda ctx: ctx.score, reverse=True
        )

    def _apply_multimodal_weights(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Apply modality-aware reranking to vector contexts before final balancing."""
        for ctx in contexts:
            if ctx.retrieval_type == "vector":
                doc_type = str(ctx.metadata.get("doc_type", ""))
                ctx.score *= self.multimodal.MODALITY_WEIGHTS.get(doc_type, 1.0)
        contexts.sort(key=lambda ctx: ctx.score, reverse=True)
        return contexts

    @staticmethod
    def _context_limit_for_intent(intent: QueryIntent) -> int:
        return {
            QueryIntent.FACTOID: 6,
            QueryIntent.ANALYTICAL: 10,
            QueryIntent.COMPARATIVE: 10,
            QueryIntent.PROCEDURAL: 8,
            QueryIntent.EXPLORATORY: 10,
        }.get(intent, 8)

    @staticmethod
    def _balanced_top_contexts(contexts: list[RetrievedContext], limit: int = 8) -> list[RetrievedContext]:
        """Keep the highest-ranking contexts while preserving hybrid source diversity."""
        selected: list[RetrievedContext] = []
        for retrieval_type in ("vector", "graph"):
            first = next((ctx for ctx in contexts if ctx.retrieval_type == retrieval_type), None)
            if first is not None and first not in selected:
                selected.append(first)

        for ctx in contexts:
            if len(selected) >= limit:
                break
            if ctx not in selected:
                selected.append(ctx)

        selected.sort(key=lambda ctx: ctx.score, reverse=True)
        return selected[:limit]

    # intent classification
    async def _classify_intent(self, question: str) -> QueryIntent:
        messages = [
            SystemMessage(content=INTENT_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        raw = resp.content.strip().lower()
        for intent in QueryIntent:
            if intent.value in raw:
                return intent
        return QueryIntent.FACTOID

    # query rewriting
    async def _rewrite_query(self, question: str) -> dict:
        import json
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return {"queries": [question], "entities": [], "keywords": []}

    # vector retrieval
    async def _vector_retrieve(self, rewritten: dict, top_k: int = 5) -> list[RetrievedContext]:
        if not self.vector_store:
            return []

        contexts: list[RetrievedContext] = []
        for query in rewritten.get("queries", []):
            results = await self.vector_store.search(query, top_k=top_k)
            for doc, score in results:
                contexts.append(RetrievedContext(
                    content=doc.get("content", ""),
                    source=doc.get("source", "vector_store"),
                    score=score,
                    retrieval_type="vector",
                    metadata=doc.get("metadata", {}),
                ))
        return contexts

    # graph retrieval
    async def _graph_retrieve(self, question: str, rewritten: dict) -> list[RetrievedContext]:
        if not self.knowledge_graph:
            return []

        import json
        entities = rewritten.get("entities", [])
        messages = [
            SystemMessage(content=CYPHER_GENERATION_PROMPT),
            HumanMessage(content=f"Question: {question}\nentities: {entities}"),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            cypher_data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            cypher_data = {"queries": []}

        contexts: list[RetrievedContext] = []
        for cypher in cypher_data.get("queries", []):
            try:
                records = await self.knowledge_graph.execute_cypher(cypher)
                for record in records:
                    contexts.append(RetrievedContext(
                        content=str(record),
                        source="knowledge_graph",
                        score=self._graph_record_score(question, record),
                        retrieval_type="graph",
                        metadata={"cypher": cypher, "score_method": "lexical_record_overlap"},
                    ))
            except Exception as exc:
                logger.warning("graph_cypher_query_failed", cypher=cypher, error=str(exc))
                continue
        return contexts

    # hybrid reranking
    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """
        Hybrid reranking with reciprocal rank fusion.

        Vector similarity and graph relevance are not guaranteed to share a
        calibrated score scale, so fusion uses rank positions per branch.
        """
        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        fused: dict[str, tuple[RetrievedContext, float, list[str]]] = {}
        for retrieval_type in ("vector", "graph"):
            branch = sorted(
                [ctx for ctx in unique if ctx.retrieval_type == retrieval_type],
                key=lambda item: item.score,
                reverse=True,
            )
            for rank, ctx in enumerate(branch, start=1):
                key = ctx.content[:100]
                contribution = 1 / (60 + rank)
                if key in fused:
                    existing, score, sources = fused[key]
                    if ctx.score > existing.score:
                        existing = ctx
                    fused[key] = (existing, score + contribution, [*sources, retrieval_type])
                else:
                    fused[key] = (ctx, contribution, [retrieval_type])

        best = max((score for _, score, _ in fused.values()), default=1.0)
        reranked: list[RetrievedContext] = []
        for ctx, score, sources in fused.values():
            ctx.metadata["rrf_score"] = round(score, 6)
            ctx.metadata["rrf_sources"] = sorted(set(sources))
            ctx.score = round(score / best, 4)
            reranked.append(ctx)

        reranked.sort(key=lambda ctx: float(ctx.metadata.get("rrf_score", 0.0)), reverse=True)
        return reranked

    @classmethod
    def _graph_record_score(cls, question: str, record: dict[str, Any]) -> float:
        record_text = " ".join(str(value) for value in record.values())
        query_tokens = cls._token_set(question)
        if not query_tokens:
            return 0.0
        record_tokens = cls._token_set(record_text)
        lexical_overlap = len(query_tokens & record_tokens) / len(query_tokens)
        structural_bonus = 0.15 if any(key in record for key in ("relations", "rel_types", "node_names")) else 0.0
        return round(min(lexical_overlap + structural_bonus, 1.0), 4)

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"\w{3,}", text.lower()))

    # answer generation
    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        context_text = "\n\n".join(
            f"[Source {i+1}: {c.source} | Type: {c.retrieval_type} | Score: {c.score:.2f}]\n{c.content}"
            for i, c in enumerate(contexts)
        )
        reasoning_steps = [
            f"Identified question intent: {intent.value}",
            f"Retrieved {len(contexts)} relevant contexts",
            f"Vector retrieval: {sum(1 for c in contexts if c.retrieval_type == 'vector')}",
            f"Graph retrieval: {sum(1 for c in contexts if c.retrieval_type == 'graph')}",
        ]

        if contexts:
            system_prompt = self._prompt_for_intent(intent)
            user_prompt = f"Context information:\n{context_text}\n\nUser question: {question}"
        else:
            system_prompt = "You are a professional enterprise knowledge QA assistant. The current knowledge base is empty, so answer the user question directly from your own knowledge. Keep the response professional, accurate, and concise."
            user_prompt = question
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        resp = await self.llm.ainvoke(messages)
        reasoning_steps.append("Answer generation complete")
        return resp.content, reasoning_steps

    @staticmethod
    def _prompt_for_intent(intent: QueryIntent) -> str:
        return {
            QueryIntent.FACTOID: FACTOID_PROMPT,
            QueryIntent.COMPARATIVE: COMPARATIVE_PROMPT,
            QueryIntent.ANALYTICAL: ANALYTICAL_PROMPT,
            QueryIntent.PROCEDURAL: PROCEDURAL_PROMPT,
            QueryIntent.EXPLORATORY: EXPLORATORY_PROMPT,
        }.get(intent, ANSWER_PROMPT)

    @staticmethod
    def _calc_retrieval_quality(contexts: list[RetrievedContext]) -> float:
        """Interpretable retrieval-quality signal, not a probability."""
        if not contexts:
            return 0.0
        best_score = max(min(max(c.score, 0.0), 1.0) for c in contexts)
        unique_sources = len({c.source for c in contexts if c.source})
        source_diversity = min(unique_sources / 3, 1.0)
        has_graph_support = any(c.retrieval_type == "graph" for c in contexts)
        retrieval_quality = (best_score * 0.5) + (source_diversity * 0.3) + (0.2 if has_graph_support else 0.0)
        return round(min(retrieval_quality, 1.0), 2)

    _calc_confidence = _calc_retrieval_quality
