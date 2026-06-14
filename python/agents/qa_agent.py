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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings


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
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)


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


class QAAgent:
    """
    QA Agent

    Workflow:
      query → intent_classify → rewrite → parallel_retrieve → rerank → generate_answer
    """

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph

    # ── public API ───────────────────────────────────────────

    async def answer(self, question: str) -> QAResult:
        """Complete QA flow"""
        intent = await self._classify_intent(question)
        rewritten = await self._rewrite_query(question)

        vector_contexts = await self._vector_retrieve(rewritten)
        graph_contexts = await self._graph_retrieve(question, rewritten)

        all_contexts = self._hybrid_rerank(vector_contexts + graph_contexts)
        top_contexts = all_contexts[:8]

        answer_text, reasoning = await self._generate_answer(question, top_contexts, intent)

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=top_contexts,
            intent=intent,
            confidence=self._calc_confidence(top_contexts),
            reasoning_steps=reasoning,
        )

    # ── intent classification ────────────────────────────────

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

    # ── query rewriting ──────────────────────────────────────

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

    # ── vector retrieval ─────────────────────────────────────

    async def _vector_retrieve(self, rewritten: dict) -> list[RetrievedContext]:
        if not self.vector_store:
            return []

        contexts: list[RetrievedContext] = []
        for query in rewritten.get("queries", []):
            results = await self.vector_store.search(query, top_k=5)
            for doc, score in results:
                contexts.append(RetrievedContext(
                    content=doc.get("content", ""),
                    source=doc.get("source", "vector_store"),
                    score=score,
                    retrieval_type="vector",
                    metadata=doc.get("metadata", {}),
                ))
        return contexts

    # ── graph retrieval ──────────────────────────────────────

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
                        score=0.8,
                        retrieval_type="graph",
                        metadata={"cypher": cypher},
                    ))
            except Exception:
                continue
        return contexts

    # ── hybrid reranking ─────────────────────────────────────

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """
        Hybrid reranking: vector score + weighted graph score
        Graph retrieval results naturally contain structured relationships, so they receive a slightly higher weight
        """
        weight_map = {"vector": 1.0, "graph": 1.2, "hybrid": 1.1}
        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.retrieval_type, 1.0)

        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        unique.sort(key=lambda c: c.score, reverse=True)
        return unique

    # ── answer generation ────────────────────────────────────

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
            system_prompt = ANSWER_PROMPT
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
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        if not contexts:
            return 0.0
        avg_score = sum(c.score for c in contexts) / len(contexts)
        return min(avg_score, 1.0)
