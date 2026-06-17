"""
Knowledge Extraction Agent - extracts entities, relations, and events from document chunks to build knowledge graph triples

Core capabilities:
  1. Named entity recognition (NER)
  2. Relation extraction (RE)
  3. Event extraction
  4. Triple generation -> write to Neo4j
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.doc_parser_agent import DocumentChunk
from utils.model_clients import create_chat_model

EXTRACTION_SYSTEM_PROMPT = """\
You are a professional knowledge extraction engine. Given a text passage, extract:
1. **entities**:people, organizations, locations, products, technologies, concepts, and similar items
2. **relations**:relationships between entities, represented as triples (head entity, relation, tail entity)
3. **events**:events mentioned in the text, including triggers and participants

Return strictly in the following JSON format:
{
  "entities": [
    {"name": "entity_name", "type": "entity_type", "description": "brief_description"}
  ],
  "relations": [
    {"head": "head_entity", "relation": "Relationship types", "tail": "tail_entity", "confidence": 0.95}
  ],
  "events": [
    {"trigger": "trigger", "type": "event_type", "participants": ["participant_1"]}
  ]
}

Notes:
- Entity types include: Person, Organization, Location, Product, Technology, Concept, Event, Time
- Relationship types include: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- confidence is a floating-point number between 0 and 1
- Return only JSON without any other text
"""


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def node_label(self) -> str:
        return self.type.replace(" ", "_")


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEvent:
    trigger: str
    type: str
    participants: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    events: list[KnowledgeEvent]
    source_chunk_id: str = ""


class KnowledgeExtractAgent:
    """
    Knowledge Extraction Agent

    Workflow:
      receive_chunks -> extract_per_chunk -> deduplicate -> resolve_entities -> output_triples
    """

    BATCH_SIZE = 5

    def __init__(self) -> None:
        self.llm = create_chat_model()

    # public API
    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        """Extract knowledge from chunks using bounded concurrent batches."""
        results: list[ExtractionResult] = []
        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            batch_results = await asyncio.gather(*(self._extract_from_chunk(chunk) for chunk in batch))
            results.extend(batch_results)
        merged = self._deduplicate(results)
        return merged

    async def extract_single(self, text: str, chunk_id: str = "") -> ExtractionResult:
        """Extract knowledge from a single text passage"""
        return await self._extract_from_text(text, chunk_id)

    # core extraction
    async def _extract_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        return await self._extract_from_text(chunk.content, chunk.chunk_id)

    async def _extract_from_text(self, text: str, source_id: str) -> ExtractionResult:
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"Extract knowledge from the following text:\n\n{text}"),
        ]
        resp = await self.llm.ainvoke(messages)
        return self._parse_response(resp.content, source_id)

    def _parse_response(self, raw: str, source_id: str) -> ExtractionResult:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)

        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "Concept"),
                description=e.get("description", ""),
                confidence=float(e.get("confidence", 1.0)),
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]
        relations = [
            Relation(
                head=r.get("head", ""),
                relation=r.get("relation", "related_to"),
                tail=r.get("tail", ""),
                confidence=float(r.get("confidence", 0.5)),
            )
            for r in data.get("relations", [])
            if r.get("head") and r.get("tail")
        ]
        events = [
            KnowledgeEvent(
                trigger=ev.get("trigger", ""),
                type=ev.get("type", ""),
                participants=ev.get("participants", []),
            )
            for ev in data.get("events", [])
        ]
        return self._filter_extraction_result(ExtractionResult(
            entities=entities,
            relations=relations,
            events=events,
            source_chunk_id=source_id,
        ))

    @staticmethod
    def _filter_extraction_result(result: ExtractionResult) -> ExtractionResult:
        """Apply lightweight quality gates before graph storage."""
        entities = [entity for entity in result.entities if entity.confidence >= 0.7]
        name_counts = Counter(entity.name for entity in entities)
        entities = [
            entity for entity in entities
            if name_counts[entity.name] >= 2 or entity.confidence >= 0.9
        ]
        valid_names = {entity.name for entity in entities}
        relations = [
            relation for relation in result.relations
            if relation.confidence >= 0.7 and relation.head in valid_names and relation.tail in valid_names
        ]
        return ExtractionResult(
            entities=entities,
            relations=relations,
            events=result.events,
            source_chunk_id=result.source_chunk_id,
        )

    # deduplication & entity resolution
    @staticmethod
    def _deduplicate(results: list[ExtractionResult]) -> list[ExtractionResult]:
        """
        Cross-chunk deduplication: merge entities with the same name and type, and deduplicate relations
        """
        seen_entities: dict[str, Entity] = {}
        seen_relations: set[tuple[str, str, str]] = set()
        deduped: list[ExtractionResult] = []

        for result in results:
            unique_entities: list[Entity] = []
            for ent in result.entities:
                entity_key = f"{ent.name}::{ent.type}"
                if entity_key not in seen_entities:
                    seen_entities[entity_key] = ent
                    unique_entities.append(ent)
                else:
                    KnowledgeExtractAgent._merge_entity(seen_entities[entity_key], ent)

            unique_relations: list[Relation] = []
            for rel in result.relations:
                relation_key = (rel.head, rel.relation, rel.tail)
                if relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    unique_relations.append(rel)

            deduped.append(ExtractionResult(
                entities=unique_entities,
                relations=unique_relations,
                events=result.events,
                source_chunk_id=result.source_chunk_id,
            ))
        return deduped

    @staticmethod
    def _merge_entity(existing: Entity, candidate: Entity) -> Entity:
        """Keep the richer/higher-confidence entity mention."""
        existing_score = (len(existing.description), existing.confidence)
        candidate_score = (len(candidate.description), candidate.confidence)
        if candidate_score > existing_score:
            existing.description = candidate.description
            existing.confidence = candidate.confidence
            existing.properties = {**existing.properties, **candidate.properties}
            return existing
        existing.properties = {**candidate.properties, **existing.properties}
        return existing
