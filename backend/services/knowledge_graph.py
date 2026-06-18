"""
Knowledge Graph Service - Neo4j graph database operations

Responsibilities:
  1. Entity (Node) CRUD with version numbers and timestamps
  2. Relationship CRUD
  3. Cypher query execution
  4. Subgraph retrieval (multi-hop traversal)
  5. Delete by source (supports incremental updates)
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

import structlog

from agents.knowledge_extract_agent import Entity, Relation
from config import settings
from services.community_summarizer import (
    CommunitySummarizer,
    StructuredCommunitySummarizer,
    create_community_summarizer,
)
from services.ingestion_registry import ingestion_registry
from services.metapaths import MetapathResult, MetapathSpec

WRITE_CYPHER_PATTERN = re.compile(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|LOAD|CALL\s+dbms)\b", re.IGNORECASE)

ALLOWED_RELATION_TYPES = {
    "BELONGS_TO",
    "DEPENDS_ON",
    "DEVELOPED_BY",
    "HOLDS",
    "LOCATED_IN",
    "OWNS",
    "PART_OF",
    "REGULATED_BY",
    "RELATED_TO",
    "SUBJECT_TO",
    "USES",
    "WORKS_AT",
}

logger = structlog.get_logger("finsight.knowledge_graph")


class KnowledgeGraphService:
    """Neo4j knowledge graph service"""

    def __init__(self, community_summarizer: CommunitySummarizer | None = None) -> None:
        self._driver: Any = None
        self._entities: dict[str, dict[str, Any]] = {}
        self._aliases: dict[str, str] = {}
        self._relations: list[dict[str, Any]] = []
        self._community_summaries: dict[str, dict[str, Any]] = {}
        self._community_summarizer = community_summarizer or create_community_summarizer()

    # lifecycle
    async def init(self) -> None:
        from neo4j import AsyncGraphDatabase
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        try:
            await self._ensure_indexes()
        except Exception as exc:
            logger.warning("neo4j_index_setup_failed_using_memory_fallback", error=str(exc))
            await self._driver.close()
            self._driver = None
            raise

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _ensure_indexes(self) -> None:
        """Create common indexes to speed up queries"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source)",
        ]
        async with self._driver.session() as session:
            for q in index_queries:
                await session.run(q)

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    # entity operations
    async def upsert_entity(self, entity: Entity, version: int = 1, source: str = "") -> None:
        self._entities[entity.name] = {
            "name": entity.name,
            "type": entity.type,
            "description": entity.description,
            "confidence": entity.confidence,
            "normalized_name": self.normalize_entity_name(entity.name),
            "aliases": self._entity_aliases(entity),
            "properties": entity.properties,
            "version": version,
            "source": source,
            "updated_at": int(time.time()),
        }
        for alias in self._entity_aliases(entity):
            self._aliases[self.normalize_entity_name(alias)] = entity.name
        if not self._driver:
            return
        """
        Create or update entity nodes using MERGE semantics
        Includes version numbers and timestamps for incremental update tracking
        """
        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.description = $description,
            e.confidence = $confidence,
            e.normalized_name = $normalized_name,
            e.aliases = $aliases,
            e.version = $version,
            e.source = $source,
            e.created_at = $now,
            e.updated_at = $now
        ON MATCH SET
            e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
            e.version = $version,
            e.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "name": entity.name,
                "type": entity.type,
            "description": entity.description,
            "confidence": entity.confidence,
            "normalized_name": self.normalize_entity_name(entity.name),
            "aliases": self._entity_aliases(entity),
            "version": version,
            "source": source,
            "now": int(time.time()),
            })

    async def add_relation(self, relation: Relation, source: str = "") -> None:
        """Create relationships between entities"""
        rel_type = self._sanitize_rel_type(relation.relation)
        rel_record = {
            "head": relation.head,
            "relation": rel_type,
            "tail": relation.tail,
            "confidence": relation.confidence,
            "source": source,
            "updated_at": int(time.time()),
        }
        if rel_record not in self._relations:
            self._relations.append(rel_record)
        if not self._driver:
            return
        # Cypher cannot parameterize relationship type tokens. rel_type is
        # therefore constrained to ALLOWED_RELATION_TYPES before interpolation.
        cypher = f"""
        MATCH (h:Entity {{name: $head}})
        MATCH (t:Entity {{name: $tail}})
        MERGE (h)-[r:{rel_type}]->(t)
        SET r.confidence = $confidence, r.source = $source, r.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "head": relation.head,
                "tail": relation.tail,
                "confidence": relation.confidence,
                "source": source,
                "now": int(time.time()),
            })

    # query operations
    async def execute_cypher(self, cypher: str, params: dict | None = None, read_only: bool = True) -> list[dict]:
        """Execute arbitrary Cypher queries"""
        if read_only and WRITE_CYPHER_PATTERN.search(cypher):
            raise ValueError("Only read-only Cypher is allowed")
        if not self._driver:
            return []

        from neo4j import READ_ACCESS, WRITE_ACCESS
        access_mode = READ_ACCESS if read_only else WRITE_ACCESS
        async with self._driver.session(default_access_mode=access_mode) as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return records

    async def get_entity(self, name: str) -> dict | None:
        """Query a single entity"""
        if not self._driver:
            entity = self._entities.get(name)
            return entity if entity and self._source_visible(entity.get("source", "")) else None
        cypher = "MATCH (e:Entity {name: $name}) RETURN e"
        records = await self.execute_cypher(cypher, {"name": name})
        return records[0] if records else None

    async def find_entity_alias(self, alias: str) -> dict | None:
        """Resolve a known alias such as a ticker or common short name."""
        normalized = self.normalize_entity_name(alias)
        if not self._driver:
            canonical = self._aliases.get(normalized)
            return self._entities.get(canonical) if canonical else None
        records = await self.execute_cypher(
            """
            MATCH (e:Entity)
            WHERE $alias IN e.aliases
            RETURN e.name AS name, e.type AS type, e.description AS description
            LIMIT 1
            """,
            {"alias": normalized},
        )
        return records[0] if records else None

    async def find_entity_normalized(self, name: str) -> dict | None:
        """Resolve by normalized name, stripping common organization suffixes."""
        normalized = self.normalize_entity_name(name)
        if not self._driver:
            for entity in self._entities.values():
                if entity.get("normalized_name") == normalized:
                    return entity
            return None
        records = await self.execute_cypher(
            """
            MATCH (e:Entity)
            WHERE e.normalized_name = $normalized
            RETURN e.name AS name, e.type AS type, e.description AS description
            LIMIT 1
            """,
            {"normalized": normalized},
        )
        return records[0] if records else None

    async def find_entities_by_name_similarity(self, mention: str, threshold: float = 0.8, limit: int = 3) -> list[dict]:
        """Lightweight fallback for entity linking when vector name indexes are unavailable."""
        normalized = self.normalize_entity_name(mention)
        names = await self.get_all_entity_names()
        scored: list[tuple[float, str]] = []
        for name in names:
            score = SequenceMatcher(None, normalized, self.normalize_entity_name(name)).ratio()
            if score >= threshold:
                scored.append((score, name))
        scored.sort(reverse=True)
        return [
            {"name": name, "similarity": score}
            for score, name in scored[:limit]
        ]

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        """
        Multi-hop subgraph retrieval - a core GraphRAG capability
        Starting from a specified entity, traverse all related entities and relationships within N hops
        """
        if not self._driver:
            return self._memory_neighbors(entity_name, hops=hops)
        cypher = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(neighbor)
        RETURN
            start.name AS source,
            [r IN relationships(path) | type(r)] AS relations,
            neighbor.name AS target,
            neighbor.type AS target_type,
            neighbor.description AS target_desc
        LIMIT 50
        """
        return await self.execute_cypher(cypher, {"name": entity_name})

    async def traverse_metapath(
        self,
        start_entities: list[str],
        metapath: MetapathSpec,
        limit: int = 20,
    ) -> list[MetapathResult]:
        """Traverse a typed finance metapath from candidate start entities."""
        if not start_entities:
            return []
        if not self._driver:
            return self._memory_traverse_metapath(start_entities, metapath, limit=limit)

        rel_types = [self._sanitize_rel_type(step.relation) for step in metapath.steps]
        direction_parts = []
        for index, rel_type in enumerate(rel_types):
            if metapath.steps[index].direction == "out":
                direction_parts.append(f"-[r{index}:{rel_type}]->")
            else:
                direction_parts.append(f"<-[r{index}:{rel_type}]-")

        node_parts = ["(n0:Entity)"]
        for index, _step in enumerate(metapath.steps, start=1):
            node_parts.append(f"(n{index}:Entity)")

        match_parts: list[str] = []
        for index, relation_part in enumerate(direction_parts):
            match_parts.append(f"{node_parts[index]}{relation_part}{node_parts[index + 1]}")
        match_clause = ",".join(match_parts)
        type_checks = [f"coalesce(n{index}.type, '') = $type_{index}" for index in range(len(metapath.steps) + 1)]
        params: dict[str, Any] = {
            "start_entities": start_entities,
            "limit": limit,
            "metapath": metapath.name,
        }
        params["type_0"] = metapath.steps[0].from_type
        for index, step in enumerate(metapath.steps, start=1):
            params[f"type_{index}"] = step.to_type

        rel_returns = ", ".join(
            f"[n{index}.name, type(r{index}), n{index + 1}.name]" for index in range(len(metapath.steps))
        )
        cypher = f"""
        MATCH {match_clause}
        WHERE n0.name IN $start_entities
          AND {' AND '.join(type_checks)}
        RETURN
          n0.name AS start_entity,
          n{len(metapath.steps)}.name AS end_entity,
          [{rel_returns}] AS path
        LIMIT $limit
        """
        records = await self.execute_cypher(cypher, params)
        return [self._metapath_result_from_record(metapath, record) for record in records]

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict]:
        """Fuzzy-search entities"""
        if not self._driver:
            lowered = keyword.lower()
            matches = [
                entity
                for entity in self._entities.values()
                if lowered in entity["name"].lower() or lowered in entity.get("description", "").lower()
                if self._source_visible(entity.get("source", ""))
            ]
            return matches[:limit]
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($keyword)
           OR toLower(coalesce(e.description, '')) CONTAINS toLower($keyword)
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    async def get_all_entity_names(self, limit: int = 1000) -> list[str]:
        """Return entity names for alias/fuzzy entity resolution."""
        if not self._driver:
            return [
                name for name, entity in self._entities.items()
                if self._source_visible(entity.get("source", ""))
            ][:limit]
        records = await self.execute_cypher(
            "MATCH (e:Entity) RETURN e.name AS name LIMIT $limit",
            {"limit": limit},
        )
        return [record["name"] for record in records if record.get("name")]

    async def refresh_community_summaries(self) -> int:
        """Detect graph communities and precompute query-time summaries.

        The public prototype avoids per-query LLM community summaries. Production
        deployments can replace the local summarizer with richer offline LLM
        summaries, but query-time retrieval remains a lookup.
        """
        if self._driver:
            return await self._refresh_neo4j_community_summaries()

        entities = self._visible_memory_entities()
        relations = self._visible_memory_relations()
        communities, algorithm = self._detect_communities(list(entities), relations)
        summaries: dict[str, dict[str, Any]] = {}
        for members in communities:
            community_id = hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:12]
            rels = [
                f"{rel['head']} -[{rel['relation']}]-> {rel['tail']}"
                for rel in relations
                if rel["head"] in members and rel["tail"] in members
            ]
            summaries[community_id] = {
                "community_id": community_id,
                "members": sorted(members),
                "relations": rels,
                "algorithm": algorithm,
                "summary": await self._summarize_community(sorted(members), rels),
            }

        self._community_summaries = summaries
        return len(summaries)

    async def get_community_summaries(self, entities: list[str], limit: int = 3) -> list[dict[str, Any]]:
        """Return precomputed summaries that contain any linked query entity."""
        if self._driver:
            records = await self.execute_cypher(
                """
                MATCH (c:CommunitySummary)
                WHERE any(entity IN $entities WHERE entity IN c.members)
                RETURN
                    c.community_id AS community_id,
                    c.members AS members,
                    c.relations AS relations,
                    c.algorithm AS algorithm,
                    c.summary AS summary
                LIMIT $limit
                """,
                {"entities": entities, "limit": limit},
            )
            return records

        matches = [
            summary for summary in self._community_summaries.values()
            if set(entities) & set(summary.get("members", []))
        ]
        return matches[:limit]

    # delete operations
    async def delete_by_source(self, source: str) -> int:
        """Delete all entities and relationships by source (delete before rebuilding during incremental updates)"""
        source_prefixes = self._source_prefixes(source)
        memory_deleted = [
            name
            for name, ent in self._entities.items()
            if self._source_matches(ent.get("source", ""), source_prefixes)
        ]
        for name in memory_deleted:
            del self._entities[name]
        self._relations = [
            rel for rel in self._relations
            if not self._source_matches(rel.get("source", ""), source_prefixes)
            and rel.get("head") not in memory_deleted
            and rel.get("tail") not in memory_deleted
        ]
        await self.refresh_community_summaries()
        if not self._driver:
            return len(memory_deleted)
        cypher = """
        MATCH (e:Entity)
        WHERE e.source = $source
           OR e.source STARTS WITH $source_chunk_prefix
           OR e.source STARTS WITH $doc_id_chunk_prefix
        DETACH DELETE e
        RETURN count(e) AS deleted
        """
        records = await self.execute_cypher(
            cypher,
            {
                "source": source,
                "source_chunk_prefix": f"{source}#chunk-",
                "doc_id_chunk_prefix": f"{hashlib.sha256(source.encode()).hexdigest()[:16]}#chunk-",
            },
            read_only=False,
        )
        return records[0].get("deleted", 0) if records else 0

    # stats
    async def get_stats(self) -> dict:
        """Get graph statistics"""
        if not self._driver:
            return {
                "backend": "memory",
                "total_entities": len(self._entities),
                "total_relations": len(self._relations),
            }
        entity_count = await self.execute_cypher("MATCH (e:Entity) RETURN count(e) AS cnt")
        rel_count = await self.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return {
            "total_entities": entity_count[0]["cnt"] if entity_count else 0,
            "total_relations": rel_count[0]["cnt"] if rel_count else 0,
        }

    def _memory_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        seen = {entity_name}
        frontier = {entity_name}
        records: list[dict[str, Any]] = []

        for _ in range(hops):
            next_frontier: set[str] = set()
            for rel in self._relations:
                if not self._source_visible(rel.get("source", "")):
                    continue
                pairs = []
                if rel["head"] in frontier:
                    pairs.append((rel["head"], rel["tail"]))
                if rel["tail"] in frontier:
                    pairs.append((rel["tail"], rel["head"]))
                for source, target in pairs:
                    target_entity = self._entities.get(target, {})
                    records.append({
                        "source": source,
                        "relations": [rel["relation"]],
                        "target": target,
                        "target_type": target_entity.get("type", ""),
                        "target_desc": target_entity.get("description", ""),
                    })
                    if target not in seen:
                        seen.add(target)
                        next_frontier.add(target)
            frontier = next_frontier
            if not frontier:
                break
        return records

    def _memory_traverse_metapath(
        self,
        start_entities: list[str],
        metapath: MetapathSpec,
        limit: int = 20,
    ) -> list[MetapathResult]:
        visible_entities = self._visible_memory_entities()
        visible_relations = self._visible_memory_relations()
        results: list[MetapathResult] = []

        for start_entity in start_entities:
            start_record = visible_entities.get(start_entity)
            if not start_record or start_record.get("type") != metapath.steps[0].from_type:
                continue
            states: list[tuple[str, list[tuple[str, str, str]]]] = [(start_entity, [])]
            for step in metapath.steps:
                next_states: list[tuple[str, list[tuple[str, str, str]]]] = []
                for current_name, path in states:
                    current_record = visible_entities.get(current_name)
                    if not current_record or current_record.get("type") != step.from_type:
                        continue
                    for rel in visible_relations:
                        if rel["relation"] != self._sanitize_rel_type(step.relation):
                            continue
                        source = rel["head"] if step.direction == "out" else rel["tail"]
                        target = rel["tail"] if step.direction == "out" else rel["head"]
                        if source != current_name:
                            continue
                        target_record = visible_entities.get(target)
                        if not target_record or target_record.get("type") != step.to_type:
                            continue
                        next_states.append((target, [*path, (source, rel["relation"], target)]))
                states = next_states
                if not states:
                    break
            for end_entity, path in states:
                results.append(self._metapath_result_from_path(metapath, start_entity, end_entity, path))
                if len(results) >= limit:
                    return results
        return results

    @classmethod
    def _metapath_result_from_record(cls, metapath: MetapathSpec, record: dict[str, Any]) -> MetapathResult:
        path = tuple((str(head), str(rel), str(tail)) for head, rel, tail in record.get("path", []))
        return cls._metapath_result_from_path(
            metapath=metapath,
            start_entity=str(record.get("start_entity", "")),
            end_entity=str(record.get("end_entity", "")),
            path=path,
        )

    @staticmethod
    def _metapath_result_from_path(
        metapath: MetapathSpec,
        start_entity: str,
        end_entity: str,
        path: list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...],
    ) -> MetapathResult:
        path_tuple = tuple(path)
        evidence = "; ".join(f"{head} -[{relation}]-> {tail}" for head, relation, tail in path_tuple)
        score = round(min(1.0, len(path_tuple) / max(len(metapath.steps), 1)), 4)
        return MetapathResult(
            metapath_name=metapath.name,
            start_entity=start_entity,
            end_entity=end_entity,
            path=path_tuple,
            evidence=evidence,
            score=score,
        )

    @staticmethod
    def _source_prefixes(source: str) -> tuple[str, str, str]:
        canonical_source = os.path.abspath(source)
        doc_id = hashlib.sha256(canonical_source.encode()).hexdigest()[:16]
        return source, f"{canonical_source}#chunk-", f"{doc_id}#chunk-"

    @staticmethod
    def _source_matches(value: str, prefixes: tuple[str, str, str]) -> bool:
        exact, source_chunk_prefix, doc_id_chunk_prefix = prefixes
        return value == exact or value.startswith(source_chunk_prefix) or value.startswith(doc_id_chunk_prefix)

    @staticmethod
    def _source_visible(source: str) -> bool:
        doc_id = source.split("#chunk-", 1)[0] if source else ""
        return ingestion_registry.is_committed(doc_id)

    @staticmethod
    def _sanitize_rel_type(raw: str) -> str:
        rel_type = raw.upper().replace(" ", "_")
        if not re.match(r"^[A-Z_]+$", rel_type):
            return "RELATED_TO"
        if rel_type not in ALLOWED_RELATION_TYPES:
            return "RELATED_TO"
        return rel_type

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", name.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        suffixes = (" incorporated", " inc", " corporation", " corp", " limited", " ltd", " llc", " plc", " ag", " sa")
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                break
        return normalized

    @classmethod
    def _entity_aliases(cls, entity_or_name: Entity | str) -> list[str]:
        if isinstance(entity_or_name, Entity):
            name = entity_or_name.name
            properties = entity_or_name.properties
        else:
            name = entity_or_name
            properties = {}

        aliases = {cls.normalize_entity_name(name)}
        compact = re.sub(r"[^A-Za-z0-9]", "", name).lower()
        if compact:
            aliases.add(compact)

        acronym = "".join(part[0] for part in re.findall(r"[A-Za-z0-9]+", name)).lower()
        if len(acronym) >= 2:
            aliases.add(acronym)

        ticker = properties.get("ticker") or properties.get("symbol")
        if ticker:
            aliases.add(cls.normalize_entity_name(str(ticker)))
            aliases.add(re.sub(r"[^A-Za-z0-9.]", "", str(ticker)).lower())

        explicit_aliases = properties.get("aliases", [])
        if isinstance(explicit_aliases, str):
            explicit_aliases = [explicit_aliases]
        if isinstance(explicit_aliases, list):
            for alias in explicit_aliases:
                aliases.add(cls.normalize_entity_name(str(alias)))
                compact_alias = re.sub(r"[^A-Za-z0-9]", "", str(alias)).lower()
                if compact_alias:
                    aliases.add(compact_alias)

        return sorted(aliases)

    @staticmethod
    def _format_community_summary(members: list[str], relations: list[str]) -> str:
        """Compatibility helper for older tests and scripts."""
        return StructuredCommunitySummarizer.format(members, relations)

    async def _summarize_community(self, members: list[str], relations: list[str]) -> str:
        try:
            return await self._community_summarizer.summarize(members, relations)
        except Exception as exc:
            logger.warning("community_summary_failed_using_structured", error=str(exc))
            return await StructuredCommunitySummarizer().summarize(members, relations)

    def _visible_memory_entities(self) -> dict[str, dict[str, Any]]:
        return {
            name: entity
            for name, entity in self._entities.items()
            if self._source_visible(entity.get("source", ""))
        }

    def _visible_memory_relations(self) -> list[dict[str, Any]]:
        visible_entities = set(self._visible_memory_entities())
        return [
            relation
            for relation in self._relations
            if relation["head"] in visible_entities
            and relation["tail"] in visible_entities
            and self._source_visible(relation.get("source", ""))
        ]

    def _detect_communities(
        self,
        entity_names: list[str],
        relations: list[dict[str, Any]],
    ) -> tuple[list[set[str]], str]:
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities

            graph = nx.Graph()
            graph.add_nodes_from(entity_names)
            graph.add_edges_from((rel["head"], rel["tail"]) for rel in relations)
            if graph.number_of_nodes() == 0:
                return [], "louvain"
            communities = [set(community) for community in louvain_communities(graph, resolution=1.0, seed=42)]
            return communities, "louvain"
        except Exception as exc:
            logger.warning("community_detection_failed_using_connected_components", error=str(exc))
            return self._connected_components(entity_names, relations), "connected_components"

    @staticmethod
    def _connected_components(entity_names: list[str], relations: list[dict[str, Any]]) -> list[set[str]]:
        adjacency: dict[str, set[str]] = {name: set() for name in entity_names}
        for relation in relations:
            head = relation["head"]
            tail = relation["tail"]
            adjacency.setdefault(head, set()).add(tail)
            adjacency.setdefault(tail, set()).add(head)

        seen: set[str] = set()
        communities: list[set[str]] = []
        for entity in adjacency:
            if entity in seen:
                continue
            stack = [entity]
            members: set[str] = set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                members.add(current)
                stack.extend(adjacency.get(current, set()) - seen)
            communities.append(members)
        return communities

    async def _refresh_neo4j_community_summaries(self) -> int:
        """Detect Neo4j graph communities and persist precomputed summaries."""
        records = await self.execute_cypher(
            """
            MATCH (e:Entity)
            RETURN collect(DISTINCT e.name) AS entities
            """,
        )
        entity_names = records[0].get("entities", []) if records else []
        relation_records = await self.execute_cypher(
            """
            MATCH (a:Entity)-[r]-(b:Entity)
            RETURN a.name AS head, type(r) AS relation, b.name AS tail
            """,
        )
        relations = [
            {"head": record["head"], "relation": record["relation"], "tail": record["tail"]}
            for record in relation_records
            if record.get("head") and record.get("tail")
        ]
        communities, algorithm = self._detect_communities(entity_names, relations)
        summaries: list[dict[str, Any]] = []
        for members in communities:
            community_id = hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:12]
            rels = [
                f"{rel['head']} -[{rel['relation']}]-> {rel['tail']}"
                for rel in relations
                if rel["head"] in members and rel["tail"] in members
            ]
            summaries.append({
                "community_id": community_id,
                "members": sorted(members),
                "relations": rels,
                "algorithm": algorithm,
                "summary": await self._summarize_community(sorted(members), rels),
            })

        await self.execute_cypher("MATCH (c:CommunitySummary) DETACH DELETE c", read_only=False)
        for summary in summaries:
            await self.execute_cypher(
                """
                MERGE (c:CommunitySummary {community_id: $community_id})
                SET c.members = $members,
                    c.relations = $relations,
                    c.algorithm = $algorithm,
                    c.summary = $summary
                """,
                summary,
                read_only=False,
            )
        return len(summaries)
