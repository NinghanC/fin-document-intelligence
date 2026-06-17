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
from typing import Any

from agents.knowledge_extract_agent import Entity, Relation
from config import settings

WRITE_CYPHER_PATTERN = re.compile(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|LOAD|CALL\s+dbms)\b", re.IGNORECASE)


class KnowledgeGraphService:
    """Neo4j knowledge graph service"""

    def __init__(self) -> None:
        self._driver: Any = None
        self._entities: dict[str, dict[str, Any]] = {}
        self._relations: list[dict[str, Any]] = []
        self._community_summaries: dict[str, dict[str, Any]] = {}

    # lifecycle
    async def init(self) -> None:
        from neo4j import AsyncGraphDatabase
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        try:
            await self._ensure_indexes()
        except Exception:
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
            "version": version,
            "source": source,
            "updated_at": int(time.time()),
        }
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
            return self._entities.get(name)
        cypher = "MATCH (e:Entity {name: $name}) RETURN e"
        records = await self.execute_cypher(cypher, {"name": name})
        return records[0] if records else None

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

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict]:
        """Fuzzy-search entities"""
        if not self._driver:
            lowered = keyword.lower()
            matches = [
                entity
                for entity in self._entities.values()
                if lowered in entity["name"].lower() or lowered in entity.get("description", "").lower()
            ]
            return matches[:limit]
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    async def get_all_entity_names(self, limit: int = 1000) -> list[str]:
        """Return entity names for alias/fuzzy entity resolution."""
        if not self._driver:
            return list(self._entities)[:limit]
        records = await self.execute_cypher(
            "MATCH (e:Entity) RETURN e.name AS name LIMIT $limit",
            {"limit": limit},
        )
        return [record["name"] for record in records if record.get("name")]

    async def refresh_community_summaries(self) -> int:
        """Precompute simple connected-component community summaries.

        The public prototype avoids per-query LLM community summaries. Production
        deployments can replace this with Leiden/Louvain plus richer offline
        summarization.
        """
        if self._driver:
            return await self._refresh_neo4j_community_summaries()

        adjacency: dict[str, set[str]] = {name: set() for name in self._entities}
        for relation in self._relations:
            head = relation["head"]
            tail = relation["tail"]
            adjacency.setdefault(head, set()).add(tail)
            adjacency.setdefault(tail, set()).add(head)

        seen: set[str] = set()
        summaries: dict[str, dict[str, Any]] = {}
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

            community_id = hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:12]
            rels = [
                f"{rel['head']} -[{rel['relation']}]-> {rel['tail']}"
                for rel in self._relations
                if rel["head"] in members and rel["tail"] in members
            ]
            summaries[community_id] = {
                "community_id": community_id,
                "members": sorted(members),
                "summary": self._format_community_summary(sorted(members), rels),
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
                RETURN c.community_id AS community_id, c.members AS members, c.summary AS summary
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
    def _sanitize_rel_type(raw: str) -> str:
        rel_type = raw.upper().replace(" ", "_")
        if not re.match(r"^[A-Z_]+$", rel_type):
            return "RELATED_TO"
        return rel_type

    @staticmethod
    def _format_community_summary(members: list[str], relations: list[str]) -> str:
        member_text = ", ".join(members[:8])
        relation_text = "; ".join(relations[:8]) if relations else "No direct relationships captured."
        return f"Community containing {member_text}. Relationships: {relation_text}"

    async def _refresh_neo4j_community_summaries(self) -> int:
        """Small connected-component approximation for Neo4j deployments without GDS."""
        records = await self.execute_cypher(
            """
            MATCH (a:Entity)
            OPTIONAL MATCH (a)-[r]-(b:Entity)
            RETURN a.name AS entity, collect(DISTINCT b.name) AS neighbors
            """,
        )
        adjacency: dict[str, set[str]] = {
            record["entity"]: {neighbor for neighbor in record.get("neighbors", []) if neighbor}
            for record in records
            if record.get("entity")
        }
        seen: set[str] = set()
        summaries: list[dict[str, Any]] = []
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
            community_id = hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:12]
            summaries.append({
                "community_id": community_id,
                "members": sorted(members),
                "summary": self._format_community_summary(sorted(members), []),
            })

        await self.execute_cypher("MATCH (c:CommunitySummary) DETACH DELETE c", read_only=False)
        for summary in summaries:
            await self.execute_cypher(
                """
                MERGE (c:CommunitySummary {community_id: $community_id})
                SET c.members = $members, c.summary = $summary
                """,
                summary,
                read_only=False,
            )
        return len(summaries)
