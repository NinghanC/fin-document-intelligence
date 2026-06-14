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

import time
from typing import Any

from agents.knowledge_extract_agent import Entity, Relation
from config import settings


class KnowledgeGraphService:
    """Neo4j knowledge graph service"""

    def __init__(self) -> None:
        self._driver: Any = None

    # ── lifecycle ────────────────────────────────────────────

    async def init(self) -> None:
        from neo4j import AsyncGraphDatabase
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._ensure_indexes()

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

    # ── entity operations ────────────────────────────────────

    async def upsert_entity(self, entity: Entity, version: int = 1, source: str = "") -> None:
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
                "version": version,
                "source": source,
                "now": int(time.time()),
            })

    async def add_relation(self, relation: Relation, source: str = "") -> None:
        """Create relationships between entities"""
        if not self._driver:
            return
        rel_type = relation.relation.upper().replace(" ", "_")
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

    # ── query operations ─────────────────────────────────────

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute arbitrary Cypher queries"""
        if not self._driver:
            return []
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return records

    async def get_entity(self, name: str) -> dict | None:
        """Query a single entity"""
        cypher = "MATCH (e:Entity {name: $name}) RETURN e"
        records = await self.execute_cypher(cypher, {"name": name})
        return records[0] if records else None

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        """
        Multi-hop subgraph retrieval - a core GraphRAG capability
        Starting from a specified entity, traverse all related entities and relationships within N hops
        """
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
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    # ── delete operations ────────────────────────────────────

    async def delete_by_source(self, source: str) -> int:
        """Delete all entities and relationships by source (delete before rebuilding during incremental updates)"""
        cypher = """
        MATCH (e:Entity {source: $source})
        DETACH DELETE e
        RETURN count(e) AS deleted
        """
        records = await self.execute_cypher(cypher, {"source": source})
        return records[0].get("deleted", 0) if records else 0

    # ── stats ────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """Get graph statistics"""
        entity_count = await self.execute_cypher("MATCH (e:Entity) RETURN count(e) AS cnt")
        rel_count = await self.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return {
            "total_entities": entity_count[0]["cnt"] if entity_count else 0,
            "total_relations": rel_count[0]["cnt"] if rel_count else 0,
        }
