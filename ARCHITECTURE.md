# Architecture

FinSight Assistant is a sanitized prototype of a financial document-intelligence system. The public code keeps the agent boundaries, ingestion workflow, hybrid retrieval flow, CDC shape, and deployment topology while replacing sensitive production integrations with local equivalents.

## Data Flow

```text
----------------------+      +-----------------------+      +-----------------------+
| Upload / API Request | ---> | LangGraph Orchestrator| ---> | Vector + Graph Stores |
+----------------------+      +-----------------------+      +-----------------------+
          |                              |                              |
          | parse: ~100-800 ms           | extract: model dependent     | store: ~50-500 ms
          v                              v                              v
  DocParserAgent              KnowledgeExtractAgent          ChromaDB/PGVector + Neo4j
          |                              |                              |
          +------------------------------+------------------------------+
                                         |
                                         v
                              QAAgent / GraphRAGPipeline
                                         |
                                         v
                              Cited answer + confidence
```

## ADR 001: LangGraph Over CrewAI

LangGraph was selected because the core flows are state-machine oriented: parse, extract, store, answer, retry, and update. The project benefits from explicit nodes, deterministic transitions, and direct control over retry/error branches. Crew-style role delegation is useful for open-ended collaboration, but this system needs auditable workflow states and predictable production behavior.

## ADR 002: Hybrid Retrieval Over Pure Vector Search

Pure vector retrieval is useful for semantic recall, but financial questions often depend on structured relationships: fund to exposure, policy to obligation, product to risk factor, and document to source lineage. The system combines vector retrieval with graph retrieval, then reranks the combined contexts before answer generation.

## ADR 003: ChromaDB Locally, PGVector For Production-Oriented Retrieval

ChromaDB is convenient for local demos and fast iteration. PGVector is the production-oriented path because it fits common managed PostgreSQL deployments, supports SQL observability and backup workflows, and keeps vector search close to metadata filtering.

## ADR 004: CDC For Incremental Updates

Reprocessing every document after each change is expensive and slow. The CDC path tracks file/API/Kafka change events, computes deltas, and refreshes only the affected document knowledge. The public version includes the processor and endpoint shape; production deployments would connect this to managed queues, audit logs, and stronger lineage metadata.

## Failure Modes

- Neo4j unavailable: graph writes and graph retrieval degrade; vector retrieval can still serve semantic answers.
- Vector store unavailable: the API reports degraded health; graph retrieval can still provide structured context when available.
- LLM timeout: extraction or answer generation fails for that request; production deployments should retry with bounded backoff and record a failed job.
- Embedding failure: vector write/search returns zero results; graph context and source-level errors remain available.
- Kafka unavailable: CDC consumer does not start or stops processing events; manual update endpoints remain available.
- Invalid uploads: files are rejected by size, extension, and magic-byte validation before ingestion.

## Production Gaps In This Prototype

The public prototype intentionally omits tenant isolation, enterprise SSO, managed secret distribution, internal prompts, proprietary documents, full evaluation datasets, regulated audit reporting, and cloud-specific infrastructure modules.
