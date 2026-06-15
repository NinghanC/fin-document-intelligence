# FinSight Assistant

This repository is an architectural extract of an internal financial document intelligence system. This code demonstrates the structure, design decisions, and integration patterns — not a runnable standalone product.

FinSight Assistant is a private document intelligence assistant designed for financial services teams. The goal is to help analysts, operations teams, and risk reviewers search internal documents, trace answers back to source material, and keep the knowledge base updated as reports and policies change.

This repository is a realistic prototype of that solution. It shows the architecture, workflows, and integration points needed for a financial document assistant, while keeping the implementation small enough to run locally with Docker Compose.

## Business Context

The target workflow is common in financial organizations. Teams often have useful knowledge scattered across:

- investment memos and research notes
- policy and compliance documents
- fund product descriptions
- client-facing reports
- risk review materials
- operational playbooks
- meeting notes and internal FAQs

Keyword search was not enough. People often needed answers such as:

- "Which funds mention exposure to renewable infrastructure?"
- "What changed in the latest risk policy compared with the previous version?"
- "Which products have liquidity constraints?"
- "Find the source document for this compliance requirement."
- "Summarize the relationship between a company, sector, product, and risk factor."

For this type of environment, a useful assistant has to be conservative. It should not simply produce fluent answers. It needs source references, confidence signals, structured retrieval, and a path toward private cloud deployment.

## Problem Statement

Financial teams need an assistant that can work across messy internal documents without losing traceability. The main requirements are:

- Parse mixed document formats including PDF, Word, spreadsheets, Markdown, and text.
- Split documents into searchable chunks with metadata.
- Extract entities such as companies, products, sectors, people, dates, policies, and risk concepts.
- Combine semantic search with relationship-aware graph retrieval.
- Cite sources so users can verify every answer.
- Support incremental updates when documents change.
- Keep model/provider configuration replaceable for Azure, AWS, or Databricks deployments.

## Solution Overview

FinSight Assistant uses a multi-agent pipeline:

- `DocParserAgent` turns uploaded documents into normalized chunks.
- `KnowledgeExtractAgent` extracts entities, relations, and events from those chunks.
- `QAAgent` answers user questions using hybrid retrieval.
- `KnowledgeUpdateAgent` handles changed documents and incremental refresh logic.

The assistant combines vector search and a Neo4j knowledge graph. Vector search helps with fuzzy semantic matching, while the graph helps with entity relationships and multi-hop questions such as "which product is connected to this sector and this risk factor?"

## Architecture

```text
Financial analyst / operations user
              |
        Web UI / REST API
              |
          FastAPI service
              |
      LangGraph orchestration
              |
  +-----------+------------+
  |                        |
Document parser      QA / update agents
  |                        |
Knowledge extractor       |
  |                        |
  +-------> Retrieval layer <------+
              |                    |
        Vector store          Neo4j graph
        Chroma/PGVector       Entities/relations
              |
      Cloud LLM provider
  Azure OpenAI / AWS Bedrock / Databricks
```

## Core Workflows

### 1. Document Ingestion

When a user uploads a financial document, the system:

1. Detects the file type.
2. Extracts text from PDFs, spreadsheets, Markdown, or plain text.
3. Falls back to vision/OCR paths for image-heavy documents.
4. Splits content into chunks.
5. Extracts structured entities and relations.
6. Stores searchable chunks and graph relationships.

### 2. Question Answering

When a user asks a question, the system:

1. Classifies the question intent.
2. Rewrites it into retrieval-friendly queries.
3. Searches the vector store for semantically relevant passages.
4. Queries the knowledge graph for entity relationships.
5. Reranks the retrieved context.
6. Generates an answer with source references.

### 3. Incremental Update

Financial documents change often. Policies, product disclosures, and risk memos may be revised weekly. The update workflow is designed to:

1. Detect file or CDC-style change events.
2. Compare before/after content where available.
3. Reprocess only affected documents or chunks.
4. Refresh vector and graph records.
5. Preserve version metadata for auditability.

## Why Hybrid RAG + Knowledge Graph

Pure vector retrieval is useful, but it is not enough for many financial questions. It can find similar text, but it does not naturally model relationships such as:

- fund -> sector exposure -> risk factor
- company -> subsidiary -> region
- policy -> control requirement -> owner
- product -> liquidity rule -> disclosure document

The graph layer gives the assistant a structured view of relationships. The vector layer gives it flexible semantic recall. Together, they support more realistic financial knowledge workflows than either approach alone.

## Cloud Provider Strategy

The model layer is intentionally provider-neutral and configured through environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `EMBEDDING_MODEL`

The intended production targets are:

- Azure OpenAI for teams already using Microsoft cloud and enterprise identity.
- AWS Bedrock through an OpenAI-compatible gateway or adapter.
- Databricks Mosaic AI Model Serving for teams keeping data and models close to the lakehouse.

## Current Implementation Status

Implemented:

- FastAPI backend with upload, QA, admin stats, update, and health endpoints.
- Static browser UI for uploads, Q&A, and status display.
- Four-agent structure for parsing, extraction, QA, and updates.
- LangGraph workflow orchestration.
- Neo4j service wrapper.
- Chroma/PGVector service abstraction.
- Docker Compose for local Neo4j, ChromaDB, Kafka, and API services.
- English source, prompts, README, and sample text data.

Partially implemented:

- Vector storage and retrieval behavior.
- Incremental update logic.
- GraphRAG path and subgraph retrieval.
- OCR and multimodal document parsing.
- Provider-specific production configuration.

Not production-ready yet:

- Authentication and role-based access control.
- Tenant isolation.
- Secure document upload scanning.
- Background job queue and retry handling.
- Full observability with traces, metrics, dashboards, and alerts.
- Production-grade CI/CD.
- Secrets management through Key Vault, Secrets Manager, or Databricks secrets.
- Comprehensive unit, integration, and end-to-end tests.
- Formal audit logging and compliance controls.

## Quick Start

Copy the environment template:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` with a real model provider endpoint:

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=your-provider-endpoint
OPENAI_MODEL=your-model
EMBEDDING_MODEL=text-embedding-3-small
```

When running through Docker Compose, use service hostnames:

```env
NEO4J_URI=bolt://neo4j:7687
CHROMA_HOST=chromadb
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
API_PORT=8080
```

Start the local stack:

```bash
docker compose up --build
```

Open the app:

```text
http://localhost:8080
```

Open API docs:

```text
http://localhost:8080/docs
```

Open Neo4j Browser:

```text
http://localhost:7474
```

Default local Neo4j credentials:

```text
neo4j / password
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/ingest/upload` | Upload and ingest one document |
| `POST` | `/api/ingest/batch` | Upload and ingest multiple documents |
| `POST` | `/api/qa/ask` | Ask a question using hybrid retrieval |
| `GET` | `/api/admin/stats` | Read vector store and graph statistics |
| `POST` | `/api/admin/update` | Trigger a document update workflow |
| `GET` | `/api/health` | Health check |

## Project Layout

```text
backend/
  agents/          Parser, extractor, QA, and update agents
  api/             FastAPI application
  config/          Environment-based settings
  orchestrator/    LangGraph workflow definitions
  services/        Vector store, graph, CDC, multimodal, and embedding helpers
  static/          Browser UI
  uploads/         Sample/input documents
docker-compose.yml Local infrastructure
```

## Roadmap

### Phase 1: Reliable Local Demo

- Make upload, ingestion, and QA stable on local Docker Compose.
- Use a real Azure/AWS/Databricks model endpoint.
- Improve vector store persistence and retrieval.
- Add basic tests for parsing, extraction, and QA flow.

### Phase 2: Internal Financial Assistant MVP

- Add login and role-based document access.
- Add asynchronous ingestion jobs.
- Add upload validation, file size limits, and malware scanning hooks.
- Add structured audit logs for answers and source documents.
- Add observability and failure retry handling.

### Phase 3: Production Hardening

- Add tenant isolation and data retention policies.
- Move secrets to cloud-native secret stores.
- Add CI/CD and cloud deployment modules.
- Add compliance review workflows.
- Add answer evaluation, hallucination checks, and prompt-injection defenses.

## Notes

This project is intentionally written as a realistic financial-document assistant prototype rather than a finished production system. It demonstrates the architecture and engineering direction, but a real financial deployment would require security, compliance, testing, and operations hardening before use with confidential data.
