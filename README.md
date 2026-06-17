# FinSight Assistant

FinSight Assistant is a private document intelligence assistant for financial services teams. It is designed to help analysts, operations teams, compliance reviewers, and risk teams search internal documents, trace answers back to source material, and keep knowledge updated as reports, policies, and product documents change.

This repository is a realistic prototype of that system. It demonstrates the architecture, code structure, ingestion flow, hybrid retrieval design, and deployment shape for a financial-document assistant, while staying small enough to run locally with Docker Compose.

## Prototype Scope

This public repository is a sanitized prototype adapted from an internal document-intelligence architecture. The README describes the target architecture and behavior of that internal project, while the code in this repository keeps the same module boundaries and workflows with sensitive production details removed.

The public version is intended to show engineering design, service decomposition, and end-to-end data flow. It is not a full production export of the internal system.

The following production-specific components have been removed, simplified, or replaced with local equivalents:

- proprietary document samples, schemas, prompts, and evaluation datasets
- cloud deployment modules, private infrastructure identifiers, and environment-specific configuration
- authentication, authorization, tenant isolation, and internal access-control policies
- production observability, alerting, audit logging, and compliance reporting integrations
- internal background job infrastructure and managed ingestion queues
- production vector-store persistence, indexing strategy, and retrieval tuning
- complete graph lineage metadata, source-policy mapping, and regulated workflow hooks
- provider-specific model routing, fallback policies, and cost-control configuration
- benchmark results, business metrics, and internal performance traces

Because of this sanitization, a few implementation details in the public code are intentionally lighter than the architecture described here. For example, the public version uses local/demo-safe fallbacks for model calls and graph storage when managed providers are not configured, while production retrieval tuning, source-policy mapping, and regulated workflow hooks are reduced compared with the internal implementation.

## What Problem It Solves

Financial teams often work with knowledge spread across:

- investment memos and research notes
- policy and compliance documents
- fund product descriptions
- client-facing reports
- risk review materials
- operational playbooks
- meeting notes and internal FAQs

Simple keyword search does not answer questions such as:

- Which funds mention exposure to renewable infrastructure?
- What changed in the latest risk policy compared with the previous version?
- Which products have liquidity constraints?
- Find the source document for this compliance requirement.
- Summarize the relationship between a company, sector, product, and risk factor.

For this environment, the assistant has to be conservative. It should not only generate fluent text. It needs source references, relationship-aware retrieval, confidence signals, and a deployment path that keeps provider choice replaceable.

## Solution Summary

FinSight Assistant uses a four-agent pipeline:

- `DocParserAgent` parses uploaded documents into normalized chunks.
- `KnowledgeExtractAgent` extracts entities, relations, and events from those chunks.
- `QAAgent` answers questions with vector and graph retrieval context.
- `KnowledgeUpdateAgent` handles changed documents and incremental refresh logic.

The architecture is designed around two complementary knowledge representations:

- Vector representation for semantic recall.
- Knowledge graph representation for entities, relationships, and multi-hop reasoning.

This hybrid design is more useful for financial workflows than a pure vector-search system because many questions depend on structured relationships, not just semantic similarity.

## Architecture

```text
+------------------------------------------------+
| User Interface Layer                           |
| Web UI (SPA, 3 tabs)                           |
| REST API (FastAPI, 7 endpoints)                |
| Swagger / OpenAPI documentation                |
+------------------------+-----------------------+
                         |
                         | HTTP
                         v
+------------------------------------------------+
| Orchestration Layer (LangGraph)                |
|                                                |
|   +----------------------------------------+   |
|   | Ingest Pipeline                        |   |
|   | parse -> extract -> store              |   |
|   +----------------------------------------+   |
|                                                |
|   +----------------------------------------+   |
|   | QA Pipeline                            |   |
|   | answer -> END                          |   |
|   +----------------------------------------+   |
|                                                |
|   +----------------------------------------+   |
|   | Update Pipeline                        |   |
|   | process -> retry -> END                |   |
|   +----------------------------------------+   |
+------------------------+-----------------------+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
+------------------------+ +-----------------------+ +----------------------+
| Agent Layer            | | Service Layer         | | Infrastructure Layer |
|                        | |                       | |                      |
| DocParserAgent         | | VectorStoreService    | | ChromaDB / PGVector  |
| KnowledgeExtractAgent  | | KnowledgeGraphService | | Neo4j                |
| QAAgent                | | GraphRAGPipeline      | | Kafka                |
| KnowledgeUpdateAgent   | | CDCProcessor          | | EmbeddingWorker      |
|                        | | MultimodalService     | | (subprocess)         |
+------------------------+ +-----------------------+ +----------------------+
```

### Runtime Services

Docker Compose starts the local infrastructure:

| Service | Purpose | Ports |
| --- | --- | --- |
| FastAPI backend | API and static UI | `8080` |
| Neo4j | Knowledge graph | `7474`, `7687` |
| ChromaDB | Local vector database service; the backend supports embedded persistent-client mode and Docker HTTP mode | `8000` |
| PGVector | PostgreSQL vector retrieval backend for production-oriented semantic search | `5432` |
| Kafka | CDC-style event queue | `29092` on the host, `9092` inside Docker |
| Zookeeper | Kafka dependency | internal |

## Repository Layout

```text
backend/
  agents/
    doc_parser_agent.py          Document parsing agent
    knowledge_extract_agent.py   Entity/relation/event extraction agent
    qa_agent.py                  Hybrid retrieval and answer generation agent
    knowledge_update_agent.py    Incremental update agent
  api/
    main.py                      FastAPI entrypoint and REST endpoints
  config/
    settings.py                  Environment-based configuration
  orchestrator/
    graph.py                     LangGraph workflow definitions
  services/
    vector_store.py              ChromaDB/PGVector abstraction
    knowledge_graph.py           Neo4j abstraction
    graph_rag.py                 GraphRAG retrieval pipeline
    cdc_processor.py             CDC event normalization and diffing
    multimodal.py                Cross-modal embedding/reranking helper
    embedding_worker.py          Isolated embedding subprocess
  static/
    index.html                   Browser UI
    app.js                       UI interactions and API calls
    style.css                    Styling
  tests/
  utils/
docker-compose.yml               Local infrastructure
README.md                        Project documentation
```

Generated runtime data is intentionally excluded from Git:

- `backend/.env`
- `backend/uploads/`
- `backend/chroma_data/`
- `.uv-cache/`

## Core Workflows

### End-to-End Fund Example

The following example shows the full path from a fund document upload to a grounded answer.

```text
Input
  Fund operations team uploads:
  "Q4_global_income_fund_risk_report.pdf"
        |
        v
+-------------------------------------------------------------+
| FastAPI upload_document()                                   |
| file = save_path                                            |
| ingest_wf.ainvoke({"file_paths": [file]})                   |
+-------------------------------------------------------------+
        |
        v
+-------------------------------------------------------------+
| parse_documents(state)                                      |
| DocParserAgent.parse_batch() -> chunks                      |
| state update: chunks = [DocumentChunk x 45]                 |
| Example content: fund exposure, liquidity terms, risk notes |
+-------------------------------------------------------------+
        |
        v
+-------------------------------------------------------------+
| extract_knowledge(state)                                    |
| KnowledgeExtractAgent.extract() -> extractions              |
| state update: extractions = [ExtractionResult ...]          |
| Example entities:                                           |
|   Global Income Fund, liquidity constraint, bond sleeve,    |
|   emerging-market debt, redemption gate                     |
+-------------------------------------------------------------+
        |
        v
        +------------------------------+------------------------------+
        |                              |                              |
        v                              v                              |
+-----------------------------+  +------------------------------------+
| store_vectors(state)        |  | store_graph(state)                 |
| VectorStoreService          |  | KnowledgeGraphService              |
| add_chunks(chunks)          |  | upsert_entity() / add_relation()   |
| vectors_stored: 45          |  | entities_stored: 23                |
+-----------------------------+  +------------------------------------+
        |                              |
        +--------------+---------------+
                       |
                       v
+-------------------------------------------------------------+
| HTTP Response                                                |
| IngestResponse                                               |
| file_name: "Q4_global_income_fund_risk_report.pdf"          |
| chunks_count: 45                                             |
| entities_count: 23                                           |
| relations_count: 18                                          |
+-------------------------------------------------------------+

User Question
  POST /api/qa/ask
  {"question": "Which funds mention liquidity constraints?"}
        |
        v
+-------------------------------------------------------------+
| process_question(state)                                     |
| QAAgent.answer() -> QAResult                                |
|                                                             |
| Internal steps:                                             |
| 1. classify_intent -> factoid / analytical                  |
| 2. rewrite_query -> queries + entities                      |
| 3. retrieve from vector store                               |
| 4. retrieve from knowledge graph                            |
| 5. hybrid_rerank(vector, graph)                             |
| 6. generate_answer with cited context                       |
+-------------------------------------------------------------+
        |
        v
+-------------------------------------------------------------+
| HTTP Response                                                |
| QuestionResponse                                             |
| answer: "Global Income Fund mentions liquidity constraints   |
|          in the redemption and bond sleeve risk sections."   |
| confidence: 0.87                                             |
| sources: [source, score, retrieval_type, snippet]            |
| reasoning_steps: [...]                                      |
+-------------------------------------------------------------+
```

### 1. Document Ingestion

The ingestion pipeline turns an uploaded document into vector-searchable chunks and graph-searchable knowledge.

```text
POST /api/ingest/upload
        |
Save file under UPLOAD_DIR
        |
LangGraph ingest workflow
        |
parse -> extract -> store_vectors -> store_graph -> END
```

Implementation flow:

1. `api/main.py` receives the uploaded file and stores it under `settings.upload_dir`.
2. `orchestrator/graph.py` invokes the `ingest` graph.
3. `DocParserAgent.parse_batch()` parses each file.
4. `KnowledgeExtractAgent.extract()` extracts entities, relations, and events.
5. `VectorStoreService.add_chunks()` handles vector-store ingestion behavior.
6. `KnowledgeGraphService.upsert_entity()` and `add_relation()` write graph records.
7. The API returns `IngestResponse` with chunk, entity, and relation counts.

### 2. Document Parsing

`DocParserAgent` routes files by extension:

| Extension | Type | Parser path |
| --- | --- | --- |
| `.pdf` | PDF | PyPDF2 text extraction, with vision fallback when text extraction fails |
| `.png`, `.jpg`, `.jpeg` | Image | Tesseract OCR plus LLM vision description |
| `.csv` | Table | CSV rows converted to structured text |
| `.xlsx`, `.xls` | Table | Excel sheets converted to structured text |
| `.txt` | Text | Direct UTF-8 read |
| `.md` | Markdown | Direct UTF-8 read |

Chunks are produced with:

```text
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
```

Each `DocumentChunk` carries:

- `content`
- `doc_id`
- `chunk_index`
- `doc_type`
- `metadata`
- optional `embedding`

### 3. Knowledge Extraction

`KnowledgeExtractAgent` converts unstructured chunks into structured knowledge:

- entities
- relations
- events

It uses an LLM JSON prompt and then parses the response into dataclasses:

- `Entity`
- `Relation`
- `KnowledgeEvent`
- `ExtractionResult`

Deduplication happens across chunks:

- entities are deduplicated by `(name, type)`
- relations are deduplicated by `(head, relation, tail)`

This prevents the same financial product, company, policy, or risk concept from being duplicated repeatedly across chunks.

### 4. Vector and Graph Storage

The system maintains two storage paths:

| Storage | Purpose |
| --- | --- |
| Vector store | Semantic similarity search over document chunks |
| Neo4j graph | Entity relationships and multi-hop reasoning |

The current vector-store implementation is intentionally defensive but functional for a local prototype:

- ChromaDB operations run through a thread pool around either a local persistent client or a Docker-hosted HTTP client.
- `add_chunks()` writes documents, metadata, IDs, and embeddings to the configured vector backend.
- ChromaDB search embeds the query and reads back matching documents, metadata, and distances.
- PGVector remains the more direct path for real vector retrieval: Docker Compose starts a pgvector-enabled PostgreSQL service, `add_chunks()` writes texts and metadata through LangChain PGVector, and `search()` reads results through `similarity_search_with_score()`.

This means the architecture is in place, while production-grade vector operations still need hardening around indexing policy, failure handling, and observability.

### 5. Question Answering

The QA pipeline is managed by `QAAgent`:

```text
question
  -> classify intent
  -> rewrite query
  -> retrieve from vector store
  -> retrieve from knowledge graph
  -> hybrid rerank
  -> generate answer
```

The intent classifier supports:

- `factoid`
- `analytical`
- `comparative`
- `procedural`
- `exploratory`

Hybrid retrieval merges vector, subgraph, path, and cached community-summary contexts. The public prototype uses neutral default weights so it does not pretend hand-picked multipliers are learned relevance scores:

```text
vector = 1.0
subgraph = 1.0
path = 1.0
community = 1.0
```

Non-uniform GraphRAG weights should be tuned with a labeled retrieval set. The `bench/` directory includes a small evaluation scaffold for expected-source checks and future recall@k tuning.

The response includes:

- answer
- retrieval-quality confidence signal
- intent
- source snippets
- reasoning steps

### 6. Incremental Update

The update flow handles document changes:

```text
changes -> process -> conditional retry -> END
```

`KnowledgeUpdateAgent` supports:

- created files
- modified files
- deleted files
- file hash comparison
- batch processing
- one retry pass through LangGraph

`CDCProcessor` also provides helpers for:

- filesystem event normalization
- Kafka message normalization
- before/after text diffing
- version bumping

This is useful for financial environments where product documents, risk policies, and operational playbooks are revised frequently.

## Why Hybrid RAG and Knowledge Graph

Pure vector retrieval is good at finding semantically similar passages, but it does not naturally model structured relationships such as:

- fund -> sector exposure -> risk factor
- company -> subsidiary -> region
- policy -> control requirement -> owner
- product -> liquidity rule -> disclosure document

Pure graph retrieval is good at relationships, but it cannot always answer broad semantic questions from long text.

FinSight combines both:

- vector search for recall
- graph retrieval for structure
- LLM answer generation for synthesis

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serve the browser UI |
| `POST` | `/api/ingest/upload` | Upload and ingest one document |
| `POST` | `/api/ingest/batch` | Upload and ingest multiple documents |
| `POST` | `/api/qa/ask` | Ask a question using hybrid retrieval |
| `GET` | `/api/admin/stats` | Read vector store and graph statistics |
| `POST` | `/api/admin/update` | Trigger a document update workflow |
| `GET` | `/api/health` | Health check |

Example:

```bash
curl -X POST http://localhost:8080/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which products mention liquidity constraints?"}'
```

## Configuration

Configuration is loaded with `pydantic-settings` from environment variables or `backend/.env`.

Important variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Provider API key |
| `OPENAI_BASE_URL` | OpenAI-compatible provider endpoint |
| `OPENAI_MODEL` | Chat model or deployment name |
| `EMBEDDING_MODEL` | Embedding model name |
| `EMBEDDING_PROVIDER` | `auto`, `openai`, `local`, or `hash`; `auto` uses provider embeddings when configured and demo-safe hash embeddings otherwise |
| `NEO4J_URI` | Neo4j Bolt URI |
| `NEO4J_USER` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `VECTOR_STORE_TYPE` | `chroma` or `pgvector` |
| `CHROMA_MODE` | `local` for embedded persistence or `http` for Docker-hosted ChromaDB |
| `CHROMA_HOST` | ChromaDB host when `CHROMA_MODE=http` |
| `CHROMA_PORT` | ChromaDB port when `CHROMA_MODE=http` |
| `PGVECTOR_DSN` | PGVector connection string |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap server |
| `UPLOAD_DIR` | File upload directory |

The intended model-provider targets are:

- Azure OpenAI
- AWS Bedrock through an OpenAI-compatible gateway or adapter
- Databricks Mosaic AI Model Serving

## Quick Start

Copy the environment template:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` with a real provider endpoint:

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=your-provider-endpoint
OPENAI_MODEL=your-model
EMBEDDING_MODEL=text-embedding-3-small
```

When running with Docker Compose, use service hostnames:

```env
NEO4J_URI=bolt://neo4j:7687
CHROMA_MODE=http
CHROMA_HOST=chromadb
PGVECTOR_DSN=postgresql://postgres:postgres@pgvector:5432/knowledge
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
API_PORT=8080
```

Start the stack:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8080
```

API docs:

```text
http://localhost:8080/docs
```

Neo4j Browser:

```text
http://localhost:7474
```

Local Neo4j credentials:

```text
neo4j / local-password
```

These credentials are local demo defaults only.

The bundled PGVector service also uses local demo defaults:

```text
postgres / postgres
```

## Local Development

You can also run the backend directly after installing dependencies:

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

The subprocess `EmbeddingWorker` is optional. Install it only when you want `EMBEDDING_PROVIDER=local`:

```bash
pip install -r requirements-local-embeddings.txt
```

For local infrastructure only:

```bash
docker compose up -d neo4j chromadb pgvector zookeeper kafka
```

## Implementation Status

Implemented:

- FastAPI backend with upload, QA, stats, update, and health endpoints.
- Static browser UI for Q&A, uploads, and dashboard stats.
- Four-agent architecture.
- LangGraph workflows for ingestion, QA, and update.
- Document parser for PDF, images, CSV, Excel, text, and Markdown.
- LLM-based entity, relation, and event extraction.
- Neo4j entity and relation writing.
- ChromaDB/PGVector vector-store abstraction.
- PGVector write and similarity-search path for production-oriented retrieval.
- CDC and file-change update scaffolding.
- Docker Compose local stack.

Partially implemented:

- Production-grade vector indexing, retrieval tuning, and observability.
- GraphRAG integration into the main QA path.
- Fine-grained incremental updates.
- OCR and vision robustness for scanned financial PDFs.
- Provider-specific deployment templates.

Known gaps:

- No authentication or role-based access control.
- No tenant isolation.
- No background job queue for ingestion.
- No malware scanning or file sandboxing for uploads.
- No full observability stack.
- No formal audit log for regulated workflows.
- Minimal automated tests.
- ChromaDB calls are isolated through sync wrappers for local runtime stability.

## Production Hardening Roadmap

### Phase 1: Reliable Local Demo

- Use a real Azure, AWS, or Databricks model endpoint.
- Stabilize vector persistence and retrieval.
- Add parser tests for PDF, CSV, Excel, text, and Markdown.
- Add basic API integration tests.

### Phase 2: Internal Financial Assistant MVP

- Add login and role-based document access.
- Move ingestion to background jobs.
- Add upload validation, file size limits, and malware scanning hooks.
- Add answer audit logs and source-document traceability.
- Add retry, timeout, and fallback policies for model calls.

### Phase 3: Production Deployment

- Add tenant isolation and retention policies.
- Move secrets to cloud-native secret stores.
- Add CI/CD and cloud deployment modules.
- Add compliance review workflows.
- Add prompt-injection defenses and answer evaluation.
- Add observability with traces, metrics, dashboards, and alerts.

## Design Notes

The current code intentionally favors clarity over production completeness. The main architecture is useful: agents are separated by lifecycle responsibility, orchestration is explicit in LangGraph, and storage concerns are isolated behind services. The next major engineering step is not adding more agents. It is making ingestion reliable, retrieval real, and operations observable.
