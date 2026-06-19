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

For this environment, the assistant has to be conservative. It should not only generate fluent text. It needs source references, relationship-aware retrieval, retrieval-quality signals, and a deployment path that keeps provider choice replaceable.

## Solution Summary

FinSight Assistant uses a four-agent pipeline:

- `DocParserAgent` parses uploaded documents into normalized chunks.
- `KnowledgeExtractAgent` extracts entities, relations, and events from those chunks.
- `QAAgent` answers questions with vector and graph retrieval context.
- `KnowledgeUpdateAgent` handles changed documents and incremental refresh logic.

The QA layer also supports finance metapath retrieval for typed graph patterns such as sector exposure, compliance chains, supplier paths, and geography-linked evidence.

Quick links:

- [Finance Metapath Retrieval](#finance-metapath-retrieval)
- [API](#api)
- [Configuration](#configuration)

The architecture is designed around two complementary knowledge representations:

- Vector representation for semantic recall.
- Knowledge graph representation for entities, relationships, shortest paths, typed metapath traversal, and rule-based multi-hop inference.

This hybrid design is more useful for financial workflows than a pure vector-search system because many questions depend on structured relationships, not just semantic similarity.

## Architecture

```text
+------------------------------------------------+
| User Interface Layer                           |
| Web UI (SPA, 3 tabs)                           |
| REST API (FastAPI, 8 endpoints)                |
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
|                        | | MultimodalReasoning   | | (subprocess)         |
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
    multimodal.py                Query-time table and image reasoning helper
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
| retrieval_quality: 0.87                                      |
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
| Neo4j graph | Entity relationships, shortest paths, typed metapath traversal, and rule-based inference |

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

Hybrid retrieval merges vector, subgraph, shortest-path, metapath, inferred-fact, and cached community-summary contexts. Each branch is scored against the query, then fused with reciprocal rank fusion so vector and graph scores do not need to pretend they live on the same scale.

The graph layer now separates three levels of graph evidence:

- traversal evidence: nearby entities and shortest paths
- metapath evidence: typed analyst-style paths such as `Fund -> holds -> Company -> belongs_to -> Sector`
- logical inference evidence: named rules derive explicit propositions from typed multi-hop paths

For example, if the graph contains `Global Income Fund -[HOLDS]-> Microsoft` and `Microsoft -[BELONGS_TO]-> Technology`, the `fund_sector_exposure` rule derives: `Global Income Fund has inferred sector exposure to Technology.` The derived fact keeps the original path as provenance, so the answer can cite both the conclusion and the evidence chain.

This is rule-based domain inference, not an unconstrained symbolic theorem prover. It is intentionally narrow: only approved financial inference rules are applied, and every inferred fact must be backed by a concrete graph path.

Non-text inputs now have a separate multimodal reasoning path in addition to ordinary vector retrieval:

- tables are parsed into headers, rows, matched columns, matched rows, and numeric facts, then emitted as `retrieval_type="multimodal"` evidence
- image chunks can be re-opened at question time and sent to the configured vision-capable provider model when a real provider key is available
- local/offline mode falls back to reasoning over the parser's OCR and vision-description output, so CI remains deterministic

This means multimodal evidence is no longer just a fixed modality weight on text embeddings. It becomes a first-class QA context with its own reasoning mode, source metadata, and retrieval-quality contribution.

Community summaries are computed offline during graph refresh, not during question answering. By default, `COMMUNITY_SUMMARY_PROVIDER=structured` produces deterministic summaries from detected community members and relationships. Setting `COMMUNITY_SUMMARY_PROVIDER=llm` uses the configured provider-backed chat model to write richer summaries at ingestion/update time, then stores those summaries for fast query-time lookup. If `llm` is requested without a real provider key, the system falls back to structured summaries rather than pretending the offline demo model is a production summarizer.

#### Finance Metapath Retrieval

The graph retrieval layer also supports finance-domain metapaths. A metapath is a typed relationship pattern that reflects how an analyst would reason over a financial graph. For example:

```text
sector_exposure:
Fund -> holds -> Company -> belongs_to -> Sector

shared_sector:
Company -> belongs_to -> Sector <- belongs_to <- Company

compliance_chain:
Fund -> holds -> Company -> subject_to -> Regulation
```

The prototype keeps these paths explicit instead of asking a model to invent them. This is intentional:

- Financial relationship patterns are domain knowledge, not arbitrary graph walks.
- Typed paths make the retrieval result explainable: the answer can show which path supported the evidence.
- Rule-based routing is easier to audit than a learned router when the labeled retrieval set is still small.
- Learned routing or HAN-style metapath attention only makes sense after the graph ontology is stable and a labeled retrieval benchmark exists.

Implementation-wise, `MetapathRouter` is now a compatibility facade over `CandidateMetapathGenerator` and `RuleMetapathRanker`. The generator produces the validated finance metapath candidate set, while the rule ranker scores terms such as `sector`, `geographic`, `supplier`, `technology`, or `compliance` and emits a trace with matched keywords, selection reason, and fallback status. `KnowledgeGraphService.traverse_metapath()` then walks the typed path in Neo4j or the local SQLite graph fallback used when Neo4j is unavailable. Metapath contexts include structured path edges, intermediate entities, start/end entities, and router trace metadata so graph evidence can be audited. `LogicalInferenceEngine` applies approved inference rules on top of those typed paths and emits `source_type="inference"` contexts. Matching paths and inferred facts are fused with vector, subgraph, path, and community evidence through RRF. The learned-ranker/HAN insertion point is the ranker interface, not the traversal or QA API.

Example:

```text
Question:
Which sectors is Global Income Fund exposed to?

Selected metapath:
sector_exposure

Traversal:
Global Income Fund -[HOLDS]-> Microsoft
Microsoft -[BELONGS_TO]-> Technology

Retrieved evidence:
Metapath sector_exposure: Global Income Fund -[HOLDS]-> Microsoft; Microsoft -[BELONGS_TO]-> Technology.

Inferred fact:
Inference rule fund_sector_exposure ... Therefore: Global Income Fund has inferred sector exposure to Technology.
```

This is deliberately a domain-guided runtime retrieval feature. The production QA path still uses transparent rule routing, while the bench layer now contains offline learned-ranker and PyTorch HAN-style prototypes that can be evaluated before anything neural is wired into retrieval.

#### Fusion Strategy Trade-off

The default application path uses RRF instead of static source weights. This is deliberate:

- Vector similarity, graph traversal scores, path scores, and community-summary scores are not naturally calibrated to the same numeric scale.
- RRF only depends on rank position inside each retrieval branch, so it is less brittle when one branch has noisier scores, including metapath results.
- The trade-off is that RRF cannot express a learned preference such as "graph evidence is more reliable for relationship questions" unless a second-stage reranker or branch boost is added.

For that reason, the benchmark keeps a `weighted-grid` mode as an experiment, not as the production default. It reranks the API-returned vector/graph source branches with candidate branch boosts and reports whether any boost improves expected-source hit rate. This is useful for deciding whether weighted fusion is worth implementing deeper in the retrieval stack. It should not be presented as learned production weighting until it is run against a labeled retrieval set with candidate-level outputs.

The `bench/` directory includes evaluation scaffolds for expected-source checks, retrieval hit rate, and future recall@k reporting on a labeled retrieval set:

```bash
python bench/run_graphrag_eval.py --mode rrf
python bench/run_graphrag_eval.py --mode weighted-grid
python bench/run_graphrag_eval.py --mode both
python bench/run_metapath_eval.py
python bench/run_real_holdings_eval.py
python bench/export_metapath_training_data.py
python bench/export_han_data.py
python bench/han_readiness_report.py
python bench/train_metapath_ranker.py
python bench/train_han_attention.py
pip install -r bench/requirements-han.txt
python bench/train_han_torch.py --output bench/results/han_torch_eval.json --model-output bench/results/han_torch_model.pt
```

`bench/metapath_dataset.json` is a larger synthetic finance graph benchmark for metapath retrieval. It contains 27 entities, 37 typed relationships, and 16 labeled questions covering sector exposure, geographic risk, supply-chain risk, technology dependency, compliance scope, shared-sector discovery, management overlap, and subsidiary-chain traversal. The benchmark is synthetic on purpose: it gives deterministic coverage for graph behavior that is hard to guarantee from a small set of public filings.

`bench/real_holdings/` adds a second benchmark shaped like public 13F holdings data. The committed sample contains 26 manager-holding rows across Berkshire Hathaway, BlackRock, and Vanguard style portfolios, plus sector and region enrichment. It is intentionally small and offline so CI can run it deterministically, while still testing graph paths that look like real public holdings analysis:

```text
Portfolio -> holds -> Company -> belongs_to -> Sector
Portfolio -> holds -> Company -> located_in -> Region
Company -> belongs_to -> Sector <- belongs_to <- Company
```

`run_metapath_eval.py` and `run_real_holdings_eval.py` build in-memory graphs, run `MetapathRouter`, and report routed and oracle path metrics separately:

- router hit rate: whether the rule router selected the expected metapath anywhere in its candidate set
- router top-1 hit rate: whether the expected metapath was the first selected path
- average router precision and average selected metapaths: how much routing noise exists before traversal
- routed path hit rate / routed end-entity recall: whether the selected metapaths reach the expected entities
- oracle path hit rate / oracle end-entity recall: whether the graph can reach expected entities when the labeled metapath is supplied

This split avoids circular scoring. Oracle traversal validates graph data and typed traversal; routed traversal validates the end-to-end metapath retrieval path that a user query actually exercises.

`export_metapath_training_data.py` prepares the pre-HAN training format. It exports pairwise query-metapath examples to `bench/han_data/metapath_training.jsonl`: each labeled question is expanded across every configured candidate metapath, with `label=1` for the benchmark path and `label=0` for the other candidates. Each row includes the query, linked start entities, candidate path steps, router-selected status, router rank, router score, matched keywords, selection reason, and a `features` object with query token counts, path length, keyword coverage, start-entity type compatibility, and router-derived numeric features. The current export contains 62 labeled questions, 8 candidate metapaths, and 496 pairwise records. This is the data bridge for a learned metapath ranker or HAN-style attention layer; the current runtime remains rule-routed until that model is trained and evaluated.

`train_metapath_ranker.py` trains a dependency-free pairwise linear ranker on the exported features. The default split trains on `synthetic_finance_graph` and evaluates on `real_13f_style_holdings`, reporting learned top-1 hit rate and MRR against the rule router baseline. On the current small benchmark, the learned baseline reports `learned_top1_hit_rate=1.0` versus `rule_top1_hit_rate=0.667`, and `learned_mrr=1.0` versus `rule_mrr=0.833`. These numbers are a pre-HAN baseline, not a production claim; the dataset is intentionally small and should be expanded before adding neural attention.

`export_han_data.py` writes HAN-ready graph artifacts under `bench/han_data/`: stable `entities.json`, `relations.json`, `relation_types.json`, `metapaths.json`, `query_metapath_labels.jsonl`, and `adjacency_by_metapath/*.json`. The current export contains 49 entities, 97 relation rows, 9 relation types, 8 metapaths, and 62 labeled queries. This is a data-preparation boundary only; the runtime still uses the transparent rule ranker until a neural model is trained, validated, and wired in.

`han_readiness_report.py` is the decision gate before implementing HAN. It checks the training JSONL, HAN artifacts, metapath label coverage, train/eval split size, and whether the lightweight learned ranker is at least as good as the rule router. With the current defaults (`min_queries=50`, `min_eval_queries=10`), the report now returns `ready_for_han=true`: the dataset has 62 labeled queries, including 12 held-out real-holdings questions, and all eight metapaths have positive labels. This means the repo is ready for a small offline HAN prototype with held-out evaluation; it is still not a production neural ranking claim until the labeled set is expanded and the HAN model beats the transparent learned-ranker baseline.

`train_han_attention.py` is that small offline prototype. It is dependency-free and HAN-style rather than a full neural HAN: it combines the pairwise query/metapath features with graph path-instance features from `adjacency_by_metapath/*.json`, then trains a pairwise attention scorer offline. On the current held-out real-holdings split it reports `attention_top1_hit_rate=1.0` versus `rule_top1_hit_rate=0.667`, and `attention_mrr=1.0` versus `rule_mrr=0.833`. The output also surfaces graph-derived weights such as `han_has_reachable_path` and `han_log_path_instance_count`, which confirms the prototype is using the HAN-ready graph artifacts rather than only keyword routing.

`train_han_torch.py` is the optional PyTorch version. Install the bench-only dependency with `pip install -r bench/requirements-han.txt`, then run `python bench/train_han_torch.py`. The model learns metapath embeddings plus an attention gate over query/metapath features and graph-path features, while remaining offline and outside the API runtime. On the current held-out real-holdings split it reports `torch_top1_hit_rate=1.0` versus `rule_top1_hit_rate=0.667`, and `torch_mrr=1.0` versus `rule_mrr=0.833`; the output includes attention-gate and metapath-embedding diagnostics so the neural component is inspectable instead of a black-box claim.

This complements the public-document API benchmark. The public filings test source and evidence retrieval only; the synthetic metapath benchmark tests full graph-pattern coverage; the real-holdings benchmark tests whether those patterns work on a public-finance data shape.

For the public-document API benchmark, the metric is retrieval hit rate: expected source plus expected evidence terms must appear in returned source snippets. Generated answer text is not scored by default because the local demo model is deterministic and should not be treated as intelligence evaluation. `--include-answer-smoke` exists only as an optional formatting smoke check.

#### Optional Live Provider Evaluation

The deterministic benchmarks above are intentionally retrieval-focused. They verify that expected sources, evidence terms, metapaths, and graph paths are reachable without relying on a live model. They do not prove answer quality from a real provider.

For provider-backed answer evaluation, use `bench/run_live_eval.py` against a running API after the public demo documents have been ingested and the API is configured with a real chat model:

```bash
python bench/run_live_eval.py \
  --base-url http://localhost:8080 \
  --api-key replace-with-a-random-local-secret \
  --output bench/live_eval/results.local.json
```

`bench/live_eval/questions.json` is scoped to provider-backed answer grounding. It contains public filing questions, table-grounded questions, and insufficient-evidence cases with expected source files, expected evidence terms, and expected answer points. It intentionally excludes graph-inference/metapath questions; those are measured by `run_metapath_eval.py` and `run_real_holdings_eval.py`, where path reachability and end-entity recall are the primary metrics.

The live runner reports:

- pass rate
- source hit rate
- evidence hit rate
- answer-point hit rate
- insufficient-evidence hit rate
- per-answer-type breakdown

If the API is already configured with provider credentials but the local shell does not expose `OPENAI_API_KEY`, pass `--allow-demo` to skip the local environment guard. Do not report deterministic demo-model output as live provider evaluation; it is useful for wiring checks only. Reported answer-quality numbers should come from a real provider such as AWS Bedrock, Azure OpenAI, or Databricks-hosted OpenAI-compatible endpoints.
#### Optional Live LLM Smoke Tests

Most automated tests use the deterministic demo model so CI can run without provider credentials. That proves orchestration, parsing, retrieval, graph traversal, and API behavior, but it does not prove a real LLM follows the prompts.

For that reason, the repo includes optional provider-backed smoke tests marked `live_llm`. They validate three narrow behaviors against the configured chat model:

- intent classification returns the expected label for a financial factoid question
- knowledge extraction returns parseable JSON with finance entities and relations
- grounded answering uses supplied context and cites the source name

Run them only when a real provider key is configured:

```bash
RUN_LIVE_LLM_TESTS=1 OPENAI_API_KEY=... pytest -m live_llm backend/tests/test_live_llm_smoke.py
```

These are smoke tests, not a full evaluation set. They are intentionally separate from the default test suite because live models add cost, latency, and occasional nondeterminism.

#### Optional Live Infrastructure Smoke Tests

The default tests use in-memory graph storage and deterministic embedding fallbacks where appropriate, so they can run without Docker. The API runtime enables a local SQLite graph fallback when Neo4j is unavailable, so fallback graph data survives process restarts during local demos. To verify that the real infrastructure paths work, the repo includes optional `live_infra` tests for:

- Neo4j entity and relationship writes/reads
- ChromaDB HTTP add/search
- PGVector add/search

Start the infrastructure first:

```bash
docker compose up -d neo4j chromadb pgvector
```

Then run:

```bash
RUN_LIVE_INFRA_TESTS=1 pytest -m live_infra backend/tests/test_live_infra_smoke.py
```

These tests intentionally fail when the services are not reachable. That makes them useful for validating the real Docker-backed path instead of silently passing through memory fallbacks.

The default public-demo questions use only publicly available annual reports and 10-K filings:

| Question | Expected source | Expected evidence |
| --- | --- | --- |
| What liquidity coverage ratio did JPMorgan Chase report for 2023? | `jpmorgan_2023_annual_report.pdf` | `Liquidity coverage ratio ... 113` |
| What did Microsoft identify as major revenue sources in fiscal 2023? | `microsoft_2023_10k.pdf` | `Revenue increased`, `Intelligent Cloud`, `Productivity and Business Processes` |
| What were Microsoft's reported revenue segments in fiscal 2023? | `microsoft_2023_10k.pdf` | `Productivity and Business Processes`, `Intelligent Cloud`, `More Personal Computing` |
| How much revenue did Apple recognize in 2023 that was included in deferred revenue as of September 24, 2022? | `apple_2023_10k.pdf` | `$8.2 billion`, `deferred revenue` |

The response includes:

- answer
- retrieval-quality signal
- intent
- source snippets
- reasoning steps

Operational hardening in the public prototype:

- Batch uploads are processed with bounded concurrency (`BATCH_UPLOAD_CONCURRENCY`) and return per-file success/failure results instead of failing the entire batch on one bad upload.
- Retrieval, parsing, graph, and storage fallbacks emit structured warning logs instead of silently swallowing exceptions; the API graph fallback is SQLite-backed rather than process-memory-only.
- Local embedding worker failures fall back to deterministic hash embeddings with a warning, rather than writing all-zero vectors.
- Rate-limit buckets and request metrics default to in-memory state for local demos, with PostgreSQL-backed persistence via `API_STATE_BACKEND=postgres` and `API_STATE_DSN`; API state degrades to memory if the persistent store is unavailable.
- The QA LangGraph includes conditional routing: low-quality answers are retried once with stronger evidence focus, then converted into an insufficient-evidence response instead of pretending to answer.

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
- SQLite-backed failed-ingestion registry with dead-letter visibility in `/api/admin/stats`, plus operator endpoints to list, retry, and clear failed ingestion records

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
| `POST` | `/api/admin/cdc/events` | Process a normalized CDC event |
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
| `MODEL_PROVIDER` | `openai_compatible` or `bedrock` |
| `OPENAI_API_KEY` | Provider API key for OpenAI-compatible providers |
| `OPENAI_BASE_URL` | OpenAI-compatible provider endpoint |
| `OPENAI_MODEL` | Chat model or deployment name for OpenAI-compatible providers |
| `AWS_REGION` | AWS region for native Bedrock Runtime |
| `AWS_PROFILE` | Optional named AWS profile for local Bedrock credentials |
| `BEDROCK_MODEL_ID` | Native Bedrock model ID when `MODEL_PROVIDER=bedrock` |
| `BEDROCK_MAX_TOKENS` | Bedrock response token budget |
| `MODEL_CALL_TIMEOUT_SECONDS` | Per-call provider timeout |
| `MODEL_CALL_MAX_RETRIES` | Provider retry count before fallback/failure |
| `MODEL_CALL_FALLBACK_TO_DEMO` | Allow provider failures to fall back to deterministic demo model; use `false` in production |
| `EMBEDDING_MODEL` | Embedding model name |
| `EMBEDDING_PROVIDER` | `auto`, `openai`, `local`, or `hash`; `auto` uses provider embeddings when configured and demo-safe hash embeddings otherwise |
| `COMMUNITY_SUMMARY_PROVIDER` | `structured` for deterministic offline summaries or `llm` for provider-backed summaries computed during graph refresh |
| `NEO4J_URI` | Neo4j Bolt URI |
| `NEO4J_USER` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `VECTOR_STORE_TYPE` | `chroma` or `pgvector` |
| `CHROMA_MODE` | `local` for embedded persistence or `http` for Docker-hosted ChromaDB |
| `CHROMA_HOST` | ChromaDB host when `CHROMA_MODE=http` |
| `CHROMA_PORT` | ChromaDB port when `CHROMA_MODE=http` |
| `CHROMA_LEXICAL_SCAN_LIMIT` | Maximum documents scanned by the Chroma lexical fallback in local/demo mode |
| `PGVECTOR_DSN` | PGVector connection string |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap server |
| `AUTH_ENABLED` | Enable API-key authentication for protected endpoints |
| `API_KEY` | Shared API key used when `AUTH_ENABLED=true` |
| `MAX_UPLOAD_SIZE_MB` | Per-file upload size limit |
| `BATCH_UPLOAD_CONCURRENCY` | Concurrency limit for batch uploads |
| `PDF_VISION_CONCURRENCY` | Concurrency limit for scanned-PDF vision fallback pages |
| `API_STATE_BACKEND` | `memory` or `postgres` for rate-limit and request metrics |
| `API_STATE_DSN` | PostgreSQL connection string for API state when enabled |
| `UPLOAD_DIR` | File upload directory |

The intended model-provider targets are:

- Azure OpenAI
- AWS Bedrock through the native Bedrock Runtime adapter
- Databricks Mosaic AI Model Serving

Native Bedrock chat is enabled with:

```env
MODEL_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_PROFILE=optional-local-profile
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
BEDROCK_MAX_TOKENS=2048
EMBEDDING_PROVIDER=local
MODEL_CALL_TIMEOUT_SECONDS=45
MODEL_CALL_MAX_RETRIES=2
MODEL_CALL_FALLBACK_TO_DEMO=false
```

The Bedrock adapter uses the AWS SDK credential chain, so credentials can come from `aws configure`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, SSO, or an IAM role in deployed environments. The current adapter covers chat and vision-capable Converse requests; embeddings should remain `local`, `hash`, or an OpenAI-compatible embedding provider until a Bedrock embedding adapter is added.
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
AUTH_ENABLED=true
API_KEY=replace-with-a-random-local-secret
MAX_UPLOAD_SIZE_MB=10
BATCH_UPLOAD_CONCURRENCY=4
```

Protected endpoints require the `X-API-Key` header when `AUTH_ENABLED=true`. For a throwaway localhost-only demo, you can explicitly set `AUTH_ENABLED=false`, but the application default is protected.

### Run With AWS Bedrock

For native Bedrock chat, configure AWS credentials with `aws configure`, SSO, environment variables, or an IAM role, then set:

```env
MODEL_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_PROFILE=optional-local-profile
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
BEDROCK_MAX_TOKENS=2048
EMBEDDING_PROVIDER=local
MODEL_CALL_TIMEOUT_SECONDS=45
MODEL_CALL_MAX_RETRIES=2
MODEL_CALL_FALLBACK_TO_DEMO=false
AUTH_ENABLED=true
API_KEY=replace-with-a-random-local-secret
```

Before running live tests, check the local setup without calling Bedrock:

```bash
python scripts/check_bedrock_config.py --env-file backend/.env
```

For a production-shaped template, see `backend/.env.production.example`.

When running with Docker Compose, the API service overrides local hostnames with container-network service names and waits for core dependencies to become healthy before startup:

```env
NEO4J_URI=bolt://neo4j:7687
CHROMA_MODE=http
CHROMA_HOST=chromadb
PGVECTOR_DSN=postgresql://postgres:postgres@pgvector:5432/knowledge
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
API_PORT=8080
```

Docker Compose includes health checks for Neo4j, PGVector, Kafka, and the API container. ChromaDB is started as a dependency and the backend still performs service-level initialization retries, so temporary startup ordering issues degrade cleanly.

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

## CI/CD

GitHub Actions runs the same deterministic quality gate used locally on every push and pull request:

```bash
ruff check backend bench scripts
mypy --config-file pyproject.toml backend
pytest -q
```

The CI job disables local embedding subprocess startup and uses hash embeddings so it stays deterministic and does not require provider credentials. Optional `live_llm` and `live_infra` tests remain manual validation steps.
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

- FastAPI backend with upload, batch upload, QA, stats, update, CDC event, and health endpoints.
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
- Chroma lexical fallback is bounded for demo safety; large corpora should use PGVector plus an indexed lexical retrieval path.
- Fine-grained incremental updates.
- OCR and vision fallback for scanned financial PDFs, including bounded per-page concurrency.
- Provider-specific deployment templates.

Known gaps:

- API-key authentication is available for protected deployments, but there is no role-based access control.
- No tenant isolation.
- No full background job queue for ingestion; failed ingestions are stored in a SQLite-backed registry and exposed as operator-manageable dead letters with list, retry, and clear endpoints.
- No malware scanning or file sandboxing for uploads.
- No full observability stack.
- No formal audit log for regulated workflows.
- Automated tests cover core paths, but broader integration coverage is still limited.
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
- Harden model-call retry, timeout, and fallback policy for provider-specific production needs.

### Phase 3: Production Deployment

- Add tenant isolation and retention policies.
- Move secrets to cloud-native secret stores.
- Add CI/CD and cloud deployment modules.
- Add compliance review workflows.
- Add prompt-injection defenses and answer evaluation.
- Add observability with traces, metrics, dashboards, and alerts.

## Design Notes

The current code intentionally favors clarity over production completeness. The main architecture is useful: agents are separated by lifecycle responsibility, orchestration is explicit in LangGraph, and storage concerns are isolated behind services. The next major engineering step is not adding more agents. It is making ingestion reliable, retrieval real, and operations observable.
