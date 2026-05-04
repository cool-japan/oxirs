# oxirs-graphrag — Architecture Overview

This document describes the internal structure of `oxirs-graphrag`, the data-flow
between modules, and the design decisions behind each layer.

---

## High-level data flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    User / Application Layer                      │
│             GraphRAGEngine<V, E, S, L>  ·  engine.query(text)   │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
  ┌──────────────────┐           ┌──────────────────────┐
  │  Embedding Layer  │           │   Keyword / SPARQL   │
  │  EmbeddingModel   │           │   SparqlEngine       │
  │  (oxirs-embed)    │           │   (oxirs-arq)        │
  └────────┬─────────┘           └──────────┬───────────┘
           │                                │
           └──────────────┬─────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   Retrieval Layer    │
              │   retrieval/fusion   │  RRF / ColBERT reranking
              │   retrieval/reranker │
              └──────────┬──────────┘
                         │
                         ▼  (seed entities)
              ┌──────────────────────┐
              │   Graph Layer        │
              │   graph/traversal    │  N-hop SPARQL expansion
              │   graph/community    │  Leiden community detection
              │   graph/subgraph     │  Subgraph extraction
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Generation Layer   │
              │   generation/context │  Subgraph → context string
              │   generation/prompt  │  Prompt templates
              │   summarizer         │  Cluster-based compression
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   LLM Client         │
              │   LlmClientTrait     │  User-provided implementation
              └──────────────────────┘
```

---

## Layer descriptions

### 1. Extraction layer (`triple_extractor`, `entity_linking`, `entity_linker`)

**Purpose:** Convert raw text or structured data into RDF-like triples.

| File | Role |
|------|------|
| `src/triple_extractor.rs` | Pattern-based NLP → TextTriple extraction |
| `src/entity_linking.rs` | String mention → KG entity (candidate ranking) |
| `src/entity_linker.rs` | Alias-aware linking with confidence scoring |
| `src/relation_extractor.rs` | Relation classification from sentence spans |

Design: The extractor uses a sliding-window pattern matcher over tokenized
sentences. Each `ExtractionPattern` specifies ordered trigger words and
produces a confidence score. No neural models required.

---

### 2. Graph storage (`graph`, `temporal`, `knowledge_fusion`)

**Purpose:** In-memory and SPARQL-backed knowledge graph representation.

| File | Role |
|------|------|
| `src/graph/mod.rs` | Module re-exports |
| `src/graph/community.rs` | CommunityDetector (Leiden algorithm) |
| `src/graph/traversal.rs` | GraphTraversal — SPARQL query generation |
| `src/graph/subgraph.rs` | Subgraph extraction helpers |
| `src/graph/embeddings.rs` | CommunityAwareEmbeddings |
| `src/temporal/` | Temporal KG retrieval (time-indexed triple storage) |
| `src/knowledge_fusion.rs` | Multi-source fusion with provenance tracking |

The primary graph storage is external (oxirs-arq / SPARQL endpoint); this
layer generates the SPARQL queries needed for N-hop expansion.

---

### 3. Embedding integration (`embeddings`, `graph_embedder`, `transe_model`)

**Purpose:** Generate dense vector representations for entities and graphs.

| File | Role |
|------|------|
| `src/embeddings/node2vec.rs` | Node2Vec with alias sampling for walk generation |
| `src/graph_embedder.rs` | Structural embeddings (degree-based, topology-only) |
| `src/transe_model.rs` | TransE KG embedding (translational model) |

`GraphEmbedder` provides two modes:
- `structural_embedding`: fast, topology-based, deterministic
- `embed`: random-walk co-occurrence, stochastic but richer

---

### 4. Retrieval engine (`retrieval`, `fusion`, `path_finder`, `path_ranker`)

**Purpose:** Combine vector and keyword search; traverse paths; rank results.

| File | Role |
|------|------|
| `src/retrieval/fusion.rs` | `FusionStrategy` enum (RRF, CombMNZ, Borda) |
| `src/retrieval/reranker.rs` | Cross-encoder reranker interface |
| `src/retrieval/streaming_sparql.rs` | Streaming SPARQL cursor for large graphs |
| `src/fusion/rrf_fusion.rs` | Reciprocal Rank Fusion implementation |
| `src/fusion/colbert_reranker.rs` | ColBERT-style late-interaction reranker |
| `src/fusion/hybrid_retrieval.rs` | BM25 + dense hybrid pipeline |
| `src/path_finder.rs` | BFS/DFS shortest-path retrieval |
| `src/path_ranker.rs` | Predicate-weighted path scoring |
| `src/community_detector.rs` | Label-propagation community grouping |

The primary fusion strategy is **Reciprocal Rank Fusion (RRF)** with configurable
weights for vector vs. keyword legs (`vector_weight`, `keyword_weight`).

---

### 5. Community detection (`graph/community.rs`, `community_detector.rs`, `graph_partitioner.rs`)

**Purpose:** Cluster entities into semantically coherent groups for hierarchical retrieval.

Two implementations are provided:

| Implementation | Algorithm | Use case |
|----------------|-----------|----------|
| `graph::community::CommunityDetector` | Leiden (refinement phase) | Production: `GraphRAGEngine` |
| `community_detector::CommunityDetector` | Greedy label propagation | Lightweight: standalone pipelines |

Both implement the same conceptual interface (detect → `DetectionResult` /
`CommunitySummary`). The Leiden variant achieves higher modularity (>0.75 target)
at the cost of more iterations.

`graph_partitioner.rs` provides graph bisection, greedy, and label-propagation
partitioning for distributed / shard-aware use cases.

---

### 6. Context building and summarization (`generation`, `summarizer`, `context_builder`, `graph_summarization`)

**Purpose:** Convert the retrieved subgraph into a string that an LLM can consume.

| File | Role |
|------|------|
| `src/generation/context_builder.rs` | Build context from triples + communities |
| `src/generation/prompt_templates.rs` | Prompt templates (QA, summarization, CoT) |
| `src/summarizer.rs` | Cluster-based subgraph compression |
| `src/context_builder.rs` | N-hop extraction + ranking + truncation |
| `src/graph_summarization.rs` | PageRank-based community summary |

`SubgraphSummarizer` groups nodes by `node_type`, selects the most-connected
representative per cluster, and produces a natural-language paragraph.

---

### 7. RAG pipeline (`lib.rs` — `GraphRAGEngine`)

**Purpose:** Orchestrate all layers into a single `query()` call.

```
engine.query(text)
  ├─ embed(text)          → Vec<f32>
  ├─ search_knn(vec, k)   → Vec<(entity, score)>
  ├─ keyword_search(text) → Vec<(entity, score)>  [via SPARQL REGEX]
  ├─ fuse_results(...)    → Vec<ScoredEntity>      [RRF]
  ├─ expand_graph(seeds)  → Vec<Triple>            [SPARQL N-hop]
  ├─ detect_communities() → Vec<CommunitySummary>
  ├─ build_context(...)   → String
  └─ llm.generate(...)    → String  → GraphRAGResult2
```

The engine is generic over four traits:
- `VectorIndexTrait` — HNSW or flat-scan vector index
- `EmbeddingModelTrait` — any dense embedding model
- `SparqlEngineTrait` — any SPARQL 1.1 endpoint
- `LlmClientTrait` — any LLM HTTP client

This design lets you mix and match backends (e.g. oxirs-vec + oxirs-embed +
oxirs-arq + your own LLM adapter) without changes to the engine core.

---

## Caching (`cache`)

`QueryCache` wraps `lru::LruCache` with adaptive TTL:

- High graph update rate → short TTL (min 5 min)
- Low graph update rate → long TTL (up to 24 h)

Call `engine.record_graph_update()` after every write to trigger TTL shortening.

---

## Distributed and federation (`distributed`, `federation`)

| Module | Purpose |
|--------|---------|
| `distributed/mod.rs` | Shard-aware query routing |
| `federation/mod.rs` | Multi-endpoint federation |
| `federation/distributed.rs` | Distributed subgraph merging |

These modules are optional extensions for multi-node deployments.

---

## Key design decisions

1. **No Python or JVM dependency** — pure Rust, zero FFI in default features.
2. **Generic engine traits** — decouple retrieval backend from orchestration.
3. **Two community detection implementations** — lightweight standalone vs. production Leiden.
4. **SPARQL-first graph traversal** — N-hop expansion as a CONSTRUCT query so any
   SPARQL 1.1 endpoint works out of the box.
5. **Adaptive cache TTL** — query results cached longer when the graph is stable,
   shorter when writes are frequent.
6. **Explainability built-in** — `explainability.rs` provides attention weights, path
   explanation, and provenance without a separate tracing system.
