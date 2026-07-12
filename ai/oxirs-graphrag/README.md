# oxirs-graphrag

**GraphRAG: Hybrid Vector + Graph Retrieval-Augmented Generation for OxiRS**

[![Crates.io](https://img.shields.io/crates/v/oxirs-graphrag.svg)](https://crates.io/crates/oxirs-graphrag)
[![docs.rs](https://docs.rs/oxirs-graphrag/badge.svg)](https://docs.rs/oxirs-graphrag)
[![Tests](https://img.shields.io/badge/tests-1125%20passing-brightgreen)](https://github.com/cool-japan/oxirs)

Microsoft-style GraphRAG implementation combining vector similarity search with knowledge graph topology for enhanced retrieval-augmented generation.

## Features

- **RRF (Reciprocal Rank Fusion)**: Combines vector and keyword search results
- **N-hop Graph Expansion**: SPARQL-based graph traversal for context retrieval
- **Community Detection**: Louvain and Leiden clustering (`graph::community`), deterministic for a given `CommunityConfig::random_seed` and guaranteed never to score below the trivial single-community partition's modularity
- **LLM Context Building**: Converts graph structures to natural language (`context_builder`, `summarizer`, `graph_summarization`)
- **SPARQL Extensions**: Custom functions for hybrid queries
- **Entity Linking & Classification**: Mention detection, candidate ranking, and type classification (`entity_linker`, `entity_linking`, `entity_classifier`)
- **Explainability**: Attention-weight contributions, explanation paths, and provenance chains (`explainability`)
- **Interactive Feedback**: Session-scoped relevance feedback that reweights subsequent retrieval (`feedback`)
- **GNN Encoder**: GraphSAGE-style structural embeddings with hand-rolled backprop (`gnn_encoder`)
- **Hybrid GNN+LLM**: Frozen-GNN soft-prompt projection into LLM context, plus a joint-training scaffold (`hybrid`)
- **Neuro-Symbolic Fusion**: PINN-driven physics-plausibility scoring blended with neural similarity (`neuro_symbolic`)
- **GGUF Model Loader** *(feature `gguf-loader`)*: Pure-Rust GGUF metadata parser and model registry, plus a LoRA adapter/trainer (`model_loader`, `hybrid::lora`)

## Architecture

```
Natural Language Query
    ↓
Query Embedding (via oxirs-embed)
    ↓
[Vector KNN Search] + [Keyword BM25 Search]
    ↓
RRF Fusion → Seed Entities
    ↓
SPARQL N-hop Expansion → Subgraph (max 500 triples)
    ↓
Community Detection (Louvain) → Hierarchical Clusters
    ↓
Context Building → Natural Language + Structured Data
    ↓
LLM Generation → Answer + Citations
```

## Quick Start

The standalone pipeline requires no external services — run immediately:

```rust
use oxirs_graphrag::triple_extractor::{ExtractionConfig, TripleExtractor};
use oxirs_graphrag::community_detector::{CommunityGraph, CommunityDetector};
use oxirs_graphrag::path_finder::{KnowledgeEdge, PathFinder, PathFinderConfig};

// 1. Extract triples from text
let extractor = TripleExtractor::with_defaults(ExtractionConfig::default());
let triples = extractor.extract("Alice works at ACME. ACME is located in Berlin.");

// 2. Detect communities
let mut cg = CommunityGraph::new();
cg.add_node(1, "Alice"); cg.add_node(2, "ACME"); cg.add_node(3, "Berlin");
cg.add_edge(1, 2, 1.0); cg.add_edge(2, 3, 1.0);
let result = CommunityDetector::new(1, 50).detect(&mut cg);
println!("Communities: {}", result.communities.len());

// 3. Find paths
let finder = PathFinder::new(
    vec![KnowledgeEdge::new("Alice", "works_at", "ACME"),
         KnowledgeEdge::new("ACME",  "located_in", "Berlin")],
    PathFinderConfig::default(),
);
let paths = finder.bfs_paths("Alice", "Berlin", 3);
println!("{}", paths[0].narrative()); // Alice —[works_at]→ ACME —[located_in]→ Berlin
```

For the full async engine with vector index + SPARQL + LLM:

```rust,ignore
use oxirs_graphrag::{GraphRAGEngine, GraphRAGConfig};
use std::sync::Arc;

let config = GraphRAGConfig {
    top_k: 20, expansion_hops: 2, enable_communities: true, ..Default::default()
};
let engine = GraphRAGEngine::new(
    Arc::new(vec_index), Arc::new(embedding_model),
    Arc::new(sparql_engine), Arc::new(llm_client), config,
);
let result = engine.query("What are quantum computing applications?").await?;
println!("Answer: {}", result.answer);
```

For a step-by-step walkthrough see **[docs/tutorial.md](docs/tutorial.md)**.
For module internals see **[docs/architecture.md](docs/architecture.md)**.

## Configuration

```rust
pub struct GraphRAGConfig {
    pub top_k: usize,                    // Default: 20
    pub expansion_hops: usize,           // Default: 2
    pub max_subgraph_size: usize,        // Default: 500
    pub enable_communities: bool,        // Default: true
    pub vector_weight: f32,              // Default: 0.7
    pub keyword_weight: f32,             // Default: 0.3
}
```

## SPARQL Extensions

```sparql
PREFIX graphrag: <http://oxirs.io/graphrag#>

SELECT ?entity ?similarity WHERE {
    ?entity graphrag:similarity ("machine learning", 0.8) .
}

SELECT ?related WHERE {
    <http://example.org/entity> graphrag:expand(2) ?related .
}
```

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| `community-detection` | ✅ | Louvain/Leiden community detection (`graph::community`) |
| `hierarchical-summarization` | ✅ | Cluster-based subgraph summarization |
| `sparql-extensions` |  | Custom SPARQL functions for hybrid queries |
| `gguf-loader` |  | Pure-Rust GGUF metadata parser + `ModelRegistry` (`model_loader`) |

## Integration with OxiRS

`oxirs-graphrag` only has one compile-time OxiRS dependency (`oxirs-rule`); the
standalone pipeline (triple extraction, community detection, path finding,
summarization) runs entirely in-process with no network calls, as shown in
the Quick Start above.

For the full async `GraphRAGEngine`, you implement four small traits with
your own backends rather than pulling in additional crates directly:

- `VectorIndexTrait` — e.g. backed by `oxirs-vec` (HNSW)
- `EmbeddingModelTrait` — e.g. backed by `oxirs-embed` (TransE, GNN, Transformers)
- `SparqlEngineTrait` — e.g. backed by `oxirs-arq`
- `LlmClientTrait` — e.g. backed by `oxirs-chat`'s LLM providers

## License

Licensed under the Apache License, Version 2.0.
