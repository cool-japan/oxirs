# oxirs-graphrag

**GraphRAG: Hybrid Vector + Graph Retrieval-Augmented Generation for OxiRS**

[![Crates.io](https://img.shields.io/crates/v/oxirs-graphrag.svg)](https://crates.io/crates/oxirs-graphrag)
[![docs.rs](https://docs.rs/oxirs-graphrag/badge.svg)](https://docs.rs/oxirs-graphrag)

Microsoft-style GraphRAG implementation combining vector similarity search with knowledge graph topology for enhanced retrieval-augmented generation.

## Features

- **RRF (Reciprocal Rank Fusion)**: Combines vector and keyword search results
- **N-hop Graph Expansion**: SPARQL-based graph traversal for context retrieval
- **Community Detection**: Louvain algorithm for hierarchical clustering
- **LLM Context Building**: Converts graph structures to natural language
- **SPARQL Extensions**: Custom functions for hybrid queries

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

```rust
use oxirs_graphrag::{GraphRAGEngine, GraphRAGConfig};
use std::sync::Arc;

let config = GraphRAGConfig {
    top_k: 20,
    expansion_hops: 2,
    enable_communities: true,
    ..Default::default()
};

let engine = GraphRAGEngine::new(
    Arc::new(vec_index),
    Arc::new(embedding_model),
    Arc::new(sparql_engine),
    Arc::new(llm_client),
    config,
);

let result = engine.query("What are quantum computing applications?").await?;
println!("Answer: {}", result.generated_text);
```

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

## Integration with OxiRS

Requires:
- `oxirs-vec` - Vector index (HNSW)
- `oxirs-embed` - Embedding models (TransE, GNN, Transformers)
- `oxirs-chat` - LLM client integration
- `oxirs-arq` - SPARQL query engine

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
