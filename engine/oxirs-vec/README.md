# OxiRS Vec - Vector Search Engine

[![Version](https://img.shields.io/badge/version-0.1.0--alpha.3-orange)](https://github.com/cool-japan/oxirs/releases)

**Status**: Alpha Release (v0.1.0-alpha.3) - Released October 12, 2025

⚠️ **Alpha Software**: This is an early alpha release. Experimental features. APIs may change without notice. Not recommended for production use.

High-performance vector search infrastructure for semantic similarity search in RDF knowledge graphs.

## Features

### Vector Indexing
- **HNSW Index** - Hierarchical Navigable Small World graphs for fast approximate nearest neighbor search
- **Flat Index** - Exact search for smaller datasets
- **IVF Index** - Inverted file index for large-scale datasets
- **Dynamic Updates** - Real-time index updates without full rebuilds

### Search Capabilities
- **Similarity Search** - Find semantically similar entities
- **Filtered Search** - Combine vector similarity with RDF constraints
- **Batch Operations** - Efficient bulk indexing and search
- **Multiple Distance Metrics** - Cosine, Euclidean, Manhattan, Dot product

### Integration
- **SPARQL Extension** - Vector search functions in SPARQL queries
- **GraphQL Support** - Vector similarity in GraphQL queries
- **Embedding Models** - Integration with various embedding providers
- **Storage Backends** - Persistent vector indices

## Installation

Add to your `Cargo.toml`:

```toml
# Experimental feature
[dependencies]
oxirs-vec = "0.1.0-alpha.3"
```

## Quick Start

### Basic Vector Search

```rust
use oxirs_vec::{VectorStore, IndexType, DistanceMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create vector store with HNSW index
    let mut store = VectorStore::builder()
        .index_type(IndexType::HNSW)
        .dimension(768)  // Embedding dimension
        .distance_metric(DistanceMetric::Cosine)
        .build()?;

    // Add vectors
    store.add_vector("entity1", &embedding1)?;
    store.add_vector("entity2", &embedding2)?;

    // Build index
    store.build_index()?;

    // Search for similar vectors
    let results = store.search(&query_vector, 10, 0.8)?;
    for result in results {
        println!("ID: {}, Score: {}", result.id, result.score);
    }

    Ok(())
}
```

### SPARQL Integration

```rust
use oxirs_vec::sparql::VectorFunctions;

let sparql = r#"
    PREFIX vec: <http://oxirs.org/vec/>

    SELECT ?entity ?score WHERE {
        ?entity a foaf:Person .

        # Vector similarity search
        ?entity vec:similarTo "machine learning researcher" .
        ?entity vec:similarity ?score .

        FILTER (?score > 0.8)
    }
    ORDER BY DESC(?score)
    LIMIT 10
"#;
```

## Architecture

### Index Types

#### HNSW (Hierarchical Navigable Small World)
- **Use Case**: General purpose, balanced performance
- **Search Time**: O(log N)
- **Build Time**: O(N log N)
- **Memory**: Moderate

#### Flat Index
- **Use Case**: Small datasets, exact search required
- **Search Time**: O(N)
- **Build Time**: O(N)
- **Memory**: Low

#### IVF (Inverted File)
- **Use Case**: Large datasets, acceptable approximate results
- **Search Time**: O(√N)
- **Build Time**: O(N)
- **Memory**: Moderate

### Distance Metrics

```rust
pub enum DistanceMetric {
    Cosine,      // For normalized embeddings
    Euclidean,   // For absolute distances
    Manhattan,   // For high-dimensional spaces
    DotProduct,  // For similarity scores
}
```

## Advanced Features

### Filtered Search

Combine vector similarity with RDF constraints:

```rust
use oxirs_vec::FilteredSearch;

let filters = FilteredSearch::builder()
    .add_constraint("rdf:type", "foaf:Person")
    .add_constraint("foaf:age", |age: i32| age > 18)
    .build();

let results = store.filtered_search(&query_vector, filters, 10)?;
```

### Batch Operations

Efficient bulk indexing:

```rust
let batch = vec![
    ("entity1", embedding1),
    ("entity2", embedding2),
    ("entity3", embedding3),
];

store.add_batch(batch)?;
store.build_index()?;
```

### Incremental Updates

```rust
// Add without full rebuild
store.add_incremental("new_entity", &embedding)?;

// Periodic optimization
store.optimize_index()?;
```

## Performance

### Benchmarks (on sample datasets)

| Dataset Size | Index Type | Build Time | Query Time (10-NN) |
|-------------|------------|------------|-------------------|
| 10K vectors | HNSW | 2.5s | 0.5ms |
| 100K vectors | HNSW | 28s | 1.2ms |
| 1M vectors | HNSW | 320s | 2.8ms |
| 10K vectors | Flat | 0.1s | 12ms |
| 100K vectors | IVF | 15s | 3.5ms |

*Benchmarked on M1 Mac with 768-dimensional vectors*

## Configuration

```rust
let config = VectorStoreConfig {
    index_type: IndexType::HNSW,
    dimension: 768,
    distance_metric: DistanceMetric::Cosine,

    // HNSW-specific parameters
    hnsw_m: 16,              // Number of connections per node
    hnsw_ef_construction: 200, // Construction time accuracy
    hnsw_ef_search: 100,      // Search time accuracy

    // Storage options
    persist_path: Some("./vector_index".into()),
    cache_size: 1000,
};
```

## Integration Examples

### With oxirs-embed

```rust
use oxirs_embed::EmbeddingModel;
use oxirs_vec::VectorStore;

// Generate embeddings
let model = EmbeddingModel::load("sentence-transformers/all-mpnet-base-v2")?;
let embedding = model.encode("Machine learning research")?;

// Index and search
let mut store = VectorStore::new(IndexType::HNSW, 768)?;
store.add_vector("doc1", &embedding)?;
```

### With oxirs-core (RDF)

```rust
use oxirs_core::Dataset;
use oxirs_vec::RdfVectorIndex;

let dataset = Dataset::from_file("knowledge_graph.ttl")?;
let mut index = RdfVectorIndex::new(&dataset)?;

// Index entities by their descriptions
for entity in dataset.subjects() {
    if let Some(description) = dataset.get_description(&entity) {
        let embedding = model.encode(&description)?;
        index.add_entity(&entity, &embedding)?;
    }
}
```

## Status

### Alpha Release (v0.1.0-alpha.3)
- ✅ HNSW/IVF/Flat indices with persisted dataset support
- ✅ SPARQL/GraphQL integration enhanced with federation-aware vector filters
- ✅ CLI pipelines for batch embedding import/export and monitoring
- ✅ SciRS2 metrics for query latency, recall, and index health
- 🚧 GPU acceleration (targeted for beta)
- 🚧 Distributed indexing (planned for v0.2.0)

## Contributing

This is an experimental module. Feedback and contributions are welcome!

## License

MIT OR Apache-2.0

## See Also

- [oxirs-embed](../oxirs-embed/) - Embedding generation
- [oxirs-arq](../oxirs-arq/) - SPARQL query engine
- [oxirs-core](../../core/oxirs-core/) - RDF data model