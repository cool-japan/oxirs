# OxiRS Core - End-to-End Tutorial

Welcome to the OxiRS Core tutorial! This guide will walk you through using OxiRS Core for RDF/SPARQL processing, from basic operations to advanced features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic RDF Operations](#basic-rdf-operations)
3. [SPARQL Queries](#sparql-queries)
4. [Transactions and ACID](#transactions-and-acid)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [AI/ML Integration](#aiml-integration)

## Getting Started

### Installation

Add OxiRS Core to your `Cargo.toml`:

```toml
[dependencies]
oxirs-core = "0.1.0-beta.2"
```

### Basic Setup

```rust
use oxirs_core::{RdfStore, NamedNode, Literal, Triple};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an in-memory RDF store
    let store = RdfStore::new();

    println!("OxiRS Core initialized!");
    Ok(())
}
```

## Basic RDF Operations

### Creating RDF Terms

```rust
use oxirs_core::model::{NamedNode, Literal, BlankNode, Term};

// Create IRIs (Named Nodes)
let person = NamedNode::new("http://example.org/person/alice")?;
let name_property = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;

// Create Literals
let name_value = Literal::new("Alice");
let age_value = Literal::new_typed(
    "30",
    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?
);

// Create Blank Nodes
let blank = BlankNode::default();
```

### Adding Triples

```rust
use oxirs_core::{RdfStore, Triple, Quad};

let store = RdfStore::new();

// Define subjects, predicates, and objects
let alice = NamedNode::new("http://example.org/person/alice")?;
let foaf_name = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
let name = Literal::new("Alice");

// Create and insert a triple
let triple = Triple::new(
    Term::NamedNode(alice.clone()),
    Term::NamedNode(foaf_name),
    Term::Literal(name)
);

store.insert_triple(&triple)?;
```

### Querying Triples

```rust
// Query all triples with a specific subject
let triples = store.triples_for_subject(&Term::NamedNode(alice))?;

for triple in triples {
    println!("Subject: {}", triple.subject());
    println!("Predicate: {}", triple.predicate());
    println!("Object: {}", triple.object());
}

// Pattern matching (None = wildcard)
let pattern_results = store.quads_matching(
    Some(&Term::NamedNode(alice)),  // subject
    None,                            // any predicate
    None,                            // any object
    None                             // default graph
)?;
```

## SPARQL Queries

### SELECT Queries

```rust
use oxirs_core::query::QueryExecutor;

let executor = QueryExecutor::new(store.clone());

let query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name ?age
    WHERE {
        ?person foaf:name ?name .
        ?person foaf:age ?age .
        FILTER(?age > 25)
    }
    ORDER BY DESC(?age)
    LIMIT 10
"#;

let results = executor.execute(query)?;

for solution in results.solutions()? {
    println!("Name: {:?}, Age: {:?}",
        solution.get("name"),
        solution.get("age")
    );
}
```

### CONSTRUCT Queries

```rust
let query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX ex: <http://example.org/>

    CONSTRUCT {
        ?person ex:profile ?profileNode .
        ?profileNode ex:hasName ?name .
        ?profileNode ex:hasAge ?age .
    }
    WHERE {
        ?person foaf:name ?name .
        ?person foaf:age ?age .
    }
"#;

let graph = executor.execute_construct(query)?;
println!("Constructed {} triples", graph.len());
```

### ASK Queries

```rust
let query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    ASK {
        ?person foaf:name "Alice" .
    }
"#;

let exists = executor.execute_ask(query)?;
println!("Alice exists in database: {}", exists);
```

### UPDATE Operations

```rust
let update = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX ex: <http://example.org/person/>

    INSERT DATA {
        ex:bob foaf:name "Bob" .
        ex:bob foaf:age 35 .
        ex:bob foaf:knows ex:alice .
    }
"#;

executor.execute_update(update)?;
```

## Transactions and ACID

### Basic Transactions

```rust
use oxirs_core::transaction::{Transaction, IsolationLevel};

// Start a transaction with snapshot isolation
let mut tx = store.begin_transaction(IsolationLevel::Snapshot)?;

// Perform operations within transaction
tx.insert_triple(&triple1)?;
tx.insert_triple(&triple2)?;
tx.remove_triple(&triple3)?;

// Commit or rollback
tx.commit()?;  // Makes changes permanent
// or
// tx.rollback()?;  // Discards all changes
```

### Isolation Levels

```rust
// Read Uncommitted - Fastest, may see uncommitted changes
let tx = store.begin_transaction(IsolationLevel::ReadUncommitted)?;

// Read Committed - See only committed data
let tx = store.begin_transaction(IsolationLevel::ReadCommitted)?;

// Repeatable Read - Consistent snapshot within transaction
let tx = store.begin_transaction(IsolationLevel::RepeatableRead)?;

// Snapshot Isolation - Full MVCC snapshot (recommended)
let tx = store.begin_transaction(IsolationLevel::Snapshot)?;

// Serializable - Strongest isolation, may have conflicts
let tx = store.begin_transaction(IsolationLevel::Serializable)?;
```

### Write-Ahead Logging (WAL)

```rust
use oxirs_core::transaction::WalConfig;

// Configure WAL for crash recovery
let config = WalConfig {
    enabled: true,
    fsync: true,  // Ensure durability
    checkpoint_interval: 1000,  // Checkpoint every 1000 operations
};

let store = RdfStore::with_wal_config(config)?;

// All transactions are now durable and recoverable
let mut tx = store.begin_transaction(IsolationLevel::Snapshot)?;
tx.insert_triple(&triple)?;
tx.commit()?;

// If system crashes, changes are recovered on restart
```

## Advanced Features

### RDF-star Support (Quoted Triples)

```rust
use oxirs_core::model::Triple;

// Create a quoted triple (RDF-star)
let statement = Triple::new(
    Term::NamedNode(alice.clone()),
    Term::NamedNode(knows),
    Term::NamedNode(bob.clone())
);

let quoted = Term::Triple(Box::new(statement));

// Add metadata about the statement
let certainty = NamedNode::new("http://example.org/certainty")?;
let high = Literal::new_typed("0.95", xsd_double)?;

let meta_triple = Triple::new(
    quoted,
    Term::NamedNode(certainty),
    Term::Literal(high)
);

store.insert_triple(&meta_triple)?;
```

### SPARQL 1.2 Advanced Functions

```rust
let query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX fn: <http://www.w3.org/2005/xpath-functions#>

    SELECT ?name ?upper ?len ?capitalized
    WHERE {
        ?person foaf:name ?name .
        BIND(UCASE(?name) AS ?upper)
        BIND(STRLEN(?name) AS ?len)
        BIND(CAPITALIZE(?name) AS ?capitalized)
    }
"#;

// Statistical aggregates
let stats_query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT
        (AVG(?age) AS ?avgAge)
        (MEDIAN(?age) AS ?medianAge)
        (STDDEV(?age) AS ?stddevAge)
        (PERCENTILE(?age, 95) AS ?p95Age)
    WHERE {
        ?person foaf:age ?age .
    }
"#;
```

### Query Profiling

```rust
use oxirs_core::profiling::QueryProfiler;

let profiler = QueryProfiler::new();

// Enable profiling for queries
profiler.enable_profiling(true);
profiler.set_slow_query_threshold_ms(100);

// Execute query with profiling
let results = executor.execute_with_profiling(query, &profiler)?;

// Get profiling statistics
let stats = profiler.get_statistics()?;
println!("Execution time: {}ms", stats.execution_time_ms);
println!("Triples matched: {}", stats.triples_matched);
println!("Index used: {:?}", stats.index_used);

// Get optimization hints
for hint in stats.optimization_hints {
    println!("💡 {}", hint);
}
```

### Query Result Caching

```rust
use oxirs_core::cache::QueryResultCache;
use std::time::Duration;

let cache = QueryResultCache::new()
    .with_max_entries(10000)
    .with_ttl(Duration::from_secs(300))  // 5 minutes
    .build();

// Cache is automatically used for repeated queries
let results1 = executor.execute_with_cache(query, &cache)?;
let results2 = executor.execute_with_cache(query, &cache)?;  // Cache hit!

// Check cache statistics
let stats = cache.get_stats();
println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
```

## Performance Optimization

### Zero-Copy Operations

```rust
use oxirs_core::parser::zero_copy::ZeroCopyParser;

// Parse RDF without copying strings
let parser = ZeroCopyParser::new();
let triples = parser.parse_ntriples_zero_copy(rdf_data)?;

// Triples reference original data - no allocations
for triple in triples {
    // Process without copying
}
```

### SIMD Pattern Matching

```rust
use oxirs_core::simd::SimdPatternMatcher;

// Automatically uses AVX2/AVX-512/NEON when available
let matcher = SimdPatternMatcher::new();

// 3-8x faster pattern matching on large datasets
let results = matcher.match_pattern(
    &store,
    pattern_subject,
    pattern_predicate,
    pattern_object
)?;
```

### Parallel Batch Processing

```rust
use oxirs_core::batch::BatchProcessor;

let processor = BatchProcessor::new()
    .with_batch_size(1000)
    .with_parallelism(8);  // Use 8 threads

// Automatically parallelizes for batches > 100 items
let triples: Vec<Triple> = vec![/* ... */];
processor.insert_batch(&store, &triples)?;

// 3-8x speedup on bulk operations
```

### Memory-Mapped Storage

```rust
use oxirs_core::store::MmapStore;

// Store data on disk with memory-mapped access
let store = MmapStore::open("./data/rdf.db")?;

// Zero-copy access to disk-backed data
let triples = store.quads_matching(None, None, None, None)?;
```

## AI/ML Integration

### Knowledge Graph Embeddings

```rust
use oxirs_core::ai::embeddings::{TransE, EmbeddingConfig};

// Train TransE embeddings on your knowledge graph
let config = EmbeddingConfig {
    embedding_dim: 128,
    learning_rate: 0.01,
    margin: 1.0,
    num_epochs: 100,
    batch_size: 1024,
};

let mut model = TransE::new(config);
model.train(&store)?;

// Use embeddings for link prediction
let score = model.predict_link(&alice, &knows, &bob)?;
println!("Link probability: {:.2}%", score * 100.0);

// Save trained model
model.save("embeddings.json")?;
```

### Vector Similarity Search

```rust
use oxirs_core::ai::vector_store::VectorStore;

let vector_store = VectorStore::new(128);  // 128-dimensional vectors

// Add entity embeddings
vector_store.insert(alice_id, alice_embedding)?;
vector_store.insert(bob_id, bob_embedding)?;

// Find similar entities
let similar = vector_store.search(
    &query_embedding,
    k: 10,  // Top 10 results
    threshold: 0.8  // Minimum similarity
)?;

for (entity_id, similarity) in similar {
    println!("Entity: {}, Similarity: {:.2}", entity_id, similarity);
}
```

### Graph Neural Networks

```rust
use oxirs_core::ai::gnn::{GNN, GNNConfig};

let config = GNNConfig {
    hidden_dim: 64,
    num_layers: 3,
    aggregation: "mean",
};

let gnn = GNN::new(config);

// Generate node embeddings using graph structure
let node_embeddings = gnn.forward(&store, &node_features)?;
```

## Next Steps

- Read the [Architecture Deep Dive](ARCHITECTURE.md) for implementation details
- Check the [Performance Guide](PERFORMANCE_GUIDE.md) for optimization strategies
- Explore the [Best Practices Guide](BEST_PRACTICES.md) for production deployment
- See [examples/](examples/) for more comprehensive examples

## Common Patterns

### Loading RDF from Files

```rust
use oxirs_core::parser::{RdfFormat, parse_file};

// Auto-detect format from extension
let triples = parse_file("data.ttl", RdfFormat::Turtle)?;
store.insert_batch(&triples)?;

// Or explicitly specify format
let triples = parse_file("data.nt", RdfFormat::NTriples)?;
```

### Exporting RDF

```rust
use oxirs_core::serializer::{RdfSerializer, RdfFormat};

let serializer = RdfSerializer::new(RdfFormat::Turtle);
let output = serializer.serialize(&store)?;
std::fs::write("output.ttl", output)?;
```

### Federated Queries

```rust
let federated_query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name ?age
    WHERE {
        ?person foaf:name ?name .
        SERVICE <http://remote-endpoint.org/sparql> {
            ?person foaf:age ?age .
        }
    }
"#;

let results = executor.execute_federated(federated_query)?;
```

## Troubleshooting

### Out of Memory

```rust
// Use memory-mapped storage for large datasets
let store = MmapStore::open("data.db")?;

// Or configure memory limits
let store = RdfStore::new()
    .with_memory_limit(1_000_000_000)  // 1GB limit
    .build()?;
```

### Slow Queries

```rust
// Enable query profiling to identify bottlenecks
let profiler = QueryProfiler::new();
profiler.enable_profiling(true);

let results = executor.execute_with_profiling(query, &profiler)?;
let stats = profiler.get_statistics()?;

// Check optimization hints
for hint in stats.optimization_hints {
    println!("{}", hint);
}
```

### Transaction Conflicts

```rust
// Use optimistic locking with retry logic
let max_retries = 3;
for attempt in 0..max_retries {
    let mut tx = store.begin_transaction(IsolationLevel::Snapshot)?;

    match tx.insert_triple(&triple) {
        Ok(_) => {
            tx.commit()?;
            break;
        }
        Err(e) if e.is_conflict() => {
            // Retry
            continue;
        }
        Err(e) => return Err(e),
    }
}
```

---

**Happy RDF Processing with OxiRS Core!** 🚀

For more information, visit the [GitHub repository](https://github.com/cool-japan/oxirs) or join our community discussions.
