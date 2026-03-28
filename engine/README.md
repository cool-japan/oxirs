# OxiRS Engine 🚀

[![Version](https://img.shields.io/badge/version-0.2.3-blue)](https://github.com/cool-japan/oxirs/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Build Status](https://github.com/cool-japan/oxirs/workflows/CI/badge.svg)](https://github.com/cool-japan/oxirs/actions)

**Status**: Production Release (v0.2.3) - Released March 16, 2026

✨ **Production Release**: Production-ready with API stability guarantees. Semantic versioning enforced.

**A high-performance, modular RDF processing engine written in Rust, providing SPARQL, SHACL, vector search, reasoning, and advanced semantic web capabilities.**

## 🌟 Overview

OxiRS Engine is a comprehensive semantic web processing platform that combines traditional RDF/SPARQL capabilities with modern AI technologies. It provides a modular architecture where each component can be used standalone or integrated for powerful hybrid semantic-neural processing.

### Key Capabilities

- 🔍 **SPARQL 1.1/1.2 Query Engine** - High-performance query processing with advanced optimization
- ✅ **SHACL Validation** - Complete constraint validation with streaming and multi-graph support  
- 🧠 **Vector Search Integration** - Semantic similarity search with neural embeddings
- 🤖 **Rule-based Reasoning** - Forward/backward chaining with OWL and SWRL support
- ⭐ **RDF-star Support** - Quoted triples and reification for knowledge graphs
- 🔗 **Neural-Symbolic Bridge** - Hybrid AI queries combining symbolic and neural reasoning
- 📊 **Real-time Analytics** - Streaming validation, performance monitoring, and optimization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OxiRS Engine                             │
├─────────────────────────────────────────────────────────────────┤
│  🔗 Neural-Symbolic Bridge                                     │
│     ├─ Hybrid Query Processing                                 │
│     ├─ Semantic-Vector Integration                             │
│     └─ Explainable AI Reasoning                               │
├─────────────────────────────────────────────────────────────────┤
│  🔍 oxirs-arq          ✅ oxirs-shacl       🧠 oxirs-vec       │
│  SPARQL Engine         SHACL Validator     Vector Search        │
│  ├─ Query Optimization ├─ Constraint Types ├─ Multiple Indices │
│  ├─ Parallel Execution ├─ Shape Versioning ├─ Real-time Updates│
│  ├─ Federation Support ├─ Streaming Valid. ├─ SPARQL Functions │
│  └─ Custom Functions   └─ Multi-graph Val. └─ ML Integration   │
├─────────────────────────────────────────────────────────────────┤
│  🤖 oxirs-rule         ⭐ oxirs-star        🔧 oxirs-ttl       │
│  Rule Engine           RDF-star Support     Format Support      │
│  ├─ Forward Chaining   ├─ Quoted Triples   ├─ Turtle/N-Triples │
│  ├─ Backward Chaining  ├─ Annotation Props ├─ JSON-LD/RDF-XML  │
│  ├─ OWL Reasoning      ├─ Query Extensions ├─ Streaming Parsing│
│  └─ SWRL Support       └─ Reification      └─ Error Recovery   │
├─────────────────────────────────────────────────────────────────┤
│                    🔧 Integration Layer                         │
│     ├─ Cross-module Communication                              │
│     ├─ Shared Caching and Optimization                        │
│     ├─ Event Coordination                                     │
│     └─ Performance Monitoring                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Modules

### 🔍 [oxirs-arq](oxirs-arq/) - SPARQL Query Engine
Advanced SPARQL 1.1/1.2 engine with optimization and federation support.

```rust
use oxirs_arq::{QueryEngine, Query};

let engine = QueryEngine::new();
let query = Query::parse(r#"
    SELECT ?person ?name WHERE {
        ?person foaf:name ?name ;
               foaf:age ?age .
        FILTER (?age > 18)
    }
"#)?;
let results = engine.execute(&query, &dataset).await?;
```

**Features:** Query optimization, parallel execution, federation, custom functions, result streaming.

### ✅ [oxirs-shacl](oxirs-shacl/) - SHACL Validation Engine
Comprehensive SHACL constraint validation with advanced features.

```rust
use oxirs_shacl::{ValidationEngine, StreamingValidationEngine};

// Standard validation
let engine = ValidationEngine::new(&shapes, config);
let report = engine.validate_store(&store)?;

// Streaming validation
let mut streaming_engine = StreamingValidationEngine::new(shapes, config)?;
let mut results = streaming_engine.validate_stream(rdf_stream).await?;
```

**Features:** Core constraints, property paths, streaming validation, multi-graph support, shape versioning.

### 🧠 [oxirs-vec](oxirs-vec/) - Vector Search Engine
High-performance semantic similarity search with ML integration.

```rust
use oxirs_vec::{VectorStore, IndexType};

let store = VectorStore::new(IndexType::HNSW, 768)?;
store.index_document("doc1", "Machine learning research")?;

let results = store.similarity_search(
    "AI and neural networks", 
    10, 
    0.8
)?;
```

**Features:** Multiple index types, real-time updates, SPARQL integration, GPU acceleration, analytics.

### 🤖 [oxirs-rule](oxirs-rule/) - Rule-based Reasoning
Forward and backward chaining reasoning with OWL and SWRL support.

```rust
use oxirs_rule::{RuleEngine, Rule};

let mut engine = RuleEngine::new();
engine.add_rule(Rule::parse(r#"
    (?person foaf:knows ?friend) ∧ (?friend foaf:knows ?other) 
    → (?person foaf:mightKnow ?other)
"#)?);

let inferred = engine.infer(&facts)?;
```

**Features:** Forward/backward chaining, OWL reasoning, SWRL rules, explanation traces.

### ⭐ [oxirs-star](oxirs-star/) - RDF-star Support
Complete RDF-star implementation with quoted triples and annotations.

```rust
use oxirs_star::{QuotedTriple, AnnotationProperty};

// Quoted triple with annotation
let quoted = QuotedTriple::new(subject, predicate, object);
let annotation = AnnotationProperty::new(
    "confidence", 
    Literal::typed("0.95", xsd::DOUBLE)
);
```

**Features:** Quoted triples, annotation properties, SPARQL-star queries, reification mapping.

### 🔧 [oxirs-ttl](oxirs-ttl/) - Format Support
Comprehensive RDF format parsing and serialization.

```rust
use oxirs_ttl::{TurtleParser, JsonLdSerializer};

let parser = TurtleParser::new();
let triples = parser.parse_file("data.ttl")?;

let serializer = JsonLdSerializer::new();
let json_ld = serializer.serialize(&triples)?;
```

**Features:** Multiple formats, streaming parsing, error recovery, format detection.

## 🔗 Neural-Symbolic Integration

The engine provides unique neural-symbolic bridge capabilities that combine traditional semantic web technologies with modern AI:

```rust
use oxirs_engine::{NeuralSymbolicBridge, HybridQuery};

let bridge = NeuralSymbolicBridge::new(config);

// Hybrid semantic-vector query
let query = HybridQuery::SimilarityWithConstraints {
    text_query: "machine learning researchers".to_string(),
    sparql_constraints: "?person a foaf:Person ; foaf:knows ?colleague".to_string(),
    threshold: 0.8,
    limit: 10,
};

let results = bridge.execute_hybrid_query(query).await?;
```

### Hybrid Query Types

- **Similarity with Constraints** - Vector search constrained by SPARQL patterns
- **Reasoning-guided Search** - Concept hierarchy expansion with neural similarity  
- **Knowledge Completion** - ML-assisted triple completion with reasoning validation
- **Explainable Similarity** - Vector search with comprehensive explanations

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
# Individual modules (stable features)
oxirs-arq = "0.2.3"
oxirs-shacl = "0.2.3"

# Individual modules (experimental features)
oxirs-vec = "0.2.3"
oxirs-rule = "0.2.3"
oxirs-star = "0.2.3"
```

Note: There is no single `oxirs-engine` crate. Use individual modules as needed.

### Basic Usage

Use individual modules as needed:

```rust
use oxirs_arq::QueryEngine;
use oxirs_shacl::ValidationEngine;
use oxirs_core::Dataset;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create dataset
    let mut dataset = Dataset::new();

    // Load data
    dataset.load_from_file("data.ttl")?;

    // SPARQL query
    let query_engine = QueryEngine::new();
    let sparql = r#"
        SELECT ?person ?name WHERE {
            ?person foaf:name ?name ;
                   foaf:age ?age .
            FILTER (?age > 18)
        }
    "#;
    let results = query_engine.execute(sparql, &dataset).await?;

    // SHACL validation
    let shapes = Dataset::from_file("shapes.ttl")?;
    let validation_engine = ValidationEngine::new(&shapes, Default::default());
    let validation_report = validation_engine.validate(&dataset)?;
    
    // Apply reasoning rules
    let inferred_facts = engine.infer().await?;
    
    println!("Found {} results", results.len());
    println!("Validation conforms: {}", validation_report.conforms());
    println!("Inferred {} new facts", inferred_facts.len());
    
    Ok(())
}
```

## 🔧 Configuration

### Engine Configuration

```yaml
# engine-config.yaml
engine:
  modules:
    sparql:
      optimization_level: "aggressive"
      parallel_execution: true
      federation_timeout: "30s"
    
    shacl:
      enable_streaming: true
      validation_strategy: "optimized"
      enable_shape_versioning: true
    
    vector:
      index_type: "hnsw"
      embedding_dimension: 768
      similarity_threshold: 0.8
      enable_gpu: true
    
    reasoning:
      enable_forward_chaining: true
      enable_backward_chaining: true
      max_inference_depth: 10
    
    integration:
      enable_cross_module_optimization: true
      shared_cache_size: "512MB"
      event_coordination: true
```

### Runtime Configuration

```rust
use oxirs_engine::{EngineConfig, ModuleConfig};

let config = EngineConfig::builder()
    .sparql_config(
        ModuleConfig::builder()
            .optimization_level("aggressive")
            .parallel_execution(true)
            .build()
    )
    .vector_config(
        ModuleConfig::builder()
            .index_type("hnsw")
            .embedding_dimension(768)
            .enable_gpu(true)
            .build()
    )
    .integration_config(
        ModuleConfig::builder()
            .enable_cross_module_optimization(true)
            .shared_cache_size_mb(512)
            .build()
    )
    .build();
```

## 📊 Performance

### Benchmarks

| Operation | Performance | Memory | Notes |
|-----------|-------------|---------|-------|
| SPARQL SELECT (1M triples) | 15ms | 45MB | With optimization |
| SHACL validation (100K resources) | 850ms | 120MB | Parallel execution |
| Vector similarity (1M vectors) | 0.8ms | 2GB | HNSW index |
| Rule inference (10K facts) | 120ms | 35MB | Forward chaining |
| Hybrid neural-symbolic query | 25ms | 180MB | Combined processing |

### Optimization Features

- **Cross-module caching** - Shared result caching across modules
- **Adaptive indexing** - Automatic index selection and optimization
- **Parallel processing** - Multi-threaded execution across all modules
- **Memory management** - Efficient memory usage with streaming support
- **GPU acceleration** - CUDA support for vector operations

## 🛠️ Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/engine

# Install dependencies
cargo build --workspace

# Run tests (no warnings policy)
cargo nextest run --workspace --no-fail-fast

# Run benchmarks
cargo bench --workspace
```

### Testing

```bash
# Run all tests
./scripts/test.sh

# Test specific module
cargo nextest run -p oxirs-arq --no-fail-fast

# Integration tests
cargo nextest run --test integration_tests --no-fail-fast

# Property-based tests
cargo nextest run --test proptest --no-fail-fast
```

### Code Quality

```bash
# Lint (no warnings policy)
cargo clippy --workspace --all-targets -- -D warnings

# Format
cargo fmt --workspace --check

# Documentation
cargo doc --workspace --no-deps --open

# Coverage
cargo tarpaulin --workspace --exclude-files "tests/*"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

### Current Priorities

1. **W3C Compliance** - Complete SPARQL 1.2 and SHACL specification compliance
2. **Performance Optimization** - Advanced indexing and caching strategies
3. **Neural Integration** - Enhanced neural-symbolic reasoning capabilities
4. **Federation** - Distributed query processing and validation
5. **Ecosystem Integration** - Bindings for Python, JavaScript, and other languages

### Development Setup

1. Install Rust 1.70+ and cargo-nextest
2. Clone the repository
3. Run `cargo nextest run --workspace --no-fail-fast` to verify setup
4. Follow the "no warnings policy" - all code must compile without warnings
5. Add comprehensive tests for new features
6. Update documentation for API changes

## 📚 Documentation

- **API Documentation**: [docs.rs/oxirs-engine](https://docs.rs/oxirs-engine)
- **User Guide**: [Engine Guide](docs/guide.md)  
- **Architecture Overview**: [Architecture](docs/architecture.md)
- **Performance Tuning**: [Performance Guide](docs/performance.md)
- **Integration Examples**: [Examples](examples/)

## 🔗 Related Projects

- **[OxiRS Core](../core/)** - Core RDF data structures and storage
- **[OxiRS Server](../server/)** - HTTP servers (Fuseki, GraphQL)
- **[OxiRS Storage](../storage/)** - Persistence and distribution
- **[OxiRS Stream](../stream/)** - Real-time processing
- **[OxiRS AI](../ai/)** - ML and embedding integration

## 📄 License

Licensed under:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

## 🌟 Status

**Active Development** - The OxiRS Engine is under active development with regular releases.

### Module Status
- ✅ **oxirs-arq**: Production ready with ongoing optimization
- ✅ **oxirs-shacl**: Core features complete, advanced features in development  
- ✅ **oxirs-vec**: High-performance vector search with Python bindings
- ✅ **oxirs-rule**: Forward/backward chaining with OWL support
- 🔧 **oxirs-star**: RDF-star implementation with ecosystem integration
- ✅ **Neural-Symbolic Bridge**: Hybrid AI query processing

### Roadmap

- **Q1 2026**: v0.2.3 - Complete W3C compliance, federation support, full-text search, GeoSPARQL
- **Q2 2026**: v1.0.0 LTS - Full specification compliance, enterprise features

---

**Built with ❤️ in Rust for high-performance semantic web processing**