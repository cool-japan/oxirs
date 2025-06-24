# OxiRS Core

**Rust-native RDF data model and core functionality for the OxiRS semantic web platform**

## Overview

`oxirs-core` provides the foundational data structures and operations for working with RDF data in Rust. It implements a clean, type-safe interface for RDF terms (IRIs, blank nodes, literals), triples, quads, graphs, and datasets while maintaining compatibility with the broader Rust ecosystem.

## Features

- **RDF Data Model**: Complete implementation of RDF 1.1 abstract syntax
  - Named nodes (IRIs) with validation
  - Blank nodes with scoped identifiers
  - Literals with datatype and language tag support
  - Variables for SPARQL queries
- **Graph Structures**: Efficient in-memory graph and dataset containers
- **Format Support**: Extensible parser and serializer framework
- **Type Safety**: Leverages Rust's type system for compile-time correctness
- **Performance**: Zero-copy operations where possible
- **Compatibility**: Interoperates with oxigraph and other RDF libraries

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-core = "0.1.0"
```

## Quick Start

```rust
use oxirs_core::{NamedNode, Triple, Graph, Literal};

// Create RDF terms
let subject = NamedNode::new("http://example.org/person/alice")?;
let predicate = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
let object = Literal::new_simple_literal("Alice");

// Create a triple
let triple = Triple::new(subject, predicate, object);

// Add to a graph
let mut graph = Graph::new();
graph.insert(triple);

// Query the graph
for triple in graph.iter() {
    println!("{}", triple);
}
```

## Architecture

### Core Types

- `Term`: Unified enum for all RDF terms
- `NamedNode`: IRI references with validation
- `BlankNode`: Anonymous nodes with scoped identifiers
- `Literal`: Typed and language-tagged strings
- `Variable`: Query variables for SPARQL

### Graph Structures

- `Triple`: Subject-predicate-object statements
- `Quad`: Named graph extension with context
- `Graph`: Collection of triples with efficient iteration
- `Dataset`: Collection of named graphs

### Error Handling

All operations return `Result` types with descriptive error messages:

```rust
use oxirs_core::{OxirsError, Result};

fn parse_iri(iri: &str) -> Result<NamedNode> {
    NamedNode::new(iri).map_err(|e| OxirsError::InvalidIri(e.to_string()))
}
```

## Integration

### With oxigraph

```rust
use oxirs_core::Graph;
use oxigraph::model::Graph as OxigraphGraph;

// Convert between formats
let oxirs_graph = Graph::new();
let oxigraph_graph: OxigraphGraph = oxirs_graph.into();
```

### With SPARQL engines

```rust
use oxirs_core::{Dataset, Variable};
use oxirs_arq::Query;

let dataset = Dataset::new();
let query = Query::parse("SELECT ?s WHERE { ?s ?p ?o }")?;
let results = query.execute(&dataset)?;
```

## Performance

- **Zero-copy parsing**: Direct access to string data without allocation
- **Efficient hashing**: Custom hash implementations for RDF terms
- **Memory layout**: Optimized for cache locality in graph structures
- **Concurrent access**: Thread-safe operations where applicable

## Related Crates

- [`oxirs-arq`](../engine/oxirs-arq/): SPARQL query execution
- [`oxirs-shacl`](../engine/oxirs-shacl/): Shape validation
- [`oxirs-fuseki`](../server/oxirs-fuseki/): SPARQL HTTP server
- [`oxirs-gql`](../server/oxirs-gql/): GraphQL interface

## Development

### Running Tests

```bash
cd core/oxirs-core
cargo test
```

### Benchmarks

```bash
cargo bench
```

### Documentation

```bash
cargo doc --open
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Status

🚧 **Under Development** - This crate is part of the initial Phase 0 implementation of OxiRS.

Current focus areas:
- Core RDF data model implementation
- Parser/serializer framework
- Integration with oxigraph
- Performance optimization