# OxiRS Core

[![Version](https://img.shields.io/badge/version-0.3.2-blue)](https://github.com/cool-japan/oxirs/releases)
[![docs.rs](https://docs.rs/oxirs-core/badge.svg)](https://docs.rs/oxirs-core)

**Foundational, Rust-native RDF data model and SPARQL engine for the OxiRS semantic web platform**

**Status**: v0.3.2 - Released 2026-07-11

✨ **Production Release**: Production-ready with API stability guarantees. Semantic versioning enforced.

## Overview

`oxirs-core` provides the foundational data structures and operations for working with RDF data in Rust. It is the base crate of the OxiRS workspace: every other `oxirs-*` crate (query engines, servers, storage backends, AI tooling) depends on it, while it depends on none of them — no other OxiRS crate sits underneath it. Originally extracted from OxiGraph's RDF implementation, it has since grown a large surface of its own — SPARQL algebra/execution, persistent storage, federation, indexing, and enterprise compliance tooling — while retaining an OxiGraph-compatible core (`oxigraph_compat`) for easy migration.

## Features

### Core RDF Data Model
- **Named nodes (IRIs)**: `NamedNode`, RFC 3987-oriented IRI validation
- **Blank nodes**: `BlankNode`, scoped identifiers
- **Literals**: `Literal`, typed/language-tagged strings with XSD datatype support
- **Variables**: `Variable`, SPARQL query variables
- **Triples/Quads**: `Triple` / `Quad`, full RDF 1.2 model with named-graph support
- **RDF-star**: quoted triples / statement reification (`rdf-star` feature, on by default)

### SPARQL Foundation
- **SPARQL 1.1/1.2 query**: SELECT, CONSTRUCT, ASK, DESCRIBE (`query`, `sparql` modules — parser, cost-based planner, streaming executor)
- **SPARQL 1.1 update**: `INSERT`/`DELETE DATA`, `LOAD`, `CLEAR`
- **Federation**: `SERVICE`-clause execution against remote SPARQL endpoints (`federation`)
- **Built-in functions**: string/numeric/date functions including `ENCODE_FOR_URI()`, backed by the new `encoding` module (see below)

### Storage & Persistence
- **`RdfStore`**: primary store type — in-memory (`RdfStore::new`) or disk-backed (`RdfStore::open`), with automatic N-Quads-based persistence
- **Multi-index access**: indexed pattern matching (`indexing`, `storage`)
- **Transactions**: ACID transaction manager (`transaction::{AcidTransaction, TransactionManager}`)
- **Named graph management**: Jena-`Dataset`-compatible API (`named_graph`)
- **Jena Assembler**: load `ja:`-vocabulary Turtle configs, Jena `fuseki-config.ttl`-style (`assembler::from_turtle`)

### Format Support & Serialization
- Parsers and serializers for Turtle, N-Triples, N-Quads, TriG, RDF/XML, and JSON-LD (`parser`, `serializer`, `rdfxml`, `jsonld`)
- Async, chunked, progress-reporting streaming parsers — `parser::{AsyncStreamingParser, AsyncRdfSink, MemoryAsyncSink, ParseProgress}` and `io::async_streaming` (behind the `async` feature)

### Performance & Concurrency
- **Lock-free structures**: epoch-based concurrent graph access (`concurrent`)
- **Arena allocation & zero-copy references**: `optimization::RdfArena`, `TermRef`/`TripleRef`
- **SIMD-accelerated triple matching**: behind the `simd` feature (`simd_triple_matching`)
- **SciRS2 integration**: `scirs2-core` provides memory management and parallel iteration (used in place of raw `ndarray`/direct `rayon` calls)
- **Benchmark & SLA harness**: a Criterion suite under `benches/` plus `perf_sla::{SloTarget, BenchmarkResult, assert_meets_slo}` for regression-gated performance targets, checked against a committed `perf_baseline.json`

### Enterprise & Compliance
- **Audit trail**: SOC2/GDPR-oriented structured event logging — `audit::{AuditEvent, InMemoryAuditLogger, JsonLineAuditLogger, AuditFilter, GdprService}`
- **W3C RDF Dataset Canonicalization (URDNA2015)**: deterministic blank-node naming for signing and Verifiable Credentials workflows — `canon::{canonicalize, Canonicalizer}` (re-exported at the crate root)
- **PROV-O provenance tracking**: `provenance` module covering the core W3C PROV-O entities, activities, and agents
- **API stability tracking**: a programmatic public-API surface snapshot (`api_baseline.json`), diffed against the live source by `tests/api_stability.rs` so unintentional breaking changes fail the test suite
- **Production hardening**: `production::{HealthCheck, CircuitBreaker, PerformanceMonitor, ResourceQuota}`

### New in 0.3.2: the `encoding` module

A pure-`std`, in-house RFC 3986 percent-encoding implementation — `percent_encode`, `percent_encode_strict`, `percent_decode` — that replaces the external `urlencoding` crate workspace-wide. It now backs SPARQL's `ENCODE_FOR_URI()` function and federated-query URL construction (`federation::client`).

```rust
use oxirs_core::encoding::{percent_encode, percent_decode};

let encoded = percent_encode("SELECT * WHERE { ?s ?p ?o }");
assert_eq!(
    encoded,
    "SELECT%20%2A%20WHERE%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D"
);
assert_eq!(percent_decode(&encoded).unwrap(), "SELECT * WHERE { ?s ?p ?o }");
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-core = "0.3.2"

# Optional: enable async streaming support
oxirs-core = { version = "0.3.2", features = ["async"] }
```

### Feature Flags

| Feature | Default | Description |
|---|---|---|
| `serde` | ✅ | Serialization support for core types |
| `parallel` | ✅ | Multi-threaded processing via `rayon` |
| `rdf-12` | ✅ | RDF 1.2 data model features |
| `rdf-star` | ✅ | RDF-star (quoted triples) support |
| `sparql-12` | ✅ | SPARQL 1.2 query features |
| `async` | – | Async I/O and streaming parsers |
| `async-tokio` | – | Tokio-backed async streaming (implies `async`) |
| `simd` | – | SIMD-accelerated string/term operations (`wide`) |
| `cuda` / `opencl` | – | GPU backend selectors consumed by `platform`; live GPU telemetry itself lives in the separate `oxirs-gpu-monitor` adapter crate (`ai::gpu_monitor::GpuMonitor` here is a Pure-Rust "no GPU" stub with a matching API) |
| `metal` / `blas` | – | Reserved for future backend crates; not yet wired to any code path in this crate |

## Quick Start

### Creating a Store and Adding Data

```rust,no_run
use oxirs_core::RdfStore;
use oxirs_core::model::{NamedNode, Triple, Literal};

# fn example() -> Result<(), oxirs_core::OxirsError> {
let mut store = RdfStore::new()?;

let alice = NamedNode::new("http://example.org/alice")?;
let knows = NamedNode::new("http://xmlns.com/foaf/0.1/knows")?;
let bob = NamedNode::new("http://example.org/bob")?;
let name = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;

store.insert_triple(Triple::new(alice.clone(), knows, bob))?;
store.insert_triple(Triple::new(alice, name, Literal::new("Alice")))?;

println!("Store contains {} triples", store.len()?);
# Ok(())
# }
```

### Querying with Pattern Matching

```rust,no_run
use oxirs_core::RdfStore;
use oxirs_core::model::{NamedNode, Predicate};

# fn query_example() -> Result<(), oxirs_core::OxirsError> {
# let mut store = RdfStore::new()?;
let knows = NamedNode::new("http://xmlns.com/foaf/0.1/knows")?;
let predicate = Predicate::NamedNode(knows);

let triples = store.query_triples(None, Some(&predicate), None)?;
for triple in triples {
    println!("{:?} knows {:?}", triple.subject(), triple.object());
}
# Ok(())
# }
```

### Executing SPARQL Queries

```rust,no_run
use oxirs_core::RdfStore;

# fn sparql_example() -> Result<(), oxirs_core::OxirsError> {
# let store = RdfStore::new()?;
let query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?name WHERE {
        ?person foaf:name ?name .
    }
"#;

let results = store.query(query)?;
println!("Query returned {} results", results.len());
# Ok(())
# }
```

### Persistent Storage

```rust,no_run
use oxirs_core::RdfStore;
use oxirs_core::model::{NamedNode, Triple, Literal};

# fn persistent_example() -> Result<(), oxirs_core::OxirsError> {
// Data is saved under ./my_rdf_store and reloaded automatically on the next open
let mut store = RdfStore::open("./my_rdf_store")?;

let subject = NamedNode::new("http://example.org/resource")?;
let predicate = NamedNode::new("http://purl.org/dc/terms/title")?;
let object = Literal::new("My Resource");

store.insert_triple(Triple::new(subject, predicate, object))?;
# Ok(())
# }
```

### Named Graphs (Quads)

```rust,no_run
use oxirs_core::RdfStore;
use oxirs_core::model::{NamedNode, Quad, GraphName, Literal};

# fn quads_example() -> Result<(), oxirs_core::OxirsError> {
# let mut store = RdfStore::new()?;
let graph = GraphName::NamedNode(NamedNode::new("http://example.org/graph1")?);
let subject = NamedNode::new("http://example.org/subject")?;
let predicate = NamedNode::new("http://example.org/predicate")?;
let object = Literal::new("value");

store.insert_quad(Quad::new(subject, predicate, object, graph))?;

let graph_node = NamedNode::new("http://example.org/graph1")?;
let quads = store.graph_quads(Some(&graph_node))?;
println!("Graph contains {} quads", quads.len());
# Ok(())
# }
```

### Bulk Loading

```rust,no_run
use oxirs_core::RdfStore;
use oxirs_core::model::{NamedNode, Triple, Quad, Literal};

# fn bulk_example() -> Result<(), oxirs_core::OxirsError> {
# let mut store = RdfStore::new()?;
let mut quads = Vec::new();
for i in 0..1000 {
    let subject = NamedNode::new(format!("http://example.org/item{i}"))?;
    let predicate = NamedNode::new("http://example.org/value")?;
    let object = Literal::new(i.to_string());
    quads.push(Quad::from_triple(Triple::new(subject, predicate, object)));
}

// Bulk insert is a single call rather than 1000 individual insert_quad() calls
let ids = store.bulk_insert_quads(quads)?;
println!("Inserted {} quads", ids.len());
# Ok(())
# }
```

More runnable examples live under [`examples/`](examples/), including `indexed_graph_demo.rs`, `concurrent_graph_demo.rs`, `async_streaming.rs`, `mmap_store_example.rs`, `sparql_algebra_demo.rs`, `query_result_cache_demo.rs`, and `zero_copy_serialization.rs`.

## Architecture

### Core Type System

- **`Term`**: unified enum over all RDF terms
- **`NamedNode`** / **`BlankNode`** / **`Literal`** / **`Variable`**: the four term kinds
- **`Triple`** / **`Quad`**: subject-predicate-object(-graph) statements
- **`Subject`** / **`Predicate`** / **`Object`**: position-typed term enums used in pattern matching
- **`TermRef<'a>`** / **`TripleRef<'a>`**: zero-copy borrowed views (`optimization` module)

### Module Organization

| Module | Purpose |
|---|---|
| `model` | Core RDF data model types (IRI, literal, blank node, triple, quad) |
| `rdf_store` | `RdfStore` and the `Store` trait — pluggable storage backends |
| `parser` / `serializer` / `rdfxml` / `jsonld` | RDF parsers and serializers for all supported formats |
| `query` / `sparql` | SPARQL algebra, parser, planner, and executor |
| `storage` / `indexing` | Storage engine and multi-index structures |
| `concurrent` | Lock-free, epoch-based concurrent graph access |
| `optimization` | Arena allocation and zero-copy term references |
| `federation` | `SERVICE`-clause distributed query execution |
| `production` | Health checks, circuit breakers, performance monitoring |
| `transaction` | ACID transaction manager |
| `named_graph` | Jena-`Dataset`-compatible named graph API |
| `assembler` | Jena Assembler (`ja:` vocabulary) config loading |
| `audit` | SOC2/GDPR structured audit trail |
| `canon` | W3C RDF Dataset Canonicalization (URDNA2015) |
| `provenance` | W3C PROV-O provenance tracking |
| `encoding` | RFC 3986 percent-encoding (new in 0.3.2) |
| `oxigraph_compat` | Drop-in-compatible `Store` API matching Oxigraph |

A handful of modules (`consciousness`, `molecular`, `quantum`) are experimental research surfaces, present in the crate but explicitly outside the API stability guarantees below.

### Error Handling

All fallible operations return `oxirs_core::Result<T>` (an alias for `std::result::Result<T, OxirsError>`):

```rust
use oxirs_core::{OxirsError, Result, NamedNode};

fn parse_iri(iri: &str) -> Result<NamedNode> {
    NamedNode::new(iri).map_err(|e| OxirsError::Parse(e.to_string()))
}
```

### API Stability

- **Stable**: core RDF model types (`NamedNode`, `Literal`, `Triple`, `Quad`), `RdfStore` operations (insert/query/remove), parser/serializer interfaces
- **Unstable**: advanced query-optimization internals (may change without a major version bump)
- **Experimental**: `consciousness`, `molecular`, and `quantum` modules

## Ecosystem Integration

### OxiGraph Compatibility

`oxigraph_compat::Store` mirrors `oxigraph::Store`'s API — including its interior-mutability semantics, where mutating methods take `&self` — so migrating existing OxiGraph code is mostly a matter of changing imports:

```rust
// Before (with OxiGraph)
use oxigraph::model::{NamedNode, Literal, Triple};

// After (with oxirs-core)
use oxirs_core::model::{NamedNode, Literal, Triple};
use oxirs_core::oxigraph_compat::Store;
```

### OxiRS Platform Components

Every other crate in the workspace builds on `oxirs-core`:

- [`oxirs-arq`](../../engine/oxirs-arq/) — SPARQL 1.1/1.2 query engine with cost-based optimization
- [`oxirs-shacl`](../../engine/oxirs-shacl/) — SHACL shape validation
- [`oxirs-fuseki`](../../server/oxirs-fuseki/) — SPARQL HTTP server
- [`oxirs-gql`](../../server/oxirs-gql/) — GraphQL interface
- [`oxirs-chat`](../../ai/oxirs-chat/) — AI-powered natural-language-to-SPARQL
- [`oxirs-embed`](../../ai/oxirs-embed/) — knowledge graph embeddings
- [`oxirs-cluster`](../../storage/oxirs-cluster/) — distributed storage with consensus
- [`oxirs-tdb`](../../storage/oxirs-tdb/) — persistent triple database with ACID transactions

## Development

### Running Tests

```bash
cd core/oxirs-core
cargo nextest run --all-features
# Current status: 2589 tests passing
```

### Benchmarks

```bash
# Criterion suite: indexed_graph, concurrent_graph, parallel_batch, rdf, query_visualization, sla_suite
cargo bench --release

# SLA regression checks (release-only; compares against perf_baseline.json)
cargo test --release -- --ignored sla_suite
```

### Documentation

```bash
cargo doc --open --all-features
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`cargo nextest run --all-features`) with zero `clippy` warnings
5. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for full guidelines.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

## Status

🚀 **Production Release (v0.3.2)** — 2,589 tests passing, zero `clippy` warnings, zero rustdoc errors.

### Current Highlights
- **RDF/SPARQL core**: RDF 1.2 data model, SPARQL 1.1/1.2 query and update, federation via `SERVICE`
- **Persistence**: `RdfStore::open` with N-Quads-backed durability
- **Pure-Rust TLS**: a `#[ctor::ctor]` constructor installs OxiTLS's Pure Rust `rustls` `CryptoProvider` as the process default for every binary and test process that links this crate, so no `ring`/`aws-lc-sys` is required at runtime
- **Compliance tooling**: SOC2/GDPR audit trail (`audit`), URDNA2015 canonicalization (`canon`), PROV-O provenance (`provenance`)
- **API stability enforcement**: `api_baseline.json` plus `tests/api_stability.rs` guard the public surface against accidental breaking changes

---

Part of the [OxiRS](https://github.com/cool-japan/oxirs) semantic web platform.
