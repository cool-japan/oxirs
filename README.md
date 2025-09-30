# OxiRS

> A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning

[![Rust](https://github.com/cool-japan/oxirs/workflows/Rust/badge.svg)](https://github.com/cool-japan/oxirs/actions)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0--alpha.1-orange)](https://github.com/cool-japan/oxirs/releases)

**Status**: Alpha Release (v0.1.0-alpha.1) - Released September 30, 2025

⚠️ **Alpha Software**: This is an early alpha release. APIs may change without notice. Not recommended for production use.

## Vision

OxiRS aims to be a **Rust-first, JVM-free** alternative to Apache Jena + Fuseki and to Jupiper, providing:

- **Protocol choice, not lock-in**: Expose both SPARQL 1.2 and GraphQL endpoints from the same dataset
- **Incremental adoption**: Each crate works stand-alone; opt into advanced features via Cargo features
- **AI readiness**: Native integration with vector search, graph embeddings, and LLM-augmented querying
- **Single static binary**: Match or exceed Jena/Fuseki feature-for-feature while keeping a <50MB footprint

## Quick Start

### Installation

```bash
# Install the CLI tool
cargo install oxirs

# Or build from source
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --workspace --release
```

### Usage

```bash
# Initialize a new knowledge graph
oxirs init mykg

# Start the server
oxirs serve mykg.toml --port 3030
```

Open:
- http://localhost:3030 for the Fuseki-style admin UI
- http://localhost:3030/graphql for GraphiQL

## Published Crates

All crates are published to [crates.io](https://crates.io) and documented on [docs.rs](https://docs.rs).

### Core

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-core]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-core.svg)](https://crates.io/crates/oxirs-core) | [![docs.rs](https://docs.rs/oxirs-core/badge.svg)](https://docs.rs/oxirs-core) | Core RDF and SPARQL functionality |

[oxirs-core]: https://crates.io/crates/oxirs-core

### Server

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-fuseki]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-fuseki.svg)](https://crates.io/crates/oxirs-fuseki) | [![docs.rs](https://docs.rs/oxirs-fuseki/badge.svg)](https://docs.rs/oxirs-fuseki) | SPARQL 1.1/1.2 HTTP server |
| **[oxirs-gql]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-gql.svg)](https://crates.io/crates/oxirs-gql) | [![docs.rs](https://docs.rs/oxirs-gql/badge.svg)](https://docs.rs/oxirs-gql) | GraphQL endpoint for RDF |

[oxirs-fuseki]: https://crates.io/crates/oxirs-fuseki
[oxirs-gql]: https://crates.io/crates/oxirs-gql

### Engine

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-arq]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-arq.svg)](https://crates.io/crates/oxirs-arq) | [![docs.rs](https://docs.rs/oxirs-arq/badge.svg)](https://docs.rs/oxirs-arq) | SPARQL query engine |
| **[oxirs-rule]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-rule.svg)](https://crates.io/crates/oxirs-rule) | [![docs.rs](https://docs.rs/oxirs-rule/badge.svg)](https://docs.rs/oxirs-rule) | Rule-based reasoning |
| **[oxirs-shacl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl.svg)](https://crates.io/crates/oxirs-shacl) | [![docs.rs](https://docs.rs/oxirs-shacl/badge.svg)](https://docs.rs/oxirs-shacl) | SHACL validation |
| **[oxirs-star]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-star.svg)](https://crates.io/crates/oxirs-star) | [![docs.rs](https://docs.rs/oxirs-star/badge.svg)](https://docs.rs/oxirs-star) | RDF-star support |
| **[oxirs-ttl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-ttl.svg)](https://crates.io/crates/oxirs-ttl) | [![docs.rs](https://docs.rs/oxirs-ttl/badge.svg)](https://docs.rs/oxirs-ttl) | Turtle parser |
| **[oxirs-vec]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-vec.svg)](https://crates.io/crates/oxirs-vec) | [![docs.rs](https://docs.rs/oxirs-vec/badge.svg)](https://docs.rs/oxirs-vec) | Vector search |

[oxirs-arq]: https://crates.io/crates/oxirs-arq
[oxirs-rule]: https://crates.io/crates/oxirs-rule
[oxirs-shacl]: https://crates.io/crates/oxirs-shacl
[oxirs-star]: https://crates.io/crates/oxirs-star
[oxirs-ttl]: https://crates.io/crates/oxirs-ttl
[oxirs-vec]: https://crates.io/crates/oxirs-vec

### Storage

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-tdb]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-tdb.svg)](https://crates.io/crates/oxirs-tdb) | [![docs.rs](https://docs.rs/oxirs-tdb/badge.svg)](https://docs.rs/oxirs-tdb) | TDB2-compatible storage |
| **[oxirs-cluster]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-cluster.svg)](https://crates.io/crates/oxirs-cluster) | [![docs.rs](https://docs.rs/oxirs-cluster/badge.svg)](https://docs.rs/oxirs-cluster) | Distributed clustering |

[oxirs-tdb]: https://crates.io/crates/oxirs-tdb
[oxirs-cluster]: https://crates.io/crates/oxirs-cluster

### Stream

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-stream]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-stream.svg)](https://crates.io/crates/oxirs-stream) | [![docs.rs](https://docs.rs/oxirs-stream/badge.svg)](https://docs.rs/oxirs-stream) | Real-time streaming |
| **[oxirs-federate]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-federate.svg)](https://crates.io/crates/oxirs-federate) | [![docs.rs](https://docs.rs/oxirs-federate/badge.svg)](https://docs.rs/oxirs-federate) | Federated queries |

[oxirs-stream]: https://crates.io/crates/oxirs-stream
[oxirs-federate]: https://crates.io/crates/oxirs-federate

### AI

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-embed]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-embed.svg)](https://crates.io/crates/oxirs-embed) | [![docs.rs](https://docs.rs/oxirs-embed/badge.svg)](https://docs.rs/oxirs-embed) | Knowledge graph embeddings |
| **[oxirs-shacl-ai]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl-ai.svg)](https://crates.io/crates/oxirs-shacl-ai) | [![docs.rs](https://docs.rs/oxirs-shacl-ai/badge.svg)](https://docs.rs/oxirs-shacl-ai) | AI-powered SHACL |
| **[oxirs-chat]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-chat.svg)](https://crates.io/crates/oxirs-chat) | [![docs.rs](https://docs.rs/oxirs-chat/badge.svg)](https://docs.rs/oxirs-chat) | RAG chat API |

[oxirs-embed]: https://crates.io/crates/oxirs-embed
[oxirs-shacl-ai]: https://crates.io/crates/oxirs-shacl-ai
[oxirs-chat]: https://crates.io/crates/oxirs-chat

### Tools

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs (CLI)]** | [![Crates.io](https://img.shields.io/crates/v/oxirs.svg)](https://crates.io/crates/oxirs) | [![docs.rs](https://docs.rs/oxirs/badge.svg)](https://docs.rs/oxirs) | CLI tool |

[oxirs (CLI)]: https://crates.io/crates/oxirs

## Architecture

```
oxirs/                  # Cargo workspace root
├─ core/                # Thin, safe re-export of oxigraph
│  └─ oxirs-core
├─ server/              # Network front ends
│  ├─ oxirs-fuseki      # SPARQL 1.1/1.2 HTTP protocol, Fuseki-compatible config
│  └─ oxirs-gql         # GraphQL façade (Juniper + mapping layer)
├─ engine/              # Query, update, reasoning
│  ├─ oxirs-arq         # Jena-style algebra + extension points
│  ├─ oxirs-rule        # Forward/backward rule engine (RDFS/OWL/SWRL)
│  ├─ oxirs-shacl       # SHACL Core + SHACL-SPARQL validator
│  ├─ oxirs-star        # RDF-star / SPARQL-star grammar support
│  └─ oxirs-vec         # Vector index abstractions (SciRS2, hnsw_rs)
├─ storage/
│  ├─ oxirs-tdb         # MVCC layer & assembler grammar (TDB2 parity)
│  └─ oxirs-cluster     # Raft-backed distributed dataset
├─ stream/              # Real-time and federation
│  ├─ oxirs-stream      # Kafka/NATS I/O, RDF Patch, SPARQL Update delta
│  └─ oxirs-federate    # SERVICE planner, GraphQL stitching
├─ ai/
│  ├─ oxirs-embed       # KG embeddings (TransE, ComplEx…)
│  ├─ oxirs-shacl-ai    # Shape induction & data repair suggestions
│  └─ oxirs-chat        # RAG chat API (LLM + SPARQL)
└─ tools/
    ├─ oxirs             # CLI (import, export, star-migrate, bench)
    └─ benchmarks/       # SP2Bench, WatDiv, LDBC SGS
```

## Feature Matrix (v0.1.0-alpha.1)

| Capability | Oxirs crate(s) | Status | Jena / Fuseki parity |
|------------|----------------|--------|----------------------|
| RDF 1.2 & syntaxes | `oxirs-core` | ✅ Alpha | ✅ |
| RDF-star parse/serialise | `oxirs-star` | 🔄 Experimental | 🔸 (Jena dev build) |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ Alpha | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (`star` flag) | 🔄 Experimental | 🔸 |
| Rule reasoning (RDFS/OWL) | `oxirs-rule` | 🔄 Experimental | ✅ |
| SHACL Core+API | `oxirs-shacl` | 🔄 Experimental | ✅ |
| Full-text search (`text:`) | `oxirs-textsearch` | ⏳ Planned | ✅ |
| GeoSPARQL | `oxirs-geosparql` (`geo`) | ⏳ Planned | ✅ |
| GraphQL API | `oxirs-gql` | ✅ Alpha | ❌ |
| Vector search / embeddings | `oxirs-vec`, `oxirs-embed` (`ai`) | 🔄 Experimental | ❌ |
| Distributed / HA store | `oxirs-cluster` (`cluster`) | 🔄 Experimental | 🔸 (Jena + external) |

Legend: ✅ Alpha - Usable but may have bugs, 🔄 Experimental - Under development, ⏳ Planned - Not yet implemented, 🔸 partial/plug-in

## Usage Examples

### Dataset Configuration (TOML)

```toml
[dataset.mykg]
type      = "tdb2"
location  = "/data"
text      = { enabled = true, analyzer = "english" }
shacl     = ["./shapes/person.ttl"]
```

### GraphQL Query (auto-generated)

```graphql
query {
  Person(where: {familyName: "Yamada"}) {
    givenName
    homepage
    knows(limit: 5) { givenName }
  }
}
```

### Vector Similarity SPARQL Service (opt-in AI)

```sparql
SELECT ?s ?score WHERE {
  SERVICE <vec:similar ( "LLM embeddings of 'semantic web'" 0.8 )> {
    ?s ?score .
  }
}
```

## Development

### Prerequisites

- Rust 1.70+ (MSRV)
- Optional: Docker for containerized deployment

### Building

```bash
# Clone the repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Build all crates
cargo build --workspace

# Run tests
cargo nextest run --no-fail-fast

# Run with all features
cargo build --workspace --all-features
```

### Feature Flags

Optional features to keep dependencies minimal:

- `geo`: GeoSPARQL support
- `text`: Full-text search with Tantivy
- `ai`: Vector search and embeddings
- `cluster`: Distributed storage with Raft
- `star`: RDF-star and SPARQL-star support
- `vec`: Vector index abstractions

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### RFC Process

- Design documents go in `./rfcs/` with lazy-consensus and 14-day comment window
- All code must pass `rustfmt + nightly 2025-06`, Clippy `--all-targets --workspace -D warnings`
- Commit sign-off required (DCO 1.1)

## Roadmap

| Version | Target Date | Milestone | Deliverables |
|---------|-------------|-----------|--------------|
| **v0.1.0-alpha.1** | **✅ Sep 2025** | **Alpha Release** | Core SPARQL/GraphQL server, basic features |
| **v0.1.0-beta.1** | **Dec 2025** | **Beta Release** | API stability, production hardening, full docs |
| **v0.2.0** | **Q1 2026** | **Enhanced Features** | Advanced optimization, AI capabilities, clustering |
| **v0.3.0** | **Q2 2026** | **Text & Geo** | Full-text search, GeoSPARQL, bulk loader |
| **v0.4.0** | **Q3 2026** | **AI & Streaming** | Vector search, embeddings, RAG, streaming |
| **v1.0.0** | **Q4 2026** | **Production Ready** | Full Jena parity, enterprise support, LTS |

## License

OxiRS is dual-licensed under either:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

See [LICENSE](LICENSE) for details.

## Contact

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **Maintainer**: @cool-japan (KitaSan)

## Release Notes (v0.1.0-alpha.1)

### What's Included
- ✅ Basic SPARQL 1.1/1.2 query engine
- ✅ GraphQL endpoint generation
- ✅ RDF parsing (Turtle, N-Triples, JSON-LD, RDF/XML)
- ✅ In-memory and disk-based storage
- ✅ Basic federated queries
- ✅ Experimental AI features (embeddings, chat)
- ✅ Experimental distributed clustering

### Known Issues
- Performance not yet optimized
- Some advanced SPARQL features incomplete
- Limited error handling in some areas
- Documentation incomplete
- API subject to change

### Feedback Welcome
Please report bugs and feature requests at: https://github.com/cool-japan/oxirs/issues

---

*"Rust makes memory safety table stakes; Oxirs makes knowledge-graph engineering table stakes."*

**First alpha release - September 30, 2025**