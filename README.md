# OxiRS

> A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning

[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0--alpha.3-orange)](https://github.com/cool-japan/oxirs/releases)

**Status**: Alpha Release (v0.1.0-alpha.3) - Released October 12, 2025

⚠️ **Alpha Software**: This is an early alpha release. APIs may change without notice. Suitable for production alpha testing and internal applications.

## Vision

OxiRS aims to be a **Rust-first, JVM-free** alternative to Apache Jena + Fuseki and to Juniper, providing:

- **Protocol choice, not lock-in**: Expose both SPARQL 1.2 and GraphQL endpoints from the same dataset
- **Incremental adoption**: Each crate works stand-alone; opt into advanced features via Cargo features
- **AI readiness**: Native integration with vector search, graph embeddings, and LLM-augmented querying
- **Single static binary**: Match or exceed Jena/Fuseki feature-for-feature while keeping a <50MB footprint

## Quick Start

### Installation

```bash
# Install the CLI tool
cargo install oxirs --version 0.1.0-alpha.3

# Or build from source
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --workspace --release
```

### What’s New in v0.1.0-alpha.3 (October 12, 2025)

- **Query Intelligence**: `oxirs explain` introduces PostgreSQL-style plans with analyze/full modes, complexity scoring, and optimization hints for SPARQL workloads.
- **Reusable SPARQL Templates**: Nine parameterizable templates (basic, federation, analytics, property paths) now ship with the CLI for faster queries.
- **Persistent Query History**: Automatic tracking, replay, search, and statistics for every CLI query, stored under `~/.local/share/oxirs/query_history.json`.
- **Industry 4.0 SAMM tooling**: Six new generators (GraphQL, TypeScript, Python, Java, Scala, SQL) plus AAS pipelines deliver 16 total codegen targets.
- **Enterprise Quality Bar**: Workspace-wide `-D warnings` enforcement, 200+ Clippy fixes, and 4,421 tests ensure a clean alpha-grade release.

### Usage

```bash
# Initialize a new knowledge graph (alphanumeric, _, - only)
oxirs init mykg

# Import RDF data (automatically persisted to mykg/data.nq)
oxirs import mykg data.ttl --format turtle

# Query the data (loaded automatically from disk)
oxirs query mykg "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Query with specific patterns
oxirs query mykg "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"

# Start the server
oxirs serve mykg/oxirs.toml --port 3030
```

**Features:**
- ✅ **Persistent storage**: Data automatically saved to disk in N-Quads format
- ✅ **SPARQL queries**: SELECT, ASK, CONSTRUCT, DESCRIBE supported
- ✅ **Auto-load**: No manual save/load needed
- 🚧 **PREFIX support**: Coming in next release

Open:
- http://localhost:3030 for the Fuseki-style admin UI
- http://localhost:3030/graphql for GraphiQL (if enabled)

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
| **[oxirs-samm]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-samm.svg)](https://crates.io/crates/oxirs-samm) | [![docs.rs](https://docs.rs/oxirs-samm/badge.svg)](https://docs.rs/oxirs-samm) | SAMM metamodel & AAS |
| **[oxirs-geosparql]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-geosparql.svg)](https://crates.io/crates/oxirs-geosparql) | [![docs.rs](https://docs.rs/oxirs-geosparql/badge.svg)](https://docs.rs/oxirs-geosparql) | GeoSPARQL support |
| **[oxirs-star]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-star.svg)](https://crates.io/crates/oxirs-star) | [![docs.rs](https://docs.rs/oxirs-star/badge.svg)](https://docs.rs/oxirs-star) | RDF-star support |
| **[oxirs-ttl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-ttl.svg)](https://crates.io/crates/oxirs-ttl) | [![docs.rs](https://docs.rs/oxirs-ttl/badge.svg)](https://docs.rs/oxirs-ttl) | Turtle parser |
| **[oxirs-vec]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-vec.svg)](https://crates.io/crates/oxirs-vec) | [![docs.rs](https://docs.rs/oxirs-vec/badge.svg)](https://docs.rs/oxirs-vec) | Vector search |

[oxirs-arq]: https://crates.io/crates/oxirs-arq
[oxirs-rule]: https://crates.io/crates/oxirs-rule
[oxirs-shacl]: https://crates.io/crates/oxirs-shacl
[oxirs-samm]: https://crates.io/crates/oxirs-samm
[oxirs-geosparql]: https://crates.io/crates/oxirs-geosparql
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
│  ├─ oxirs-samm        # SAMM metamodel + AAS integration (Industry 4.0)
│  ├─ oxirs-geosparql   # GeoSPARQL spatial queries and topological relations
│  ├─ oxirs-shacl       # SHACL Core + SHACL-SPARQL validator
│  ├─ oxirs-star        # RDF-star / SPARQL-star grammar support
│  ├─ oxirs-ttl         # Turtle/TriG parser and serializer
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

## Feature Matrix (v0.1.0-alpha.3)

| Capability | Oxirs crate(s) | Status | Jena / Fuseki parity |
|------------|----------------|--------|----------------------|
| **Core RDF & SPARQL** | | | |
| RDF 1.2 & syntaxes (7 formats) | `oxirs-core` | ✅ Alpha (519 tests) | ✅ |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ Alpha (466 tests) | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (`star` flag) | ✅ Alpha | 🔸 |
| Persistent storage (N-Quads) | `oxirs-core` | ✅ Alpha | ✅ |
| **Semantic Web Extensions** | | | |
| RDF-star parse/serialise | `oxirs-star` | ✅ Alpha (157 tests) | 🔸 (Jena dev build) |
| SHACL Core+API (W3C compliant) | `oxirs-shacl` | ✅ Alpha (344 tests, 27/27 W3C) | ✅ |
| Rule reasoning (RDFS/OWL) | `oxirs-rule` | ✅ Alpha (170 tests) | ✅ |
| SAMM 2.0-2.3 & AAS (Industry 4.0) | `oxirs-samm` | ✅ Alpha (16 generators) | ❌ |
| **Query & Federation** | | | |
| GraphQL API | `oxirs-gql` | ✅ Alpha (118 tests) | ❌ |
| SPARQL Federation (SERVICE) | `oxirs-federate` | ✅ Alpha (285 tests, 2PC) | ✅ |
| Federated authentication | `oxirs-federate` | ✅ Alpha (OAuth2/SAML/JWT) | 🔸 |
| **Real-time & Streaming** | | | |
| Stream processing (Kafka/NATS) | `oxirs-stream` | ✅ Alpha (214 tests, SIMD) | 🔸 (Jena + external) |
| RDF Patch & SPARQL Update delta | `oxirs-stream` | ✅ Alpha | 🔸 |
| **Search & Geo** | | | |
| Full-text search (`text:`) | `oxirs-textsearch` | ⏳ Planned | ✅ |
| GeoSPARQL (OGC 1.1) | `oxirs-geosparql` (`geo`) | ✅ Alpha (183 tests) | ✅ |
| Vector search / embeddings | `oxirs-vec` (323 tests), `oxirs-embed` (296 tests) | ✅ Alpha | ❌ |
| **Storage & Distribution** | | | |
| TDB2-compatible storage | `oxirs-tdb` | ✅ Alpha (193 tests) | ✅ |
| Distributed / HA store (Raft) | `oxirs-cluster` (`cluster`) | ✅ Alpha | 🔸 (Jena + external) |
| **AI & Advanced Features** | | | |
| RAG chat API (LLM integration) | `oxirs-chat` | ✅ Alpha | ❌ |
| AI-powered SHACL validation | `oxirs-shacl-ai` | ✅ Alpha (278 tests) | ❌ |

**Legend:**
- ✅ Alpha: Usable with 100+ tests, may have bugs, suitable for alpha testing
- 🔄 Experimental: Under active development, APIs unstable
- ⏳ Planned: Not yet implemented
- 🔸 Partial/plug-in support in Jena

**Quality Metrics (v0.1.0-alpha.3):**
- 4,421 tests passing (99.98% pass rate)
- Zero compilation warnings (enforced with `-D warnings`)
- 200+ clippy lints fixed
- 7/7 integration tests passing

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

| Version | Target Date | Milestone | Deliverables | Status |
|---------|-------------|-----------|--------------|---------|
| **v0.1.0-alpha.1** | **✅ Sep 30, 2025** | **Initial Alpha** | Core RDF/SPARQL, GraphQL, AI modules foundation | ✅ Released |
| **v0.1.0-alpha.2** | **✅ Oct 4, 2025** | **Alpha Enhancements** | Persistent storage, CLI parity, federation, observability | ✅ Released |
| **v0.1.0-alpha.3** | **✅ Oct 12, 2025** | **SAMM & Quality** | Industry 4.0 (16 generators), zero-warning compilation, 4,421 tests | ✅ Released |
| **v0.1.0-beta.1** | **Dec 2025** | **Beta Release** | API stability, production hardening, 95%+ test coverage, full docs | 🎯 Next |
| **v0.2.0** | **Q1 2026** | **Performance** | Query optimization (10x), AI production-ready, multi-region clustering | 📋 Planned |
| **v0.3.0** | **Q2 2026** | **Search & Geo** | Full-text search (Tantivy), GeoSPARQL, bulk loader, performance SLAs | 📋 Planned |
| **v1.0.0** | **Q4 2026** | **Production Ready** | Full Jena parity verified, enterprise support, LTS guarantees | 📋 Planned |

### Alpha.3 Achievements (October 12, 2025)

**SAMM & AAS Integration:**
- ✅ 16 code generators (GraphQL, TypeScript, Python, Java, Scala, SQL, OpenAPI, AsyncAPI, HTML, JSON Schema, Markdown, Rust, AAS, Turtle, Sample, Diagram)
- ✅ 100% Java ESMF SDK command coverage (19/19 commands)
- ✅ AAS to SAMM conversion pipeline (XML/JSON/AASX support)
- ✅ Package management with namespace sharing

**Federation & Distribution:**
- ✅ oxirs-federate: 100% Beta Release targets achieved in alpha.3
- ✅ Distributed transactions (2PC, Saga pattern, eventual consistency)
- ✅ Multi-provider authentication (OAuth2, SAML, JWT, API keys)
- ✅ OpenTelemetry integration with circuit breakers

**Code Quality Excellence:**
- ✅ Zero-warning compilation enforced (`-D warnings`)
- ✅ 200+ clippy lints fixed across 13+ crates
- ✅ 4,421 tests passing (99.98% pass rate, 88.8s execution)
- ✅ SHACL: 100% W3C compliance (27/27 constraints, 344 tests)

**Production Features:**
- ✅ Performance module (caching, profiling, batch processing)
- ✅ Template engine with custom filters
- ✅ Metrics, health checks, structured logging

**GeoSPARQL & Spatial Features:**
- ✅ OGC GeoSPARQL 1.1 compliance (183 tests)
- ✅ R-tree spatial indexing with stress tests (50k points)
- ✅ Performance optimization module (parallel, streaming for large datasets)
- ✅ Simple Features, Egenhofer-9, RCC-8 topological relations
- ✅ WKT/GML parsing, CRS transformations (PROJ integration)
- ✅ Comprehensive spatial queries (bbox, within-distance, k-NN)

### Next Milestone: Beta.1 (December 2025)

**Beta.1 Features Already Complete in Alpha.3:** 🎉
- ✅ **Production Hardening** - oxirs-core, oxirs-arq, oxirs-fuseki
  - Circuit breakers for fault tolerance
  - Performance monitoring with latency statistics
  - Resource quotas and rate limiting
  - Health checks for all components
  - Comprehensive benchmarking suites (17 benchmark groups total)
  - Stress testing suites (20 comprehensive tests total)

**Remaining Focus Areas:**
- 🎯 API stability and versioning guarantees
- 🎯 Production performance benchmarking (validate 10x claims)
- 🎯 Security audit and hardening
- 🎯 Comprehensive documentation (95%+ coverage)
- 🎯 Test coverage increase to 95%+
- 🎯 Migration guides and examples
- 🎯 Performance SLAs and optimization

## License

OxiRS is dual-licensed under either:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

See [LICENSE](LICENSE) for details.

## Contact

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **Maintainer**: @cool-japan (KitaSan)

## Release Notes (v0.1.0-alpha.3)

📄 Full notes live in [CHANGELOG.md](CHANGELOG.md).

### Highlights
- 🏭 **SAMM & AAS Integration**: Industry 4.0 digital twin support with SAMM (Semantic Aspect Meta Model) 2.0.0-2.3.0 parser and bidirectional AAS (Asset Administration Shell) conversion
- 🎨 **16 Code Generators**: GraphQL, TypeScript, Python, Java, Scala, Rust, SQL, OpenAPI, AsyncAPI, HTML, JSON Schema, Markdown, and more
- 🔄 **Java ESMF SDK Compatible**: Drop-in replacement syntax (`samm` → `oxirs`) for seamless migration from Java tooling
- ⚙️ **Persistent RDF pipeline**: Automatic on-disk save/load in N-Quads, streaming import/export/migrate flows, and configurable parallel batch ingestion
- 🧠 **Interactive SPARQL tooling**: Full-featured CLI REPL with history search, templates, syntax hints, SELECT */wildcard fixes, and multi-line editing
- 🌐 **Federated querying**: SPARQL 1.1 `SERVICE` support with retries, `SERVICE SILENT`, JSON results merging, and verified interoperability with DBpedia/Wikidata
- 🔐 **Production safeguards**: OAuth2/OIDC + JWT, seven security headers, HSTS, structured logging, and Prometheus metrics with slow-query tracing
- 🚀 **Performance improvements**: SIMD-accelerated SciRS2 operators, streaming pipelines, and 4,421+ tests (including 7 integration suites) covering the new workflow
- ✨ **Code quality**: Zero-warning compilation enforced with `-D warnings` across all 21 crates - 200+ clippy lints fixed

### Known Issues
- Large dataset (>100M triples) performance optimization continues; benchmark feedback appreciated
- AI-centric crates (`oxirs-chat`, `oxirs-embed`, `oxirs-shacl-ai`) remain experimental
- Advanced serialization documentation being expanded

### Quality Metrics (v0.1.0-alpha.3)
- ✅ **Zero warnings** - Strict `-D warnings` enforced across all 21 crates
- ✅ **4,421 tests passing** - 99.98% pass rate (88.8s execution time)
- ✅ **200+ clippy lints fixed** - Comprehensive code quality improvements
- ✅ **7/7 integration tests passing** - Complete RDF pipeline validated

### Upgrade Notes
- Install the new CLI with `cargo install oxirs --version 0.1.0-alpha.3` or update individual crates via `Cargo.toml`
- **Breaking change**: CLI syntax updated to match Java ESMF SDK - replace `oxirs samm` with `oxirs aspect` (see [CHANGELOG.md](CHANGELOG.md) for migration guide)
- New `oxirs aas` command for AAS integration (XML/JSON/AASX support)
- Existing dataset directories from alpha.1/alpha.2 remain compatible; the new persistence layer will automatically detect and upgrade saved N-Quads data
- Outbound HTTP access is required for federation; configure firewall rules and timeouts before enabling cross-endpoint queries

---

*"Rust makes memory safety table stakes; Oxirs makes knowledge-graph engineering table stakes."*

**Third alpha release - October 12, 2025**