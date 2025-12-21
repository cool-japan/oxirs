# OxiRS

> A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning

[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0--beta.2-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: Beta Release (v0.1.0-beta.2) - Released December 21, 2025

⚡ **Beta Software**: API stability guaranteed. Feature complete for 1.0. Suitable for production use with comprehensive testing.

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
cargo install oxirs --version 0.1.0-beta.2

# Or build from source
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --workspace --release
```

### What's New in v0.1.0-beta.2 (December 2025)

- **CUDA Support**: GPU acceleration for knowledge graph embeddings and vector operations.
- **Enhanced AI Modules**: Improved vision-language graph processing and Tucker decomposition models.
- **Performance Improvements**: Memory-mapped optimizations for oxirs-tdb storage and enhanced SIMD operations.
- **SAMM Enhancements**: Performance regression tests and improved metamodel code generation.
- **Documentation Updates**: Refreshed API documentation and examples across all crates.
- **Bug Fixes**: Various stability improvements and edge case handling across all modules.

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

## Feature Matrix (v0.1.0-beta.2)

| Capability | Oxirs crate(s) | Status | Jena / Fuseki parity |
|------------|----------------|--------|----------------------|
| **Core RDF & SPARQL** | | | |
| RDF 1.2 & syntaxes (7 formats) | `oxirs-core` | ✅ Beta (600+ tests) | ✅ |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ Beta (550+ tests) | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (`star` flag) | ✅ Beta | 🔸 |
| Persistent storage (N-Quads) | `oxirs-core` | ✅ Beta | ✅ |
| **Semantic Web Extensions** | | | |
| RDF-star parse/serialise | `oxirs-star` | ✅ Beta (200+ tests) | 🔸 (Jena dev build) |
| SHACL Core+API (W3C compliant) | `oxirs-shacl` | ✅ Beta (400+ tests, 27/27 W3C) | ✅ |
| Rule reasoning (RDFS/OWL) | `oxirs-rule` | ✅ Beta (200+ tests) | ✅ |
| SAMM 2.0-2.3 & AAS (Industry 4.0) | `oxirs-samm` | ✅ Beta (16 generators) | ❌ |
| **Query & Federation** | | | |
| GraphQL API | `oxirs-gql` | ✅ Beta (150+ tests) | ❌ |
| SPARQL Federation (SERVICE) | `oxirs-federate` | ✅ Beta (350+ tests, 2PC) | ✅ |
| Federated authentication | `oxirs-federate` | ✅ Beta (OAuth2/SAML/JWT) | 🔸 |
| **Real-time & Streaming** | | | |
| Stream processing (Kafka/NATS) | `oxirs-stream` | ✅ Beta (300+ tests, SIMD) | 🔸 (Jena + external) |
| RDF Patch & SPARQL Update delta | `oxirs-stream` | ✅ Beta | 🔸 |
| **Search & Geo** | | | |
| Full-text search (`text:`) | `oxirs-textsearch` | ⏳ Planned | ✅ |
| GeoSPARQL (OGC 1.1) | `oxirs-geosparql` (`geo`) | ✅ Beta (250+ tests) | ✅ |
| Vector search / embeddings | `oxirs-vec` (400+ tests), `oxirs-embed` (350+ tests) | ✅ Beta | ❌ |
| **Storage & Distribution** | | | |
| TDB2-compatible storage | `oxirs-tdb` | ✅ Beta (250+ tests) | ✅ |
| Distributed / HA store (Raft) | `oxirs-cluster` (`cluster`) | ✅ Beta | 🔸 (Jena + external) |
| **AI & Advanced Features** | | | |
| RAG chat API (LLM integration) | `oxirs-chat` | ✅ Beta | ❌ |
| AI-powered SHACL validation | `oxirs-shacl-ai` | ✅ Beta (350+ tests) | ❌ |
| **Security & Authorization** | | | |
| ReBAC (Relationship-Based Access Control) | `oxirs-fuseki` | ✅ Beta (83 tests) | ❌ |
| Graph-level authorization | `oxirs-fuseki` | ✅ Beta | ❌ |
| SPARQL-based authorization storage | `oxirs-fuseki` | ✅ Beta | ❌ |
| OAuth2/OIDC/SAML authentication | `oxirs-fuseki` | ✅ Beta | 🔸 |

**Legend:**
- ✅ Beta: Production-ready with comprehensive tests, API stability guaranteed
- 🔄 Experimental: Under active development, APIs unstable
- ⏳ Planned: Not yet implemented
- 🔸 Partial/plug-in support in Jena

**Quality Metrics (v0.1.0-beta.2):**
- **12,248 tests passing** (100% pass rate, 100 skipped)
- **Zero compilation warnings** (enforced with `-D warnings`)
- **95%+ test coverage** across all modules
- **95%+ documentation coverage**
- **All integration tests passing**
- **Production-grade security audit completed**
- **CUDA GPU support** for AI acceleration

## Usage Examples

### Dataset Configuration (TOML)

```toml
[dataset.mykg]
type      = "tdb2"
location  = "/data"
text      = { enabled = true, analyzer = "english" }
shacl     = ["./shapes/person.ttl"]

# ReBAC Authorization (optional)
[security.policy_engine]
mode = "Combined"  # RbacOnly | RebacOnly | Combined | Both

[security.rebac]
backend = "InMemory"  # InMemory | RdfNative
namespace = "http://oxirs.org/auth#"
inference_enabled = true

[[security.rebac.initial_relationships]]
subject = "user:alice"
relation = "owner"
object = "dataset:mykg"
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
| **v0.1.0-alpha.3** | **✅ Oct 12, 2025** | **Code Quality** | Zero-warning compilation, 200+ clippy lints fixed, module compliance | ✅ Released |
| **v0.1.0-beta.1** | **✅ Nov 16, 2025** | **Beta Release** | API stability, production hardening, 8,690 tests, 95%+ coverage, ReBAC | ✅ Released |
| **v0.1.0-beta.2** | **✅ Dec 21, 2025** | **Beta Enhancement** | CUDA support, AI improvements, performance optimizations, mmap storage | ✅ Released |
| **v0.2.0** | **Q1 2026** | **Performance** | Query optimization (10x), AI production-ready, multi-region clustering | 🎯 Next |
| **v0.3.0** | **Q2 2026** | **Search & Geo** | Full-text search (Tantivy), GeoSPARQL, bulk loader, performance SLAs | 📋 Planned |
| **v1.0.0** | **Q4 2026** | **Production Ready** | Full Jena parity verified, enterprise support, LTS guarantees | 📋 Planned |

### Beta.2 Achievements (December 2025)

**CUDA & GPU Acceleration:**
- ✅ CUDA backend for knowledge graph embeddings
- ✅ GPU-accelerated vector operations
- ✅ Enhanced tensor processing for AI modules

**AI Module Improvements:**
- ✅ Vision-language graph processing enhancements
- ✅ Tucker decomposition model improvements
- ✅ Enhanced embedding algorithms

**Performance & Storage:**
- ✅ Memory-mapped file optimizations for oxirs-tdb
- ✅ SIMD operation enhancements
- ✅ SAMM performance regression testing

### Beta.1 Achievements (November 2025)

**API Stability & Production Readiness:**
- ✅ API stability guarantees with semantic versioning
- ✅ Comprehensive error handling and recovery patterns
- ✅ Production-grade logging and observability
- ✅ Resource management and leak prevention
- ✅ Graceful degradation and fault tolerance

**Documentation Excellence:**
- ✅ 95%+ documentation coverage across all crates
- ✅ Comprehensive API documentation with examples
- ✅ Migration guides from alpha to beta
- ✅ Production deployment guides
- ✅ Performance tuning documentation

**Testing & Quality:**
- ✅ 95%+ test coverage across all modules
- ✅ **8,690 tests passing** (100% pass rate, 79 skipped)
- ✅ Comprehensive integration test suites
- ✅ Performance benchmarks and stress tests
- ✅ Security testing and vulnerability scanning

**Codebase Scale:**
- ✅ **1,577,497 lines of Rust** across 3,126 files
- ✅ **1.29M lines of production code** with 66,158 comments
- ✅ **149,010 lines of inline documentation**
- ✅ **Comprehensive guides and docs** across all modules

**Performance Optimization:**
- ✅ Query engine optimization and caching
- ✅ Memory usage optimization
- ✅ Parallel processing enhancements
- ✅ Connection pooling and resource management
- ✅ Production performance validation

**Security Enhancements:**
- ✅ Security audit completed
- ✅ **ReBAC (Relationship-Based Access Control)** - Production-ready authorization system
  - Google Zanzibar-inspired design with subject-relation-object tuples
  - Graph-level and dataset-level authorization with inheritance
  - Dual backends: In-memory (O(1), 1M relationships) + RDF-native (SPARQL, 10M relationships)
  - REST API (POST/DELETE/GET/BATCH) + CLI tools (export/import/migrate/verify/stats)
  - Permission implication (Manage → Read/Write/Delete) and conditional relationships
  - Unified RBAC+ReBAC policy engine with 4 modes
  - **83 tests passing** across all ReBAC components
- ✅ Authentication and authorization hardening
- ✅ Input validation and sanitization
- ✅ Rate limiting and DoS protection
- ✅ Secure defaults and best practices

### Next Milestone: v0.2.0 (Q1 2026)

**Focus Areas:**
- 🎯 10x query performance improvements
- 🎯 AI features production hardening
- 🎯 Multi-region clustering
- 🎯 Advanced caching strategies
- 🎯 Performance SLAs and guarantees

## License

OxiRS is dual-licensed under either:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

See [LICENSE](LICENSE) for details.

## Contact

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **Maintainer**: @cool-japan (KitaSan)

## Release Notes (v0.1.0-beta.2)

📄 Full notes live in [CHANGELOG.md](CHANGELOG.md).

### Highlights
- 🚀 **CUDA Support**: GPU acceleration for knowledge graph embeddings and AI operations
- 🧠 **AI Enhancements**: Improved vision-language processing and Tucker decomposition models
- ⚡ **Performance**: Memory-mapped storage optimizations and enhanced SIMD operations
- 🔧 **SAMM Improvements**: Performance regression testing and improved code generation
- 📚 **Documentation**: Updated API docs and examples across all crates
- 🐛 **Bug Fixes**: Various stability improvements and edge case handling

### Known Issues
- Large dataset (>100M triples) performance optimization ongoing
- Full-text search (`oxirs-textsearch`) planned for v0.3.0
- Advanced AI features continue to mature towards v0.2.0

### Quality Metrics (v0.1.0-beta.2)
- ✅ **Zero warnings** - Strict `-D warnings` enforced across all 22 crates
- ✅ **12,248 tests passing** - 100% pass rate (100 skipped)
- ✅ **95%+ test coverage** - Comprehensive test suites
- ✅ **95%+ documentation coverage** - Complete API documentation
- ✅ **CUDA GPU support** - Hardware acceleration for AI
- ✅ **Memory-mapped storage** - Enhanced I/O performance

### Upgrade Notes from Beta.1
- Install the new CLI with `cargo install oxirs --version 0.1.0-beta.2`
- **No Breaking Changes**: All beta.1 APIs remain compatible
- **New Features**: CUDA support is opt-in via feature flags
- **Performance**: Memory-mapped storage provides automatic improvements
- Existing datasets remain fully compatible
- See [CHANGELOG.md](CHANGELOG.md) for detailed migration guide

---

*"Rust makes memory safety table stakes; Oxirs makes knowledge-graph engineering table stakes."*

**Beta.2 release - December 21, 2025**