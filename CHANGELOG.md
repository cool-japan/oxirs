# Changelog

All notable changes to OxiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.1] - 2025-09-30

### Overview

First alpha release of OxiRS - a Rust-native semantic web platform with SPARQL 1.2, GraphQL, and AI capabilities.

⚠️ **Alpha Software**: This is an early alpha release. APIs may change without notice. Not recommended for production use.

### Added

#### Core Platform
- **oxirs-core**: Native RDF/SPARQL implementation extracted from OxiGraph
  - Zero-dependency RDF data model
  - Complete RDF 1.2 support (Turtle, N-Triples, JSON-LD, RDF/XML)
  - SPARQL 1.1 query engine
  - 519 tests passing (99.1% success rate)

- **oxirs-fuseki**: SPARQL 1.2 HTTP server
  - Basic SPARQL query endpoint
  - SPARQL update endpoint (basic)
  - Multi-dataset support (basic)
  - Jena Fuseki compatibility layer
  - 349 tests passing

- **oxirs-arq**: SPARQL query engine with optimization
  - SPARQL 1.1 parsing and execution
  - Query optimization passes
  - Custom function framework
  - 114 tests passing

- **oxirs-gql**: GraphQL interface
  - Basic GraphQL server
  - Schema generation from RDF vocabularies (basic)
  - GraphQL to SPARQL translation (basic)
  - 118 tests passing

#### Storage & Data Management
- **oxirs-tdb**: High-performance RDF storage
  - MVCC (Multi-Version Concurrency Control)
  - ACID transactions
  - B+ Tree indexing
  - TDB2 compatibility

- **oxirs-cluster**: Distributed storage (experimental)
  - Raft consensus implementation
  - Distributed RDF storage
  - High availability features

#### Semantic Web Extensions
- **oxirs-shacl**: SHACL validation framework
  - Core constraint types
  - Property path support
  - Basic validation engine

- **oxirs-star**: RDF-star support (experimental)
  - Quoted triple support
  - RDF-star parsing
  - SPARQL-star queries

- **oxirs-rule**: Rule-based reasoning (experimental)
  - RDFS reasoning
  - Basic rule engine

#### AI & Machine Learning (Experimental)
- **oxirs-chat**: RAG and natural language interface
  - Natural language to SPARQL conversion
  - LLM integration (OpenAI, Anthropic, Ollama)
  - Basic RAG capabilities

- **oxirs-embed**: Vector embeddings
  - Knowledge graph embeddings
  - Semantic similarity search

- **oxirs-vec**: Vector search infrastructure
  - Vector indexing
  - Similarity search

#### Tools & Utilities
- **oxirs**: Command-line interface
  - Data import/export
  - Query execution
  - Basic server management

### Known Limitations

- **Alpha Quality**: Not recommended for production use
- **API Stability**: APIs subject to change without notice
- **Performance**: Not yet fully optimized
- **Features**: Some advanced features incomplete or experimental
- **Documentation**: Comprehensive documentation in progress
- **Error Handling**: Limited error handling in some areas

### Performance

- **Test Coverage**: 3,740 tests passing (99.8% success rate)
- **Code Size**: ~845k lines of Rust code across 21 crates
- **Build Status**: Clean compilation without errors/warnings

### Technical Highlights

- **Zero OxiGraph Dependency**: Successfully eliminated external OxiGraph dependency
- **Native Implementation**: Pure Rust implementation of RDF/SPARQL
- **SciRS2 Integration**: Full integration with SciRS2 scientific computing framework
- **Modular Architecture**: 21-crate workspace with clear separation of concerns
- **Type Safety**: Leveraging Rust's type system for correctness

### Platform Support

- **Operating Systems**: Linux, macOS, Windows
- **Rust Version**: 1.70+ (MSRV)
- **Architecture**: x86_64, aarch64

### Installation

Available on crates.io:

```toml
[dependencies]
oxirs-core = "0.1.0-alpha.1"
oxirs-fuseki = "0.1.0-alpha.1"
oxirs-arq = "0.1.0-alpha.1"
oxirs-gql = "0.1.0-alpha.1"
# ... other crates as needed
```

Or install the CLI tool:

```bash
cargo install oxirs
```

### Next Steps (Beta Release - Q4 2025)

- API stability and freeze
- Production hardening
- Performance optimization
- Complete documentation
- Security audit
- Full test coverage
- Migration guides

### Contributors

- @cool-japan (KitaSan) - Project lead and primary developer

### Links

- **Repository**: https://github.com/cool-japan/oxirs
- **Issues**: https://github.com/cool-japan/oxirs/issues
- **Documentation**: Coming soon

---

## Release Notes Format

### [Version] - YYYY-MM-DD

#### Added
- New features

#### Changed
- Changes in existing functionality

#### Deprecated
- Soon-to-be removed features

#### Removed
- Removed features

#### Fixed
- Bug fixes

#### Security
- Security improvements

---

*First alpha release - September 30, 2025*