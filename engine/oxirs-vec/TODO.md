# OxiRS Vec - TODO

*Last Updated: October 12, 2025*

## ✅ Current Status: v0.1.0-alpha.3 Released (Experimental)

**oxirs-vec** provides vector search infrastructure for semantic similarity (experimental feature).

### Alpha.3 Release Status (October 12, 2025)
- **323 tests passing** (unit + integration) with zero compilation warnings
- **Vector indexing** with persisted storage and streaming ingestion pipelines
- **Similarity search** exposed via SPARQL `vec:` SERVICE bindings and GraphQL filters
- **Embedding integrations** expanded with CLI batch tooling & SciRS2 telemetry
- **Observability** hooks for index health and slow-query tracing
- **Released on crates.io**: `oxirs-vec = "0.1.0-alpha.3"` (experimental)

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Performance
- [ ] HNSW index optimization
- [ ] Approximate nearest neighbor improvements
- [ ] Memory usage optimization
- [ ] Query performance tuning

#### Features
- [ ] Multiple distance metrics
- [ ] Dynamic index updates
- [ ] Filtered search
- [ ] Batch operations

#### Integration
- [ ] SPARQL vector search extension
- [ ] GraphQL vector queries
- [ ] Embedding model integration
- [ ] Storage backend integration

#### Stability
- [ ] Index persistence
- [ ] Crash recovery
- [ ] Data validation
- [ ] Comprehensive testing

### v0.2.0 Targets (Q1 2026)
- [ ] GPU acceleration
- [ ] Distributed vector search
- [ ] Advanced indexing algorithms
- [ ] Hybrid search support