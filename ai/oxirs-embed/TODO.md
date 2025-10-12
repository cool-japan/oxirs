# OxiRS Embed - TODO

*Last Updated: October 12, 2025*

## ✅ Current Status: v0.1.0-alpha.3 Released (Experimental)

**oxirs-embed** provides vector embeddings for knowledge graphs (experimental feature).

### Alpha.3 Release Status (October 12, 2025)
- **296 tests passing** (unit + integration) with zero warnings
- **Knowledge graph embeddings** integrated with persisted dataset pipelines
- **Multiple embedding models** with provider failover and batch streaming
- **Semantic similarity** surfaced via `vec:` SPARQL SERVICE bindings
- **Telemetry & caching** via SciRS2 metrics and embedding cache
- **Released on crates.io**: `oxirs-embed = "0.1.0-alpha.3"` (experimental)

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Embedding Models
- [ ] Additional model support
- [ ] Fine-tuning capabilities
- [ ] Model selection guidance
- [ ] Performance optimization

#### Features
- [ ] Entity linking
- [ ] Relation prediction
- [ ] Link prediction
- [ ] Clustering support

#### Performance
- [ ] Batch processing
- [ ] GPU acceleration
- [ ] Memory optimization
- [ ] Caching strategies

#### Integration
- [ ] Vector search integration
- [ ] SPARQL extension
- [ ] GraphQL support
- [ ] Storage backend integration

### v0.2.0 Targets (Q1 2026)
- [ ] Multi-modal embeddings
- [ ] Temporal embeddings
- [ ] Transfer learning
- [ ] Production optimization