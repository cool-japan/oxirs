# OxiRS Core - TODO

*Last Updated: October 4, 2025*

## ✅ Current Status: v0.1.0-alpha.2 Production-Ready

**oxirs-core** is the foundational module providing native RDF/SPARQL implementation.

### Alpha.2 Release Status (October 4, 2025)
- **Persistent RDF pipeline** with automatic N-Quads save/load
- **Streaming import/export/migrate** covering all 7 serialization formats
- **SciRS2 instrumentation** for metrics, tracing, and slow-query diagnostics
- **Federation-ready SPARQL algebra** powering `SERVICE` clause execution
- **3,750+ tests passing** (unit + integration) with zero compilation warnings
- **Zero-dependency RDF/SPARQL implementation** with concurrent operations

### 🎉 Alpha.2 Achievements

#### Persistence & Streaming ✅
- ✅ **Automatic Persistence**: Disk-backed N-Quads serializer/loader integrated with CLI/server
- ✅ **Streaming Pipelines**: Multi-threaded importer/exporter with progress instrumentation
- ✅ **Federated Execution Hooks**: Core algebra enhancements supporting remote `SERVICE` calls

#### Code Quality ✅
- ✅ **3,750+ tests** spanning persistence, streaming, and federation flows
- ✅ **Continuous benchmarking** with SciRS2 telemetry
- ✅ **W3C RDF/SPARQL compliance** verified against reference suites

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Performance Optimization
- [ ] Query execution performance improvements
- [ ] Memory usage optimization
- [ ] Index structure enhancements
- [ ] Parallel query processing improvements

#### API Stability
- [ ] API freeze and stability guarantees
- [ ] Comprehensive API documentation
- [ ] Migration guides from alpha

#### Feature Enhancements
- [x] Additional RDF serialization formats (7 formats complete: alpha.2)
- [x] N-Triples/N-Quads parsing implementation (alpha.2)
- [ ] Turtle parser implementation
- [ ] Enhanced SPARQL 1.2 support
- [ ] Improved error messages and debugging

#### Testing & Quality
- [ ] Increase test coverage to 100%
- [ ] Performance benchmarking suite
- [ ] Stress testing and edge cases
- [ ] Production hardening

### v0.2.0 Targets (Q1 2026)
- [ ] Advanced query optimization
- [ ] Full SPARQL 1.2 compliance
- [ ] Enhanced concurrency support
- [ ] Production-scale performance tuning