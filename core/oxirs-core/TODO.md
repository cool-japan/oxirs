# OxiRS Core - TODO

*Last Updated: October 12, 2025*

## ✅ Current Status: v0.1.0-alpha.3 Production-Ready

**oxirs-core** is the foundational module providing native RDF/SPARQL implementation.

### Alpha.3 Release Status (October 12, 2025) - **Beta.1 Features Complete!** 🎉
- **Persistent RDF pipeline** with automatic N-Quads save/load
- **Streaming import/export/migrate** covering all 7 serialization formats
- **SciRS2 instrumentation** for metrics, tracing, and slow-query diagnostics
- **Federation-ready SPARQL algebra** powering `SERVICE` clause execution
- **4,421 tests passing** (unit + integration) with zero compilation warnings
- **Zero-dependency RDF/SPARQL implementation** with concurrent operations
- **✨ NEW: Query plan introspection hooks** consumed by `oxirs explain`
- **✨ NEW: Comprehensive benchmarking suite** (8 benchmark groups covering all operations)
- **✨ NEW: Production hardening** (Circuit breakers, health checks, resource quotas, performance monitoring)

### 🎉 Alpha.3 Achievements

#### Persistence & Streaming ✅
- ✅ **Automatic Persistence**: Disk-backed N-Quads serializer/loader integrated with CLI/server
- ✅ **Streaming Pipelines**: Multi-threaded importer/exporter with progress instrumentation
- ✅ **Federated Execution Hooks**: Core algebra enhancements supporting remote `SERVICE` calls

#### Code Quality ✅
- ✅ **4,421 tests** spanning persistence, streaming, and federation flows
- ✅ **Continuous benchmarking** with SciRS2 telemetry
- ✅ **W3C RDF/SPARQL compliance** verified against reference suites

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Performance Optimization ✅ (Complete in Alpha.3)
- [x] Query execution performance improvements (optimization module)
- [x] Memory usage optimization (RdfArena, zero-copy operations)
- [x] Index structure enhancements (OptimizedGraph with multi-index)
- [x] Parallel query processing improvements (BatchProcessor, concurrent operations)

#### API Stability 🚧 (In Progress)
- [x] Production-ready error handling (ProductionError with context)
- [x] Comprehensive monitoring (PerformanceMonitor, HealthCheck)
- [ ] API freeze and stability guarantees
- [ ] Comprehensive API documentation
- [ ] Migration guides from alpha

#### Feature Enhancements ✅ (Complete in Alpha.3)
- [x] Additional RDF serialization formats (7 formats complete: alpha.3)
- [x] N-Triples/N-Quads parsing implementation (alpha.3)
- [x] Turtle parser implementation (complete with serialization)
- [x] Improved error messages and debugging (ProductionError with detailed context)

#### Testing & Quality ✅ (Complete in Alpha.3)
- [x] Performance benchmarking suite (comprehensive_bench.rs - 8 benchmark groups)
- [x] Stress testing and edge cases (stress_tests.rs - 10 comprehensive tests)
- [x] Production hardening (Circuit breaker, resource quotas, health checks)

### v0.2.0 Targets (Q1 2026)
- [ ] Advanced query optimization
- [ ] Full SPARQL 1.2 compliance
- [ ] Enhanced concurrency support
- [ ] Production-scale performance tuning