# OxiRS ARQ - TODO

*Last Updated: October 4, 2025*

## ✅ Current Status: v0.1.0-alpha.2 Production-Ready

**oxirs-arq** provides a SPARQL 1.1/1.2 query engine with optimization.

### Alpha.2 Release Status (October 4, 2025)
- **114 tests** (unit + integration) with zero compilation warnings
- **SPARQL 1.1/1.2 support** with persisted dataset awareness
- **Federation (`SERVICE`)** with retries, failover, and JSON result merging
- **SciRS2 instrumentation** powering query metrics and slow-query tracing
- **CLI Integration** ✨ Full parity with `oxirs query` and REPL workflows
- **Production Tested**: SELECT/ASK/CONSTRUCT/DESCRIBE across persisted stores

### 🎉 Alpha.2 Achievements

#### Federation & Tooling ✅
- ✅ **Query Command Integration**: Full SPARQL execution via CLI & REPL with streaming output
- ✅ **Federated Execution**: Resilient remote endpoint calls with backoff and `SERVICE SILENT`
- ✅ **Instrumentation**: Exposed metrics and tracing hooks through SciRS2
- ✅ **Production Testing**: Validated with 7 integration tests plus federation smoke tests

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Query Optimization
- [ ] Cost-based optimization
- [ ] Advanced join ordering
- [ ] Filter pushdown improvements
- [ ] Statistics-based cardinality estimation

#### SPARQL Compliance
- [ ] Complete SPARQL 1.2 support
- [ ] Additional aggregate functions
- [ ] Property path optimization
- [ ] Federated query improvements

#### Performance
- [ ] Parallel query execution
- [ ] Query result streaming
- [ ] Memory-efficient processing
- [ ] Query plan caching

#### Developer Experience
- [ ] Query explain and profiling
- [ ] Better error messages
- [ ] Query validation tools
- [ ] Debugging utilities

### v0.2.0 Targets (Q1 2026)
- [ ] Advanced query rewriting
- [ ] Adaptive query execution
- [ ] Query result materialization strategies
- [ ] Integration with distributed storage