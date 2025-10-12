# OxiRS ARQ - TODO

*Last Updated: October 12, 2025*

## ✅ Current Status: v0.1.0-alpha.3 Production-Ready - **Beta.1 Features Complete!** 🎉

**oxirs-arq** provides a SPARQL 1.1/1.2 query engine with optimization.

### Alpha.3 Release Status (October 12, 2025) - **Beta.1 Features Complete!** 🎉
- **228 tests** (unit + integration) passing with zero failures
- **SPARQL 1.1/1.2 support** with persisted dataset awareness
- **Federation (`SERVICE`)** with retries, failover, and JSON result merging
- **SciRS2 instrumentation** powering query metrics and slow-query tracing
- **CLI Integration** ✨ Full parity with `oxirs query` and REPL workflows
- **Query Explainability** ✨ PostgreSQL-style plans (`oxirs explain`) with analyze/full modes
- **Production Tested**: SELECT/ASK/CONSTRUCT/DESCRIBE across persisted stores
- **✨ NEW: Comprehensive SPARQL benchmarking suite** (9 benchmark groups covering all SPARQL operations)
- **✨ NEW: SPARQL stress testing suite** (10 comprehensive tests for edge cases and high load)
- **✨ NEW: Production hardening** (Query circuit breakers, SPARQL performance monitoring, resource quotas, health checks)

### 🎉 Alpha.3 Achievements

#### Federation & Tooling ✅
- ✅ **Query Command Integration**: Full SPARQL execution via CLI & REPL with streaming output
- ✅ **Federated Execution**: Resilient remote endpoint calls with backoff and `SERVICE SILENT`
- ✅ **Instrumentation**: Exposed metrics and tracing hooks through SciRS2
- ✅ **Production Testing**: Validated with 7 integration tests plus federation smoke tests

#### Beta.1 Production Features ✅ (Complete in Alpha.3)
- ✅ **Comprehensive SPARQL Benchmarking** (comprehensive_sparql_bench.rs - 9 benchmark groups)
  - Query parsing performance
  - Pattern matching scalability (100 → 10K triples)
  - Join operations (2-way, 3-way joins)
  - Filter operations (equality, regex, numeric, compound)
  - OPTIONAL pattern performance
  - UNION operations
  - Aggregation operations (COUNT, GROUP BY, HAVING)
  - Query forms (SELECT, ASK, CONSTRUCT, DESCRIBE)
  - Scalability testing (1K → 100K triples)
- ✅ **SPARQL Stress Testing** (stress_tests.rs - 10 comprehensive tests)
  - High volume query execution (10K queries)
  - Complex multi-way joins
  - Concurrent query execution (16 threads × 500 queries)
  - Memory intensive queries
  - Filter performance stress
  - Aggregation performance stress
  - UNION operations stress
  - OPTIONAL patterns stress
  - Sustained query load (30 seconds)
  - Query edge cases
- ✅ **Production Hardening** (production.rs)
  - SPARQL-specific error handling with query context
  - Query execution circuit breakers
  - SPARQL performance monitoring (latency, pattern complexity, result sizes)
  - Query result size quotas
  - Health checks for query engine components
  - Global statistics tracking
- ✅ **SPARQL 1.2 / SPARQL-star Integration** (star_integration.rs)
  - Complete RDF-star / SPARQL-star support via oxirs-star
  - Quoted triple patterns (`<<subject predicate object>>`)
  - SPARQL-star built-in functions (TRIPLE, isTRIPLE, SUBJECT, PREDICATE, OBJECT)
  - Term conversion between ARQ algebra and RDF-star
  - Pattern matching utilities for nested quoted triples
  - Statistics tracking for SPARQL-star queries
  - 6 comprehensive unit tests validating all functionality
  - Working example demonstrating provenance tracking and meta-metadata
  - Comprehensive benchmark suite (sparql_star_bench.rs - 8 benchmark groups)
    - Quoted triple creation and insertion performance
    - Pattern matching by subject/predicate/object
    - Nesting depth queries (depth 0-2 and ranges)
    - SPARQL-star utility functions overhead
    - Term conversion performance (ARQ ↔ RDF-star)
    - Statistics tracking overhead
    - Scalability testing with increasing nesting depth (1-4 levels)
    - Benchmark coverage: 100-10K triples, nesting depths 1-4

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Query Optimization
- [ ] Cost-based optimization
- [ ] Advanced join ordering
- [ ] Filter pushdown improvements
- [ ] Statistics-based cardinality estimation

#### SPARQL Compliance
- [x] Complete SPARQL 1.2 / SPARQL-star support ✨ (Alpha.3)
- [ ] Additional aggregate functions
- [ ] Property path optimization
- [ ] Federated query improvements

#### Performance
- [ ] Parallel query execution
- [ ] Query result streaming
- [ ] Memory-efficient processing
- [ ] Query plan caching

#### Developer Experience
- [x] Query explain and profiling ✨ (Alpha.3)
- [ ] Better error messages
- [ ] Query validation tools
- [ ] Debugging utilities

### v0.2.0 Targets (Q1 2026)
- [ ] Advanced query rewriting
- [ ] Adaptive query execution
- [ ] Query result materialization strategies
- [ ] Integration with distributed storage