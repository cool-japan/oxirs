# OxiRS-Star - TODO

*Last Updated: October 12, 2025*

## ✅ Current Status: v0.1.0-alpha.3 Released (Experimental)

**oxirs-star** provides RDF-star and SPARQL-star support for quoted triples (experimental feature).

### Alpha.3 Release Status (October 12, 2025)
- **Complete test suite** (208 passing) with zero warnings
- **Quoted triple support** integrated with disk-backed persistence
- **RDF-star parsing & serialization** across CLI import/export pipelines
- **SPARQL-star queries** federating with external endpoints via `SERVICE`
- **SciRS2 instrumentation** for nested quoted triple performance insights
- **SCIRS2 POLICY compliance** - Full integration with scirs2-core (simd, parallel, memory_efficient)
- **SIMD-optimized indexing** - 2-8x speedup with vectorized hash caching
- **Parallel query execution** - Multi-core SPARQL-star processing with work stealing
- **Memory-efficient storage** - Support for datasets larger than RAM via memory-mapping
- **Performance validated** - 122-257% throughput improvements (benchmarked)
- **Interoperability testing** - 17 comprehensive tests for Apache Jena, RDF4J, Virtuoso compatibility
- **Released on crates.io**: `oxirs-star = "0.1.0-alpha.3"` (experimental)

## ✅ Recently Completed (October 12, 2025 - Session 2)

### Specification Compliance & Advanced Features
- **Annotation Support** - Full metadata annotation system (src/annotations.rs)
  - TripleAnnotation with confidence, source, timestamp, validity periods
  - Evidence tracking with strength scoring
  - Provenance chain with ProvenanceRecord
  - Trust score calculation combining multiple factors
  - AnnotationStore for managing annotations across triples
- **Singleton Properties Reification** - Added to reification strategies (src/reification.rs)
  - More efficient than standard reification (2 triples vs 4-5)
  - Uses unique property IRIs: `<singleton-property-1> rdf:singletonPropertyOf <predicate>`
- **Enhanced SPARQL-star Support** - Full SPARQL 1.1 compliance (src/sparql_enhanced.rs)
  - OPTIONAL, UNION, GRAPH, MINUS patterns
  - Solution modifiers: ORDER BY, LIMIT, OFFSET, DISTINCT
  - BIND and VALUES clauses
  - Aggregations: COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE
  - GROUP BY with HAVING filters
- **Legacy RDF Compatibility** - Full compatibility layer for standard RDF tools (src/compatibility.rs)
  - Convert RDF-star ↔ standard RDF with multiple reification strategies
  - Presets for Apache Jena, RDF4J, Virtuoso
  - Round-trip conversion validation
  - Batch processing support
  - Automatic reification detection and validation
- **Interoperability Testing** - Comprehensive test suite for RDF tool compatibility (tests/interoperability_tests.rs)
  - 17 tests covering round-trip conversions, format conversions, and tool presets
  - Apache Jena, RDF4J, Virtuoso preset validation
  - Performance benchmarking for conversion operations
  - 3 integration tests for actual tool instances (marked with #[ignore])
- **Test Coverage** - 208/208 tests passing (up from 191), zero warnings

### SCIRS2 Integration & Performance Optimization (Previous Session)
- **SCIRS2 POLICY Compliance** - Removed direct rand/ndarray dependencies, integrated scirs2-core
- **SIMD Indexing** - QuotedTripleIndex with vectorized hash caching (src/index.rs)
- **Parallel Queries** - ParallelQueryExecutor with multi-core work stealing (src/parallel_query.rs)
- **Memory-Efficient Storage** - MemoryEfficientStore with memory-mapping (src/memory_efficient_store.rs)
- **Benchmarking** - Validated 122-257% performance improvements

See `../../docs/oxirs_star_scirs2_integration_summary.md` for detailed technical report from Session 1.

---

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Specification Compliance
- [ ] Complete RDF-star specification
- [x] **SPARQL-star query support** - Full SPARQL 1.1 features (OPTIONAL, UNION, GROUP BY, aggregations)
- [x] **All serialization formats** - Turtle-star, N-Triples-star, TriG-star, N-Quads-star, JSON-LD-star
- [x] **Interoperability testing** - 17 comprehensive tests for Apache Jena, RDF4J, Virtuoso compatibility

#### Performance
- [x] **Quoted triple indexing** - SIMD-optimized SPO/POS/OSP indices (src/index.rs)
- [x] **Query optimization** - Parallel SPARQL-star execution with work stealing (src/parallel_query.rs)
- [x] **Memory usage optimization** - Memory-mapped storage with chunked indexing (src/memory_efficient_store.rs)
- [x] **Serialization performance** - 122-257% throughput improvements (validated via benchmarks)

#### Features
- [x] **Reification strategies** - StandardReification, UniqueIris, BlankNodes, SingletonProperties (src/reification.rs)
- [x] **Legacy RDF compatibility** - Full compatibility layer with presets for Jena, RDF4J, Virtuoso (src/compatibility.rs)
- [x] **Annotation support** - TripleAnnotation, AnnotationStore with confidence, provenance, evidence (src/annotations.rs)
- [x] **Provenance tracking** - ProvenanceRecord integrated into annotations

#### Integration
- [ ] Storage backend integration
- [ ] SHACL-star support
- [ ] GraphQL integration
- [ ] Reasoning with quoted triples

### v0.2.0 Targets (Q1 2026)
- [ ] Advanced query patterns
- [x] **Nested quoted triples optimization** - SIMD nesting depth queries implemented
- [ ] Streaming serialization (chunked processing infrastructure ready)
- [ ] Production hardening