# OxiRS-Star - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Status: Production Ready

OxiRS-Star provides RDF-star and SPARQL-star support for quoted triples with enterprise-ready features.

### Features

- **RDF-star Parsing** - Turtle-star, N-Triples-star, TriG-star, N-Quads-star, JSON-LD-star formats
- **SPARQL-star Query Engine** - Full SPARQL-star query support with extensions
- **Quoted Triple Store** - Efficient storage and indexing for quoted triples
- **Annotation Support** - Statement-level annotations and metadata
- **Adaptive Query Optimization** - ML, quantum-inspired, classical, and hybrid strategies
- **HDT-star Format** - Binary serialization with dictionary compression
- **Streaming Query Processor** - Continuous queries with window types
- **Property Graph Bridge** - RDF-star to Neo4j/Cypher conversion
- **SHACL-star Validation** - Shape validation for quoted triples
- **GraphQL Integration** - GraphQL API for RDF-star data
- **Reasoning Engine** - RDFS and OWL 2 RL support
- **Cryptographic Provenance** - Secure triple signing and verification
- **Production Features** - Circuit breaker, rate limiter, health monitoring
- **1628 tests passing** with zero warnings

### Key Capabilities

- Parse and serialize quoted triples in all major RDF formats
- Execute SPARQL-star queries with nested quoted triples
- Adaptive optimization with auto-tuning based on workload
- Real-time streaming queries with CEP pattern matching
- Bidirectional LPG/RDF-star conversion with Neo4j support
- Memory-efficient storage with SIMD-accelerated indexing
- Full SciRS2 integration for scientific computing

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ RDF-star/SPARQL-star support with all major formats
- ✅ Adaptive query optimization, HDT-star, streaming, property graph bridge
- ✅ 1628 tests passing

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Advanced HNSW graph optimizations
- ✅ Enhanced parallel query execution
- ✅ Improved cache management
- ✅ Additional storage backends
- ✅ Extended HDT-star compression options
- ✅ Enhanced streaming query capabilities
- ✅ Advanced property graph integration
- ✅ Distributed query processing

### v0.3.0 - Planned (Q2 2026)
- [x] Full W3C RDF-star compliance (completed 2026-04-28)
  - **Goal:** Pass W3C RDF 1.2/RDF-star conformance suite for parsing, querying, and serializing quoted triples.
  - **Design:** Vendor W3C test suite fixtures → driver reads manifest → dispatch per-test → fix parser/serializer/query gaps (asserted vs unasserted semantics, SPARQL-star in BIND/FILTER/OPTIONAL, blank-node scope in quoted triples, round-trip NQ-star/TriG-star).
  - **Files:** src/parser/{turtle_star,trig_star,nquads_star}.rs, src/sparql_star/query_executor.rs, src/serializer.rs, tests/w3c_rdf_star_conformance.rs (new)
  - **Tests:** W3C corpus driver pass_rate >= 0.99; parse→serialize→parse round-trip property test
  - **Risk:** RDF 1.2 spec still settling — pin to specific commit hash
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive benchmarks (completed 2026-04-29)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS-Star v0.2.3 - RDF-star and SPARQL-star support*
