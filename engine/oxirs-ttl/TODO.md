# OxiRS TTL - TODO

*Version: 0.3.2 | Last Updated: July 12, 2026*

## Status: Production Ready

OxiRS TTL provides streaming RDF parsing and serialization with comprehensive format support and production-ready features.

### Features

- **Turtle Parser** - Full W3C Turtle 1.1 parsing with zero-copy optimization
- **N-Triples Parser** - Streaming N-Triples parsing with Unicode support
- **N-Quads Parser** - Named graph support with N-Quads format
- **TriG Parser** - Named graph support with full Turtle syntax
- **N3 Parser** - Advanced N3 support with formulas, variables, and implications
- **RDF 1.2 Support** - Quoted triples (RDF-star) and directional language tags
- **Streaming Parsing** - Memory-efficient streaming for large files
- **Async I/O** - Tokio-based async parsing and serialization
- **Parallel Processing** - Rayon-based parallel parsing
- **Serialization** - Pretty-printing with predicate grouping and object lists
- **Incremental Parsing** - Parse as bytes arrive with checkpoint resume
- **Format Detection** - Automatic format detection from extension, MIME, and content
- **Error Recovery** - Lenient mode with comprehensive error collection
- **IRI Normalization** - RFC 3987 compliant IRI normalization and resolution
- **N3 Reasoning** - Pattern matching, variable substitution, forward chaining
- **Graph Utilities** - Merging, diff, transformation, and statistics
- **Format Conversion** - Universal converter between all RDF formats
- **Pattern Matching** - SPARQL-like in-memory queries without full engine
- **1779 tests passing** with zero warnings

### Key Capabilities

- Parse and serialize Turtle, N-Triples, N-Quads, TriG, and N3 formats
- Process RDF 1.2 features including quoted triples and directional tags
- Stream large files with configurable batch sizes
- Async I/O for non-blocking operations
- Parallel processing for maximum throughput
- Pretty-print with smart prefix generation
- Validate and normalize IRIs per RFC 3987
- Perform basic N3 reasoning operations
- Convert between all supported formats

### Performance Targets

| Format | Parse Speed | Serialize Speed |
|--------|-------------|-----------------|
| Turtle | 300K triples/s | 200K triples/s |
| N-Triples | 500K triples/s | 400K triples/s |
| TriG | 250K triples/s | 180K triples/s |
| N-Quads | 450K triples/s | 350K triples/s |

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ Full Turtle/N-Triples/N-Quads/TriG/N3 parsing and serialization
- ✅ RDF 1.2 support, streaming, async I/O, parallel processing
- ✅ 1726 tests passing

### v0.2.3 - Released (March 16, 2026)
- ✅ Additional N3 built-in predicates
- ✅ Enhanced error recovery strategies
- ✅ Performance optimizations for edge cases
- ✅ Extended format detection heuristics
- ✅ N3 backward chaining inference
- ✅ Enhanced N3 reasoning capabilities
- ✅ Additional serialization optimizations
- ✅ Extended format conversion options

### v0.3.0 - Released (Q2 2026)
- [x] Full W3C compliance certification (completed 2026-04-30)
  - **Goal:** Pass every test in the W3C RDF 1.1 / 1.2 conformance suite for Turtle, N-Triples, N-Quads, and TriG that the parser/serializer does not yet pass; document any constructs that remain unsupported as typed errors (no silent passes). N3 is best-effort (no W3C conformance suite).
  - **Design:** Vendor W3C RDF test corpus under `tests/fixtures/w3c-rdf-tests/` (pinned commit) covering Turtle, N-Triples, TriG, N-Quads (positive/negative parser tests + eval). Driver `tests/w3c_compliance.rs` reads per-format manifest, dispatches by `mf:Test` type. Cluster failures by category and close in `src/parser/{turtle,nquads,trig,ntriples}.rs` and `src/serializer/`. RDF 1.2 quoted-triple coverage: TriG-star and N-Quads-star round-trip (oxirs-star handles SPARQL-star).
  - **Files:** `src/parser/{turtle,nquads,trig,ntriples,n3}.rs`, `src/serializer/{turtle,nquads,trig,ntriples}.rs`, `tests/w3c_compliance.rs` (new), `tests/fixtures/w3c-rdf-tests/` (vendored, pinned)
  - **Tests:** unit per-parser positive + negative (escapes, IRI refs, blank scope, prefix declarations, language tags, datatype IRIs); integration W3C corpus per format with `pass_rate >= 0.99`; property parse → serialize → parse round-trip stability over N=1000 generated graphs
  - **Risk:** RDF 1.2 spec settling; some tests may flap. Mitigation: pin to specific test-suite commit; document version in test header.
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive benchmarks (completed 2026-04-29)

### v0.3.2 - Current Release (July 12, 2026)
- [x] Maintenance release: no functional/API changes. Workspace-wide dependency modernization (`lazy_static` → `once_cell::sync::Lazy`) and rustdoc intra-doc link fixes

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS TTL v0.3.2 - Streaming RDF parser and serializer*
