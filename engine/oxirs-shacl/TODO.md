# OxiRS SHACL - TODO

*Version: 0.3.2 | Last Updated: July 12, 2026*

## Current Status

OxiRS SHACL v0.3.2 is production-ready, providing W3C SHACL validation with cross-module integration and advanced features.

### Production Features
- ✅ **W3C SHACL Compliance** - 27/27 W3C constraint types passing
- ✅ **SHACL-SPARQL** - Advanced validation with custom SPARQL constraints
- ✅ **Cross-Module Integration** - GraphQL, Fuseki, Stream, AI modules
- ✅ **Shape Quality Metrics** - Complexity, maintainability, performance analysis
- ✅ **Constraint Component Library** - 30+ pre-built validators
- ✅ **CI/CD Integration** - JUnit, TAP, SARIF, JSON output formats
- ✅ **Distributed Validation** - Coordinator-worker architecture
- ✅ **Documentation Generator** - Markdown, HTML, reStructuredText, AsciiDoc
- ✅ **Testing Framework** - Comprehensive test suite organization
- ✅ **ShEx Migration** - ShEx to SHACL migration tool
- ✅ **LSP Integration** - Language Server Protocol for IDE support
- ✅ **Interactive Designer** - Step-by-step shape creation wizard
- ✅ **Subclass-Aware Targeting** - Reflexive+transitive `rdfs:subClassOf` closure so `sh:class`/implicit-class targets honor subclassing
- ✅ **Live SPARQL & Property-Path Targets** - `sh:target` SPARQLTarget and single-hop property-path targets execute against the store
- ✅ **2140 tests passing** with zero warnings

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ W3C SHACL compliance with 27/27 constraint types
- ✅ SHACL-SPARQL, cross-module integration, distributed validation
- ✅ 2008 tests passing

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Advanced constraint types
- ✅ Performance optimization for large datasets
- ✅ Enhanced error reporting
- ✅ Additional integration patterns
- ✅ Constraint marketplace expansion
- ✅ Enhanced LSP features
- ✅ Advanced analytics
- ✅ Production monitoring

### v0.3.0 - Planned (Q2 2026)
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Complete SHACL 1.0 compliance (completed 2026-04-28)
  - 47/47 real conformance tests pass (shacl_core_1_0_conformance.rs)
  - Covers all 27 W3C SHACL Core constraint types: class, datatype, nodeKind, minCount, maxCount, minInclusive, maxInclusive, minLength, maxLength, pattern, languageIn, uniqueLang, equals, disjoint, lessThan, in, hasValue, and, or, not, xone, qualifiedValueShape, closed
  - Removed all eprintln! debug prints from shape_constraints.rs, validation/engine.rs, targets/selector.rs (replaced with tracing::trace!)
  - Fixed logical constraints (and/or/not/xone) to use shapes_registry for real sub-shape validation
  - 2040 total tests passing, zero clippy warnings
- [x] Enterprise validation features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive documentation (completed 2026-04-30)
  - **Goal:** Comprehensive documentation pass on oxirs-shacl: rustdoc enrichment + cookbook + spec mapping.
  - **Design:** Comprehensive rustdoc on every public API in `src/lib.rs` and submodules. Cookbook `COOKBOOK.md` covering common shape patterns (cardinality, string constraints, datatype, qualified value, target chains, SHACL-AF SPARQL constraints). Spec mapping `SPEC_MAPPING.md` listing every SHACL Core / SHACL-AF construct with the symbol that implements it.
  - **Files:** `src/lib.rs`, `src/constraints/comparison_constraints.rs`, `src/constraints/shape_constraints.rs`, `src/constraints/logical_constraints.rs`, `src/paths/mod.rs`, `src/sparql_af/mod.rs` (rustdoc enrichment), `COOKBOOK.md` (new), `SPEC_MAPPING.md` (new)
  - **Tests:** `cargo doc --no-deps -p oxirs-shacl` warning-free; 29 doctests pass; 2040/2040 unit tests pass; clippy clean; fmt clean.
  - **Risk:** docs drift from code. Mitigation: spec-mapping table is generated from a single doctest harness later (out of scope for this run; spec-mapping is hand-authored and reviewed).

### v0.3.2 - Current Release (July 12, 2026)
- [x] Subclass-aware target closure (completed 2026-07-11)
  - **Goal:** `sh:class` and implicit-class targets only matched exact `rdf:type` values; a resource typed as a subclass of the target class was silently skipped.
  - **Design:** `advanced_features::subclass_closure` computes the reflexive+transitive closure of `rdfs:subClassOf` over a boolean adjacency matrix via Floyd–Warshall, then target evaluation consults the closure instead of exact-type matching.
  - **Files:** `src/advanced_features/subclass_closure.rs` (new, 201 lines), `src/advanced_features/mod.rs`, `src/advanced_features/advanced_targets.rs`, `src/constraints/value_constraints.rs`.
- [x] Live SPARQL-based and property-path targets (completed 2026-07-11)
  - **Goal:** `sh:target` SPARQLTarget and single-hop property-path targets were stub implementations that returned empty or unchanged results.
  - **Design:** Both target kinds now execute for real against the backing store during target selection.
  - **Files:** `src/advanced_features/advanced_targets.rs`, `src/advanced_features/shape_inference.rs`.
- [x] `sh:languageIn` BCP47/RFC-4647 matching (completed 2026-07-11)
  - **Goal:** `sh:languageIn` compared language tags with exact string equality, so `"de"` did not match a value tagged `"de-CH"`.
  - **Design:** Basic-filtering range matching per BCP47/RFC-4647 §3.3.
  - **Files:** `src/constraints/string_constraints.rs`.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS SHACL v0.3.2 - W3C SHACL validation engine*
