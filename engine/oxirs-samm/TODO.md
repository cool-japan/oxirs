# OxiRS SAMM - TODO

*Version: 0.3.2 | Last Updated: July 12, 2026*

## Status: Production Ready

OxiRS SAMM provides comprehensive support for the Semantic Aspect Meta Model (SAMM) and Asset Administration Shell (AAS), enabling Industry 4.0 digital twin modeling.

### Features

- **SAMM Parser** - Full SAMM/AAS model parsing with validation
- **Code Generation** - 16 built-in generators (Rust, TypeScript, Java, Python, JSON Schema, etc.)
- **Model Validation** - Comprehensive validation with detailed error reporting
- **Model Analytics** - Quality metrics, complexity analysis, correlation analysis
- **SIMD Operations** - Accelerated batch processing with SIMD optimization
- **Plugin System** - Extensible architecture for custom generators and validators
- **Extension Support** - User-defined extensions for model enrichment
- **Incremental Parsing** - Efficient parsing for large model files
- **Documentation Generation** - HTML, Markdown, JSON output formats
- **Cloud Storage** - Trait-based abstraction for S3, GCS, Azure integration
- **Graph Analytics** - Dependency analysis with scirs2-graph integration
- **Graph Visualization** - DOT format generation with Graphviz rendering
- **SciRS2 Integration** - Full compliance with SciRS2 policy
- **1549 tests passing** with zero warnings

### Key Capabilities

- Parse and validate SAMM aspect models
- Generate code in multiple programming languages
- Analyze model quality and complexity
- Compute property correlations using statistical methods
- Detect circular dependencies and design issues
- Generate comprehensive documentation
- Store and retrieve models from cloud storage
- Visualize model relationships

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ SAMM/AAS parsing, 16 code generators, model analytics, SIMD, plugin system
- ✅ 1409 tests passing

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Additional correlation analysis methods (Spearman, Kendall)
- ✅ Partial correlation analysis
- ✅ Distribution fitting for model metrics
- ✅ Time-series analysis for model evolution
- ✅ Real AWS S3 backend implementation
- ✅ Google Cloud Storage backend
- ✅ Azure Blob Storage backend
- ✅ Presigned URL generation for sharing

### v0.3.0 - Planned (Q2 2026)
- [x] GPU-accelerated batch validation
- [x] Parallel code generation
- [x] Batch correlation matrix computation
- [x] ESMF SDK feature-parity matrix and status report (completed 2026-05-01)
  - **Goal:** Build a *parity matrix* — a structured catalogue of every documented ESMF SDK 2.x feature tagged `implemented` / `partial` / `missing` against current `oxirs-samm`. Emit a markdown report. Surface the top-3 gaps as new TODO.md items. Does not close gaps — inventories them.
  - **Design:** New module `engine/oxirs-samm/src/parity/` with `mod.rs`, `matrix.rs`, `report.rs`. `ParityMatrix` data type — `HashMap<FeatureCategory, Vec<FeatureEntry>>`. Hand-curated `esmf_catalog.toml` (ESMF SDK 2.x core: aspect modeling, validation, code-gen, OpenAPI emission, JSON-LD profiles). `report::generate_report()` produces markdown for `docs/esmf_parity.md`. New bin `src/bin/parity_report.rs`. New test `tests/parity_test.rs` — round-trips TOML, validates `oxirs_module` references resolve to real paths in this crate. Top-3 `missing` entries appended as new `[ ]` items to this TODO.md.
  - **Files:** `engine/oxirs-samm/src/parity/{mod.rs,matrix.rs,report.rs}`, `src/parity/esmf_catalog.toml`, `src/bin/parity_report.rs`, `docs/esmf_parity.md`, `tests/parity_test.rs`, `src/lib.rs` (re-export `pub mod parity`), `Cargo.toml` (bin entry), `TODO.md` (appended gap list).
  - **Prerequisites:** `toml`, `serde` already in workspace.
  - **Tests:** TOML round-trip; catalog references resolve; report contains every category; missing-status rows generate proper TODO entries; status enum exhaustiveness.
  - **Risk:** ESMF SDK feature list is large. Mitigation: scope to ESMF SDK 2.x core only; profile add-ons deferred.
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive benchmarks (completed 2026-04-29)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS SAMM v0.3.2 - Semantic Aspect Meta Model support*

## Proposed follow-ups

- ESMF parity gap closure: after the parity matrix identifies top-3 missing features, each gap is one dedicated plan round.

## Identified ESMF parity gaps

Top-3 `Missing` features surfaced by the ESMF SDK 2.x parity matrix (see `docs/esmf_parity.md`):

- [x] **Either characteristic** — implement the `samm-c:Either` union-type characteristic in `metamodel::characteristic` so that aspects can model properties whose value may be one of two distinct types. Reference: <https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#either-characteristic>
- [x] **CLI `generate` command** — wire the existing code generators (Java, TypeScript, Python) to an oxirs CLI sub-command `samm generate` so that developers can invoke code generation without writing Rust code. Reference: <https://eclipse-esmf.github.io/esmf-sdk/2.9.7/samm-cli.html#generate> (completed 2026-05-01)
- [x] **OpenAPI 3.1 schema generation** — extend `codegen::openapi` to emit OpenAPI 3.1 specifications aligned with JSON Schema 2020-12, enabling validation of the emitted document with modern tools. Reference: <https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#open-api-generation>

## New ESMF parity gaps (surfaced 2026-05-17)

Top-3 `Missing` features from the updated ESMF SDK 2.x parity matrix (see `docs/esmf_parity.md`):

- [ ] **Cross-model reference validation** — validate external URN references that span independently loaded SAMM model files, so that `oxirs-samm` can report errors when a property references a type defined in a separately resolved model. Affects: `validator`. Reference: <https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#cross-model-reference>
- [ ] **Pagination extension in OpenAPI output** — emit the `x-samm-pagination` extension block in OpenAPI specs generated from Collection-type aspects, enabling pagination-aware clients. Affects: `codegen::openapi`. Reference: <https://eclipse-esmf.github.io/esmf-sdk/2.9.7/java-aspect-tooling.html#open-api-generation>
- [x] **JSON-LD compaction / framing** — implemented the JSON-LD 1.1 compaction algorithm and framing API so that emitted JSON-LD documents can be reshaped into application-specific frames. Landed as `src/jsonld/compaction.rs` (IRI compaction against a `@context`) and `src/jsonld/framing.rs` (reshapes a flat `@graph` into a nested, application-specific tree), rather than under the originally anticipated `generators::jsonld`/`serializer::jsonld` paths. Reference: <https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#mapping-to-json-ld>
