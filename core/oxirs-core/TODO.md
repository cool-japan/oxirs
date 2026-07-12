# OxiRS Core - TODO

*Version: 0.3.2 | Last Updated: July 11, 2026*

## Current Status

OxiRS Core v0.3.2 is production-ready, providing the foundation for semantic web operations with complete RDF/SPARQL support.

### Production Features
- ✅ RDF 1.2 data model with 7 format parsers
- ✅ SPARQL 1.1/1.2 query engine with adaptive optimization
- ✅ Persistent storage with N-Quads serialization
- ✅ ML-based query optimization with adaptive learning
- ✅ Federation support with SERVICE clause execution
- ✅ SciRS2 integration for scientific computing
- ✅ Pure-`std` RFC 3986 percent-encoding (`encoding` module, replaces the external `urlencoding` crate; backs SPARQL `ENCODE_FOR_URI()`)
- ✅ 2589 tests passing (100% pass rate)

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ RDF 1.2 data model with 7 format parsers
- ✅ SPARQL 1.1/1.2 query engine with adaptive optimization
- ✅ Federation support with SERVICE clause execution
- ✅ SciRS2 integration, 850 tests passing

### v0.2.3 - Released (March 16, 2026)
- ✅ Cost-based optimizer enhancements
- ✅ Advanced join ordering strategies
- ✅ Query result caching with invalidation
- ✅ Incremental view maintenance
- ✅ Parallel query execution
- ✅ Distributed storage backend
- ✅ Horizontal sharding support
- ✅ Advanced indexing strategies
- ✅ 2332 tests passing (100% pass rate)

### v0.3.0 - Released (May 3, 2026)
- [x] API stability guarantees (planned 2026-05-01)
  - **Goal:** Concrete engineering interpretation: a programmatic public-API
    surface baseline that the test suite enforces, blocking unintentional
    breaking changes. Per CLAUDE.md IMPLEMENT POLICY, choosing the bold
    engineering interpretation rather than punting on the policy ambiguity.
  - **Design:**
    - New module `core/oxirs-core/src/api_surface.rs`:
      - `ApiSurface { types: Vec<TypeSig>, fns: Vec<FnSig>, traits: Vec<TraitSig>, modules: Vec<String> }`.
      - `parse_lib(path: &Path) -> Result<ApiSurface>` using the existing
        `syn` dep (already in workspace) — walks `pub` items in `lib.rs` +
        `pub use` re-exports.
    - `core/oxirs-core/api_baseline.json` — committed snapshot of the current
      public surface, generated initially by the bin tool below.
    - New bin: `core/oxirs-core/src/bin/api_snapshot.rs` — emits a JSON
      snapshot. Manual workflow: maintainer runs it after intentional API
      changes, commits the new baseline.
    - New test: `core/oxirs-core/tests/api_stability.rs`
      - Loads baseline, parses current source, computes the diff.
      - Pass iff: no removed item AND no modified signature AND no
        relocation. Additions allowed.
      - Failure prints the offending diff with file:line refs.
  - **Files:** `core/oxirs-core/src/api_surface.rs`,
    `core/oxirs-core/src/bin/api_snapshot.rs`,
    `core/oxirs-core/api_baseline.json` (initial snapshot),
    `core/oxirs-core/tests/api_stability.rs`,
    `core/oxirs-core/Cargo.toml` (bin entry).
  - **Prerequisites:** `syn` (already in workspace), `serde_json` (already in workspace).
  - **Tests:** parse a fixture lib with known signatures → verify surface;
    diff a known modified surface → verify it's flagged as breaking;
    additive change → verify it's allowed; baseline round-trip JSON.
  - **Risk:** false positives on internal moves. Mitigation: only walk `pub`
    items, ignore `pub(crate)` / `#[doc(hidden)]`. Document the bypass workflow
    (regenerate baseline) in the test failure message.
- [x] Long-term support commitments (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Performance SLAs (planned 2026-05-01)
  - **Goal:** Concrete engineering interpretation: criterion-based benchmark
    suite + stored baseline + regression-detection harness. Three named SLOs
    that ride the test suite (runs only with `--ignored` to avoid CI-perf
    noise). Per CLAUDE.md IMPLEMENT POLICY, build the engineering machinery
    rather than waiting for the policy text.
  - **Design:**
    - New module `core/oxirs-core/src/perf_sla.rs`:
      - `SloTarget { name, p50_us, p99_us, throughput_ops_s, allow_regression_pct: f64 }`.
      - `BenchmarkResult { name, p50_us, p99_us, throughput_ops_s, samples }`.
      - `assert_meets_slo(result, target) -> Result<(), SloViolation>`.
    - New benches `core/oxirs-core/benches/sla_suite.rs` (criterion):
      - `sla_parse_ntriples_100k` — target: <1.0s.
      - `sla_term_equality` — target: <100ns/op median.
      - `sla_dataset_insert_1m` — target: <10s for 1M triples.
    - Baseline `core/oxirs-core/perf_baseline.json` — committed reference
      run (relaxed thresholds via `cfg(debug_assertions)`).
    - New test `core/oxirs-core/tests/perf_sla.rs`:
      - Uses `#[ignore]` so it doesn't run in default CI.
      - `cargo test --release -p oxirs-core -- --ignored sla_suite` runs the
        bench in test mode and asserts thresholds.
    - Document operator workflow in `docs/performance.md` (new permanent doc).
  - **Files:** `core/oxirs-core/src/perf_sla.rs`,
    `core/oxirs-core/benches/sla_suite.rs`,
    `core/oxirs-core/perf_baseline.json`,
    `core/oxirs-core/tests/perf_sla.rs`,
    `core/oxirs-core/docs/performance.md`,
    `core/oxirs-core/Cargo.toml` (criterion dev-dep — already in workspace).
  - **Prerequisites:** `criterion` (already in workspace dev-deps).
  - **Tests:** unit-test the SLO assertion logic (regression detected /
    accepted), baseline JSON parse round-trip; the bench compiles; the
    `#[ignore]`d test passes locally on a release build.
  - **Risk:** machine-dependent timing → false fails on slow CI hardware.
    Mitigation: SLOs run only with `--release --ignored`; debug builds skip
    threshold assertion (still measure + report).
- [x] Enterprise audit trail — SOC2/GDPR-compliant structured event logging (completed 2026-05-02)

### v0.3.1 - Released (June 6, 2026)
- [x] RDF-star quoted triples in pattern matching and query execution — quoted-triple support across query algebra, executor, JIT, planner, pattern unification, and SIMD triple matching
- [x] Fixed a latent compile break in `jsonld::{compaction,flattening}` by declaring `indexmap` and `toml` as workspace dependencies
- [x] Pure-Rust migration: `crypto_provider.rs` installs OxiTLS's Pure Rust `rustls` `CryptoProvider` as the process default via a `#[ctor::ctor]` constructor (no `ring`/`aws-lc-sys` required at runtime, including in test harnesses)
- [x] Compression migrated to `oxiarc-*` crates workspace-wide (COOLJAPAN Pure Rust Policy)

### v0.3.2 - Released (July 11, 2026)
- [x] New `encoding` module — pure-`std` RFC 3986 percent-encoding (`percent_encode`, `percent_encode_strict`, `percent_decode`), replacing the external `urlencoding` crate; now backs SPARQL's `ENCODE_FOR_URI()` (`query/functions/string.rs`) and federated-query URL construction (`federation/client.rs`)
- [x] Fixed: `rdfxml::serializer` no longer collides two or more distinct unmapped namespaces on the same subject onto one hardcoded `xmlns:oxprefix` attribute; each namespace now gets its own synthetic prefix
- [x] Fixed: `query::update`'s `INSERT DATA`/`DELETE DATA` blocks whose final triple omits the trailing `.` (legal in SPARQL's grammar, illegal in the Turtle grammar used to re-parse the block) now parse correctly
- [x] Fixed: JSON-LD expansion no longer silently drops `@index` on indexed containers and node objects; `@protected` term-redefinition detection now also compares the `protected` flag itself; `@set` containers now expand to flat triples instead of a no-op
- [x] Pure-Rust Policy v2: the `gpu` feature (NVML via `nvml-wrapper`) was removed; live GPU telemetry moved to the separate `publish = false` `oxirs-gpu-monitor` adapter crate, leaving `ai::gpu_monitor::GpuMonitor` as a Pure-Rust "no GPU" stub with an unchanged public API
- [x] 2589 tests passing (100% pass rate)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Core v0.3.2 - RDF/SPARQL foundation for the OxiRS platform (zero dependencies on other OxiRS crates)*

## Proposed follow-ups

- [x] LTS commitments — RFC published at `docs/policies/lts.md` (completed 2026-05-17 via RFC-001).
- [x] Enterprise features — decomposed in `docs/policies/enterprise.md`. Audit logging completed 2026-05-02 (`core/oxirs-core/src/audit/`); RBAC templates and FIPS completed 2026-05-17. All enterprise items done.

## Jena Parity Gaps (identified 2026-05-01)

- [x] Jena Assembler vocabulary — RDF-based dataset/model configuration using the `ja:` vocabulary. Allows dataset construction from a `.ttl` config file at startup rather than from TOML config or code. The Assembler reads `ja:RDFDataset`, `ja:MemoryDataset`, `tdb2:DatasetTDB2`, `ja:namedGraph`, `ja:graphName`, `ja:graph`, `ja:defaultGraph`, `ja:contentURL`, and `tdb2:location` triples and wires the described datasets/reasoners together. This is the mechanism Jena uses for its `fuseki-config.ttl` startup configuration; supporting it would allow direct migration of Jena Fuseki configurations to OxiRS. Implemented in `core/oxirs-core/src/assembler/` with 16 tests (2026-05-01).
