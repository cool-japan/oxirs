# OxiRS Core - TODO

*Version: 0.3.0 | Last Updated: May 3, 2026*

## Current Status

OxiRS Core v0.2.3 is production-ready, providing the foundation for semantic web operations with complete RDF/SPARQL support.

### Production Features
- ✅ RDF 1.2 data model with 7 format parsers
- ✅ SPARQL 1.1/1.2 query engine with adaptive optimization
- ✅ Persistent storage with N-Quads serialization
- ✅ ML-based query optimization with adaptive learning
- ✅ Federation support with SERVICE clause execution
- ✅ SciRS2 integration for scientific computing
- ✅ 2332 tests passing (100% pass rate)

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ RDF 1.2 data model with 7 format parsers
- ✅ SPARQL 1.1/1.2 query engine with adaptive optimization
- ✅ Federation support with SERVICE clause execution
- ✅ SciRS2 integration, 850 tests passing

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Cost-based optimizer enhancements
- ✅ Advanced join ordering strategies
- ✅ Query result caching with invalidation
- ✅ Incremental view maintenance
- ✅ Parallel query execution
- ✅ Distributed storage backend
- ✅ Horizontal sharding support
- ✅ Advanced indexing strategies
- ✅ 2332 tests passing (100% pass rate)

### v0.3.0 - Planned (Q2 2026)
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
- [~] Long-term support commitments (policy: docs/policies/lts.md)
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

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Core v0.2.3 - Zero-dependency RDF/SPARQL foundation*

## Proposed follow-ups

- [~] LTS commitments — RFC published at `docs/policies/lts.md`.
- [~] Enterprise features — decomposed in `docs/policies/enterprise.md`. Audit logging completed 2026-05-02 (`core/oxirs-core/src/audit/`); SSO, RBAC, multi-tenant, compliance posture, support SLA still pending.

## Jena Parity Gaps (identified 2026-05-01)

- [x] Jena Assembler vocabulary — RDF-based dataset/model configuration using the `ja:` vocabulary. Allows dataset construction from a `.ttl` config file at startup rather than from TOML config or code. The Assembler reads `ja:RDFDataset`, `ja:MemoryDataset`, `tdb2:DatasetTDB2`, `ja:namedGraph`, `ja:graphName`, `ja:graph`, `ja:defaultGraph`, `ja:contentURL`, and `tdb2:location` triples and wires the described datasets/reasoners together. This is the mechanism Jena uses for its `fuseki-config.ttl` startup configuration; supporting it would allow direct migration of Jena Fuseki configurations to OxiRS. Implemented in `core/oxirs-core/src/assembler/` with 16 tests (2026-05-01).
