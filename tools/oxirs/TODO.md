# OxiRS CLI - TODO

*Version: 0.3.0 | Last Updated: May 3, 2026*

## Current Status

OxiRS CLI v0.2.3 is production-ready with comprehensive SPARQL support, interactive REPL, and complete RDF management capabilities.

### Production Features
- ✅ Complete SPARQL 1.1/1.2 query execution
- ✅ Interactive mode with autocomplete and syntax highlighting
- ✅ All 7 RDF formats (Turtle, N-Triples, RDF/XML, JSON-LD, N-Quads, TriG, N3)
- ✅ Dataset management and server administration
- ✅ Performance profiling and optimization tools
- ✅ ReBAC authorization and security features
- ✅ Diff command (RDF set-diff + Dice similarity scoring)
- ✅ Convert command (cross-format RDF conversion)
- ✅ Validate command (SHACL validation from CLI)
- ✅ Benchmark command (query performance benchmarking)
- ✅ Profile command (execution profiling)
- ✅ Import/export commands
- ✅ Inspect, merge, query commands
- ✅ Serve command
- ✅ 1615 tests passing (100% pass rate)

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ SPARQL query execution, REPL, all 7 RDF formats, 532 tests

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Enhanced caching strategies for large datasets
- ✅ Distributed query execution improvements
- ✅ Advanced indexing optimizations
- ✅ Streaming query results for massive datasets
- ✅ Full-text search integration
- ✅ Machine learning integration for query optimization
- ✅ Real-time monitoring and alerting
- ✅ Diff, convert, validate, benchmark, profile commands
- ✅ Import, export, inspect, merge, query, serve commands
- ✅ 1615 tests passing

### v0.3.0 - Planned (Q2 2026)
- [~] Long-term support guarantees (policy: docs/policies/lts.md)
- [~] Enterprise features and support (policy: docs/policies/enterprise.md, decomposed items listed therein)
- [x] Complete Jena/Fuseki parity verification (CLI side) (completed 2026-04-30)
  - **Goal:** Verify behavioral parity between OxiRS CLI tools and the Jena `riot`/`sparql`/`tdb2.tdbquery`/`shacl` CLI tools over a defined request matrix and fix any gaps surfaced.
  - **Design:** Built parity matrix in `tests/jena_cli_parity.rs` (driver + 17 tests including 11 self-tests + matrix walker). 44 fixtures cover the parser tools `q-parse`/`u-parse` (8), the Jena `arq` equivalent and `query` against initialised datasets (5), `riot`/`rdf-copy`/`rdf-parse`/`rdf-diff` for conversion (12), `shacl` (5), `tdb-stats`/`tdb-query` (7), and utilities `lang-tag`/`iri`/`www-enc`/`www-dec` (7). Each fixture is a TOML manifest pinning argv, stdin, exit code, and per-stream contract (`exact` / `contains` / `regex` / `absent`). Per-fixture `${TMPDIR}` and `${FIXTURE}` substitution + optional `setup = [argv...]` for fixtures that need an initialised TDB store. Spec-required mismatches FAIL the matrix; impl-detail mismatches print warnings.
  - **Gaps closed:** (1) `tdb-query --results` default flipped from `"table"` (rejected by tdbquery) to `"text"` (accepted); (2) `q-parse Variables :` line no longer prints `??s` double sigils — leans on `Variable::Display` instead of an extra `?` prefix; (3) `arq` `parse_and_validate_query` now case-preserves projected variable names and de-duplicates them (was uppercasing and emitting one entry per occurrence so `?s ?p ?o` came out as `["?S","?P","?O","?S","?P","?O"]`); also stops at `WHERE` / `{` so projection-only tokens are captured.
  - **Files:** `tests/jena_cli_parity.rs` (new), `tests/fixtures/jena-cli-ref/` (44 manifest fixtures + shared `_data/`), `src/lib.rs` (default value fix), `src/tools/qparse.rs` (variable-print fix), `src/tools/arq.rs` (variable extraction fix).
  - **Tests:** 1753 passed / 4 skipped on `cargo nextest run -p oxirs`; driver self-tests cover parser, classification, stream comparators, substitution; matrix walker covers all 44 vendored fixtures (42 spec-required, 2 impl-detail).
  - **Risk mitigation:** Jena-style banner/footer differences (riot, rdf-copy) are tagged `impl-detail` so they do not fail the gate; specification-level contracts (exit codes, stream attribution, format-flag rejection, file-not-found errors) are `spec-required`.
- [x] Comprehensive performance benchmarks (completed 2026-04-29)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on how to contribute to OxiRS development.

---

*OxiRS CLI v0.2.3 - Production-ready semantic web toolkit*
