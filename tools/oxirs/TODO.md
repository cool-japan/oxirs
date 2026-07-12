# OxiRS CLI - TODO

*Version: 0.3.2 | Last Updated: July 12, 2026*

## Current Status

OxiRS CLI v0.3.2 is production-ready with comprehensive SPARQL support, an interactive REPL, and ~60 CLI subcommands covering RDF data management, SPARQL query/update, SHACL/ShEx validation, TDB storage tools, industrial connectivity (TSDB/Modbus/CANbus), and SAMM/AAS model tooling (`oxirs --help` lists all of them).

### Production Features
- ✅ Complete SPARQL 1.1/1.2 query execution (`query`, `update`, `arq`, `r-sparql`, `r-update`)
- ✅ Interactive REPL mode (`interactive`) with history, autocomplete, and dot-commands
- ✅ All 7 RDF formats (Turtle, N-Triples, RDF/XML, JSON-LD, N-Quads, TriG, N3)
- ✅ Dataset management (`init`, `import`, `export`) and server administration (`serve`, `config`)
- ✅ Performance profiling and optimization tools (`explain`, `optimize`, `profile`, `performance`)
- ✅ ReBAC authorization CLI (`rebac`: export/import/migrate/verify/stats)
- ✅ RDF diff (`rdf-diff`), format conversion (`riot`, `migrate format`), SHACL/ShEx validation (`shacl`, `shex`)
- ✅ Benchmark command (`benchmark run/generate/analyze/compare`)
- ✅ Industrial connectivity: TSDB, Modbus, CANbus/J1939 (`tsdb`, `modbus`, `canbus`)
- ✅ SAMM/AAS/package tooling, Java ESMF SDK compatible (`aspect`, `aas`, `package`)
- ✅ 1799 tests passing (0 failed)

Note: `commands/diff_command.rs`, `convert_command.rs`, `validate_command.rs`, `merge_command.rs`,
`inspect_command.rs`, `query_command.rs`, `export_command.rs`, `import_command.rs`,
`benchmark_command.rs`, `profile_command.rs`, and `serve_command.rs` are implemented, unit-tested
internal library modules (doc-commented for `use oxirs::commands::<name>::{...}`) — they are not wired
to standalone `oxirs diff` / `oxirs convert` / `oxirs validate` / `oxirs merge` / `oxirs inspect` CLI
subcommands. The CLI-reachable equivalents are `rdf-diff`, `migrate format`, `shacl`/`shex`, and
`tdb-stats`; there is currently no CLI-level `merge` or `inspect` subcommand.

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ SPARQL query execution, REPL, all 7 RDF formats, 532 tests

### v0.2.3 - Released (March 16, 2026)
- ✅ Enhanced caching strategies for large datasets
- ✅ Distributed query execution improvements
- ✅ Advanced indexing optimizations
- ✅ Streaming query results for massive datasets
- ✅ Full-text search integration
- ✅ Machine learning integration for query optimization
- ✅ Real-time monitoring and alerting
- ✅ 1615 tests passing

### v0.3.0 - Released (May 3, 2026)
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise features and support (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Complete Jena/Fuseki parity verification (CLI side) (completed 2026-04-30)
  - **Goal:** Verify behavioral parity between OxiRS CLI tools and the Jena `riot`/`sparql`/`tdb2.tdbquery`/`shacl` CLI tools over a defined request matrix and fix any gaps surfaced.
  - **Design:** Built parity matrix in `tests/jena_cli_parity.rs` (driver + 17 tests including 11 self-tests + matrix walker). 44 fixtures cover the parser tools `q-parse`/`u-parse` (8), the Jena `arq` equivalent and `query` against initialised datasets (5), `riot`/`rdf-copy`/`rdf-parse`/`rdf-diff` for conversion (12), `shacl` (5), `tdb-stats`/`tdb-query` (7), and utilities `lang-tag`/`iri`/`www-enc`/`www-dec` (7). Each fixture is a TOML manifest pinning argv, stdin, exit code, and per-stream contract (`exact` / `contains` / `regex` / `absent`). Per-fixture `${TMPDIR}` and `${FIXTURE}` substitution + optional `setup = [argv...]` for fixtures that need an initialised TDB store. Spec-required mismatches FAIL the matrix; impl-detail mismatches print warnings.
  - **Gaps closed:** (1) `tdb-query --results` default flipped from `"table"` (rejected by tdbquery) to `"text"` (accepted); (2) `q-parse Variables :` line no longer prints `??s` double sigils — leans on `Variable::Display` instead of an extra `?` prefix; (3) `arq` `parse_and_validate_query` now case-preserves projected variable names and de-duplicates them (was uppercasing and emitting one entry per occurrence so `?s ?p ?o` came out as `["?S","?P","?O","?S","?P","?O"]`); also stops at `WHERE` / `{` so projection-only tokens are captured.
  - **Files:** `tests/jena_cli_parity.rs` (new), `tests/fixtures/jena-cli-ref/` (44 manifest fixtures + shared `_data/`), `src/lib.rs` (default value fix), `src/tools/qparse.rs` (variable-print fix), `src/tools/arq.rs` (variable extraction fix).
  - **Tests:** 1753 passed / 4 skipped on `cargo nextest run -p oxirs`; driver self-tests cover parser, classification, stream comparators, substitution; matrix walker covers all 44 vendored fixtures (42 spec-required, 2 impl-detail).
  - **Risk mitigation:** Jena-style banner/footer differences (riot, rdf-copy) are tagged `impl-detail` so they do not fail the gate; specification-level contracts (exit codes, stream attribution, format-flag rejection, file-not-found errors) are `spec-required`.
- [x] Comprehensive performance benchmarks (completed 2026-04-29)

### v0.3.1 - Released (June 6, 2026)
- [x] Fully Pure-Rust build: `cargo install --path .` links zero `ring`/`aws-lc-sys` C/asm crypto — compression via oxiarc, crypto via oxicrypto, TLS via oxitls
- [x] Large command modules (import, interactive, aspect, jena-parity) split so every source file stays under 2,000 lines
- [x] Dependency refresh: SciRS2 0.5.0, oxiarc 0.3.3 consumed directly from crates.io

### v0.3.2 - Current (July 11, 2026)
- [x] Pure-Rust Policy v2: GPU/GEOS/DuckDB/Kafka/Pulsar C-FFI integrations extracted to separate `publish = false` adapter crates; the `tsdb-duckdb` feature now depends on `oxirs-tsdb-adapter-duckdb` instead of an in-tree DuckDB integration
- [x] SHACL `sh:class`/implicit-class targets now honor `rdfs:subClassOf` closure, improving `oxirs shacl` validation accuracy
- [x] Dependency refresh: SciRS2 0.6.0, oxiarc-* 0.3.5, oxicrypto/oxitls 0.2.0
- [x] 1799 tests passing (0 failed)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on how to contribute to OxiRS development.

---

*OxiRS CLI v0.3.2 - Production-ready semantic web toolkit*
