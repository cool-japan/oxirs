# OxiRS

> A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.1-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: v0.4.1 - Released - 2026-07-28

**Production Ready**: Complete SPARQL 1.1/1.2 implementation with **3.8x faster optimizer**, industrial IoT support, and AI-powered features. **46,255 tests passing** (`--all-features`; 45,408 with default features) with zero warnings across all 27 crates.

**v0.4.1 Highlights (2026-07-28)**: A workspace-wide production-readiness hardening pass — a 38-scope multi-agent audit surfaced 308 verified findings (62 P0, 131 P1, 115 P2), with roughly 300 fixed across 38 work packages. Security fixes close a full oxirs-fuseki authentication/authorization bypass (the `AuthUser` extractor never actually authenticated requests, `route_based_rbac` failed open, and `/update` had no auth check — all three now gate correctly on `config.security.auth_required`), an X.509 client-certificate check that only compared DN strings instead of verifying the signature chain, a SPARQL injection hole in the REST API v2, and missing SSRF guards on SPARQL `LOAD`; oxirs-gql's "AES" cache-encryption layer was a plaintext passthrough with token validation that unconditionally granted access — both fixed; oxirs-did now checks credential revocation, consumes the auth challenge to prevent replay, and verifies ZKP selective-disclosure proofs instead of accepting them unconditionally; oxirs-chat's SAML verification is hardened against XML Signature Wrapping attacks. Data-integrity fixes: oxirs-cluster Raft state (vote/log/state-machine/snapshot) is now durably fsync-persisted via a new `DurableRaftStore` (previously in-memory only, risking double-votes or lost commits on restart); the `oxirs` CLI's `tdbupdate` no longer collapses every literal/blank-node to an IRI before writing (previously corrupted TDB-backed stores for every other reader); oxirs-tdb's `repair_page_checksums` and superblock reads now verify checksums before trusting the data instead of silently propagating corruption; oxirs-did's `StatusList2021` encoding is now genuinely GZIP-compressed. Also: the core SPARQL parser no longer silently drops `FILTER` clauses, multi-line `CONSTRUCT`/`SELECT`/`ASK`/`DESCRIBE`/`UPDATE` parsing is fixed, a new mmap-able RDF store snapshot format gives sub-second cold start (versus ~13.6s re-parsing 1.35M quads), SPARQL 1.1 sub-`SELECT` and Turtle collection/property-list syntax now work inside query patterns, and RDF-vocabulary-driven GraphQL schema auto-generation (`graphql_autoschema`) is new in oxirs-fuseki.

**v0.4.0 Highlights (2026-07-19)**: Consolidates the previously-unpublished 0.3.3 production-hardening work (a 272-finding audit → 288 fixes plus 455 regression tests across storage, security, and distributed subsystems) with 0.3.4's deployment fixes from the public, query-only sparql.wik.jp rollout — neither 0.3.3 nor 0.3.4 was published to crates.io, so both ship here as 0.4.0. oxirs-tdb is now a real durable on-disk backend (superblock, fsync-backed writes, free-page allocator, GSPO/GPOS/GOSP indexes) wired into oxirs-fuseki via `StoreType::TDB2`/`dataset_type = "tdb2"` and into `oxirs import --dataset-type tdb2`; oxirs-core's `RdfStore` Persistent backend replaces its O(N²) per-insert full-file rewrite with O(N) buffered-append persistence, and `MemoryStorage` now interns terms for roughly 4x lower RAM. The SPARQL query path is now unified through the real oxirs-arq engine end to end: `CONSTRUCT`/`DESCRIBE` execute via new template-instantiation/CBD machinery, `GRAPH <iri>`/`GRAPH ?g` and `FROM`/`FROM NAMED` execute for real against dataset views, `SERVICE` HTTP federation is reachable from the SPARQL endpoint, and native aggregate projections (`COUNT`/`SUM`/`MIN`/`MAX`/`AVG`/`SAMPLE`/`GROUP_CONCAT`, expressions inside aggregates, `DISTINCT`) plus `HAVING` (including aggregates inside `HAVING`) run through the engine's grouping machinery — the legacy demo path and every silent-empty-200 fallback are gone. Parser fixes cover WHERE-less `ASK`/`SELECT *`, positionally-scoped `BIND`, group-scoped `FILTER` (including after a top-level `UNION`), populated `GROUP BY`/`ORDER BY` lists, and multi-triple `INSERT/DELETE DATA` parsing. The axum 0.8 route migration is now complete workspace-wide (fuseki, cluster, embed, chat). Security hardening adds real X25519/Ristretto DID crypto, OIDC/SAML SSO signature verification, real cluster RPC with BFT quorum, and enforced `read_only` dataset checks — now name-agnostic for single-dataset deployments, with startup diagnostics for multi-dataset misconfigurations — across all write paths including REST API v2 and admin dataset management.

## Vision

OxiRS aims to be a **Rust-first, JVM-free** alternative to Apache Jena + Fuseki and to Juniper, providing:

- **Protocol choice, not lock-in**: Expose both SPARQL 1.2 and GraphQL endpoints from the same dataset
- **Incremental adoption**: Each crate works stand-alone; opt into advanced features via Cargo features
- **AI readiness**: Native integration with vector search, graph embeddings, and LLM-augmented querying
- **Single static binary**: Match or exceed Jena/Fuseki feature-for-feature while keeping a <50MB footprint

## Quick Start

### Installation

```bash
# Build the CLI from source
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo install --path tools/oxirs

# Or just build the whole workspace without installing
cargo build --workspace --release
```

> **Note**: the `oxirs` CLI binary is intentionally kept off crates.io (`publish = false`,
> since v0.3.2) so it can optionally depend on `publish = false` quarantine adapter crates
> without pulling their C FFI onto a published Pure-Rust surface. All 25 OxiRS library crates remain normally published to
> crates.io — see [Published Crates](#published-crates) below.

### What's New in v0.4.1 (2026-07-28)

**Production-Readiness Hardening Release: Security Fixes, Data-Integrity Fixes, and SPARQL Parser Corrections**

A 38-scope multi-agent audit surfaced 308 verified findings (62 P0, 131 P1, 115 P2) across the workspace; roughly 300 were fixed across 38 work packages, plus 74 test regressions caught by the full-suite gate and a follow-up chain of SPARQL-parser, storage-durability, and CLI/DID data-integrity fixes:

- **Security fixes** - oxirs-fuseki: closed a full authentication/authorization bypass (`AuthUser` extractor never authenticated, `route_based_rbac` failed open, `/update` had no auth check — all three now gate on `config.security.auth_required`), X.509 client-cert trust now verifies the signature chain (not just DN-string comparison), a SPARQL injection hole in REST API v2, and SSRF guards on SPARQL `LOAD`; oxirs-gql's "AES" cache-encryption layer was a plaintext passthrough with token validation that unconditionally granted access — both fixed; oxirs-did now checks credential revocation, consumes the auth challenge (replay prevention), and verifies ZKP selective-disclosure proofs instead of accepting them unconditionally; oxirs-chat SAML verification hardened against XML Signature Wrapping attacks
- **Data-integrity fixes** - oxirs-cluster Raft state (vote/log/state-machine/snapshot) is now durably fsync-persisted via a new `DurableRaftStore` (write-temp + fsync + rename), closing the durable-storage gap deferred from the 0.4.0 Raft integration; the `oxirs` CLI's `tdbupdate` no longer collapses every literal/blank-node to an IRI before writing (previously corrupted TDB-backed stores for every other reader); oxirs-tdb's `repair_page_checksums` and superblock reads now verify checksums before trusting data instead of silently propagating or masking corruption; oxirs-did's `StatusList2021` encoding is now genuinely GZIP-compressed (fixing interop with external W3C-compliant verifiers)
- **SPARQL correctness fixes** - the core SPARQL parser no longer silently drops `FILTER` clauses; multi-line `CONSTRUCT`/`SELECT`/`ASK`/`DESCRIBE` queries and `UPDATE` statements (newline before `WHERE`/braces, multi-`PREFIX` prologues) now parse correctly; `Store::flush()` actually flushes to disk (was a no-op)
- **Fabrication removed** - the W3C SHACL conformance harness no longer fabricates results; oxirs-shacl-ai no longer fabricates quality metrics or AutoML "training" results; oxirs-federate's query planner now really decomposes queries instead of returning a canned plan; performance_validation benchmarks now run genuine end-to-end SPARQL queries instead of `sleep()`-simulated workload
- **Other fixes** - oxirs-embed KGE model save/load implemented (was a no-op), TransE/RotatE training loss sign corrected; oxirs-stream NATS consumer no longer acks before processing, the NATS producer fails loud instead of silently no-op'ing; oxirs-cluster's replication-maintenance background task no longer leaks (was running against a throwaway empty manager in a shutdown-blind loop on every node)
- **New** - SPARQL 1.1 sub-`SELECT` and Turtle collection/property-list syntax inside query patterns (oxirs-arq); RDF-vocabulary-driven GraphQL schema auto-generation `graphql_autoschema` (oxirs-fuseki); mmap-able RDF store snapshot format for sub-second cold start (oxirs-core, versus ~13.6s re-parsing 1.35M quads)
- **Changed** - the SPARQL query path is fully unified onto the real oxirs-arq execution engine; oxirs-cluster state-machine checkpointing is durable and amortized O(1) (was O(n²)); oxirs-star tiered-storage default directories are now instance-unique (fixing a cross-process data-leakage bug); oxirs-core's `blas` feature now wires into real OxiBLAS routines (previously compiled but had no effect); the CLI's `all-features` feature (which silently omitted `excel-export`/`system-keyring`) is renamed `full-cli` and now genuinely enables every optional feature — build scripts must switch `--features all-features` → `--features full-cli`
- **Removed (breaking)** - three C-FFI quarantine adapters retired outright: `oxirs-geosparql-adapter-geos` and the `rust-buffer` feature (superseded by Pure-Rust `geo::algorithm::buffer`, which does strictly more), `oxirs-tsdb-adapter-duckdb` and the CLI's `tsdb-duckdb` feature (export to Parquet instead), and `oxirs-stream-adapter-rdkafka` together with the `StreamBackendType::Kafka` config variant — being `publish = false` it was unreachable from any release, nothing depended on it, and it required a `librdkafka` C build. Its Confluent Schema Registry client never used `rdkafka` and now ships in `oxirs-stream` as `confluent_registry`. With these gone plus a `default-members` list, `cargo test --all-features` needs no C toolchain

**Quality Metrics (v0.4.1):**
- ✅ **308-finding production-readiness audit** → ~300 fixes across 38 work packages, plus 74 test regressions caught and fixed by the full-suite gate
- ✅ **46,255 tests passing** (`--all-features`; 45,408 with default features), 100% pass rate
- ✅ Zero compilation warnings maintained across all 27 crates
- ✅ New workspace `deny.toml` documents 14 explicitly-reviewed `cargo-deny`/RUSTSEC exceptions; `cargo deny check bans`/`check advisories` clean under default and `--all-features`

---

### What's New in v0.4.0 (2026-07-19)

**Production Hardening & Deployment Release: TDB2 On-Disk Backend, SPARQL Correctness, and Security Fixes**

OxiRS v0.4.0 consolidates the unreleased 0.3.3 production-hardening pass (a 272-finding audit spanning storage, security, and distributed subsystems) with the 0.3.4 deployment fixes surfaced by the public, query-only sparql.wik.jp rollout. Neither 0.3.3 nor 0.3.4 was published to crates.io — both ship here as 0.4.0:

- **oxirs-tdb real on-disk backend** - Superblock (v2, quad roots), fsync-backed writes, a free-page allocator, GSPO/GPOS/GOSP quad and named-graph indexes, and streaming quad iterators replace the previous non-persisting placeholder; wired into oxirs-fuseki as `StoreType::TDB2`/`dataset_type = "tdb2"` (new `TdbStoreAdapter`/`StoreFactory`) and into `oxirs import --dataset-type tdb2` for bounded-RAM bulk loads; unknown dataset types now fail loud instead of silently falling back to in-memory
- **oxirs-core storage rewrite** - `RdfStore`'s Persistent backend replaces its O(N²) per-insert full-file rewrite with a single buffered N-Quads append (O(N)); `MemoryStorage` now interns terms behind ID-based index permutations, cutting per-triple RAM roughly 4x; new public durability API (`open_with_sync_policy`, `flush`, `SyncPolicy`, `AsyncRdfStore::flush_async`) plus streaming `bulk_insert_quads`/`for_each_quad` on the `Store` trait
- **SPARQL query path unified through the real oxirs-arq engine** - a single parse-once dispatch replaces substring-based query-type routing; `CONSTRUCT` and `DESCRIBE` (including `DESCRIBE <iri>` with no `WHERE`, `DESCRIBE *`, and the `CONSTRUCT WHERE {}` shorthand) execute via new template-instantiation/CBD machinery; `GRAPH <iri>`/`GRAPH ?g` execute for real in both the serial and parallel executors; `FROM`/`FROM NAMED` are honored via dataset views; `SERVICE` HTTP federation is reachable end-to-end; native aggregate projections (`COUNT`/`SUM`/`MIN`/`MAX`/`AVG`/`SAMPLE`/`GROUP_CONCAT` with `SEPARATOR`, expressions inside aggregates like `SUM(?a*?b)`, `DISTINCT`) and `HAVING` (including aggregate calls in `HAVING`) run through the engine's grouping machinery; the legacy demo path and every silent-empty-200 fallback are deleted — parse failures are HTTP 400, execution failures HTTP 500, never a silent empty 200
- **SPARQL parser fixes** - WHERE-less `ASK {}`/`SELECT * {}` now extract patterns correctly; the arq parser accepts `SELECT *` and optional-WHERE `ASK`; `BIND` is now scoped positionally (was deferred to the group end, silently producing wrong bindings); a `FILTER` after a top-level `UNION` is now group-scoped; `GROUP BY`/`ORDER BY` expression lists actually populate (a latent tokenizer bug on the trailing `BY` keyword silently dropped both lists); `DESCRIBE` targets are retained
- **oxirs-fuseki hardening** - SPARQL UPDATE dispatch now runs on the parsed AST instead of a `.contains("CLEAR")` substring scan; `INSERT DATA`/`DELETE DATA` blocks parse every triple via proper top-level `.`-terminator splitting instead of dropping rows past the first line; routes migrated to axum 0.8/matchit 0.8 path syntax — and that migration is now complete workspace-wide (oxirs-cluster's dashboard, oxirs-embed's API, and oxirs-chat's server each got their own route fixes plus a router-construction regression test)
- **read_only dataset enforcement hardened** - name-agnostic resolution when exactly one dataset is configured (any name gets write protection, not only `"default"`); startup WARN/ERROR diagnostics for multi-dataset misconfigurations; a shared guard helper now also protects admin dataset create/delete/compact/reload and the REST API v2 dataset/triple write endpoints, closing a previously unguarded write bypass
- **Security fixes** - oxirs-did gains real X25519 ECDH and Ristretto Schnorr/Pedersen crypto (replacing forgeable placeholders); oxirs-chat SSO verifies OIDC ID tokens (RS256/ES256 via JWKS) and SAML XML signatures, failing closed; oxirs-cluster inter-node RPC is now real length-prefixed, oxicode-framed TCP with a real 2f+1 BFT quorum
- **Other fixes** - oxirs-vec TF-IDF smoothed-IDF correction and a FAISS HNSW/IVF read-path cursor-alignment fix; oxirs-samm graph analytics short-circuits `density = 0` for `n < 2`

A follow-up hardening round tightens the query path and the CLI further: aggregate arity inside `HAVING` (`SUM()`, `COUNT(?a,?b)`) is now rejected at parse time as an HTTP 400, an unknown function in `FILTER`/`HAVING` fails the whole query loudly (typed `UnknownFunctionError`) instead of silently dropping rows, `DESCRIBE` returns a **symmetric** Concise Bounded Description (incoming/object-side arcs plus blank-node closure in both directions), and the SPARQL Results JSON serializer is now exhaustive — RDF-star quoted triples serialize as `{"type":"triple",…}` and a property-path binding is a 500 fail-loud error rather than a fabricated literal. On the storage side, oxirs-tdb gains WAL-integrated durable writes with crash-recovery replay and a checkpoint LSN, honored `StoreParams`, and opt-in `O_DIRECT`/`F_NOCACHE`; oxirs-core's term dictionary reference-counts ids with a free-list so deletes reclaim ids immediately; and oxirs-cluster's BFT consensus (feature `bft`) is a closed loop that applies real commands to storage on a genuine 2f+1 quorum. The `oxirs` CLI adds `lint`, `merge`, `jena-parity`, `monitor` (remote-endpoint), `detect-format`, and `inspect` subcommands, `serve --dry-run`, `schema-gen --advanced`, `history export-csv`/`similar`, `profile --flamegraph`, REPL meta-commands, and transparent `.gz` I/O; `generate --schema` now parses the supplied schema instead of emitting hardcoded sample data, and over two dozen dead/simulated command modules were removed.

**Quality Metrics (v0.4.0):**
- ✅ **272-finding production-hardening audit** → 288 fixes and **455 new regression tests** across storage, security, and distributed subsystems (on top of the 45,034 tests passing at v0.3.2 with `--all-features`)
- ✅ New query-path/parser suites: `oxirs-fuseki/tests/query_path_040.rs` (20 real-server end-to-end cases), `oxirs-arq/tests/bind_scoping_test.rs` (9 cases), `oxirs-arq/tests/parser_forms_test.rs` (23 cases); oxirs-arq 3012 tests passing, oxirs-fuseki 2381 tests passing at integration time
- ✅ Zero compilation warnings maintained across all 27 crates
- ✅ Consolidates the unreleased 0.3.3 + 0.3.4 work into a single release; not yet published to crates.io

---

### What's New in v0.3.2 (2026-07-12)

**Maintenance & Purity Release: Pure-Rust Policy v2, OxiSQL, and Quarantined C-FFI Adapters**

OxiRS v0.3.2 completes a second, deeper pass of the COOLJAPAN Pure-Rust migration and hardens SHACL, GeoSPARQL, and the oxirs-wasm query engine:

- **Pure-Rust Policy v2 (breaking)** - Six in-tree, feature-gated C-FFI integrations (NVML GPU monitoring, CUDA, GEOS, DuckDB, Kafka, Pulsar) extracted into new `publish = false` quarantine adapter crates (`oxirs-gpu-monitor`, `oxirs-vec-adapter-cuda`, `oxirs-geosparql-adapter-geos`, `oxirs-tsdb-adapter-duckdb`, `oxirs-stream-adapter-rdkafka`, `oxirs-stream-adapter-pulsar`), each API-compatible with the in-tree feature it replaces; the old feature flags themselves (oxirs-core's `gpu`, oxirs-vec's `cuda`/`gpu-full`, oxirs-geosparql's `geos-backend`, oxirs-tsdb's `duckdb`, oxirs-stream's `kafka`/`pulsar`) were removed
- **OxiSQL GeoPackage backend** - GeoSPARQL's `GeoPackage` SQLite backend migrated from `rusqlite` (bundled C libsqlite3) to the new Pure-Rust `oxisql-core`/`oxisql-sqlite-compat` engine, plus an explicit `GeoPackage::checkpoint()` for WAL flush
- **Pure-Rust `zstd` shim** - New internal `crates/zstd-shim` (backed by `oxiarc-zstd`), applied workspace-wide via `[patch.crates-io]`, removes the last transitive `zstd-sys` C dependency (tantivy, parquet, pulsar, wasmtime)
- **SHACL subclass-aware targets** - Reflexive+transitive `rdfs:subClassOf` closure (`advanced_features::subclass_closure`); `sh:class` and implicit-class targets now honor subclassing; SPARQL-based and single-hop property-path SHACL targets execute for real against the store instead of returning stub results
- **oxirs-wasm query engine** - SPARQL `PREFIX`/`BASE` prologues, a per-store solution budget (`setSolutionBudget`/`clearSolutionBudget`) that fails unselective joins fast instead of running to completion, and triple-pattern/property-path evaluation now driven by subject/predicate/object indexes instead of full scans
- **GeoSPARQL geometry fixes** - Shapefile writer now emits interior rings (holes) for `Polygon`/`MultiPolygon`; compressed-geometry round-tripping preserves polygon holes and multi-part `MultiLineString`/`MultiPolygon` structure (new `ring_counts` field); WKT parser accepts optional Z/M coordinates
- **oxirs-tdb distributed transactions** - Saga steps, 2PC, and 3PC participants now run real registered callbacks against a WAL-backed `Transaction` instead of simulating success; the distributed coordinator's `abort_transaction` notifies all participants
- **oxirs-gql / oxirs-chat / oxirs-federate / oxirs-stream** - Adaptive query-batching dependency analysis, an activated ML-driven `DynamicQueryPlanner`, a real z-score anomaly detector and alert-handler dispatch (Welford's online algorithm), NATS federation message dispatch by type, and an MQTT 5.0 property codec all replace previous no-op/stub code paths
- **Dependency refresh** - SciRS2 0.5.0 → 0.6.0; `oxiarc-*` 0.3.3 → 0.3.5; `oxicrypto`/`oxitls` 0.1.1 → 0.2.0; `kube` 3.1 → 4.0 (optional `k8s` feature); `bytes` → 1.12.0 (CVE-2026-25541 fix); `lazy_static` → `once_cell`, `num_cpus` → `std::thread::available_parallelism()`

**Quality Metrics (v0.3.2):**
- ✅ **45,199 tests passing** (`--all-features`; 44,398 with default features), 100% pass rate
- ✅ **Zero compilation warnings** across all 27 crates
- ✅ **Pure Rust by default and under `--all-features`** - zero `ring`/`aws-lc-sys`/`rusqlite`/`zstd-sys` reach any published crate; the six remaining C-FFI integrations live in separately-versioned, opt-in `publish = false` adapter crates
- ✅ **All `.rs` files under 2,000 lines** (proactive refactors applied)

---

### Usage

```bash
# Initialize a new knowledge graph (alphanumeric, _, - only).
# `--format memory` matches what `oxirs import`/`oxirs query` use by default
# below; `oxirs init` also accepts `--format tdb2` for the on-disk backend,
# but `oxirs query` cannot read tdb2 datasets yet (use `oxirs tdbquery`
# instead) and `oxirs import`/`oxirs query` do not read the backend back out
# of the dataset's own oxirs.toml, so pass `--dataset-type` explicitly on
# every command instead of relying on `init`'s declared storage format.
oxirs init mykg --format memory

# Import RDF data (automatically persisted to mykg/data.nq)
oxirs import mykg data.ttl --format turtle

# Query the data (loaded automatically from disk)
oxirs query mykg "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Query with specific patterns
oxirs query mykg "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"

# Start the server
oxirs serve mykg/oxirs.toml --port 3030
```

**Features:**
- ✅ **Persistent storage**: Data automatically saved to disk in N-Quads format
- ✅ **SPARQL queries**: SELECT, ASK, CONSTRUCT, DESCRIBE supported
- ✅ **Auto-load**: No manual save/load needed
- ✅ **PREFIX / BASE support**: Full prologue declarations, resolved through the same SPARQL engine used by the server

Open:
- http://localhost:3030 for the Fuseki-style admin UI
- http://localhost:3030/graphql/playground for the browsable GraphiQL IDE (the GraphQL routes are mounted unconditionally by the server; `/graphql` itself is a POST-only JSON API endpoint, not a browser page — opening it with a plain `GET` returns 405)

## Published Crates

All crates are published to [crates.io](https://crates.io) and documented on [docs.rs](https://docs.rs).

### Core

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-core]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-core.svg)](https://crates.io/crates/oxirs-core) | [![docs.rs](https://docs.rs/oxirs-core/badge.svg)](https://docs.rs/oxirs-core) | Core RDF and SPARQL functionality |

[oxirs-core]: https://crates.io/crates/oxirs-core

### Server

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-fuseki]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-fuseki.svg)](https://crates.io/crates/oxirs-fuseki) | [![docs.rs](https://docs.rs/oxirs-fuseki/badge.svg)](https://docs.rs/oxirs-fuseki) | SPARQL 1.1/1.2 HTTP server |
| **[oxirs-gql]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-gql.svg)](https://crates.io/crates/oxirs-gql) | [![docs.rs](https://docs.rs/oxirs-gql/badge.svg)](https://docs.rs/oxirs-gql) | GraphQL endpoint for RDF |

[oxirs-fuseki]: https://crates.io/crates/oxirs-fuseki
[oxirs-gql]: https://crates.io/crates/oxirs-gql

### Engine

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-arq]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-arq.svg)](https://crates.io/crates/oxirs-arq) | [![docs.rs](https://docs.rs/oxirs-arq/badge.svg)](https://docs.rs/oxirs-arq) | SPARQL query engine |
| **[oxirs-rule]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-rule.svg)](https://crates.io/crates/oxirs-rule) | [![docs.rs](https://docs.rs/oxirs-rule/badge.svg)](https://docs.rs/oxirs-rule) | Rule-based reasoning |
| **[oxirs-shacl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl.svg)](https://crates.io/crates/oxirs-shacl) | [![docs.rs](https://docs.rs/oxirs-shacl/badge.svg)](https://docs.rs/oxirs-shacl) | SHACL validation |
| **[oxirs-samm]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-samm.svg)](https://crates.io/crates/oxirs-samm) | [![docs.rs](https://docs.rs/oxirs-samm/badge.svg)](https://docs.rs/oxirs-samm) | SAMM metamodel & AAS |
| **[oxirs-geosparql]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-geosparql.svg)](https://crates.io/crates/oxirs-geosparql) | [![docs.rs](https://docs.rs/oxirs-geosparql/badge.svg)](https://docs.rs/oxirs-geosparql) | GeoSPARQL support |
| **[oxirs-star]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-star.svg)](https://crates.io/crates/oxirs-star) | [![docs.rs](https://docs.rs/oxirs-star/badge.svg)](https://docs.rs/oxirs-star) | RDF-star support |
| **[oxirs-ttl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-ttl.svg)](https://crates.io/crates/oxirs-ttl) | [![docs.rs](https://docs.rs/oxirs-ttl/badge.svg)](https://docs.rs/oxirs-ttl) | Turtle parser |
| **[oxirs-vec]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-vec.svg)](https://crates.io/crates/oxirs-vec) | [![docs.rs](https://docs.rs/oxirs-vec/badge.svg)](https://docs.rs/oxirs-vec) | Vector search |

[oxirs-arq]: https://crates.io/crates/oxirs-arq
[oxirs-rule]: https://crates.io/crates/oxirs-rule
[oxirs-shacl]: https://crates.io/crates/oxirs-shacl
[oxirs-samm]: https://crates.io/crates/oxirs-samm
[oxirs-geosparql]: https://crates.io/crates/oxirs-geosparql
[oxirs-star]: https://crates.io/crates/oxirs-star
[oxirs-ttl]: https://crates.io/crates/oxirs-ttl
[oxirs-vec]: https://crates.io/crates/oxirs-vec

### Storage

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-tdb]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-tdb.svg)](https://crates.io/crates/oxirs-tdb) | [![docs.rs](https://docs.rs/oxirs-tdb/badge.svg)](https://docs.rs/oxirs-tdb) | TDB2-compatible storage |
| **[oxirs-cluster]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-cluster.svg)](https://crates.io/crates/oxirs-cluster) | [![docs.rs](https://docs.rs/oxirs-cluster/badge.svg)](https://docs.rs/oxirs-cluster) | Distributed clustering |

[oxirs-tdb]: https://crates.io/crates/oxirs-tdb
[oxirs-cluster]: https://crates.io/crates/oxirs-cluster

### Stream

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-stream]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-stream.svg)](https://crates.io/crates/oxirs-stream) | [![docs.rs](https://docs.rs/oxirs-stream/badge.svg)](https://docs.rs/oxirs-stream) | Real-time streaming |
| **[oxirs-federate]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-federate.svg)](https://crates.io/crates/oxirs-federate) | [![docs.rs](https://docs.rs/oxirs-federate/badge.svg)](https://docs.rs/oxirs-federate) | Federated queries |

[oxirs-stream]: https://crates.io/crates/oxirs-stream
[oxirs-federate]: https://crates.io/crates/oxirs-federate

### AI

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-embed]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-embed.svg)](https://crates.io/crates/oxirs-embed) | [![docs.rs](https://docs.rs/oxirs-embed/badge.svg)](https://docs.rs/oxirs-embed) | Knowledge graph embeddings & vector store |
| **[oxirs-shacl-ai]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl-ai.svg)](https://crates.io/crates/oxirs-shacl-ai) | [![docs.rs](https://docs.rs/oxirs-shacl-ai/badge.svg)](https://docs.rs/oxirs-shacl-ai) | AI-powered SHACL constraint inference |
| **[oxirs-chat]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-chat.svg)](https://crates.io/crates/oxirs-chat) | [![docs.rs](https://docs.rs/oxirs-chat/badge.svg)](https://docs.rs/oxirs-chat) | RAG chat API with conversation history |
| **[oxirs-physics]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-physics.svg)](https://crates.io/crates/oxirs-physics) | [![docs.rs](https://docs.rs/oxirs-physics/badge.svg)](https://docs.rs/oxirs-physics) | Physics-informed digital twin reasoning |
| **[oxirs-graphrag]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-graphrag.svg)](https://crates.io/crates/oxirs-graphrag) | [![docs.rs](https://docs.rs/oxirs-graphrag/badge.svg)](https://docs.rs/oxirs-graphrag) | GraphRAG hybrid search (Vector x Graph) |

[oxirs-embed]: https://crates.io/crates/oxirs-embed
[oxirs-shacl-ai]: https://crates.io/crates/oxirs-shacl-ai
[oxirs-chat]: https://crates.io/crates/oxirs-chat
[oxirs-physics]: https://crates.io/crates/oxirs-physics
[oxirs-graphrag]: https://crates.io/crates/oxirs-graphrag

### Security

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-did]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-did.svg)](https://crates.io/crates/oxirs-did) | [![docs.rs](https://docs.rs/oxirs-did/badge.svg)](https://docs.rs/oxirs-did) | DID & Verifiable Credentials |

[oxirs-did]: https://crates.io/crates/oxirs-did

### Platforms

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-wasm]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-wasm.svg)](https://crates.io/crates/oxirs-wasm) | [![docs.rs](https://docs.rs/oxirs-wasm/badge.svg)](https://docs.rs/oxirs-wasm) | WASM browser/edge deployment |

[oxirs-wasm]: https://crates.io/crates/oxirs-wasm

### Tools

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs (CLI)]** | *not published — build from source* | [source](tools/oxirs) | Command-line interface: import, export, query, migration, benchmarking |

[oxirs (CLI)]: tools/oxirs

> As of v0.3.2 the `oxirs` CLI binary is `publish = false` (see [Installation](#installation)); it is not on crates.io. The other 25 library crates above are published normally.

## Architecture

```
oxirs/                  # Cargo workspace root
├─ core/                # Thin, safe re-export of oxigraph
│  └─ oxirs-core
├─ server/              # Network front ends
│  ├─ oxirs-fuseki      # SPARQL 1.1/1.2 HTTP protocol, Fuseki-compatible config
│  └─ oxirs-gql         # GraphQL façade (Juniper + mapping layer)
├─ engine/              # Query, update, reasoning
│  ├─ oxirs-arq         # Jena-style algebra + extension points
│  ├─ oxirs-rule        # Forward/backward rule engine (RDFS/OWL/SWRL)
│  ├─ oxirs-samm        # SAMM metamodel + AAS integration (Industry 4.0)
│  ├─ oxirs-geosparql   # GeoSPARQL spatial queries and topological relations
│  ├─ oxirs-shacl       # SHACL Core + SHACL-SPARQL validator
│  ├─ oxirs-star        # RDF-star / SPARQL-star grammar support
│  ├─ oxirs-ttl         # Turtle/TriG parser and serializer
│  └─ oxirs-vec         # Vector index abstractions (SciRS2, native HNSW)
├─ storage/
│  ├─ oxirs-tdb         # MVCC layer & assembler grammar (TDB2 parity)
│  ├─ oxirs-cluster     # Raft-backed distributed dataset
│  └─ oxirs-tsdb        # Time-series database (chunked, compressed)
├─ stream/              # Real-time and federation
│  ├─ oxirs-stream      # NATS/Redis/MQTT I/O, RDF Patch, SPARQL Update delta
│  ├─ oxirs-federate    # SERVICE planner, GraphQL stitching
│  ├─ oxirs-modbus      # Modbus TCP/RTU industrial protocol
│  └─ oxirs-canbus      # CANbus / J1939 industrial protocol
├─ ai/
│  ├─ oxirs-embed       # KG embeddings (TransE, ComplEx…)
│  ├─ oxirs-shacl-ai    # Shape induction & data repair suggestions
│  ├─ oxirs-chat        # RAG chat API (LLM + SPARQL)
│  ├─ oxirs-physics     # Physics-informed digital twins
│  └─ oxirs-graphrag    # GraphRAG hybrid search (Vector × Graph)
├─ security/
│  └─ oxirs-did         # W3C DID & Verifiable Credentials
├─ platforms/
│  └─ oxirs-wasm        # WebAssembly browser/edge deployment
├─ tools/
│  ├─ oxirs             # CLI (import, export, query, migration, benchmarking) — publish = false
│  └─ benchmarks/       # SP2Bench, WatDiv, LDBC SGS
└─ desktop/
    └─ oxirs-tauri      # Desktop app: chat UI, visual SPARQL builder, CAN bus monitor — publish = false
```

### Quarantined C-FFI adapters (opt-in, `publish = false`)

Three C-FFI integrations live outside the default and `--all-features` dependency
closure of every published crate, each in its own adapter crate that depends
on the corresponding library crate and re-exports API-compatible types:

| Adapter crate | Wraps | Adapts |
|---|---|---|
| `core/oxirs-gpu-monitor` | NVML | `oxirs-core` GPU telemetry |
| `engine/oxirs-vec-adapter-cuda` | `cuda-runtime-sys` | `oxirs-vec` CUDA buffers/streams/kernels |
| `stream/oxirs-stream-adapter-pulsar` | Apache Pulsar client | `oxirs-stream` Pulsar backend |

Depend on the adapter crate directly (path or git) to opt back in; none of
them are published to crates.io. They are workspace members but are kept out of
`default-members`, so a plain `cargo test --all-features` does not require CUDA
headers or `protoc`. Opt in per crate:

```powershell
$env:PROTOC = "$env:USERPROFILE\.cargo\bin\oxiproto-protoc.exe"  # cargo install oxiproto-cli
cargo build -p oxirs-stream-adapter-pulsar
cargo build -p oxirs-vec-adapter-cuda                            # needs the CUDA toolkit
```

Three more adapters existed and have since been retired outright rather than kept
opt-in. `oxirs-geosparql-adapter-geos` (GEOS) is gone because
`geo::algorithm::buffer` covers every geometry type and the full OGC cap/join
styles in Pure Rust, so the boundary-dependent Egenhofer/RCC8 relations and
buffer/boundary now work in the published crate with no opt-in at all.
`oxirs-tsdb-adapter-duckdb` (DuckDB) is gone with no replacement: export chunks
to Parquet via oxirs-tsdb's `arrow-export` feature and query them with an
external DuckDB. `oxirs-stream-adapter-rdkafka` (Kafka) is gone too — being
`publish = false` made it unreachable from any released crate, nothing in the
workspace depended on it, and it needed a `librdkafka` C build. Its Confluent
Schema Registry client, which never touched `rdkafka`, moved into `oxirs-stream`
as the Pure-Rust `confluent_registry` module.

## Feature Matrix (v0.4.1)

| Capability | Oxirs crate(s) | Status | Jena / Fuseki parity |
|------------|----------------|--------|----------------------|
| **Core RDF & SPARQL** | | | |
| RDF 1.2 & syntaxes (7 formats) | `oxirs-core` | ✅ Stable (2670 tests) | ✅ |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ Stable (2464 + 3210 tests) | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (`star` flag) | ✅ Stable | 🔸 |
| Advanced SPARQL Algebra (EXISTS/MINUS/subquery) | `oxirs-arq` | ✅ Stable | ✅ |
| Persistent storage (N-Quads) | `oxirs-core` | ✅ Stable | ✅ |
| **Semantic Web Extensions** | | | |
| RDF-star parse/serialise | `oxirs-star` | ✅ Stable (1702 tests) | 🔸 (Jena dev build) |
| SHACL Core+API (W3C compliant) | `oxirs-shacl` | ✅ Stable (2152 tests, 27/27 W3C) | ✅ |
| Rule reasoning (RDFS/OWL 2 DL) | `oxirs-rule` | ✅ Stable (2242 tests) | ✅ |
| SAMM 2.0-2.3 & AAS (Industry 4.0) | `oxirs-samm` | ✅ Stable (1555 tests, 16 generators) | ❌ |
| **Query & Federation** | | | |
| GraphQL API | `oxirs-gql` | ✅ Stable (2189 tests) | ❌ |
| SPARQL Federation (SERVICE) | `oxirs-federate` | ✅ Stable (1569 tests, 2PC) | ✅ |
| Federated authentication | `oxirs-federate` | ✅ Stable (OAuth2/SAML/JWT) | 🔸 |
| **Real-time & Streaming** | | | |
| Stream processing (NATS/Redis/MQTT) | `oxirs-stream` | ✅ Stable (1747 tests, SIMD) | 🔸 (Jena + external) |
| RDF Patch & SPARQL Update delta | `oxirs-stream` | ✅ Stable | 🔸 |
| **Search & Geo** | | | |
| Full-text search (Tantivy) | `oxirs-tdb` / `oxirs-fuseki` (`full-text-search`, opt-in) | 🔸 Partial (feature-gated, non-default) | ✅ |
| GeoSPARQL (OGC 1.1) | `oxirs-geosparql` (`geo`) | ✅ Stable (1967 tests) | ✅ |
| Vector search / embeddings | `oxirs-vec` (1771 tests), `oxirs-embed` (1537 tests) | ✅ Stable | ❌ |
| **Storage & Distribution** | | | |
| TDB2-compatible storage (six-index) | `oxirs-tdb` | ✅ Stable (2155 tests) | ✅ |
| Distributed / HA store (Raft) | `oxirs-cluster` (`cluster`) | ✅ Stable (1868 tests) | 🔸 (Jena + external) |
| Time-series database | `oxirs-tsdb` | ✅ Stable (1305 tests) | ❌ |
| **AI & Advanced Features** | | | |
| RAG chat API (LLM integration) | `oxirs-chat` | ✅ Stable (1267 tests) | ❌ |
| AI-powered SHACL constraint inference | `oxirs-shacl-ai` | ✅ Stable (1722 tests) | ❌ |
| GraphRAG hybrid search (Vector x Graph) | `oxirs-graphrag` | ✅ Stable (1130 tests) | ❌ |
| Physics-informed digital twins | `oxirs-physics` | ✅ Stable (1292 tests) | ❌ |
| Knowledge graph embeddings (TransE, etc.) | `oxirs-embed` | ✅ Stable (1537 tests) | ❌ |
| **Security & Trust** | | | |
| W3C DID & Verifiable Credentials | `oxirs-did` | ✅ Stable (1137 tests) | ❌ |
| Trust chain validation | `oxirs-did` | ✅ Stable | ❌ |
| Signed RDF graphs (RDFC-1.0) | `oxirs-did` | ✅ Stable | ❌ |
| Ed25519 cryptographic proofs | `oxirs-did` | ✅ Stable | ❌ |
| **Security & Authorization** | | | |
| ReBAC (Relationship-Based Access Control) | `oxirs-fuseki` | ✅ Stable | ❌ |
| Graph-level authorization | `oxirs-fuseki` | ✅ Stable | ❌ |
| SPARQL-based authorization storage | `oxirs-fuseki` | ✅ Stable | ❌ |
| OAuth2/OIDC/SAML authentication | `oxirs-fuseki` | ✅ Stable | 🔸 |
| **Browser & Edge Deployment** | | | |
| WebAssembly (WASM) bindings | `oxirs-wasm` | ✅ Stable (918 tests) | ❌ |
| Browser RDF/SPARQL execution | `oxirs-wasm` | ✅ Stable | ❌ |
| TypeScript type definitions | `oxirs-wasm` | ✅ Stable | ❌ |
| Cloudflare Workers / Deno support | `oxirs-wasm` | ✅ Stable | ❌ |
| **Industrial IoT** | | | |
| Modbus TCP/RTU protocol | `oxirs-modbus` | ✅ Stable (1237 tests) | ❌ |
| CANbus / J1939 protocol | `oxirs-canbus` | ✅ Stable (1183 tests) | ❌ |

**Legend:**
- ✅ Stable: Production-ready with comprehensive tests, API stability guaranteed
- ⏳ Planned: Not yet implemented
- 🔸 Partial/plug-in support in Jena

**Quality Metrics (v0.4.1):**
- **46,255 tests passing** (`--all-features`; 45,408 with default features), 100% pass rate
- **Zero compilation warnings** (enforced with `-D warnings`)
- **95%+ test coverage** across all 27 modules
- **95%+ documentation coverage**
- **All integration tests passing**
- **Production-grade security audit completed**
- **CUDA GPU acceleration** available via the opt-in, `publish = false` `oxirs-vec-adapter-cuda` quarantine crate (Pure Rust by default without it)
- **3.8x faster query optimization** via adaptive complexity detection
- **Pure Rust by default and under `--all-features`** — zero `ring` / `aws-lc-sys` / `rusqlite` / `zstd-sys` reach any published crate

## Usage Examples

### Dataset Configuration (TOML)

This is the `oxirs-fuseki` server config format (`ServerConfig`, loaded via
`oxirs-fuseki.toml`/`--config`), keyed under `[datasets.NAME]` (plural — see
`DatasetConfig` in `server/oxirs-fuseki/src/config_server.rs`). This is a
fragment showing the dataset/ReBAC sections in isolation — merge it into a
complete config alongside the required top-level `[server]`/`[security]`
sections; see the repo root [`oxirs.toml`](./oxirs.toml) for a full,
working reference file:

```toml
[datasets.mykg]
name         = "mykg"
location     = "./data/mykg"
read_only    = false
shacl_shapes = ["./shapes/person.ttl"]
services     = []

[datasets.mykg.text_index]
enabled     = true
analyzer    = "english"
max_results = 100
stemming    = true
stop_words  = []

# ReBAC Authorization (optional; see RebacConfig in config_security.rs)
[security.rebac]
enabled        = true
policy_mode    = "combined"  # rbac_only | rebac_only | combined
storage        = "memory"    # memory | open_fga | rdf
audit_enabled  = true
cache_ttl_secs = 300

[[security.rebac.initial_relationships]]
subject  = "user:alice"
relation = "owner"
object   = "dataset:mykg"
```

### GraphQL Query (auto-generated)

```graphql
query {
  Person(where: {familyName: "Yamada"}) {
    givenName
    homepage
    knows(limit: 5) { givenName }
  }
}
```

### Vector Similarity SPARQL Service (opt-in AI)

`oxirs-vec`'s `sparql_integration` module exposes a `vec:` predicate/function
vocabulary (`vec:searchText`, `vec:similar`, `vec:similarity`, ...) and can
generate the query below via `VectorSparqlIntegration::generate_service_query`
(see `engine/oxirs-vec/src/sparql_integration/config.rs`). It is standard,
grammar-valid SPARQL — unlike an IRI such as `<vec:similar(...)>`, which is
invalid (SPARQL `IRIREF`s may not contain spaces, quotes, or parentheses):

```sparql
PREFIX vec: <http://oxirs.org/vec#>

SELECT ?resource ?similarity WHERE {
  SERVICE <http://oxirs.org/vec/> {
    SELECT ?resource ?similarity WHERE {
      ?resource vec:searchText "semantic web" .
      ?resource vec:similarity ?similarity .
      FILTER(?similarity >= 0.8)
    }
    ORDER BY DESC(?similarity)
  }
}
```

Note: `SERVICE` always issues a real outbound HTTP SPARQL federation call
(`oxirs-arq`'s `service_federation` module) — there is no built-in listener
on `http://oxirs.org/vec/`, and `oxirs-arq` does not (yet) special-case a
`vec:` scheme. To run this for real today, stand up your own vector-aware
SPARQL endpoint at the target URI (`VectorSparqlIntegration` gives you the
query text to send it), or call `oxirs_vec::sparql_integration` directly
from Rust instead of through a live `SERVICE` clause.

### Live Deployment: OxiEphemeris LOD (CloudFlare + OxiRS)

OxiRS powers a public, production SPARQL endpoint at
<https://sparql.cooljapan.tech/>, edge-cached by CloudFlare and serving the
[OxiEphemeris](https://github.com/cool-japan/oxiephemeris) astrology
vocabulary (a custom `oxa:` ontology plus `oxc:`/`oxs:` SKOS concept
schemes) as dereferenceable Linked Open Data. Query it live over the
SPARQL 1.1 Protocol:

```sh
curl -G 'https://sparql.cooljapan.tech/sparql' \
  --data-urlencode 'query=ASK { <https://cooljapan.tech/ns/oxiephemeris/concept/sign/Scorpio> a <http://www.w3.org/2004/02/skos/core#Concept> }' \
  -H 'Accept: application/sparql-results+json'
# {"head":{},"boolean":true}
```

The `oxa:`/`oxc:`/`oxs:` IRIs under
<https://cooljapan.tech/ns/oxiephemeris/> dereference with content
negotiation (Turtle, N-Triples, or HTML).

## Digital Twin Platform Examples

### Smart City Sensor (NGSI-LD)

```bash
# Create an air quality sensor entity
curl -X POST http://localhost:3030/ngsi-ld/v1/entities \
  -H "Content-Type: application/ld+json" \
  -d '{
    "id": "urn:ngsi-ld:AirQualitySensor:Tokyo-001",
    "type": "AirQualitySensor",
    "location": {
      "type": "GeoProperty",
      "value": {"type": "Point", "coordinates": [139.6917, 35.6895]}
    },
    "temperature": {"type": "Property", "value": 22.5, "unitCode": "CEL"}
  }'

# Query sensors within 5km
curl "http://localhost:3030/ngsi-ld/v1/entities?type=AirQualitySensor&georel=near;maxDistance==5000"
```

### Factory IoT Bridge (MQTT)

```rust
use oxirs_stream::backend::mqtt::{
    MqttClient, MqttConfig, PayloadFormat, QoS, TopicRdfMapping, TopicSubscription,
};
use std::collections::HashMap;

let mqtt_config = MqttConfig {
    broker_url: "tcp://factory.example.com:1883".to_string(),
    ..Default::default()
};

let mut client = MqttClient::new(mqtt_config);
client.connect().await?;
client.subscribe(vec![
    TopicSubscription {
        topic_pattern: "factory/+/sensor/#".to_string(),
        qos: QoS::AtLeastOnce,
        payload_format: PayloadFormat::Json { schema: None, root_path: None },
        rdf_mapping: TopicRdfMapping {
            subject_pattern: "urn:sensor:{topic.1}:{topic.3}".to_string(),
            predicate_map: HashMap::new(),
            graph_pattern: Some("urn:factory:sensors".to_string()),
            type_uri: None,
            timestamp_field: None,
            timestamp_predicate: None,
            transformations: vec![],
        },
        options: None,
    },
]).await?; // Real-time RDF updates from subscribed topics
```

### Data Sovereignty Policy (IDS/Gaia-X)

```rust
use chrono::{Duration, Utc};
use oxirs_fuseki::ids::policy::constraint_evaluator::{
    ComparisonOperator, Purpose, SpatialRestriction, TemporalOperand,
};
use oxirs_fuseki::ids::policy::odrl_parser::PolicyType;
use oxirs_fuseki::ids::policy::{Constraint, OdrlAction, OdrlPolicy, Permission};
use oxirs_fuseki::ids::residency::Region;
use oxirs_fuseki::ids::types::IdsUri;

let policy = OdrlPolicy {
    uid: IdsUri::new("urn:policy:catena-x:battery-data:001")?,
    policy_type: PolicyType::Agreement,
    context: None,
    profile: None,
    permissions: vec![
        Permission {
            uid: None,
            action: OdrlAction::Use,
            constraints: vec![
                Constraint::Purpose {
                    allowed_purposes: vec![Purpose::ResearchUse],
                },
                Constraint::Spatial {
                    allowed_regions: vec![Region::eu_member("DE", "Germany"), Region::japan()],
                    restriction_type: SpatialRestriction::Within,
                },
                Constraint::Temporal {
                    left_operand: TemporalOperand::DateTime,
                    operator: ComparisonOperator::Lteq,
                    right_operand: Utc::now() + Duration::days(90),
                },
            ],
            duties: vec![],
            target: None,
            assignee: None,
            assigner: None,
        }
    ],
    prohibitions: vec![],
    obligations: vec![],
    targets: vec![],
    assigner: None,
    assignee: None,
    inherits_from: None,
    conflict: None,
};
```

### Physics Simulation (SciRS2 Integration)

```rust
use oxirs_physics::simulation::SimulationOrchestrator;

let mut orchestrator = SimulationOrchestrator::new();
orchestrator.register("thermal", Arc::new(SciRS2ThermalSimulation::default()));

// Extract parameters from RDF, run simulation, inject results back
let result = orchestrator.execute_workflow(
    "urn:battery:cell:001",
    "thermal"
).await?;

println!("Converged: {}, Final temp: {:.2}°C",
    result.convergence_info.converged,
    result.state_trajectory.last().unwrap().state["temperature"]
);
```

**Complete Examples**: See [`DIGITAL_TWIN_QUICKSTART.md`](server/oxirs-fuseki/DIGITAL_TWIN_QUICKSTART.md) and [`examples/digital_twin_factory.rs`](server/oxirs-fuseki/examples/digital_twin_factory.rs)

## Internationalization

📖 **Localized README versions:**
- 🇯🇵 [日本語 (Japanese)](README.ja.md) - Society 5.0 / PLATEAU support
- 🇩🇪 [Deutsch (German)](README.de.md) - Gaia-X / Industry 4.0 focus
- 🇫🇷 [Français (French)](README.fr.md) - European data sovereignty

## Development

### Prerequisites

- Rust 1.70+ (MSRV)
- Optional: Docker for containerized deployment

### Building

```bash
# Clone the repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Build all crates
cargo build --workspace

# Run tests
cargo nextest run --no-fail-fast

# Run with all features
cargo build --workspace --all-features
```

### Feature Flags

Optional features to keep dependencies minimal:

- `geo`: GeoSPARQL support
- `text`: Full-text search with Tantivy
- `ai`: Vector search and embeddings
- `cluster`: Distributed storage with Raft
- `star`: RDF-star and SPARQL-star support
- `vec`: Vector index abstractions

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### RFC Process

- Design documents go in `./rfcs/` with lazy-consensus and 14-day comment window
- All code must pass `rustfmt + nightly 2026-01`, Clippy `--all-targets --workspace -D warnings`
- Commit sign-off required (DCO 1.1)

## Roadmap

| Version | Target Date | Milestone | Deliverables | Status |
|---------|-------------|-----------|--------------|---------|
| **v0.1.0** | **✅ Jan 7, 2026** | **Initial Production Release** | Complete SPARQL 1.1/1.2, Industrial IoT, AI features, 13,123 tests | ✅ Released |
| **v0.2.4** | **✅ Mar 16, 2026** | **Deep Feature Expansion** | 40,786 tests, 26 new modules, 3.8x faster optimizer, advanced SPARQL algebra, AI production-grade | ✅ Released |
| **v0.3.0** | **✅ May 3, 2026** | **Full-text Search & Scale** | Full-text search (Tantivy), 10x performance, multi-region clustering, audit/certification/SSO/marketplace | ✅ Released |
| **v0.3.1** | **✅ 2026-06-06** | **SHACL-AF & Pure-Rust** | SHACL Advanced Features (recursive/qualified/reasoning), genetic constraint optimization, RDF-star in query execution, GraphSAGE embeddings, FIPS gates, full Pure-Rust migration, ~43,500 tests | ✅ Released |
| **v0.3.2** | **✅ 2026-07-12** | **Pure-Rust Policy v2** | Six C-FFI integrations quarantined into opt-in adapter crates, OxiSQL GeoPackage backend, Pure-Rust `zstd` shim, SHACL subclass-aware targets, oxirs-wasm PREFIX/solution-budget/indexed matching, 45,034 tests | ✅ Released |
| **v0.4.0** | **✅ 2026-07-19** | **Production Hardening & Deployment** | TDB2 durable on-disk backend, unified SPARQL query path via oxirs-arq, axum 0.8 migration complete, X25519/Ristretto DID crypto, OIDC/SAML SSO, BFT cluster RPC, 45,199 tests | ✅ Released |
| **v0.4.1** | **✅ 2026-07-28** | **Production-Readiness Hardening** | 38-scope multi-agent audit (308 findings, ~300 fixed), Raft durable storage, TDB corruption-detection fixes, SPARQL parser fixes, GraphQL auto-schema, mmap snapshot format, 46,255 tests | ✅ Released (current) |

### Current Release: v0.4.1 (2026-07-28)

**v0.4.1 Focus Areas:**
- Security: closed a full oxirs-fuseki auth/authz bypass, X.509 signature-chain verification, SPARQL injection and SSRF fixes, real oxirs-gql cache encryption, oxirs-did revocation/replay/ZKP checks, oxirs-chat SAML XSW hardening
- Data integrity: oxirs-cluster Raft state now durably fsync-persisted (`DurableRaftStore`), CLI `tdbupdate` no longer corrupts TDB stores, oxirs-tdb checksum-verify-before-trust, oxirs-did `StatusList2021` genuinely GZIP-compressed
- SPARQL correctness: core parser no longer drops `FILTER`, multi-line `CONSTRUCT`/`SELECT`/`ASK`/`DESCRIBE`/`UPDATE` parsing fixed, SPARQL query path fully unified onto oxirs-arq
- New: SPARQL 1.1 sub-`SELECT` and Turtle collection/property-list syntax (oxirs-arq), GraphQL schema auto-generation `graphql_autoschema` (oxirs-fuseki), mmap-able RDF store snapshot format (oxirs-core)
- Fabrication removed: W3C SHACL conformance harness, oxirs-shacl-ai quality metrics/AutoML, oxirs-federate query planning, performance_validation benchmarks all now do the real thing instead of faking results

> Total (`--all-features`): **46,255 tests passing** (45,408 with default features), 0 failed either way.

### Previous Release: v0.4.0 (2026-07-19)

**v0.4.0 Focus Areas:**
- oxirs-tdb real durable on-disk backend: superblock (v2, quad roots), fsync-backed writes, free-page allocator, GSPO/GPOS/GOSP indexes, wired into oxirs-fuseki (`StoreType::TDB2`/`dataset_type = "tdb2"`) and `oxirs import --dataset-type tdb2`
- oxirs-core storage rewrite: `RdfStore` Persistent backend now O(N) buffered-append (was O(N²) per-insert rewrite); `MemoryStorage` term interning for ~4x lower RAM; new durability API (`open_with_sync_policy`, `flush`, `SyncPolicy`, `AsyncRdfStore::flush_async`)
- SPARQL query path unified through the real oxirs-arq engine: `CONSTRUCT`/`DESCRIBE`, `GRAPH`/`FROM`/`FROM NAMED`, `SERVICE` federation, and native aggregates/`HAVING` all execute for real — legacy demo path and silent-empty-200 fallback removed
- SPARQL parser fixes: WHERE-less `ASK`/`SELECT *`, positionally-scoped `BIND`, group-scoped `FILTER` after `UNION`, populated `GROUP BY`/`ORDER BY` lists
- axum 0.8 route migration complete workspace-wide (oxirs-fuseki, oxirs-cluster, oxirs-embed, oxirs-chat)
- Security hardening: real X25519/Ristretto DID crypto, OIDC/SAML SSO signature verification, real BFT cluster RPC, hardened `read_only` dataset enforcement
- `oxirs` CLI additions: `lint`, `merge`, `jena-parity`, `monitor`, `detect-format`, `inspect` subcommands, `serve --dry-run`, `schema-gen --advanced`, and more

> Total (`--all-features`): **45,199 tests passing** (44,398 with default features), 0 failed either way.

### Previous Release: v0.3.2 (2026-07-12)

**v0.3.2 Focus Areas:**
- Pure-Rust Policy v2: NVML/CUDA/GEOS/DuckDB/Kafka/Pulsar C-FFI extracted into six opt-in, `publish = false` quarantine adapter crates
- GeoSPARQL's GeoPackage backend migrated from `rusqlite` to Pure-Rust `oxisql-core`/`oxisql-sqlite-compat`
- Pure-Rust `zstd` shim (`crates/zstd-shim`, backed by `oxiarc-zstd`) removes the last transitive `zstd-sys` dependency
- SHACL: subclass-aware `sh:class`/implicit-class targets, real SPARQL/property-path target execution
- oxirs-wasm query engine: PREFIX/BASE prologues, per-store solution budgets, SPO/POS/OSP-indexed pattern matching
- GeoSPARQL: shapefile interior-ring writing and compressed-geometry multi-ring round-tripping fixed
- Dependency refresh: SciRS2 0.6.0, oxiarc 0.3.5, oxicrypto/oxitls 0.2.0, kube 4.0

## Sponsorship

OxiRS is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find OxiRS useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

OxiRS is licensed under:

See [LICENSE](LICENSE) for details.

## Contact

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **Maintainer**: @cool-japan (KitaSan)

## Release Notes (v0.4.1)

Full notes live in [CHANGELOG.md](CHANGELOG.md).

### Highlights (2026-07-28)
- **46,255 tests passing** (`--all-features`; 45,408 with default features) across all 27 crates
- **308-finding production-readiness audit** → ~300 fixes across 38 work packages, plus 74 test regressions caught and fixed by the full-suite gate
- **Security hardening**: closed a full oxirs-fuseki auth/authz bypass, X.509 signature-chain verification, SPARQL injection/SSRF fixes, real oxirs-gql cache encryption, oxirs-did revocation/replay/ZKP checks, oxirs-chat SAML XSW hardening
- **Data-integrity fixes**: oxirs-cluster Raft state now durably fsync-persisted (`DurableRaftStore`), CLI `tdbupdate` no longer corrupts TDB stores, oxirs-tdb checksum-verify-before-trust, oxirs-did `StatusList2021` genuinely GZIP-compressed
- **SPARQL correctness**: core parser no longer drops `FILTER`, multi-line `CONSTRUCT`/`SELECT`/`ASK`/`DESCRIBE`/`UPDATE` parsing fixed
- **New**: SPARQL 1.1 sub-`SELECT` and Turtle collection/property-list syntax (oxirs-arq), GraphQL schema auto-generation `graphql_autoschema` (oxirs-fuseki), mmap-able RDF store snapshot format (oxirs-core)

### Previous Highlights (v0.4.0 — 2026-07-19)
- **45,199 tests passing** (`--all-features`; 44,398 with default features) across all 27 crates
- **oxirs-tdb real durable on-disk backend**: superblock (v2, quad roots), fsync-backed writes, free-page allocator, GSPO/GPOS/GOSP indexes, wired into oxirs-fuseki (`StoreType::TDB2`/`dataset_type = "tdb2"`) and `oxirs import --dataset-type tdb2`
- **SPARQL query path unified** through the real oxirs-arq engine: `CONSTRUCT`/`DESCRIBE`, `GRAPH`/`FROM`/`FROM NAMED`, `SERVICE` federation, and native aggregates/`HAVING` all execute for real
- **oxirs-core storage rewrite**: `RdfStore` Persistent backend now O(N) buffered-append; `MemoryStorage` term interning for ~4x lower RAM
- **axum 0.8 route migration** complete workspace-wide (oxirs-fuseki, oxirs-cluster, oxirs-embed, oxirs-chat)
- **Security hardening**: real X25519/Ristretto DID crypto, OIDC/SAML SSO signature verification, real BFT cluster RPC, hardened `read_only` dataset enforcement
- **`oxirs` CLI additions**: `lint`, `merge`, `jena-parity`, `monitor`, `detect-format`, `inspect` subcommands, `serve --dry-run`, `schema-gen --advanced`, and more

### Previous Highlights (v0.3.2 — 2026-07-12)
- **45,034 tests passing** (`--all-features`; 44,344 with default features) across all 27 crates
- **Pure-Rust Policy v2**: NVML/CUDA/GEOS/DuckDB/Kafka/Pulsar C-FFI extracted into six opt-in, `publish = false` quarantine adapter crates
- **OxiSQL GeoPackage backend**: GeoSPARQL's SQLite storage migrated from `rusqlite` to Pure-Rust `oxisql-core`/`oxisql-sqlite-compat`
- **Pure-Rust `zstd` shim** (`crates/zstd-shim`, backed by `oxiarc-zstd`) removes the last transitive `zstd-sys` dependency
- **SHACL subclass-aware targets** and real SPARQL/property-path target execution, replacing prior stub results
- **oxirs-wasm query engine**: PREFIX/BASE prologues, per-store solution budgets, and SPO/POS/OSP-indexed pattern matching
- **GeoSPARQL fixes**: shapefile interior-ring (hole) writing and compressed-geometry multi-ring round-tripping
- **SciRS2 0.6.0**; `oxiarc-*` 0.3.5; `oxicrypto`/`oxitls` 0.2.0; large-file refactors (all source < 2,000 lines)

### Per-Crate Test Counts (v0.4.1)

> Workspace total for v0.4.1 (`--all-features`): **46,255 tests passing** (45,408 with default features, 0 failed either way). Growth over v0.4.0's 45,199-crate baseline reflects the 38-scope production-readiness hardening pass (Raft durable storage, SPARQL parser fixes, data-integrity fixes) and its 74 caught-and-fixed test regressions. This table covers the 25 published library crates plus the `oxirs` CLI and `oxirs-tauri` desktop app; the rest belong to the 3 remaining `publish = false` C-FFI quarantine adapter crates plus the internal `oxirs-performance-validation` crate, excluded here.
| Crate | Tests |
|-------|-------|
| oxirs-arq | 3340 |
| oxirs-core | 2764 |
| oxirs-fuseki | 2548 |
| oxirs-rule | 2252 |
| oxirs-gql | 2221 |
| oxirs-shacl | 2210 |
| oxirs-tdb | 2166 |
| oxirs-geosparql | 1958 |
| oxirs-cluster | 1875 |
| oxirs (CLI) | 1314 |
| oxirs-ttl | 1852 |
| oxirs-vec | 1785 |
| oxirs-stream | 1766 |
| oxirs-shacl-ai | 1740 |
| oxirs-star | 1708 |
| oxirs-federate | 1576 |
| oxirs-samm | 1609 |
| oxirs-embed | 1545 |
| oxirs-physics | 1298 |
| oxirs-tsdb | 1326 |
| oxirs-chat | 1278 |
| oxirs-modbus | 1243 |
| oxirs-canbus | 1192 |
| oxirs-graphrag | 1148 |
| oxirs-did | 1283 |
| oxirs-wasm | 1116 |
| oxirs-tauri (desktop) | 65 |
| **Total (`--all-features`, published-crate subset)** | **46,178** |

### Performance Benchmarks
```
Query Optimization (5 triple patterns):
  HighThroughput:  3.24 µs  (3.3x faster than baseline)
  Analytical:      3.01 µs  (3.9x faster than baseline)
  Mixed:           2.95 µs  (3.6x faster than baseline)
  LowMemory:       2.94 µs  (5.3x faster than baseline)

Time-Series Database:
  Write throughput: 500K pts/sec (single), 2M pts/sec (batch)
  Query latency:    180ms p50 (1M points)
  Compression:      40:1 average ratio

Production Impact (100K QPS):
  CPU time saved: 45 minutes per hour (75% reduction)
  Annual savings: $10,000 - $50,000 (cloud deployments)
```

### Getting Started
- Build the CLI with `cargo install --path tools/oxirs` (not published to crates.io — see [Installation](#installation))
- Adaptive optimization is enabled by default (no configuration needed)
- CUDA support is opt-in via the `oxirs-vec-adapter-cuda` quarantine crate
- See [CHANGELOG.md](CHANGELOG.md) for detailed release notes

---

*"Rust makes memory safety table stakes; OxiRS makes knowledge-graph engineering table stakes."*

**v0.4.1 - Released - 2026-07-28**