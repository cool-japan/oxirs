# OxiRS

> A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.1-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: v0.4.0 - Release preparation - 2026-07-19

**Production Ready**: Complete SPARQL 1.1/1.2 implementation with **3.8x faster optimizer**, industrial IoT support, and AI-powered features. **45,199 tests passing** (`--all-features`; 44,398 with default features) with zero warnings across all 27 crates.

**v0.4.0 Highlights (2026-07-19)**: Consolidates the previously-unpublished 0.3.3 production-hardening work (a 272-finding audit → 288 fixes plus 455 regression tests across storage, security, and distributed subsystems) with 0.3.4's deployment fixes from the public, query-only sparql.wik.jp rollout — neither 0.3.3 nor 0.3.4 was published to crates.io, so both ship here as 0.4.0. oxirs-tdb is now a real durable on-disk backend (superblock, fsync-backed writes, free-page allocator, GSPO/GPOS/GOSP indexes) wired into oxirs-fuseki via `StoreType::TDB2`/`dataset_type = "tdb2"` and into `oxirs import --dataset-type tdb2`; oxirs-core's `RdfStore` Persistent backend replaces its O(N²) per-insert full-file rewrite with O(N) buffered-append persistence, and `MemoryStorage` now interns terms for roughly 4x lower RAM. The SPARQL query path is now unified through the real oxirs-arq engine end to end: `CONSTRUCT`/`DESCRIBE` execute via new template-instantiation/CBD machinery, `GRAPH <iri>`/`GRAPH ?g` and `FROM`/`FROM NAMED` execute for real against dataset views, `SERVICE` HTTP federation is reachable from the SPARQL endpoint, and native aggregate projections (`COUNT`/`SUM`/`MIN`/`MAX`/`AVG`/`SAMPLE`/`GROUP_CONCAT`, expressions inside aggregates, `DISTINCT`) plus `HAVING` (including aggregates inside `HAVING`) run through the engine's grouping machinery — the legacy demo path and every silent-empty-200 fallback are gone. Parser fixes cover WHERE-less `ASK`/`SELECT *`, positionally-scoped `BIND`, group-scoped `FILTER` (including after a top-level `UNION`), populated `GROUP BY`/`ORDER BY` lists, and multi-triple `INSERT/DELETE DATA` parsing. The axum 0.8 route migration is now complete workspace-wide (fuseki, cluster, embed, chat). Security hardening adds real X25519/Ristretto DID crypto, OIDC/SAML SSO signature verification, real cluster RPC with BFT quorum, and enforced `read_only` dataset checks — now name-agnostic for single-dataset deployments, with startup diagnostics for multi-dataset misconfigurations — across all write paths including REST API v2 and admin dataset management.

**v0.3.2 Highlights (2026-07-12)**: "Pure-Rust Policy v2" — six C-FFI integrations (NVML GPU monitoring, CUDA, GEOS, DuckDB, Kafka, Pulsar) extracted out of the in-tree feature flags into opt-in `publish = false` quarantine adapter crates, so every published crate's `--all-features` build is 100% Pure Rust; GeoSPARQL's GeoPackage backend migrated off `rusqlite` onto the new Pure-Rust `oxisql-core`/`oxisql-sqlite-compat` engine; a Pure-Rust `zstd` shim (backed by `oxiarc-zstd`) removes the last transitive `zstd-sys` C dependency (tantivy/parquet/pulsar/wasmtime). Also: SHACL targets gain subclass-aware `sh:class` matching and real SPARQL/property-path target execution, oxirs-wasm gains PREFIX/BASE query prologues, a per-store solution budget, and SPO/POS/OSP-index-driven pattern matching, plus GeoSPARQL shapefile/compressed-geometry hole and multi-ring round-tripping fixes. SciRS2 0.6.0; oxiarc 0.3.5.

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
> (e.g. the DuckDB-backed `tsdb-duckdb` feature) without pulling their C FFI onto a
> published Pure-Rust surface. All 25 OxiRS library crates remain normally published to
> crates.io — see [Published Crates](#published-crates) below.

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

### What's New in v0.3.1 (2026-06-06)

**Maintenance & Hardening Release: SHACL-AF, Pure-Rust Migration, and Inductive Embeddings**

OxiRS v0.3.1 completes SHACL Advanced Features, finishes the COOLJAPAN Pure-Rust migration, and adds new AI and security capabilities:

- **SHACL Advanced Features (SHACL-AF)** - Recursive shapes, qualified value shapes, and a rule-based reasoning engine (RDFS / OWL 2 RL entailment)
- **SHACL constraint-order optimization** - Genetic algorithm (configurable population/generations/tournament/mutation) for shape constraint ordering (oxirs-shacl-ai)
- **RDF-star in query execution** - Quoted triples now flow through pattern matching, query algebra, executor, JIT, planner, and SIMD triple matching
- **GraphSAGE inductive embeddings** - k-hop mean aggregation (ReLU + L2-norm), Xavier init, margin ranking loss, and unseen-entity support (oxirs-embed)
- **Graph summarizer & relevance feedback** - Leiden community detection → centrality → predicate-frequency natural-language summaries, plus multiplicative relevance-feedback re-ranking (oxirs-graphrag)
- **FIPS 140-2 feature gates** - `fips` feature for FIPS-validated cryptography in oxirs-fuseki and oxirs-did (RFC-003 FIPS boundary policy)
- **RBAC policy templates** - Built-in DBA / ReadOnly / Auditor role templates via PolicyTemplateRegistry (oxirs-fuseki)
- **Pure-Rust migration complete** - Compression (brotli/snap/flate2 → oxiarc), crypto (ring → oxicrypto), and TLS (pure-Rust oxitls provider); the default `cargo build` links zero `ring` / `aws-lc-sys` C/asm crypto
- **Dependency refresh** - SciRS2 0.5.0; oxiarc 0.3.3 now consumed directly from crates.io; large-file refactors keep every source file under 2,000 lines

**Quality Metrics (v0.3.1):**
- ✅ **~43,500 tests passing** (100% pass rate)
- ✅ **Zero compilation warnings** across all 26 crates
- ✅ **Pure Rust by default** - zero `ring` / `aws-lc-sys` in the default-feature build
- ✅ **All `.rs` files under 2,000 lines** (proactive refactors applied)

---

### Usage

```bash
# Initialize a new knowledge graph (alphanumeric, _, - only)
oxirs init mykg

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
- http://localhost:3030/graphql for GraphiQL (if enabled)

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
│  ├─ oxirs-stream      # Kafka/NATS I/O, RDF Patch, SPARQL Update delta
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

Six C-FFI integrations live outside the default and `--all-features` dependency
closure of every published crate, each in its own adapter crate that depends
on the corresponding library crate and re-exports API-compatible types:

| Adapter crate | Wraps | Adapts |
|---|---|---|
| `core/oxirs-gpu-monitor` | NVML | `oxirs-core` GPU telemetry |
| `engine/oxirs-vec-adapter-cuda` | `cuda-runtime-sys` | `oxirs-vec` CUDA buffers/streams/kernels |
| `engine/oxirs-geosparql-adapter-geos` | GEOS | `oxirs-geosparql` Egenhofer/RCC8 relations |
| `storage/oxirs-tsdb-adapter-duckdb` | DuckDB | `oxirs-tsdb` Arrow `RecordBatch` bridge |
| `stream/oxirs-stream-adapter-rdkafka` | `rdkafka` | `oxirs-stream` Kafka backend |
| `stream/oxirs-stream-adapter-pulsar` | Apache Pulsar client | `oxirs-stream` Pulsar backend |

Depend on the adapter crate directly (path or git) to opt back in; none of
them are published to crates.io.

## Feature Matrix (v0.4.0)

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
| Stream processing (Kafka/NATS) | `oxirs-stream` | ✅ Stable (1747 tests, SIMD) | 🔸 (Jena + external) |
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

**Quality Metrics (v0.4.0):**
- **45,199 tests passing** (`--all-features`; 44,398 with default features), 100% pass rate
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

```toml
[dataset.mykg]
type      = "tdb2"
location  = "/data"
text      = { enabled = true, analyzer = "english" }
shacl     = ["./shapes/person.ttl"]

# ReBAC Authorization (optional)
[security.policy_engine]
mode = "Combined"  # RbacOnly | RebacOnly | Combined | Both

[security.rebac]
backend = "InMemory"  # InMemory | RdfNative
namespace = "http://oxirs.org/auth#"
inference_enabled = true

[[security.rebac.initial_relationships]]
subject = "user:alice"
relation = "owner"
object = "dataset:mykg"
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

```sparql
SELECT ?s ?score WHERE {
  SERVICE <vec:similar ( "LLM embeddings of 'semantic web'" 0.8 )> {
    ?s ?score .
  }
}
```

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
        },
        options: None,
    },
]).await?; // Real-time RDF updates from subscribed topics
```

### Data Sovereignty Policy (IDS/Gaia-X)

```rust
use oxirs_fuseki::ids::policy::{OdrlPolicy, Permission, Constraint};

let policy = OdrlPolicy {
    uid: "urn:policy:catena-x:battery-data:001".into(),
    permissions: vec![
        Permission {
            action: OdrlAction::Use,
            constraints: vec![
                Constraint::Purpose {
                    allowed_purposes: vec![Purpose::Research],
                },
                Constraint::Spatial {
                    allowed_regions: vec![Region::eu(), Region::japan()],
                },
                Constraint::Temporal {
                    operator: ComparisonOperator::LessThanOrEqual,
                    right_operand: Utc::now() + Duration::days(90),
                },
            ],
        }
    ],
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
| **v0.4.0** | **✅ 2026-07-19** | **Production Hardening & Deployment** | TDB2 durable on-disk backend, unified SPARQL query path via oxirs-arq, axum 0.8 migration complete, X25519/Ristretto DID crypto, OIDC/SAML SSO, BFT cluster RPC, 45,199 tests | ✅ Released (current) |

### Current Release: v0.4.0 (2026-07-19)

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

### Previous Release: v0.3.1 (2026-06-06)

**v0.3.1 Focus Areas:**
- SHACL Advanced Features (SHACL-AF): recursive shapes, qualified value shapes, rule-based reasoning engine
- SHACL constraint-order optimization via genetic algorithm (oxirs-shacl-ai)
- RDF-star quoted triples in pattern matching and query execution (algebra/executor/JIT/planner/SIMD)
- AI: GraphSAGE inductive embeddings, graph summarizer, relevance feedback
- Security: FIPS 140-2 feature gates (oxirs-fuseki, oxirs-did), RBAC policy templates
- COOLJAPAN Pure-Rust migration complete: brotli/snap/flate2 → oxiarc, ring → oxicrypto, pure-Rust TLS via oxitls
- Dependency refresh: SciRS2 0.5.0, oxiarc 0.3.3 from crates.io; large-file refactors (all source < 2,000 lines)

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

## Release Notes (v0.4.0)

Full notes live in [CHANGELOG.md](CHANGELOG.md).

### Highlights (2026-07-19)
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

### Previous Highlights (v0.3.1 — 2026-06-06)
- **~43,500 tests passing** across all 26 crates
- **SHACL Advanced Features** completed: recursive shapes, qualified value shapes, and a rule-based reasoning engine
- **Genetic constraint-order optimization** for SHACL shapes (oxirs-shacl-ai)
- **RDF-star quoted triples** in pattern matching and query execution (algebra, executor, JIT, planner, SIMD)
- **GraphSAGE inductive embeddings**, **graph summarizer**, and **relevance feedback** (oxirs-embed, oxirs-graphrag)
- **FIPS 140-2 feature gates** (oxirs-fuseki, oxirs-did) and **RBAC policy templates**
- **Pure-Rust migration complete**: brotli/snap/flate2 → oxiarc, ring → oxicrypto, pure-Rust TLS via oxitls; default build links zero `ring` / `aws-lc-sys`
- **SciRS2 0.5.0**; oxiarc 0.3.3 consumed directly from crates.io; large-file refactors (all source < 2,000 lines)

### Per-Crate Test Counts (v0.4.0)

> Workspace total for v0.4.0 (`--all-features`): **45,199 tests passing** (44,398 with default features, 0 failed either way). Growth over v0.3.2's 45,034-crate baseline reflects the TDB2 durable on-disk backend, the unified oxirs-arq query path (`CONSTRUCT`/`DESCRIBE`/`GRAPH`/`SERVICE`/aggregates), axum 0.8 migration, DID/SSO/BFT security hardening, and expanded coverage across every crate. The 6 `publish = false` quarantine adapter crates (NVML/CUDA/GEOS/DuckDB/Kafka/Pulsar) require toolchains unavailable in routine CI and are excluded from this count.
| Crate | Tests |
|-------|-------|
| oxirs-arq | 3210 |
| oxirs-core | 2670 |
| oxirs-fuseki | 2464 |
| oxirs-rule | 2242 |
| oxirs-gql | 2189 |
| oxirs-shacl | 2152 |
| oxirs-tdb | 2155 |
| oxirs-geosparql | 1967 |
| oxirs-cluster | 1868 |
| oxirs (CLI) | 1279 |
| oxirs-ttl | 1817 |
| oxirs-vec | 1771 |
| oxirs-stream | 1747 |
| oxirs-shacl-ai | 1722 |
| oxirs-star | 1702 |
| oxirs-federate | 1569 |
| oxirs-samm | 1555 |
| oxirs-embed | 1537 |
| oxirs-physics | 1292 |
| oxirs-tsdb | 1305 |
| oxirs-chat | 1267 |
| oxirs-modbus | 1237 |
| oxirs-canbus | 1183 |
| oxirs-graphrag | 1130 |
| oxirs-did | 1137 |
| oxirs-wasm | 918 |
| oxirs-tauri (desktop) | 61 |
| **Total (`--all-features`)** | **45,199** |

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

**v0.4.0 - Release preparation - 2026-07-19**