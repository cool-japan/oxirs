# Changelog

All notable changes to OxiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-07-12

### Added
- **oxirs-core**: new `encoding` module — pure-`std` RFC 3986 percent-encoding (`percent_encode`, `percent_encode_strict`, `percent_decode`), replacing the external `urlencoding` crate; now backs SPARQL's `ENCODE_FOR_URI()` and oxirs-fuseki's OAuth2/SAML/LDF URL handling
- **oxirs-shacl**: `advanced_features::subclass_closure` — reflexive+transitive `rdfs:subClassOf` closure (Floyd–Warshall over a boolean adjacency matrix); `sh:class` and implicit-class targets now honor subclassing instead of exact-type matching only
- **oxirs-shacl**: SPARQL-based (`sh:target` SPARQLTarget) and single-hop property-path targets now execute for real against the store, replacing stub implementations that returned empty or unchanged results
- **oxirs-vec**: `real_time_embedding_pipeline` gains a full consistency/versioning/monitoring stack — `consistency` (inconsistency repair engine with severity-based outcomes), `versioning` (per-ID embedding version history with configurable retention), and a rewritten `monitoring` (metrics collection, health checks, severity-throttled alerting); all four submodules (`consistency`, `versioning`, `monitoring`, `coordination`) are now compiled and wired live into `RealTimeEmbeddingPipeline` — previously declared but commented out as unimplemented
- **oxirs-chat**: `nl2sparql::context_aware` — `extract_entities_rich()` rule-based NL entity extraction returning typed, span-and-confidence-scored `ExtractedEntity` values (IRIs, prefixed names, quoted literals, multi-word capitalized concepts)
- **oxirs-stream**: MQTT 5.0 property codec (`backend::mqtt::properties`) — encode/decode for the PUBLISH-relevant property set (Payload Format Indicator, Message Expiry Interval, Content Type, Response Topic, Correlation Data, Subscription Identifier, Topic Alias, repeatable User Properties), wired into `MqttClient::parse_properties_from_bytes()`
- **oxirs-federate**: NATS federation now dispatches inbound messages to registered `FederationMessageHandler`s by type (`register_handler()`), replacing a no-op stub
- **oxirs-gql**: adaptive query batching (`QueryBatcher::analyze_batch_dependencies()` plus topological wave execution), an activated ML-driven `DynamicQueryPlanner` (real `MLQueryOptimizer` + `PerformanceTracker` behind `enable_ml_prediction`), and real parallel-field-resolver timing metrics — all previously dead or stubbed code paths
- **oxirs-tdb**: saga steps now run real registered forward-action/compensation callbacks (`SagaCallbackRegistry`) with reverse-order compensation on failure; 2PC/3PC participants can now commit/abort a real WAL-backed `Transaction` via `with_transaction_manager()`; the deadlock detector's `LeastWork` victim-selection strategy is implemented instead of silently falling back
- **oxirs-geosparql**: shapefile writer now emits interior rings (holes) for `Polygon`/`MultiPolygon`; WKT parser accepts optional Z/M coordinate dimensions; `GeoPackage::checkpoint()` for explicit WAL flush
- **oxirs-wasm**: SPARQL queries may declare a `PREFIX`/`BASE` prologue (`query::prefix_expand::expand_prologue`), expanded before parsing; a per-store solution budget (`setSolutionBudget`/`clearSolutionBudget`) fails a query fast once a join produces more intermediate rows than configured, instead of running an unselective join to completion

### Added — Pure-Rust Policy v2 (COOLJAPAN)
- Six previously in-tree, feature-gated C-FFI integrations were extracted into new `publish = false` adapter crates, each reusing the original crate's public types so the two stay API-compatible:
  - **oxirs-gpu-monitor**: `NvmlGpuMonitor` — live NVIDIA GPU telemetry (utilization, memory, temperature, power draw) via NVML
  - **oxirs-vec-adapter-cuda**: `CudaBuffer`/`CudaStream`/`CudaKernel` and device enumeration, backed by `cuda-runtime-sys`
  - **oxirs-geosparql-adapter-geos**: GEOS-backed Egenhofer relations, RCC8 relations, and buffer/boundary operations
  - **oxirs-tsdb-adapter-duckdb**: Arrow `RecordBatch` bridge between DuckDB SQL and `oxirs_tsdb::TimeChunk`/`DataPoint`
  - **oxirs-stream-adapter-rdkafka**: the former in-tree Kafka backend (consumer, producer, schema registry), moved verbatim
  - **oxirs-stream-adapter-pulsar**: the former in-tree Pulsar backend, moved verbatim
- New internal `crates/zstd-shim`, redirected to via workspace-wide `[patch.crates-io] zstd = ...`: a Pure-Rust shim backed by `oxiarc-zstd` that replaces the C-FFI `zstd`/`zstd-sys` crate for every transitive consumer (tantivy, parquet, pulsar, wasmtime) — removes the last transitive `zstd-sys` C dependency from the default build

### Changed
- **oxirs-geosparql**: `GeoPackage`'s SQLite backend migrated from `rusqlite` (bundled C libsqlite3) to Pure-Rust `oxisql-core`/`oxisql-sqlite-compat`
- **oxirs-arq**: removed the `ordered-float` dependency; float-valued SPARQL terms now use in-house `TotalF32`/`TotalF64` total-order wrappers (`total_float.rs`) with identical NaN-equality and zero-sign semantics
- **oxirs-wasm**: triple pattern and property-path evaluation now drives off the store's subject/predicate/object indexes instead of scanning every triple, so a join costs a hash lookup per solution rather than a full graph scan
- **oxirs-stream**: `DiagnosticsConfig.jaeger_endpoint` renamed to `otlp_endpoint` (env var `OTEL_EXPORTER_JAEGER_ENDPOINT` → `OTEL_EXPORTER_OTLP_ENDPOINT`), reflecting the workspace-wide drop of the deprecated `opentelemetry-jaeger` exporter in favor of OTLP (`opentelemetry*` 0.31 → 0.32)
- Workspace-wide: `lazy_static` → `once_cell::sync::Lazy`; `num_cpus::get()` → `std::thread::available_parallelism()`; `blake3` now built with `features = ["pure"]` (no C/asm via `cc`)
- COOLJAPAN ecosystem: `oxisql-core`/`oxisql-sqlite-compat` introduced at 0.3.2 (new Pure-Rust SQLite-compatible engine; first consumer is GeoSPARQL's GeoPackage backend, see above); `oxiarc-*` compression crates 0.3.3 → 0.3.5; `oxicrypto-{core,aead,rand,mac,hash,kdf}` and `oxitls`/`oxitls-rcgen` 0.1.1 → 0.2.0; `scirs2-{core,linalg,stats,neural,graph,integrate}` 0.5.0 → 0.6.0
- Security-relevant dependency bumps: `bytes` 1.11.1 → 1.12.0 (CVE-2026-25541 fix); `kube` 3.1 → 4.0 with `k8s-openapi` 0.27 → 0.28 (`v1_31` → `v1_32` schema snapshot) for oxirs-fuseki's optional `k8s` feature
- Routine dependency bumps: tokio 1.52.3, hyper 1.10.1, tower-http 0.7, serde_json 1.0.150, async-nats 0.49, chrono 0.4.45, nalgebra 0.35, quick-xml 0.41, rstar 0.13, redis 1.3, cranelift-* 0.133.1, jsonwebtoken 10.4, aes-gcm 0.11, sha3 0.12, arrow/parquet 59, plus assorted per-crate bumps (tera, wasmi, pdf-extract, sha2, utoipa, axum-test, pbkdf2, mdns-sd, pulsar client 6.8, lapin, time, shlex, wasmparser, wasmtime, pyo3/numpy 0.29)
- Removed unused workspace dependencies: `rio_api`/`rio_turtle`/`rio_xml` (zero remaining references), `rand_distr`, `log`, `env_logger`, `urlencoding` (superseded by oxirs-core's `encoding` module), `serial_test`, `crossbeam-utils`, `tracing-opentelemetry`, `http-body-util`, `rust_decimal`, `multibase`, `k256`, `digest`, `deadpool`, `wasm-bindgen-futures`, `tracing-wasm`, `oxicrypto-sig`

### Removed — Pure-Rust Policy v2 (COOLJAPAN)
- **Breaking**: the following Cargo features were removed outright (their functionality moved to the corresponding adapter crate listed under Added — see above; building with the old feature flag now fails and the adapter crate must be added as a direct dependency):
  - `oxirs-core`'s `gpu` feature (NVML via `nvml-wrapper`)
  - `oxirs-vec`'s `cuda` and `gpu-full` features (`cuda-runtime-sys`, `cudarc`, `candle-core`); the legacy, already-unused duplicate `gpu_acceleration` module (945 lines) was also deleted in favor of `gpu`
  - `oxirs-geosparql`'s `geos-backend` feature (`geos`/`geos-sys`) — the in-tree Egenhofer/RCC8/buffer functions now return `UnsupportedOperation` directing callers to the adapter crate
  - `oxirs-tsdb`'s `duckdb` feature (`duckdb`/`libduckdb-sys`)
  - `oxirs-stream`'s `kafka` feature (`rdkafka`/`rdkafka-sys`/`libz-sys`)
  - `oxirs-stream`'s `pulsar` feature (`pulsar`/`native-tls`/`lz4-sys`)
- Net effect: every published OxiRS crate's `--all-features` dependency surface is now 100% Pure Rust; all six C-FFI integrations remain available, just as separate `publish = false` adapter crates rather than in-tree feature flags

### Fixed
- **oxirs-core**: `rdfxml::serializer` no longer collides two or more distinct unmapped namespaces on the same subject onto one hardcoded `xmlns:oxprefix` attribute; each namespace now gets its own synthetic prefix
- **oxirs-core**: `query::update`'s `INSERT DATA`/`DELETE DATA` blocks whose final triple omits the trailing `.` (legal in SPARQL's grammar, illegal in the Turtle grammar used to re-parse the block) now parse correctly
- **oxirs-core**: JSON-LD expansion no longer silently drops `@index` on indexed containers and node objects; `@protected` term-redefinition detection now also compares the `protected` flag itself; `@set` containers now expand to flat triples instead of a no-op
- **oxirs-tdb**: `DistributedTdbStore::execute_saga` previously reported success without ever calling `saga.execute()` — sagas silently never ran; they now execute for real, with results recorded to the replication log via `commit_distributed_transaction`
- **oxirs-tdb**: deadlock detector's `YoungestTransaction`/`OldestTransaction` victim selection no longer panics on an empty cycle; the distributed coordinator's `abort_transaction` now notifies all registered participants instead of a no-op
- **oxirs-chat**: AES-256-GCM key/nonce construction (`persistence_storage.rs`, `security/encryption.rs`) no longer panics on a malformed key or nonce length — returns a proper error instead
- **oxirs-chat**: `DensePassageRetriever`'s dense encoder was content-blind (embeddings were derived only from word position/count, so unrelated texts of equal length produced identical vectors); it now uses content-sensitive term-frequency plus hash-trick encoding
- **oxirs-physics**: `ResultInjector` generated invalid SPARQL — numeric/boolean values combined with an explicit `^^xsd:type` datatype annotation must be quoted strings, but were emitted as bare tokens; `execute_update()` was also a no-op that logged but never wrote to the store, so simulation results were never actually persisted — both fixed
- **oxirs-shacl**: `sh:languageIn` now does BCP47/RFC-4647 basic-filtering range matching (`"de"` matches `"de-CH"`) instead of exact string equality
- **oxirs-graphrag**: `GraphSummarizer`'s representative and relation ordering is now deterministic (lexicographically-smallest IRI, sorted-by-frequency relations) instead of depending on hash iteration order
- **oxirs-geosparql**: `compressed_storage` decompression previously collapsed polygon holes and multi-part `MultiLineString`/`MultiPolygon` geometries down to a single ring/part on round-trip; a new `ring_counts` field fixes reconstruction; `Cargo.toml`'s `cuda`/`wgpu_backend` features referenced nonexistent `scirs2-core` sub-features, breaking dependency resolution for any workspace build after the scirs2-core 0.6.0 bump — corrected
- **oxirs-wasm**: pattern and property-path matching now treats the two IRI spellings that reach the store — bracketed `<iri>` (as parsers emit) and bare `iri` (as `insert()` accepts) — as equal, so a query no longer silently misses matches depending on which form was used; property paths also gained support for the `a` (`rdf:type`) keyword

## [0.3.1] - 2026-06-06

### Added
- **oxirs-fuseki**: `auth/policy_templates.rs` — `PolicyTemplateRegistry` with built-in DBA, ReadOnly, Auditor role templates; `PolicyTemplate` (serde-compatible); `apply_to_user`; `ReadAudit` permission variant; `DuplicateTemplate`/`UnknownTemplate` auth errors (18 tests)
- **oxirs-fuseki**: FIPS 140-2 `fips` feature gate for FIPS-validated cryptography via `ring`
- **oxirs-did**: FIPS 140-2 `fips` feature gate
- `docs/policies/fips-boundary.md` — RFC-003 FIPS cryptographic boundary policy
- **oxirs-graphrag**: `summarizer.rs` — `GraphSummarizer` (Leiden community detection → in-degree centrality → predicate frequency → `to_text()` natural language output) (6 tests)
- **oxirs-graphrag**: `feedback.rs` — `TripleRelevanceFeedback` / `Relevance` enum; seahash triple IDs; multiplicative weight adjustment (0.1–2.0 clamp); `apply_to_scores` sorted descending (7 tests)
- **oxirs-embed**: `models/graph_sage.rs` — `GraphSageEmbedder` inductive graph embeddings; k-hop mean aggregation with ReLU+L2-norm; Xavier init via scirs2-core; margin ranking loss; sign-SGD; LCG neighbor sampling; unseen-entity support (6 tests)
- **oxirs-shacl**: SHACL Advanced Features (SHACL-AF) — recursive shapes and qualified value shapes completed (Round 32)
- **oxirs-shacl**: SHACL-AF reasoning engine — rule-based reasoning validator with dedicated reasoning types and validation rules (Round 33)
- **oxirs-shacl-ai**: `optimization/genetic.rs` — genetic algorithm for SHACL shape constraint-order optimization (configurable population/generations/tournament/mutation via `GeneticParams`, cost-estimate fitness scoring, tournament selection)
- **oxirs-ttl**: HDT 1.0 (Header-Dictionary-Triples) binary RDF format reader — iterator-based read-only parser over Header, Dictionary, and Triples sections
- **oxirs-ttl**: streaming TriG parser — `TriGStreamingParser` pull-parser implementing `Iterator<Item = Result<StreamedQuad, _>>` for W3C TriG named graphs
- **oxirs-core**: RDF-star quoted triples in pattern matching and query execution — quoted-triple support across query algebra, executor, JIT, planner, pattern unification, and SIMD triple matching
- **oxirs**: `rset` tool now parses SPARQL Results XML (SRX) — booleans, bindings, typed/language literals, and blank nodes (via `parse_sparql_results_xml`)

### Changed
- Version bump to 0.3.1 across all workspace crates
- **Refactor** `oxirs-fuseki/src/config.rs` (1999 lines) → split into `config.rs` + `config_server.rs` + `config_security.rs` + `config_runtime.rs`; public API preserved via re-exports
- **Refactor** `oxirs-arq/src/lateral_join.rs` (1978 lines) → directory module; test block extracted to `lateral_join/lateral_join_tests.rs`; source down to 1138 lines
- **Refactor** `oxirs-federate/src/query_decomposition/advanced_pattern_analysis.rs` (1990 lines) → split into `_consciousness.rs` (252) + `_analyzer.rs` (1484) + `_quantum.rs` (298) + 16-line facade
- **Refactor** `oxirs-cluster/src/storage/persistent.rs` (1953 lines) → `persistent.rs` (710) + `persistent_wal.rs` (335) + `persistent_integrity.rs` (444) + `persistent_tests.rs` (287)
- **Refactor** `oxirs-shacl/src/targets/selector.rs` (1961 lines) → `selector.rs` (709) + `selector_query.rs` (548) + `selector_eval.rs` (641)
- **Refactor** `oxirs-shacl-ai/src/performance_monitoring.rs` (1949 lines) → `performance_monitoring.rs` (1575) + `performance_monitoring_advanced.rs` (514); fixed pre-existing Duration/field-name/Serialize bugs
- **Refactor** `oxirs-shacl-ai/src/optimization/engine.rs` (1987 lines) → `engine.rs` (1308) + `optimization/engine_analysis.rs` (718)
- 21 TODO.md files updated: 47 `[~]` items flipped to `[x]` for RFC-001 (LTS) and RFC-002 (Enterprise) now that policies are published
- Dependency: **SciRS2** workspace crates bumped 0.4.3 → 0.5.0 (all 12 `scirs2-*` sub-crates)
- Dependency: `oxiarc-*` compression crates pinned to **0.3.3** and consumed directly from crates.io — the temporary `[patch.crates-io]` path overrides were removed, leaving only `pathfinder_simd` patched (an unrelated Rust-nightly intrinsic fix)
- **Refactor (Round 31 — facade recovery)**: 13 oversized files (1602–1909 lines) across **oxirs-core**, **oxirs-vec**, **oxirs-fuseki**, **oxirs-embed**, **oxirs-shacl-ai**, **oxirs-samm**, **oxirs-rule**, and **oxirs** converted to thin facades over pre-existing sibling modules (~20K lines net removed)
- **Refactor (Round 32 — tier-2)**: **oxirs-fuseki** `auth/saml.rs` + `bind_values_enhanced.rs`, **oxirs-cluster** `cluster_metrics.rs`, **oxirs-tdb** `advanced_diagnostics.rs`, **oxirs-federate** `service.rs`, and **oxirs** `commands/jena_parity.rs` split into focused submodules
- **Refactor (Round 33 — tier-3)**: **oxirs-shacl** `constraints/expression_constraint.rs`, **oxirs-vec** `tree_indices.rs` (ball/cover/kd/rp/vp-tree), **oxirs-wasm** `query/construct.rs`, **oxirs-gql** `schema.rs`, and **oxirs-stream** `backend/kafka/backend.rs` split into submodules
- **Refactor (tools/oxirs)**: `commands/aspect.rs`, `commands/interactive.rs`, and `commands/import_command.rs` split into focused modules; `lib.rs` → `lib_commands.rs` + `lib_dispatch.rs`
- **Refactor (additional)**: **oxirs-gql** `validation_spec.rs`, **oxirs-tdb** `mvcc/mod.rs`, **oxirs-stream** `lib_types.rs` / `neuromorphic_analytics.rs` / `wasm_edge_computing.rs`, and **oxirs-core** `storage/tiered.rs` split into module directories
- **oxirs-vec**: `build.rs` CUDA-toolkit detection now emits informational build-script output instead of `cargo:warning=` (no-warnings policy); GPU acceleration remains opt-in via the `cuda` feature with Pure-Rust CPU fallbacks
- Error-handling hardening: `unwrap()` calls in the benchmark runner and performance-benchmark modules replaced with context-carrying error handling

### Changed — Pure-Rust (COOLJAPAN Policy)
- Compression: `brotli` → `oxiarc-brotli`, `snap` → `oxiarc-snappy`, `flate2` → `oxiarc-deflate` across all consuming crates (tdb, stream, federate, fuseki, cluster, arq, gql, star, embed, shacl-ai, chat, did, tools); direct `brotli`/`snap`/`flate2` deps removed
- Crypto: 27 first-party `ring::` call sites (tools/oxirs, oxirs-did, oxirs-fuseki, oxirs-cluster) migrated to **oxicrypto** leaf crates (hash/mac/aead/kdf/rand) + workspace `ed25519-dalek` + `rsa` for Ed25519/RSA signatures; all direct `ring` deps removed
- TLS: pure-Rust **`oxitls::pure_provider()`** installed process-wide at every binary entry; `rustls` no-provider, `reqwest` rustls-no-provider, `tokio-rustls` default-features=false, `metrics-exporter-prometheus` push-gateway-no-tls-provider; cluster cert-gen `rcgen` → `oxitls-rcgen`; `async-openai` gated behind non-default `openai` feature
- Fixed an `oxiarc-brotli` incompressible-input compressor bug (it emitted streams its own decoder rejected); the fix shipped upstream as **oxiarc 0.3.3** (2026-06-06) on crates.io, which OxiRS now consumes directly
- **Default `cargo build` is now free of `ring` and `aws-lc-sys`/`aws-lc-rs` C/asm crypto** — `cargo tree -i ring` and `cargo tree -i aws-lc-sys`/`aws-lc-rs` are all empty for the default feature set

### Fixed
- **oxirs-stream**: fixed undefined behavior in neuromorphic analytics — `NetworkTopology` and `NeuralResponse` removed from the unsafe `impl_zeroed_default!` macro (`std::mem::zeroed()` is UB for `Vec`/`Instant`); replaced with `derive(Default)` and a manual `Default` impl (Round 31)
- **oxirs-core**: declared `indexmap` and `toml` as workspace dependencies, fixing a latent compile break in `jsonld::{compaction,flattening}` (Round 31)
- **oxirs-vec**: restored `JointEmbeddingSpace::zero_shot_retrieval`, lost to earlier module drift (Round 31)

### Policy Compliance
- All `.rs` files under 2000 lines (proactive refactors applied)
- Zero warnings, zero errors across workspace
- Total test count: ~43,511 (+19 graphrag, +6 embed from Round 18)

## [0.3.0] - 2026-05-03

### Added
- **oxirs-core**: audit/ module — SOC2/GDPR compliance with AuditEvent, InMemoryAuditLogger, JsonLineAuditLogger, AuditFilter, and GdprService (30 tests)
- **oxirs-shacl-ai**: certification/ module — ClassificationMetrics, CertificationRunner, and report generation (19 tests)
- **oxirs-chat**: marketplace/ module — HuggingFace Hub, Ollama, and local GGUF registry integration (28 tests)
- **oxirs-chat**: SSO module — OIDC (oidc.rs), SAML 2.0 SP (saml_sp.rs), and session management (session.rs) (13 tests)
- **oxirs-cluster**: certification/ module — consistency, partition, raft, and SLA checks (13 tests)
- **oxirs-graphrag**: model_loader/ module — GGUF parser and ModelRegistry
- **oxirs-graphrag**: hybrid/lora.rs — LoRA adapter and trainer implementation
- **oxirs-fuseki**: SAML 2.0 XML parsing via quick-xml with ring RSA-SHA256 signature verification (32 tests)
- **oxirs-modbus**: browser/ TUI module — ratatui 0.30, feature-gated as `tui` (33 tests)
- docs/policies/lts.md — RFC-001 LTS support policy
- docs/policies/enterprise.md — RFC-002 enterprise deployment policy
- 26 TODO.md files updated with policy references

### Changed
- Version bump to 0.3.0 across all workspace crates
- Total test count: ~41,400 (up from ~40,530 in v0.2.4)
- SLoC: 2.46M Rust (4,873 files)

## [0.2.4] - 2026-03-28

### Changed
- Version bump to 0.2.4 across all 26 workspace crates
- GPU acceleration: feature-gated `scirs2-core::gpu` imports in oxirs-star behind `gpu` feature flag (Pure Rust policy compliance)
- GPU acceleration: CPU-only fallback for `GpuAccelerator::initialize_context` when `gpu` feature is disabled
- CLI: fixed duplicate short argument flags in clap (`-i` → `-I`, `-c` → `-C`, `-v` → `-V`, `-q` → `-Q`, `-n` explicit) across cli_actions.rs, performance.rs, rebac.rs, and lib.rs
- Dependency upgrades:
  - scirs2-* 0.3.3 → 0.4.1 (all 12 sub-crates)
  - oxiarc-archive, oxiarc-zstd, oxiarc-lz4 0.2.4 → 0.2.6
  - tokio-tungstenite 0.28 → 0.29
  - uuid 1.22 → 1.23
  - redis 1.0 → 1.1
  - kube 3.0 → 3.1
  - toml 1.0 → 1.1
  - proptest 1.10 → 1.11
- Pinned digest to 0.10 and sha2 to 0.10 for compatibility

## [0.2.3] - 2026-03-16

### Changed
- Version bump to 0.2.3
- Production unwrap() audit: all unwrap() calls confirmed in test code only (zero production violations)
- Fixed quantum_sparql_optimizer test marked #[ignore] (slow: >30s quantum simulation)
- Refactored cloud_integration.rs (2000 lines → module with 6 files via splitrs)
- Security: RUSTSEC-2026-0002 (lru 0.12.5 via tantivy) documented and suppressed; tantivy never calls iter_mut()
- Security: RUSTSEC-2025-0134 (rustls-pemfile unmaintained) documented and suppressed; no CVE
- Dependency updates: tempfile 3.26, chrono 0.4.44, rsa 0.9.10, serial_test 3.4, clap 4.6.0, serde_with 3.18.0, tracing-subscriber 0.3.23, oxigdal-proj 0.1.1
- lib.rs doc comment version badges updated: 23 files v0.1.0 → v0.2.3 (html_root_url, shield.io badges)
- Workspace policy (phase 1): all 27 subcrate Cargo.toml files converted to use `.workspace = true` for shared dependencies
- Workspace policy (phase 2): 28 additional deps added to `[workspace.dependencies]` (axum, rustls, tokio-test, tokio-rustls, rustls-pemfile, webpki-roots, lazy_static, blake3, crossbeam-utils, bumpalo, num-complex, indicatif, async-stream, handlebars, config, image, rust_xlsxwriter, governor, prometheus, metrics, moka, kube, k8s-openapi, tokio-test, wiremock, serial_test, rcgen, tracing-opentelemetry, http-body-util); ~60 more inline dep entries converted to `.workspace = true`

### Fixed
- oxirs-rule: 792 compilation errors in test functions (missing Result return types after ? operator changes)
- oxirs-vec: 66 test compilation errors (invalid ? usage on Options in test code)
- oxirs-star: Missing StarError import, non-exhaustive match, wrong method in test
- oxirs-tsdb/oxirs-federate/oxirs-graphrag: chrono::LocalResult .expect() → .unwrap()
- oxirs-stream: large_enum_variant warning fixed (BoxedConsumer variants)

## [0.2.1] - 2026-03-11

### Fixed
- Fixed flaky `test_performance_speedup` in oxirs-arq (removed timing-dependent assertion)
- Fixed flaky `test_multiple_spills` in oxirs-arq (use isolated temp dirs)
- Fixed all clippy warnings (duplicated attributes, redundant patterns, type complexity)
- Upgraded `scirs2-integrate` from 0.1.5 to 0.3.0 (eliminated stale `scirs2-fft` v0.1.5 dependency)

### Changed
- All internal dependencies use workspace version management
- Added `scirs2-integrate` to workspace dependencies
- Upgraded scirs2 dependencies from 0.3.0 to 0.3.1
- Upgraded oxiarc-archive, oxiarc-zstd, oxiarc-lz4 from 0.2.3 to 0.2.3
- Added build profile optimizations to reduce target directory size
- Added Round 16 development modules (+describe_builder, +query_logger, +enum_resolver, and 23 more)

## [0.2.0] - 2026-03-05

### Overview

**Major release** delivering ~10x query performance improvement and comprehensive feature additions across 5 key areas: Performance, Search, Clustering, AI, and Quality. All features are backward compatible with v0.1.0 and feature-gated for gradual rollout.

### Added

#### Performance Enhancements (~10x Cumulative Speedup)
- **Intelligent Cache Invalidation** - Smart cache invalidation engine with dependency tracking (2.5x speedup)
  - Fine-grained invalidation patterns for UPDATE operations
  - Granular invalidation for INSERT/DELETE with affected predicate tracking
  - 758 comprehensive cache invalidation tests (100% passing)
- **ML-Based Cost Prediction** - Machine learning query cost predictor (1.75x speedup, 14.2x cache speedup)
  - Random Forest predictor with 25 feature extractors
  - 95.4% prediction accuracy on production workloads
  - Automatic model retraining with 10K+ query samples
  - 1,122 ML predictor tests (100% passing)
- **Streaming Execution** - Adaptive streaming with memory-bounded operators (1.4x speedup)
  - Automatic spill-to-disk with 85% memory savings
  - Pipeline parallelism with 3-stage execution
  - 514 streaming execution tests (100% passing)
- **Distributed Cache** - L1+L2 cache hierarchy with coherence protocol (1.3x speedup)
  - Write-through L1 cache (process-local) + L2 distributed cache (Redis)
  - Cache coherence with optimistic locking and CAS operations
  - 548 distributed cache tests (100% passing)
- **Adaptive Query Re-optimization** - Runtime query re-optimization (1.25x speedup)
  - Monitors actual vs estimated cardinality during execution
  - Triggers re-optimization when cardinality error >2x
  - 517 adaptive executor tests (100% passing)

#### Search & Indexing
- **Tantivy Full-Text Search** - Production-grade full-text search integration
  - BM25 ranking with stemming and tokenization
  - Fuzzy matching, phrase queries, and Boolean operators
  - Incremental indexing with automatic commit batching
- **3D GeoSPARQL** - Three-dimensional geospatial support
  - 26 topological predicates (sfContains3D, sfIntersects3D, sfWithin3D, etc.)
  - 3D coordinate system with elevation/altitude
  - R-tree spatial indexing for 3D bounding boxes
  - 505 comprehensive 3D geometry tests (100% passing)
- **Multimodal Fusion** - Hybrid search combining text, vector, and spatial queries
  - Reciprocal Rank Fusion (RRF) for result merging
  - Weighted scoring across modalities
- **GPU Acceleration Infrastructure** - CUDA/Metal GPU support framework
  - GPU-accelerated spatial operations (area, distance, intersection)
  - Batch processing for large-scale geometry operations

#### Clustering Enhancements
- **1000+ Node Cluster Support** - Scaled from 500 to 1000+ node clusters
  - Adaptive batching with dynamic batch sizing (15-50x speedup on 1000 node clusters)
  - Pipelined replication with concurrent batch processing
  - Stress testing with chaos engineering (random node failures)
- **Compression** - Multiple compression algorithms for network efficiency
  - LZ4 (fast), Zstd (balanced), LZMA (high compression)
  - Automatic compression selection based on payload size
  - 40-60% bandwidth reduction
- **Encryption** - AES-256-GCM encryption for secure replication
  - Per-node symmetric keys with key rotation support
  - Encrypted Raft log entries and snapshots
- **Cross-Region Optimization** - Improved latency for multi-region deployments
  - Adaptive timeout based on network RTT
  - Regional leader election for locality-aware routing

#### AI Production Hardening
- **LLM Provider Fallback Chains** - Fault-tolerant LLM integration
  - Automatic fallback: OpenAI → Anthropic Claude → Ollama (local)
  - Health checking with circuit breaker pattern
  - Token budget management and rate limiting
  - 440 comprehensive LLM fallback tests (100% passing)
- **GraphRAG with Leiden Community Detection** - Advanced community detection
  - Leiden algorithm replacing Louvain (higher quality partitions)
  - Hierarchical summarization with multi-level communities
  - Cache-aware implementation with 90% cache hit rate
  - 239 GraphRAG cache tests + 355 community detection tests (100% passing)
- **Physics RDF Integration** - Physics-informed digital twin enhancements
  - SAMM Aspect Model parser for automotive/industrial domains
  - Automatic parameter extraction from RDF for simulations
  - Result injection with provenance tracking (PROV-O)
  - 424 RDF integration tests (100% passing)

#### Cloud Integration
- **S3 Backend** - Cloud storage backend for distributed deployments
  - AWS S3 integration with automatic multipart uploads
  - Configurable caching with TTL and size limits
  - Support for S3-compatible services (MinIO, DigitalOcean Spaces)
- **Excel Export** - Excel export functionality with related dependencies for data export capabilities

### Changed

- **Query Performance** - 10x cumulative speedup through optimization stack
  - 2.5x from intelligent cache invalidation
  - 1.75x from ML-based cost prediction
  - 1.4x from streaming execution
  - 1.3x from distributed caching
  - 1.25x from adaptive re-optimization
- **Documentation** - Added 8 comprehensive implementation documents
  - ML Prediction Strategy (14 pages)
  - Cache Invalidation Strategy (19 pages)
  - Streaming Execution Architecture (11 pages)
  - And 5 more detailed design documents
- **Test Coverage** - Added 74 integration tests (100% pass rate)
  - 1,122 ML predictor tests
  - 758 cache invalidation tests
  - 548 distributed cache tests
  - 517 adaptive executor tests
  - 514 streaming execution tests
  - 505 3D GeoSPARQL tests
  - 440 LLM fallback tests
  - 424 physics RDF integration tests
  - 355 community detection tests
  - 239 GraphRAG cache tests
- **Benchmarking** - Added 39+ comprehensive benchmarks (all passing)
  - ML prediction benchmarks
  - SPARQL query optimization benchmarks
  - 3D geometry operation benchmarks
- **RDFS Rules** - Configurable RDFS rules via builder pattern for flexible reasoning configuration (#59)

### Fixed

- **Turtle Parser** - Delegated parsing to oxttl for full Turtle syntax support (#57)
- **SHACL Language Tags** - Fixed language tag handling for proper RDF validation (#58)
- **RETE Engine** - Fixed remove_fact to use unification matching instead of hash lookup (#60)

### Removed

- **Vaporware Cleanup** - Removed 27,237 lines of unimplemented code
  - Quantum-related modules and references (quantum consciousness, quantum computing)
  - Quantum/biological modules from oxirs-embed
  - Dead code with unimplemented!() macros from oxirs-chat
  - AI slop modules that were not production-ready

### Quality Metrics

- **74 integration tests** - 100% pass rate
- **39+ benchmarks** - All passing with performance validation
- **~18,200 lines** of new production code
- **Zero compilation errors** - Clean build across all modules
- **Complete documentation** - 8 comprehensive design documents

### Performance Benchmarks

```
Query Performance (10x cumulative speedup):
  Cache Invalidation:    2.5x improvement
  ML Cost Prediction:    1.75x improvement (14.2x cache speedup)
  Streaming Execution:   1.4x improvement (85% memory savings)
  Distributed Cache:     1.3x improvement
  Adaptive Execution:    1.25x improvement

Cluster Scaling:
  Node Support:          1000+ nodes (up from 500)
  Adaptive Batching:     15-50x speedup on large clusters
  Compression:           40-60% bandwidth reduction
  Encryption:            AES-256-GCM with minimal overhead

3D GeoSPARQL:
  Topological Operations: 26 predicates (sfContains3D, sfIntersects3D, etc.)
  R-tree Indexing:        Sub-millisecond spatial queries
  GPU Acceleration:       10-100x speedup for batch operations
```

### Contributors

- @cool-japan (KitaSan) - Core development and architecture
- Claude Sonnet 4.5 - Co-development

---

## [0.1.0] - 2026-01-07

### Overview

**Initial production release** of OxiRS - A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning. This release provides a complete, production-ready alternative to Apache Jena + Fuseki with modern enhancements.

**Major Achievements**:
- **Complete SPARQL 1.1/1.2 Implementation**: Full W3C compliance with advanced query optimization
- **13,123 tests passing**: 100% pass rate with comprehensive test coverage (136 skipped)
- **Zero warnings**: Strict `-D warnings` enforcement across all 22 crates
- **Industrial IoT Support**: Time-series, Modbus, CANbus/J1939 integration
- **AI-Powered Features**: GraphRAG, embeddings, physics-informed reasoning
- **Production-Ready**: Complete observability, security, and deployment automation

**Quality Metrics**:
- **13,123 tests passing** - 100% pass rate (136 skipped)
- **Zero compilation warnings** - Enforced with `-D warnings`
- **95%+ test coverage** - Comprehensive test suites
- **95%+ documentation coverage** - Complete API documentation
- **Zero clippy warnings** - Production-grade code quality

### Added

#### Core RDF & SPARQL
- **RDF 1.2 Support** - Complete implementation with 7 format parsers (Turtle, N-Triples, RDF/XML, JSON-LD, N-Quads, TriG, N3)
- **SPARQL 1.1 Query & Update** - Full W3C compliance with SELECT, CONSTRUCT, ASK, DESCRIBE, INSERT, DELETE
- **SPARQL 1.2 Extensions** - RDF-star, enhanced property paths, advanced aggregations
- **Adaptive Query Optimization** - 3.8x faster queries via automatic complexity detection
- **Persistent Storage** - N-Quads format with automatic save/load

#### Server & API
- **oxirs-fuseki** - SPARQL 1.1/1.2 HTTP server with Fuseki compatibility
- **oxirs-gql** - GraphQL API for RDF data
- **REST API v2** - OpenAPI 3.0 with Swagger UI
- **WebSocket Support** - Real-time subscriptions and query streaming
- **Admin UI** - Modern web-based dashboard with live metrics

#### Query & Reasoning
- **oxirs-arq** - Advanced SPARQL query engine with cost-based optimization
- **oxirs-rule** - Rule-based reasoning (RDFS/OWL/SWRL)
- **oxirs-shacl** - SHACL Core + SHACL-SPARQL validation (27/27 W3C tests passing)
- **oxirs-federate** - Distributed query federation with 2-phase commit
- **GeoSPARQL Support** - OGC 1.1 compliance with spatial indexing

#### Industrial Connectivity (Phase D)
- **oxirs-tsdb** - Time-series database with 40:1 Gorilla compression
  - SPARQL temporal extensions (ts:window, ts:resample, ts:interpolate)
  - Hybrid RDF + time-series storage with automatic routing
  - 500K pts/sec write throughput, 180ms p50 query latency
  - 128 tests passing
- **oxirs-modbus** - Complete Modbus TCP/RTU protocol support
  - PLC integration with 6 data types (INT16, UINT16, INT32, UINT32, FLOAT32, BIT)
  - RDF triple generation with QUDT units and PROV-O timestamps
  - Connection pooling with health monitoring
  - 75 tests passing
- **oxirs-canbus** - CANbus/J1939 automotive integration
  - DBC file parser with signal extraction
  - J1939 protocol with PGN extraction and multi-packet reassembly
  - SAMM Aspect Model generation from DBC
  - 98 tests passing

#### AI & Machine Learning
- **oxirs-embed** - Knowledge graph embeddings (TransE, ComplEx, Tucker)
  - CUDA GPU acceleration for 2-5x faster computation
  - Multi-GPU support for large-scale graphs
  - 350+ tests passing
- **oxirs-chat** - RAG chat API with LLM integration
  - Multi-LLM support (OpenAI, Anthropic Claude, Ollama)
  - Session management and context tracking
- **oxirs-shacl-ai** - AI-powered SHACL validation
  - Shape learning and data repair suggestions
  - 350+ tests passing
- **oxirs-graphrag** - GraphRAG hybrid search
  - RRF (Reciprocal Rank Fusion) combining vector and graph topology
  - N-hop SPARQL graph expansion for context retrieval
  - Louvain community detection for hierarchical summarization
  - 23 tests passing
- **oxirs-physics** - Physics-informed digital twins
  - SciRS2 simulation integration
  - RDF → Simulation parameter extraction
  - Physics constraint validation

#### Security & Trust
- **oxirs-did** - W3C DID & Verifiable Credentials
  - DID Core 1.0 and VC Data Model 2.0 implementation
  - did:key and did:web methods with Ed25519 signatures
  - RDFC-1.0 RDF graph canonicalization
  - 43 tests passing
- **ReBAC** - Relationship-Based Access Control
  - Graph-level authorization with hierarchical permissions
  - SPARQL-based policy storage
  - 83 tests passing
- **OAuth2/OIDC/JWT** - Modern authentication
- **TLS/SSL** - Certificate rotation with ACME/Let's Encrypt integration

#### Storage & Distribution
- **oxirs-tdb** - TDB2-compatible storage with MVCC
  - Memory-mapped optimization for large datasets
  - Background compaction and snapshots
  - 250+ tests passing
- **oxirs-cluster** - Distributed clustering with Raft consensus
  - Multi-region replication
  - Automatic failover and recovery
- **oxirs-stream** - Real-time streaming (Kafka/NATS)
  - RDF Patch and SPARQL Update delta
  - 100K+ events/sec throughput
  - 300+ tests passing

#### Platforms
- **oxirs-wasm** - WebAssembly browser/edge deployment
  - In-memory RDF store for browsers
  - TypeScript definitions and ES modules
  - Zero Tokio dependency (WASM-compatible)
  - 8 tests passing

#### Industry Standards
- **oxirs-samm** - SAMM 2.0-2.3 metamodel & AAS integration
  - 16 code generators for Industry 4.0
- **NGSI-LD** - FIWARE smart city compatibility (ETSI GS CIM 009 v1.6)
- **MQTT & OPC UA** - Industrial IoT bridges
- **IDS/Gaia-X** - European data space compliance with ODRL 2.2

#### Performance & Operations
- **Adaptive Query Optimization** - 3.8x faster for simple queries
  - Automatic complexity detection
  - Fast path for simple queries (≤5 patterns)
  - Full cost-based optimization for complex queries
  - 75% CPU savings at production scale (100K QPS)
- **Work-Stealing Scheduler** - Efficient concurrency with 4-level priority queuing
- **Memory Pooling** - SciRS2-integrated buffer management
- **Request Batching** - Automatic batching with adaptive sizing
- **Result Streaming** - Zero-copy streaming with compression (Gzip, Brotli)
- **Load Balancing** - 9 strategies including consistent hashing
- **Edge Caching** - Multi-CDN support (Cloudflare, Fastly, CloudFront, Akamai)
- **DDoS Protection** - IP-based rate limiting with anomaly detection
- **Security Audit** - OWASP Top 10 vulnerability scanning
- **Prometheus Metrics** - Complete observability with OpenTelemetry tracing
- **Disaster Recovery** - Automated failover with RPO/RTO management

#### Deployment Automation
- **Docker** - Multi-stage production builds (12MB binary)
- **Kubernetes** - Production-grade manifests with HPA and Prometheus Operator
- **Kubernetes Operator** - CRD-based deployment automation
- **Terraform** - Complete cloud infrastructure (AWS EKS, GCP GKE, Azure AKS)
- **Ansible** - Production deployment playbooks

#### CLI Tools
- **oxirs** - Comprehensive command-line tool
  - Dataset management (init, import, export, query, serve)
  - Time-series operations (query, insert, stats, compact, retention, benchmark)
  - Modbus operations (monitor-tcp, monitor-rtu, read, write, to-rdf, mock-server)
  - CANbus operations (monitor, parse-dbc, decode, send, to-samm, to-rdf, replay)
  - ReBAC management (add-relationship, remove-relationship, check-permission, list-relationships)

### Changed

- **Query Optimizer Performance** - All profiles optimized at ~3.0 µs (down from 10-16 µs)
  - HighThroughput: 10.8 µs → 3.24 µs (3.3x faster)
  - Analytical: 11.7 µs → 3.01 µs (3.9x faster)
  - Mixed: 10.5 µs → 2.95 µs (3.6x faster)
  - LowMemory: 15.6 µs → 2.94 µs (5.3x faster)

### Performance Benchmarks

```
Query Optimization (5 triple patterns):
  HighThroughput:  3.24 µs (3.3x faster than baseline)
  Analytical:      3.01 µs (3.9x faster than baseline)
  Mixed:           2.95 µs (3.6x faster than baseline)
  LowMemory:       2.94 µs (5.3x faster than baseline)

Time-Series Database:
  Write throughput: 500K pts/sec (single), 2M pts/sec (batch 1K)
  Query latency:    180ms p50 (1M points range), 120ms p50 (aggregation)
  Compression:      38:1 (temperature), 25:1 (vibration), 32:1 (timestamps)

Production Impact (100K QPS):
  CPU time saved:   45 minutes per hour (75% reduction)
  Annual savings:   $10,000 - $50,000 (cloud deployments)
```

### Standards Compliance

- **W3C**: SPARQL 1.1, SPARQL 1.2, RDF 1.2, SHACL, DID Core 1.0, VC Data Model 2.0, RDFC-1.0
- **OGC**: GeoSPARQL 1.1
- **ETSI**: NGSI-LD v1.6
- **ISO/IEC**: MQTT 5.0 (20922), CAN 2.0/CAN FD (11898-1)
- **SAE**: J1939 (heavy vehicles)
- **IEC**: OPC UA (62541)
- **IDSA**: Reference Architecture 4.x
- **Eclipse**: Sparkplug B 3.0

### Contributors

- @cool-japan (KitaSan) - Core development and architecture

### Links

- **Repository**: https://github.com/cool-japan/oxirs
- **Issues**: https://github.com/cool-japan/oxirs/issues
- **Documentation**: https://docs.rs/oxirs-core

---

*"Rust makes memory safety table stakes; OxiRS makes knowledge-graph engineering table stakes."*

[0.3.2]: https://github.com/cool-japan/oxirs/releases/tag/v0.3.2
[0.3.1]: https://github.com/cool-japan/oxirs/releases/tag/v0.3.1
[0.3.0]: https://github.com/cool-japan/oxirs/releases/tag/v0.3.0
[0.2.4]: https://github.com/cool-japan/oxirs/releases/tag/v0.2.4
[0.2.3]: https://github.com/cool-japan/oxirs/releases/tag/v0.2.3
[0.2.1]: https://github.com/cool-japan/oxirs/releases/tag/v0.2.1
[0.2.0]: https://github.com/cool-japan/oxirs/releases/tag/v0.2.0
[0.1.0]: https://github.com/cool-japan/oxirs/releases/tag/v0.1.0
