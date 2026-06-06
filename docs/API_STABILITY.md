# OxiRS API Stability Guarantees

**Version**: 0.3.1
**Last Updated**: June 6, 2026 (initial release: January 7, 2026)
**Status**: Production-Ready
**Stability Level**: Production Release

---

## 🎯 Overview

This document defines OxiRS's API stability guarantees, versioning policy, and deprecation procedures. Starting with v0.1.0, we committed to **backward compatibility** within each published minor series (currently the v0.3.x patch line) and to providing migration guidance across minor releases — a contract that has held in practice across the v0.1.0 → v0.2.x → v0.3.1 progression. As of v0.3.1, stability guarantees cover an expanded surface, with distributed storage and AI modules having matured through the 0.2.x and 0.3.x series.

---

## 📜 Stability Levels

OxiRS APIs are classified into three stability levels:

### 🟢 Stable (Production-Ready)

**Guarantee**: These APIs will NOT change in backward-incompatible ways within the same major version (v0.x → v0.y)

**Policy**:
- ✅ Guaranteed backward compatibility within the current minor series (v0.3.x)
- ✅ Safe for production use
- ✅ Deprecations announced 3 months before removal
- ✅ Migration path provided for all changes

**Examples**:
```rust
// Stable APIs in oxirs-core
pub trait Store {
    fn insert(&mut self, quad: Quad) -> Result<()>;
    fn contains(&self, quad: &Quad) -> Result<bool>;
    fn quads(&self) -> Box<dyn Iterator<Item = Result<Quad>>>;
}

// Stable APIs in oxirs-arq
pub struct QueryEngine { /* ... */ }
impl QueryEngine {
    pub fn new() -> Self;
    pub fn execute(&self, query: &str, dataset: &dyn Store) -> Result<QueryResults>;
}
```

### 🟡 Unstable (Pre-Production)

**Guarantee**: These APIs may change between minor versions (e.g., v0.2.x → v0.3.x) but will provide migration guides

**Policy**:
- ⚠️ May change in a future minor release
- ⚠️ Suitable for testing and evaluation
- ⚠️ Changes documented in CHANGELOG
- ⚠️ Migration examples provided

**Examples**:
```rust
// Unstable APIs in oxirs-shacl-ai
pub struct ShapeLearner { /* ... */ }
impl ShapeLearner {
    pub fn learn_shapes(&self, graph: &Graph) -> Result<ShapeSchema>;
    // May change to take additional parameters in a future minor release
}

// Unstable APIs in oxirs-embed
pub struct EmbeddingModel { /* ... */ }
impl EmbeddingModel {
    pub fn encode(&self, text: &str) -> Result<Vec<f32>>;
    // Vector size may become configurable in a future minor release
}
```

### 🔴 Experimental (Research Preview)

**Guarantee**: These APIs may change at ANY time, including patch versions

**Policy**:
- 🚨 No stability guarantees
- 🚨 For research and experimentation only
- 🚨 NOT recommended for production
- 🚨 May be removed without notice

**Examples**:
```rust
// Experimental APIs (hidden behind feature flags)
#[cfg(feature = "experimental-quantum")]
pub mod quantum {
    pub struct QuantumOptimizer { /* ... */ }
}

#[cfg(feature = "experimental-neuro")]
pub mod neuro {
    pub struct NeuroSymbolicReasoner { /* ... */ }
}
```

---

## 🔢 Semantic Versioning Policy

OxiRS follows [Semantic Versioning 2.0.0](https://semver.org/) with Rust-specific interpretations:

### Version Format: `MAJOR.MINOR.PATCH`

#### Major Version (0.x.y → 1.x.y)
**Breaking changes** to stable APIs:
- Removal of public APIs
- Changes to function signatures
- Changes to trait requirements
- Incompatible data format changes

**Timeline**: v0.1.0 (Jan 2026) → v0.2.x (Mar 2026) → v0.3.0 (May 2026) → v0.3.1 (Jun 2026) have shipped. A future v1.0.0 (date TBD) is planned as the long-term-stable milestone.

#### Minor Version (0.1.x → 0.2.x)
**Non-breaking additions and unstable API changes**:
- New stable APIs
- New features behind feature flags
- Deprecations (with 3-month notice)
- Changes to unstable APIs
- Performance improvements

**Timeline**: The v0.2.x minor series shipped in March 2026 (v0.2.0 on 2026-03-05 through v0.2.4 on 2026-03-28); v0.3.0 shipped 2026-05-03 and v0.3.1 on 2026-06-06.

#### Patch Version (0.3.1 → 0.3.2)
**Bug fixes only**:
- Security patches
- Bug fixes
- Documentation updates
- Performance optimizations (non-breaking)

**Timeline**: Patch releases as needed (typically monthly)

---

## 🔄 Deprecation Policy

### Deprecation Process

1. **Announcement** (Version N)
   - Mark API with `#[deprecated]` attribute
   - Add deprecation message with migration path
   - Document in CHANGELOG
   - Update migration guide

2. **Grace Period** (3 months minimum)
   - API remains functional with warnings
   - Migration examples provided
   - Support for migration questions

3. **Removal** (Version N+2 minimum)
   - API removed in next major/minor version
   - Final migration notice in CHANGELOG

### Example Deprecation

```rust
// Version 0.1.0 (Deprecation announcement)
#[deprecated(
    since = "0.1.0",
    note = "Use `MemoryStore::new()` or `TdbStore::open()` instead. \
            See migration guide: docs/MIGRATION_ALPHA3_BETA1.md"
)]
pub fn ConcreteStore::new() -> Result<Self> {
    // Still works, but emits warning
}

// Version 0.2.3 (Removal - 3 months later)
// ConcreteStore::new() removed entirely
```

### Current Deprecations (v0.3.1)

*No deprecations are currently active as of v0.3.1.*

---

## 📦 Module Stability Matrix

### Core Foundation (🟢 Stable)

#### oxirs-core (v0.3.1)
**Stability**: 🟢 **Stable** (95% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `rdf_model::Term` | 🟢 Stable | Core RDF term types (IRI, Literal, BlankNode) |
| `rdf_model::Triple` | 🟢 Stable | RDF triple structure |
| `rdf_model::Quad` | 🟢 Stable | RDF quad structure (with graph) |
| `store::Store` trait | 🟢 Stable | Core storage interface |
| `store::MemoryStore` | 🟢 Stable | In-memory RDF store |
| `error::OxirsError` | 🟢 Stable | Error type hierarchy |
| `dataset::Dataset` | 🟢 Stable | Dataset abstraction |
| `parser::Parser` trait | 🟡 Unstable | May add streaming support |
| `serializer::Serializer` trait | 🟡 Unstable | May add async support |

**Breaking Change Risk**: **LOW** (< 5%)

#### oxirs-tdb (v0.3.1)
**Stability**: 🟢 **Stable** (90% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `TdbStore::open()` | 🟢 Stable | Open persistent TDB store |
| `TdbStore::create()` | 🟢 Stable | Create new TDB store |
| `TdbStore::close()` | 🟢 Stable | Close TDB store |
| `TdbOptions` | 🟡 Unstable | May add compression options |
| `TdbStats` | 🟡 Unstable | Statistics interface may expand |

**Breaking Change Risk**: **LOW** (< 10%)

---

### Query Engine (🟢 Stable)

#### oxirs-arq (v0.3.1)
**Stability**: 🟢 **Stable** (90% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `QueryEngine::new()` | 🟢 Stable | Create query engine |
| `QueryEngine::execute()` | 🟢 Stable | Execute SPARQL query |
| `QueryEngine::execute_with_options()` | 🟢 Stable | Execute with options (timeout, limits) |
| `QueryResults` enum | 🟢 Stable | Query result types (SELECT, ASK, CONSTRUCT) |
| `QueryOptions` | 🟡 Unstable | May add caching options |
| `Algebra` types | 🟡 Unstable | Internal algebra may evolve |

**Breaking Change Risk**: **LOW** (< 10%)

#### oxirs-rule (v0.3.1)
**Stability**: 🟡 **Unstable** (70% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `RuleEngine::new()` | 🟡 Unstable | May add rule compilation options |
| `RuleEngine::execute()` | 🟡 Unstable | Execution semantics may change |
| `Rule` structure | 🟡 Unstable | Rule DSL may expand |
| `RuleSet` | 🟡 Unstable | Set operations may change |

**Breaking Change Risk**: **MEDIUM** (30%)

---

### Server & HTTP (🟢 Stable)

#### oxirs-fuseki (v0.3.1)
**Stability**: 🟢 **Stable** (95% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `FusekiServer::new()` | 🟢 Stable | Create server instance |
| `FusekiServer::start()` | 🟢 Stable | Start HTTP server |
| `FusekiConfig` | 🟢 Stable | Server configuration |
| HTTP endpoints (`/$/ping`, `/query`, `/update`) | 🟢 Stable | SPARQL 1.1 Protocol compliant |
| Admin UI | 🟡 Unstable | UI may be redesigned |

**Breaking Change Risk**: **VERY LOW** (< 5%)

#### oxirs-gql (v0.3.1)
**Stability**: 🟡 **Unstable** (80% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `GraphQLServer::new()` | 🟡 Unstable | May add schema customization |
| GraphQL schema | 🟡 Unstable | Schema may expand with new types |
| Query resolvers | 🟡 Unstable | Resolver optimization may change signatures |

**Breaking Change Risk**: **MEDIUM** (20%)

---

### Storage & Distribution (🟡 Unstable)

#### oxirs-cluster (v0.3.1)
**Stability**: 🟡 **Unstable** (70% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `ClusterNode::new()` | 🟡 Unstable | May add gossip protocol options |
| `ClusterConfig` | 🟡 Unstable | Configuration may expand significantly |
| `ReplicationFactor` | 🟡 Unstable | May add dynamic replication |
| Raft consensus | 🟡 Unstable | Internal implementation may change |

**Breaking Change Risk**: **MEDIUM** (40%)

---

### Validation & Reasoning (🟡 Unstable)

#### oxirs-shacl (v0.3.1)
**Stability**: 🟡 **Unstable** (75% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `ShaclValidator::validate()` | 🟡 Unstable | Validation API may add async support |
| `ValidationReport` | 🟡 Unstable | Report format may expand |
| `Shape` types | 🟡 Unstable | SHACL-SPARQL support may change shape types |

**Breaking Change Risk**: **MEDIUM** (25%)

#### oxirs-shacl-ai (v0.3.1)
**Stability**: 🔴 **Experimental** (50% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `ShapeLearner` | 🔴 Experimental | AI-based shape learning is research preview |
| `ShapeInferenceModel` | 🔴 Experimental | Model architecture may change significantly |

**Breaking Change Risk**: **HIGH** (50%)

---

### AI & Machine Learning (🔴 Experimental)

#### oxirs-embed (v0.3.1)
**Stability**: 🔴 **Experimental** (60% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `EmbeddingModel::encode()` | 🟡 Unstable | Encoding API stabilizing across the 0.x series |
| `VectorStore` | 🔴 Experimental | Storage format may change |
| Similarity search | 🔴 Experimental | Algorithm may be replaced |

**Breaking Change Risk**: **HIGH** (40%)

#### oxirs-chat (v0.3.1)
**Stability**: 🔴 **Experimental** (50% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `ChatEngine` | 🔴 Experimental | RAG pipeline is research preview |
| `ContextRetriever` | 🔴 Experimental | Retrieval strategy may change |
| Response generation | 🔴 Experimental | LLM integration may change |

**Breaking Change Risk**: **HIGH** (50%)

---

### Streaming & Federation (🟡 Unstable)

#### oxirs-stream (v0.3.1)
**Stability**: 🟡 **Unstable** (65% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `StreamProcessor` | 🟡 Unstable | Kafka/NATS integration may change |
| `EventStream` | 🟡 Unstable | Event format may evolve |
| Watermark handling | 🟡 Unstable | Watermark strategy may change |

**Breaking Change Risk**: **MEDIUM** (35%)

#### oxirs-federate (v0.3.1)
**Stability**: 🟡 **Unstable** (80% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `FederationClient::builder()` | 🟢 Stable | Builder API frozen |
| `FederationClient::query()` | 🟡 Unstable | Query execution may add caching |
| Endpoint discovery | 🔴 Experimental | Discovery mechanism may change |

**Breaking Change Risk**: **MEDIUM** (20%)

---

### Extensions (🔴 Experimental)

#### oxirs-star (v0.3.1)
**Stability**: 🟡 **Unstable** (85% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `StarTerm` | 🟡 Unstable | RDF-star types stabilizing |
| Reification strategies | 🟡 Unstable | Strategies may expand |
| Annotation syntax | 🟡 Unstable | W3C spec still evolving |

**Breaking Change Risk**: **MEDIUM** (15%)

#### oxirs-geosparql (v0.3.1)
**Stability**: 🔴 **Experimental** (60% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| Geo functions | 🟡 Unstable | OGC GeoSPARQL compliance targeted |
| Spatial indexing | 🔴 Experimental | Index structure may change |
| CRS handling | 🔴 Experimental | Coordinate system support expanding |

**Breaking Change Risk**: **MEDIUM** (40%)

---

### Digital Twin & Simulation (🟡 Unstable)

> The following crates were added during the v0.2.x / v0.3.x series and have a smaller stable surface than the core modules above. Their public APIs are **🟡 Unstable** (newer; may change between minor releases).

#### oxirs-samm (v0.3.1)
**Stability**: 🟡 **Unstable** (newer module; APIs evolving)

Semantic Aspect Meta Model (SAMM) implementation for describing digital-twin aspect models — Turtle parsing, SHACL validation, and multi-language code generation.

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `parser::parse_aspect_model()` | 🟡 Unstable | Aspect-model loading API may change |
| `metamodel::Aspect` / `ModelElement` | 🟡 Unstable | Meta-model types still evolving |
| `generators::*` | 🟡 Unstable | Code-generation surface (TS/GraphQL/SQL/…) expanding |

**Breaking Change Risk**: **MEDIUM** (40%)

#### oxirs-physics (v0.3.1)
**Stability**: 🟡 **Unstable** (newer module; APIs evolving)

Physics-informed digital-twin simulation bridge connecting RDF knowledge graphs with SciRS2 simulations — parameter extraction, result injection, and conservation-law validation.

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `simulation::SimulationOrchestrator` | 🟡 Unstable | Orchestration API may change |
| `digital_twin::DigitalTwin` | 🟡 Unstable | Twin-synchronization surface evolving |
| `rdf::*` (SPARQL builder, literal parser) | 🟡 Unstable | RDF / unit-conversion bridge may change |

**Breaking Change Risk**: **MEDIUM** (40%)

---

### Knowledge Graph RAG (🟡 Unstable)

#### oxirs-graphrag (v0.3.1)
**Stability**: 🟡 **Unstable** (newer module; APIs evolving)

Hybrid vector + graph Retrieval-Augmented Generation — triple extraction, community detection, path finding, and subgraph summarization for LLM context.

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `triple_extractor::TripleExtractor` | 🟡 Unstable | NLP → RDF extraction evolving |
| `community_detector` / `path_finder` | 🟡 Unstable | Graph-retrieval primitives may change |
| `summarizer::SubgraphSummarizer` | 🟡 Unstable | Context-building API evolving |

**Breaking Change Risk**: **MEDIUM** (40%)

---

### Identity & Trust (🟡 Unstable)

#### oxirs-did (v0.3.1)
**Stability**: 🟡 **Unstable** (newer module; APIs evolving)

W3C Decentralized Identifiers (DID) and Verifiable Credentials with signed RDF graphs — did:key / did:web, VC Data Model 2.0, and Ed25519 dataset canonicalization.

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `Did` / `DidResolver` | 🟡 Unstable | DID creation / resolution API may change |
| `VerifiableCredential` / `CredentialIssuer` | 🟡 Unstable | VC issuance / verification evolving |
| `signed_graph` module | 🟡 Unstable | RDF canonicalization + signing evolving |

**Breaking Change Risk**: **MEDIUM** (40%)

---

### Browser / WebAssembly (🟡 Unstable)

#### oxirs-wasm (v0.3.1)
**Stability**: 🟡 **Unstable** (newer module; APIs evolving)

WebAssembly bindings to run RDF/SPARQL in the browser — streaming parser, compact triple store, and a `wasm_bindgen` JS/TS surface.

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `OxiRSStore` (JS/TS binding) | 🟡 Unstable | `wasm_bindgen` surface may change |
| `query` / `update` modules | 🟡 Unstable | SPARQL query / update API evolving |
| `parser` (streaming) | 🟡 Unstable | Incremental parsing API may change |

**Breaking Change Risk**: **MEDIUM** (40%)

---

### Industrial Connectivity (🟡 Unstable)

#### oxirs-modbus (v0.3.1)
**Stability**: 🟡 **Unstable** (newer module; APIs evolving)

Modbus TCP/RTU/ASCII/TLS protocol support for industrial-IoT ingestion into RDF — register mapping with QUDT units and W3C PROV-O provenance.

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `ModbusTcpClient` / `ModbusConfig` | 🟡 Unstable | Client / config API may change |
| `mapping::RegisterMap` | 🟡 Unstable | Register-mapping surface evolving |

**Breaking Change Risk**: **MEDIUM** (40%)

#### oxirs-canbus (v0.3.1)
**Stability**: 🟡 **Unstable** (newer module; APIs evolving)

CANbus/J1939 protocol support for automotive and heavy-machinery data — DBC parsing, PGN decoding, multi-packet reassembly, and RDF mapping.

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `J1939Processor` / `PgnRegistry` | 🟡 Unstable | J1939 processing API may change |
| `CanFrame` / `CanId` | 🟡 Unstable | Frame / ID types still evolving |

**Breaking Change Risk**: **MEDIUM** (40%)

---

### Time-Series Storage (🟡 Unstable)

#### oxirs-tsdb (v0.3.1)
**Stability**: 🟡 **Unstable** (newer module; APIs evolving)

Time-series optimizations for IoT-scale RDF — Gorilla compression, delta-of-delta timestamps, SPARQL temporal extensions, and hybrid RDF + time-series storage.

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `HybridStore` | 🟡 Unstable | Hybrid RDF + time-series API may change |
| `query` / `sparql` modules | 🟡 Unstable | Temporal query extensions evolving |
| `write` (WAL / compaction) | 🟡 Unstable | Write-path API may change |

**Breaking Change Risk**: **MEDIUM** (40%)

---

## 🔒 API Stability Contract (v0.3.1 → v1.0.0)

### What We Guarantee

#### ✅ Backward Compatibility Within v0.3.x
- **All patch releases (v0.3.1, v0.3.2, ...)** maintain 100% backward compatibility
- **No API removals** in patch releases
- **No signature changes** in patch releases
- **Only bug fixes and docs** in patch releases

#### ✅ Migration Path for the Next Minor (v0.4.0)
- **Deprecations announced 3 months in advance**
- **Migration guide** provided for all breaking changes
- **Examples** for all deprecated APIs
- **Compatibility shims** where possible

#### ✅ Data Format Stability
- **TDB format** stable across v0.3.x
- **RDF serialization** compatible across v0.3.x
- **Configuration format** backward compatible (new fields additive only)

#### ✅ Protocol Stability
- **SPARQL 1.1 Protocol** fully compliant (no deviations)
- **HTTP endpoints** stable (no URL changes)
- **GraphQL schema** additive only in v0.3.x

### What We Don't Guarantee

#### ❌ Internal Implementation Details
- Private APIs may change without notice
- Internal data structures not covered
- Performance characteristics (may improve)
- Memory layout (may optimize)

#### ❌ Experimental Features
- APIs marked 🔴 Experimental may change anytime
- Features behind `experimental-*` flags unstable
- Research previews (oxirs-chat, oxirs-shacl-ai) evolving

#### ❌ Compile-Time Guarantees
- MSRV (Minimum Supported Rust Version) may increase
- Dependency versions may update (following semver)
- Feature flag names may change (with migration guide)

---

## 📊 Stability Roadmap

### v0.1.0 (January 2026) — Shipped
**Focus**: API freeze for core modules

- 🟢 **Stable**: oxirs-core, oxirs-arq, oxirs-fuseki, oxirs-tdb (95% frozen)
- 🟡 **Unstable**: oxirs-cluster, oxirs-shacl, oxirs-gql (70-80% frozen)
- 🔴 **Experimental**: oxirs-embed, oxirs-chat, oxirs-shacl-ai (50-60% frozen)

**Breaking Change Risk**: 10% overall

### v0.2.x (March 2026) — Shipped
**Focus**: Performance, search, clustering scale, and AI hardening

The v0.2 series (v0.2.0 on 2026-03-05 through v0.2.4 on 2026-03-28) delivered a ~10x cumulative query speedup, Tantivy full-text search, 3D GeoSPARQL, 1000+ node clustering, and AI production hardening. All additions were backward compatible with v0.1.0 and feature-gated for gradual rollout.

- 🟢 **Stable**: core modules (oxirs-core, oxirs-arq, oxirs-fuseki, oxirs-tdb) unchanged
- 🟡 **Unstable**: distributed (oxirs-cluster), GraphQL (oxirs-gql), and streaming (oxirs-stream) continued to mature
- 🔴 **Experimental**: AI modules (oxirs-embed, oxirs-chat, oxirs-shacl-ai) continued as research previews

**Breaking Change Risk**: low (additive, backward-compatible)

### v0.3.0 (May 2026) — Shipped
**Focus**: Audit/compliance, certification, SSO, and marketplace

Shipped 2026-05-03: a SOC2/GDPR audit module (oxirs-core), SHACL-AI certification, a chat marketplace (HuggingFace Hub / Ollama / local GGUF), SSO (OIDC + SAML 2.0), cluster certification, and GraphRAG model loading.

**Breaking Change Risk**: low (additive)

### v0.3.1 (June 2026) — Shipped (Current)
**Focus**: Pure-Rust crypto/compression/TLS migration, FIPS feature gates, SHACL-AF

Shipped 2026-06-06: migration of compression (brotli/snap/flate2 → oxiarc-*), crypto (ring → oxicrypto), and TLS (process-wide pure-Rust oxitls provider); FIPS 140-2 feature gates for oxirs-fuseki and oxirs-did; additional SHACL Advanced Features; an HDT 1.0 reader; a streaming TriG parser; and a wave of file-size refactors.

**Breaking Change Risk**: low (additive; the default build is now fully Pure Rust)

### v1.0.0 (Planned — date TBD)
**Focus**: Stable release with long-term API guarantees

- 🟢 **Stable**: 100% API freeze for core modules
- 🟡 **Unstable**: Experimental features moved to separate workspace
- 🔴 **Experimental**: None (moved to oxirs-experimental workspace)

**Breaking Change Risk**: 0% for stable APIs (aspirational target)

---

## 🛡️ MSRV (Minimum Supported Rust Version) Policy

### Current MSRV: **Rust 1.70** (declared MSRV; CI builds on newer stable)

**Policy**:
- MSRV increases are **NOT** considered breaking changes
- MSRV updates follow Rust's 6-week release cycle
- We support **N-2** stable Rust releases (approximately 12 weeks)
- MSRV bumps documented in CHANGELOG

**Rationale**:
- Rust's rapid evolution provides significant performance and safety improvements
- Supporting older Rust versions limits our ability to use new language features
- Most production Rust users stay within 1-2 releases of stable

**Examples**:
```toml
# Cargo.toml
[package]
name = "oxirs-core"
version = "0.2.3"
rust-version = "1.70"  # MSRV declared (conservative; builds on newer stable)
```

**Upgrade Guidance**:
```bash
# Check current Rust version
rustc --version

# Update Rust to latest stable
rustup update stable

# Verify OxiRS compiles
cargo build --release
```

---

## 🔧 Cargo Feature Stability

### Stable Features (Always Available)
```toml
[features]
# Default build is fully Pure Rust (no ring / aws-lc-sys / native-tls).
default = ["metrics", "auth"]
metrics = ["prometheus", "axum-prometheus"]  # Prometheus metrics + exporter
auth = ["jsonwebtoken", "bcrypt"]            # JWT + bcrypt authentication
# TLS is pure-Rust rustls. The `rustls`/`oxitls` crates are always compiled in
# (non-optional) so a process-wide pure-Rust crypto provider is installed at
# startup; the optional `tls` feature adds the HTTPS listener + PEM key loading.
tls = ["tokio-rustls", "rustls-pemfile", "axum-server"]
```

### Unstable Features (May Change)
```toml
[features]
# Distributed features
cluster = ["dep:oxirs-cluster"]        # Raft consensus clustering
federation = ["dep:oxirs-federate"]    # Federated SPARQL queries

# Advanced query features
text-search = ["dep:tantivy"]          # Full-text search
geosparql = ["dep:oxirs-geosparql"]    # GeoSPARQL support
rdf-star = ["dep:oxirs-star"]          # RDF-star annotations
```

### Experimental Features (Research Preview)
```toml
[features]
# AI features (unstable, may change significantly)
experimental-ai = ["dep:oxirs-embed", "dep:oxirs-chat"]
experimental-shacl-ai = ["dep:oxirs-shacl-ai"]

# Quantum optimization (research only)
experimental-quantum = ["dep:oxirs-quantum"]

# Neuro-symbolic reasoning (research only)
experimental-neuro = ["dep:oxirs-neuro"]
```

**Policy**:
- **Stable features**: Never removed within a minor series (e.g., v0.3.x)
- **Unstable features**: May change in a future minor release with migration guide
- **Experimental features**: May be removed at any time

---

## 📝 Breaking Change Examples

### ❌ Breaking Changes (Require Major/Minor Version Bump)

```rust
// BEFORE (v0.1.0)
pub fn execute(&self, query: &str) -> Result<QueryResults>;

// AFTER (v0.2.3) - BREAKING: New required parameter
pub fn execute(&self, query: &str, options: QueryOptions) -> Result<QueryResults>;
```

```rust
// BEFORE (v0.1.0)
pub struct Triple {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
}

// AFTER (v0.2.3) - BREAKING: Field removed
pub struct Triple {
    pub subject: Term,
    pub predicate: IriRef,  // Changed type
    pub object: Term,
}
```

### ✅ Non-Breaking Changes (Allowed in Patch Releases)

```rust
// BEFORE (v0.1.0)
pub fn execute(&self, query: &str) -> Result<QueryResults>;

// AFTER (v0.2.3) - NON-BREAKING: New optional parameter via overload
pub fn execute_with_options(&self, query: &str, options: QueryOptions) -> Result<QueryResults>;
```

```rust
// BEFORE (v0.1.0)
pub struct Triple {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
}

// AFTER (v0.2.3) - NON-BREAKING: New optional field with default
pub struct Triple {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
    #[serde(default)]
    pub metadata: Option<Metadata>,  // New optional field
}
```

---

## 🔔 Staying Updated

### How to Track API Changes

#### 1. CHANGELOG.md
**Most Important**: Always read CHANGELOG.md before upgrading
```bash
# View changes between versions
git log v0.3.0..v0.3.1 --oneline CHANGELOG.md
```

#### 2. Compiler Warnings
**Deprecation warnings** guide you to new APIs
```rust
warning: use of deprecated function `ConcreteStore::new`:
Use `MemoryStore::new()` or `TdbStore::open()` instead.
See API documentation for migration guidance.
```

#### 3. Migration Guides
**Step-by-step** upgrade instructions provided for each major/minor version transition.

#### 4. API Documentation
**docs.rs** updated with each release
- Stability annotations on all public APIs
- Migration examples in doc comments

#### 5. GitHub Releases
**Release notes** summarize all changes
- Breaking changes highlighted
- New features listed
- Performance improvements noted

---

## 📞 Support and Feedback

### Getting Help with API Changes

#### Documentation
- **API Docs**: https://docs.rs/oxirs-core/0.3.1
- **Migration Guides**: `/docs/MIGRATION_*.md`
- **Architecture Guide**: `/docs/ARCHITECTURE.md`

#### Community
- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Discussions**: https://github.com/cool-japan/oxirs/discussions
- **Discord**: https://discord.gg/oxirs

#### Reporting Breaking Changes
If you encounter an **undocumented breaking change**, please report:
1. Open GitHub issue with `breaking-change` label
2. Include code example showing breakage
3. Specify versions (before/after)

---

## 🎯 Summary: What You Can Rely On

### ✅ Safe for Production (v0.3.1)

**Core RDF Operations**:
- ✅ `oxirs-core`: Store trait, RDF model, error types
- ✅ `oxirs-tdb`: Persistent storage (TDB format stable)
- ✅ `oxirs-arq`: SPARQL 1.1 query engine
- ✅ `oxirs-fuseki`: SPARQL HTTP protocol server

**Configuration**:
- ✅ `oxirs.toml` format (additive only)
- ✅ Environment variables (no changes)
- ✅ CLI arguments (no breaking changes)

**Data Formats**:
- ✅ TDB storage format (stable)
- ✅ RDF serialization (Turtle, N-Triples, RDF/XML, JSON-LD)
- ✅ SPARQL 1.1 syntax (W3C compliant)

**Deployment**:
- ✅ Docker images (same interface)
- ✅ Kubernetes manifests (backward compatible)
- ✅ HTTP endpoints (SPARQL 1.1 Protocol compliant)

### ⚠️ Use with Caution (Unstable)

**Distributed Features**:
- ⚠️ `oxirs-cluster`: Raft configuration may change
- ⚠️ `oxirs-stream`: Event processing API evolving
- ⚠️ `oxirs-federate`: Federation strategy may change

**Advanced Queries**:
- ⚠️ `oxirs-rule`: Rule DSL may expand
- ⚠️ `oxirs-gql`: GraphQL schema may change

**Validation**:
- ⚠️ `oxirs-shacl`: Validation API may add async support

### 🚨 Experimental Only (Research Preview)

**AI Features**:
- 🚨 `oxirs-embed`: Embedding models may change
- 🚨 `oxirs-chat`: RAG pipeline evolving
- 🚨 `oxirs-shacl-ai`: Shape learning research preview

**Specialized Extensions**:
- 🚨 `oxirs-geosparql`: Spatial indexing experimental
- 🚨 Quantum optimization (research only)
- 🚨 Neuro-symbolic reasoning (research only)

---

## 🔮 Future Stability (Post-v1.0.0)

### Long-Term API Guarantees (v1.x.y)

Once we reach v1.0.0 (planned, date TBD):

- ✅ **10-year stability guarantee** for core APIs
- ✅ **Zero breaking changes** within v1.x series
- ✅ **Deprecation-only removals** (with 12-month notice)
- ✅ **LTS releases** with extended support (3 years)

### Post-v1.0.0 Policy

```
v1.0.0 ─────────────────────────────────────────► v2.0.0
  │                                                   │
  ├─ v1.1.0 (new features, no breaking changes)      │
  ├─ v1.2.0 (new features, deprecations)             │
  ├─ v1.3.0 (new features)                           │
  │                                                   │
  └─ v1.x.y LTS (3-year support)                     └─ Breaking changes
```

**v1.x → v2.x transition**:
- Minimum 12-month deprecation notice
- Comprehensive migration tooling
- Gradual migration path (compatibility shims)
- LTS support for v1.x during transition

---

## ✅ Conclusion

**OxiRS v0.3.1** establishes a **clear stability contract**:

1. **Core APIs (🟢 Stable)**: Safe for production, backward compatible within the current minor series (v0.3.x)
2. **Distributed APIs (🟡 Unstable → 🟢 Stable)**: Most now stable, safe for production
3. **AI APIs (🔴→🟡 Unstable)**: Most now unstable, approaching stability
4. **Research APIs (🔴 Experimental)**: Shape learning remains research preview

**Commitment**: We prioritize **smooth upgrades** with **clear migration paths** over rapid breaking changes.

**Timeline**:
- **v0.1.0 (January 2026)**: Initial production release with stable APIs
- **v0.2.0–v0.2.4 (March 2026)**: ~10x query performance, full-text search, 3D GeoSPARQL, 1000+ node clustering, AI hardening
- **v0.3.0 (May 2026)**: Audit/compliance, certification, SSO, marketplace, additional SHACL features
- **v0.3.1 (June 2026, current)**: Pure-Rust crypto/compression/TLS migration, FIPS feature gates, additional SHACL-AF
- **v1.0.0 (planned, date TBD)**: Long-term stability guarantee

---

*API Stability Guarantees - Last Updated June 6, 2026 (initial: January 7, 2026)*
*Version: 0.3.1*
*Status: Production-Ready*
