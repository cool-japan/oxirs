# OxiRS API Stability Guarantees

**Version**: v0.1.0-rc.2
**Date**: December 21, 2025
**Status**: Production-Ready
**Stability Level**: Release Candidate

---

## 🎯 Overview

This document defines OxiRS's API stability guarantees, versioning policy, and deprecation procedures. Starting with v0.1.0-rc.2, we commit to **backward compatibility** within the v0.1.x series and establish a clear path to v1.0.0 stability. As of v0.1.0-rc.2, stability guarantees are now expanded with distributed storage and AI modules reaching unstable/stable status.

---

## 📜 Stability Levels

OxiRS APIs are classified into three stability levels:

### 🟢 Stable (Production-Ready)

**Guarantee**: These APIs will NOT change in backward-incompatible ways within the same major version (v0.x → v0.y)

**Policy**:
- ✅ Guaranteed backward compatibility within v0.1.x
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

**Guarantee**: These APIs may change between minor versions (v0.1.x → v0.2.x) but will provide migration guides

**Policy**:
- ⚠️ May change in v0.2.0
- ⚠️ Suitable for testing and evaluation
- ⚠️ Changes documented in CHANGELOG
- ⚠️ Migration examples provided

**Examples**:
```rust
// Unstable APIs in oxirs-shacl-ai
pub struct ShapeLearner { /* ... */ }
impl ShapeLearner {
    pub fn learn_shapes(&self, graph: &Graph) -> Result<ShapeSchema>;
    // May change to take additional parameters in v0.2.0
}

// Unstable APIs in oxirs-embed
pub struct EmbeddingModel { /* ... */ }
impl EmbeddingModel {
    pub fn encode(&self, text: &str) -> Result<Vec<f32>>;
    // Vector size may become configurable in v0.2.0
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

**Timeline**: v1.0.0 planned for Q2 2026 (after RC.1 → RC.1 → RC.1 → Stable)

#### Minor Version (0.1.x → 0.2.x)
**Non-breaking additions and unstable API changes**:
- New stable APIs
- New features behind feature flags
- Deprecations (with 3-month notice)
- Changes to unstable APIs
- Performance improvements

**Timeline**: v0.2.0 planned for Q1 2026 (3 months after RC.1)

#### Patch Version (0.1.0 → 0.1.1)
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
// Version 0.1.0-rc.2 (Deprecation announcement)
#[deprecated(
    since = "0.1.0-rc.2",
    note = "Use `MemoryStore::new()` or `TdbStore::open()` instead. \
            See migration guide: docs/MIGRATION_ALPHA3_BETA1.md"
)]
pub fn ConcreteStore::new() -> Result<Self> {
    // Still works, but emits warning
}

// Version 0.2.0 (Removal - 3 months later)
// ConcreteStore::new() removed entirely
```

### Current Deprecations (v0.1.0-rc.2)

| Deprecated API | Replacement | Removal Version | Migration Guide |
|----------------|-------------|-----------------|-----------------|
| `ConcreteStore::new()` | `MemoryStore::new()` | v0.2.0 | [Migration Guide](MIGRATION_ALPHA3_BETA1.md#1-concretestore-construction) |
| `QueryExecutor::execute_query()` | `QueryExecutor::execute()` | v0.2.0 | [Migration Guide](MIGRATION_ALPHA3_BETA1.md#2-query-execution-api) |
| `FederationConfig` struct | `FederationClient::builder()` | v0.2.0 | [Migration Guide](MIGRATION_ALPHA3_BETA1.md#3-federation-configuration) |
| String-based errors | `OxirsError` enum | v0.2.0 | [Migration Guide](MIGRATION_ALPHA3_BETA1.md#4-error-handling) |
| Manual health endpoints | Built-in `/$/ping` endpoints | v0.2.0 | [Migration Guide](MIGRATION_ALPHA3_BETA1.md#5-manual-health-check-endpoints) |

---

## 📦 Module Stability Matrix

### Core Foundation (🟢 Stable)

#### oxirs-core (v0.1.0-rc.2)
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

#### oxirs-tdb (v0.1.0-rc.2)
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

#### oxirs-arq (v0.1.0-rc.2)
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

#### oxirs-rule (v0.1.0-rc.2)
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

#### oxirs-fuseki (v0.1.0-rc.2)
**Stability**: 🟢 **Stable** (95% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `FusekiServer::new()` | 🟢 Stable | Create server instance |
| `FusekiServer::start()` | 🟢 Stable | Start HTTP server |
| `FusekiConfig` | 🟢 Stable | Server configuration |
| HTTP endpoints (`/$/ping`, `/query`, `/update`) | 🟢 Stable | SPARQL 1.1 Protocol compliant |
| Admin UI | 🟡 Unstable | UI may be redesigned |

**Breaking Change Risk**: **VERY LOW** (< 5%)

#### oxirs-gql (v0.1.0-rc.2)
**Stability**: 🟡 **Unstable** (80% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `GraphQLServer::new()` | 🟡 Unstable | May add schema customization |
| GraphQL schema | 🟡 Unstable | Schema may expand with new types |
| Query resolvers | 🟡 Unstable | Resolver optimization may change signatures |

**Breaking Change Risk**: **MEDIUM** (20%)

---

### Storage & Distribution (🟡 Unstable)

#### oxirs-cluster (v0.1.0-rc.2)
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

#### oxirs-shacl (v0.1.0-rc.2)
**Stability**: 🟡 **Unstable** (75% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `ShaclValidator::validate()` | 🟡 Unstable | Validation API may add async support |
| `ValidationReport` | 🟡 Unstable | Report format may expand |
| `Shape` types | 🟡 Unstable | SHACL-SPARQL support may change shape types |

**Breaking Change Risk**: **MEDIUM** (25%)

#### oxirs-shacl-ai (v0.1.0-rc.2)
**Stability**: 🔴 **Experimental** (50% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `ShapeLearner` | 🔴 Experimental | AI-based shape learning is research preview |
| `ShapeInferenceModel` | 🔴 Experimental | Model architecture may change significantly |

**Breaking Change Risk**: **HIGH** (50%)

---

### AI & Machine Learning (🔴 Experimental)

#### oxirs-embed (v0.1.0-rc.2)
**Stability**: 🔴 **Experimental** (60% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `EmbeddingModel::encode()` | 🟡 Unstable | Encoding API stabilizing in RC.1 |
| `VectorStore` | 🔴 Experimental | Storage format may change |
| Similarity search | 🔴 Experimental | Algorithm may be replaced |

**Breaking Change Risk**: **HIGH** (40%)

#### oxirs-chat (v0.1.0-rc.2)
**Stability**: 🔴 **Experimental** (50% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `ChatEngine` | 🔴 Experimental | RAG pipeline is research preview |
| `ContextRetriever` | 🔴 Experimental | Retrieval strategy may change |
| Response generation | 🔴 Experimental | LLM integration may change |

**Breaking Change Risk**: **HIGH** (50%)

---

### Streaming & Federation (🟡 Unstable)

#### oxirs-stream (v0.1.0-rc.2)
**Stability**: 🟡 **Unstable** (65% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `StreamProcessor` | 🟡 Unstable | Kafka/NATS integration may change |
| `EventStream` | 🟡 Unstable | Event format may evolve |
| Watermark handling | 🟡 Unstable | Watermark strategy may change |

**Breaking Change Risk**: **MEDIUM** (35%)

#### oxirs-federate (v0.1.0-rc.2)
**Stability**: 🟡 **Unstable** (80% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `FederationClient::builder()` | 🟢 Stable | Builder API frozen |
| `FederationClient::query()` | 🟡 Unstable | Query execution may add caching |
| Endpoint discovery | 🔴 Experimental | Discovery mechanism may change |

**Breaking Change Risk**: **MEDIUM** (20%)

---

### Extensions (🔴 Experimental)

#### oxirs-star (v0.1.0-rc.2)
**Stability**: 🟡 **Unstable** (85% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| `StarTerm` | 🟡 Unstable | RDF-star types stabilizing |
| Reification strategies | 🟡 Unstable | Strategies may expand |
| Annotation syntax | 🟡 Unstable | W3C spec still evolving |

**Breaking Change Risk**: **MEDIUM** (15%)

#### oxirs-geosparql (v0.1.0-rc.2)
**Stability**: 🔴 **Experimental** (60% frozen)

| API Surface | Stability | Notes |
|-------------|-----------|-------|
| Geo functions | 🟡 Unstable | OGC GeoSPARQL compliance targeted |
| Spatial indexing | 🔴 Experimental | Index structure may change |
| CRS handling | 🔴 Experimental | Coordinate system support expanding |

**Breaking Change Risk**: **MEDIUM** (40%)

---

## 🔒 API Stability Contract (RC.1 → v1.0.0)

### What We Guarantee

#### ✅ Backward Compatibility Within v0.1.x
- **All patch releases (v0.1.1, v0.1.2, ...)** maintain 100% backward compatibility
- **No API removals** in patch releases
- **No signature changes** in patch releases
- **Only bug fixes and docs** in patch releases

#### ✅ Migration Path for v0.2.0
- **Deprecations announced 3 months in advance**
- **Migration guide** provided for all breaking changes
- **Examples** for all deprecated APIs
- **Compatibility shims** where possible

#### ✅ Data Format Stability
- **TDB format** stable across v0.1.x
- **RDF serialization** compatible across v0.1.x
- **Configuration format** backward compatible (new fields additive only)

#### ✅ Protocol Stability
- **SPARQL 1.1 Protocol** fully compliant (no deviations)
- **HTTP endpoints** stable (no URL changes)
- **GraphQL schema** additive only in v0.1.x

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

### v0.1.0-rc.2 (December 2025)
**Focus**: API freeze for core modules

- 🟢 **Stable**: oxirs-core, oxirs-arq, oxirs-fuseki, oxirs-tdb (95% frozen)
- 🟡 **Unstable**: oxirs-cluster, oxirs-shacl, oxirs-gql (70-80% frozen)
- 🔴 **Experimental**: oxirs-embed, oxirs-chat, oxirs-shacl-ai (50-60% frozen)

**Breaking Change Risk**: 10% overall

### v0.1.0-rc.2 (Current - December 2025)
**Focus**: Stabilize distributed storage, GraphQL, and AI modules

- 🟢 **Stable**: oxirs-cluster, oxirs-gql promoted to stable (90%+ frozen)
- 🟡 **Unstable**: oxirs-embed, oxirs-stream, oxirs-chat promoted to unstable (75%+ frozen)
- 🔴 **Experimental**: oxirs-shacl-ai remains experimental (research preview)
- **New**: CUDA GPU acceleration for embeddings, memory-mapped storage optimization

**Breaking Change Risk**: 5% overall

### v0.1.0-rc.2 (Q1 2026)
**Focus**: Release candidate with full API freeze

- 🟢 **Stable**: All core modules frozen (99% frozen)
- 🟡 **Unstable**: Only experimental features remain unstable
- 🔴 **Experimental**: AI research features remain experimental

**Breaking Change Risk**: < 1% overall

### v1.0.0 (Q2 2026)
**Focus**: Stable release with long-term API guarantees

- 🟢 **Stable**: 100% API freeze for core modules
- 🟡 **Unstable**: Experimental features moved to separate workspace
- 🔴 **Experimental**: None (moved to oxirs-experimental workspace)

**Breaking Change Risk**: 0% for stable APIs

---

## 🛡️ MSRV (Minimum Supported Rust Version) Policy

### Current MSRV: **Rust 1.75.0**

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
version = "0.1.0-rc.2"
rust-version = "1.75.0"  # MSRV declared
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
default = ["native-tls"]
native-tls = ["dep:native-tls"]  # TLS via system libraries
rustls = ["dep:rustls"]          # Pure Rust TLS
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
- **Stable features**: Never removed in v0.1.x
- **Unstable features**: May change in v0.2.x with migration guide
- **Experimental features**: May be removed at any time

---

## 📝 Breaking Change Examples

### ❌ Breaking Changes (Require Major/Minor Version Bump)

```rust
// BEFORE (v0.1.0-rc.2)
pub fn execute(&self, query: &str) -> Result<QueryResults>;

// AFTER (v0.2.0) - BREAKING: New required parameter
pub fn execute(&self, query: &str, options: QueryOptions) -> Result<QueryResults>;
```

```rust
// BEFORE (v0.1.0-rc.2)
pub struct Triple {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
}

// AFTER (v0.2.0) - BREAKING: Field removed
pub struct Triple {
    pub subject: Term,
    pub predicate: IriRef,  // Changed type
    pub object: Term,
}
```

### ✅ Non-Breaking Changes (Allowed in Patch Releases)

```rust
// BEFORE (v0.1.0-rc.2)
pub fn execute(&self, query: &str) -> Result<QueryResults>;

// AFTER (v0.1.1) - NON-BREAKING: New optional parameter via overload
pub fn execute_with_options(&self, query: &str, options: QueryOptions) -> Result<QueryResults>;
```

```rust
// BEFORE (v0.1.0-rc.2)
pub struct Triple {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
}

// AFTER (v0.1.1) - NON-BREAKING: New optional field with default
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
git log v0.1.0-rc.2..v0.1.0-rc.2 --oneline CHANGELOG.md
```

#### 2. Compiler Warnings
**Deprecation warnings** guide you to new APIs
```rust
warning: use of deprecated function `ConcreteStore::new`:
Use `MemoryStore::new()` or `TdbStore::open()` instead.
See migration guide: docs/MIGRATION_ALPHA3_BETA1.md
```

#### 3. Migration Guides
**Step-by-step** upgrade instructions
- [Alpha.3 → RC.1](MIGRATION_ALPHA3_BETA1.md)
- [RC.1 → RC.1](MIGRATION_BETA1_BETA2.md) *(coming soon)*

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
- **API Docs**: https://docs.rs/oxirs-core/0.1.0-rc.2
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

### ✅ Safe for Production (v0.1.0-rc.2)

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

Once we reach v1.0.0 (Q2 2026):

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

**OxiRS v0.1.0-rc.2** establishes a **clear stability contract**:

1. **Core APIs (🟢 Stable)**: Safe for production, backward compatible within v0.1.x
2. **Distributed APIs (🟡 Unstable → 🟢 Stable)**: Most now stable, safe for production
3. **AI APIs (🔴→🟡 Unstable)**: Most now unstable, approaching stability
4. **Research APIs (🔴 Experimental)**: Shape learning remains research preview

**Commitment**: We prioritize **smooth upgrades** with **clear migration paths** over rapid breaking changes.

**Timeline**:
- **RC.1 (Oct 2025)**: Core APIs frozen
- **RC.1 (Dec 2025, current)**: Distributed and AI APIs stabilized
- **RC.1 (Q1 2026)**: Full API freeze
- **v1.0.0 (Q2 2026)**: Long-term stability guarantee

---

*API Stability Guarantees - December 21, 2025*
*Version: v0.1.0-rc.2*
*Status: Production-Ready*
