# OxiRS Enhancement Plan: GraphRAG, WASM, DID/VC

## Executive Summary

This plan addresses three strategic enhancements to transform OxiRS into a next-generation semantic platform:

| Enhancement | Feasibility | Effort | Priority | Impact |
|------------|-------------|--------|----------|--------|
| **1. GraphRAG (Hybrid Search)** | High (8/10) | 3-4 weeks | P1 | Very High |
| **2. WASM Support** | Medium (6/10) | 4-6 weeks | P2 | High |
| **3. DID/VC (Signed Graphs)** | High (8/10) | 3-4 weeks | P1 | Very High |

**Recommended Order:** GraphRAG → DID/VC → WASM (parallel start possible for 1+3)

---

## Current State Analysis

### Existing Foundations

```
┌─────────────────────────────────────────────────────────────────┐
│                        OxiRS Current Stack                       │
├─────────────────────────────────────────────────────────────────┤
│  AI Layer                                                        │
│  ├─ oxirs-vec (69K LoC)     Vector search, HNSW, 15+ metrics    │
│  ├─ oxirs-embed (60K LoC)   TransE, ComplEx, GNN, Transformers  │
│  ├─ oxirs-chat (33K LoC)    RAG, multi-stage retrieval          │
│  └─ oxirs-physics (1K LoC)  Thermal simulation, PINN bridge     │
├─────────────────────────────────────────────────────────────────┤
│  Engine Layer                                                    │
│  ├─ oxirs-arq               SPARQL 1.2 query engine             │
│  ├─ oxirs-rule              OWL/RDFS reasoning                  │
│  └─ oxirs-shacl             Shape validation                    │
├─────────────────────────────────────────────────────────────────┤
│  Security Layer (IDS/Gaia-X)                                     │
│  ├─ DAPS client             JWT token retrieval                 │
│  ├─ VerifiableCredential    W3C struct (incomplete)             │
│  ├─ ODRL 2.2 engine         Policy parsing & evaluation         │
│  └─ PROV-O                  Provenance tracking                 │
├─────────────────────────────────────────────────────────────────┤
│  Core Layer                                                      │
│  ├─ oxirs-core (105K LoC)   RDF store, transactions, MVCC       │
│  ├─ Named Graphs            Quad support, graph-level ops       │
│  └─ Storage backends        Memory, TDB, Cluster                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Enhancement 1: GraphRAG (Hybrid Search)

### 1.1 Problem Statement

Current OxiRS has:
- **oxirs-vec**: Vector similarity search (semantic)
- **oxirs-arq**: SPARQL graph traversal (structural)
- **Hybrid search**: BM25 + Vector fusion (keyword + semantic)

**Missing:** True GraphRAG combines **Vector Similarity × Graph Topology** in a single query:
1. Find semantically similar nodes (vector space)
2. Expand context via graph structure (2-hop neighbors, paths)
3. Generate hierarchical summaries (community detection)

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GraphRAG Query Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query: "What safety issues affect battery cells?"          │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────┐                │
│  │         1. Query Embedding (oxirs-embed)    │                │
│  │         "safety issues battery cells" → v̄   │                │
│  └─────────────────────────────────────────────┘                │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              ▼                         ▼                        │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │  2a. Vector KNN  │      │  2b. Keyword     │                │
│  │  (oxirs-vec)     │      │  Search (BM25)   │                │
│  │  Top-K similar   │      │  Term matching   │                │
│  └────────┬─────────┘      └────────┬─────────┘                │
│           │                         │                           │
│           └────────────┬────────────┘                           │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │     3. Seed Node Selection (Fusion)         │                │
│  │     RRF/Linear combination → Top-M seeds    │                │
│  └─────────────────────────────────────────────┘                │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │     4. Graph Expansion (oxirs-arq)          │                │
│  │     SPARQL: 2-hop neighbors, paths          │                │
│  │     Community detection (Louvain/Leiden)    │                │
│  └─────────────────────────────────────────────┘                │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │     5. Subgraph Extraction                  │                │
│  │     Contextual triples + metadata           │                │
│  └─────────────────────────────────────────────┘                │
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────┐                │
│  │     6. LLM Generation (oxirs-chat)          │                │
│  │     Subgraph → Natural language answer      │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Directory Structure

```
oxirs/
├─ ai/
│  └─ oxirs-graphrag/                    # [NEW] GraphRAG Orchestrator
│     ├─ Cargo.toml
│     ├─ src/
│     │  ├─ lib.rs                       # Public API
│     │  ├─ config.rs                    # GraphRAG configuration
│     │  ├─ query/
│     │  │  ├─ mod.rs
│     │  │  ├─ parser.rs                 # Natural language query parsing
│     │  │  └─ planner.rs                # Query execution planning
│     │  ├─ retrieval/
│     │  │  ├─ mod.rs
│     │  │  ├─ vector_stage.rs           # Vector KNN retrieval
│     │  │  ├─ graph_stage.rs            # Graph expansion stage
│     │  │  ├─ fusion.rs                 # RRF, linear combination
│     │  │  └─ reranker.rs               # Cross-encoder reranking
│     │  ├─ graph/
│     │  │  ├─ mod.rs
│     │  │  ├─ traversal.rs              # N-hop neighbor expansion
│     │  │  ├─ community.rs              # Community detection (Louvain)
│     │  │  ├─ summarization.rs          # Hierarchical summaries
│     │  │  └─ subgraph.rs               # Subgraph extraction
│     │  ├─ generation/
│     │  │  ├─ mod.rs
│     │  │  ├─ context_builder.rs        # Build LLM context from subgraph
│     │  │  ├─ prompt_templates.rs       # GraphRAG-specific prompts
│     │  │  └─ answer_generator.rs       # LLM answer generation
│     │  └─ sparql/
│     │     ├─ mod.rs
│     │     └─ graph_functions.rs        # SPARQL extension functions
│     └─ tests/
│        └─ integration_tests.rs
│
├─ engine/
│  └─ oxirs-vec/
│     └─ src/
│        └─ graphrag_integration.rs      # [EXTEND] GraphRAG hooks
```

### 1.4 Core API Design

```rust
// ai/oxirs-graphrag/src/lib.rs

/// GraphRAG query result
pub struct GraphRAGResult {
    /// Natural language answer
    pub answer: String,
    /// Source subgraph (RDF triples)
    pub subgraph: Vec<Triple>,
    /// Seed entities with scores
    pub seeds: Vec<ScoredEntity>,
    /// Community summaries (if enabled)
    pub communities: Vec<CommunitySummary>,
    /// Provenance information
    pub provenance: QueryProvenance,
}

/// Main GraphRAG engine
pub struct GraphRAGEngine {
    vec_index: Arc<dyn VectorIndex>,
    sparql_engine: Arc<SparqlEngine>,
    embedding_model: Arc<dyn EmbeddingModel>,
    llm_client: Arc<dyn LlmClient>,
    config: GraphRAGConfig,
}

impl GraphRAGEngine {
    /// Execute a GraphRAG query
    pub async fn query(&self, query: &str) -> Result<GraphRAGResult> {
        // 1. Embed query
        let query_vec = self.embedding_model.embed(query).await?;

        // 2. Vector retrieval (Top-K)
        let vector_results = self.vec_index.search_knn(&query_vec, self.config.top_k)?;

        // 3. Keyword retrieval (BM25)
        let keyword_results = self.keyword_search(query).await?;

        // 4. Fusion (RRF)
        let seeds = self.fuse_results(vector_results, keyword_results)?;

        // 5. Graph expansion (SPARQL)
        let subgraph = self.expand_graph(&seeds).await?;

        // 6. Community detection (optional)
        let communities = if self.config.enable_communities {
            self.detect_communities(&subgraph)?
        } else {
            vec![]
        };

        // 7. Generate answer
        let context = self.build_context(&subgraph, &communities)?;
        let answer = self.llm_client.generate(&context, query).await?;

        Ok(GraphRAGResult { answer, subgraph, seeds, communities, provenance })
    }

    /// SPARQL-integrated GraphRAG query
    pub async fn sparql_graphrag(&self, sparql: &str) -> Result<GraphRAGResult> {
        // Parse SPARQL with GraphRAG extensions
        // SELECT ?answer WHERE {
        //   GRAPHRAG:query("battery safety") ?subgraph ?answer .
        // }
    }
}

/// Configuration
pub struct GraphRAGConfig {
    /// Number of seed nodes from vector search
    pub top_k: usize,                    // Default: 20
    /// Graph expansion hops
    pub expansion_hops: usize,           // Default: 2
    /// Maximum subgraph size (triples)
    pub max_subgraph_size: usize,        // Default: 500
    /// Enable community detection
    pub enable_communities: bool,        // Default: true
    /// Community detection algorithm
    pub community_algorithm: CommunityAlgorithm, // Louvain, Leiden
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy, // RRF, Linear, Learned
    /// Path patterns for expansion
    pub path_patterns: Vec<String>,      // e.g., "?s :partOf+ ?o"
}
```

### 1.5 SPARQL Extension Functions

```sparql
PREFIX graphrag: <http://oxirs.io/graphrag#>

# GraphRAG query function
SELECT ?entity ?score ?context WHERE {
  BIND(graphrag:query("battery thermal runaway") AS ?result)
  ?result graphrag:entity ?entity ;
          graphrag:score ?score ;
          graphrag:context ?context .
}

# Hybrid search with graph expansion
SELECT ?related WHERE {
  ?seed graphrag:similar("safety hazard", 0.8) .
  ?seed ((:relatedTo|:partOf)|^(:relatedTo|:partOf)){1,2} ?related .
}

# Community-based retrieval
SELECT ?community ?summary WHERE {
  graphrag:communities(?graph, "Louvain") ?community .
  ?community graphrag:summary ?summary ;
             graphrag:entities ?entities .
}
```

### 1.6 Implementation Tasks

| Task | Description | Effort | Dependencies |
|------|-------------|--------|--------------|
| 1.1 | Create `oxirs-graphrag` crate scaffold | 2h | - |
| 1.2 | Implement query embedding pipeline | 4h | oxirs-embed |
| 1.3 | Implement vector retrieval stage | 4h | oxirs-vec |
| 1.4 | Implement graph expansion (SPARQL) | 8h | oxirs-arq |
| 1.5 | Implement RRF/fusion strategies | 4h | - |
| 1.6 | Implement community detection (Louvain) | 12h | scirs2-graph |
| 1.7 | Implement hierarchical summarization | 8h | oxirs-chat |
| 1.8 | Implement context builder | 4h | - |
| 1.9 | Implement SPARQL extension functions | 8h | oxirs-arq |
| 1.10 | Integration tests | 8h | All above |
| 1.11 | Benchmarking & optimization | 8h | - |

**Total: ~70 hours (3-4 weeks)**

---

## Enhancement 2: WASM Support

### 2.1 Feasibility Analysis

| Module | WASM Feasibility | Primary Blocker |
|--------|-----------------|-----------------|
| oxirs-ttl | ✅ Easy | None |
| oxirs-star | ✅ Easy | None |
| oxirs-arq | ⚠️ Moderate | Tokio (conditional) |
| oxirs-rule | ⚠️ Moderate | Tokio (optional) |
| oxirs-shacl | ⚠️ Moderate | Has wasmi support |
| oxirs-core | ❌ Hard | File I/O, LMDB |
| oxirs-vec | ❌ Hard | SIMD, GPU |
| oxirs-fuseki | ❌ Impossible | HTTP Server |
| oxirs-cluster | ❌ Impossible | LMDB mandatory |

**Critical Blockers:**
1. **Tokio runtime** - Used in 24+ crates, incompatible with WASM single-thread
2. **File I/O** - 60+ files with filesystem operations
3. **Memory-mapped files** - OS-specific, not available in WASM
4. **System libraries** - LMDB, RocksDB, GEOS, CUDA

### 2.2 Recommended Approach: Minimal WASM Library

**Target:** Browser-based RDF processing without server communication

```
┌─────────────────────────────────────────────────────────────────┐
│                    OxiRS WASM Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Browser/Edge Device                                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    oxirs-wasm                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │ RDF Parser  │  │ In-Memory   │  │ SPARQL      │       │  │
│  │  │ (Turtle,    │  │ Store       │  │ Engine      │       │  │
│  │  │ N-Triples)  │  │ (HashMap)   │  │ (SELECT)    │       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │ Reasoning   │  │ SHACL       │  │ JSON-LD     │       │  │
│  │  │ (RDFS)      │  │ Validation  │  │ Support     │       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           │ wasm-bindgen                         │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    JavaScript API                          │  │
│  │  const store = new OxiRSStore();                          │  │
│  │  await store.loadTurtle(ttlString);                       │  │
│  │  const results = store.query("SELECT * WHERE {...}");     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Directory Structure

```
oxirs/
├─ platforms/
│  └─ oxirs-wasm/                        # [NEW] WASM bindings
│     ├─ Cargo.toml
│     ├─ src/
│     │  ├─ lib.rs                       # wasm-bindgen exports
│     │  ├─ store.rs                     # In-memory RDF store
│     │  ├─ parser.rs                    # Turtle/N-Triples parser
│     │  ├─ query.rs                     # SPARQL query interface
│     │  ├─ reasoning.rs                 # Basic RDFS reasoning
│     │  ├─ shacl.rs                     # SHACL validation
│     │  └─ jsonld.rs                    # JSON-LD support
│     ├─ js/
│     │  ├─ index.js                     # JS wrapper
│     │  └─ types.d.ts                   # TypeScript definitions
│     ├─ tests/
│     │  └─ wasm_tests.rs
│     └─ examples/
│        ├─ browser/
│        │  ├─ index.html
│        │  └─ app.js
│        └─ node/
│           └─ example.js
│
├─ core/
│  └─ oxirs-core/
│     └─ src/
│        ├─ storage/
│        │  ├─ mod.rs
│        │  ├─ memory_storage.rs         # [EXTEND] WASM-compatible
│        │  └─ wasm_storage.rs           # [NEW] WASM-specific
│        └─ lib.rs                       # [EXTEND] Feature flags
```

### 2.4 Feature Flag Strategy

```toml
# core/oxirs-core/Cargo.toml
[features]
default = ["std", "tokio-runtime"]
std = []
wasm = ["js-sys", "web-sys", "wasm-bindgen"]
tokio-runtime = ["tokio"]

# Conditional compilation
[target.'cfg(target_arch = "wasm32")'.dependencies]
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console", "Window"] }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
getrandom = { version = "0.2", features = ["js"] }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
tokio = { version = "1", features = ["full"] }
```

### 2.5 Storage Abstraction Layer

```rust
// core/oxirs-core/src/storage/traits.rs

/// Storage backend trait (WASM-compatible)
pub trait StorageBackend: Send + Sync {
    fn insert(&mut self, triple: &Triple) -> Result<()>;
    fn delete(&mut self, triple: &Triple) -> Result<bool>;
    fn contains(&self, triple: &Triple) -> bool;
    fn iter(&self) -> Box<dyn Iterator<Item = &Triple> + '_>;
    fn len(&self) -> usize;

    // Optional persistence (not available in WASM)
    #[cfg(not(target_arch = "wasm32"))]
    fn persist(&self, path: &Path) -> Result<()>;

    #[cfg(not(target_arch = "wasm32"))]
    fn load(path: &Path) -> Result<Self> where Self: Sized;
}

/// In-memory storage (works everywhere)
pub struct MemoryStorage {
    triples: HashSet<Triple>,
    indexes: TripleIndexes,
}

impl StorageBackend for MemoryStorage {
    // Implementation works in both native and WASM
}
```

### 2.6 WASM JavaScript API

```typescript
// platforms/oxirs-wasm/js/types.d.ts

export class OxiRSStore {
    constructor();

    // Data loading
    loadTurtle(ttl: string): Promise<number>;
    loadNTriples(nt: string): Promise<number>;
    loadJsonLd(jsonld: object): Promise<number>;

    // Querying
    query(sparql: string): QueryResult[];
    ask(sparql: string): boolean;
    construct(sparql: string): Triple[];

    // Data manipulation
    insert(subject: string, predicate: string, object: string): void;
    delete(subject: string, predicate: string, object: string): boolean;

    // Validation
    validateShacl(shapesGraph: string): ValidationReport;

    // Reasoning
    infer(profile: "rdfs" | "owl-lite"): number;

    // Export
    toTurtle(): string;
    toNTriples(): string;
    toJsonLd(): object;

    // Statistics
    size(): number;
    subjects(): string[];
    predicates(): string[];
}

export interface Triple {
    subject: string;
    predicate: string;
    object: string;
    graph?: string;
}

export interface QueryResult {
    [variable: string]: string | number | boolean;
}

export interface ValidationReport {
    conforms: boolean;
    results: ValidationResult[];
}
```

### 2.7 Implementation Tasks

| Task | Description | Effort | Dependencies |
|------|-------------|--------|--------------|
| 2.1 | Create `oxirs-wasm` crate scaffold | 2h | - |
| 2.2 | Abstract storage trait (platform-agnostic) | 8h | oxirs-core |
| 2.3 | Implement in-memory WASM storage | 8h | Task 2.2 |
| 2.4 | Port Turtle parser (no I/O) | 8h | oxirs-ttl |
| 2.5 | Port N-Triples parser | 4h | oxirs-ttl |
| 2.6 | Port SPARQL engine (sync) | 16h | oxirs-arq |
| 2.7 | wasm-bindgen JS bindings | 8h | All above |
| 2.8 | TypeScript definitions | 4h | Task 2.7 |
| 2.9 | Basic RDFS reasoning | 8h | oxirs-rule |
| 2.10 | SHACL validation (in-memory) | 12h | oxirs-shacl |
| 2.11 | JSON-LD support | 8h | - |
| 2.12 | Browser example application | 8h | All above |
| 2.13 | npm package setup | 4h | - |
| 2.14 | Testing & benchmarking | 12h | All above |

**Total: ~110 hours (4-6 weeks)**

---

## Enhancement 3: DID / Verifiable Credentials

### 3.1 Problem Statement

**Current State:**
- VerifiableCredential struct exists (incomplete)
- DAPS client for IDS authentication
- ODRL policy engine
- W3C PROV-O provenance tracking
- Named graph handling

**Missing:**
- DID resolver/registrar
- Cryptographic proof generation
- Key management system
- Signed quads (graph-level signatures)

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DID/VC Trust Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     DID Layer                                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ││
│  │  │ DID:key     │  │ DID:web     │  │ DID:ebsi    │         ││
│  │  │ (Ed25519)   │  │ (.well-known│  │ (EU EBSI)   │         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘         ││
│  │                         │                                   ││
│  │                         ▼                                   ││
│  │  ┌─────────────────────────────────────────────┐           ││
│  │  │           Universal DID Resolver             │           ││
│  │  │  did:key:z6Mk... → Public Key + Metadata     │           ││
│  │  └─────────────────────────────────────────────┘           ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Verifiable Credentials Layer                 ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         ││
│  │  │ VC Issuer   │  │ VC Holder   │  │ VC Verifier │         ││
│  │  │ (Create &   │  │ (Store &    │  │ (Validate & │         ││
│  │  │  Sign)      │  │  Present)   │  │  Trust)     │         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Signed Graphs Layer                        ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │  Named Graph: <urn:graph:sensor-data-2025-12>           │││
│  │  │  ├─ :sensor1 :temperature "25.5"^^xsd:double .          │││
│  │  │  ├─ :sensor1 :timestamp "2025-12-25T10:30:00Z" .        │││
│  │  │  └─ Signature: Ed25519(hash(graph), issuer_key)         │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Directory Structure

```
oxirs/
├─ security/
│  └─ oxirs-did/                         # [NEW] DID/VC implementation
│     ├─ Cargo.toml
│     ├─ src/
│     │  ├─ lib.rs                       # Public API
│     │  ├─ did/
│     │  │  ├─ mod.rs
│     │  │  ├─ document.rs               # DID Document structure
│     │  │  ├─ resolver.rs               # Universal resolver
│     │  │  ├─ methods/
│     │  │  │  ├─ mod.rs
│     │  │  │  ├─ did_key.rs             # did:key method (Ed25519)
│     │  │  │  ├─ did_web.rs             # did:web method
│     │  │  │  └─ did_ebsi.rs            # did:ebsi (EU blockchain)
│     │  │  └─ registrar.rs              # DID creation/update
│     │  ├─ vc/
│     │  │  ├─ mod.rs
│     │  │  ├─ credential.rs             # VC structure (W3C)
│     │  │  ├─ presentation.rs           # VP structure
│     │  │  ├─ issuer.rs                 # Credential issuance
│     │  │  ├─ verifier.rs               # Credential verification
│     │  │  └─ holder.rs                 # Credential storage
│     │  ├─ proof/
│     │  │  ├─ mod.rs
│     │  │  ├─ ed25519.rs                # Ed25519Signature2020
│     │  │  ├─ rsa.rs                    # RsaSignature2018
│     │  │  ├─ ecdsa.rs                  # EcdsaSecp256k1Signature2019
│     │  │  └─ data_integrity.rs         # Data Integrity Proofs
│     │  ├─ signed_graph/
│     │  │  ├─ mod.rs
│     │  │  ├─ canonicalization.rs       # RDF Dataset Normalization
│     │  │  ├─ hash.rs                   # Graph hashing (SHA-256)
│     │  │  ├─ signature.rs              # Graph signing
│     │  │  └─ verification.rs           # Signature verification
│     │  ├─ key_management/
│     │  │  ├─ mod.rs
│     │  │  ├─ keystore.rs               # Encrypted key storage
│     │  │  ├─ key_derivation.rs         # HD key derivation
│     │  │  └─ hardware.rs               # HSM/TPM support (optional)
│     │  └─ rdf_integration/
│     │     ├─ mod.rs
│     │     ├─ vc_to_rdf.rs              # VC → RDF conversion
│     │     └─ sparql_functions.rs       # SPARQL extensions
│     └─ tests/
│        ├─ did_tests.rs
│        ├─ vc_tests.rs
│        └─ signed_graph_tests.rs
│
├─ server/
│  └─ oxirs-fuseki/
│     └─ src/
│        └─ ids/
│           └─ identity/
│              ├─ verifiable_credentials.rs  # [EXTEND] Complete impl
│              └─ daps.rs                    # [EXTEND] Real signing
```

### 3.4 Core API Design

```rust
// security/oxirs-did/src/lib.rs

/// DID Document (W3C DID Core 1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidDocument {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    pub id: Did,
    pub verification_method: Vec<VerificationMethod>,
    pub authentication: Vec<String>,
    pub assertion_method: Vec<String>,
    pub key_agreement: Vec<String>,
    pub service: Vec<Service>,
}

/// DID identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Did(String);

impl Did {
    pub fn new_key(public_key: &[u8]) -> Self {
        // did:key:z6Mk...
        let multibase = multibase::encode(Base::Base58Btc, public_key);
        Did(format!("did:key:{}", multibase))
    }

    pub fn new_web(domain: &str, path: Option<&str>) -> Self {
        // did:web:example.com:path
        Did(format!("did:web:{}{}", domain, path.unwrap_or("")))
    }

    pub fn method(&self) -> &str {
        self.0.split(':').nth(1).unwrap_or("unknown")
    }
}

/// Universal DID Resolver
pub struct DidResolver {
    resolvers: HashMap<String, Box<dyn MethodResolver>>,
    cache: Arc<RwLock<LruCache<Did, DidDocument>>>,
}

impl DidResolver {
    pub async fn resolve(&self, did: &Did) -> Result<DidDocument> {
        // Check cache
        if let Some(doc) = self.cache.read().await.get(did) {
            return Ok(doc.clone());
        }

        // Resolve by method
        let method = did.method();
        let resolver = self.resolvers.get(method)
            .ok_or(DidError::UnsupportedMethod(method.to_string()))?;

        let doc = resolver.resolve(did).await?;
        self.cache.write().await.put(did.clone(), doc.clone());
        Ok(doc)
    }
}

/// Verifiable Credential (W3C VC Data Model 2.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiableCredential {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    pub issuer: CredentialIssuer,
    #[serde(rename = "issuanceDate")]
    pub issuance_date: DateTime<Utc>,
    #[serde(rename = "expirationDate", skip_serializing_if = "Option::is_none")]
    pub expiration_date: Option<DateTime<Utc>>,
    #[serde(rename = "credentialSubject")]
    pub credential_subject: CredentialSubject,
    pub proof: Option<Proof>,
}

/// Credential Issuer
pub struct CredentialIssuer {
    resolver: Arc<DidResolver>,
    keystore: Arc<Keystore>,
}

impl CredentialIssuer {
    pub async fn issue(
        &self,
        subject: CredentialSubject,
        credential_type: Vec<String>,
        issuer_did: &Did,
    ) -> Result<VerifiableCredential> {
        // 1. Build credential
        let mut vc = VerifiableCredential {
            context: vec![
                "https://www.w3.org/2018/credentials/v1".to_string(),
                "https://www.w3.org/2018/credentials/examples/v1".to_string(),
            ],
            id: Some(format!("urn:uuid:{}", uuid::Uuid::new_v4())),
            credential_type: vec!["VerifiableCredential".to_string()]
                .into_iter().chain(credential_type).collect(),
            issuer: CredentialIssuer::Simple(issuer_did.clone()),
            issuance_date: Utc::now(),
            expiration_date: None,
            credential_subject: subject,
            proof: None,
        };

        // 2. Get signing key
        let key = self.keystore.get_signing_key(issuer_did).await?;

        // 3. Create proof
        let proof = self.create_proof(&vc, &key).await?;
        vc.proof = Some(proof);

        Ok(vc)
    }

    async fn create_proof(&self, vc: &VerifiableCredential, key: &SigningKey) -> Result<Proof> {
        let canonical = self.canonicalize(vc)?;
        let hash = sha256(&canonical);
        let signature = key.sign(&hash)?;

        Ok(Proof {
            proof_type: "Ed25519Signature2020".to_string(),
            created: Utc::now(),
            verification_method: key.id.clone(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: base64::encode(&signature),
        })
    }
}

/// Signed RDF Graph
pub struct SignedGraph {
    pub graph_uri: NamedNode,
    pub triples: Vec<Triple>,
    pub signature: GraphSignature,
    pub issuer: Did,
    pub issuance_date: DateTime<Utc>,
}

impl SignedGraph {
    /// Create a signed graph from triples
    pub async fn sign(
        graph_uri: NamedNode,
        triples: Vec<Triple>,
        issuer: &Did,
        keystore: &Keystore,
    ) -> Result<Self> {
        // 1. Canonicalize (RDF Dataset Normalization Algorithm)
        let canonical = rdfc10_canonicalize(&triples)?;

        // 2. Hash (SHA-256)
        let hash = sha256(canonical.as_bytes());

        // 3. Sign
        let key = keystore.get_signing_key(issuer).await?;
        let signature_bytes = key.sign(&hash)?;

        Ok(SignedGraph {
            graph_uri,
            triples,
            signature: GraphSignature {
                algorithm: "Ed25519".to_string(),
                value: base64::encode(&signature_bytes),
                hash: hex::encode(&hash),
            },
            issuer: issuer.clone(),
            issuance_date: Utc::now(),
        })
    }

    /// Verify graph signature
    pub async fn verify(&self, resolver: &DidResolver) -> Result<bool> {
        // 1. Resolve issuer DID
        let did_doc = resolver.resolve(&self.issuer).await?;

        // 2. Get verification key
        let key = did_doc.get_verification_key()?;

        // 3. Re-canonicalize and hash
        let canonical = rdfc10_canonicalize(&self.triples)?;
        let hash = sha256(canonical.as_bytes());

        // 4. Verify signature
        let signature = base64::decode(&self.signature.value)?;
        key.verify(&hash, &signature)
    }
}
```

### 3.5 SPARQL Extension Functions

```sparql
PREFIX did: <http://oxirs.io/did#>
PREFIX vc: <http://oxirs.io/vc#>

# Resolve DID to document
SELECT ?pubkey WHERE {
  BIND(did:resolve("did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK") AS ?doc)
  ?doc did:verificationMethod ?vm .
  ?vm did:publicKeyMultibase ?pubkey .
}

# Verify credential
SELECT ?valid ?issuer WHERE {
  ?credential a vc:VerifiableCredential .
  BIND(vc:verify(?credential) AS ?valid)
  ?credential vc:issuer ?issuer .
  FILTER(?valid = true)
}

# Query signed graphs
SELECT ?graph ?issuer ?valid WHERE {
  GRAPH ?graph {
    ?s ?p ?o .
  }
  ?graph did:signedBy ?issuer ;
         did:signatureValid ?valid .
  FILTER(?valid = true)
}

# Issue credential (UPDATE)
INSERT {
  ?vc a vc:VerifiableCredential ;
      vc:issuer <did:key:z6Mk...> ;
      vc:credentialSubject [
        vc:id <urn:employee:12345> ;
        vc:role "Engineer"
      ] .
}
WHERE {
  BIND(vc:issue(<did:key:z6Mk...>, "EmployeeCredential") AS ?vc)
}
```

### 3.6 Implementation Tasks

| Task | Description | Effort | Dependencies |
|------|-------------|--------|--------------|
| 3.1 | Create `oxirs-did` crate scaffold | 2h | - |
| 3.2 | Implement DID Document structure | 4h | - |
| 3.3 | Implement did:key method | 8h | ed25519-dalek |
| 3.4 | Implement did:web method | 8h | reqwest |
| 3.5 | Implement Universal Resolver | 8h | Tasks 3.3-3.4 |
| 3.6 | Implement VerifiableCredential (complete) | 8h | - |
| 3.7 | Implement Ed25519Signature2020 proof | 8h | Task 3.3 |
| 3.8 | Implement RDF canonicalization (RDFC-1.0) | 12h | oxirs-core |
| 3.9 | Implement graph hashing | 4h | Task 3.8 |
| 3.10 | Implement signed graphs | 8h | Tasks 3.8-3.9 |
| 3.11 | Implement key management | 12h | - |
| 3.12 | SPARQL extension functions | 8h | oxirs-arq |
| 3.13 | Integrate with IDS connector | 8h | oxirs-fuseki |
| 3.14 | Integration tests | 8h | All above |

**Total: ~106 hours (3-4 weeks)**

---

## Integration Points

### Cross-Enhancement Synergies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GraphRAG + DID/VC                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  GraphRAG query → Only include VERIFIED subgraphs           ││
│  │  LLM context → Include provenance chain from signed graphs  ││
│  │  Answer attribution → DID-based source identification       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  WASM + DID/VC                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Browser wallet → DID key management                        ││
│  │  Local verification → VC validation without server          ││
│  │  Privacy → Selective disclosure in browser                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  GraphRAG + WASM                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Edge GraphRAG → Local knowledge graph + vector search      ││
│  │  Offline queries → Browser-based semantic search            ││
│  │  Federated → WASM nodes contribute to distributed GraphRAG  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Schedule

### Recommended Parallel Execution

```
Week 1-2: Foundation
├─ GraphRAG: Crate scaffold, query pipeline (Tasks 1.1-1.5)
├─ DID/VC: Crate scaffold, DID methods (Tasks 3.1-3.5)
└─ WASM: Storage abstraction research

Week 3-4: Core Implementation
├─ GraphRAG: Graph expansion, community detection (Tasks 1.4-1.6)
├─ DID/VC: VC implementation, proofs (Tasks 3.6-3.7)
└─ WASM: Crate scaffold, storage impl (Tasks 2.1-2.3)

Week 5-6: Integration
├─ GraphRAG: Summarization, SPARQL extensions (Tasks 1.7-1.9)
├─ DID/VC: Signed graphs, canonicalization (Tasks 3.8-3.10)
└─ WASM: Parser porting (Tasks 2.4-2.5)

Week 7-8: WASM Core
├─ GraphRAG: Testing & optimization (Tasks 1.10-1.11)
├─ DID/VC: Key management, IDS integration (Tasks 3.11-3.13)
└─ WASM: SPARQL engine porting (Task 2.6)

Week 9-10: WASM Completion
├─ All: Integration testing
└─ WASM: JS bindings, examples, npm (Tasks 2.7-2.14)
```

### Total Effort Estimate

| Enhancement | Hours | Duration |
|-------------|-------|----------|
| GraphRAG | 70h | 3-4 weeks |
| WASM | 110h | 4-6 weeks |
| DID/VC | 106h | 3-4 weeks |
| **Total** | **286h** | **8-10 weeks** (parallel) |

---

## Dependencies

### New Crate Dependencies

```toml
# GraphRAG
petgraph = "0.6"          # Graph algorithms
community-detection = "*"  # Louvain/Leiden (or custom impl)

# WASM
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = "0.3"
wasm-bindgen-futures = "0.4"
getrandom = { version = "0.2", features = ["js"] }

# DID/VC
ed25519-dalek = "2.1"     # Ed25519 signatures
multibase = "0.9"         # Multibase encoding
multicodec = "0.2"        # Multicodec
sha2 = "0.10"             # SHA-256
bs58 = "0.5"              # Base58
jsonwebtoken = "9"        # JWT handling
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| WASM Tokio incompatibility | High | High | Abstract async runtime, use wasm-bindgen-futures |
| Community detection performance | Medium | Medium | Use scirs2-graph, optimize for RDF |
| DID method diversity | Medium | Low | Start with did:key, add others incrementally |
| Browser memory limits | Medium | Medium | Implement streaming, limit graph size |
| Cryptography in WASM | Low | High | Use ring/rustcrypto with WASM support |

---

## Success Criteria

### GraphRAG
- [ ] Vector + Graph hybrid queries return relevant results
- [ ] Community detection completes in <5s for 100K triples
- [ ] SPARQL extension functions work correctly
- [ ] LLM answers include accurate provenance

### WASM
- [ ] `oxirs-wasm` compiles to wasm32-unknown-unknown
- [ ] Browser example loads 10K triples in <1s
- [ ] SPARQL SELECT queries execute in browser
- [ ] npm package published with TypeScript types

### DID/VC
- [ ] did:key resolution works offline
- [ ] VC issuance/verification passes W3C test suite
- [ ] Signed graphs verify correctly
- [ ] IDS connector uses real JWT signing

---

## Questions for User

Before implementation, please clarify:

1. **GraphRAG Priority:**
   - Focus on Microsoft-style hierarchical summarization?
   - Or simpler vector + 2-hop expansion first?

2. **WASM Scope:**
   - Full SPARQL support (complex)?
   - Or basic SELECT-only (faster)?

3. **DID Methods:**
   - Start with did:key only (offline)?
   - Include did:web (requires HTTP)?
   - Include did:ebsi (EU blockchain)?

4. **Integration Priority:**
   - GraphRAG + DID (trusted answers)?
   - WASM + DID (browser wallet)?
   - All three equally?
