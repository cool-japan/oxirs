# OxiRS Architecture Guide

**Version**: 0.3.1
**Date**: June 6, 2026
**Status**: Production Architecture

## 🏗️ System Overview

OxiRS is a modular, high-performance semantic web platform built in Rust. It provides a complete SPARQL 1.1/1.2 implementation with AI augmentation, distributed storage, and real-time streaming capabilities.

### Key Design Principles

1. **Modularity**: Each component is independently usable
2. **Performance**: SIMD, parallel processing, zero-copy operations
3. **Safety**: Rust's memory safety guarantees throughout
4. **Scalability**: Horizontal scaling via clustering
5. **Observability**: Built-in metrics, tracing, and monitoring

---

## 📦 Module Organization

```
oxirs/
├── core/               # Foundation modules
│   └── oxirs-core      # RDF data model, SPARQL basics
├── engine/             # Query processors
│   ├── oxirs-arq       # SPARQL query engine
│   ├── oxirs-rule      # RDFS/OWL reasoning
│   ├── oxirs-shacl     # SHACL validation
│   ├── oxirs-shacl-ai  # AI-enhanced SHACL
│   ├── oxirs-star      # RDF-star support
│   └── oxirs-geosparql # GeoSPARQL support
├── server/             # HTTP servers
│   ├── oxirs-fuseki    # SPARQL HTTP server
│   └── oxirs-gql       # GraphQL server
├── storage/            # Persistence layers
│   ├── oxirs-tdb       # Disk storage (B+ trees)
│   └── oxirs-cluster   # Distributed storage (Raft)
├── stream/             # Real-time processing
│   ├── oxirs-stream    # Stream processing
│   └── oxirs-federate  # Federation support
├── ai/                 # ML integration
│   ├── oxirs-embed     # Vector embeddings
│   ├── oxirs-chat      # RAG system
│   └── oxirs-vec       # Vector search
└── tools/              # CLI utilities
    └── oxirs           # Command-line tool
```

---

## 🔍 Core Architecture

### RDF Data Model (oxirs-core)

#### Term Hierarchy
```rust
pub enum Term {
    NamedNode(NamedNode),    // IRI
    BlankNode(BlankNode),    // _:identifier
    Literal(Literal),        // "value"^^datatype
    Variable(Variable),      // ?var or $var
}

pub struct Triple {
    subject: NamedOrBlankNode,
    predicate: NamedNode,
    object: Term,
}

pub struct Quad {
    triple: Triple,
    graph: GraphName,  // Named graph or default
}
```

#### Storage Interface
```rust
pub trait Store {
    fn insert_quad(&mut self, quad: Quad) -> Result<()>;
    fn delete_quad(&self, quad: &Quad) -> Result<()>;
    fn quads(&self, pattern: QuadPattern) -> Box<dyn Iterator<Item = Quad>>;
    fn contains(&self, quad: &Quad) -> bool;
}

pub struct ConcreteStore {
    // In-memory hashmap-based implementation
    // O(1) lookups by subject, predicate, object, graph
    index: MultiIndex<Quad>,
}
```

### Design Patterns

#### 1. Multi-Index Pattern
**Purpose**: Efficient triple pattern matching

```rust
struct MultiIndex<T> {
    spog: BTreeMap<(S, P, O, G), T>,  // Subject-Predicate-Object-Graph
    posg: BTreeMap<(P, O, S, G), T>,  // Predicate-Object-Subject-Graph
    ospg: BTreeMap<(O, S, P, G), T>,  // Object-Subject-Predicate-Graph
    // ... additional indexes
}

impl MultiIndex {
    fn query(&self, pattern: Pattern) -> impl Iterator<Item = &T> {
        // Select optimal index based on pattern
        match pattern {
            (Some(s), Some(p), _, _) => self.spog.range(...),
            (_, Some(p), Some(o), _) => self.posg.range(...),
            // ...
        }
    }
}
```

**Benefits**:
- O(log n) lookups instead of O(n) scans
- Index selection based on query pattern
- Memory overhead: 4-6x data size

#### 2. Streaming Iterator Pattern
**Purpose**: Memory-efficient result processing

```rust
pub trait StreamingIterator {
    type Item;
    fn next(&mut self) -> Option<&Self::Item>;
}

// Used for large result sets without allocating all at once
impl Store {
    fn quads_streaming(&self, pattern: QuadPattern)
        -> impl StreamingIterator<Item = Quad> {
        // Stream results without buffering
    }
}
```

**Benefits**:
- Constant memory usage regardless of result size
- Efficient pipelining of operations
- Zero-copy when possible

---

## ⚙️ Query Engine Architecture (oxirs-arq)

### Query Execution Pipeline

```
SPARQL Query
    ↓
[Parser] → Parse SPARQL syntax
    ↓
[Algebra Generator] → Convert to algebra tree
    ↓
[Optimizer] → Cost-based optimization
    ↓
[Physical Planner] → Select join algorithms
    ↓
[Executor] → Vectorized execution
    ↓
Results
```

### SPARQL Algebra

#### Algebra Types
```rust
pub enum Algebra {
    BGP(Vec<TriplePattern>),           // Basic graph pattern
    Join(Box<Algebra>, Box<Algebra>),  // Inner join
    LeftJoin(Box<Algebra>, Box<Algebra>, Expression),  // OPTIONAL
    Union(Box<Algebra>, Box<Algebra>), // UNION
    Filter(Box<Algebra>, Expression),  // FILTER
    Extend(Box<Algebra>, Variable, Expression),  // BIND
    Group(Box<Algebra>, Vec<Variable>, Vec<Aggregate>),  // GROUP BY
    OrderBy(Box<Algebra>, Vec<OrderCondition>),  // ORDER BY
    Project(Box<Algebra>, Vec<Variable>),  // SELECT
    Distinct(Box<Algebra>),  // DISTINCT
    Slice(Box<Algebra>, Option<usize>, Option<usize>),  // LIMIT/OFFSET
}
```

#### Optimization Rules

**1. Filter Pushdown**
```rust
// Before
Filter(Join(A, B), expr)

// After (if expr only uses variables from A)
Join(Filter(A, expr), B)
```

**2. Join Reordering**
```rust
// Cost-based join ordering
fn optimize_join_order(joins: Vec<Algebra>) -> Algebra {
    let costs: Vec<_> = joins.iter()
        .map(|j| estimate_cardinality(j))
        .collect();

    // Dynamic programming to find optimal order
    find_optimal_join_order(&joins, &costs)
}
```

**3. Common Subexpression Elimination**
```rust
// Before
Join(
    Filter(BGP(...), expr1),
    Filter(BGP(...), expr2)  // Same BGP
)

// After
let bgp = BGP(...);
Join(Filter(bgp.clone(), expr1), Filter(bgp, expr2))
```

### Join Algorithms

#### 1. Hash Join (Default)
```rust
pub struct HashJoin {
    left: Vec<Solution>,     // Materialized
    right: impl Iterator<Item = Solution>,
    hash_table: HashMap<Vec<Term>, Vec<Solution>>,
}

impl HashJoin {
    fn execute(&mut self) -> impl Iterator<Item = Solution> {
        // Build phase: hash left side
        for solution in &self.left {
            let key = extract_join_keys(solution);
            self.hash_table.entry(key).or_default().push(solution.clone());
        }

        // Probe phase: stream right side
        self.right.flat_map(|right_solution| {
            let key = extract_join_keys(&right_solution);
            self.hash_table.get(&key).into_iter().flatten()
                .map(move |left_solution| merge(left_solution, &right_solution))
        })
    }
}
```

**Characteristics**:
- Time: O(n + m)
- Space: O(n) for hash table
- Best for: Equi-joins with small left side

#### 2. Nested Loop Join
```rust
pub struct NestedLoopJoin {
    left: impl Iterator<Item = Solution>,
    right: impl Iterator<Item = Solution> + Clone,
}

impl NestedLoopJoin {
    fn execute(&mut self) -> impl Iterator<Item = Solution> {
        self.left.flat_map(|left_solution| {
            self.right.clone()
                .filter_map(move |right_solution| {
                    try_merge(&left_solution, &right_solution)
                })
        })
    }
}
```

**Characteristics**:
- Time: O(n * m)
- Space: O(1)
- Best for: Very small inputs or non-equi-joins

#### 3. Merge Join
```rust
pub struct MergeJoin {
    left: impl Iterator<Item = Solution>,   // Sorted
    right: impl Iterator<Item = Solution>,  // Sorted
}

impl MergeJoin {
    fn execute(&mut self) -> impl Iterator<Item = Solution> {
        // Merge sorted streams
        // O(n + m) time, O(1) space
    }
}
```

**Characteristics**:
- Time: O(n + m) if sorted, O(n log n + m log m) with sort
- Space: O(1) streaming
- Best for: Already sorted inputs

### Vectorized Execution

#### SIMD Operations (SciRS2 Integration)
```rust
use scirs2_core::simd::{SimdArray, auto_vectorize};

pub fn filter_numeric(values: &[i64], threshold: i64) -> Vec<bool> {
    // SIMD-optimized comparison
    auto_vectorize(values, |chunk| {
        chunk.iter().map(|&v| v > threshold).collect()
    })
}

pub fn vector_similarity(a: &[f32], b: &[f32]) -> f32 {
    // SIMD dot product
    scirs2_core::simd_ops::simd_dot_product(a, b)
}
```

**Performance**:
- 4-8x speedup for numeric operations
- Automatic vectorization for supported types
- Fallback to scalar for edge cases

---

## 🗄️ Storage Architecture

### TDB Storage (oxirs-tdb)

#### B+ Tree Design
```
Index Structure:
┌──────────────────────────────────────┐
│       B+ Tree (SPOG Index)           │
├──────────────────────────────────────┤
│ Root Node                            │
│  ├─ Pointer to Node 1                │
│  └─ Pointer to Node 2                │
├──────────────────────────────────────┤
│ Leaf Node 1                          │
│  ├─ (S1, P1, O1, G1) → Data          │
│  ├─ (S1, P1, O2, G1) → Data          │
│  └─ Next leaf pointer                │
├──────────────────────────────────────┤
│ Leaf Node 2                          │
│  ├─ (S2, P1, O1, G1) → Data          │
│  └─ ...                              │
└──────────────────────────────────────┘
```

#### Dictionary Encoding
```rust
pub struct Dictionary {
    // String → ID
    forward: HashMap<String, u64>,
    // ID → String
    reverse: HashMap<u64, String>,
    next_id: AtomicU64,
}

// Example:
// "http://xmlns.com/foaf/0.1/name" → 1
// "Alice" → 2
// "Bob" → 3

// Quad before encoding:
// (<http://example.org/alice>, <foaf:name>, "Alice", <default>)

// Quad after encoding:
// (100, 1, 2, 0)  // 4 × 8 bytes = 32 bytes instead of ~100 bytes
```

**Benefits**:
- 3-5x space reduction
- Faster comparisons (integer vs string)
- Cache-friendly memory layout

#### Buffer Pool
```rust
pub struct BufferPool {
    // LRU cache of disk pages
    cache: LruCache<PageId, Page>,
    capacity: usize,
    disk: File,
}

impl BufferPool {
    fn get_page(&mut self, page_id: PageId) -> &Page {
        if !self.cache.contains(&page_id) {
            // Cache miss: load from disk
            let page = self.disk.read_page(page_id)?;
            self.cache.put(page_id, page);
        }
        self.cache.get(&page_id).unwrap()
    }
}
```

**Performance Characteristics**:
- Cache hit: ~10ns
- Cache miss: ~1ms (disk I/O)
- Typical hit rate: 95%+ for OLTP workloads

### Distributed Storage (oxirs-cluster)

#### Raft Consensus

```
Leader Election:
┌─────────┐  heartbeat  ┌─────────┐
│ Leader  │─────────────>│Follower1│
│ Node 1  │<─────────────│         │
└─────────┘   ack        └─────────┘
     │
     │ heartbeat
     v
┌─────────┐
│Follower2│
│         │
└─────────┘

Log Replication:
Leader:   [Entry1][Entry2][Entry3]
Follower1:[Entry1][Entry2][Entry3]  ← Replicated
Follower2:[Entry1][Entry2][ ]        ← Replicating
```

#### Data Partitioning

**Consistent Hashing**:
```rust
pub struct ConsistentHashRing {
    // Virtual nodes for even distribution
    ring: BTreeMap<u64, NodeId>,
    virtual_nodes: usize,  // 150 per physical node
}

impl ConsistentHashRing {
    fn get_node(&self, key: &str) -> NodeId {
        let hash = fnv1a_hash(key);
        // Find first node >= hash in ring
        self.ring.range(hash..).next()
            .or_else(|| self.ring.iter().next())  // Wrap around
            .map(|(_, node)| *node)
            .unwrap()
    }
}
```

**Benefits**:
- Minimal data movement on node add/remove
- Even load distribution
- O(log n) node lookup

---

## 🌐 Server Architecture

### HTTP Server Stack (oxirs-fuseki)

```
HTTP Request
    ↓
[TLS Termination] → HTTPS support
    ↓
[Compression] → Gzip/Brotli
    ↓
[Security Headers] → HSTS, CSP, etc.
    ↓
[CORS] → Cross-origin handling
    ↓
[Authentication] → OAuth2/JWT
    ↓
[Authorization] → RBAC
    ↓
[Rate Limiting] → Token bucket
    ↓
[Request Logging] → Correlation IDs
    ↓
[Query Routing] → SPARQL/GraphQL
    ↓
[Response Formatting] → JSON/XML/CSV
    ↓
HTTP Response
```

### Middleware Pattern
```rust
pub trait Middleware {
    fn call(&self, req: Request) -> Response;
}

pub struct MiddlewareChain {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewareChain {
    fn execute(&self, req: Request) -> Response {
        let mut req = req;
        for middleware in &self.middlewares {
            req = middleware.call(req);
        }
        req
    }
}
```

---

## 🔄 Streaming Architecture (oxirs-stream)

### Event Processing Pipeline

```
Event Source (Kafka/NATS)
    ↓
[Deserializer] → Convert to RDF
    ↓
[Window Operator] → Tumbling/Sliding/Session windows
    ↓
[Aggregation] → Count/Sum/Avg per window
    ↓
[Filter] → SPARQL-based filtering
    ↓
[Pattern Detection] → Complex event processing
    ↓
[Sink] → Output to store/stream
```

### Watermark Strategy
```rust
pub enum WatermarkStrategy {
    Periodic(Duration),      // Emit watermark every N seconds
    PerRecord,               // Emit watermark per record
    Bounded(Duration),       // Max out-of-orderness
}

pub struct WatermarkGenerator {
    strategy: WatermarkStrategy,
    current_watermark: Instant,
}

impl WatermarkGenerator {
    fn should_emit(&mut self, event: &Event) -> bool {
        match self.strategy {
            WatermarkStrategy::Bounded(max_delay) => {
                event.timestamp + max_delay > self.current_watermark
            }
            // ...
        }
    }
}
```

---

## 🧠 AI Architecture

### RAG Pipeline (oxirs-chat)

```
User Query
    ↓
[Query Embedding] → Convert to vector
    ↓
[Vector Search] → Find relevant RDF triples
    ↓
[Context Assembly] → Build prompt with retrieved data
    ↓
[LLM Inference] → Generate response
    ↓
[Response Validation] → Verify against RDF
    ↓
User Response
```

### Embedding Storage
```rust
pub struct EmbeddingStore {
    // Vector index for similarity search
    index: HNSWIndex<f32>,
    // Triple → Embedding mapping
    triple_to_embedding: HashMap<Triple, Vec<f32>>,
    // Embedding → Triple mapping
    embedding_to_triple: HashMap<EmbeddingId, Triple>,
}

impl EmbeddingStore {
    fn similar_triples(&self, query: &str, k: usize) -> Vec<Triple> {
        let query_embedding = self.embed(query);
        let nearest = self.index.search(&query_embedding, k);
        nearest.into_iter()
            .map(|id| self.embedding_to_triple[&id].clone())
            .collect()
    }
}
```

---

## 📊 Performance Characteristics

### Time Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Insert quad | O(log n) | B+ tree insert |
| Delete quad | O(log n) | B+ tree delete |
| Pattern match | O(log n + k) | Index lookup + results |
| Simple SELECT | O(log n + k) | Single pattern |
| 2-way JOIN | O(n + m) | Hash join |
| n-way JOIN | O(n₁ + n₂ + ... + nₙ) | Optimal ordering |
| GROUP BY | O(n log n) | Sort + aggregate |
| ORDER BY | O(n log n) | Sort |

### Space Complexity

| Component | Space Usage | Notes |
|-----------|-------------|-------|
| Raw triples | 100-200 bytes/triple | Depends on IRI length |
| Dictionary encoded | 32 bytes/triple | 4 × 8-byte integers |
| Indexes | 4-6x data size | Multiple B+ trees |
| Buffer pool | 1-10GB | Configurable |
| Query cache | 100MB-1GB | LRU eviction |

---

## 🔒 Security Architecture

### Authentication Flow

```
1. Client → Server: GET /query (no token)
2. Server → Client: 401 Unauthorized + WWW-Authenticate header
3. Client → OAuth2 Provider: Authorization request
4. OAuth2 Provider → Client: Authorization code
5. Client → OAuth2 Provider: Exchange code for token
6. OAuth2 Provider → Client: Access token + refresh token
7. Client → Server: GET /query + Authorization: Bearer <token>
8. Server: Validate token signature
9. Server: Check token expiration
10. Server: Verify scopes
11. Server → Client: 200 OK + query results
```

### Defense in Depth

**Layer 1: Network**
- TLS 1.3 required
- Certificate pinning
- Firewall rules

**Layer 2: Application**
- Rate limiting
- CSRF tokens
- Security headers

**Layer 3: Authentication**
- OAuth2/OIDC
- JWT validation
- Session management

**Layer 4: Authorization**
- RBAC
- Resource-level permissions
- Audit logging

**Layer 5: Data**
- Encryption at rest
- Input validation
- Output sanitization

---

## 🎯 Design Trade-offs

### Memory vs Speed
**Choice**: Multiple indexes (4-6x memory overhead)
**Rationale**: Query performance is critical, memory is cheap
**Alternative**: Single index with sequential scans (slower but less memory)

### Consistency vs Availability (Cluster)
**Choice**: Strong consistency (Raft consensus)
**Rationale**: Data integrity is paramount for RDF
**Alternative**: Eventual consistency (higher availability, risk of conflicts)

### Compiled vs Interpreted Queries
**Choice**: Interpreted execution
**Rationale**: Flexibility and simplicity
**Alternative**: JIT compilation (faster but more complex)

### Synchronous vs Asynchronous I/O
**Choice**: Asynchronous (Tokio)
**Rationale**: Better resource utilization for I/O-bound operations
**Alternative**: Synchronous (simpler but lower throughput)

---

## 🚀 Scaling Patterns

### Vertical Scaling
- Add more CPU cores → Parallel query execution
- Add more memory → Larger buffer pool and caches
- Add faster disks → SSD/NVMe for TDB storage

### Horizontal Scaling
- Add cluster nodes → Distributed storage (Raft)
- Add read replicas → Query load distribution
- Add stream processors → Event processing parallelism

### Optimization Strategies
1. **Caching**: Query results, compiled expressions
2. **Batching**: Bulk inserts, parallel processing
3. **Indexing**: Multiple indexes for fast lookups
4. **Compression**: Dictionary encoding, column compression
5. **Partitioning**: Shard data across nodes

---

## 📚 References

- **RDF 1.1**: https://www.w3.org/TR/rdf11-concepts/
- **SPARQL 1.1**: https://www.w3.org/TR/sparql11-query/
- **Raft Consensus**: https://raft.github.io/
- **B+ Trees**: Comer, D. (1979). "Ubiquitous B-Tree"
- **Join Algorithms**: Graefe, G. (1993). "Query Evaluation Techniques"

---

*Architecture Guide - June 6, 2026*
*Production-ready architecture for v0.3.1*
