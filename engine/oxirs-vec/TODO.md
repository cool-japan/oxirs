# OxiRS Vec - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Status: Production Ready

OxiRS Vec provides comprehensive vector search infrastructure for semantic similarity with full SPARQL integration.

### Features

- **HNSW Index** - Hierarchical Navigable Small World graph for approximate nearest neighbors
- **IVF Index** - Inverted File Index for large-scale search
- **Product Quantization** - PQ and OPQ for memory-efficient storage
- **Scalar Quantization** - SQ for reduced memory footprint
- **LSH Index** - Locality Sensitive Hashing for approximate search
- **NSG Index** - Navigable Small World Graph
- **DiskANN** - Billion-scale vector search with SSD optimization
- **Learned Indexes** - Neural network-based indexing with RMI architecture
- **20+ Distance Metrics** - Cosine, Euclidean, Manhattan, KL-divergence, Pearson, and more
- **GPU Acceleration** - CUDA kernels with mixed-precision and tensor core support
- **Hybrid Search** - Keyword + semantic search combination with BM25
- **Re-ranking** - Cross-encoder re-ranking with diversity-aware strategies
- **Multi-modal Search** - Text, image, audio, video with production encoders
- **Personalized Search** - User embeddings, collaborative filtering, contextual bandits
- **SPARQL Integration** - Custom functions for vector similarity queries
- **GraphQL Integration** - Vector queries through GraphQL API
- **Multi-tenancy** - Tenant isolation, quota management, billing engine
- **Write-Ahead Logging** - Crash recovery with WAL support
- **Persistence** - Zstd compression, incremental checkpointing
- **Monitoring** - Performance metrics, alerting, health monitoring
- **1598 tests passing** with zero warnings

### Key Capabilities

- Multiple indexing strategies for different use cases
- GPU-accelerated distance calculations
- Real-time index updates with streaming ingestion
- Filtered search with metadata and complex conditions
- Batch operations with transactional semantics
- Memory-efficient storage with compression
- Production-grade crash recovery
- Comprehensive monitoring and analytics

### Documentation

Production-ready guides available in `/docs`:
- Production Deployment Guide
- Performance Tuning Guide
- Best Practices Guide
- WAL Configuration Guide
- GPU Acceleration Guide
- Migration Guide from FAISS/Annoy

### Known Limitations

- **Tree indices** (Ball Tree, KD-Tree, VP-Tree): Marked as experimental with conservative depth limits. Use HNSW, IVF, or LSH for production workloads.

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ HNSW, IVF, PQ, SQ, LSH, NSG, DiskANN, learned indexes
- ✅ 20+ distance metrics, GPU acceleration, hybrid search, multi-modal
- ✅ 1598 tests passing

### v0.2.3 - Current Release (March 16, 2026)
- ✅ GPU-based index building
- ✅ Multi-GPU load balancing
- ✅ Enhanced CUDA kernel optimization
- ✅ Consensus protocol (Raft) integration
- ✅ Fault tolerance testing
- ✅ Cross-datacenter replication
- ✅ Geo-distributed deployment

### v0.3.0 - Planned (Q2 2026)
- [x] Enhanced snapshot and restore
- [x] SLA-based resource allocation
- [x] Advanced query optimization (completed 2026-04-30)
  - **Goal:** Selectivity-aware vector index choice (HNSW vs IVF vs LSH vs PQ) per-query at runtime.
  - **Design:** Cost model estimates query cost per index given (data_size, dim, requested_recall, query_density). HNSW: O(log n × M); IVF: O(n / nprobe); LSH: O(buckets); PQ: O(centroids × subquantizers). At query time dispatcher picks index with lowest estimated cost meeting requested recall. Fallback: if dispatcher misclassifies and recall drops below threshold, re-issue against next-best index. Persist runtime statistics (per-index hit/miss + recall observed) in `query_stats` for online learning of cost-model weights.
  - **Files:** `src/optimizer/{cost_model,index_dispatcher,query_stats,mod}.rs`, `src/index_dispatcher.rs`
  - **Tests:** unit on cost-model formulas + index pick under various (n, dim, recall) tuples; integration recall-vs-latency test on synthetic dataset
  - **Risk:** cost-model weights drift. Mitigation: seed with empirical data from criterion benchmarks shipped in round 1.
- [x] Full production deployment validation (completed 2026-04-30)
  - **Goal:** Chaos-test harness validating production deployment (multi-tenant, snapshot-restore, SLA) under fault injection.
  - **Design:** Build `tests/production_deployment.rs` with scenarios: multi-tenant load (10 tenants × 4 SLA classes × 1000 queries); snapshot during load (trigger PIT snapshot mid-load, verify queries continue, restore on fresh node); chaos (kill writer, verify reads continue; kill reader, verify others pick up; corrupt a checkpoint file, verify error path). Pass criteria: zero data loss, zero corruption, RTO < 30s, RPO < 1s.
  - **Files:** `tests/production_deployment.rs`
  - **Prerequisites:** snapshot+SLA from round 1 (W2-S6, already shipped)
  - **Tests:** full chaos-deployment scenario under fault injection (heavy 40k-op variant `#[ignore]`-gated)
  - **Risk:** test flakiness on slower hardware. Mitigation: relaxed thresholds in `cfg(debug_assertions)` branches.
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive benchmarks (completed 2026-04-29)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Vec v0.2.3 - Vector search infrastructure*
