# OxiRS TDB - TODO

*Version: 0.3.0 | Last Updated: May 3, 2026*

## Status: Production Ready

**oxirs-tdb** provides high-performance RDF storage with MVCC, ACID transactions, and distributed systems support.

### Features
- **MVCC + ACID Transactions** - Multi-version concurrency control
- **B+ Tree Indexing** - Optimized for streaming operations
- **TDB2 Feature Parity** - Quad indexes, inline values, RDF-star support, prefix compression
- **Database Operations** - Lifecycle management, compaction, repair
- **Observability** - Metrics collection, health checks, distributed tracing
- **Bulk Loader** - High-performance data loading with parallel processing
- **Distributed Transactions** - Two-phase commit, three-phase commit, Paxos consensus
- **Database Replication** - Master-slave and master-master replication
- **Advanced Diagnostics** - Production-ready diagnostic engine
- **GeoSPARQL Indexing** - R*-tree based spatial queries
- **Asynchronous I/O** - Non-blocking operations with io_uring support
- **Cost-Based Optimizer** - Intelligent index selection
- **Production Features** - Resource quotas, materialized views, WAL archiving, connection pooling
- **2005 tests passing** with clean build

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ MVCC + ACID, B+ Tree, TDB2 parity, bulk loader, distributed transactions, 950+ tests

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Advanced compression algorithms
- ✅ Enhanced index strategies (adaptive index, bloom index)
- ✅ Improved query optimization
- ✅ Distributed storage enhancements
- ✅ Multi-tenant isolation
- ✅ Advanced backup/restore
- ✅ Enhanced monitoring
- ✅ Six-index store, page cache, write batch, checkpoint manager
- ✅ 2005 tests passing

### v0.3.0 - Planned (Q2 2026)
- [~] Long-term support guarantees (policy: docs/policies/lts.md)
- [x] Complete TDB2 parity verification (completed 2026-04-28)
  - **Goal:** Verify behavioral parity with Apache Jena TDB2 over a defined operation matrix and fix any gaps surfaced.
  - **Scope:** Storage-engine level (oxirs-tdb layer). SPARQL execution (FILTER/OPTIONAL/UNION/aggregation/paths) lives in oxirs-arq and is excluded — classified as impl-detail divergence, not a failure. Parity is measured at the triple-pattern, MVCC transaction, six-index lookup, BNode ID, prefix-compression, and bulk-load layers.
  - **Design:** Build parity matrix in tests/tdb2_parity.rs — operations: bulk-load (Term API + BulkLoader), triple-pattern query (all 7 wildcard combos), MVCC txn (RW/RO/abort/concurrent), six-index lookup (SixIndexStore all 7 patterns), BNode ID interning via Tdb2Database, prefix-table compress/expand. Reference outputs are pinned inside the test file as inline constants.
  - **Files:** tests/tdb2_parity.rs (new), src/tdb2/mod.rs + src/six_index_store.rs + src/prefix_table.rs (gap closure if needed)
  - **Tests:** unit per-op + integration full parity matrix (load:1.0, triple-pattern:1.0, txn:1.0, index:1.0, bnode:1.0, prefix:1.0)
  - **Risk:** Jena-specific behaviors not in spec → classify as impl-detail divergence, not failure
- [~] Enterprise support (policy: docs/policies/enterprise.md, decomposed items listed therein)
- [x] Comprehensive benchmarks (completed 2026-04-29)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS TDB v0.2.3 - High-performance RDF storage*
