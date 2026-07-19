# OxiRS TDB - High-Performance RDF Storage Engine

[![Version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/cool-japan/oxirs/releases)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Status**: v0.4.0 - Release preparation 2026-07-19 (2155 tests passing)

✨ **Production Release**: Production-ready with API stability guarantees. Semantic versioning enforced.

A high-performance, ACID-compliant RDF storage engine with multi-version concurrency control (MVCC) and advanced transaction support. OxiRS TDB provides TDB2-equivalent functionality with modern Rust performance optimizations.

## Features

### Core Storage Engine
- **MVCC (Multi-Version Concurrency Control)**: Snapshot isolation with conflict detection
- **ACID Transactions**: Full transaction support with rollback capabilities
- **B+ Tree Indexing**: Six standard RDF indices (SPO, POS, OSP, SOP, PSO, OPS)
- **Advanced Page Management**: LRU buffer pools with efficient memory management
- **Crash Recovery**: ARIES-style write-ahead logging with analysis/redo/undo phases

### Durability & Crash Recovery (v0.4.0)

`TdbStore` mutations are now durable through an integrated write-ahead log
(`TdbConfig.enable_wal`, **default `true`**):

- **WAL-ordered writes**: every mutating op is logged Begin/Update/Commit before
  it is acknowledged, and `sync()` flushes in a strict order — WAL flush → page
  flush → superblock → WAL truncate — so a crash never leaves a committed record
  behind an already-truncated log.
- **Recovery replay on open**: `TdbStore::open` replays committed WAL operations
  recorded since the last checkpoint (`recover_from_wal`) before serving any read,
  and advances past any id a crashed session left in the WAL so replayed and
  freshly-issued dictionary ids never collide. A checkpoint records its LSN in the
  superblock; replay is idempotent, so re-applying already-persisted operations is
  safe. A crash test (`test_crash_without_sync_replays_committed_writes`) proves
  that committed-but-unsynced writes survive a reopen.
- **`StoreParams` honored end to end**: open a store with an explicit
  `StoreParams` via `TdbStore::open_with_params`, which threads the parameters
  into the engine `TdbConfig` (`TdbConfig::from_store_params`) — buffer pool size,
  bloom-filter fpr/size, query-cache size, statistics sample rate, slow-query /
  query-timeout monitors, spatial-index bounds, quad-index and WAL toggles. A
  `page_size` that differs from the compile-time page size is rejected loudly, and
  `store_params.json` is persisted so a reopen restores the same parameters.
- **Sorted bulk build**: `insert_triples_bulk` / `insert_quads_bulk` intern and
  encode a whole batch, validate it up front (a literal subject or a quad while
  quad indexes are disabled fails loudly *before* any mutation), build each index
  in sorted key order, and issue a single WAL batch + one sync per batch.
- **Opt-in direct I/O**: set `StoreParams.enable_direct_io` to bypass the OS page
  cache — Linux `O_DIRECT`, macOS `F_NOCACHE` via `fcntl` (failing loud if the
  syscall errors). The default (`false`) path is plain `std::fs`, keeping the
  default build 100% Pure Rust with no `unsafe` fcntl on the hot path.

### Distributed Transactions & Fault Tolerance
- **Two-Phase & Three-Phase Commit**: `TwoPhaseParticipant`/`ThreePhaseParticipant` can be constructed via `with_transaction_manager()` so PREPARE/COMMIT/ABORT drive a real WAL-backed `Transaction`, not just protocol bookkeeping
- **Saga Pattern**: `SagaOrchestrator` plus a `SagaCallbackRegistry` runs real registered forward-action/compensation callbacks per step, with automatic reverse-order compensation when a step fails
- **Distributed Deadlock Detection**: wait-for graph cycle detection with four victim-selection strategies — `YoungestTransaction`, `OldestTransaction`, `LeastWork` (fewest outstanding wait-for edges), and `Random`
- **Paxos Consensus**: `consensus::paxos` for coordinator agreement, alongside 2PC/3PC
- **Replication Manager**: master-slave and master-master replication with async/sync modes

  *The coordinator/participant/saga engines are an in-process protocol state machine — network transport across real nodes is provided by integration layers such as `oxirs-cluster`.*

### Performance & Scalability
- **High Throughput**: Designed for 100M+ triples with sub-second query response
- **Concurrent Access**: Support for 1000+ concurrent read/write operations
- **Efficient Storage**: Node compression with dictionary encoding
- **Memory Optimized**: <8GB memory footprint for 100M triple datasets

### Integration
- **OxiRS Ecosystem**: Seamless integration with oxirs-core and oxirs-arq
- **TDB2 Compatibility**: Feature parity with Apache Jena TDB2
- **Modern Rust**: Safe, fast, and memory-efficient implementation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query Layer   │    │ Transaction Mgr │    │   WAL Recovery  │
│   (oxirs-arq)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                        TDB Storage Engine                       │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│   Triple Store  │   Node Table    │   B+ Tree      │   MVCC   │
│                 │                 │   Indices       │  Storage │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
         │                       │                       │
┌─────────────────┬─────────────────┬─────────────────┬──────────┐
│  Page Manager   │   Assembler     │   Buffer Pool   │   WAL    │
│                 │                 │                 │          │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
```

## Quick Start

### Dependencies

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-tdb = "0.4.0"
```

### Basic Usage

```rust
use oxirs_tdb::TdbStore;

// Open (or create) a TDB store at a directory
let mut store = TdbStore::open("./data/tdb")?;

// Insert a triple (string-based convenience API)
store.insert(
    "http://example.org/subject",
    "http://example.org/predicate",
    "Hello, World!",
)?;

// Query by pattern (any of subject/predicate/object may be `None` as a wildcard)
let results = store.query_triples(None, None, None)?;
for (subject, predicate, object) in results {
    println!("{subject} {predicate} {object}");
}

println!("{} triples stored", store.count());
```

### Advanced Usage

```rust
use oxirs_core::model::{NamedNode, Term};
use oxirs_tdb::{TdbConfig, TdbStore};

// Configure with advanced options
let config = TdbConfig::new("./data/tdb")
    .with_buffer_pool_size(2000) // pages kept in the buffer pool
    .with_compression(true)
    .with_bloom_filters(true)
    .with_statistics(true);

let mut store = TdbStore::open_with_config(config)?;

// Bulk insert using Term-typed triples
let subject = Term::NamedNode(NamedNode::new("http://example.org/subject")?);
let predicate = Term::NamedNode(NamedNode::new("http://example.org/predicate")?);
let object = Term::NamedNode(NamedNode::new("http://example.org/object")?);
store.insert_triples_bulk(&[(subject.clone(), predicate.clone(), object)])?;

// Explicit WAL-backed transaction (governs locking; mutate via the store API
// while it is active, then hand the transaction back to commit/abort it)
let txn = store.begin_transaction()?;
store.commit_transaction(txn)?;
```

## Configuration

### TdbConfig Options

```rust
let config = TdbConfig::new("./data/tdb")
    // Memory management
    .with_buffer_pool_size(2000) // number of pages kept in the buffer pool

    // Storage engine features
    .with_compression(true)
    .with_bloom_filters(true)
    .with_spatial_indexing(true) // GeoSPARQL support

    // Query features
    .with_query_cache(true)
    .with_statistics(true)
    .with_query_monitoring(true);
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests with nextest (recommended)
cargo nextest run --no-fail-fast

# Run specific test categories
cargo nextest run --no-fail-fast -p oxirs-tdb

# Run with all features
cargo nextest run --no-fail-fast --all-features

# Run performance tests
cargo nextest run --no-fail-fast --release -- --ignored
```

## Performance Benchmarks

### Target Performance (100M triples)
- **Query Response**: <1 second for complex queries
- **Load Performance**: 10M triples/minute bulk loading
- **Transaction Throughput**: 10K transactions/second
- **Memory Usage**: <8GB for 100M triple database
- **Recovery Time**: <30 seconds for 1GB database

### Benchmark Results

```bash
# Run all benchmarks (bloom_filter_benchmark, storage_bench)
cargo bench -p oxirs-tdb

# Run a specific benchmark
cargo bench -p oxirs-tdb --bench storage_bench
```

## Development

### Building

```bash
# Build with all features
cargo build --all-features

# Build optimized release
cargo build --release --all-features

# Run clippy
cargo clippy --workspace --all-targets -- -D warnings

# Format code
cargo fmt --all
```

### Project Structure

```
oxirs-tdb/
├── src/
│   ├── lib.rs             # Public API surface
│   ├── store/             # TdbStore, TdbConfig, TdbStats
│   ├── transaction/       # WAL, lock manager, 2PC, 3PC, group commit
│   ├── distributed/       # Coordinator, saga, deadlock detector, replication
│   ├── consensus/         # Paxos
│   ├── btree/, btree_index.rs         # B+ tree storage engine
│   ├── index/, six_index_store.rs     # SPO/POS/OSP/SOP/PSO/OPS indices
│   ├── dictionary/        # Term dictionary / interning
│   ├── storage/           # Page & buffer pool management
│   ├── tdb2/              # TDB2-parity node table & triple index
│   ├── mvcc/              # Snapshot isolation
│   └── recovery.rs, backup.rs, wal_archive.rs, ...  # Crash recovery & backup
├── tests/                 # Integration tests
├── benches/               # bloom_filter_benchmark, storage_bench
├── examples/              # Usage examples
└── docs/                  # BENCHMARK_RESULTS.md, etc.
```

### Contributing

1. Follow the existing code style
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure all tests pass with `cargo nextest run --no-fail-fast`
5. Run clippy with `cargo clippy --workspace --all-targets -- -D warnings`

## Compatibility

### Apache Jena TDB2 Compatibility

OxiRS TDB provides feature parity with Apache Jena TDB2:

- ✅ Six standard RDF indices (SPO, POS, OSP, SOP, PSO, OPS)
- ✅ ACID transactions with MVCC
- ✅ Node compression and dictionary encoding
- ✅ B+ tree storage with efficient page management
- ✅ Write-ahead logging for crash recovery
- ✅ Statistics collection for query optimization

### TDB2-Compatible Storage

`oxirs_tdb::tdb2` provides a TDB2-parity node table and triple index (BNode ID
interning, prefix compression) for workloads that need on-disk semantics
equivalent to Apache Jena's TDB2 layer via `tdb2::Tdb2Database`.

Note: this is an OxiRS-native TDB2-compatible storage layer, not a bundled
converter for existing Apache Jena TDB2 database files — there is currently no
`migration` module.

## Troubleshooting

### Common Issues

**Q: Database corruption after crash**
A: Use the store's built-in recovery API:
```rust
let store = TdbStore::open("./data/tdb")?;
let recovery_report = store.recover()?;
let corruption_report = store.detect_corruption()?;
store.verify_indexes()?;
```

**Q: Poor query performance**
A: Inspect statistics and slow-query history:
```rust
let stats = store.enhanced_stats();
let slow_queries = store.slow_query_history();
```

**Q: High memory usage**
A: Reduce the buffer pool size in configuration:
```rust
let config = TdbConfig::new("./data/tdb")
    .with_buffer_pool_size(256); // fewer pages kept resident
```

### Debug Mode

Enable debug logging and run a deep diagnostic pass:

```rust
tracing_subscriber::fmt()
    .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
    .init();

let config = TdbConfig::new("./data/tdb").with_statistics(true);
let store = TdbStore::open_with_config(config)?;
let report = store.run_deep_diagnostics();
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Apache Jena TDB2 for reference implementation
- Oxigraph for RDF storage patterns
- The Rust community for excellent database libraries

---

For more information, see the [OxiRS project documentation](https://github.com/cool-japan/oxirs).