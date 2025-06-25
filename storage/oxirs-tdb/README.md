# OxiRS TDB - High-Performance RDF Storage Engine

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance, ACID-compliant RDF storage engine with multi-version concurrency control (MVCC) and advanced transaction support. OxiRS TDB provides TDB2-equivalent functionality with modern Rust performance optimizations.

## Features

### Core Storage Engine
- **MVCC (Multi-Version Concurrency Control)**: Snapshot isolation with conflict detection
- **ACID Transactions**: Full transaction support with rollback capabilities
- **B+ Tree Indexing**: Six standard RDF indices (SPO, POS, OSP, SOP, PSO, OPS)
- **Advanced Page Management**: LRU buffer pools with efficient memory management
- **Crash Recovery**: ARIES-style write-ahead logging with analysis/redo/undo phases

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
oxirs-tdb = "0.1.0"
```

### Basic Usage

```rust
use oxirs_tdb::{TdbStore, TdbConfig};
use oxirs_core::{Triple, Quad, Term};

// Create a new TDB store
let config = TdbConfig::new()
    .with_directory("./data/tdb")
    .with_cache_size(1024 * 1024 * 1024) // 1GB cache
    .with_sync_mode(true);

let mut store = TdbStore::new(config)?;

// Start a transaction
let mut txn = store.begin_transaction()?;

// Insert triples
let subject = Term::iri("http://example.org/subject")?;
let predicate = Term::iri("http://example.org/predicate")?;
let object = Term::literal("Hello, World!")?;

let triple = Triple::new(subject, predicate, object);
txn.insert_triple(&triple)?;

// Commit transaction
txn.commit()?;

// Query data
let results = store.query_pattern(
    Some(&subject),
    Some(&predicate),
    None
)?;

for triple in results {
    println!("{}", triple);
}
```

### Advanced Usage

```rust
use oxirs_tdb::{TdbStore, TdbConfig, TransactionOptions};

// Configure with advanced options
let config = TdbConfig::new()
    .with_directory("./data/tdb")
    .with_cache_size(2 * 1024 * 1024 * 1024) // 2GB cache
    .with_page_size(8192) // 8KB pages
    .with_wal_enabled(true)
    .with_checkpoint_interval(Duration::from_secs(300))
    .with_mvcc_enabled(true);

let store = TdbStore::new(config)?;

// Use transactions with options
let txn_options = TransactionOptions::new()
    .with_isolation_level(IsolationLevel::Snapshot)
    .with_timeout(Duration::from_secs(30));

let mut txn = store.begin_transaction_with_options(txn_options)?;

// Bulk insert
let triples = vec![
    Triple::new(/* ... */),
    Triple::new(/* ... */),
    // ... more triples
];

txn.insert_triples_batch(&triples)?;
txn.commit()?;
```

## Configuration

### TdbConfig Options

```rust
let config = TdbConfig::new()
    // Storage location
    .with_directory("./data/tdb")
    
    // Memory management
    .with_cache_size(1_073_741_824) // 1GB
    .with_page_size(8192)           // 8KB pages
    
    // Transaction settings
    .with_mvcc_enabled(true)
    .with_wal_enabled(true)
    .with_checkpoint_interval(Duration::from_secs(300))
    
    // Performance tuning
    .with_sync_mode(true)
    .with_compression_enabled(true)
    .with_statistics_enabled(true)
    
    // Concurrency
    .with_max_concurrent_transactions(1000)
    .with_deadlock_detection_enabled(true);
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
# Run benchmarks
cargo bench

# Profile with specific datasets
cargo run --release --bin tdb-benchmark -- --dataset large --queries complex
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
│   ├── lib.rs              # Public API
│   ├── assembler.rs        # Low-level operations
│   ├── nodes.rs            # Node table implementation
│   ├── page.rs             # Page management
│   ├── triple_store.rs     # Triple storage engine
│   └── wal/                # Write-ahead logging
│       ├── mod.rs
│       ├── recovery.rs
│       └── log.rs
├── tests/                  # Integration tests
├── benches/               # Performance benchmarks
├── examples/              # Usage examples
└── data/                  # Test datasets
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

### Migration from TDB2

```rust
// Convert TDB2 database to OxiRS TDB
use oxirs_tdb::migration::tdb2_converter;

let converter = tdb2_converter::Tdb2Converter::new()
    .with_source_directory("./jena-tdb2-data")
    .with_target_directory("./oxirs-tdb-data");

converter.convert()?;
```

## Troubleshooting

### Common Issues

**Q: Database corruption after crash**
A: Run the recovery tool:
```bash
cargo run --bin tdb-recovery -- --database ./data/tdb --verify
```

**Q: Poor query performance**
A: Check statistics and indices:
```bash
cargo run --bin tdb-analyze -- --database ./data/tdb --verbose
```

**Q: High memory usage**
A: Adjust cache size in configuration:
```rust
let config = TdbConfig::new()
    .with_cache_size(512 * 1024 * 1024) // Reduce to 512MB
    .with_page_size(4096);               // Smaller pages
```

### Debug Mode

Enable debug logging:

```rust
env_logger::init();
let config = TdbConfig::new()
    .with_debug_mode(true)
    .with_statistics_enabled(true);
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Apache Jena TDB2 for reference implementation
- Oxigraph for RDF storage patterns
- The Rust community for excellent database libraries

---

For more information, see the [OxiRS project documentation](https://github.com/cool-japan/oxirs).