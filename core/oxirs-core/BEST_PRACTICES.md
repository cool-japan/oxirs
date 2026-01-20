# OxiRS Core - Best Practices Guide

This guide provides best practices for using OxiRS Core in production environments, covering performance optimization, error handling, monitoring, and scalability.

## Table of Contents

1. [Performance Best Practices](#performance-best-practices)
2. [Error Handling](#error-handling)
3. [Monitoring and Observability](#monitoring-and-observability)
4. [Security](#security)
5. [Scalability Patterns](#scalability-patterns)
6. [Data Management](#data-management)
7. [Testing Strategies](#testing-strategies)

## Performance Best Practices

### 1. Choose the Right Storage Backend

```rust
// For small datasets (<1M triples) - Use in-memory store
let store = RdfStore::new();

// For medium datasets (1M-10M triples) - Use memory-mapped store
let store = MmapStore::open("data.db")?;

// For large datasets (>10M triples) - Use clustered store
let store = ClusteredStore::new(cluster_config)?;
```

### 2. Use Batch Operations

```rust
// ❌ BAD: Insert one-by-one
for triple in triples {
    store.insert_triple(&triple)?;  // Slow!
}

// ✅ GOOD: Use batch insert (50-100x faster)
store.insert_batch(&triples)?;
```

### 3. Enable Query Result Caching

```rust
use oxirs_core::cache::QueryResultCache;
use std::time::Duration;

let cache = QueryResultCache::new()
    .with_max_entries(10_000)
    .with_ttl(Duration::from_secs(300))  // 5 minutes
    .build();

// Cache frequently executed queries
let results = executor.execute_with_cache(query, &cache)?;

// Expected performance:
// - Cache hits: <1ms
// - Cache misses: original query time
// - Hit rate target: >80% for production workloads
```

### 4. Optimize SPARQL Queries

```rust
// ❌ BAD: Unbounded queries
let bad_query = r#"
    SELECT * WHERE { ?s ?p ?o }
"#;

// ✅ GOOD: Use LIMIT and specific patterns
let good_query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?name WHERE {
        ?person a foaf:Person .
        ?person foaf:name ?name .
    }
    LIMIT 1000
"#;
```

### 5. Use Proper Indexing

```rust
// Let the query optimizer choose indexes automatically
// But you can hint with specific patterns:

// For subject-focused queries
let results = store.quads_matching(
    Some(&subject),  // Specific subject
    None,           // Any predicate
    None,           // Any object
    None            // Any graph
)?;  // Uses SPO index

// For predicate-focused queries (like property search)
let results = store.quads_matching(
    None,                  // Any subject
    Some(&predicate),      // Specific predicate
    None,                  // Any object
    None                   // Any graph
)?;  // Uses POS index
```

### 6. Parallel Processing for Large Datasets

```rust
use oxirs_core::batch::BatchProcessor;

let processor = BatchProcessor::new()
    .with_batch_size(1000)
    .with_parallelism(num_cpus::get());  // Use all available cores

processor.insert_batch(&store, &large_triple_set)?;

// Performance: 3-8x speedup on bulk operations
```

### 7. Use Zero-Copy Parsing

```rust
use oxirs_core::parser::zero_copy::ZeroCopyParser;

// ❌ BAD: Regular parsing (allocates memory)
let triples = parse_ntriples(data)?;

// ✅ GOOD: Zero-copy parsing (60-80% less memory)
let parser = ZeroCopyParser::new();
let triples = parser.parse_ntriples_zero_copy(data)?;
```

## Error Handling

### 1. Use Result Types Properly

```rust
use oxirs_core::OxirsError;

fn process_rdf_data(path: &str) -> Result<(), OxirsError> {
    let store = RdfStore::new();

    // Use ? operator for error propagation
    let triples = parse_file(path, RdfFormat::Turtle)?;
    store.insert_batch(&triples)?;

    Ok(())
}
```

### 2. Handle Transaction Failures

```rust
use oxirs_core::transaction::TransactionError;

fn safe_transaction(store: &RdfStore, data: &[Triple]) -> Result<(), OxirsError> {
    let mut tx = store.begin_transaction(IsolationLevel::Snapshot)?;

    // Use match for specific error handling
    match tx.insert_batch(data) {
        Ok(_) => tx.commit(),
        Err(TransactionError::Conflict) => {
            // Retry on conflicts
            tx.rollback()?;
            // Implement retry logic here
            Err(OxirsError::Transaction("Conflict detected".into()))
        }
        Err(e) => {
            tx.rollback()?;
            Err(e.into())
        }
    }
}
```

### 3. Implement Retry Logic

```rust
use std::time::Duration;
use std::thread;

fn execute_with_retry<F, T>(
    mut f: F,
    max_retries: u32,
    backoff_ms: u64
) -> Result<T, OxirsError>
where
    F: FnMut() -> Result<T, OxirsError>,
{
    let mut attempts = 0;

    loop {
        match f() {
            Ok(result) => return Ok(result),
            Err(e) if attempts < max_retries && e.is_retryable() => {
                attempts += 1;
                thread::sleep(Duration::from_millis(backoff_ms * attempts as u64));
            }
            Err(e) => return Err(e),
        }
    }
}

// Usage
execute_with_retry(
    || store.insert_triple(&triple),
    max_retries: 3,
    backoff_ms: 100
)?;
```

## Monitoring and Observability

### 1. Enable Metrics Collection

```rust
use oxirs_core::metrics::MetricsCollector;

let metrics = MetricsCollector::new();

// Track query performance
metrics.record_query_execution(query_id, duration);

// Track cache performance
metrics.record_cache_hit();
metrics.record_cache_miss();

// Track storage metrics
metrics.record_storage_size(bytes);
metrics.record_triple_count(count);

// Export metrics (Prometheus format)
let prometheus_output = metrics.export_prometheus();
```

### 2. Query Profiling in Production

```rust
use oxirs_core::profiling::QueryProfiler;

let profiler = QueryProfiler::new()
    .with_sampling_rate(0.1)  // Profile 10% of queries
    .with_slow_query_threshold_ms(1000)  // Log queries >1s
    .build();

// Only slow queries and sampled queries are profiled
let results = executor.execute_with_profiling(query, &profiler)?;

// Get aggregated statistics
let stats = profiler.get_aggregated_stats()?;
println!("Avg query time: {}ms", stats.avg_execution_time_ms);
println!("P95 query time: {}ms", stats.p95_execution_time_ms);
println!("Slow queries: {}", stats.slow_query_count);
```

### 3. Health Checks

```rust
use oxirs_core::health::HealthChecker;

let health = HealthChecker::new();

// Register health checks
health.register_check("store_size", || {
    let size = store.size()?;
    if size < 1_000_000_000 {  // < 1GB
        Ok(HealthStatus::Healthy)
    } else {
        Ok(HealthStatus::Warning("Store size approaching limit".into()))
    }
});

health.register_check("query_latency", || {
    let latency = metrics.avg_query_latency_ms();
    if latency < 100.0 {
        Ok(HealthStatus::Healthy)
    } else {
        Ok(HealthStatus::Degraded("High query latency".into()))
    }
});

// Check overall health
let status = health.check_all()?;
```

### 4. Logging Best Practices

```rust
use log::{debug, info, warn, error};

// ✅ GOOD: Structured logging with context
info!(
    "Query executed successfully";
    "query_id" => query_id,
    "execution_time_ms" => duration.as_millis(),
    "triples_matched" => count
);

// ❌ BAD: Unstructured logging
println!("Query done in {}ms", duration.as_millis());
```

## Security

### 1. Input Validation

```rust
use oxirs_core::validation::validate_sparql;

fn execute_user_query(query: &str) -> Result<QueryResults, OxirsError> {
    // Validate query syntax and check for injection attacks
    validate_sparql(query)?;

    // Limit query complexity
    let complexity = analyze_query_complexity(query)?;
    if complexity.depth > 10 || complexity.joins > 100 {
        return Err(OxirsError::Query("Query too complex".into()));
    }

    executor.execute(query)
}
```

### 2. Resource Limits

```rust
use oxirs_core::limits::ResourceLimits;

let limits = ResourceLimits::new()
    .with_max_query_time(Duration::from_secs(30))
    .with_max_memory_per_query(100_000_000)  // 100MB
    .with_max_results(10_000)
    .build();

let executor = QueryExecutor::new(store)
    .with_limits(limits);
```

### 3. Access Control

```rust
use oxirs_core::security::{AccessControl, Permission};

let acl = AccessControl::new();

// Define permissions
acl.grant_permission(user_id, Permission::Read)?;
acl.grant_permission(admin_id, Permission::Write)?;
acl.grant_permission(admin_id, Permission::Admin)?;

// Check permissions before operations
fn insert_triple(
    store: &RdfStore,
    triple: &Triple,
    user_id: &str,
    acl: &AccessControl
) -> Result<(), OxirsError> {
    if !acl.has_permission(user_id, Permission::Write)? {
        return Err(OxirsError::Unauthorized);
    }

    store.insert_triple(triple)
}
```

## Scalability Patterns

### 1. Horizontal Scaling with Clustering

```rust
use oxirs_core::cluster::{ClusterConfig, ClusterStore};

let config = ClusterConfig {
    nodes: vec![
        "node1.example.com:5000",
        "node2.example.com:5000",
        "node3.example.com:5000",
    ],
    replication_factor: 3,
    consistency_level: ConsistencyLevel::Quorum,
};

let store = ClusterStore::new(config)?;

// Data automatically distributed across nodes
// Queries automatically parallelized
```

### 2. Read Replicas

```rust
use oxirs_core::replication::{PrimaryStore, ReplicaStore};

// Primary store (for writes)
let primary = PrimaryStore::new("primary.db")?;

// Read replicas (for read scaling)
let replica1 = ReplicaStore::connect(&primary)?;
let replica2 = ReplicaStore::connect(&primary)?;

// Route reads to replicas, writes to primary
fn handle_query(query: &str, is_write: bool) -> Result<QueryResults, OxirsError> {
    if is_write {
        primary.execute(query)
    } else {
        // Load balance across replicas
        let replica = select_replica(&[replica1, replica2]);
        replica.execute(query)
    }
}
```

### 3. Caching Strategies

```rust
// Multi-level caching
use oxirs_core::cache::{L1Cache, L2Cache, CacheHierarchy};

let cache = CacheHierarchy::new()
    .with_l1(L1Cache::new(1000))      // In-memory, hot queries
    .with_l2(L2Cache::new(10_000))    // Larger, warm queries
    .with_ttl(Duration::from_secs(300))
    .build();

// Cache automatically manages eviction and promotion
```

## Data Management

### 1. Regular Backups

```rust
use oxirs_core::backup::{BackupManager, BackupType};

let backup_mgr = BackupManager::new();

// Full backup
backup_mgr.create_backup(
    &store,
    "backup/full_20251225.db",
    BackupType::Full
)?;

// Incremental backup (only changes since last backup)
backup_mgr.create_backup(
    &store,
    "backup/incr_20251225_1200.db",
    BackupType::Incremental
)?;

// Restore from backup chain
backup_mgr.restore(
    &store,
    &["backup/full_20251220.db", "backup/incr_20251225_1200.db"]
)?;
```

### 2. Data Compaction

```rust
// Compact store to remove deleted triples and reclaim space
store.compact()?;

// Schedule regular compaction
use std::time::Duration;

let compaction_interval = Duration::from_secs(3600);  // Every hour
schedule_periodic_task(compaction_interval, || {
    if store.deleted_ratio() > 0.2 {  // >20% deleted
        store.compact()?;
    }
    Ok(())
});
```

### 3. Data Versioning

```rust
use oxirs_core::versioning::VersionedStore;

let store = VersionedStore::new();

// Create snapshots
let snapshot_id = store.create_snapshot("v1.0")?;

// Make changes
store.insert_triple(&triple)?;

// Revert to snapshot if needed
store.restore_snapshot(snapshot_id)?;
```

## Testing Strategies

### 1. Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_insertion() {
        let store = RdfStore::new();
        let triple = create_test_triple();

        assert!(store.insert_triple(&triple).is_ok());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_query_execution() {
        let store = setup_test_store();
        let executor = QueryExecutor::new(store);

        let results = executor.execute("SELECT * WHERE { ?s ?p ?o } LIMIT 10")
            .expect("Query should succeed");

        assert!(results.len() <= 10);
    }
}
```

### 2. Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_full_workflow() {
        // Setup
        let store = RdfStore::new();
        let executor = QueryExecutor::new(store.clone());

        // Load data
        let triples = load_test_data("test_data.ttl")?;
        store.insert_batch(&triples)?;

        // Execute query
        let results = executor.execute(TEST_QUERY)?;

        // Verify results
        assert_eq!(results.len(), EXPECTED_COUNT);
    }
}
```

### 3. Performance Testing

```rust
#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn benchmark_insert(c: &mut Criterion) {
        let store = RdfStore::new();
        let triples = generate_test_triples(1000);

        c.bench_function("insert_1000_triples", |b| {
            b.iter(|| {
                store.insert_batch(black_box(&triples))
            });
        });
    }

    criterion_group!(benches, benchmark_insert);
    criterion_main!(benches);
}
```

### 4. Property-Based Testing

```rust
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_triple_roundtrip(
            subject in any::<String>(),
            predicate in any::<String>(),
            object in any::<String>()
        ) {
            let store = RdfStore::new();
            let triple = create_triple(&subject, &predicate, &object);

            store.insert_triple(&triple)?;
            let retrieved = store.get_triple(&triple)?;

            prop_assert_eq!(triple, retrieved);
        }
    }
}
```

## Production Checklist

Before deploying to production, ensure:

- [ ] **Performance**
  - [ ] Query result caching enabled
  - [ ] Batch operations used for bulk data
  - [ ] Appropriate storage backend selected
  - [ ] Query profiling configured

- [ ] **Reliability**
  - [ ] Transactions enabled with appropriate isolation
  - [ ] WAL configured for durability
  - [ ] Backup strategy implemented
  - [ ] Error handling and retry logic in place

- [ ] **Monitoring**
  - [ ] Metrics collection enabled
  - [ ] Health checks configured
  - [ ] Logging properly structured
  - [ ] Alerts set up for critical issues

- [ ] **Security**
  - [ ] Input validation implemented
  - [ ] Resource limits configured
  - [ ] Access control in place
  - [ ] Query complexity limits set

- [ ] **Scalability**
  - [ ] Capacity planning completed
  - [ ] Horizontal scaling strategy defined
  - [ ] Caching strategy implemented
  - [ ] Read replicas configured if needed

- [ ] **Testing**
  - [ ] Unit tests passing
  - [ ] Integration tests passing
  - [ ] Performance benchmarks completed
  - [ ] Load testing performed

---

Following these best practices will help ensure your OxiRS Core deployment is performant, reliable, and scalable in production environments.

For more information, see:
- [Tutorial](TUTORIAL.md) - Getting started guide
- [Architecture](ARCHITECTURE.md) - Deep dive into internals
- [Performance Guide](PERFORMANCE_GUIDE.md) - Detailed optimization strategies
- [Deployment Handbook](DEPLOYMENT.md) - Production deployment guide
