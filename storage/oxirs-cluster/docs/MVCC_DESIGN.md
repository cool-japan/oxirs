# MVCC Design for OxiRS Cluster

## Overview

The Multi-Version Concurrency Control (MVCC) system in OxiRS provides high-performance concurrent access to RDF data by maintaining multiple versions of each triple. This design allows readers to access consistent snapshots without blocking writers, significantly improving throughput in read-heavy workloads.

## Key Components

### 1. Hybrid Logical Clock (HLC)

The HLC combines physical time with logical counters to provide globally ordered timestamps across distributed nodes:

```rust
pub struct HLCTimestamp {
    physical: u64,      // Milliseconds since epoch
    logical: u64,       // Logical counter
    node_id: u64,       // Node identifier
}
```

**Features:**
- Monotonically increasing timestamps
- Handles clock skew between nodes
- Unique ordering across the cluster
- Efficient timestamp generation

### 2. Version Management

Each triple can have multiple versions, tracked with metadata:

```rust
pub struct Version {
    timestamp: HLCTimestamp,
    transaction_id: TransactionId,
    is_deleted: bool,
    data: Option<Triple>,
}
```

**Version Storage:**
- Versions stored in BTreeMap for efficient range queries
- Indexed by HLC timestamp for temporal ordering
- Supports both insertions and deletions (tombstones)

### 3. Transaction Snapshots

Transactions operate on consistent snapshots of the data:

```rust
pub struct TransactionSnapshot {
    transaction_id: TransactionId,
    timestamp: HLCTimestamp,
    isolation_level: IsolationLevel,
    read_set: HashSet<String>,
    write_set: HashSet<String>,
}
```

## Isolation Levels

The MVCC system supports all standard SQL isolation levels:

### Read Uncommitted
- Reads latest version including uncommitted changes
- No read locks required
- Highest performance, lowest consistency

### Read Committed (Default)
- Reads only committed versions
- Each query sees latest committed data
- Prevents dirty reads

### Repeatable Read
- All queries see data as of transaction start
- Prevents non-repeatable reads
- Uses snapshot timestamp for consistency

### Serializable
- Full serializability guarantee
- Detects read-write conflicts
- Uses optimistic concurrency control

## Conflict Detection

The system implements optimistic concurrency control with conflict detection at commit time:

1. **Write-Write Conflicts**: Detected when two transactions modify the same key
2. **Read-Write Conflicts**: Detected for serializable isolation when a read key is modified
3. **Phantom Prevention**: Serializable transactions track read patterns

## Version Storage Architecture

### Multi-Version Index

The `MVCCIndex` provides efficient version-aware lookups:

```rust
pub struct MVCCIndex {
    // IndexKey -> Versions -> Triple Keys
    primary_index: DashMap<IndexKey, BTreeMap<HLCTimestamp, HashSet<String>>>,
    // Triple Key -> Index Keys
    reverse_index: DashMap<String, HashSet<IndexKey>>,
}
```

**Index Types:**
- Subject index (s:subject)
- Predicate index (p:predicate)
- Object index (o:object)
- Composite indices (sp:, po:, so:)
- Full triple index (spo:)

### Storage Backend

The `MVCCStorage` integrates MVCC with persistent storage:

1. **Write Path:**
   - Generate new version with HLC timestamp
   - Update indices
   - Write to transaction log
   - Persist to disk on commit

2. **Read Path:**
   - Determine snapshot timestamp
   - Query indices for matching keys
   - Retrieve appropriate versions
   - Apply isolation level rules

## Garbage Collection

The system implements configurable garbage collection strategies:

### Time-Based Retention
```rust
CompactionStrategy::TimeBasedRetention(Duration)
```
- Removes versions older than specified duration
- Preserves recent history

### Version Count Limit
```rust
CompactionStrategy::KeepLatest(n)
```
- Keeps only N most recent versions per key
- Bounds storage usage

### Hybrid Strategy (Default)
```rust
CompactionStrategy::Hybrid {
    max_versions: 100,
    retention_period: Duration::from_secs(86400),
}
```
- Combines time and count limits
- Balances history preservation with storage efficiency

## Performance Optimizations

### 1. Lock-Free Reads
- Readers never block writers
- No read locks required for most isolation levels
- Concurrent readers scale linearly

### 2. Efficient Version Lookup
- BTreeMap provides O(log n) version lookups
- Range queries for temporal scans
- Cached latest versions for common case

### 3. Batched Operations
- Group multiple operations per transaction
- Amortize timestamp generation cost
- Reduce index update overhead

### 4. Lazy Deletion
- Tombstones mark deletions without immediate removal
- Garbage collection handles physical deletion
- Maintains read consistency

## Integration with 2PC

The MVCC system integrates seamlessly with the existing Two-Phase Commit protocol:

1. **Prepare Phase:**
   - Validate read/write sets
   - Check for conflicts
   - Acquire necessary locks

2. **Commit Phase:**
   - Finalize version timestamps
   - Update committed transaction log
   - Release resources

3. **Rollback:**
   - Remove uncommitted versions
   - Clear transaction state
   - No impact on other transactions

## Configuration

Key configuration parameters:

```rust
pub struct MVCCConfig {
    enable_snapshot_isolation: bool,      // Enable snapshot isolation
    gc_interval: Duration,                // GC run frequency
    gc_min_age: Duration,                 // Minimum version age for GC
    max_versions_per_key: usize,          // Version limit per key
    enable_conflict_detection: bool,      // Enable conflict detection
}
```

## Usage Example

```rust
// Start MVCC manager
let mvcc = MVCCManager::new(node_id, MVCCConfig::default());
mvcc.start().await?;

// Begin transaction
let tx_id = "tx123";
let snapshot = mvcc.begin_transaction(
    tx_id,
    IsolationLevel::RepeatableRead
).await?;

// Read with MVCC
let value = mvcc.read(tx_id, "key1").await?;

// Write with MVCC
mvcc.write(tx_id, "key2", Some(triple)).await?;

// Commit (with conflict detection)
mvcc.commit_transaction(tx_id).await?;
```

## Monitoring and Debugging

The system provides comprehensive statistics:

```rust
pub struct MVCCStatistics {
    total_keys: usize,
    total_versions: usize,
    max_versions_per_key: usize,
    active_transactions: usize,
    committed_transactions: usize,
}
```

Access version history for debugging:
```rust
let versions = mvcc.get_all_versions("key").await;
```

## Future Enhancements

1. **Adaptive Isolation:** Automatically adjust isolation levels based on conflict patterns
2. **Version Compression:** Compress older versions to reduce storage
3. **Distributed GC:** Coordinate garbage collection across cluster nodes
4. **Read Replicas:** Serve historical reads from dedicated replicas
5. **Time-Travel Queries:** Query data as of specific timestamps

## Conclusion

The MVCC implementation provides a robust foundation for concurrent access to RDF data in OxiRS. By maintaining multiple versions and using optimistic concurrency control, the system achieves high performance while maintaining strong consistency guarantees. The flexible configuration and multiple isolation levels allow users to optimize for their specific workload requirements.