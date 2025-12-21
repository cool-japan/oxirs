# OxiRS Vec - Write-Ahead Logging (WAL) Configuration Guide

**Version**: v0.1.0-beta.2
**Last Updated**: December 6, 2025

## Table of Contents

1. [Introduction](#introduction)
2. [WAL Architecture](#wal-architecture)
3. [Configuration](#configuration)
4. [Crash Recovery](#crash-recovery)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring](#monitoring)
7. [Maintenance](#maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Introduction

Write-Ahead Logging (WAL) provides crash recovery and durability guarantees for OxiRS Vec. This guide covers configuration, operation, and best practices for production use.

### What is WAL?

WAL ensures data durability by writing all changes to a log file **before** applying them to the index. In case of a crash, the system can replay the log to restore state.

### Benefits

- **Crash Recovery**: Automatic recovery from unexpected shutdowns
- **Data Durability**: No data loss on crashes
- **Transaction Support**: ACID-compliant operations
- **Performance**: Minimal overhead (1-5% latency increase)

### When to Use WAL

✅ **Use WAL when**:
- Data loss is unacceptable
- Compliance requires audit logs
- Running in production environments
- Index rebuilding is expensive

❌ **Skip WAL when**:
- Development/testing environments
- Read-only workloads
- Index can be quickly rebuilt from source
- Extreme performance requirements

---

## WAL Architecture

### Components

```
┌─────────────────────────────────────────────┐
│              Application                    │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│           WAL Manager                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Write   │  │Checkpoint│  │ Recovery │  │
│  │  Path    │  │  Manager │  │  Engine  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│            WAL File(s)                      │
│  ┌──────────────────────────────────────┐   │
│  │ [Header][Entry1][Entry2]...[EntryN] │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Write Path

1. **Operation Received**: Insert/Update/Delete
2. **Write to WAL**: Append operation to log file
3. **Sync to Disk**: fsync() for durability (optional)
4. **Apply to Index**: Update in-memory index
5. **Return Success**: Acknowledge to client

### Recovery Path

1. **Detect Crash**: Check for incomplete WAL
2. **Read WAL**: Parse all log entries
3. **Replay Operations**: Apply operations to index
4. **Verify Checksums**: Detect corrupted entries
5. **Complete Recovery**: Resume normal operation

---

## Configuration

### Basic Configuration

```rust
use oxirs_vec::wal::{WalConfig, WalManager};

fn setup_basic_wal() -> anyhow::Result<WalManager> {
    let config = WalConfig {
        wal_dir: "/var/lib/oxirs/wal".to_string(),
        max_wal_size: 1_000_000_000,  // 1 GB
        sync_interval_ms: 1000,        // Sync every 1 second
        checkpoint_interval: 10_000,   // Checkpoint every 10k operations
    };

    WalManager::new(config)
}
```

### Production Configuration

```rust
fn setup_production_wal() -> anyhow::Result<WalManager> {
    let config = WalConfig {
        wal_dir: "/mnt/ssd/oxirs/wal".to_string(),  // Fast SSD
        max_wal_size: 10_000_000_000,               // 10 GB
        sync_interval_ms: 100,                       // Sync every 100ms
        checkpoint_interval: 100_000,                // Checkpoint every 100k ops
        enable_compression: true,                    // Compress old WAL segments
        rotation_policy: RotationPolicy::SizeBased,  // Rotate at max size
        retention_policy: RetentionPolicy::KeepLast(7), // Keep last 7 checkpoints
    };

    WalManager::new(config)
}
```

### Configuration Parameters

| Parameter | Default | Description | Tuning Guide |
|-----------|---------|-------------|--------------|
| `wal_dir` | `/tmp/wal` | WAL directory | Use fast SSD |
| `max_wal_size` | 1 GB | Max WAL file size | 1-10 GB typical |
| `sync_interval_ms` | 1000 ms | Sync frequency | Lower = more durable, slower |
| `checkpoint_interval` | 10,000 | Operations between checkpoints | Higher = less I/O, slower recovery |
| `enable_compression` | false | Compress WAL | Enable for disk savings |

### Durability vs Performance Trade-offs

#### Maximum Durability (Safest)

```rust
let config = WalConfig {
    sync_interval_ms: 0,  // fsync after every write
    checkpoint_interval: 1000,
    ..Default::default()
};

// Characteristics:
// - No data loss on crash
// - 20-30% performance overhead
// - Use for: Financial, healthcare, critical data
```

#### Balanced (Recommended)

```rust
let config = WalConfig {
    sync_interval_ms: 100,  // fsync every 100ms
    checkpoint_interval: 10_000,
    ..Default::default()
};

// Characteristics:
// - Max 100ms data loss window
// - 5-10% performance overhead
// - Use for: Most production workloads
```

#### Performance-Optimized

```rust
let config = WalConfig {
    sync_interval_ms: 5000,  // fsync every 5 seconds
    checkpoint_interval: 100_000,
    ..Default::default()
};

// Characteristics:
// - Max 5 seconds data loss window
// - 1-2% performance overhead
// - Use for: High-throughput, less critical data
```

---

## Crash Recovery

### Automatic Recovery

```rust
use oxirs_vec::crash_recovery::{CrashRecoveryManager, RecoveryConfig};

fn recover_from_crash() -> anyhow::Result<VectorStore> {
    let recovery_config = RecoveryConfig {
        wal_dir: "/var/lib/oxirs/wal".to_string(),
        data_dir: "/var/lib/oxirs/data".to_string(),
        verify_checksums: true,      // Detect corruption
        max_recovery_time_ms: 60000, // Fail if recovery > 1 min
    };

    let manager = CrashRecoveryManager::new(recovery_config)?;

    // Automatic recovery on startup
    let store = manager.recover()?;

    println!("Recovery complete. Index restored.");
    Ok(store)
}
```

### Recovery Process

```
1. Check for WAL files
   │
   ├─ WAL exists → Continue to step 2
   └─ No WAL → Normal startup
   │
2. Read WAL header
   │
   ├─ Valid header → Continue to step 3
   └─ Corrupted → Fail recovery
   │
3. Replay operations
   │
   ├─ Read entry
   ├─ Verify checksum
   ├─ Apply operation
   └─ Repeat until end of WAL
   │
4. Verify consistency
   │
5. Checkpoint and cleanup
   │
6. Resume normal operation
```

### Manual Recovery

```rust
use oxirs_vec::wal::{WalManager, WalEntry};

fn manual_recovery(wal_path: &str) -> anyhow::Result<()> {
    let wal_manager = WalManager::new_from_path(wal_path)?;

    // Read all entries
    let entries = wal_manager.read_all_entries()?;

    println!("Found {} WAL entries", entries.len());

    // Inspect entries
    for (i, entry) in entries.iter().enumerate() {
        match entry {
            WalEntry::Insert { id, vector, .. } => {
                println!("[{}] INSERT id={}", i, id);
            }
            WalEntry::Update { id, vector, .. } => {
                println!("[{}] UPDATE id={}", i, id);
            }
            WalEntry::Delete { id, .. } => {
                println!("[{}] DELETE id={}", i, id);
            }
            WalEntry::Checkpoint { seq, .. } => {
                println!("[{}] CHECKPOINT seq={}", i, seq);
            }
        }
    }

    // Apply entries to index
    let mut store = VectorStore::new();
    for entry in entries {
        wal_manager.apply_entry(&mut store, entry)?;
    }

    Ok(())
}
```

### Recovery Statistics

```rust
fn display_recovery_stats(manager: &CrashRecoveryManager) {
    let stats = manager.get_recovery_stats();

    println!("Recovery Statistics:");
    println!("  Total entries: {}", stats.total_entries);
    println!("  Applied: {}", stats.applied_entries);
    println!("  Skipped: {}", stats.skipped_entries);
    println!("  Corrupted: {}", stats.corrupted_entries);
    println!("  Recovery time: {} ms", stats.recovery_time_ms);
}
```

---

## Performance Tuning

### Write Performance

#### Batch Operations

```rust
use oxirs_vec::wal::WalTransaction;

// ❌ DON'T: Write individually (slow)
for vector in vectors {
    wal_manager.log_insert(&vector)?;
    wal_manager.sync()?; // fsync per operation!
}

// ✅ DO: Batch writes
let mut transaction = wal_manager.begin_transaction()?;
for vector in vectors {
    transaction.log_insert(&vector)?;
}
transaction.commit()?; // Single fsync
```

#### Async Syncing

```rust
use oxirs_vec::wal::AsyncWalManager;

// Use async WAL for better throughput
let async_wal = AsyncWalManager::new(config)?;

// Write returns immediately, sync happens in background
async_wal.log_insert_async(&vector).await?;

// Flush when needed (e.g., before checkpoint)
async_wal.flush().await?;
```

### Read Performance

WAL has minimal impact on read performance (queries don't access WAL).

### Recovery Performance

```rust
// Faster recovery with larger checkpoint intervals
let config = WalConfig {
    checkpoint_interval: 100_000,  // Fewer checkpoints
    ..Default::default()
};

// Trade-off:
// - Longer checkpoint intervals = faster writes
// - But slower recovery (more entries to replay)
```

### Disk I/O Optimization

```rust
// Use dedicated SSD for WAL
let config = WalConfig {
    wal_dir: "/mnt/nvme-ssd/wal".to_string(),  // NVMe SSD
    ..Default::default()
};

// Separate WAL and data on different disks
// - WAL: Sequential writes (SSD)
// - Data: Random access (can use HDD)
```

---

## Monitoring

### WAL Metrics

```rust
use oxirs_vec::wal::WalMetrics;

fn monitor_wal(wal_manager: &WalManager) {
    let metrics = wal_manager.get_metrics();

    println!("WAL Metrics:");
    println!("  Current size: {} MB", metrics.current_size_mb);
    println!("  Total entries: {}", metrics.total_entries);
    println!("  Write throughput: {} ops/s", metrics.write_ops_per_sec);
    println!("  Sync latency: {} ms", metrics.avg_sync_latency_ms);
    println!("  Last checkpoint: {} ago", metrics.last_checkpoint_age);
}
```

### Key Metrics to Track

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| WAL size | Current WAL file size | > 80% of max_wal_size |
| Write latency | Time to write + sync | > 10 ms |
| Sync latency | Time for fsync | > 50 ms |
| Entries per second | Write throughput | Baseline ± 50% |
| Checkpoint age | Time since last checkpoint | > 1 hour |

### Health Checks

```rust
fn wal_health_check(wal_manager: &WalManager) -> bool {
    let metrics = wal_manager.get_metrics();

    // Check 1: WAL not too large
    if metrics.current_size_mb > 0.9 * metrics.max_size_mb {
        eprintln!("WAL size critical: {} MB", metrics.current_size_mb);
        return false;
    }

    // Check 2: Sync latency acceptable
    if metrics.avg_sync_latency_ms > 100.0 {
        eprintln!("High sync latency: {} ms", metrics.avg_sync_latency_ms);
        return false;
    }

    // Check 3: Recent checkpoint
    if metrics.last_checkpoint_age.as_secs() > 3600 {
        eprintln!("Stale checkpoint: {} seconds old",
                  metrics.last_checkpoint_age.as_secs());
        return false;
    }

    true
}
```

---

## Maintenance

### Manual Checkpoint

```rust
use oxirs_vec::wal::WalManager;

fn manual_checkpoint(wal_manager: &mut WalManager) -> anyhow::Result<()> {
    println!("Starting manual checkpoint...");

    let start = std::time::Instant::now();
    wal_manager.checkpoint()?;
    let elapsed = start.elapsed();

    println!("Checkpoint completed in {} ms", elapsed.as_millis());
    Ok(())
}
```

### WAL Rotation

```rust
// Automatic rotation when WAL reaches max size
let config = WalConfig {
    max_wal_size: 1_000_000_000,  // 1 GB
    rotation_policy: RotationPolicy::SizeBased,
    ..Default::default()
};

// Or time-based rotation
let config = WalConfig {
    rotation_policy: RotationPolicy::TimeBased {
        interval_hours: 24,  // Daily rotation
    },
    ..Default::default()
};
```

### WAL Cleanup

```rust
use oxirs_vec::wal::WalCleaner;

fn cleanup_old_wal_files() -> anyhow::Result<()> {
    let cleaner = WalCleaner::new("/var/lib/oxirs/wal")?;

    // Remove WAL files older than 7 days
    let removed = cleaner.cleanup_old_files(7)?;

    println!("Removed {} old WAL files", removed);
    Ok(())
}
```

### Compression

```rust
// Enable compression for old WAL segments
let config = WalConfig {
    enable_compression: true,
    compression_threshold_age_hours: 24,  // Compress after 24 hours
    ..Default::default()
};

// Compressed WAL segments are read-only (for recovery only)
```

---

## Troubleshooting

### Issue 1: High Write Latency

**Symptoms**: Writes taking > 10 ms

**Diagnosis**:
```rust
let metrics = wal_manager.get_metrics();
println!("Write latency: {} ms", metrics.avg_write_latency_ms);
println!("Sync latency: {} ms", metrics.avg_sync_latency_ms);
```

**Solutions**:
1. Reduce sync frequency: `sync_interval_ms: 1000`
2. Use faster disk (NVMe SSD)
3. Enable batching: Use transactions
4. Disable compression during writes

### Issue 2: WAL Growing Too Large

**Symptoms**: WAL file > max_wal_size

**Diagnosis**:
```rust
let metrics = wal_manager.get_metrics();
println!("WAL size: {} MB (max: {} MB)",
         metrics.current_size_mb,
         metrics.max_size_mb);
```

**Solutions**:
1. Trigger manual checkpoint
2. Increase `max_wal_size`
3. Reduce `checkpoint_interval`
4. Enable WAL compression

### Issue 3: Recovery Failure

**Symptoms**: Recovery fails with corruption error

**Diagnosis**:
```rust
let entries = wal_manager.verify_wal()?;
for (i, entry) in entries.iter().enumerate() {
    if !entry.checksum_valid {
        eprintln!("Corrupted entry at position {}", i);
    }
}
```

**Solutions**:
1. Check disk health (SMART status)
2. Restore from backup
3. Skip corrupted entries (data loss):
   ```rust
   let config = RecoveryConfig {
       skip_corrupted_entries: true,  // Dangerous!
       ..Default::default()
   };
   ```

### Issue 4: Slow Recovery

**Symptoms**: Recovery taking > 1 minute

**Diagnosis**:
```rust
let stats = recovery_manager.get_recovery_stats();
println!("Entries to replay: {}", stats.total_entries);
println!("Recovery time: {} ms", stats.recovery_time_ms);
```

**Solutions**:
1. Reduce `checkpoint_interval` (fewer entries to replay)
2. Enable parallel recovery
3. Use faster disk for WAL
4. Consider shorter retention policy

---

## Advanced Topics

### Two-Phase Commit

```rust
use oxirs_vec::wal::TwoPhaseCommit;

fn distributed_transaction() -> anyhow::Result<()> {
    let mut tpc = TwoPhaseCommit::new()?;

    // Phase 1: Prepare
    tpc.prepare(vec![
        Operation::Insert { id: "1", vector: vec1 },
        Operation::Insert { id: "2", vector: vec2 },
    ])?;

    // Phase 2: Commit (or rollback on failure)
    tpc.commit()?;

    Ok(())
}
```

### Point-in-Time Recovery

```rust
use chrono::{DateTime, Utc};

fn recover_to_point_in_time(
    wal_manager: &WalManager,
    target_time: DateTime<Utc>
) -> anyhow::Result<VectorStore> {
    let entries = wal_manager.read_all_entries()?;

    let mut store = VectorStore::new();
    for entry in entries {
        // Only apply entries before target time
        if entry.timestamp <= target_time {
            wal_manager.apply_entry(&mut store, entry)?;
        }
    }

    Ok(store)
}
```

### WAL Replication

```rust
use oxirs_vec::wal::WalReplicator;

fn setup_wal_replication() -> anyhow::Result<()> {
    let replicator = WalReplicator::new()?;

    // Replicate WAL to standby nodes
    replicator.add_standby("standby-1:5432")?;
    replicator.add_standby("standby-2:5432")?;

    // Automatic failover on primary failure
    replicator.enable_auto_failover()?;

    Ok(())
}
```

### Custom WAL Entries

```rust
use oxirs_vec::wal::{WalEntry, CustomEntry};

// Extend WAL with custom operations
#[derive(serde::Serialize, serde::Deserialize)]
struct CustomOperation {
    operation_type: String,
    payload: Vec<u8>,
}

fn log_custom_operation(
    wal_manager: &mut WalManager,
    op: CustomOperation
) -> anyhow::Result<()> {
    let entry = WalEntry::Custom(CustomEntry {
        operation: op,
        timestamp: chrono::Utc::now(),
    });

    wal_manager.log_entry(entry)?;
    Ok(())
}
```

---

## Configuration Examples

### Development

```rust
// Fast, no durability guarantees
let config = WalConfig {
    wal_dir: "/tmp/wal".to_string(),
    sync_interval_ms: 10000,  // 10 seconds
    checkpoint_interval: 100_000,
    enable_compression: false,
};
```

### Production - Balanced

```rust
// Good balance of durability and performance
let config = WalConfig {
    wal_dir: "/mnt/ssd/wal".to_string(),
    max_wal_size: 10_000_000_000,  // 10 GB
    sync_interval_ms: 100,          // 100 ms
    checkpoint_interval: 50_000,
    enable_compression: true,
    rotation_policy: RotationPolicy::SizeBased,
    retention_policy: RetentionPolicy::KeepLast(7),
};
```

### Production - Maximum Durability

```rust
// Critical data, no data loss acceptable
let config = WalConfig {
    wal_dir: "/mnt/nvme/wal".to_string(),
    max_wal_size: 50_000_000_000,  // 50 GB
    sync_interval_ms: 0,            // Immediate fsync
    checkpoint_interval: 10_000,
    enable_compression: true,
    rotation_policy: RotationPolicy::SizeBased,
    retention_policy: RetentionPolicy::KeepAll,  // Never delete
    enable_replication: true,
};
```

---

## Best Practices Summary

✅ **DO**:
- Use fast SSD for WAL directory
- Enable compression for old WAL segments
- Monitor WAL size and checkpoint age
- Test recovery regularly
- Batch operations when possible
- Set appropriate sync intervals
- Enable automatic checkpoints

❌ **DON'T**:
- Store WAL on slow disks (HDD)
- Skip fsync in production
- Ignore WAL size limits
- Delete WAL files manually
- Use WAL for read-heavy workloads
- Set sync_interval_ms to 0 unless necessary

---

## Quick Reference

| Scenario | sync_interval_ms | checkpoint_interval | Expected Impact |
|----------|------------------|---------------------|-----------------|
| Maximum Durability | 0 | 1,000 | 20-30% slower writes |
| Production | 100 | 10,000 | 5-10% slower writes |
| High Throughput | 1,000 | 100,000 | 1-2% slower writes |
| Development | 10,000 | 100,000 | Negligible impact |

---

**Document Version**: 1.0
**OxiRS Vec Version**: v0.1.0-beta.2
**Last Updated**: December 6, 2025
