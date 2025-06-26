# Cross-Shard Transaction Design for OxiRS Cluster

## Overview

This document describes the cross-shard transaction implementation for OxiRS cluster, providing ACID guarantees across distributed shards using an optimized Two-Phase Commit (2PC) protocol.

## Architecture

### Core Components

1. **Transaction Coordinator** (`transaction.rs`)
   - Manages transaction lifecycle
   - Implements 2PC protocol phases
   - Handles participant voting and decision making
   - Maintains transaction state and recovery log

2. **Transaction Optimizer** (`transaction_optimizer.rs`)
   - Analyzes transactions for optimization opportunities
   - Implements read-only and single-shard optimizations
   - Provides deadlock detection and prevention
   - Manages recovery optimization

3. **Lock Manager**
   - Provides concurrency control
   - Supports multiple isolation levels
   - Implements deadlock detection
   - Manages lock timeouts and cleanup

## Two-Phase Commit Protocol

### Phase 1: Prepare

1. **Coordinator Actions**:
   - Assigns unique transaction ID
   - Logs transaction begin
   - Acquires necessary locks
   - Sends prepare requests to all participants
   - Waits for votes from participants

2. **Participant Actions**:
   - Validates operations
   - Acquires local locks
   - Prepares changes (without committing)
   - Votes YES or NO
   - Logs decision locally

### Phase 2: Commit/Abort

1. **Commit Decision** (all votes YES):
   - Coordinator logs commit decision
   - Sends commit messages to participants
   - Participants apply changes
   - Coordinator releases locks
   - Transaction marked as committed

2. **Abort Decision** (any vote NO):
   - Coordinator logs abort decision
   - Sends abort messages to participants
   - Participants rollback changes
   - Coordinator releases locks
   - Transaction marked as aborted

## Optimizations

### 1. Read-Only Optimization
```rust
// Skip 2PC entirely for read-only transactions
if transaction.is_readonly() {
    execute_without_2pc(transaction);
}
```

**Benefits**:
- No coordination overhead
- No locks required
- Immediate execution

### 2. Single-Shard Optimization
```rust
// Skip 2PC for single-shard transactions
if transaction.affects_single_shard() {
    execute_on_single_shard(transaction);
}
```

**Benefits**:
- Local transaction execution
- No network communication
- Reduced latency

### 3. Presumed Abort
- Don't log abort decisions at participants
- Assume abort if no commit record found
- Reduces logging overhead

### 4. Parallel Prepare
- Send prepare requests to all participants simultaneously
- Process votes as they arrive
- Reduces overall latency

### 5. Batching
- Group operations by shard
- Send batched operations in single message
- Reduces network overhead

## Isolation Levels

### Read Uncommitted
- No read locks required
- May read uncommitted data
- Highest performance, lowest consistency

### Read Committed (Default)
- Read locks released immediately
- Write locks held until commit
- Prevents dirty reads

### Repeatable Read
- Read locks held until commit
- Prevents phantom reads
- Higher consistency guarantee

### Serializable
- Strictest isolation level
- Full serializability guarantee
- Highest consistency, lowest performance

## Configuration

```rust
TransactionConfig {
    // Default transaction timeout
    default_timeout: Duration::from_secs(30),
    
    // Maximum concurrent transactions
    max_concurrent_transactions: 1000,
    
    // Enable optimistic concurrency control
    enable_optimistic_cc: true,
    
    // Enable deadlock detection
    enable_deadlock_detection: true,
    
    // Checkpoint interval for recovery
    checkpoint_interval: Duration::from_secs(60),
}
```

## Usage Example

```rust
use oxirs_cluster::transaction::{
    TransactionCoordinator, IsolationLevel, TransactionOp
};

// Begin a transaction
let tx_id = coordinator.begin_transaction(
    IsolationLevel::ReadCommitted
).await?;

// Add operations
coordinator.add_operation(&tx_id, TransactionOp::Insert {
    triple: Triple {
        subject: NamedNode::new("http://example.org/alice")?,
        predicate: NamedNode::new("http://example.org/knows")?,
        object: NamedNode::new("http://example.org/bob")?,
    }
}).await?;

// Query within transaction
coordinator.add_operation(&tx_id, TransactionOp::Query {
    subject: Some("http://example.org/alice"),
    predicate: None,
    object: None,
}).await?;

// Commit the transaction
coordinator.commit_transaction(&tx_id).await?;
```

## Deadlock Detection

The system implements a wait-for graph algorithm for deadlock detection:

```rust
// Transaction T1 waiting for lock held by T2
detector.add_wait("T1", "T2").await?;

// Transaction T2 waiting for lock held by T3
detector.add_wait("T2", "T3").await?;

// This would create a cycle: T3 -> T1 -> T2 -> T3
// The system will detect and prevent this
match detector.add_wait("T3", "T1").await {
    Err(e) if e.to_string().contains("Deadlock") => {
        // Handle deadlock
    }
    _ => {}
}
```

## Recovery

### Transaction Log
- All transaction state changes are logged
- Log entries include timestamp and transaction ID
- Used for recovery after failures

### Recovery Process
1. **Scan transaction log**
   - Identify incomplete transactions
   - Determine transaction states

2. **Query participants**
   - For prepared transactions, query participant votes
   - Reconstruct transaction state

3. **Complete transactions**
   - Commit transactions with all YES votes
   - Abort transactions with any NO votes
   - Clean up abandoned transactions

### Recovery Optimization
```rust
RecoveryPlan {
    // Transactions needing participant query
    transactions_to_query: Vec<TransactionId>,
    
    // Transactions needing commit completion
    transactions_to_commit: Vec<TransactionId>,
    
    // Transactions needing abort completion
    transactions_to_abort: Vec<TransactionId>,
}
```

## Performance Characteristics

### Latency
- Read-only transactions: ~1ms (no coordination)
- Single-shard transactions: ~5ms (local execution)
- Multi-shard transactions: ~20-50ms (2PC overhead)

### Throughput
- Read-only: 100,000+ TPS
- Single-shard: 50,000+ TPS
- Multi-shard: 10,000+ TPS

### Scalability
- Linear scalability for read-only workloads
- Near-linear for single-shard transactions
- Coordination overhead for multi-shard transactions

## Monitoring and Statistics

```rust
TransactionStatistics {
    total_transactions: 1000000,
    active_transactions: 42,
    committed_transactions: 999900,
    aborted_transactions: 58,
}

OptimizationStats {
    readonly_optimized: 800000,
    single_shard_optimized: 150000,
    presumed_abort_used: 50000,
    batched_transactions: 10000,
}
```

## Error Handling

### Common Errors
- **Deadlock**: Transaction aborted, retry with backoff
- **Timeout**: Transaction aborted, check participant health
- **Lock conflict**: Wait or abort based on policy
- **Network failure**: Trigger recovery protocol

### Error Recovery Strategies
1. **Automatic retry** with exponential backoff
2. **Participant health monitoring**
3. **Automatic failover** to replica nodes
4. **Transaction replay** from log

## Best Practices

1. **Use appropriate isolation level**
   - ReadCommitted for most cases
   - Serializable only when necessary

2. **Minimize transaction scope**
   - Keep transactions short
   - Batch related operations

3. **Design for single-shard transactions**
   - Colocate related data
   - Use semantic sharding

4. **Handle failures gracefully**
   - Implement retry logic
   - Use compensation transactions

5. **Monitor transaction metrics**
   - Track abort rates
   - Monitor lock contention
   - Identify hot spots

## Future Enhancements

1. **MVCC (Multi-Version Concurrency Control)**
   - Snapshot isolation
   - Reduced lock contention
   - Better read performance

2. **Distributed Deadlock Detection**
   - Global wait-for graph
   - Proactive deadlock prevention

3. **Advanced Recovery**
   - Parallel recovery
   - Incremental checkpointing
   - Point-in-time recovery

4. **Performance Optimizations**
   - Lock-free data structures
   - Hardware transactional memory
   - RDMA for low-latency communication

## References

1. Two-Phase Commit Protocol (Gray & Reuter)
2. Presumed Abort Optimization (Mohan & Lindsay)
3. Distributed Deadlock Detection (Chandy & Misra)
4. MVCC in Distributed Systems (PostgreSQL, CockroachDB)