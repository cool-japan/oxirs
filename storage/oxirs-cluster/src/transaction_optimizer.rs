//! Optimizations for Two-Phase Commit protocol

use crate::transaction::{Transaction, TransactionId, TransactionOp, TransactionState, IsolationLevel};
use crate::shard::ShardId;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Optimization strategies for 2PC
#[derive(Debug, Clone)]
pub struct TwoPhaseOptimizer {
    /// Enable read-only optimization
    enable_readonly_opt: bool,
    /// Enable single-shard optimization
    enable_single_shard_opt: bool,
    /// Enable presumed abort optimization
    enable_presumed_abort: bool,
    /// Enable operation batching
    enable_batching: bool,
    /// Batch size for operations
    batch_size: usize,
    /// Statistics for optimization effectiveness
    stats: Arc<RwLock<OptimizationStats>>,
}

impl TwoPhaseOptimizer {
    /// Create a new optimizer with default settings
    pub fn new() -> Self {
        Self {
            enable_readonly_opt: true,
            enable_single_shard_opt: true,
            enable_presumed_abort: true,
            enable_batching: true,
            batch_size: 100,
            stats: Arc::new(RwLock::new(OptimizationStats::default())),
        }
    }

    /// Analyze a transaction and determine optimization opportunities
    pub async fn analyze_transaction(&self, transaction: &Transaction) -> TransactionOptimization {
        let mut optimization = TransactionOptimization::default();

        // Check if transaction is read-only
        let is_readonly = transaction.operations.iter().all(|(_, op)| {
            matches!(op, TransactionOp::Query { .. })
        });

        if is_readonly && self.enable_readonly_opt {
            optimization.skip_2pc = true;
            optimization.reason = "Read-only transaction".to_string();
            self.stats.write().await.readonly_optimized += 1;
            return optimization;
        }

        // Check if transaction affects only one shard
        let affected_shards: HashSet<_> = transaction.operations.iter()
            .map(|(shard_id, _)| *shard_id)
            .collect();

        if affected_shards.len() == 1 && self.enable_single_shard_opt {
            optimization.skip_2pc = true;
            optimization.single_shard = Some(*affected_shards.iter().next().unwrap());
            optimization.reason = "Single-shard transaction".to_string();
            self.stats.write().await.single_shard_optimized += 1;
            return optimization;
        }

        // Determine if we can use presumed abort
        if self.enable_presumed_abort {
            optimization.use_presumed_abort = true;
            self.stats.write().await.presumed_abort_used += 1;
        }

        // Check if operations can be batched
        if self.enable_batching && transaction.operations.len() > self.batch_size {
            optimization.batch_operations = true;
            optimization.batch_size = self.batch_size;
            self.stats.write().await.batched_transactions += 1;
        }

        // Analyze isolation level optimizations
        match transaction.isolation_level {
            IsolationLevel::ReadUncommitted => {
                optimization.skip_locking = true;
            }
            IsolationLevel::ReadCommitted => {
                optimization.release_locks_early = true;
            }
            _ => {}
        }

        optimization
    }

    /// Optimize the prepare phase for parallel execution
    pub async fn optimize_prepare_phase(
        &self,
        transaction: &Transaction,
    ) -> PrepareOptimization {
        let mut optimization = PrepareOptimization::default();

        // Group operations by shard for batch processing
        let mut shard_ops: HashMap<ShardId, Vec<TransactionOp>> = HashMap::new();
        for (shard_id, op) in &transaction.operations {
            shard_ops.entry(*shard_id).or_insert_with(Vec::new).push(op.clone());
        }

        // Determine parallel groups (shards that can be prepared in parallel)
        let parallel_groups = self.compute_parallel_groups(&shard_ops);
        optimization.parallel_groups = parallel_groups;

        // Identify critical path (longest chain of dependencies)
        optimization.critical_path = self.compute_critical_path(&shard_ops);

        // Determine timeout optimization
        optimization.optimized_timeout = self.compute_optimized_timeout(transaction);

        optimization
    }

    /// Compute parallel groups for prepare phase
    fn compute_parallel_groups(
        &self,
        shard_ops: &HashMap<ShardId, Vec<TransactionOp>>,
    ) -> Vec<Vec<ShardId>> {
        // Simple implementation: all shards can be prepared in parallel
        // In a more sophisticated implementation, we would analyze dependencies
        vec![shard_ops.keys().cloned().collect()]
    }

    /// Compute critical path for transaction
    fn compute_critical_path(&self, shard_ops: &HashMap<ShardId, Vec<TransactionOp>>) -> Vec<ShardId> {
        // Simple implementation: return shards ordered by operation count
        let mut shards: Vec<_> = shard_ops.iter()
            .map(|(shard_id, ops)| (*shard_id, ops.len()))
            .collect();
        shards.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        shards.into_iter().map(|(shard_id, _)| shard_id).collect()
    }

    /// Compute optimized timeout based on transaction characteristics
    fn compute_optimized_timeout(&self, transaction: &Transaction) -> std::time::Duration {
        use std::time::Duration;
        
        // Base timeout
        let mut timeout = Duration::from_secs(10);

        // Add time based on number of operations
        timeout += Duration::from_millis(transaction.operations.len() as u64 * 100);

        // Add time based on number of participants
        timeout += Duration::from_millis(transaction.participants.len() as u64 * 500);

        // Adjust based on isolation level
        match transaction.isolation_level {
            IsolationLevel::Serializable => timeout *= 2,
            IsolationLevel::RepeatableRead => timeout = timeout * 3 / 2,
            _ => {}
        }

        timeout.min(Duration::from_secs(60)) // Cap at 60 seconds
    }

    /// Optimize commit phase for faster completion
    pub async fn optimize_commit_phase(
        &self,
        transaction: &Transaction,
    ) -> CommitOptimization {
        let mut optimization = CommitOptimization::default();

        // Enable asynchronous commit for non-critical transactions
        if transaction.isolation_level != IsolationLevel::Serializable {
            optimization.async_commit = true;
        }

        // Enable group commit for better throughput
        optimization.group_commit = self.enable_batching;

        // Determine if we can skip logging for some participants
        if self.enable_presumed_abort {
            optimization.skip_participant_logging = true;
        }

        optimization
    }

    /// Get optimization statistics
    pub async fn get_statistics(&self) -> OptimizationStats {
        self.stats.read().await.clone()
    }

    /// Reset optimization statistics
    pub async fn reset_statistics(&self) {
        *self.stats.write().await = OptimizationStats::default();
    }
}

/// Transaction optimization result
#[derive(Debug, Default)]
pub struct TransactionOptimization {
    /// Skip 2PC entirely
    pub skip_2pc: bool,
    /// Single shard involved (if skip_2pc is true)
    pub single_shard: Option<ShardId>,
    /// Use presumed abort optimization
    pub use_presumed_abort: bool,
    /// Batch operations for better performance
    pub batch_operations: bool,
    /// Batch size for operations
    pub batch_size: usize,
    /// Skip locking for read uncommitted
    pub skip_locking: bool,
    /// Release locks early for read committed
    pub release_locks_early: bool,
    /// Reason for optimization
    pub reason: String,
}

/// Prepare phase optimization
#[derive(Debug, Default)]
pub struct PrepareOptimization {
    /// Groups of shards that can be prepared in parallel
    pub parallel_groups: Vec<Vec<ShardId>>,
    /// Critical path of shards (process these first)
    pub critical_path: Vec<ShardId>,
    /// Optimized timeout for prepare phase
    pub optimized_timeout: std::time::Duration,
}

/// Commit phase optimization
#[derive(Debug, Default)]
pub struct CommitOptimization {
    /// Use asynchronous commit
    pub async_commit: bool,
    /// Use group commit
    pub group_commit: bool,
    /// Skip logging for some participants
    pub skip_participant_logging: bool,
}

/// Optimization statistics
#[derive(Debug, Default, Clone)]
pub struct OptimizationStats {
    /// Transactions optimized as read-only
    pub readonly_optimized: u64,
    /// Transactions optimized as single-shard
    pub single_shard_optimized: u64,
    /// Transactions using presumed abort
    pub presumed_abort_used: u64,
    /// Transactions with batched operations
    pub batched_transactions: u64,
    /// Total transactions analyzed
    pub total_analyzed: u64,
}

/// Deadlock detection and prevention
pub struct DeadlockDetector {
    /// Wait-for graph
    wait_graph: Arc<RwLock<HashMap<TransactionId, HashSet<TransactionId>>>>,
}

impl DeadlockDetector {
    pub fn new() -> Self {
        Self {
            wait_graph: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a wait dependency
    pub async fn add_wait(&self, waiter: &str, holder: &str) -> Result<()> {
        let mut graph = self.wait_graph.write().await;
        
        // Check if adding this edge would create a cycle
        if self.would_create_cycle(&graph, waiter, holder) {
            return Err(anyhow::anyhow!("Deadlock detected"));
        }

        graph.entry(waiter.to_string())
            .or_insert_with(HashSet::new)
            .insert(holder.to_string());

        Ok(())
    }

    /// Remove wait dependencies for a transaction
    pub async fn remove_transaction(&self, tx_id: &str) {
        let mut graph = self.wait_graph.write().await;
        
        // Remove as waiter
        graph.remove(tx_id);
        
        // Remove as holder
        for waiters in graph.values_mut() {
            waiters.remove(tx_id);
        }
    }

    /// Check if adding an edge would create a cycle
    fn would_create_cycle(
        &self,
        graph: &HashMap<TransactionId, HashSet<TransactionId>>,
        from: &str,
        to: &str,
    ) -> bool {
        // Simple DFS to detect cycle
        let mut visited = HashSet::new();
        let mut stack = vec![to.to_string()];

        while let Some(node) = stack.pop() {
            if node == from {
                return true; // Cycle detected
            }

            if visited.insert(node.clone()) {
                if let Some(neighbors) = graph.get(&node) {
                    stack.extend(neighbors.iter().cloned());
                }
            }
        }

        false
    }
}

/// Recovery optimization for 2PC
pub struct RecoveryOptimizer {
    /// Checkpoint interval
    checkpoint_interval: std::time::Duration,
    /// Last checkpoint time
    last_checkpoint: Arc<RwLock<std::time::Instant>>,
}

impl RecoveryOptimizer {
    pub fn new(checkpoint_interval: std::time::Duration) -> Self {
        Self {
            checkpoint_interval,
            last_checkpoint: Arc::new(RwLock::new(std::time::Instant::now())),
        }
    }

    /// Determine if checkpoint is needed
    pub async fn should_checkpoint(&self) -> bool {
        let last = *self.last_checkpoint.read().await;
        last.elapsed() >= self.checkpoint_interval
    }

    /// Update checkpoint time
    pub async fn update_checkpoint(&self) {
        *self.last_checkpoint.write().await = std::time::Instant::now();
    }

    /// Optimize recovery by analyzing transaction log
    pub fn optimize_recovery_plan(
        &self,
        pending_transactions: Vec<(TransactionId, TransactionState)>,
    ) -> RecoveryPlan {
        let mut plan = RecoveryPlan::default();

        for (tx_id, state) in pending_transactions {
            match state {
                TransactionState::Preparing | TransactionState::Prepared => {
                    // Need to query participants and decide
                    plan.transactions_to_query.push(tx_id);
                }
                TransactionState::Committing => {
                    // Need to complete commit
                    plan.transactions_to_commit.push(tx_id);
                }
                TransactionState::Aborting => {
                    // Need to complete abort
                    plan.transactions_to_abort.push(tx_id);
                }
                _ => {}
            }
        }

        plan
    }
}

/// Recovery plan for pending transactions
#[derive(Debug, Default)]
pub struct RecoveryPlan {
    /// Transactions that need participant query
    pub transactions_to_query: Vec<TransactionId>,
    /// Transactions that need commit completion
    pub transactions_to_commit: Vec<TransactionId>,
    /// Transactions that need abort completion
    pub transactions_to_abort: Vec<TransactionId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_readonly_optimization() {
        let optimizer = TwoPhaseOptimizer::new();
        
        let mut transaction = Transaction {
            id: "test-tx".to_string(),
            state: TransactionState::Active,
            operations: vec![
                (0, TransactionOp::Query {
                    subject: Some("test".to_string()),
                    predicate: None,
                    object: None,
                }),
            ],
            participants: HashMap::new(),
            created_at: std::time::Instant::now(),
            timeout: std::time::Duration::from_secs(30),
            isolation_level: IsolationLevel::ReadCommitted,
        };

        let optimization = optimizer.analyze_transaction(&transaction).await;
        assert!(optimization.skip_2pc);
        assert_eq!(optimization.reason, "Read-only transaction");
    }

    #[tokio::test]
    async fn test_single_shard_optimization() {
        let optimizer = TwoPhaseOptimizer::new();
        
        let transaction = Transaction {
            id: "test-tx".to_string(),
            state: TransactionState::Active,
            operations: vec![
                (0, TransactionOp::Insert {
                    triple: oxirs_core::model::Triple {
                        subject: oxirs_core::model::NamedNode::new("http://example.org/s").unwrap().into(),
                        predicate: oxirs_core::model::NamedNode::new("http://example.org/p").unwrap().into(),
                        object: oxirs_core::model::NamedNode::new("http://example.org/o").unwrap().into(),
                    },
                }),
            ],
            participants: HashMap::new(),
            created_at: std::time::Instant::now(),
            timeout: std::time::Duration::from_secs(30),
            isolation_level: IsolationLevel::ReadCommitted,
        };

        let optimization = optimizer.analyze_transaction(&transaction).await;
        assert!(optimization.skip_2pc);
        assert_eq!(optimization.single_shard, Some(0));
        assert_eq!(optimization.reason, "Single-shard transaction");
    }

    #[test]
    fn test_deadlock_detection() {
        let detector = DeadlockDetector::new();
        
        // Test cycle detection
        let mut graph = HashMap::new();
        graph.insert("tx1".to_string(), vec!["tx2".to_string()].into_iter().collect());
        graph.insert("tx2".to_string(), vec!["tx3".to_string()].into_iter().collect());
        
        // This would create a cycle: tx3 -> tx1 -> tx2 -> tx3
        assert!(detector.would_create_cycle(&graph, "tx3", "tx1"));
        
        // This would not create a cycle
        assert!(!detector.would_create_cycle(&graph, "tx3", "tx4"));
    }
}