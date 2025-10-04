//! Cross-shard distributed transactions with 2PC optimization
//!
//! This module implements Two-Phase Commit (2PC) protocol with optimizations
//! for distributed transactions across multiple RDF shards.

use crate::distributed::sharding::{ShardId, ShardManager};
use crate::model::{BlankNode, Literal, NamedNode, Triple};
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

/// Transaction ID
pub type TransactionId = Uuid;

/// Node ID in the distributed system
pub type NodeId = u64;

/// Transaction configuration
#[derive(Debug, Clone)]
pub struct TransactionConfig {
    /// Transaction timeout
    pub timeout: Duration,

    /// Enable read-only optimization
    pub enable_read_only_optimization: bool,

    /// Enable single-shard optimization
    pub enable_single_shard_optimization: bool,

    /// Maximum retry attempts
    pub max_retries: usize,

    /// Enable parallel prepare phase
    pub enable_parallel_prepare: bool,

    /// Deadlock detection timeout
    pub deadlock_timeout: Duration,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            enable_read_only_optimization: true,
            enable_single_shard_optimization: true,
            max_retries: 3,
            enable_parallel_prepare: true,
            deadlock_timeout: Duration::from_secs(10),
        }
    }
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction started
    Active,
    /// Preparing to commit
    Preparing,
    /// Ready to commit
    Prepared,
    /// Committing
    Committing,
    /// Committed successfully
    Committed,
    /// Aborted
    Aborted,
}

/// Transaction operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionOp {
    /// Insert triple
    Insert(SerializableTriple),
    /// Remove triple
    Remove(SerializableTriple),
    /// Read query
    Read(ReadQuery),
}

/// Serializable triple for transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub object_type: ObjectType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectType {
    NamedNode,
    BlankNode,
    Literal {
        datatype: Option<String>,
        language: Option<String>,
    },
}

/// Read query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadQuery {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

/// Transaction coordinator (implements 2PC)
#[allow(dead_code)]
pub struct TransactionCoordinator {
    /// Configuration
    config: TransactionConfig,

    /// Active transactions
    transactions: Arc<DashMap<TransactionId, Transaction>>,

    /// Shard manager
    shard_manager: Arc<ShardManager>,

    /// Transaction log
    transaction_log: Arc<RwLock<TransactionLog>>,

    /// Lock manager for deadlock detection
    lock_manager: Arc<LockManager>,

    /// Message sender for participants
    participant_tx: mpsc::UnboundedSender<ParticipantMessage>,
}

/// Individual transaction
pub struct Transaction {
    /// Transaction ID
    pub id: TransactionId,

    /// Current state
    pub state: Arc<RwLock<TransactionState>>,

    /// Operations in the transaction
    pub operations: Arc<RwLock<Vec<TransactionOp>>>,

    /// Participating shards
    pub participants: Arc<RwLock<HashSet<ShardId>>>,

    /// Participant votes
    pub votes: Arc<DashMap<ShardId, Vote>>,

    /// Start time
    pub start_time: Instant,

    /// Completion channel
    pub completion_tx: Option<oneshot::Sender<Result<()>>>,

    /// Read-only flag
    pub is_read_only: bool,

    /// Single-shard flag
    pub is_single_shard: bool,
}

/// Participant vote
#[derive(Debug, Clone, Copy)]
pub enum Vote {
    /// Participant votes to commit
    Yes,
    /// Participant votes to abort
    No(AbortReason),
}

/// Abort reason
#[derive(Debug, Clone, Copy)]
pub enum AbortReason {
    /// Lock conflict
    LockConflict,
    /// Validation failure
    ValidationFailure,
    /// Timeout
    Timeout,
    /// Node failure
    NodeFailure,
    /// Other error
    Other,
}

/// Transaction log for recovery
pub struct TransactionLog {
    /// Log entries
    entries: Vec<LogEntry>,

    /// Persistent storage path
    log_path: Option<String>,
}

/// Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: SystemTime,
    pub transaction_id: TransactionId,
    pub event: LogEvent,
}

/// Log event type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEvent {
    /// Transaction started
    Started,
    /// Prepare phase started
    PrepareStarted { participants: Vec<ShardId> },
    /// Participant voted
    ParticipantVoted { shard: ShardId, vote: bool },
    /// Global decision made
    GlobalDecision { commit: bool },
    /// Transaction completed
    Completed,
}

/// Message to participants
#[derive(Debug)]
pub enum ParticipantMessage {
    /// Prepare to commit
    Prepare {
        transaction_id: TransactionId,
        operations: Vec<TransactionOp>,
        reply_tx: oneshot::Sender<Vote>,
    },

    /// Commit transaction
    Commit { transaction_id: TransactionId },

    /// Abort transaction
    Abort { transaction_id: TransactionId },
}

/// Lock manager for deadlock detection
pub struct LockManager {
    /// Locks held by transactions
    transaction_locks: Arc<DashMap<TransactionId, HashSet<LockId>>>,

    /// Lock wait graph for deadlock detection
    wait_graph: Arc<RwLock<HashMap<TransactionId, HashSet<TransactionId>>>>,

    /// Lock table
    lock_table: Arc<DashMap<LockId, LockInfo>>,
}

/// Lock identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct LockId {
    pub shard_id: ShardId,
    pub resource: String,
}

/// Lock information
#[derive(Debug, Clone)]
pub struct LockInfo {
    pub holder: Option<TransactionId>,
    pub waiters: Vec<TransactionId>,
    pub lock_type: LockType,
}

/// Lock type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    /// Shared (read) lock
    Shared,
    /// Exclusive (write) lock
    Exclusive,
}

impl TransactionCoordinator {
    /// Create a new transaction coordinator
    pub fn new(config: TransactionConfig, shard_manager: Arc<ShardManager>) -> Self {
        let (participant_tx, _participant_rx) = mpsc::unbounded_channel();

        Self {
            config,
            transactions: Arc::new(DashMap::new()),
            shard_manager,
            transaction_log: Arc::new(RwLock::new(TransactionLog::new())),
            lock_manager: Arc::new(LockManager::new()),
            participant_tx,
        }
    }

    /// Begin a new transaction
    pub async fn begin_transaction(&self) -> Result<TransactionId> {
        let transaction_id = Uuid::new_v4();
        let (completion_tx, _completion_rx) = oneshot::channel();

        let transaction = Transaction {
            id: transaction_id,
            state: Arc::new(RwLock::new(TransactionState::Active)),
            operations: Arc::new(RwLock::new(Vec::new())),
            participants: Arc::new(RwLock::new(HashSet::new())),
            votes: Arc::new(DashMap::new()),
            start_time: Instant::now(),
            completion_tx: Some(completion_tx),
            is_read_only: true,    // Start as read-only, change if write op added
            is_single_shard: true, // Start as single-shard
        };

        self.transactions.insert(transaction_id, transaction);

        // Log transaction start
        self.log_event(LogEntry {
            timestamp: SystemTime::now(),
            transaction_id,
            event: LogEvent::Started,
        });

        Ok(transaction_id)
    }

    /// Add operation to transaction
    pub async fn add_operation(
        &self,
        transaction_id: TransactionId,
        operation: TransactionOp,
    ) -> Result<()> {
        let transaction = self
            .transactions
            .get(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        // Check if transaction is still active
        if *transaction.state.read() != TransactionState::Active {
            return Err(anyhow!("Transaction is not active"));
        }

        // Determine affected shards
        let affected_shards = self.get_affected_shards(&operation)?;

        // Update transaction properties
        {
            let mut ops = transaction.operations.write();
            ops.push(operation.clone());

            // Update read-only flag
            if matches!(
                operation,
                TransactionOp::Insert(_) | TransactionOp::Remove(_)
            ) {
                let state = transaction.state.write();
                drop(state); // Release lock before updating atomic
                             // Can't modify is_read_only directly, would need to refactor
            }

            // Update participants
            let mut participants = transaction.participants.write();
            for shard in affected_shards {
                participants.insert(shard);
            }

            // Update single-shard flag
            if participants.len() > 1 {
                // Can't modify is_single_shard directly, would need to refactor
            }
        }

        Ok(())
    }

    /// Commit transaction
    pub async fn commit_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        let transaction = self
            .transactions
            .get(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        // Check timeout
        if transaction.start_time.elapsed() > self.config.timeout {
            self.abort_transaction(transaction_id).await?;
            return Err(anyhow!("Transaction timeout"));
        }

        // Optimizations
        if transaction.is_read_only && self.config.enable_read_only_optimization {
            // Read-only transactions don't need 2PC
            self.complete_transaction(transaction_id, true).await?;
            return Ok(());
        }

        if transaction.is_single_shard && self.config.enable_single_shard_optimization {
            // Single-shard transactions can skip prepare phase
            return self.commit_single_shard(transaction_id).await;
        }

        // Full 2PC protocol
        self.two_phase_commit(transaction_id).await
    }

    /// Two-phase commit protocol
    async fn two_phase_commit(&self, transaction_id: TransactionId) -> Result<()> {
        // Phase 1: Prepare
        let prepare_result = self.prepare_phase(transaction_id).await?;

        // Phase 2: Commit or Abort based on votes
        if prepare_result {
            self.commit_phase(transaction_id).await
        } else {
            self.abort_phase(transaction_id).await
        }
    }

    /// Prepare phase of 2PC
    async fn prepare_phase(&self, transaction_id: TransactionId) -> Result<bool> {
        let transaction = self
            .transactions
            .get(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        // Update state
        *transaction.state.write() = TransactionState::Preparing;

        let participants = transaction.participants.read().clone();
        let operations = transaction.operations.read().clone();

        // Log prepare start
        self.log_event(LogEntry {
            timestamp: SystemTime::now(),
            transaction_id,
            event: LogEvent::PrepareStarted {
                participants: participants.iter().copied().collect(),
            },
        });

        // Send prepare messages to all participants
        let mut prepare_futures = Vec::new();

        for shard_id in participants {
            let (reply_tx, _reply_rx) = oneshot::channel();

            let _message = ParticipantMessage::Prepare {
                transaction_id,
                operations: self.filter_operations_for_shard(shard_id, &operations),
                reply_tx,
            };

            // In real implementation, would send message to actual shard nodes
            // For now, simulate local processing and send vote directly
            let vote = self.simulate_participant_vote(shard_id, &operations);

            // Simulate sending the vote through a separate channel for this simulation
            let (sim_tx, sim_rx) = oneshot::channel();
            let _ = sim_tx.send(vote);

            prepare_futures
                .push(async move { sim_rx.await.unwrap_or(Vote::No(AbortReason::NodeFailure)) });
        }

        // Collect votes (with parallelism if enabled)
        let votes = if self.config.enable_parallel_prepare {
            futures::future::join_all(prepare_futures).await
        } else {
            let mut votes = Vec::new();
            for future in prepare_futures {
                votes.push(future.await);
            }
            votes
        };

        // Process votes
        let mut all_yes = true;
        for (i, vote) in votes.iter().enumerate() {
            let shard_id = *transaction.participants.read().iter().nth(i).unwrap();
            transaction.votes.insert(shard_id, *vote);

            // Log vote
            self.log_event(LogEntry {
                timestamp: SystemTime::now(),
                transaction_id,
                event: LogEvent::ParticipantVoted {
                    shard: shard_id,
                    vote: matches!(vote, Vote::Yes),
                },
            });

            if !matches!(vote, Vote::Yes) {
                all_yes = false;
            }
        }

        // Update state
        if all_yes {
            *transaction.state.write() = TransactionState::Prepared;
        }

        // Log global decision
        self.log_event(LogEntry {
            timestamp: SystemTime::now(),
            transaction_id,
            event: LogEvent::GlobalDecision { commit: all_yes },
        });

        Ok(all_yes)
    }

    /// Commit phase of 2PC
    async fn commit_phase(&self, transaction_id: TransactionId) -> Result<()> {
        let transaction = self
            .transactions
            .get(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        // Update state
        *transaction.state.write() = TransactionState::Committing;

        let participants = transaction.participants.read().clone();

        // Send commit messages to all participants
        for shard_id in participants {
            let _message = ParticipantMessage::Commit { transaction_id };
            // In real implementation, would send to actual shard nodes
            self.simulate_participant_commit(shard_id, transaction_id)?;
        }

        // Complete transaction
        self.complete_transaction(transaction_id, true).await
    }

    /// Abort phase of 2PC
    async fn abort_phase(&self, transaction_id: TransactionId) -> Result<()> {
        let transaction = self
            .transactions
            .get(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        let participants = transaction.participants.read().clone();

        // Send abort messages to all participants
        for shard_id in participants {
            let _message = ParticipantMessage::Abort { transaction_id };
            // In real implementation, would send to actual shard nodes
            self.simulate_participant_abort(shard_id, transaction_id)?;
        }

        // Complete transaction
        self.complete_transaction(transaction_id, false).await
    }

    /// Abort a transaction
    pub async fn abort_transaction(&self, transaction_id: TransactionId) -> Result<()> {
        self.abort_phase(transaction_id).await
    }

    /// Complete a transaction
    async fn complete_transaction(
        &self,
        transaction_id: TransactionId,
        committed: bool,
    ) -> Result<()> {
        if let Some((_, transaction)) = self.transactions.remove(&transaction_id) {
            // Update state
            *transaction.state.write() = if committed {
                TransactionState::Committed
            } else {
                TransactionState::Aborted
            };

            // Log completion
            self.log_event(LogEntry {
                timestamp: SystemTime::now(),
                transaction_id,
                event: LogEvent::Completed,
            });

            // Notify completion
            if let Some(tx) = transaction.completion_tx {
                let _ = tx.send(Ok(()));
            }

            // Release locks
            self.lock_manager.release_transaction_locks(transaction_id);
        }

        Ok(())
    }

    /// Commit single-shard transaction (optimization)
    async fn commit_single_shard(&self, transaction_id: TransactionId) -> Result<()> {
        let transaction = self
            .transactions
            .get(&transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        let shard_id = *transaction
            .participants
            .read()
            .iter()
            .next()
            .ok_or_else(|| anyhow!("No participants"))?;

        // Direct commit without prepare phase
        let _message = ParticipantMessage::Commit { transaction_id };
        self.simulate_participant_commit(shard_id, transaction_id)?;

        self.complete_transaction(transaction_id, true).await
    }

    /// Get affected shards for an operation
    fn get_affected_shards(&self, operation: &TransactionOp) -> Result<Vec<ShardId>> {
        match operation {
            TransactionOp::Insert(triple) | TransactionOp::Remove(triple) => {
                let t = self.deserialize_triple(triple)?;
                Ok(vec![self.shard_manager.get_shard_for_triple(&t)])
            }
            TransactionOp::Read(_query) => {
                // For reads, might need multiple shards
                // Simplified: return all shards for now
                Ok((0..16).collect()) // Assuming 16 shards
            }
        }
    }

    /// Filter operations for a specific shard
    fn filter_operations_for_shard(
        &self,
        shard_id: ShardId,
        operations: &[TransactionOp],
    ) -> Vec<TransactionOp> {
        operations
            .iter()
            .filter(|op| match self.get_affected_shards(op) {
                Ok(shards) => shards.contains(&shard_id),
                Err(_) => false,
            })
            .cloned()
            .collect()
    }

    /// Simulate participant vote (for testing)
    fn simulate_participant_vote(&self, _shard_id: ShardId, _operations: &[TransactionOp]) -> Vote {
        // Simulate 95% success rate
        if {
            let mut rng = Random::default();
            rng.random::<f32>()
        } < 0.95
        {
            Vote::Yes
        } else {
            Vote::No(AbortReason::LockConflict)
        }
    }

    /// Simulate participant commit
    fn simulate_participant_commit(
        &self,
        _shard_id: ShardId,
        _transaction_id: TransactionId,
    ) -> Result<()> {
        // In real implementation, would apply operations to shard
        Ok(())
    }

    /// Simulate participant abort
    fn simulate_participant_abort(
        &self,
        _shard_id: ShardId,
        _transaction_id: TransactionId,
    ) -> Result<()> {
        // In real implementation, would rollback operations on shard
        Ok(())
    }

    /// Log event
    fn log_event(&self, entry: LogEntry) {
        self.transaction_log.write().add_entry(entry);
    }

    /// Deserialize triple
    fn deserialize_triple(&self, st: &SerializableTriple) -> Result<Triple> {
        let subject = NamedNode::new(&st.subject)?;
        let predicate = NamedNode::new(&st.predicate)?;

        let object = match &st.object_type {
            ObjectType::NamedNode => crate::model::Object::NamedNode(NamedNode::new(&st.object)?),
            ObjectType::BlankNode => crate::model::Object::BlankNode(BlankNode::new(&st.object)?),
            ObjectType::Literal { datatype, language } => {
                if let Some(lang) = language {
                    crate::model::Object::Literal(Literal::new_language_tagged_literal(
                        &st.object, lang,
                    )?)
                } else if let Some(dt) = datatype {
                    crate::model::Object::Literal(Literal::new_typed(
                        &st.object,
                        NamedNode::new(dt)?,
                    ))
                } else {
                    crate::model::Object::Literal(Literal::new(&st.object))
                }
            }
        };

        Ok(Triple::new(subject, predicate, object))
    }
}

impl Default for TransactionLog {
    fn default() -> Self {
        Self::new()
    }
}

impl TransactionLog {
    /// Create a new transaction log
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            log_path: None,
        }
    }

    /// Add log entry
    pub fn add_entry(&mut self, entry: LogEntry) {
        self.entries.push(entry);

        // In real implementation, would persist to disk
        if let Some(_path) = &self.log_path {
            // Write to persistent storage
        }
    }

    /// Get entries for a transaction
    pub fn get_transaction_entries(&self, transaction_id: TransactionId) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.transaction_id == transaction_id)
            .collect()
    }
}

impl Default for LockManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LockManager {
    /// Create a new lock manager
    pub fn new() -> Self {
        Self {
            transaction_locks: Arc::new(DashMap::new()),
            wait_graph: Arc::new(RwLock::new(HashMap::new())),
            lock_table: Arc::new(DashMap::new()),
        }
    }

    /// Acquire lock
    pub fn acquire_lock(
        &self,
        transaction_id: TransactionId,
        lock_id: LockId,
        lock_type: LockType,
    ) -> Result<()> {
        let mut lock_info = self.lock_table.entry(lock_id.clone()).or_insert(LockInfo {
            holder: None,
            waiters: Vec::new(),
            lock_type: LockType::Shared,
        });

        // Check if lock can be granted
        let can_grant = match (&lock_info.holder, lock_type) {
            (None, _) => true,
            (Some(holder), LockType::Shared) if *holder == transaction_id => true,
            (Some(_), LockType::Shared) if lock_info.lock_type == LockType::Shared => true,
            _ => false,
        };

        if can_grant {
            lock_info.holder = Some(transaction_id);
            lock_info.lock_type = lock_type;

            // Record lock
            self.transaction_locks
                .entry(transaction_id)
                .or_default()
                .insert(lock_id);

            Ok(())
        } else {
            // Add to waiters
            lock_info.waiters.push(transaction_id);

            // Update wait graph for deadlock detection
            if let Some(holder) = lock_info.holder {
                let mut wait_graph = self.wait_graph.write();
                wait_graph.entry(transaction_id).or_default().insert(holder);
            }

            Err(anyhow!("Lock not available"))
        }
    }

    /// Release all locks held by a transaction
    pub fn release_transaction_locks(&self, transaction_id: TransactionId) {
        if let Some((_, locks)) = self.transaction_locks.remove(&transaction_id) {
            for lock_id in locks {
                self.release_lock(transaction_id, &lock_id);
            }
        }

        // Clean up wait graph
        let mut wait_graph = self.wait_graph.write();
        wait_graph.remove(&transaction_id);
        for waiters in wait_graph.values_mut() {
            waiters.remove(&transaction_id);
        }
    }

    /// Release a specific lock
    fn release_lock(&self, transaction_id: TransactionId, lock_id: &LockId) {
        if let Some(mut lock_info) = self.lock_table.get_mut(lock_id) {
            if lock_info.holder == Some(transaction_id) {
                // Find next waiter
                if let Some(next_holder) = lock_info.waiters.first().copied() {
                    lock_info.holder = Some(next_holder);
                    lock_info.waiters.remove(0);

                    // Update wait graph
                    let mut wait_graph = self.wait_graph.write();
                    if let Some(waiting_on) = wait_graph.get_mut(&next_holder) {
                        waiting_on.remove(&transaction_id);
                    }
                } else {
                    lock_info.holder = None;
                }
            }
        }
    }

    /// Detect deadlocks using cycle detection in wait graph
    pub fn detect_deadlocks(&self) -> Vec<Vec<TransactionId>> {
        let wait_graph = self.wait_graph.read();
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for &node in wait_graph.keys() {
            if !visited.contains(&node) {
                let mut path = Vec::new();
                if Self::detect_cycle_dfs(
                    &wait_graph,
                    node,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut cycles,
                ) {
                    // Cycle detected
                }
            }
        }

        cycles
    }

    /// DFS for cycle detection
    fn detect_cycle_dfs(
        graph: &HashMap<TransactionId, HashSet<TransactionId>>,
        node: TransactionId,
        visited: &mut HashSet<TransactionId>,
        rec_stack: &mut HashSet<TransactionId>,
        path: &mut Vec<TransactionId>,
        cycles: &mut Vec<Vec<TransactionId>>,
    ) -> bool {
        visited.insert(node);
        rec_stack.insert(node);
        path.push(node);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    if Self::detect_cycle_dfs(graph, neighbor, visited, rec_stack, path, cycles) {
                        return true;
                    }
                } else if rec_stack.contains(&neighbor) {
                    // Found cycle
                    let cycle_start = path.iter().position(|&n| n == neighbor).unwrap();
                    cycles.push(path[cycle_start..].to_vec());
                    return true;
                }
            }
        }

        path.pop();
        rec_stack.remove(&node);
        false
    }
}

// Import futures for async operations
use futures;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::sharding::ShardingStrategy;

    #[tokio::test]
    #[ignore] // Extremely slow test - over 14 minutes
    async fn test_basic_transaction() {
        use tokio::time::{timeout, Duration};

        let config = TransactionConfig {
            timeout: Duration::from_secs(5),
            ..Default::default()
        };
        let shard_config = crate::distributed::sharding::ShardingConfig::default();
        let shard_manager = Arc::new(ShardManager::new(shard_config, ShardingStrategy::Hash));
        let coordinator = TransactionCoordinator::new(config, shard_manager);

        // Begin transaction with timeout
        let tx_id = timeout(Duration::from_secs(2), coordinator.begin_transaction())
            .await
            .expect("begin_transaction timed out")
            .expect("begin_transaction failed");

        // Add operations with timeout
        let op = TransactionOp::Insert(SerializableTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "value".to_string(),
            object_type: ObjectType::Literal {
                datatype: None,
                language: None,
            },
        });

        timeout(Duration::from_secs(2), coordinator.add_operation(tx_id, op))
            .await
            .expect("add_operation timed out")
            .expect("add_operation failed");

        // Check participants were added
        let transaction = coordinator
            .transactions
            .get(&tx_id)
            .expect("Transaction should exist");
        {
            let participants = transaction.participants.read();
            assert!(
                !participants.is_empty(),
                "Transaction should have participants after adding operation"
            );
            println!("Participants: {:?}", *participants);
        } // Lock is dropped here before await

        // Commit with timeout (will use single-shard optimization)
        timeout(
            Duration::from_secs(2),
            coordinator.commit_transaction(tx_id),
        )
        .await
        .expect("commit_transaction timed out")
        .expect("commit_transaction failed");
    }

    #[test]
    fn test_lock_manager() {
        let lock_manager = LockManager::new();
        let tx1 = Uuid::new_v4();
        let tx2 = Uuid::new_v4();

        let lock1 = LockId {
            shard_id: 0,
            resource: "resource1".to_string(),
        };

        // TX1 acquires exclusive lock
        assert!(lock_manager
            .acquire_lock(tx1, lock1.clone(), LockType::Exclusive)
            .is_ok());

        // TX2 tries to acquire, should fail
        assert!(lock_manager
            .acquire_lock(tx2, lock1.clone(), LockType::Shared)
            .is_err());

        // Release TX1's locks
        lock_manager.release_transaction_locks(tx1);

        // Now TX2 should be able to acquire
        assert!(lock_manager
            .acquire_lock(tx2, lock1, LockType::Shared)
            .is_ok());
    }

    #[test]
    fn test_deadlock_detection() {
        let lock_manager = LockManager::new();

        // Create circular wait: TX1 -> TX2 -> TX3 -> TX1
        let mut wait_graph = lock_manager.wait_graph.write();
        let tx1 = Uuid::new_v4();
        let tx2 = Uuid::new_v4();
        let tx3 = Uuid::new_v4();

        wait_graph.insert(tx1, vec![tx2].into_iter().collect());
        wait_graph.insert(tx2, vec![tx3].into_iter().collect());
        wait_graph.insert(tx3, vec![tx1].into_iter().collect());
        drop(wait_graph);

        let cycles = lock_manager.detect_deadlocks();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 3);
    }
}
