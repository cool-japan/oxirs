//! Cross-shard transaction support with Two-Phase Commit (2PC) optimization

use crate::shard::{ShardId, ShardRouter};
use crate::shard_manager::ShardManager;
use crate::storage::StorageBackend;
use crate::network::{NetworkService, RpcMessage};
use crate::raft::OxirsNodeId;
use anyhow::Result;
use oxirs_core::model::Triple;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Transaction identifier
pub type TransactionId = String;

/// Transaction state for 2PC protocol
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransactionState {
    /// Transaction has been initiated
    Active,
    /// Prepare phase - waiting for participant votes
    Preparing,
    /// All participants voted to commit
    Prepared,
    /// Transaction is committing
    Committing,
    /// Transaction has been committed
    Committed,
    /// Transaction is aborting
    Aborting,
    /// Transaction has been aborted
    Aborted,
}

/// Transaction operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionOp {
    /// Insert a triple
    Insert { triple: Triple },
    /// Delete a triple
    Delete { triple: Triple },
    /// Query operation (read-only)
    Query {
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    },
}

/// Transaction participant information
#[derive(Debug, Clone)]
pub struct TransactionParticipant {
    pub node_id: OxirsNodeId,
    pub shard_id: ShardId,
    pub vote: Option<bool>,
    pub last_contact: Instant,
}

/// Cross-shard transaction
#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: TransactionId,
    pub state: TransactionState,
    pub operations: Vec<(ShardId, TransactionOp)>,
    pub participants: HashMap<ShardId, TransactionParticipant>,
    pub created_at: Instant,
    pub timeout: Duration,
    pub isolation_level: IsolationLevel,
}

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IsolationLevel {
    /// Read uncommitted - lowest isolation
    ReadUncommitted,
    /// Read committed - default for most operations
    ReadCommitted,
    /// Repeatable read - prevents phantom reads
    RepeatableRead,
    /// Serializable - highest isolation
    Serializable,
}

impl Default for IsolationLevel {
    fn default() -> Self {
        Self::ReadCommitted
    }
}

/// Transaction coordinator configuration
#[derive(Debug, Clone)]
pub struct TransactionConfig {
    /// Default transaction timeout
    pub default_timeout: Duration,
    /// Maximum concurrent transactions
    pub max_concurrent_transactions: usize,
    /// Enable optimistic concurrency control
    pub enable_optimistic_cc: bool,
    /// Enable deadlock detection
    pub enable_deadlock_detection: bool,
    /// Checkpoint interval for recovery
    pub checkpoint_interval: Duration,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            max_concurrent_transactions: 1000,
            enable_optimistic_cc: true,
            enable_deadlock_detection: true,
            checkpoint_interval: Duration::from_secs(60),
        }
    }
}

/// Transaction coordinator for 2PC protocol
pub struct TransactionCoordinator {
    node_id: OxirsNodeId,
    shard_router: Arc<ShardRouter>,
    shard_manager: Arc<ShardManager>,
    storage: Arc<dyn StorageBackend>,
    network: Arc<NetworkService>,
    config: TransactionConfig,
    transactions: Arc<RwLock<HashMap<TransactionId, Transaction>>>,
    transaction_log: Arc<Mutex<TransactionLog>>,
    lock_manager: Arc<LockManager>,
}

impl TransactionCoordinator {
    /// Create a new transaction coordinator
    pub fn new(
        node_id: OxirsNodeId,
        shard_router: Arc<ShardRouter>,
        shard_manager: Arc<ShardManager>,
        storage: Arc<dyn StorageBackend>,
        network: Arc<NetworkService>,
        config: TransactionConfig,
    ) -> Self {
        Self {
            node_id,
            shard_router,
            shard_manager,
            storage,
            network,
            config,
            transactions: Arc::new(RwLock::new(HashMap::new())),
            transaction_log: Arc::new(Mutex::new(TransactionLog::new())),
            lock_manager: Arc::new(LockManager::new()),
        }
    }

    /// Begin a new transaction
    pub async fn begin_transaction(&self, isolation_level: IsolationLevel) -> Result<TransactionId> {
        let tx_id = Uuid::new_v4().to_string();
        
        let transaction = Transaction {
            id: tx_id.clone(),
            state: TransactionState::Active,
            operations: Vec::new(),
            participants: HashMap::new(),
            created_at: Instant::now(),
            timeout: self.config.default_timeout,
            isolation_level,
        };

        // Check concurrent transaction limit
        {
            let transactions = self.transactions.read().await;
            if transactions.len() >= self.config.max_concurrent_transactions {
                return Err(anyhow::anyhow!("Maximum concurrent transactions exceeded"));
            }
        }

        // Log transaction start
        self.transaction_log.lock().await.log_begin(&tx_id).await?;

        // Store transaction
        self.transactions.write().await.insert(tx_id.clone(), transaction);

        Ok(tx_id)
    }

    /// Add an operation to a transaction
    pub async fn add_operation(
        &self,
        tx_id: &str,
        operation: TransactionOp,
    ) -> Result<()> {
        let mut transactions = self.transactions.write().await;
        let transaction = transactions.get_mut(tx_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;

        if transaction.state != TransactionState::Active {
            return Err(anyhow::anyhow!("Transaction is not active"));
        }

        // Determine which shard(s) are affected
        let shard_id = match &operation {
            TransactionOp::Insert { triple } | TransactionOp::Delete { triple } => {
                self.shard_router.route_triple(triple).await?
            }
            TransactionOp::Query { .. } => {
                // For queries, we might need multiple shards
                // For now, we'll handle this separately
                0 // Placeholder
            }
        };

        // Add participant if not already present
        if !transaction.participants.contains_key(&shard_id) {
            let node_id = self.shard_manager.get_primary_node(shard_id).await?;
            transaction.participants.insert(shard_id, TransactionParticipant {
                node_id,
                shard_id,
                vote: None,
                last_contact: Instant::now(),
            });
        }

        transaction.operations.push((shard_id, operation));

        Ok(())
    }

    /// Execute a transaction using 2PC protocol
    pub async fn commit_transaction(&self, tx_id: &str) -> Result<()> {
        // Phase 1: Prepare
        self.prepare_phase(tx_id).await?;

        // Phase 2: Commit or Abort
        let should_commit = self.check_votes(tx_id).await?;
        
        if should_commit {
            self.commit_phase(tx_id).await?;
        } else {
            self.abort_phase(tx_id).await?;
        }

        Ok(())
    }

    /// Phase 1: Prepare phase of 2PC
    async fn prepare_phase(&self, tx_id: &str) -> Result<()> {
        // Update transaction state
        {
            let mut transactions = self.transactions.write().await;
            let transaction = transactions.get_mut(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.state = TransactionState::Preparing;
        }

        // Log prepare phase
        self.transaction_log.lock().await.log_prepare(tx_id).await?;

        // Send prepare messages to all participants
        let participants = {
            let transactions = self.transactions.read().await;
            let transaction = transactions.get(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.participants.clone()
        };

        // Acquire locks for all operations
        let operations = {
            let transactions = self.transactions.read().await;
            let transaction = transactions.get(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.operations.clone()
        };

        // Try to acquire locks
        for (shard_id, op) in &operations {
            match op {
                TransactionOp::Insert { triple } | TransactionOp::Delete { triple } => {
                    self.lock_manager.acquire_write_lock(
                        tx_id,
                        *shard_id,
                        &triple.subject.to_string(),
                    ).await?;
                }
                TransactionOp::Query { subject, .. } => {
                    if let Some(subj) = subject {
                        self.lock_manager.acquire_read_lock(
                            tx_id,
                            *shard_id,
                            subj,
                        ).await?;
                    }
                }
            }
        }

        // Send prepare requests to participants
        for (shard_id, participant) in participants {
            let ops: Vec<_> = operations.iter()
                .filter(|(s, _)| *s == shard_id)
                .map(|(_, op)| op.clone())
                .collect();

            self.send_prepare_request(tx_id, participant.node_id, shard_id, ops).await?;
        }

        // Update state to prepared if we get here
        {
            let mut transactions = self.transactions.write().await;
            let transaction = transactions.get_mut(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.state = TransactionState::Prepared;
        }

        Ok(())
    }

    /// Check if all participants voted to commit
    async fn check_votes(&self, tx_id: &str) -> Result<bool> {
        let transactions = self.transactions.read().await;
        let transaction = transactions.get(tx_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;

        // Check if all participants have voted
        for participant in transaction.participants.values() {
            match participant.vote {
                Some(true) => continue,
                Some(false) => return Ok(false),
                None => {
                    // Timeout - treat as abort vote
                    if participant.last_contact.elapsed() > transaction.timeout {
                        return Ok(false);
                    }
                    // Still waiting
                    return Err(anyhow::anyhow!("Not all participants have voted"));
                }
            }
        }

        Ok(true)
    }

    /// Phase 2: Commit phase of 2PC
    async fn commit_phase(&self, tx_id: &str) -> Result<()> {
        // Update transaction state
        {
            let mut transactions = self.transactions.write().await;
            let transaction = transactions.get_mut(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.state = TransactionState::Committing;
        }

        // Log commit decision
        self.transaction_log.lock().await.log_commit(tx_id).await?;

        // Send commit messages to all participants
        let participants = {
            let transactions = self.transactions.read().await;
            let transaction = transactions.get(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.participants.clone()
        };

        for (shard_id, participant) in participants {
            self.send_commit_request(tx_id, participant.node_id, shard_id).await?;
        }

        // Release locks
        self.lock_manager.release_transaction_locks(tx_id).await;

        // Update final state
        {
            let mut transactions = self.transactions.write().await;
            let transaction = transactions.get_mut(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.state = TransactionState::Committed;
        }

        // Log completion
        self.transaction_log.lock().await.log_complete(tx_id, true).await?;

        Ok(())
    }

    /// Phase 2: Abort phase of 2PC
    async fn abort_phase(&self, tx_id: &str) -> Result<()> {
        // Update transaction state
        {
            let mut transactions = self.transactions.write().await;
            let transaction = transactions.get_mut(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.state = TransactionState::Aborting;
        }

        // Log abort decision
        self.transaction_log.lock().await.log_abort(tx_id).await?;

        // Send abort messages to all participants
        let participants = {
            let transactions = self.transactions.read().await;
            let transaction = transactions.get(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.participants.clone()
        };

        for (shard_id, participant) in participants {
            self.send_abort_request(tx_id, participant.node_id, shard_id).await?;
        }

        // Release locks
        self.lock_manager.release_transaction_locks(tx_id).await;

        // Update final state
        {
            let mut transactions = self.transactions.write().await;
            let transaction = transactions.get_mut(tx_id)
                .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;
            transaction.state = TransactionState::Aborted;
        }

        // Log completion
        self.transaction_log.lock().await.log_complete(tx_id, false).await?;

        Ok(())
    }

    /// Send prepare request to a participant
    async fn send_prepare_request(
        &self,
        tx_id: &str,
        node_id: OxirsNodeId,
        shard_id: ShardId,
        operations: Vec<TransactionOp>,
    ) -> Result<()> {
        let message = RpcMessage::TransactionPrepare {
            tx_id: tx_id.to_string(),
            shard_id,
            operations,
        };

        self.network.send_message(node_id, message).await?;
        Ok(())
    }

    /// Send commit request to a participant
    async fn send_commit_request(
        &self,
        tx_id: &str,
        node_id: OxirsNodeId,
        shard_id: ShardId,
    ) -> Result<()> {
        let message = RpcMessage::TransactionCommit {
            tx_id: tx_id.to_string(),
            shard_id,
        };

        self.network.send_message(node_id, message).await?;
        Ok(())
    }

    /// Send abort request to a participant
    async fn send_abort_request(
        &self,
        tx_id: &str,
        node_id: OxirsNodeId,
        shard_id: ShardId,
    ) -> Result<()> {
        let message = RpcMessage::TransactionAbort {
            tx_id: tx_id.to_string(),
            shard_id,
        };

        self.network.send_message(node_id, message).await?;
        Ok(())
    }

    /// Handle prepare vote from participant
    pub async fn handle_prepare_vote(
        &self,
        tx_id: &str,
        shard_id: ShardId,
        vote: bool,
    ) -> Result<()> {
        let mut transactions = self.transactions.write().await;
        let transaction = transactions.get_mut(tx_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction not found"))?;

        if let Some(participant) = transaction.participants.get_mut(&shard_id) {
            participant.vote = Some(vote);
            participant.last_contact = Instant::now();
        }

        Ok(())
    }

    /// Get transaction statistics
    pub async fn get_statistics(&self) -> TransactionStatistics {
        let transactions = self.transactions.read().await;
        let mut stats = TransactionStatistics::default();

        for transaction in transactions.values() {
            stats.total_transactions += 1;
            match transaction.state {
                TransactionState::Active => stats.active_transactions += 1,
                TransactionState::Committed => stats.committed_transactions += 1,
                TransactionState::Aborted => stats.aborted_transactions += 1,
                _ => {}
            }
        }

        stats
    }

    /// Cleanup completed transactions
    pub async fn cleanup_transactions(&self, retention: Duration) {
        let mut transactions = self.transactions.write().await;
        let now = Instant::now();
        
        transactions.retain(|_, tx| {
            match tx.state {
                TransactionState::Committed | TransactionState::Aborted => {
                    now.duration_since(tx.created_at) < retention
                }
                _ => true
            }
        });
    }
}

/// Transaction log for recovery
#[derive(Debug)]
struct TransactionLog {
    entries: Vec<LogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogEntry {
    timestamp: Instant,
    tx_id: String,
    entry_type: LogEntryType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LogEntryType {
    Begin,
    Prepare,
    Commit,
    Abort,
    Complete { committed: bool },
}

impl TransactionLog {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    async fn log_begin(&mut self, tx_id: &str) -> Result<()> {
        self.entries.push(LogEntry {
            timestamp: Instant::now(),
            tx_id: tx_id.to_string(),
            entry_type: LogEntryType::Begin,
        });
        Ok(())
    }

    async fn log_prepare(&mut self, tx_id: &str) -> Result<()> {
        self.entries.push(LogEntry {
            timestamp: Instant::now(),
            tx_id: tx_id.to_string(),
            entry_type: LogEntryType::Prepare,
        });
        Ok(())
    }

    async fn log_commit(&mut self, tx_id: &str) -> Result<()> {
        self.entries.push(LogEntry {
            timestamp: Instant::now(),
            tx_id: tx_id.to_string(),
            entry_type: LogEntryType::Commit,
        });
        Ok(())
    }

    async fn log_abort(&mut self, tx_id: &str) -> Result<()> {
        self.entries.push(LogEntry {
            timestamp: Instant::now(),
            tx_id: tx_id.to_string(),
            entry_type: LogEntryType::Abort,
        });
        Ok(())
    }

    async fn log_complete(&mut self, tx_id: &str, committed: bool) -> Result<()> {
        self.entries.push(LogEntry {
            timestamp: Instant::now(),
            tx_id: tx_id.to_string(),
            entry_type: LogEntryType::Complete { committed },
        });
        Ok(())
    }
}

/// Lock manager for concurrency control
#[derive(Debug)]
struct LockManager {
    locks: Arc<RwLock<HashMap<(ShardId, String), Lock>>>,
    tx_locks: Arc<RwLock<HashMap<TransactionId, HashSet<(ShardId, String)>>>>,
}

#[derive(Debug)]
struct Lock {
    lock_type: LockType,
    tx_id: TransactionId,
    acquired_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LockType {
    Read,
    Write,
}

impl LockManager {
    fn new() -> Self {
        Self {
            locks: Arc::new(RwLock::new(HashMap::new())),
            tx_locks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn acquire_read_lock(
        &self,
        tx_id: &str,
        shard_id: ShardId,
        resource: &str,
    ) -> Result<()> {
        let key = (shard_id, resource.to_string());
        let mut locks = self.locks.write().await;

        // Check for conflicting write locks
        if let Some(existing) = locks.get(&key) {
            if existing.lock_type == LockType::Write && existing.tx_id != tx_id {
                return Err(anyhow::anyhow!("Resource is write-locked by another transaction"));
            }
        }

        // Acquire read lock
        locks.insert(key.clone(), Lock {
            lock_type: LockType::Read,
            tx_id: tx_id.to_string(),
            acquired_at: Instant::now(),
        });

        // Track lock for transaction
        let mut tx_locks = self.tx_locks.write().await;
        tx_locks.entry(tx_id.to_string())
            .or_insert_with(HashSet::new)
            .insert(key);

        Ok(())
    }

    async fn acquire_write_lock(
        &self,
        tx_id: &str,
        shard_id: ShardId,
        resource: &str,
    ) -> Result<()> {
        let key = (shard_id, resource.to_string());
        let mut locks = self.locks.write().await;

        // Check for any existing locks
        if let Some(existing) = locks.get(&key) {
            if existing.tx_id != tx_id {
                return Err(anyhow::anyhow!("Resource is locked by another transaction"));
            }
        }

        // Acquire write lock
        locks.insert(key.clone(), Lock {
            lock_type: LockType::Write,
            tx_id: tx_id.to_string(),
            acquired_at: Instant::now(),
        });

        // Track lock for transaction
        let mut tx_locks = self.tx_locks.write().await;
        tx_locks.entry(tx_id.to_string())
            .or_insert_with(HashSet::new)
            .insert(key);

        Ok(())
    }

    async fn release_transaction_locks(&self, tx_id: &str) {
        let mut tx_locks = self.tx_locks.write().await;
        if let Some(locks_to_release) = tx_locks.remove(tx_id) {
            let mut locks = self.locks.write().await;
            for key in locks_to_release {
                locks.remove(&key);
            }
        }
    }
}

/// Transaction statistics
#[derive(Debug, Default, Clone)]
pub struct TransactionStatistics {
    pub total_transactions: usize,
    pub active_transactions: usize,
    pub committed_transactions: usize,
    pub aborted_transactions: usize,
}

/// RPC message extensions for transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionRpcMessage {
    /// Prepare request from coordinator
    TransactionPrepare {
        tx_id: TransactionId,
        shard_id: ShardId,
        operations: Vec<TransactionOp>,
    },
    /// Vote response from participant
    TransactionVote {
        tx_id: TransactionId,
        shard_id: ShardId,
        vote: bool,
    },
    /// Commit request from coordinator
    TransactionCommit {
        tx_id: TransactionId,
        shard_id: ShardId,
    },
    /// Abort request from coordinator
    TransactionAbort {
        tx_id: TransactionId,
        shard_id: ShardId,
    },
    /// Acknowledgment from participant
    TransactionAck {
        tx_id: TransactionId,
        shard_id: ShardId,
    },
}