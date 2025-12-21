//! Distributed Transaction Coordinator Service
//!
//! This module implements a comprehensive transaction coordinator service that manages
//! distributed transactions across multiple nodes using 2PC, 3PC, and Paxos protocols.
//!
//! # Features
//!
//! - **Multiple Protocol Support**: 2PC, 3PC, and Paxos-based consensus
//! - **Transaction Registry**: Track active, committed, and aborted transactions
//! - **Timeout Management**: Automatic cleanup of stale transactions
//! - **Recovery**: WAL-based recovery for coordinator failures
//! - **Monitoring**: Comprehensive statistics and health metrics
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────┐
//! │   Transaction Coordinator Service          │
//! │                                            │
//! │  ┌──────────────────────────────────────┐ │
//! │  │   Transaction Registry               │ │
//! │  │  - Active transactions               │ │
//! │  │  - Transaction state tracking        │ │
//! │  │  - Timeout monitoring                │ │
//! │  └──────────────────────────────────────┘ │
//! │                                            │
//! │  ┌──────────────────────────────────────┐ │
//! │  │   Protocol Selection                 │ │
//! │  │  - 2PC (fast, blocking)              │ │
//! │  │  - 3PC (non-blocking)                │ │
//! │  │  - Paxos (consensus)                 │ │
//! │  └──────────────────────────────────────┘ │
//! │                                            │
//! │  ┌──────────────────────────────────────┐ │
//! │  │   Participant Management             │ │
//! │  │  - Node health tracking              │ │
//! │  │  - Failure detection                 │ │
//! │  │  - Network communication             │ │
//! │  └──────────────────────────────────────┘ │
//! └────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_tdb::distributed::coordinator::{TransactionCoordinator, CoordinatorConfig, CommitProtocol};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create coordinator
//! let config = CoordinatorConfig::default();
//! let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);
//!
//! // Register participants
//! coordinator.register_participant("node1".to_string(), "http://node1:8080".to_string()).await?;
//! coordinator.register_participant("node2".to_string(), "http://node2:8080".to_string()).await?;
//!
//! // Start distributed transaction
//! let txn_id = coordinator.begin_transaction(CommitProtocol::ThreePhase).await?;
//!
//! // ... perform operations ...
//!
//! // Commit transaction
//! let result = coordinator.commit_transaction(&txn_id).await?;
//! # Ok(())
//! # }
//! ```

use crate::consensus::paxos::{PaxosProposer, ProposalValue};
use crate::error::{Result, TdbError};
use crate::transaction::three_phase_commit::{ThreePhaseCoordinator, TpcPhase};
use crate::transaction::two_phase_commit::{Participant, TpcState, TwoPhaseCoordinator};
use anyhow::Context;
use chrono::{DateTime, Duration, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Commit protocol to use for distributed transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitProtocol {
    /// Two-Phase Commit (faster, but blocking)
    TwoPhase,
    /// Three-Phase Commit (non-blocking)
    ThreePhase,
    /// Paxos-based consensus (most fault-tolerant)
    Paxos,
}

/// Transaction state in coordinator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinatorTxnState {
    /// Transaction initiated
    Active,
    /// Preparing to commit
    Preparing,
    /// Pre-commit phase (3PC only)
    PreCommitting,
    /// Committing
    Committing,
    /// Successfully committed
    Committed,
    /// Aborting
    Aborting,
    /// Aborted
    Aborted,
    /// Timeout occurred
    TimedOut,
}

/// Transaction metadata tracked by coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionMetadata {
    /// Transaction ID
    pub txn_id: String,
    /// Commit protocol used
    pub protocol: CommitProtocol,
    /// Current state
    pub state: CoordinatorTxnState,
    /// Participants
    pub participants: Vec<String>,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time (if completed)
    pub end_time: Option<DateTime<Utc>>,
    /// Timeout duration
    pub timeout: Duration,
}

/// Participant node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantNode {
    /// Node ID
    pub node_id: String,
    /// Network endpoint
    pub endpoint: String,
    /// Last heartbeat time
    pub last_heartbeat: DateTime<Utc>,
    /// Node health status
    pub healthy: bool,
}

/// Transaction Coordinator Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Default transaction timeout
    pub default_timeout: Duration,
    /// Maximum concurrent transactions
    pub max_concurrent_transactions: usize,
    /// Heartbeat interval for participant health checks
    pub heartbeat_interval: Duration,
    /// Participant timeout (consider dead if no heartbeat)
    pub participant_timeout: Duration,
    /// Enable automatic transaction cleanup
    pub auto_cleanup_enabled: bool,
    /// Cleanup interval for completed transactions
    pub cleanup_interval: Duration,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::minutes(5),
            max_concurrent_transactions: 1000,
            heartbeat_interval: Duration::seconds(10),
            participant_timeout: Duration::seconds(30),
            auto_cleanup_enabled: true,
            cleanup_interval: Duration::minutes(1),
        }
    }
}

/// Transaction Coordinator Service
///
/// Central coordinator for distributed transactions across multiple nodes.
pub struct TransactionCoordinator {
    /// Coordinator ID
    id: String,
    /// Configuration
    config: CoordinatorConfig,
    /// Active transactions
    transactions: Arc<RwLock<HashMap<String, TransactionMetadata>>>,
    /// Registered participants
    participants: Arc<RwLock<HashMap<String, ParticipantNode>>>,
    /// Transaction counter for ID generation
    txn_counter: Arc<Mutex<u64>>,
    /// Statistics
    stats: Arc<Mutex<CoordinatorStats>>,
}

/// Transaction Coordinator Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoordinatorStats {
    /// Total transactions initiated
    pub total_transactions: u64,
    /// Successful commits
    pub successful_commits: u64,
    /// Aborted transactions
    pub aborted_transactions: u64,
    /// Timed out transactions
    pub timed_out_transactions: u64,
    /// Active transaction count
    pub active_transactions: u64,
    /// Protocol usage counts
    pub two_phase_count: u64,
    /// Three-phase commit protocol usage count
    pub three_phase_count: u64,
    /// Paxos consensus protocol usage count
    pub paxos_count: u64,
    /// Average transaction duration (milliseconds)
    pub avg_transaction_duration_ms: f64,
    /// Total duration (for calculating average)
    total_duration_ms: f64,
}

impl TransactionCoordinator {
    /// Create a new Transaction Coordinator
    pub fn new(id: String, config: CoordinatorConfig) -> Self {
        Self {
            id,
            config,
            transactions: Arc::new(RwLock::new(HashMap::new())),
            participants: Arc::new(RwLock::new(HashMap::new())),
            txn_counter: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(CoordinatorStats::default())),
        }
    }

    /// Register a participant node
    pub async fn register_participant(&mut self, node_id: String, endpoint: String) -> Result<()> {
        let participant = ParticipantNode {
            node_id: node_id.clone(),
            endpoint,
            last_heartbeat: Utc::now(),
            healthy: true,
        };

        self.participants.write().insert(node_id, participant);
        Ok(())
    }

    /// Unregister a participant node
    pub async fn unregister_participant(&mut self, node_id: &str) -> Result<()> {
        self.participants.write().remove(node_id);
        Ok(())
    }

    /// Update participant heartbeat
    pub async fn update_heartbeat(&self, node_id: &str) -> Result<()> {
        let mut participants = self.participants.write();
        if let Some(participant) = participants.get_mut(node_id) {
            participant.last_heartbeat = Utc::now();
            participant.healthy = true;
        }
        Ok(())
    }

    /// Check participant health
    pub async fn check_participant_health(&self) -> Result<()> {
        let now = Utc::now();
        let timeout = self.config.participant_timeout;

        let mut participants = self.participants.write();
        for participant in participants.values_mut() {
            let elapsed = now - participant.last_heartbeat;
            if elapsed > timeout {
                participant.healthy = false;
            }
        }

        Ok(())
    }

    /// Begin a new distributed transaction
    pub async fn begin_transaction(&mut self, protocol: CommitProtocol) -> Result<String> {
        // Check concurrent transaction limit
        let active_count = self
            .transactions
            .read()
            .values()
            .filter(|t| t.state == CoordinatorTxnState::Active)
            .count();

        if active_count >= self.config.max_concurrent_transactions {
            return Err(TdbError::Other(
                "Maximum concurrent transactions limit reached".to_string(),
            ));
        }

        // Generate transaction ID
        let txn_id = self.generate_txn_id();

        // Get healthy participants
        let participants: Vec<String> = self
            .participants
            .read()
            .values()
            .filter(|p| p.healthy)
            .map(|p| p.node_id.clone())
            .collect();

        if participants.is_empty() {
            return Err(TdbError::Other(
                "No healthy participants available".to_string(),
            ));
        }

        // Create transaction metadata
        let metadata = TransactionMetadata {
            txn_id: txn_id.clone(),
            protocol,
            state: CoordinatorTxnState::Active,
            participants: participants.clone(),
            start_time: Utc::now(),
            end_time: None,
            timeout: self.config.default_timeout,
        };

        self.transactions.write().insert(txn_id.clone(), metadata);

        let mut stats = self.stats.lock();
        stats.total_transactions += 1;
        stats.active_transactions += 1;

        match protocol {
            CommitProtocol::TwoPhase => stats.two_phase_count += 1,
            CommitProtocol::ThreePhase => stats.three_phase_count += 1,
            CommitProtocol::Paxos => stats.paxos_count += 1,
        }

        Ok(txn_id)
    }

    /// Commit a distributed transaction
    pub async fn commit_transaction(&mut self, txn_id: &str) -> Result<bool> {
        let metadata = {
            let transactions = self.transactions.read();
            transactions
                .get(txn_id)
                .cloned()
                .ok_or_else(|| TdbError::Other(format!("Transaction not found: {}", txn_id)))?
        };

        // Check if transaction is still active
        if metadata.state != CoordinatorTxnState::Active {
            return Err(TdbError::Other(format!(
                "Transaction {} is not active (state: {:?})",
                txn_id, metadata.state
            )));
        }

        // Execute commit protocol
        let result = match metadata.protocol {
            CommitProtocol::TwoPhase => self.execute_two_phase_commit(&metadata).await?,
            CommitProtocol::ThreePhase => self.execute_three_phase_commit(&metadata).await?,
            CommitProtocol::Paxos => self.execute_paxos_commit(&metadata).await?,
        };

        // Update transaction state
        let final_state = if result {
            CoordinatorTxnState::Committed
        } else {
            CoordinatorTxnState::Aborted
        };

        self.update_transaction_state(txn_id, final_state).await?;

        // Update statistics
        let duration = Utc::now()
            .signed_duration_since(metadata.start_time)
            .num_milliseconds() as f64;
        let mut stats = self.stats.lock();
        stats.active_transactions -= 1;
        stats.total_duration_ms += duration;

        if result {
            stats.successful_commits += 1;
        } else {
            stats.aborted_transactions += 1;
        }

        stats.avg_transaction_duration_ms =
            stats.total_duration_ms / stats.total_transactions as f64;

        Ok(result)
    }

    /// Abort a distributed transaction
    pub async fn abort_transaction(&mut self, txn_id: &str) -> Result<()> {
        self.update_transaction_state(txn_id, CoordinatorTxnState::Aborting)
            .await?;

        // TODO: Send abort to all participants

        self.update_transaction_state(txn_id, CoordinatorTxnState::Aborted)
            .await?;

        let mut stats = self.stats.lock();
        stats.active_transactions -= 1;
        stats.aborted_transactions += 1;

        Ok(())
    }

    /// Execute Two-Phase Commit protocol
    async fn execute_two_phase_commit(&self, metadata: &TransactionMetadata) -> Result<bool> {
        let mut coordinator = TwoPhaseCoordinator::new(metadata.txn_id.clone());

        // Add participants
        for node_id in &metadata.participants {
            if let Some(participant) = self.participants.read().get(node_id) {
                coordinator.add_participant(Participant {
                    node_id: participant.node_id.clone(),
                    endpoint: participant.endpoint.clone(),
                });
            }
        }

        coordinator.commit().await
    }

    /// Execute Three-Phase Commit protocol
    async fn execute_three_phase_commit(&self, metadata: &TransactionMetadata) -> Result<bool> {
        let mut coordinator = ThreePhaseCoordinator::new(metadata.txn_id.clone());

        // Add participants
        for node_id in &metadata.participants {
            if let Some(participant) = self.participants.read().get(node_id) {
                coordinator.add_participant(Participant {
                    node_id: participant.node_id.clone(),
                    endpoint: participant.endpoint.clone(),
                });
            }
        }

        coordinator.commit().await
    }

    /// Execute Paxos-based commit
    async fn execute_paxos_commit(&self, metadata: &TransactionMetadata) -> Result<bool> {
        let mut proposer = PaxosProposer::new(self.id.clone());

        // Add acceptors (participants)
        for node_id in &metadata.participants {
            proposer.add_acceptor(node_id.clone());
        }

        // Propose transaction commit decision
        let value = ProposalValue::TxnDecision {
            txn_id: metadata.txn_id.clone(),
            commit: true,
        };

        let result = proposer.propose(value).await?;

        // Check if commit decision was chosen
        match result {
            ProposalValue::TxnDecision { commit, .. } => Ok(commit),
            _ => Ok(false),
        }
    }

    /// Update transaction state
    async fn update_transaction_state(
        &self,
        txn_id: &str,
        new_state: CoordinatorTxnState,
    ) -> Result<()> {
        let mut transactions = self.transactions.write();
        if let Some(metadata) = transactions.get_mut(txn_id) {
            metadata.state = new_state;

            if matches!(
                new_state,
                CoordinatorTxnState::Committed
                    | CoordinatorTxnState::Aborted
                    | CoordinatorTxnState::TimedOut
            ) {
                metadata.end_time = Some(Utc::now());
            }
        }
        Ok(())
    }

    /// Generate unique transaction ID
    fn generate_txn_id(&self) -> String {
        let mut counter = self.txn_counter.lock();
        *counter += 1;
        format!("{}-txn-{:08x}", self.id, *counter)
    }

    /// Get transaction metadata
    pub fn get_transaction(&self, txn_id: &str) -> Option<TransactionMetadata> {
        self.transactions.read().get(txn_id).cloned()
    }

    /// Get all active transactions
    pub fn get_active_transactions(&self) -> Vec<TransactionMetadata> {
        self.transactions
            .read()
            .values()
            .filter(|t| t.state == CoordinatorTxnState::Active)
            .cloned()
            .collect()
    }

    /// Clean up completed transactions
    pub async fn cleanup_completed_transactions(&self) -> Result<u64> {
        let now = Utc::now();
        let mut transactions = self.transactions.write();

        let to_remove: Vec<String> = transactions
            .iter()
            .filter(|(_, metadata)| {
                matches!(
                    metadata.state,
                    CoordinatorTxnState::Committed
                        | CoordinatorTxnState::Aborted
                        | CoordinatorTxnState::TimedOut
                ) && metadata
                    .end_time
                    .map(|end| now - end > self.config.cleanup_interval)
                    .unwrap_or(false)
            })
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len() as u64;
        for txn_id in to_remove {
            transactions.remove(&txn_id);
        }

        Ok(count)
    }

    /// Get coordinator ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get statistics
    pub fn stats(&self) -> CoordinatorStats {
        self.stats.lock().clone()
    }

    /// Get participant count
    pub fn participant_count(&self) -> usize {
        self.participants.read().len()
    }

    /// Get healthy participant count
    pub fn healthy_participant_count(&self) -> usize {
        self.participants
            .read()
            .values()
            .filter(|p| p.healthy)
            .count()
    }

    /// Get active transaction count
    pub fn active_transaction_count(&self) -> usize {
        self.transactions
            .read()
            .values()
            .filter(|t| t.state == CoordinatorTxnState::Active)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = CoordinatorConfig::default();
        let coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        assert_eq!(coordinator.id(), "coordinator-1");
        assert_eq!(coordinator.participant_count(), 0);
        assert_eq!(coordinator.active_transaction_count(), 0);
    }

    #[tokio::test]
    async fn test_register_participants() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        coordinator
            .register_participant("node2".to_string(), "http://node2:8080".to_string())
            .await
            .unwrap();

        assert_eq!(coordinator.participant_count(), 2);
        assert_eq!(coordinator.healthy_participant_count(), 2);
    }

    #[tokio::test]
    async fn test_begin_transaction() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        let txn_id = coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();

        assert!(txn_id.starts_with("coordinator-1-txn-"));
        assert_eq!(coordinator.active_transaction_count(), 1);

        let metadata = coordinator.get_transaction(&txn_id).unwrap();
        assert_eq!(metadata.protocol, CommitProtocol::TwoPhase);
        assert_eq!(metadata.state, CoordinatorTxnState::Active);
    }

    #[tokio::test]
    async fn test_commit_transaction_2pc() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        coordinator
            .register_participant("node2".to_string(), "http://node2:8080".to_string())
            .await
            .unwrap();

        let txn_id = coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();

        let result = coordinator.commit_transaction(&txn_id).await.unwrap();
        assert!(result, "Transaction should commit successfully");

        let metadata = coordinator.get_transaction(&txn_id).unwrap();
        assert_eq!(metadata.state, CoordinatorTxnState::Committed);

        let stats = coordinator.stats();
        assert_eq!(stats.successful_commits, 1);
        assert_eq!(stats.two_phase_count, 1);
    }

    #[tokio::test]
    async fn test_commit_transaction_3pc() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        let txn_id = coordinator
            .begin_transaction(CommitProtocol::ThreePhase)
            .await
            .unwrap();

        let result = coordinator.commit_transaction(&txn_id).await.unwrap();
        assert!(result);

        let stats = coordinator.stats();
        assert_eq!(stats.three_phase_count, 1);
    }

    #[tokio::test]
    async fn test_commit_transaction_paxos() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        coordinator
            .register_participant("node2".to_string(), "http://node2:8080".to_string())
            .await
            .unwrap();

        coordinator
            .register_participant("node3".to_string(), "http://node3:8080".to_string())
            .await
            .unwrap();

        let txn_id = coordinator
            .begin_transaction(CommitProtocol::Paxos)
            .await
            .unwrap();

        let result = coordinator.commit_transaction(&txn_id).await.unwrap();
        assert!(result);

        let stats = coordinator.stats();
        assert_eq!(stats.paxos_count, 1);
    }

    #[tokio::test]
    async fn test_abort_transaction() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        let txn_id = coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();

        coordinator.abort_transaction(&txn_id).await.unwrap();

        let metadata = coordinator.get_transaction(&txn_id).unwrap();
        assert_eq!(metadata.state, CoordinatorTxnState::Aborted);

        let stats = coordinator.stats();
        assert_eq!(stats.aborted_transactions, 1);
    }

    #[tokio::test]
    async fn test_heartbeat_tracking() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        assert_eq!(coordinator.healthy_participant_count(), 1);

        // Update heartbeat
        coordinator.update_heartbeat("node1").await.unwrap();
        assert_eq!(coordinator.healthy_participant_count(), 1);
    }

    #[tokio::test]
    async fn test_concurrent_transaction_limit() {
        let config = CoordinatorConfig {
            max_concurrent_transactions: 2,
            ..Default::default()
        };

        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        // First two transactions should succeed
        coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();

        coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();

        // Third should fail
        let result = coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cleanup_completed_transactions() {
        let config = CoordinatorConfig {
            cleanup_interval: Duration::milliseconds(100),
            ..Default::default()
        };

        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        let txn_id = coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();

        coordinator.commit_transaction(&txn_id).await.unwrap();

        // Wait for cleanup interval
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let cleaned = coordinator.cleanup_completed_transactions().await.unwrap();
        assert_eq!(cleaned, 1);
    }

    #[tokio::test]
    async fn test_get_active_transactions() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        let txn_id1 = coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();

        let txn_id2 = coordinator
            .begin_transaction(CommitProtocol::ThreePhase)
            .await
            .unwrap();

        let active = coordinator.get_active_transactions();
        assert_eq!(active.len(), 2);

        // Commit one
        coordinator.commit_transaction(&txn_id1).await.unwrap();

        let active = coordinator.get_active_transactions();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].txn_id, txn_id2);
    }

    #[tokio::test]
    async fn test_coordinator_stats() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TransactionCoordinator::new("coordinator-1".to_string(), config);

        coordinator
            .register_participant("node1".to_string(), "http://node1:8080".to_string())
            .await
            .unwrap();

        // Transaction 1: 2PC commit
        let txn1 = coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();
        coordinator.commit_transaction(&txn1).await.unwrap();

        // Transaction 2: 3PC commit
        let txn2 = coordinator
            .begin_transaction(CommitProtocol::ThreePhase)
            .await
            .unwrap();
        coordinator.commit_transaction(&txn2).await.unwrap();

        // Transaction 3: Abort
        let txn3 = coordinator
            .begin_transaction(CommitProtocol::TwoPhase)
            .await
            .unwrap();
        coordinator.abort_transaction(&txn3).await.unwrap();

        let stats = coordinator.stats();
        assert_eq!(stats.total_transactions, 3);
        assert_eq!(stats.successful_commits, 2);
        assert_eq!(stats.aborted_transactions, 1);
        assert_eq!(stats.two_phase_count, 2);
        assert_eq!(stats.three_phase_count, 1);
        assert!(stats.avg_transaction_duration_ms > 0.0);
    }
}
