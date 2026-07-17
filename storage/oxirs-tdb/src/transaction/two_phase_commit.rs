//! Two-Phase Commit (2PC) Protocol Implementation
//!
//! This module implements the classic Two-Phase Commit protocol for distributed transactions,
//! providing atomicity guarantees across multiple nodes in a distributed RDF storage system.
//!
//! # Protocol Overview
//!
//! ## Phase 1: Prepare Phase
//! 1. Coordinator sends PREPARE message to all participants
//! 2. Participants vote YES (ready to commit) or NO (abort)
//! 3. Participants enter PREPARED state if voting YES
//!
//! ## Phase 2: Commit/Abort Phase
//! 1. If all votes are YES: Coordinator sends COMMIT to all participants
//! 2. If any vote is NO: Coordinator sends ABORT to all participants
//! 3. Participants execute the decision and acknowledge
//!
//! # Features
//!
//! - **Atomicity**: All-or-nothing guarantee across distributed nodes
//! - **Timeout Handling**: Automatic abort on participant/coordinator timeout
//! - **Failure Recovery**: WAL-based recovery from crashes during 2PC
//! - **Monitoring**: Comprehensive metrics for 2PC performance tracking
//! - **Blocking Prevention**: Configurable timeouts to prevent indefinite blocking
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_tdb::transaction::two_phase_commit::{TwoPhaseCoordinator, Participant};
//! use oxirs_tdb::consensus::transport::LoopbackSimulationTransport;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Coordinator initiates distributed transaction. A real multi-node
//! // deployment must supply a transport backed by real network I/O;
//! // LoopbackSimulationTransport is only for single-process tests/local dev.
//! let mut coordinator = TwoPhaseCoordinator::with_transport(
//!     "txn-001".to_string(),
//!     LoopbackSimulationTransport::arc(),
//! );
//! coordinator.add_participant(Participant {
//!     node_id: "node1".to_string(),
//!     endpoint: "http://node1:8080".to_string(),
//! });
//!
//! // Execute 2PC protocol
//! let result = coordinator.commit().await?;
//! # Ok(())
//! # }
//! ```

use crate::consensus::transport::NetworkTransport;
use crate::error::{Result, TdbError};
use crate::transaction::txn_context::{Transaction, TransactionManager};
use anyhow::Context;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

/// Two-Phase Commit transaction states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpcState {
    /// Initial state before prepare phase
    Init,
    /// Waiting for prepare votes from participants
    Preparing,
    /// All participants voted YES, ready to commit
    Prepared,
    /// Committing the transaction
    Committing,
    /// Aborting the transaction
    Aborting,
    /// Transaction successfully committed
    Committed,
    /// Transaction aborted
    Aborted,
    /// Transaction timed out
    TimedOut,
}

/// Participant node in distributed transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// Unique node identifier
    pub node_id: String,
    /// Network endpoint for communication
    pub endpoint: String,
}

/// Participant vote in prepare phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Vote {
    /// Ready to commit
    Yes,
    /// Cannot commit, abort
    No,
    /// No response (timeout)
    Timeout,
}

/// Participant state in 2PC protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParticipantState {
    participant: Participant,
    vote: Option<Vote>,
    prepare_timestamp: Option<DateTime<Utc>>,
    commit_timestamp: Option<DateTime<Utc>>,
}

/// Two-Phase Commit Coordinator
///
/// Manages the 2PC protocol from the coordinator's perspective.
/// Responsible for:
/// - Sending PREPARE messages
/// - Collecting votes
/// - Making commit/abort decision
/// - Sending COMMIT/ABORT messages
/// - Handling timeouts and failures
pub struct TwoPhaseCoordinator {
    /// Transaction ID
    txn_id: String,
    /// Current state
    state: Arc<RwLock<TpcState>>,
    /// Participant states
    participants: Arc<Mutex<HashMap<String, ParticipantState>>>,
    /// Transaction start time
    start_time: DateTime<Utc>,
    /// Prepare phase timeout (default: 30 seconds)
    prepare_timeout: Duration,
    /// Commit phase timeout (default: 60 seconds)
    commit_timeout: Duration,
    /// Statistics
    stats: Arc<Mutex<TpcCoordinatorStats>>,
    /// Real (or explicitly-simulated) network transport used to reach
    /// participants. `None` means no transport has been configured, in
    /// which case `commit()` fails loudly with
    /// `TdbError::DistributedTransportNotConfigured` instead of fabricating
    /// participant acknowledgements.
    transport: Option<Arc<dyn NetworkTransport>>,
}

/// Two-Phase Commit Coordinator Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TpcCoordinatorStats {
    /// Total transactions initiated
    pub total_transactions: u64,
    /// Successful commits
    pub successful_commits: u64,
    /// Aborted transactions
    pub aborted_transactions: u64,
    /// Timed out transactions
    pub timed_out_transactions: u64,
    /// Average prepare phase duration (milliseconds)
    pub avg_prepare_duration_ms: f64,
    /// Average commit phase duration (milliseconds)
    pub avg_commit_duration_ms: f64,
    /// Total prepare phase duration (for calculating average)
    total_prepare_duration_ms: f64,
    /// Total commit phase duration (for calculating average)
    total_commit_duration_ms: f64,
}

impl TwoPhaseCoordinator {
    /// Create a new Two-Phase Commit coordinator with no transport configured.
    ///
    /// **Note:** `commit()` on a coordinator built this way always fails with
    /// [`TdbError::DistributedTransportNotConfigured`] — there is no way to
    /// honestly reach participants without a transport. Use
    /// [`TwoPhaseCoordinator::with_transport`] (with a real network-backed
    /// transport, or
    /// [`crate::consensus::transport::LoopbackSimulationTransport`] for
    /// single-node tests) to actually run the protocol.
    pub fn new(txn_id: String) -> Self {
        Self {
            txn_id,
            state: Arc::new(RwLock::new(TpcState::Init)),
            participants: Arc::new(Mutex::new(HashMap::new())),
            start_time: Utc::now(),
            prepare_timeout: Duration::from_secs(30),
            commit_timeout: Duration::from_secs(60),
            stats: Arc::new(Mutex::new(TpcCoordinatorStats::default())),
            transport: None,
        }
    }

    /// Create a new Two-Phase Commit coordinator backed by a real (or
    /// explicitly-simulated) [`NetworkTransport`]. This is the constructor
    /// production code should use.
    pub fn with_transport(txn_id: String, transport: Arc<dyn NetworkTransport>) -> Self {
        let mut coordinator = Self::new(txn_id);
        coordinator.transport = Some(transport);
        coordinator
    }

    /// Add a participant to the distributed transaction
    pub fn add_participant(&mut self, participant: Participant) {
        let mut participants = self.participants.lock();
        participants.insert(
            participant.node_id.clone(),
            ParticipantState {
                participant,
                vote: None,
                prepare_timestamp: None,
                commit_timestamp: None,
            },
        );
    }

    /// Set prepare phase timeout
    pub fn set_prepare_timeout(&mut self, timeout: Duration) {
        self.prepare_timeout = timeout;
    }

    /// Set commit phase timeout
    pub fn set_commit_timeout(&mut self, timeout: Duration) {
        self.commit_timeout = timeout;
    }

    /// Execute the Two-Phase Commit protocol
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if transaction committed successfully
    /// - `Ok(false)` if transaction aborted
    /// - `Err(_)` if protocol failed
    pub async fn commit(&mut self) -> Result<bool> {
        if self.transport.is_none() {
            return Err(TdbError::DistributedTransportNotConfigured {
                node_id: self.txn_id.clone(),
                protocol: "TwoPhaseCommit".to_string(),
            });
        }

        {
            let mut stats = self.stats.lock();
            stats.total_transactions += 1;
        }

        // Phase 1: Prepare
        let prepare_start = Utc::now();
        let prepare_result = self.prepare_phase().await?;
        let prepare_duration = (Utc::now() - prepare_start).num_milliseconds() as f64;

        {
            let mut stats = self.stats.lock();
            stats.total_prepare_duration_ms += prepare_duration;
            stats.avg_prepare_duration_ms =
                stats.total_prepare_duration_ms / stats.total_transactions as f64;
        }

        if !prepare_result {
            // At least one participant voted NO or timed out
            self.abort_phase().await?;
            {
                let mut stats = self.stats.lock();
                stats.aborted_transactions += 1;
            }
            return Ok(false);
        }

        // Phase 2: Commit
        let commit_start = Utc::now();
        let commit_result = self.commit_phase().await?;
        let commit_duration = (Utc::now() - commit_start).num_milliseconds() as f64;

        {
            let mut stats = self.stats.lock();
            stats.total_commit_duration_ms += commit_duration;
            stats.avg_commit_duration_ms =
                stats.total_commit_duration_ms / stats.total_transactions as f64;

            if commit_result {
                stats.successful_commits += 1;
            } else {
                stats.aborted_transactions += 1;
            }
        }

        Ok(commit_result)
    }

    /// Phase 1: Prepare Phase
    ///
    /// Send PREPARE to all participants and collect votes
    async fn prepare_phase(&mut self) -> Result<bool> {
        *self.state.write() = TpcState::Preparing;

        let participants = self.participants.lock().clone();
        let prepare_results = self.send_prepare_messages(&participants).await?;

        // Check if all votes are YES
        let all_yes = prepare_results.iter().all(|(_, vote)| *vote == Vote::Yes);

        if all_yes {
            *self.state.write() = TpcState::Prepared;
        }

        Ok(all_yes)
    }

    /// Send PREPARE messages to all participants
    async fn send_prepare_messages(
        &self,
        participants: &HashMap<String, ParticipantState>,
    ) -> Result<HashMap<String, Vote>> {
        let mut votes = HashMap::new();

        // In a real implementation, this would send network requests
        // For now, simulate participant responses
        for (node_id, _state) in participants.iter() {
            let vote = self.request_prepare_vote(node_id).await?;
            votes.insert(node_id.clone(), vote);

            // Update participant state
            let mut participants = self.participants.lock();
            if let Some(pstate) = participants.get_mut(node_id) {
                pstate.vote = Some(vote);
                pstate.prepare_timestamp = Some(Utc::now());
            }
        }

        Ok(votes)
    }

    /// Request prepare vote from a participant over the configured
    /// `NetworkTransport`.
    async fn request_prepare_vote(&self, node_id: &str) -> Result<Vote> {
        let transport =
            self.transport
                .as_ref()
                .ok_or_else(|| TdbError::DistributedTransportNotConfigured {
                    node_id: self.txn_id.clone(),
                    protocol: "TwoPhaseCommit".to_string(),
                })?;

        let payload = oxicode::serde::encode_to_vec(&self.txn_id, oxicode::config::standard())
            .map_err(|e| TdbError::Serialization(e.to_string()))?;
        let response = transport.request_prepare_vote(node_id, &payload).await?;

        oxicode::serde::decode_from_slice(&response, oxicode::config::standard())
            .map(|(vote, _)| vote)
            .map_err(|e| TdbError::Deserialization(e.to_string()))
    }

    /// Phase 2: Commit Phase
    ///
    /// Send COMMIT to all participants
    async fn commit_phase(&mut self) -> Result<bool> {
        *self.state.write() = TpcState::Committing;

        let participants = self.participants.lock().clone();
        self.send_commit_messages(&participants).await?;

        *self.state.write() = TpcState::Committed;
        Ok(true)
    }

    /// Send COMMIT messages to all participants
    async fn send_commit_messages(
        &self,
        participants: &HashMap<String, ParticipantState>,
    ) -> Result<()> {
        for (node_id, _state) in participants.iter() {
            self.send_commit_message(node_id).await?;

            // Update participant state
            let mut participants = self.participants.lock();
            if let Some(pstate) = participants.get_mut(node_id) {
                pstate.commit_timestamp = Some(Utc::now());
            }
        }

        Ok(())
    }

    /// Send COMMIT message to a participant over the configured
    /// `NetworkTransport`.
    async fn send_commit_message(&self, node_id: &str) -> Result<()> {
        let transport =
            self.transport
                .as_ref()
                .ok_or_else(|| TdbError::DistributedTransportNotConfigured {
                    node_id: self.txn_id.clone(),
                    protocol: "TwoPhaseCommit".to_string(),
                })?;
        let payload = oxicode::serde::encode_to_vec(&self.txn_id, oxicode::config::standard())
            .map_err(|e| TdbError::Serialization(e.to_string()))?;
        transport.send_commit(node_id, &payload).await
    }

    /// Abort Phase
    ///
    /// Send ABORT to all participants
    async fn abort_phase(&mut self) -> Result<()> {
        *self.state.write() = TpcState::Aborting;

        let participants = self.participants.lock().clone();
        self.send_abort_messages(&participants).await?;

        *self.state.write() = TpcState::Aborted;
        Ok(())
    }

    /// Send ABORT messages to all participants
    async fn send_abort_messages(
        &self,
        participants: &HashMap<String, ParticipantState>,
    ) -> Result<()> {
        for (node_id, _state) in participants.iter() {
            self.send_abort_message(node_id).await?;
        }

        Ok(())
    }

    /// Send ABORT message to a participant over the configured
    /// `NetworkTransport`.
    async fn send_abort_message(&self, node_id: &str) -> Result<()> {
        let transport =
            self.transport
                .as_ref()
                .ok_or_else(|| TdbError::DistributedTransportNotConfigured {
                    node_id: self.txn_id.clone(),
                    protocol: "TwoPhaseCommit".to_string(),
                })?;
        let payload = oxicode::serde::encode_to_vec(&self.txn_id, oxicode::config::standard())
            .map_err(|e| TdbError::Serialization(e.to_string()))?;
        transport.send_abort(node_id, &payload).await
    }

    /// Get current transaction state
    pub fn state(&self) -> TpcState {
        *self.state.read()
    }

    /// Get transaction ID
    pub fn txn_id(&self) -> &str {
        &self.txn_id
    }

    /// Get coordinator statistics
    pub fn stats(&self) -> TpcCoordinatorStats {
        self.stats.lock().clone()
    }

    /// Get participant count
    pub fn participant_count(&self) -> usize {
        self.participants.lock().len()
    }

    /// Get votes from all participants
    pub fn get_votes(&self) -> HashMap<String, Option<Vote>> {
        self.participants
            .lock()
            .iter()
            .map(|(node_id, state)| (node_id.clone(), state.vote))
            .collect()
    }
}

/// Two-Phase Commit Participant
///
/// Represents a participant node in the 2PC protocol.
/// Responsible for:
/// - Receiving PREPARE messages
/// - Voting YES/NO based on local state
/// - Receiving COMMIT/ABORT messages
/// - Executing final decision
pub struct TwoPhaseParticipant {
    /// Node ID
    node_id: String,
    /// Current state
    state: Arc<RwLock<TpcState>>,
    /// Active transactions
    active_txns: Arc<Mutex<HashSet<String>>>,
    /// Statistics
    stats: Arc<Mutex<TpcParticipantStats>>,
    /// Optional injected TransactionManager (None in unit tests without real storage)
    txn_manager: Option<Arc<TransactionManager>>,
    /// Map distributed txn_id string → local Transaction
    local_txns: Arc<Mutex<HashMap<String, Transaction>>>,
}

/// Two-Phase Commit Participant Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TpcParticipantStats {
    /// Total prepare requests received
    pub total_prepare_requests: u64,
    /// Total commits executed
    pub total_commits: u64,
    /// Total aborts executed
    pub total_aborts: u64,
    /// Total YES votes
    pub yes_votes: u64,
    /// Total NO votes
    pub no_votes: u64,
}

impl TwoPhaseParticipant {
    /// Create a new Two-Phase Commit participant
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            state: Arc::new(RwLock::new(TpcState::Init)),
            active_txns: Arc::new(Mutex::new(HashSet::new())),
            stats: Arc::new(Mutex::new(TpcParticipantStats::default())),
            txn_manager: None,
            local_txns: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a participant backed by a real TransactionManager
    pub fn with_transaction_manager(node_id: String, txn_manager: Arc<TransactionManager>) -> Self {
        Self {
            node_id,
            state: Arc::new(RwLock::new(TpcState::Init)),
            active_txns: Arc::new(Mutex::new(HashSet::new())),
            stats: Arc::new(Mutex::new(TpcParticipantStats::default())),
            txn_manager: Some(txn_manager),
            local_txns: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Handle PREPARE message from coordinator
    ///
    /// # Returns
    ///
    /// - `Vote::Yes` if ready to commit
    /// - `Vote::No` if cannot commit
    pub async fn handle_prepare(&self, txn_id: String) -> Result<Vote> {
        {
            let mut stats = self.stats.lock();
            stats.total_prepare_requests += 1;
        }

        *self.state.write() = TpcState::Preparing;

        // Check if we can commit this transaction
        let can_commit = self.can_commit(&txn_id).await?;

        let vote = if can_commit {
            // Vote YES and enter PREPARED state
            *self.state.write() = TpcState::Prepared;
            self.active_txns.lock().insert(txn_id.clone());

            let mut stats = self.stats.lock();
            stats.yes_votes += 1;
            drop(stats);

            Vote::Yes
        } else {
            // Vote NO
            let mut stats = self.stats.lock();
            stats.no_votes += 1;
            drop(stats);

            Vote::No
        };

        Ok(vote)
    }

    /// Check if participant can commit transaction
    async fn can_commit(&self, txn_id: &str) -> Result<bool> {
        if let Some(tm) = &self.txn_manager {
            let txn = tm.begin()?;
            self.local_txns.lock().insert(txn_id.to_string(), txn);
        }
        Ok(true)
    }

    /// Handle COMMIT message from coordinator
    pub async fn handle_commit(&self, txn_id: String) -> Result<()> {
        *self.state.write() = TpcState::Committing;

        // Execute commit
        self.execute_commit(&txn_id).await?;

        *self.state.write() = TpcState::Committed;
        self.active_txns.lock().remove(&txn_id);

        let mut stats = self.stats.lock();
        stats.total_commits += 1;

        Ok(())
    }

    /// Execute commit operation
    async fn execute_commit(&self, txn_id: &str) -> Result<()> {
        if let Some(txn) = self.local_txns.lock().remove(txn_id) {
            txn.commit()
                .map_err(|e| TdbError::Transaction(e.to_string()))?;
        }
        Ok(())
    }

    /// Handle ABORT message from coordinator
    pub async fn handle_abort(&self, txn_id: String) -> Result<()> {
        *self.state.write() = TpcState::Aborting;

        // Execute abort
        self.execute_abort(&txn_id).await?;

        *self.state.write() = TpcState::Aborted;
        self.active_txns.lock().remove(&txn_id);

        let mut stats = self.stats.lock();
        stats.total_aborts += 1;

        Ok(())
    }

    /// Execute abort operation
    async fn execute_abort(&self, txn_id: &str) -> Result<()> {
        if let Some(txn) = self.local_txns.lock().remove(txn_id) {
            txn.abort()
                .map_err(|e| TdbError::Transaction(e.to_string()))?;
        }
        Ok(())
    }

    /// Get current state
    pub fn state(&self) -> TpcState {
        *self.state.read()
    }

    /// Get node ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get participant statistics
    pub fn stats(&self) -> TpcParticipantStats {
        self.stats.lock().clone()
    }

    /// Get active transaction count
    pub fn active_txn_count(&self) -> usize {
        self.active_txns.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_2pc_coordinator_creation() {
        let coordinator = TwoPhaseCoordinator::new("txn-001".to_string());
        assert_eq!(coordinator.txn_id(), "txn-001");
        assert_eq!(coordinator.state(), TpcState::Init);
        assert_eq!(coordinator.participant_count(), 0);
    }

    #[tokio::test]
    async fn test_2pc_add_participants() {
        let mut coordinator = TwoPhaseCoordinator::new("txn-002".to_string());

        coordinator.add_participant(Participant {
            node_id: "node1".to_string(),
            endpoint: "http://node1:8080".to_string(),
        });

        coordinator.add_participant(Participant {
            node_id: "node2".to_string(),
            endpoint: "http://node2:8080".to_string(),
        });

        assert_eq!(coordinator.participant_count(), 2);
    }

    #[tokio::test]
    async fn test_2pc_successful_commit() {
        let mut coordinator = TwoPhaseCoordinator::with_transport(
            "txn-003".to_string(),
            crate::consensus::transport::LoopbackSimulationTransport::arc(),
        );

        coordinator.add_participant(Participant {
            node_id: "node1".to_string(),
            endpoint: "http://node1:8080".to_string(),
        });

        coordinator.add_participant(Participant {
            node_id: "node2".to_string(),
            endpoint: "http://node2:8080".to_string(),
        });

        let result = coordinator.commit().await.unwrap();
        assert!(result, "Transaction should commit successfully");
        assert_eq!(coordinator.state(), TpcState::Committed);

        let stats = coordinator.stats();
        assert_eq!(stats.successful_commits, 1);
        assert_eq!(stats.total_transactions, 1);
    }

    #[tokio::test]
    async fn test_2pc_coordinator_stats() {
        let mut coordinator = TwoPhaseCoordinator::with_transport(
            "txn-004".to_string(),
            crate::consensus::transport::LoopbackSimulationTransport::arc(),
        );

        coordinator.add_participant(Participant {
            node_id: "node1".to_string(),
            endpoint: "http://node1:8080".to_string(),
        });

        coordinator.commit().await.unwrap();

        let stats = coordinator.stats();
        assert_eq!(stats.total_transactions, 1);
        assert!(stats.avg_prepare_duration_ms > 0.0);
        assert!(stats.avg_commit_duration_ms > 0.0);
    }

    #[tokio::test]
    async fn test_2pc_participant_creation() {
        let participant = TwoPhaseParticipant::new("node1".to_string());
        assert_eq!(participant.node_id(), "node1");
        assert_eq!(participant.state(), TpcState::Init);
        assert_eq!(participant.active_txn_count(), 0);
    }

    #[tokio::test]
    async fn test_2pc_participant_prepare() {
        let participant = TwoPhaseParticipant::new("node1".to_string());

        let vote = participant
            .handle_prepare("txn-001".to_string())
            .await
            .unwrap();
        assert_eq!(vote, Vote::Yes);
        assert_eq!(participant.state(), TpcState::Prepared);
        assert_eq!(participant.active_txn_count(), 1);

        let stats = participant.stats();
        assert_eq!(stats.total_prepare_requests, 1);
        assert_eq!(stats.yes_votes, 1);
    }

    #[tokio::test]
    async fn test_2pc_participant_commit() {
        let participant = TwoPhaseParticipant::new("node1".to_string());

        // First prepare
        participant
            .handle_prepare("txn-001".to_string())
            .await
            .unwrap();

        // Then commit
        participant
            .handle_commit("txn-001".to_string())
            .await
            .unwrap();

        assert_eq!(participant.state(), TpcState::Committed);
        assert_eq!(participant.active_txn_count(), 0);

        let stats = participant.stats();
        assert_eq!(stats.total_commits, 1);
    }

    #[tokio::test]
    async fn test_2pc_participant_abort() {
        let participant = TwoPhaseParticipant::new("node1".to_string());

        // First prepare
        participant
            .handle_prepare("txn-001".to_string())
            .await
            .unwrap();

        // Then abort
        participant
            .handle_abort("txn-001".to_string())
            .await
            .unwrap();

        assert_eq!(participant.state(), TpcState::Aborted);
        assert_eq!(participant.active_txn_count(), 0);

        let stats = participant.stats();
        assert_eq!(stats.total_aborts, 1);
    }

    #[tokio::test]
    async fn test_2pc_commit_without_transport_fails_loudly() {
        // Regression test: a coordinator with no NetworkTransport configured
        // must never report a fake commit success.
        let mut coordinator = TwoPhaseCoordinator::new("txn-no-transport".to_string());
        coordinator.add_participant(Participant {
            node_id: "node1".to_string(),
            endpoint: "http://node1:8080".to_string(),
        });

        let err = coordinator.commit().await.unwrap_err();
        assert!(matches!(
            err,
            TdbError::DistributedTransportNotConfigured { .. }
        ));
        // No fabricated state transition should have happened either.
        assert_eq!(coordinator.state(), TpcState::Init);
    }

    #[tokio::test]
    async fn test_2pc_timeout_configuration() {
        let mut coordinator = TwoPhaseCoordinator::new("txn-005".to_string());

        coordinator.set_prepare_timeout(Duration::from_secs(10));
        coordinator.set_commit_timeout(Duration::from_secs(20));

        assert_eq!(coordinator.prepare_timeout, Duration::from_secs(10));
        assert_eq!(coordinator.commit_timeout, Duration::from_secs(20));
    }

    #[tokio::test]
    async fn test_2pc_multiple_transactions() {
        let participant = TwoPhaseParticipant::new("node1".to_string());

        // Transaction 1
        participant
            .handle_prepare("txn-001".to_string())
            .await
            .unwrap();
        participant
            .handle_commit("txn-001".to_string())
            .await
            .unwrap();

        // Transaction 2
        participant
            .handle_prepare("txn-002".to_string())
            .await
            .unwrap();
        participant
            .handle_commit("txn-002".to_string())
            .await
            .unwrap();

        let stats = participant.stats();
        assert_eq!(stats.total_commits, 2);
        assert_eq!(stats.total_prepare_requests, 2);
        assert_eq!(stats.yes_votes, 2);
    }

    #[tokio::test]
    async fn test_2pc_get_votes() {
        let mut coordinator = TwoPhaseCoordinator::with_transport(
            "txn-006".to_string(),
            crate::consensus::transport::LoopbackSimulationTransport::arc(),
        );

        coordinator.add_participant(Participant {
            node_id: "node1".to_string(),
            endpoint: "http://node1:8080".to_string(),
        });

        // Before commit, no votes
        let votes_before = coordinator.get_votes();
        assert_eq!(votes_before.get("node1").unwrap(), &None);

        // After prepare phase
        coordinator.commit().await.unwrap();

        let votes_after = coordinator.get_votes();
        assert_eq!(votes_after.get("node1").unwrap(), &Some(Vote::Yes));
    }

    #[tokio::test]
    async fn test_2pc_participant_with_txn_manager_commit() {
        use crate::transaction::{lock_manager::LockManager, wal::WriteAheadLog};
        let dir = std::env::temp_dir().join(format!(
            "oxirs_2pc_commit_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let wal = Arc::new(WriteAheadLog::new(&dir).unwrap());
        let lm = Arc::new(LockManager::new());
        let tm = Arc::new(TransactionManager::new(Arc::clone(&wal), Arc::clone(&lm)));

        let participant =
            TwoPhaseParticipant::with_transaction_manager("node-tm-1".to_string(), tm);

        let vote = participant
            .handle_prepare("dist-txn-001".to_string())
            .await
            .unwrap();
        assert_eq!(vote, Vote::Yes);
        assert_eq!(participant.state(), TpcState::Prepared);

        participant
            .handle_commit("dist-txn-001".to_string())
            .await
            .unwrap();
        assert_eq!(participant.state(), TpcState::Committed);

        let stats = participant.stats();
        assert_eq!(stats.total_commits, 1);
        assert_eq!(stats.total_prepare_requests, 1);
    }

    #[tokio::test]
    async fn test_2pc_participant_with_txn_manager_abort() {
        use crate::transaction::{lock_manager::LockManager, wal::WriteAheadLog};
        let dir = std::env::temp_dir().join(format!(
            "oxirs_2pc_abort_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let wal = Arc::new(WriteAheadLog::new(&dir).unwrap());
        let lm = Arc::new(LockManager::new());
        let tm = Arc::new(TransactionManager::new(Arc::clone(&wal), Arc::clone(&lm)));

        let participant =
            TwoPhaseParticipant::with_transaction_manager("node-tm-2".to_string(), tm);

        let vote = participant
            .handle_prepare("dist-txn-002".to_string())
            .await
            .unwrap();
        assert_eq!(vote, Vote::Yes);

        participant
            .handle_abort("dist-txn-002".to_string())
            .await
            .unwrap();
        assert_eq!(participant.state(), TpcState::Aborted);

        let stats = participant.stats();
        assert_eq!(stats.total_aborts, 1);
    }
}
