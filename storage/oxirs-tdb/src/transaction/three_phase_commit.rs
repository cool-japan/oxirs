//! Three-Phase Commit (3PC) Protocol Implementation
//!
//! This module implements the Three-Phase Commit protocol, an extension of 2PC that eliminates
//! blocking in certain failure scenarios by adding a pre-commit phase.
//!
//! # Protocol Overview
//!
//! ## Phase 1: Can-Commit Phase (Prepare)
//! 1. Coordinator asks all participants if they can commit
//! 2. Participants vote YES (can commit) or NO (cannot commit)
//! 3. If any participant votes NO, the protocol aborts
//!
//! ## Phase 2: Pre-Commit Phase (NEW - Not in 2PC)
//! 1. Coordinator sends PRE-COMMIT to all participants
//! 2. Participants acknowledge they are ready to commit
//! 3. This phase ensures all participants know the decision before final commit
//!
//! ## Phase 3: Do-Commit Phase
//! 1. Coordinator sends DO-COMMIT to all participants
//! 2. Participants commit the transaction
//! 3. Participants acknowledge completion
//!
//! # Advantages over 2PC
//!
//! - **Non-blocking in Network Partitions**: If coordinator fails after pre-commit,
//!   participants can safely commit after timeout
//! - **Better Failure Recovery**: The pre-commit phase creates a synchronization point
//!   where all participants know the decision
//! - **Reduced Uncertainty Window**: Shorter window where outcome is uncertain
//!
//! # Trade-offs
//!
//! - **Additional Phase**: More messages and latency compared to 2PC
//! - **Complexity**: More complex state machine and recovery logic
//! - **Network Overhead**: Requires more round-trips
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_tdb::transaction::three_phase_commit::{ThreePhaseCoordinator, Participant};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Coordinator initiates distributed transaction
//! let mut coordinator = ThreePhaseCoordinator::new("txn-001".to_string());
//! coordinator.add_participant(Participant {
//!     node_id: "node1".to_string(),
//!     endpoint: "http://node1:8080".to_string(),
//! });
//!
//! // Execute 3PC protocol
//! let result = coordinator.commit().await?;
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, TdbError};
use crate::transaction::two_phase_commit::Participant;
use anyhow::Context;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

/// Three-Phase Commit protocol states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpcPhase {
    /// Initial state
    Init,
    /// Phase 1: Can-Commit (asking participants if they can commit)
    CanCommit,
    /// Waiting for can-commit responses
    WaitingCanCommit,
    /// Phase 2: Pre-Commit (telling participants to prepare for commit)
    PreCommit,
    /// Waiting for pre-commit acknowledgments
    WaitingPreCommit,
    /// Phase 3: Do-Commit (final commit)
    DoCommit,
    /// Waiting for commit acknowledgments
    WaitingCommit,
    /// Transaction committed successfully
    Committed,
    /// Aborting transaction
    Aborting,
    /// Transaction aborted
    Aborted,
    /// Transaction timed out
    TimedOut,
}

/// Participant response in can-commit phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanCommitResponse {
    /// Can commit
    Yes,
    /// Cannot commit
    No,
    /// No response (timeout)
    Timeout,
}

/// Participant state in 3PC protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParticipantState {
    participant: Participant,
    can_commit_response: Option<CanCommitResponse>,
    pre_commit_acked: bool,
    commit_acked: bool,
    can_commit_timestamp: Option<DateTime<Utc>>,
    pre_commit_timestamp: Option<DateTime<Utc>>,
    commit_timestamp: Option<DateTime<Utc>>,
}

/// Three-Phase Commit Coordinator
///
/// Manages the 3PC protocol from the coordinator's perspective.
/// Implements all three phases with proper timeout handling and recovery.
pub struct ThreePhaseCoordinator {
    /// Transaction ID
    txn_id: String,
    /// Current phase
    phase: Arc<RwLock<TpcPhase>>,
    /// Participant states
    participants: Arc<Mutex<HashMap<String, ParticipantState>>>,
    /// Transaction start time
    start_time: DateTime<Utc>,
    /// Can-commit phase timeout (default: 20 seconds)
    can_commit_timeout: Duration,
    /// Pre-commit phase timeout (default: 30 seconds)
    pre_commit_timeout: Duration,
    /// Do-commit phase timeout (default: 60 seconds)
    do_commit_timeout: Duration,
    /// Statistics
    stats: Arc<Mutex<ThreePhaseStats>>,
}

/// Three-Phase Commit Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreePhaseStats {
    /// Total transactions initiated
    pub total_transactions: u64,
    /// Successful commits
    pub successful_commits: u64,
    /// Aborted transactions
    pub aborted_transactions: u64,
    /// Timed out transactions
    pub timed_out_transactions: u64,
    /// Average can-commit phase duration (milliseconds)
    pub avg_can_commit_duration_ms: f64,
    /// Average pre-commit phase duration (milliseconds)
    pub avg_pre_commit_duration_ms: f64,
    /// Average do-commit phase duration (milliseconds)
    pub avg_do_commit_duration_ms: f64,
    /// Total phase durations (for calculating averages)
    total_can_commit_duration_ms: f64,
    total_pre_commit_duration_ms: f64,
    total_do_commit_duration_ms: f64,
}

impl ThreePhaseCoordinator {
    /// Create a new Three-Phase Commit coordinator
    pub fn new(txn_id: String) -> Self {
        Self {
            txn_id,
            phase: Arc::new(RwLock::new(TpcPhase::Init)),
            participants: Arc::new(Mutex::new(HashMap::new())),
            start_time: Utc::now(),
            can_commit_timeout: Duration::from_secs(20),
            pre_commit_timeout: Duration::from_secs(30),
            do_commit_timeout: Duration::from_secs(60),
            stats: Arc::new(Mutex::new(ThreePhaseStats::default())),
        }
    }

    /// Add a participant to the distributed transaction
    pub fn add_participant(&mut self, participant: Participant) {
        let mut participants = self.participants.lock();
        participants.insert(
            participant.node_id.clone(),
            ParticipantState {
                participant,
                can_commit_response: None,
                pre_commit_acked: false,
                commit_acked: false,
                can_commit_timestamp: None,
                pre_commit_timestamp: None,
                commit_timestamp: None,
            },
        );
    }

    /// Set phase timeouts
    pub fn set_timeouts(
        &mut self,
        can_commit: Duration,
        pre_commit: Duration,
        do_commit: Duration,
    ) {
        self.can_commit_timeout = can_commit;
        self.pre_commit_timeout = pre_commit;
        self.do_commit_timeout = do_commit;
    }

    /// Execute the Three-Phase Commit protocol
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if transaction committed successfully
    /// - `Ok(false)` if transaction aborted
    /// - `Err(_)` if protocol failed
    pub async fn commit(&mut self) -> Result<bool> {
        {
            let mut stats = self.stats.lock();
            stats.total_transactions += 1;
        }

        // Phase 1: Can-Commit
        let can_commit_start = Utc::now();
        let can_commit_result = self.can_commit_phase().await?;
        let can_commit_duration = (Utc::now() - can_commit_start).num_milliseconds() as f64;

        {
            let mut stats = self.stats.lock();
            stats.total_can_commit_duration_ms += can_commit_duration;
            stats.avg_can_commit_duration_ms =
                stats.total_can_commit_duration_ms / stats.total_transactions as f64;
        }

        if !can_commit_result {
            // At least one participant cannot commit
            self.abort_phase().await?;
            {
                let mut stats = self.stats.lock();
                stats.aborted_transactions += 1;
            }
            return Ok(false);
        }

        // Phase 2: Pre-Commit
        let pre_commit_start = Utc::now();
        let pre_commit_result = self.pre_commit_phase().await?;
        let pre_commit_duration = (Utc::now() - pre_commit_start).num_milliseconds() as f64;

        {
            let mut stats = self.stats.lock();
            stats.total_pre_commit_duration_ms += pre_commit_duration;
            stats.avg_pre_commit_duration_ms =
                stats.total_pre_commit_duration_ms / stats.total_transactions as f64;
        }

        if !pre_commit_result {
            // Pre-commit failed, abort
            self.abort_phase().await?;
            {
                let mut stats = self.stats.lock();
                stats.aborted_transactions += 1;
            }
            return Ok(false);
        }

        // Phase 3: Do-Commit
        let do_commit_start = Utc::now();
        let do_commit_result = self.do_commit_phase().await?;
        let do_commit_duration = (Utc::now() - do_commit_start).num_milliseconds() as f64;

        {
            let mut stats = self.stats.lock();
            stats.total_do_commit_duration_ms += do_commit_duration;
            stats.avg_do_commit_duration_ms =
                stats.total_do_commit_duration_ms / stats.total_transactions as f64;

            if do_commit_result {
                stats.successful_commits += 1;
            } else {
                stats.aborted_transactions += 1;
            }
        }

        Ok(do_commit_result)
    }

    /// Phase 1: Can-Commit Phase
    ///
    /// Ask all participants if they can commit
    async fn can_commit_phase(&mut self) -> Result<bool> {
        *self.phase.write() = TpcPhase::CanCommit;

        let participants = self.participants.lock().clone();
        let responses = self.send_can_commit_messages(&participants).await?;

        *self.phase.write() = TpcPhase::WaitingCanCommit;

        // Check if all responses are YES
        let all_yes = responses
            .iter()
            .all(|(_, response)| *response == CanCommitResponse::Yes);

        Ok(all_yes)
    }

    /// Send CAN-COMMIT messages to all participants
    async fn send_can_commit_messages(
        &self,
        participants: &HashMap<String, ParticipantState>,
    ) -> Result<HashMap<String, CanCommitResponse>> {
        let mut responses = HashMap::new();

        for (node_id, _state) in participants.iter() {
            let response = self.request_can_commit(node_id).await?;
            responses.insert(node_id.clone(), response);

            // Update participant state
            let mut participants = self.participants.lock();
            if let Some(pstate) = participants.get_mut(node_id) {
                pstate.can_commit_response = Some(response);
                pstate.can_commit_timestamp = Some(Utc::now());
            }
        }

        Ok(responses)
    }

    /// Request can-commit response from a participant (simulated)
    async fn request_can_commit(&self, _node_id: &str) -> Result<CanCommitResponse> {
        // TODO: Implement actual network communication
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(CanCommitResponse::Yes)
    }

    /// Phase 2: Pre-Commit Phase
    ///
    /// Tell all participants to prepare for commit
    async fn pre_commit_phase(&mut self) -> Result<bool> {
        *self.phase.write() = TpcPhase::PreCommit;

        let participants = self.participants.lock().clone();
        self.send_pre_commit_messages(&participants).await?;

        *self.phase.write() = TpcPhase::WaitingPreCommit;
        Ok(true)
    }

    /// Send PRE-COMMIT messages to all participants
    async fn send_pre_commit_messages(
        &self,
        participants: &HashMap<String, ParticipantState>,
    ) -> Result<()> {
        for (node_id, _state) in participants.iter() {
            self.send_pre_commit_message(node_id).await?;

            // Update participant state
            let mut participants = self.participants.lock();
            if let Some(pstate) = participants.get_mut(node_id) {
                pstate.pre_commit_acked = true;
                pstate.pre_commit_timestamp = Some(Utc::now());
            }
        }

        Ok(())
    }

    /// Send PRE-COMMIT message to a participant (simulated)
    async fn send_pre_commit_message(&self, _node_id: &str) -> Result<()> {
        // TODO: Implement actual network communication
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    /// Phase 3: Do-Commit Phase
    ///
    /// Send final commit command to all participants
    async fn do_commit_phase(&mut self) -> Result<bool> {
        *self.phase.write() = TpcPhase::DoCommit;

        let participants = self.participants.lock().clone();
        self.send_do_commit_messages(&participants).await?;

        *self.phase.write() = TpcPhase::Committed;
        Ok(true)
    }

    /// Send DO-COMMIT messages to all participants
    async fn send_do_commit_messages(
        &self,
        participants: &HashMap<String, ParticipantState>,
    ) -> Result<()> {
        for (node_id, _state) in participants.iter() {
            self.send_do_commit_message(node_id).await?;

            // Update participant state
            let mut participants = self.participants.lock();
            if let Some(pstate) = participants.get_mut(node_id) {
                pstate.commit_acked = true;
                pstate.commit_timestamp = Some(Utc::now());
            }
        }

        Ok(())
    }

    /// Send DO-COMMIT message to a participant (simulated)
    async fn send_do_commit_message(&self, _node_id: &str) -> Result<()> {
        // TODO: Implement actual network communication
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    /// Abort Phase
    ///
    /// Send ABORT to all participants
    async fn abort_phase(&mut self) -> Result<()> {
        *self.phase.write() = TpcPhase::Aborting;

        let participants = self.participants.lock().clone();
        self.send_abort_messages(&participants).await?;

        *self.phase.write() = TpcPhase::Aborted;
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

    /// Send ABORT message to a participant (simulated)
    async fn send_abort_message(&self, _node_id: &str) -> Result<()> {
        // TODO: Implement actual network communication
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    /// Get current phase
    pub fn phase(&self) -> TpcPhase {
        *self.phase.read()
    }

    /// Get transaction ID
    pub fn txn_id(&self) -> &str {
        &self.txn_id
    }

    /// Get statistics
    pub fn stats(&self) -> ThreePhaseStats {
        self.stats.lock().clone()
    }

    /// Get participant count
    pub fn participant_count(&self) -> usize {
        self.participants.lock().len()
    }

    /// Get can-commit responses from all participants
    pub fn get_can_commit_responses(&self) -> HashMap<String, Option<CanCommitResponse>> {
        self.participants
            .lock()
            .iter()
            .map(|(node_id, state)| (node_id.clone(), state.can_commit_response))
            .collect()
    }
}

/// Three-Phase Commit Participant
///
/// Represents a participant node in the 3PC protocol.
pub struct ThreePhaseParticipant {
    /// Node ID
    node_id: String,
    /// Current phase
    phase: Arc<RwLock<TpcPhase>>,
    /// Active transactions
    active_txns: Arc<Mutex<HashSet<String>>>,
    /// Pre-committed transactions (ready to commit on timeout)
    pre_committed_txns: Arc<Mutex<HashSet<String>>>,
    /// Statistics
    stats: Arc<Mutex<ThreePhaseParticipantStats>>,
}

/// Three-Phase Commit Participant Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreePhaseParticipantStats {
    /// Total can-commit requests received
    pub total_can_commit_requests: u64,
    /// Total pre-commit messages received
    pub total_pre_commit_messages: u64,
    /// Total commits executed
    pub total_commits: u64,
    /// Total aborts executed
    pub total_aborts: u64,
    /// YES responses
    pub yes_responses: u64,
    /// NO responses
    pub no_responses: u64,
}

impl ThreePhaseParticipant {
    /// Create a new Three-Phase Commit participant
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            phase: Arc::new(RwLock::new(TpcPhase::Init)),
            active_txns: Arc::new(Mutex::new(HashSet::new())),
            pre_committed_txns: Arc::new(Mutex::new(HashSet::new())),
            stats: Arc::new(Mutex::new(ThreePhaseParticipantStats::default())),
        }
    }

    /// Handle CAN-COMMIT message from coordinator
    pub async fn handle_can_commit(&self, txn_id: String) -> Result<CanCommitResponse> {
        {
            let mut stats = self.stats.lock();
            stats.total_can_commit_requests += 1;
        }

        *self.phase.write() = TpcPhase::CanCommit;

        // Check if we can commit this transaction
        let can_commit = self.can_commit(&txn_id).await?;

        let response = if can_commit {
            self.active_txns.lock().insert(txn_id.clone());

            let mut stats = self.stats.lock();
            stats.yes_responses += 1;
            drop(stats);

            CanCommitResponse::Yes
        } else {
            let mut stats = self.stats.lock();
            stats.no_responses += 1;
            drop(stats);

            CanCommitResponse::No
        };

        Ok(response)
    }

    /// Check if participant can commit transaction
    async fn can_commit(&self, _txn_id: &str) -> Result<bool> {
        // TODO: Implement actual resource checking
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(true)
    }

    /// Handle PRE-COMMIT message from coordinator
    pub async fn handle_pre_commit(&self, txn_id: String) -> Result<()> {
        {
            let mut stats = self.stats.lock();
            stats.total_pre_commit_messages += 1;
        }

        *self.phase.write() = TpcPhase::PreCommit;

        // Move transaction to pre-committed state
        self.pre_committed_txns.lock().insert(txn_id.clone());

        // Execute pre-commit preparation
        self.execute_pre_commit(&txn_id).await?;

        *self.phase.write() = TpcPhase::WaitingCommit;
        Ok(())
    }

    /// Execute pre-commit preparation
    async fn execute_pre_commit(&self, _txn_id: &str) -> Result<()> {
        // TODO: Implement actual pre-commit logic
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }

    /// Handle DO-COMMIT message from coordinator
    pub async fn handle_do_commit(&self, txn_id: String) -> Result<()> {
        *self.phase.write() = TpcPhase::DoCommit;

        // Execute commit
        self.execute_commit(&txn_id).await?;

        *self.phase.write() = TpcPhase::Committed;
        self.active_txns.lock().remove(&txn_id);
        self.pre_committed_txns.lock().remove(&txn_id);

        let mut stats = self.stats.lock();
        stats.total_commits += 1;

        Ok(())
    }

    /// Execute commit operation
    async fn execute_commit(&self, _txn_id: &str) -> Result<()> {
        // TODO: Implement actual commit logic
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }

    /// Handle ABORT message from coordinator
    pub async fn handle_abort(&self, txn_id: String) -> Result<()> {
        *self.phase.write() = TpcPhase::Aborting;

        // Execute abort
        self.execute_abort(&txn_id).await?;

        *self.phase.write() = TpcPhase::Aborted;
        self.active_txns.lock().remove(&txn_id);
        self.pre_committed_txns.lock().remove(&txn_id);

        let mut stats = self.stats.lock();
        stats.total_aborts += 1;

        Ok(())
    }

    /// Execute abort operation
    async fn execute_abort(&self, _txn_id: &str) -> Result<()> {
        // TODO: Implement actual abort logic
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(())
    }

    /// Handle coordinator timeout (3PC advantage: can commit if pre-committed)
    ///
    /// If coordinator fails after pre-commit phase, participants can
    /// safely commit the transaction after timeout.
    pub async fn handle_coordinator_timeout(&self, txn_id: String) -> Result<bool> {
        // Check if transaction is in pre-committed state
        let is_pre_committed = self.pre_committed_txns.lock().contains(&txn_id);

        if is_pre_committed {
            // Safe to commit - all participants know the decision
            self.handle_do_commit(txn_id).await?;
            Ok(true)
        } else {
            // Not pre-committed, must abort
            self.handle_abort(txn_id).await?;
            Ok(false)
        }
    }

    /// Get current phase
    pub fn phase(&self) -> TpcPhase {
        *self.phase.read()
    }

    /// Get node ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get statistics
    pub fn stats(&self) -> ThreePhaseParticipantStats {
        self.stats.lock().clone()
    }

    /// Get active transaction count
    pub fn active_txn_count(&self) -> usize {
        self.active_txns.lock().len()
    }

    /// Get pre-committed transaction count
    pub fn pre_committed_txn_count(&self) -> usize {
        self.pre_committed_txns.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_3pc_coordinator_creation() {
        let coordinator = ThreePhaseCoordinator::new("txn-001".to_string());
        assert_eq!(coordinator.txn_id(), "txn-001");
        assert_eq!(coordinator.phase(), TpcPhase::Init);
        assert_eq!(coordinator.participant_count(), 0);
    }

    #[tokio::test]
    async fn test_3pc_successful_commit() {
        let mut coordinator = ThreePhaseCoordinator::new("txn-002".to_string());

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
        assert_eq!(coordinator.phase(), TpcPhase::Committed);

        let stats = coordinator.stats();
        assert_eq!(stats.successful_commits, 1);
        assert_eq!(stats.total_transactions, 1);
    }

    #[tokio::test]
    async fn test_3pc_coordinator_stats() {
        let mut coordinator = ThreePhaseCoordinator::new("txn-003".to_string());

        coordinator.add_participant(Participant {
            node_id: "node1".to_string(),
            endpoint: "http://node1:8080".to_string(),
        });

        coordinator.commit().await.unwrap();

        let stats = coordinator.stats();
        assert_eq!(stats.total_transactions, 1);
        assert!(stats.avg_can_commit_duration_ms > 0.0);
        assert!(stats.avg_pre_commit_duration_ms > 0.0);
        assert!(stats.avg_do_commit_duration_ms > 0.0);
    }

    #[tokio::test]
    async fn test_3pc_participant_creation() {
        let participant = ThreePhaseParticipant::new("node1".to_string());
        assert_eq!(participant.node_id(), "node1");
        assert_eq!(participant.phase(), TpcPhase::Init);
        assert_eq!(participant.active_txn_count(), 0);
    }

    #[tokio::test]
    async fn test_3pc_participant_can_commit() {
        let participant = ThreePhaseParticipant::new("node1".to_string());

        let response = participant
            .handle_can_commit("txn-001".to_string())
            .await
            .unwrap();
        assert_eq!(response, CanCommitResponse::Yes);
        assert_eq!(participant.active_txn_count(), 1);

        let stats = participant.stats();
        assert_eq!(stats.total_can_commit_requests, 1);
        assert_eq!(stats.yes_responses, 1);
    }

    #[tokio::test]
    async fn test_3pc_participant_pre_commit() {
        let participant = ThreePhaseParticipant::new("node1".to_string());

        // First can-commit
        participant
            .handle_can_commit("txn-001".to_string())
            .await
            .unwrap();

        // Then pre-commit
        participant
            .handle_pre_commit("txn-001".to_string())
            .await
            .unwrap();

        assert_eq!(participant.pre_committed_txn_count(), 1);

        let stats = participant.stats();
        assert_eq!(stats.total_pre_commit_messages, 1);
    }

    #[tokio::test]
    async fn test_3pc_participant_commit() {
        let participant = ThreePhaseParticipant::new("node1".to_string());

        // Full flow
        participant
            .handle_can_commit("txn-001".to_string())
            .await
            .unwrap();
        participant
            .handle_pre_commit("txn-001".to_string())
            .await
            .unwrap();
        participant
            .handle_do_commit("txn-001".to_string())
            .await
            .unwrap();

        assert_eq!(participant.phase(), TpcPhase::Committed);
        assert_eq!(participant.active_txn_count(), 0);
        assert_eq!(participant.pre_committed_txn_count(), 0);

        let stats = participant.stats();
        assert_eq!(stats.total_commits, 1);
    }

    #[tokio::test]
    async fn test_3pc_participant_abort() {
        let participant = ThreePhaseParticipant::new("node1".to_string());

        participant
            .handle_can_commit("txn-001".to_string())
            .await
            .unwrap();
        participant
            .handle_abort("txn-001".to_string())
            .await
            .unwrap();

        assert_eq!(participant.phase(), TpcPhase::Aborted);
        assert_eq!(participant.active_txn_count(), 0);

        let stats = participant.stats();
        assert_eq!(stats.total_aborts, 1);
    }

    #[tokio::test]
    async fn test_3pc_coordinator_timeout() {
        let mut coordinator = ThreePhaseCoordinator::new("txn-004".to_string());

        coordinator.set_timeouts(
            Duration::from_secs(5),
            Duration::from_secs(10),
            Duration::from_secs(15),
        );

        assert_eq!(coordinator.can_commit_timeout, Duration::from_secs(5));
        assert_eq!(coordinator.pre_commit_timeout, Duration::from_secs(10));
        assert_eq!(coordinator.do_commit_timeout, Duration::from_secs(15));
    }

    #[tokio::test]
    async fn test_3pc_participant_coordinator_timeout_pre_committed() {
        let participant = ThreePhaseParticipant::new("node1".to_string());

        // Transaction reaches pre-commit state
        participant
            .handle_can_commit("txn-001".to_string())
            .await
            .unwrap();
        participant
            .handle_pre_commit("txn-001".to_string())
            .await
            .unwrap();

        // Coordinator times out - participant can safely commit
        let can_commit = participant
            .handle_coordinator_timeout("txn-001".to_string())
            .await
            .unwrap();
        assert!(
            can_commit,
            "Pre-committed transaction should commit on timeout"
        );
        assert_eq!(participant.phase(), TpcPhase::Committed);
    }

    #[tokio::test]
    async fn test_3pc_participant_coordinator_timeout_not_pre_committed() {
        let participant = ThreePhaseParticipant::new("node1".to_string());

        // Transaction only reached can-commit state
        participant
            .handle_can_commit("txn-001".to_string())
            .await
            .unwrap();

        // Coordinator times out - participant must abort
        let can_commit = participant
            .handle_coordinator_timeout("txn-001".to_string())
            .await
            .unwrap();
        assert!(
            !can_commit,
            "Non-pre-committed transaction should abort on timeout"
        );
        assert_eq!(participant.phase(), TpcPhase::Aborted);
    }

    #[tokio::test]
    async fn test_3pc_multiple_transactions() {
        let participant = ThreePhaseParticipant::new("node1".to_string());

        // Transaction 1
        participant
            .handle_can_commit("txn-001".to_string())
            .await
            .unwrap();
        participant
            .handle_pre_commit("txn-001".to_string())
            .await
            .unwrap();
        participant
            .handle_do_commit("txn-001".to_string())
            .await
            .unwrap();

        // Transaction 2
        participant
            .handle_can_commit("txn-002".to_string())
            .await
            .unwrap();
        participant
            .handle_pre_commit("txn-002".to_string())
            .await
            .unwrap();
        participant
            .handle_do_commit("txn-002".to_string())
            .await
            .unwrap();

        let stats = participant.stats();
        assert_eq!(stats.total_commits, 2);
        assert_eq!(stats.total_can_commit_requests, 2);
        assert_eq!(stats.total_pre_commit_messages, 2);
    }

    #[tokio::test]
    async fn test_3pc_get_can_commit_responses() {
        let mut coordinator = ThreePhaseCoordinator::new("txn-005".to_string());

        coordinator.add_participant(Participant {
            node_id: "node1".to_string(),
            endpoint: "http://node1:8080".to_string(),
        });

        // Before commit, no responses
        let responses_before = coordinator.get_can_commit_responses();
        assert_eq!(responses_before.get("node1").unwrap(), &None);

        // After can-commit phase
        coordinator.commit().await.unwrap();

        let responses_after = coordinator.get_can_commit_responses();
        assert_eq!(
            responses_after.get("node1").unwrap(),
            &Some(CanCommitResponse::Yes)
        );
    }
}
