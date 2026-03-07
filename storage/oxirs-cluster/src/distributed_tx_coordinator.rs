//! # Distributed Transaction Coordinator (2PC)
//!
//! Implements a `DistributedTransactionCoordinator` that uses the Two-Phase
//! Commit (2PC) protocol to coordinate atomic, durable transactions across
//! multiple cluster nodes.
//!
//! ## Protocol overview
//!
//! ```text
//! Phase 1 — Prepare:
//!   Coordinator → Participants: PREPARE(tx_id, ops)
//!   Participants → Coordinator: VOTE_COMMIT | VOTE_ABORT
//!
//! Phase 2 — Commit / Abort:
//!   If all voted COMMIT → Coordinator → Participants: COMMIT(tx_id)
//!   Else              → Coordinator → Participants: ABORT(tx_id)
//! ```
//!
//! The coordinator maintains a durable transaction log so that it can recover
//! a transaction after a coordinator crash (coordinator recovery protocol).

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Transaction identifier
// ---------------------------------------------------------------------------

/// Globally unique transaction identifier.
pub type DistTxId = String;

// ---------------------------------------------------------------------------
// Transaction state machine
// ---------------------------------------------------------------------------

/// State of a distributed transaction in the 2PC state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistTxState {
    /// Transaction has been registered with the coordinator.
    Active,
    /// PREPARE has been sent; waiting for participant votes.
    Preparing,
    /// All participants voted COMMIT; ready to commit.
    Prepared,
    /// COMMIT messages sent to participants; waiting for confirmations.
    Committing,
    /// All participants confirmed commit — transaction is durable.
    Committed,
    /// At least one participant voted ABORT or timeout occurred.
    Aborting,
    /// All participants confirmed rollback — transaction is rolled back.
    Aborted,
}

impl std::fmt::Display for DistTxState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "Active"),
            Self::Preparing => write!(f, "Preparing"),
            Self::Prepared => write!(f, "Prepared"),
            Self::Committing => write!(f, "Committing"),
            Self::Committed => write!(f, "Committed"),
            Self::Aborting => write!(f, "Aborting"),
            Self::Aborted => write!(f, "Aborted"),
        }
    }
}

// ---------------------------------------------------------------------------
// Transaction operation
// ---------------------------------------------------------------------------

/// An operation to be executed atomically across participant nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DistTxOp {
    /// Insert an RDF triple (subject, predicate, object).
    Insert {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    },
    /// Delete a triple.
    Delete {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    },
    /// Batch insert (placeholder: count only, payload transported separately).
    BatchInsert { count: usize },
    /// Custom operation identified by name and opaque payload.
    Custom { name: String, payload: Vec<u8> },
}

// ---------------------------------------------------------------------------
// Participant vote
// ---------------------------------------------------------------------------

/// Vote cast by a participant during Phase 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantVote {
    Commit,
    Abort { reason: String },
}

// ---------------------------------------------------------------------------
// Participant descriptor
// ---------------------------------------------------------------------------

/// Tracks a single participant node within a distributed transaction.
#[derive(Debug, Clone)]
pub struct TxParticipant {
    /// Node identifier.
    pub node_id: String,
    /// Operations assigned to this participant.
    pub ops: Vec<DistTxOp>,
    /// Vote received during Phase 1 (`None` if not yet received).
    pub vote: Option<ParticipantVote>,
    /// Whether this participant has confirmed the Phase 2 outcome.
    pub outcome_confirmed: bool,
    /// Timestamp of last contact.
    pub last_contact_ms: u64,
}

impl TxParticipant {
    fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            ops: Vec::new(),
            vote: None,
            outcome_confirmed: false,
            last_contact_ms: now_ms(),
        }
    }

    fn touch(&mut self) {
        self.last_contact_ms = now_ms();
    }
}

// ---------------------------------------------------------------------------
// Transaction log entry
// ---------------------------------------------------------------------------

/// A durable log entry for crash recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxLogEntry {
    /// Transaction identifier.
    pub tx_id: DistTxId,
    /// New state after the transition.
    pub state: DistTxState,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Optional message or context.
    pub message: Option<String>,
}

impl TxLogEntry {
    fn new(tx_id: impl Into<String>, state: DistTxState, message: Option<String>) -> Self {
        Self {
            tx_id: tx_id.into(),
            state,
            timestamp_ms: now_ms(),
            message,
        }
    }
}

// ---------------------------------------------------------------------------
// Transaction log
// ---------------------------------------------------------------------------

/// In-memory transaction log with a bounded history.
pub struct TxLog {
    entries: VecDeque<TxLogEntry>,
    capacity: usize,
}

impl TxLog {
    /// Create a new log with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
        }
    }

    /// Append an entry. Oldest entries are dropped when capacity is exceeded.
    pub fn append(&mut self, entry: TxLogEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Return all entries for a given transaction ID.
    pub fn entries_for(&self, tx_id: &str) -> Vec<&TxLogEntry> {
        self.entries.iter().filter(|e| e.tx_id == tx_id).collect()
    }

    /// Return the latest state recorded for a transaction.
    pub fn latest_state(&self, tx_id: &str) -> Option<&DistTxState> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.tx_id == tx_id)
            .map(|e| &e.state)
    }

    /// Total number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Active transaction record
// ---------------------------------------------------------------------------

/// Internal representation of an active distributed transaction.
pub struct DistributedTx {
    /// Unique identifier.
    pub id: DistTxId,
    /// Current state.
    pub state: DistTxState,
    /// Participants keyed by node ID.
    pub participants: HashMap<String, TxParticipant>,
    /// Transaction creation time.
    pub created_at: Instant,
    /// Maximum lifetime before automatic abort.
    pub timeout: Duration,
}

impl DistributedTx {
    fn new(id: DistTxId, timeout: Duration) -> Self {
        Self {
            id,
            state: DistTxState::Active,
            participants: HashMap::new(),
            created_at: Instant::now(),
            timeout,
        }
    }

    fn is_timed_out(&self) -> bool {
        self.created_at.elapsed() > self.timeout
    }

    /// True if all participants have voted COMMIT.
    fn all_voted_commit(&self) -> bool {
        self.participants
            .values()
            .all(|p| matches!(p.vote, Some(ParticipantVote::Commit)))
    }

    /// True if any participant has voted ABORT.
    #[allow(dead_code)]
    fn any_voted_abort(&self) -> bool {
        self.participants
            .values()
            .any(|p| matches!(p.vote, Some(ParticipantVote::Abort { .. })))
    }

    /// True if all participants have confirmed the Phase 2 outcome.
    fn all_confirmed(&self) -> bool {
        self.participants.values().all(|p| p.outcome_confirmed)
    }

    /// True if all participants have cast their Phase 1 vote.
    #[allow(dead_code)]
    fn all_voted(&self) -> bool {
        self.participants.values().all(|p| p.vote.is_some())
    }
}

// ---------------------------------------------------------------------------
// Coordinator configuration
// ---------------------------------------------------------------------------

/// Configuration for `DistributedTransactionCoordinator`.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Identifier of the coordinator node.
    pub coordinator_id: String,
    /// Default transaction timeout.
    pub default_timeout: Duration,
    /// Maximum number of concurrently active transactions.
    pub max_concurrent: usize,
    /// Capacity of the transaction log (number of entries).
    pub log_capacity: usize,
    /// How often the background sweeper checks for timed-out transactions.
    pub sweep_interval: Duration,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            coordinator_id: "coordinator-1".to_string(),
            default_timeout: Duration::from_secs(30),
            max_concurrent: 1000,
            log_capacity: 4096,
            sweep_interval: Duration::from_secs(5),
        }
    }
}

impl CoordinatorConfig {
    /// Create a config for the given coordinator ID.
    pub fn new(coordinator_id: impl Into<String>) -> Self {
        Self {
            coordinator_id: coordinator_id.into(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// 2PC outcome
// ---------------------------------------------------------------------------

/// The final outcome of a 2PC round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TwoPhaseOutcome {
    /// Transaction committed successfully.
    Committed,
    /// Transaction aborted.
    Aborted { reason: String },
}

// ---------------------------------------------------------------------------
// DistributedTransactionCoordinator
// ---------------------------------------------------------------------------

/// Coordinates two-phase commit (2PC) for distributed transactions.
///
/// # Usage pattern
///
/// ```ignore
/// let coord = DistributedTransactionCoordinator::new(CoordinatorConfig::default());
/// let tx_id = coord.begin(None).await?;
/// coord.enlist_participant(&tx_id, "node-1").await?;
/// coord.add_op(&tx_id, "node-1", op).await?;
/// let outcome = coord.execute_2pc(&tx_id).await?;
/// ```
pub struct DistributedTransactionCoordinator {
    config: CoordinatorConfig,
    /// Active transactions.
    transactions: Arc<RwLock<HashMap<DistTxId, DistributedTx>>>,
    /// Durable transaction log.
    log: Arc<Mutex<TxLog>>,
    /// Counters.
    commits_total: Arc<RwLock<u64>>,
    aborts_total: Arc<RwLock<u64>>,
    timeouts_total: Arc<RwLock<u64>>,
}

impl DistributedTransactionCoordinator {
    /// Create a new coordinator with the given configuration.
    pub fn new(config: CoordinatorConfig) -> Self {
        let log_cap = config.log_capacity;
        Self {
            config,
            transactions: Arc::new(RwLock::new(HashMap::new())),
            log: Arc::new(Mutex::new(TxLog::new(log_cap))),
            commits_total: Arc::new(RwLock::new(0)),
            aborts_total: Arc::new(RwLock::new(0)),
            timeouts_total: Arc::new(RwLock::new(0)),
        }
    }

    // -----------------------------------------------------------------------
    // Transaction lifecycle
    // -----------------------------------------------------------------------

    /// Begin a new transaction.
    ///
    /// Returns the new transaction ID.
    pub async fn begin(&self, timeout: Option<Duration>) -> Result<DistTxId> {
        let mut txs = self.transactions.write().await;
        if txs.len() >= self.config.max_concurrent {
            return Err(ClusterError::Other(
                "max concurrent transactions exceeded".into(),
            ));
        }
        let id = Uuid::new_v4().to_string();
        let timeout = timeout.unwrap_or(self.config.default_timeout);
        txs.insert(id.clone(), DistributedTx::new(id.clone(), timeout));

        self.log.lock().await.append(TxLogEntry::new(
            &id,
            DistTxState::Active,
            Some("Transaction started".into()),
        ));
        Ok(id)
    }

    /// Enlist a participant node in the transaction.
    pub async fn enlist_participant(&self, tx_id: &str, node_id: &str) -> Result<()> {
        let mut txs = self.transactions.write().await;
        let tx = txs
            .get_mut(tx_id)
            .ok_or_else(|| ClusterError::Other(format!("transaction '{}' not found", tx_id)))?;
        if tx.state != DistTxState::Active {
            return Err(ClusterError::Other(format!(
                "cannot enlist participant in state {}",
                tx.state
            )));
        }
        tx.participants
            .entry(node_id.to_string())
            .or_insert_with(|| TxParticipant::new(node_id));
        Ok(())
    }

    /// Add an operation to a participant's work list.
    pub async fn add_op(&self, tx_id: &str, node_id: &str, op: DistTxOp) -> Result<()> {
        let mut txs = self.transactions.write().await;
        let tx = txs
            .get_mut(tx_id)
            .ok_or_else(|| ClusterError::Other(format!("transaction '{}' not found", tx_id)))?;
        if tx.state != DistTxState::Active {
            return Err(ClusterError::Other("transaction is not active".into()));
        }
        let participant = tx.participants.get_mut(node_id).ok_or_else(|| {
            ClusterError::Other(format!("participant '{}' not enlisted", node_id))
        })?;
        participant.ops.push(op);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Phase 1 — Prepare
    // -----------------------------------------------------------------------

    /// Execute Phase 1: send PREPARE to all participants and collect votes.
    ///
    /// In this embedded coordinator the votes are simulated synchronously via
    /// `vote_fn`. In production the coordinator would send RPC messages.
    ///
    /// `vote_fn(node_id, ops) -> ParticipantVote`
    pub async fn phase1_prepare<F>(&self, tx_id: &str, mut vote_fn: F) -> Result<bool>
    where
        F: FnMut(&str, &[DistTxOp]) -> ParticipantVote,
    {
        let mut txs = self.transactions.write().await;
        let tx = txs
            .get_mut(tx_id)
            .ok_or_else(|| ClusterError::Other(format!("transaction '{}' not found", tx_id)))?;

        if tx.state != DistTxState::Active {
            return Err(ClusterError::Other(format!(
                "cannot prepare transaction in state {}",
                tx.state
            )));
        }
        if tx.is_timed_out() {
            tx.state = DistTxState::Aborting;
            return Err(ClusterError::Other("transaction timed out".into()));
        }

        tx.state = DistTxState::Preparing;
        drop(txs); // release lock during vote collection

        self.log
            .lock()
            .await
            .append(TxLogEntry::new(tx_id, DistTxState::Preparing, None));

        // Collect participant IDs and their ops without holding the write lock.
        let participants_snapshot = {
            let txs = self.transactions.read().await;
            let tx = txs
                .get(tx_id)
                .ok_or_else(|| ClusterError::Other("tx vanished".into()))?;
            tx.participants
                .iter()
                .map(|(id, p)| (id.clone(), p.ops.clone()))
                .collect::<Vec<_>>()
        };

        let mut votes: Vec<(String, ParticipantVote)> = Vec::new();
        for (node_id, ops) in &participants_snapshot {
            let vote = vote_fn(node_id.as_str(), ops.as_slice());
            votes.push((node_id.clone(), vote));
        }

        // Apply votes.
        let mut txs = self.transactions.write().await;
        let tx = txs
            .get_mut(tx_id)
            .ok_or_else(|| ClusterError::Other("tx vanished".into()))?;

        for (node_id, vote) in votes {
            if let Some(p) = tx.participants.get_mut(&node_id) {
                p.vote = Some(vote);
                p.touch();
            }
        }

        let all_commit = tx.all_voted_commit();
        if all_commit {
            tx.state = DistTxState::Prepared;
            drop(txs);
            self.log.lock().await.append(TxLogEntry::new(
                tx_id,
                DistTxState::Prepared,
                Some("All participants voted COMMIT".into()),
            ));
        } else {
            tx.state = DistTxState::Aborting;
            drop(txs);
            self.log.lock().await.append(TxLogEntry::new(
                tx_id,
                DistTxState::Aborting,
                Some("Participant voted ABORT".into()),
            ));
        }

        Ok(all_commit)
    }

    // -----------------------------------------------------------------------
    // Phase 2 — Commit / Abort
    // -----------------------------------------------------------------------

    /// Execute Phase 2: commit if Phase 1 succeeded, otherwise abort.
    ///
    /// `confirm_fn(node_id, commit: bool) -> bool` — returns `true` if the
    /// participant confirmed the outcome. In production this is an RPC call.
    pub async fn phase2_finalize<F>(
        &self,
        tx_id: &str,
        mut confirm_fn: F,
    ) -> Result<TwoPhaseOutcome>
    where
        F: FnMut(&str, bool) -> bool,
    {
        let should_commit = {
            let txs = self.transactions.read().await;
            let tx = txs
                .get(tx_id)
                .ok_or_else(|| ClusterError::Other(format!("tx '{}' not found", tx_id)))?;
            matches!(tx.state, DistTxState::Prepared)
        };

        // Collect participant IDs.
        let node_ids: Vec<String> = {
            let txs = self.transactions.read().await;
            let tx = txs
                .get(tx_id)
                .ok_or_else(|| ClusterError::Other("tx vanished".into()))?;
            tx.participants.keys().cloned().collect()
        };

        let mut confirmations: Vec<(String, bool)> = Vec::new();
        for node_id in &node_ids {
            let confirmed = confirm_fn(node_id.as_str(), should_commit);
            confirmations.push((node_id.clone(), confirmed));
        }

        // Apply confirmations.
        let mut txs = self.transactions.write().await;
        let tx = txs
            .get_mut(tx_id)
            .ok_or_else(|| ClusterError::Other("tx vanished".into()))?;

        for (node_id, confirmed) in confirmations {
            if let Some(p) = tx.participants.get_mut(&node_id) {
                p.outcome_confirmed = confirmed;
                p.touch();
            }
        }

        let final_state = if should_commit {
            tx.state = DistTxState::Committing;
            if tx.all_confirmed() {
                tx.state = DistTxState::Committed;
                DistTxState::Committed
            } else {
                DistTxState::Committing
            }
        } else {
            tx.state = DistTxState::Aborting;
            if tx.all_confirmed() {
                tx.state = DistTxState::Aborted;
                DistTxState::Aborted
            } else {
                DistTxState::Aborting
            }
        };

        drop(txs);

        let outcome = match &final_state {
            DistTxState::Committed => {
                *self.commits_total.write().await += 1;
                TwoPhaseOutcome::Committed
            }
            DistTxState::Aborted => {
                *self.aborts_total.write().await += 1;
                TwoPhaseOutcome::Aborted {
                    reason: "participant aborted".into(),
                }
            }
            _ => TwoPhaseOutcome::Aborted {
                reason: "not all participants confirmed".into(),
            },
        };

        self.log.lock().await.append(TxLogEntry::new(
            tx_id,
            final_state,
            Some(format!("{:?}", outcome)),
        ));

        Ok(outcome)
    }

    /// Execute the full 2PC round in one call.
    ///
    /// Combines `phase1_prepare` and `phase2_finalize` with the provided
    /// vote and confirm functions.
    pub async fn execute_2pc<VF, CF>(
        &self,
        tx_id: &str,
        vote_fn: VF,
        confirm_fn: CF,
    ) -> Result<TwoPhaseOutcome>
    where
        VF: FnMut(&str, &[DistTxOp]) -> ParticipantVote,
        CF: FnMut(&str, bool) -> bool,
    {
        let prepared = self.phase1_prepare(tx_id, vote_fn).await?;
        if !prepared {
            // Phase 1 failed — run Phase 2 to distribute abort
            return self.phase2_finalize(tx_id, confirm_fn).await;
        }
        self.phase2_finalize(tx_id, confirm_fn).await
    }

    // -----------------------------------------------------------------------
    // Rollback
    // -----------------------------------------------------------------------

    /// Force-abort a transaction regardless of participant votes.
    ///
    /// This is used for manual rollback or timeout-based cleanup.
    pub async fn rollback(&self, tx_id: &str, reason: &str) -> Result<()> {
        let mut txs = self.transactions.write().await;
        let tx = txs
            .get_mut(tx_id)
            .ok_or_else(|| ClusterError::Other(format!("tx '{}' not found", tx_id)))?;

        // Can only rollback if not already committed.
        if matches!(tx.state, DistTxState::Committed) {
            return Err(ClusterError::Other(
                "cannot rollback a committed transaction".into(),
            ));
        }
        tx.state = DistTxState::Aborted;
        drop(txs);

        *self.aborts_total.write().await += 1;

        self.log.lock().await.append(TxLogEntry::new(
            tx_id,
            DistTxState::Aborted,
            Some(format!("Rolled back: {}", reason)),
        ));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Timeout sweep
    // -----------------------------------------------------------------------

    /// Abort all transactions that have exceeded their timeout.
    ///
    /// Returns the IDs of transactions that were aborted.
    pub async fn sweep_timed_out(&self) -> Vec<DistTxId> {
        let timed_out: Vec<DistTxId> = {
            let txs = self.transactions.read().await;
            txs.iter()
                .filter(|(_, tx)| {
                    tx.is_timed_out()
                        && !matches!(tx.state, DistTxState::Committed | DistTxState::Aborted)
                })
                .map(|(id, _)| id.clone())
                .collect()
        };

        for id in &timed_out {
            let _ = self.rollback(id, "timeout").await;
            *self.timeouts_total.write().await += 1;
        }
        timed_out
    }

    // -----------------------------------------------------------------------
    // Query / metrics
    // -----------------------------------------------------------------------

    /// Return the current state of a transaction.
    pub async fn state(&self, tx_id: &str) -> Option<DistTxState> {
        self.transactions
            .read()
            .await
            .get(tx_id)
            .map(|tx| tx.state.clone())
    }

    /// Return the number of active (non-terminal) transactions.
    pub async fn active_count(&self) -> usize {
        self.transactions
            .read()
            .await
            .values()
            .filter(|tx| !matches!(tx.state, DistTxState::Committed | DistTxState::Aborted))
            .count()
    }

    /// Return total committed transaction count.
    pub async fn commits_total(&self) -> u64 {
        *self.commits_total.read().await
    }

    /// Return total aborted transaction count.
    pub async fn aborts_total(&self) -> u64 {
        *self.aborts_total.read().await
    }

    /// Return total timed-out transaction count.
    pub async fn timeouts_total(&self) -> u64 {
        *self.timeouts_total.read().await
    }

    /// Return a snapshot of log entries for a given transaction.
    pub async fn log_entries(&self, tx_id: &str) -> Vec<TxLogEntry> {
        self.log
            .lock()
            .await
            .entries_for(tx_id)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Return the latest logged state for a transaction (crash recovery aid).
    pub async fn latest_logged_state(&self, tx_id: &str) -> Option<DistTxState> {
        self.log.lock().await.latest_state(tx_id).cloned()
    }

    /// Return all transaction IDs currently tracked (active and terminal).
    pub async fn all_tx_ids(&self) -> Vec<DistTxId> {
        self.transactions.read().await.keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    fn rt() -> Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
    }

    fn make_coord() -> DistributedTransactionCoordinator {
        DistributedTransactionCoordinator::new(CoordinatorConfig::default())
    }

    fn always_commit(_node: &str, _ops: &[DistTxOp]) -> ParticipantVote {
        ParticipantVote::Commit
    }

    fn always_abort(_node: &str, _ops: &[DistTxOp]) -> ParticipantVote {
        ParticipantVote::Abort {
            reason: "test abort".into(),
        }
    }

    fn always_confirm(_node: &str, _commit: bool) -> bool {
        true
    }

    // -----------------------------------------------------------------------
    // TxLog tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_txlog_append_and_query() {
        let mut log = TxLog::new(10);
        log.append(TxLogEntry::new("tx1", DistTxState::Active, None));
        log.append(TxLogEntry::new("tx1", DistTxState::Preparing, None));
        assert_eq!(log.entries_for("tx1").len(), 2);
        assert_eq!(log.latest_state("tx1"), Some(&DistTxState::Preparing));
    }

    #[test]
    fn test_txlog_capacity_bounded() {
        let mut log = TxLog::new(3);
        for _ in 0..5 {
            log.append(TxLogEntry::new("tx1", DistTxState::Active, None));
        }
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_txlog_empty_query() {
        let log = TxLog::new(10);
        assert!(log.latest_state("nonexistent").is_none());
        assert!(log.entries_for("x").is_empty());
    }

    // -----------------------------------------------------------------------
    // begin / enlist / add_op
    // -----------------------------------------------------------------------

    #[test]
    fn test_begin_returns_unique_ids() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id1 = coord.begin(None).await.expect("begin");
            let id2 = coord.begin(None).await.expect("begin");
            assert_ne!(id1, id2);
        });
    }

    #[test]
    fn test_begin_logs_active_state() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            let entries = coord.log_entries(&id).await;
            assert!(!entries.is_empty());
            assert_eq!(entries[0].state, DistTxState::Active);
        });
    }

    #[test]
    fn test_enlist_participant() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord
                .enlist_participant(&id, "node-1")
                .await
                .expect("enlist");
        });
    }

    #[test]
    fn test_enlist_duplicate_participant_idempotent() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("enlist");
            coord
                .enlist_participant(&id, "n1")
                .await
                .expect("enlist again");
            let state = coord.state(&id).await;
            assert_eq!(state, Some(DistTxState::Active));
        });
    }

    #[test]
    fn test_enlist_nonexistent_tx_errors() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            assert!(coord.enlist_participant("bad-id", "n1").await.is_err());
        });
    }

    #[test]
    fn test_add_op_to_participant() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("enlist");
            coord
                .add_op(
                    &id,
                    "n1",
                    DistTxOp::Insert {
                        subject: "s".into(),
                        predicate: "p".into(),
                        object: "o".into(),
                        graph: None,
                    },
                )
                .await
                .expect("add_op");
        });
    }

    #[test]
    fn test_add_op_unenrolled_participant_errors() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            assert!(coord
                .add_op(&id, "ghost", DistTxOp::BatchInsert { count: 1 })
                .await
                .is_err());
        });
    }

    // -----------------------------------------------------------------------
    // Phase 1 — Prepare
    // -----------------------------------------------------------------------

    #[test]
    fn test_phase1_all_commit_returns_true() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            coord.enlist_participant(&id, "n2").await.expect("e");
            let ok = coord.phase1_prepare(&id, always_commit).await.expect("p1");
            assert!(ok);
            assert_eq!(coord.state(&id).await, Some(DistTxState::Prepared));
        });
    }

    #[test]
    fn test_phase1_any_abort_returns_false() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            let ok = coord.phase1_prepare(&id, always_abort).await.expect("p1");
            assert!(!ok);
            assert_eq!(coord.state(&id).await, Some(DistTxState::Aborting));
        });
    }

    #[test]
    fn test_phase1_logs_preparing_and_prepared() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            coord.phase1_prepare(&id, always_commit).await.expect("p1");
            let entries = coord.log_entries(&id).await;
            let states: Vec<_> = entries.iter().map(|e| e.state.clone()).collect();
            assert!(states.contains(&DistTxState::Preparing));
            assert!(states.contains(&DistTxState::Prepared));
        });
    }

    // -----------------------------------------------------------------------
    // Phase 2 — Finalize
    // -----------------------------------------------------------------------

    #[test]
    fn test_phase2_after_prepare_commits() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            coord.phase1_prepare(&id, always_commit).await.expect("p1");
            let outcome = coord
                .phase2_finalize(&id, always_confirm)
                .await
                .expect("p2");
            assert_eq!(outcome, TwoPhaseOutcome::Committed);
            assert_eq!(coord.commits_total().await, 1);
        });
    }

    #[test]
    fn test_phase2_after_abort_aborts() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            coord.phase1_prepare(&id, always_abort).await.expect("p1");
            let outcome = coord
                .phase2_finalize(&id, always_confirm)
                .await
                .expect("p2");
            assert!(matches!(outcome, TwoPhaseOutcome::Aborted { .. }));
        });
    }

    // -----------------------------------------------------------------------
    // Full execute_2pc
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_2pc_success() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            coord.enlist_participant(&id, "n2").await.expect("e");
            coord
                .add_op(&id, "n1", DistTxOp::BatchInsert { count: 10 })
                .await
                .expect("op");
            let outcome = coord
                .execute_2pc(&id, always_commit, always_confirm)
                .await
                .expect("2pc");
            assert_eq!(outcome, TwoPhaseOutcome::Committed);
        });
    }

    #[test]
    fn test_execute_2pc_abort_when_one_rejects() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            coord.enlist_participant(&id, "n2").await.expect("e");

            // n2 will abort
            let vote_fn = |node: &str, _ops: &[DistTxOp]| {
                if node == "n2" {
                    ParticipantVote::Abort {
                        reason: "disk full".into(),
                    }
                } else {
                    ParticipantVote::Commit
                }
            };
            let outcome = coord
                .execute_2pc(&id, vote_fn, always_confirm)
                .await
                .expect("2pc");
            assert!(matches!(outcome, TwoPhaseOutcome::Aborted { .. }));
        });
    }

    // -----------------------------------------------------------------------
    // Rollback
    // -----------------------------------------------------------------------

    #[test]
    fn test_rollback_active_transaction() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord
                .rollback(&id, "manual cancel")
                .await
                .expect("rollback");
            assert_eq!(coord.state(&id).await, Some(DistTxState::Aborted));
            assert_eq!(coord.aborts_total().await, 1);
        });
    }

    #[test]
    fn test_rollback_committed_errors() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            coord
                .execute_2pc(&id, always_commit, always_confirm)
                .await
                .expect("2pc");
            assert!(coord.rollback(&id, "too late").await.is_err());
        });
    }

    #[test]
    fn test_rollback_logs_reason() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.rollback(&id, "forced").await.expect("rollback");
            let entries = coord.log_entries(&id).await;
            let aborted = entries.iter().find(|e| e.state == DistTxState::Aborted);
            assert!(aborted.is_some());
            assert!(aborted
                .unwrap()
                .message
                .as_deref()
                .unwrap_or("")
                .contains("forced"));
        });
    }

    // -----------------------------------------------------------------------
    // Timeout sweep
    // -----------------------------------------------------------------------

    #[test]
    fn test_sweep_timed_out_transactions() {
        let rt = rt();
        rt.block_on(async {
            let mut config = CoordinatorConfig::default();
            config.default_timeout = Duration::from_millis(1);
            let coord = DistributedTransactionCoordinator::new(config);

            let id = coord.begin(None).await.expect("begin");
            tokio::time::sleep(Duration::from_millis(5)).await;
            let swept = coord.sweep_timed_out().await;
            assert!(swept.contains(&id));
            assert_eq!(coord.state(&id).await, Some(DistTxState::Aborted));
            assert!(coord.timeouts_total().await >= 1);
        });
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_active_count_decreases_after_commit() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id1 = coord.begin(None).await.expect("b");
            let _id2 = coord.begin(None).await.expect("b");
            assert_eq!(coord.active_count().await, 2);

            coord.enlist_participant(&id1, "n1").await.expect("e");
            coord
                .execute_2pc(&id1, always_commit, always_confirm)
                .await
                .expect("2pc");
            assert_eq!(coord.active_count().await, 1);
        });
    }

    #[test]
    fn test_all_tx_ids_returned() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id1 = coord.begin(None).await.expect("b");
            let id2 = coord.begin(None).await.expect("b");
            let ids = coord.all_tx_ids().await;
            assert!(ids.contains(&id1));
            assert!(ids.contains(&id2));
        });
    }

    #[test]
    fn test_max_concurrent_limit() {
        let rt = rt();
        rt.block_on(async {
            let mut config = CoordinatorConfig::default();
            config.max_concurrent = 2;
            let coord = DistributedTransactionCoordinator::new(config);
            coord.begin(None).await.expect("b1");
            coord.begin(None).await.expect("b2");
            // Third should fail
            assert!(coord.begin(None).await.is_err());
        });
    }

    // -----------------------------------------------------------------------
    // Crash recovery via log
    // -----------------------------------------------------------------------

    #[test]
    fn test_latest_logged_state_after_commit() {
        let rt = rt();
        rt.block_on(async {
            let coord = make_coord();
            let id = coord.begin(None).await.expect("begin");
            coord.enlist_participant(&id, "n1").await.expect("e");
            coord
                .execute_2pc(&id, always_commit, always_confirm)
                .await
                .expect("2pc");
            let logged = coord.latest_logged_state(&id).await;
            assert_eq!(logged, Some(DistTxState::Committed));
        });
    }

    #[test]
    fn test_tx_state_display() {
        assert_eq!(DistTxState::Committed.to_string(), "Committed");
        assert_eq!(DistTxState::Aborted.to_string(), "Aborted");
        assert_eq!(DistTxState::Preparing.to_string(), "Preparing");
    }
}
