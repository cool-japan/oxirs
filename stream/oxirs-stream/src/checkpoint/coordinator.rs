//! # Chandy-Lamport Checkpoint Coordinator
//!
//! Distributed checkpoint coordinator inspired by the Chandy-Lamport algorithm
//! for consistent global snapshots.  Enables failure recovery by periodically
//! snapshotting all operator states.
//!
//! ## How it works
//!
//! 1. The coordinator decides it is time for a new checkpoint.
//! 2. It broadcasts a *barrier marker* to every registered operator (modelled
//!    here as the `initiate()` call returning a `checkpoint_id` that is
//!    forwarded to operators out-of-band).
//! 3. Each operator:
//!    a. Drains its in-flight channel messages.
//!    b. Serialises its state via its `StateBackend`.
//!    c. Calls `coordinator.operator_reported(snapshot)`.
//! 4. Once all operators have acknowledged the coordinator declares the
//!    checkpoint *complete* and stores the `GlobalCheckpoint`.
//! 5. On recovery the coordinator replays the latest `GlobalCheckpoint`.

use crate::error::StreamError;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

// ─── Types ────────────────────────────────────────────────────────────────────

/// Lifecycle phase of the checkpoint coordinator.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointPhase {
    /// No checkpoint in progress.
    Idle,
    /// Coordinator sent barrier markers; waiting for operators to respond.
    InProgress {
        checkpoint_id: u64,
        started_at: Instant,
        acked_operators: HashSet<String>,
    },
    /// All operators have reported; checkpoint is permanently stored.
    Completed {
        checkpoint_id: u64,
        completed_at: Instant,
        size_bytes: usize,
    },
    /// Checkpoint was abandoned.
    Failed { checkpoint_id: u64, reason: String },
}

impl CheckpointPhase {
    /// Return the checkpoint ID for phases that have one.
    pub fn checkpoint_id(&self) -> Option<u64> {
        match self {
            Self::Idle => None,
            Self::InProgress { checkpoint_id, .. } => Some(*checkpoint_id),
            Self::Completed { checkpoint_id, .. } => Some(*checkpoint_id),
            Self::Failed { checkpoint_id, .. } => Some(*checkpoint_id),
        }
    }
}

/// Snapshot of a single operator's state at checkpoint time.
#[derive(Debug, Clone)]
pub struct OperatorSnapshot {
    /// ID of the operator that produced this snapshot.
    pub operator_id: String,
    /// The checkpoint this snapshot belongs to.
    pub checkpoint_id: u64,
    /// Serialised state bytes (format is operator-specific).
    pub state_bytes: Vec<u8>,
    /// In-flight messages that were in the channel when the barrier arrived.
    pub in_flight_messages: Vec<Vec<u8>>,
    /// Wall-clock time at snapshot creation.
    pub created_at: Instant,
    /// Total byte size for budget tracking.
    pub size_bytes: usize,
}

impl OperatorSnapshot {
    /// Create a new operator snapshot, auto-computing `size_bytes`.
    pub fn new(
        operator_id: impl Into<String>,
        checkpoint_id: u64,
        state_bytes: Vec<u8>,
        in_flight_messages: Vec<Vec<u8>>,
    ) -> Self {
        let in_flight_size: usize = in_flight_messages.iter().map(|m| m.len()).sum();
        let size_bytes = state_bytes.len() + in_flight_size;
        Self {
            operator_id: operator_id.into(),
            checkpoint_id,
            state_bytes,
            in_flight_messages,
            created_at: Instant::now(),
            size_bytes,
        }
    }
}

/// Consistent global snapshot across all operators in the streaming job.
#[derive(Debug, Clone)]
pub struct GlobalCheckpoint {
    pub checkpoint_id: u64,
    pub operator_snapshots: HashMap<String, OperatorSnapshot>,
    pub created_at: Instant,
    pub total_size_bytes: usize,
    /// Stream read offsets at the time of the checkpoint.
    /// Maps `stream_id → offset`, allowing the job to replay exactly from here.
    pub stream_positions: HashMap<String, u64>,
}

impl GlobalCheckpoint {
    /// Create an empty global checkpoint container.
    pub fn new(checkpoint_id: u64) -> Self {
        Self {
            checkpoint_id,
            operator_snapshots: HashMap::new(),
            created_at: Instant::now(),
            total_size_bytes: 0,
            stream_positions: HashMap::new(),
        }
    }

    /// Add an operator snapshot to this global checkpoint.
    pub fn add_operator_snapshot(&mut self, snapshot: OperatorSnapshot) {
        self.total_size_bytes += snapshot.size_bytes;
        self.operator_snapshots
            .insert(snapshot.operator_id.clone(), snapshot);
    }

    /// Set the committed read offset for a stream.
    pub fn set_stream_position(&mut self, stream_id: impl Into<String>, offset: u64) {
        self.stream_positions.insert(stream_id.into(), offset);
    }

    /// Returns `true` when every expected operator has contributed a snapshot.
    pub fn is_complete(&self, expected_operators: &[String]) -> bool {
        expected_operators
            .iter()
            .all(|op_id| self.operator_snapshots.contains_key(op_id))
    }

    /// Total byte size of all operator state in this checkpoint.
    pub fn total_bytes(&self) -> usize {
        self.total_size_bytes
    }
}

// ─── Coordinator ─────────────────────────────────────────────────────────────

/// Central checkpoint coordinator.
///
/// Manages the lifecycle of periodic checkpoints: scheduling, barrier
/// initiation, acknowledgement collection, and storage of completed snapshots.
pub struct CheckpointCoordinator {
    current_phase: CheckpointPhase,
    /// How often to trigger a checkpoint.
    checkpoint_interval: Duration,
    /// Wall-clock time when the last checkpoint completed (or was initiated).
    last_checkpoint: Option<Instant>,
    /// Completed checkpoints, newest last.
    completed_checkpoints: Vec<GlobalCheckpoint>,
    /// Maximum number of completed checkpoints to retain in memory.
    max_retained_checkpoints: usize,
    /// Operators that must acknowledge before a checkpoint is complete.
    registered_operators: Vec<String>,
    /// Monotonically increasing checkpoint counter.
    next_checkpoint_id: u64,
    /// In-progress global checkpoint being assembled.
    in_progress_checkpoint: Option<GlobalCheckpoint>,
    /// Timeout for operators to acknowledge before a checkpoint is aborted.
    operator_timeout: Duration,
}

impl CheckpointCoordinator {
    /// Create a coordinator with the given checkpoint interval.
    pub fn new(interval: Duration) -> Self {
        Self {
            current_phase: CheckpointPhase::Idle,
            checkpoint_interval: interval,
            last_checkpoint: None,
            completed_checkpoints: Vec::new(),
            max_retained_checkpoints: 10,
            registered_operators: Vec::new(),
            next_checkpoint_id: 1,
            in_progress_checkpoint: None,
            operator_timeout: Duration::from_secs(60),
        }
    }

    /// Override the maximum number of retained completed checkpoints.
    pub fn with_max_retained(mut self, n: usize) -> Self {
        self.max_retained_checkpoints = n;
        self
    }

    /// Override the per-operator acknowledgement timeout.
    pub fn with_operator_timeout(mut self, timeout: Duration) -> Self {
        self.operator_timeout = timeout;
        self
    }

    /// Register an operator that must participate in every checkpoint.
    pub fn register_operator(&mut self, operator_id: String) {
        if !self.registered_operators.contains(&operator_id) {
            self.registered_operators.push(operator_id);
        }
    }

    /// Register multiple operators at once.
    pub fn register_operators(&mut self, operator_ids: impl IntoIterator<Item = String>) {
        for id in operator_ids {
            self.register_operator(id);
        }
    }

    /// Returns `true` if enough time has elapsed and no checkpoint is in
    /// progress.
    pub fn should_checkpoint(&self) -> bool {
        if !matches!(self.current_phase, CheckpointPhase::Idle) {
            return false;
        }

        match self.last_checkpoint {
            None => true,
            Some(last) => last.elapsed() >= self.checkpoint_interval,
        }
    }

    /// Initiate a new checkpoint.
    ///
    /// The returned `checkpoint_id` should be forwarded to all registered
    /// operators as a barrier token.
    ///
    /// Returns an error if a checkpoint is already in progress.
    pub fn initiate(&mut self) -> Result<u64, StreamError> {
        if !matches!(self.current_phase, CheckpointPhase::Idle) {
            return Err(StreamError::InvalidOperation(format!(
                "cannot initiate checkpoint while in phase {:?}",
                self.current_phase.checkpoint_id()
            )));
        }

        let checkpoint_id = self.next_checkpoint_id;
        self.next_checkpoint_id += 1;

        self.current_phase = CheckpointPhase::InProgress {
            checkpoint_id,
            started_at: Instant::now(),
            acked_operators: HashSet::new(),
        };

        self.in_progress_checkpoint = Some(GlobalCheckpoint::new(checkpoint_id));
        self.last_checkpoint = Some(Instant::now());

        Ok(checkpoint_id)
    }

    /// Called by an operator after it has snapshotted its state.
    ///
    /// Returns `true` when all operators have acknowledged (checkpoint
    /// complete).
    pub fn operator_reported(&mut self, snapshot: OperatorSnapshot) -> Result<bool, StreamError> {
        let (checkpoint_id, started_at) = match &self.current_phase {
            CheckpointPhase::InProgress {
                checkpoint_id,
                started_at,
                ..
            } => (*checkpoint_id, *started_at),
            other => {
                return Err(StreamError::InvalidOperation(format!(
                    "operator_reported called but coordinator is in {:?} phase",
                    other.checkpoint_id()
                )));
            }
        };

        if snapshot.checkpoint_id != checkpoint_id {
            return Err(StreamError::InvalidInput(format!(
                "snapshot checkpoint_id {} does not match in-progress {}",
                snapshot.checkpoint_id, checkpoint_id
            )));
        }

        // Check operator timeout
        if started_at.elapsed() > self.operator_timeout {
            let reason = format!(
                "operator {} timed out after {:?}",
                snapshot.operator_id,
                started_at.elapsed()
            );
            self.abort(&reason);
            return Err(StreamError::Timeout(reason));
        }

        // Record acknowledgement
        let operator_id = snapshot.operator_id.clone();
        if let CheckpointPhase::InProgress {
            ref mut acked_operators,
            ..
        } = self.current_phase
        {
            acked_operators.insert(operator_id.clone());
        }

        // Accumulate into the global snapshot
        if let Some(ref mut global) = self.in_progress_checkpoint {
            global.add_operator_snapshot(snapshot);
        }

        // Check if all operators have reported
        let all_done = if let CheckpointPhase::InProgress {
            ref acked_operators,
            ..
        } = self.current_phase
        {
            self.registered_operators
                .iter()
                .all(|op| acked_operators.contains(op))
        } else {
            false
        };

        if all_done {
            self.finalize_checkpoint(checkpoint_id)?;
            return Ok(true);
        }

        Ok(false)
    }

    fn finalize_checkpoint(&mut self, checkpoint_id: u64) -> Result<(), StreamError> {
        let global = self.in_progress_checkpoint.take().ok_or_else(|| {
            StreamError::Other("in_progress_checkpoint missing at finalize".into())
        })?;

        let size_bytes = global.total_size_bytes;

        self.completed_checkpoints.push(global);

        // Trim retained checkpoints
        while self.completed_checkpoints.len() > self.max_retained_checkpoints {
            self.completed_checkpoints.remove(0);
        }

        self.current_phase = CheckpointPhase::Completed {
            checkpoint_id,
            completed_at: Instant::now(),
            size_bytes,
        };

        Ok(())
    }

    /// Abort the in-progress checkpoint, transitioning back to `Idle`.
    pub fn abort(&mut self, reason: &str) {
        let checkpoint_id = self.current_phase.checkpoint_id().unwrap_or(0);
        self.in_progress_checkpoint = None;
        self.current_phase = CheckpointPhase::Failed {
            checkpoint_id,
            reason: reason.to_string(),
        };
    }

    /// Reset from a Failed or Completed phase back to Idle.
    pub fn reset_to_idle(&mut self) {
        match self.current_phase {
            CheckpointPhase::Completed { .. } | CheckpointPhase::Failed { .. } => {
                self.current_phase = CheckpointPhase::Idle;
            }
            _ => {}
        }
    }

    /// Return a reference to the most recent completed checkpoint.
    pub fn latest_checkpoint(&self) -> Option<&GlobalCheckpoint> {
        self.completed_checkpoints.last()
    }

    /// Return a reference to a completed checkpoint by ID.
    pub fn get_checkpoint(&self, id: u64) -> Option<&GlobalCheckpoint> {
        self.completed_checkpoints
            .iter()
            .find(|cp| cp.checkpoint_id == id)
    }

    /// Number of retained completed checkpoints.
    pub fn completed_count(&self) -> usize {
        self.completed_checkpoints.len()
    }

    /// The checkpoint ID currently in progress, if any.
    pub fn current_checkpoint_id(&self) -> Option<u64> {
        match &self.current_phase {
            CheckpointPhase::InProgress { checkpoint_id, .. } => Some(*checkpoint_id),
            _ => None,
        }
    }

    /// Current phase (for diagnostics).
    pub fn phase(&self) -> &CheckpointPhase {
        &self.current_phase
    }

    /// Pending operator acknowledgements for the in-progress checkpoint.
    ///
    /// Returns `None` if no checkpoint is in progress.
    pub fn pending_operators(&self) -> Option<Vec<String>> {
        if let CheckpointPhase::InProgress {
            ref acked_operators,
            ..
        } = self.current_phase
        {
            let pending: Vec<String> = self
                .registered_operators
                .iter()
                .filter(|op| !acked_operators.contains(*op))
                .cloned()
                .collect();
            Some(pending)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(operator_id: &str, checkpoint_id: u64, state: &[u8]) -> OperatorSnapshot {
        OperatorSnapshot::new(operator_id, checkpoint_id, state.to_vec(), vec![])
    }

    // ── CheckpointPhase helpers ───────────────────────────────────────────────

    #[test]
    fn test_phase_checkpoint_id() {
        let idle = CheckpointPhase::Idle;
        assert_eq!(idle.checkpoint_id(), None);

        let in_progress = CheckpointPhase::InProgress {
            checkpoint_id: 7,
            started_at: Instant::now(),
            acked_operators: HashSet::new(),
        };
        assert_eq!(in_progress.checkpoint_id(), Some(7));
    }

    // ── GlobalCheckpoint ─────────────────────────────────────────────────────

    #[test]
    fn test_global_checkpoint_completeness() {
        let mut cp = GlobalCheckpoint::new(1);
        let ops = vec!["op_a".to_string(), "op_b".to_string()];

        assert!(!cp.is_complete(&ops));

        cp.add_operator_snapshot(make_snapshot("op_a", 1, b"state_a"));
        assert!(!cp.is_complete(&ops));

        cp.add_operator_snapshot(make_snapshot("op_b", 1, b"state_b"));
        assert!(cp.is_complete(&ops));
    }

    #[test]
    fn test_global_checkpoint_bytes() {
        let mut cp = GlobalCheckpoint::new(1);
        cp.add_operator_snapshot(make_snapshot("op_a", 1, &[0u8; 100]));
        cp.add_operator_snapshot(make_snapshot("op_b", 1, &[0u8; 200]));
        assert_eq!(cp.total_bytes(), 300);
    }

    // ── CheckpointCoordinator ─────────────────────────────────────────────────

    #[test]
    fn test_should_checkpoint_when_no_last() {
        let coord = CheckpointCoordinator::new(Duration::from_secs(60));
        assert!(coord.should_checkpoint()); // No last checkpoint
    }

    #[test]
    fn test_should_not_checkpoint_when_in_progress() {
        let mut coord = CheckpointCoordinator::new(Duration::from_secs(60));
        coord.register_operator("op1".to_string());
        coord.initiate().unwrap();
        assert!(!coord.should_checkpoint());
    }

    #[test]
    fn test_initiate_returns_incrementing_ids() {
        let mut coord = CheckpointCoordinator::new(Duration::from_millis(0));
        coord.register_operator("op1".to_string());

        let id1 = coord.initiate().unwrap();
        assert_eq!(id1, 1);
        assert_eq!(coord.current_checkpoint_id(), Some(1));

        // Report so we can initiate again
        let snap = make_snapshot("op1", 1, b"state");
        coord.operator_reported(snap).unwrap();
        coord.reset_to_idle();

        let id2 = coord.initiate().unwrap();
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_single_operator_full_lifecycle() {
        let mut coord = CheckpointCoordinator::new(Duration::from_secs(300));
        coord.register_operator("worker".to_string());

        assert!(coord.should_checkpoint());

        let cp_id = coord.initiate().unwrap();
        assert_eq!(cp_id, 1);

        let snap = make_snapshot("worker", cp_id, b"my_state_data");
        let complete = coord.operator_reported(snap).unwrap();
        assert!(complete);

        assert_eq!(coord.completed_count(), 1);
        let latest = coord.latest_checkpoint().unwrap();
        assert_eq!(latest.checkpoint_id, 1);
        assert!(latest.operator_snapshots.contains_key("worker"));
        assert_eq!(latest.total_bytes(), 13); // "my_state_data"

        coord.reset_to_idle();
        assert!(!coord.should_checkpoint()); // interval hasn't elapsed
    }

    #[test]
    fn test_multi_operator_checkpoint() {
        let mut coord = CheckpointCoordinator::new(Duration::from_secs(300));
        coord.register_operators(["op_a".to_string(), "op_b".to_string(), "op_c".to_string()]);

        let cp_id = coord.initiate().unwrap();

        // First two operators report → not complete yet
        let not_done = coord
            .operator_reported(make_snapshot("op_a", cp_id, b"state_a"))
            .unwrap();
        assert!(!not_done);

        let not_done2 = coord
            .operator_reported(make_snapshot("op_b", cp_id, b"state_b"))
            .unwrap();
        assert!(!not_done2);

        // Pending operators should be just "op_c"
        let pending = coord.pending_operators().unwrap();
        assert_eq!(pending, vec!["op_c".to_string()]);

        // Last operator reports → complete
        let done = coord
            .operator_reported(make_snapshot("op_c", cp_id, b"state_c"))
            .unwrap();
        assert!(done);

        let cp = coord.get_checkpoint(cp_id).unwrap();
        assert_eq!(cp.operator_snapshots.len(), 3);
    }

    #[test]
    fn test_abort_checkpoint() {
        let mut coord = CheckpointCoordinator::new(Duration::from_secs(300));
        coord.register_operator("op".to_string());

        coord.initiate().unwrap();
        coord.abort("operator crashed");

        assert!(matches!(coord.phase(), CheckpointPhase::Failed { .. }));
        assert_eq!(coord.completed_count(), 0);

        coord.reset_to_idle();
        assert!(matches!(coord.phase(), CheckpointPhase::Idle));
    }

    #[test]
    fn test_max_retained_checkpoints() {
        let mut coord = CheckpointCoordinator::new(Duration::from_millis(0)).with_max_retained(3);
        coord.register_operator("op".to_string());

        for _ in 0..5 {
            coord.initiate().unwrap();
            let cp_id = coord.current_checkpoint_id().unwrap();
            coord
                .operator_reported(make_snapshot("op", cp_id, b"s"))
                .unwrap();
            coord.reset_to_idle();
        }

        assert_eq!(coord.completed_count(), 3);
        // The latest should be checkpoint 5
        assert_eq!(coord.latest_checkpoint().unwrap().checkpoint_id, 5);
    }

    #[test]
    fn test_wrong_checkpoint_id_rejected() {
        let mut coord = CheckpointCoordinator::new(Duration::from_secs(300));
        coord.register_operator("op".to_string());

        coord.initiate().unwrap(); // checkpoint_id = 1

        // Report with wrong ID
        let snap = make_snapshot("op", 999, b"state");
        let result = coord.operator_reported(snap);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_initiate_fails() {
        let mut coord = CheckpointCoordinator::new(Duration::from_secs(300));
        coord.register_operator("op".to_string());

        coord.initiate().unwrap();
        let result = coord.initiate();
        assert!(result.is_err());
    }

    #[test]
    fn test_get_checkpoint_by_id() {
        let mut coord = CheckpointCoordinator::new(Duration::from_millis(0));
        coord.register_operator("op".to_string());

        for _ in 0..3 {
            coord.initiate().unwrap();
            let cp_id = coord.current_checkpoint_id().unwrap();
            coord
                .operator_reported(make_snapshot("op", cp_id, b"s"))
                .unwrap();
            coord.reset_to_idle();
        }

        assert!(coord.get_checkpoint(1).is_some());
        assert!(coord.get_checkpoint(2).is_some());
        assert!(coord.get_checkpoint(3).is_some());
        assert!(coord.get_checkpoint(99).is_none());
    }

    #[test]
    fn test_operator_snapshot_size() {
        let state = vec![0u8; 500];
        let in_flight = vec![vec![0u8; 100], vec![0u8; 50]];
        let snap = OperatorSnapshot::new("op", 1, state, in_flight);
        assert_eq!(snap.size_bytes, 650);
    }

    #[test]
    fn test_stream_positions() {
        let mut cp = GlobalCheckpoint::new(1);
        cp.set_stream_position("topic-A", 1024);
        cp.set_stream_position("topic-B", 2048);

        assert_eq!(cp.stream_positions.get("topic-A"), Some(&1024));
        assert_eq!(cp.stream_positions.get("topic-B"), Some(&2048));
    }
}
