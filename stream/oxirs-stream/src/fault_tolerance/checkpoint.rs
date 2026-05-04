//! # Chandy-Lamport-style checkpointing for stream operators
//!
//! Implements a marker-based snapshot algorithm in the spirit of the original
//! Chandy-Lamport protocol, adapted to stream processing topologies.
//!
//! ## Protocol summary
//!
//! 1. The coordinator broadcasts a [`Marker`] with a unique
//!    [`CheckpointId`] to every *source* operator.
//! 2. When an operator receives a marker on **input edge `e`**:
//!    a. If this is the first marker observed in this checkpoint, the operator
//!    (i) snapshots its local state, (ii) emits the marker on every output
//!    edge, (iii) starts recording the messages that arrive on every other
//!    input edge.
//!    b. Otherwise, the operator stops recording on `e` and writes the
//!    recorded prefix as part of the snapshot.
//! 3. When the operator has received the marker on every input edge, the
//!    snapshot is complete.
//! 4. Snapshots are persisted via [`CheckpointStore`] (the production
//!    implementation forwards them to the cluster snapshot store).
//!
//! ## Marker propagation
//!
//! [`MarkerPropagator`] tracks the per-operator marker state and surfaces a
//! [`MarkerPropagatorEvent`] every time a checkpoint completes for an
//! operator. A [`CheckpointController`] builds on top of the propagator to
//! coordinate a global checkpoint.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info};

// ─── Identifiers ────────────────────────────────────────────────────────────

/// Unique identifier of a checkpoint round.
pub type CheckpointId = u64;

/// Identifier of an operator participating in a checkpoint.
pub type OperatorId = String;

/// Identifier of an input edge feeding into an operator.
pub type InputEdgeId = String;

// ─── Marker ─────────────────────────────────────────────────────────────────

/// Chandy-Lamport marker.
///
/// Markers carry the [`CheckpointId`] and a wall-clock timestamp. They flow
/// in-band with regular events on every operator edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Marker {
    pub checkpoint_id: CheckpointId,
    pub emitted_at_ms: u64,
}

impl Marker {
    /// Build a marker with the current wall-clock time.
    pub fn new(checkpoint_id: CheckpointId) -> Self {
        Self {
            checkpoint_id,
            emitted_at_ms: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }
}

// ─── Snapshot ──────────────────────────────────────────────────────────────

/// State snapshot produced by a single operator during a checkpoint round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorSnapshot {
    pub operator_id: OperatorId,
    pub checkpoint_id: CheckpointId,
    /// Opaque serialised state.
    pub state_blob: Vec<u8>,
    /// Pre-snapshot in-flight events recorded per input edge after the marker
    /// arrived on the *first* edge but before it arrived on others.
    pub channel_logs: HashMap<InputEdgeId, Vec<Vec<u8>>>,
    /// Wall-clock time at which the snapshot completed (Unix ms).
    pub completed_at_ms: u64,
}

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by the checkpointing pipeline.
#[derive(Debug, Error)]
pub enum CheckpointError {
    /// Marker for an unknown operator.
    #[error("unknown operator: {0}")]
    UnknownOperator(OperatorId),
    /// Marker for an unknown input edge on an operator.
    #[error("unknown edge {edge} on operator {op}")]
    UnknownEdge { op: OperatorId, edge: InputEdgeId },
    /// Internal store error.
    #[error("store error: {0}")]
    Store(String),
}

/// Convenience alias.
pub type CheckpointResult<T> = std::result::Result<T, CheckpointError>;

// ─── CheckpointStore ───────────────────────────────────────────────────────

/// Where completed snapshots are persisted.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Persist a single operator snapshot.
    async fn put(&self, snap: OperatorSnapshot) -> CheckpointResult<()>;

    /// Load a snapshot by `(operator, checkpoint)` pair, if any.
    async fn get(
        &self,
        operator: &OperatorId,
        checkpoint: CheckpointId,
    ) -> CheckpointResult<Option<OperatorSnapshot>>;

    /// Latest committed checkpoint id, if any. Used by the recovery path to
    /// pick the most recent global snapshot.
    async fn latest(&self) -> CheckpointResult<Option<CheckpointId>>;
}

/// In-memory snapshot store used in tests and as a default for embedded
/// deployments. Production deployments swap this for an
/// `oxirs-cluster`-backed snapshot store.
pub struct InMemoryCheckpointStore {
    inner: RwLock<HashMap<(OperatorId, CheckpointId), OperatorSnapshot>>,
    latest: RwLock<Option<CheckpointId>>,
}

impl Default for InMemoryCheckpointStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryCheckpointStore {
    /// Build an empty store.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            latest: RwLock::new(None),
        }
    }

    /// Total number of snapshots currently in the store.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// True when no snapshots are stored.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

#[async_trait]
impl CheckpointStore for InMemoryCheckpointStore {
    async fn put(&self, snap: OperatorSnapshot) -> CheckpointResult<()> {
        let cp = snap.checkpoint_id;
        self.inner
            .write()
            .insert((snap.operator_id.clone(), cp), snap);
        let mut latest = self.latest.write();
        if latest.map_or(true, |old| cp > old) {
            *latest = Some(cp);
        }
        Ok(())
    }

    async fn get(
        &self,
        operator: &OperatorId,
        checkpoint: CheckpointId,
    ) -> CheckpointResult<Option<OperatorSnapshot>> {
        Ok(self
            .inner
            .read()
            .get(&(operator.clone(), checkpoint))
            .cloned())
    }

    async fn latest(&self) -> CheckpointResult<Option<CheckpointId>> {
        Ok(*self.latest.read())
    }
}

// ─── MarkerPropagator ──────────────────────────────────────────────────────

/// Per-operator state used by the marker propagator.
#[derive(Debug, Default)]
struct OperatorMarkerState {
    /// Set of input edges that have observed the marker so far.
    seen_on: HashSet<InputEdgeId>,
    /// Set of *all* configured input edges.
    expected: HashSet<InputEdgeId>,
    /// Recorded pre-snapshot messages keyed by input edge.
    channel_logs: HashMap<InputEdgeId, Vec<Vec<u8>>>,
}

impl OperatorMarkerState {
    fn new(expected: HashSet<InputEdgeId>) -> Self {
        Self {
            seen_on: HashSet::new(),
            expected,
            channel_logs: HashMap::new(),
        }
    }
}

/// Event emitted by [`MarkerPropagator::on_marker`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkerPropagatorEvent {
    /// First marker observed for this checkpoint id; the operator should
    /// snapshot its state and emit the marker on every output edge.
    StartSnapshot,
    /// Subsequent marker on a different edge; the recording for that edge is
    /// closed.
    EdgeClosed,
    /// Marker observed on every input edge: the operator's checkpoint round is
    /// complete.
    Completed,
}

/// Tracks marker arrival per (operator, edge) pair.
pub struct MarkerPropagator {
    state: RwLock<HashMap<(OperatorId, CheckpointId), OperatorMarkerState>>,
    expected_edges: RwLock<HashMap<OperatorId, HashSet<InputEdgeId>>>,
}

impl Default for MarkerPropagator {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkerPropagator {
    /// Build an empty propagator.
    pub fn new() -> Self {
        Self {
            state: RwLock::new(HashMap::new()),
            expected_edges: RwLock::new(HashMap::new()),
        }
    }

    /// Register an operator with the set of input edges it expects to see
    /// markers on.
    pub fn register_operator(
        &self,
        op: OperatorId,
        edges: impl IntoIterator<Item = impl Into<InputEdgeId>>,
    ) {
        let edges: HashSet<InputEdgeId> = edges.into_iter().map(Into::into).collect();
        self.expected_edges.write().insert(op, edges);
    }

    /// Returns true if the operator is registered.
    pub fn is_registered(&self, op: &OperatorId) -> bool {
        self.expected_edges.read().contains_key(op)
    }

    /// Record an in-flight event arriving on `edge` for an operator that has
    /// already started recording for the given checkpoint. Events are stored
    /// in the order they arrive.
    pub fn record_inflight(
        &self,
        op: &OperatorId,
        checkpoint: CheckpointId,
        edge: &InputEdgeId,
        payload: Vec<u8>,
    ) -> CheckpointResult<bool> {
        let mut states = self.state.write();
        let st = match states.get_mut(&(op.clone(), checkpoint)) {
            Some(s) => s,
            None => return Ok(false),
        };
        if st.seen_on.is_empty() {
            // No marker yet — nothing to record.
            return Ok(false);
        }
        if st.seen_on.contains(edge) {
            // Marker already seen on this edge: do not record.
            return Ok(false);
        }
        st.channel_logs
            .entry(edge.clone())
            .or_default()
            .push(payload);
        Ok(true)
    }

    /// Process a marker arrival on `edge` for `operator`.
    pub fn on_marker(
        &self,
        op: &OperatorId,
        edge: &InputEdgeId,
        marker: &Marker,
    ) -> CheckpointResult<MarkerPropagatorEvent> {
        let expected = {
            let edges = self.expected_edges.read();
            edges
                .get(op)
                .ok_or_else(|| CheckpointError::UnknownOperator(op.clone()))?
                .clone()
        };
        if !expected.contains(edge) {
            return Err(CheckpointError::UnknownEdge {
                op: op.clone(),
                edge: edge.clone(),
            });
        }
        let mut states = self.state.write();
        let entry = states
            .entry((op.clone(), marker.checkpoint_id))
            .or_insert_with(|| OperatorMarkerState::new(expected.clone()));

        let event = if entry.seen_on.is_empty() {
            entry.seen_on.insert(edge.clone());
            MarkerPropagatorEvent::StartSnapshot
        } else if entry.seen_on.contains(edge) {
            MarkerPropagatorEvent::EdgeClosed
        } else {
            entry.seen_on.insert(edge.clone());
            MarkerPropagatorEvent::EdgeClosed
        };

        let completed = entry.seen_on == entry.expected;
        if completed {
            Ok(MarkerPropagatorEvent::Completed)
        } else {
            Ok(event)
        }
    }

    /// Drain the recorded channel logs for an operator's checkpoint round.
    pub fn drain_channel_logs(
        &self,
        op: &OperatorId,
        checkpoint: CheckpointId,
    ) -> HashMap<InputEdgeId, Vec<Vec<u8>>> {
        let mut states = self.state.write();
        match states.remove(&(op.clone(), checkpoint)) {
            Some(s) => s.channel_logs,
            None => HashMap::new(),
        }
    }

    /// Reset all per-checkpoint state for an operator.
    ///
    /// Clears every checkpoint round currently tracked for `op`. The
    /// operator registration (its expected input edges) is preserved so the
    /// next checkpoint round can resume immediately.
    pub fn reset(&self, op: &OperatorId) {
        let mut states = self.state.write();
        states.retain(|(o, _), _| o != op);
    }
}

// ─── CheckpointController ──────────────────────────────────────────────────

/// Configuration for [`CheckpointController`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointControllerConfig {
    /// How often to issue a new checkpoint.
    pub interval: Duration,
    /// Maximum time an operator has to commit its snapshot before it is
    /// considered failed (the controller will issue a fresh checkpoint).
    pub timeout: Duration,
}

impl Default for CheckpointControllerConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(60),
        }
    }
}

/// Per-checkpoint completion progress.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckpointProgress {
    pub checkpoint_id: CheckpointId,
    /// Set of operators that have committed a snapshot for this round.
    pub committed: HashSet<OperatorId>,
    /// Total expected operators.
    pub expected: HashSet<OperatorId>,
    pub started_at_ms: u64,
    pub completed_at_ms: Option<u64>,
}

impl CheckpointProgress {
    pub fn is_complete(&self) -> bool {
        !self.expected.is_empty() && self.committed == self.expected
    }
}

/// Coordinator that drives checkpoint rounds across the operator topology.
pub struct CheckpointController {
    config: CheckpointControllerConfig,
    propagator: Arc<MarkerPropagator>,
    store: Arc<dyn CheckpointStore>,
    next_id: AtomicU64,
    /// Set of operators participating.
    operators: RwLock<HashSet<OperatorId>>,
    /// Round-by-round progress.
    progress: RwLock<HashMap<CheckpointId, CheckpointProgress>>,
}

impl CheckpointController {
    /// Build a controller.
    pub fn new(
        config: CheckpointControllerConfig,
        propagator: Arc<MarkerPropagator>,
        store: Arc<dyn CheckpointStore>,
    ) -> Self {
        Self {
            config,
            propagator,
            store,
            next_id: AtomicU64::new(1),
            operators: RwLock::new(HashSet::new()),
            progress: RwLock::new(HashMap::new()),
        }
    }

    /// Configuration accessor.
    pub fn config(&self) -> &CheckpointControllerConfig {
        &self.config
    }

    /// Marker propagator handle (so operator code can re-use the same
    /// propagator).
    pub fn propagator(&self) -> &Arc<MarkerPropagator> {
        &self.propagator
    }

    /// Register an operator with the controller.
    pub fn register_operator(&self, op: OperatorId) {
        self.operators.write().insert(op);
    }

    /// Open a new checkpoint round.
    pub fn open(&self) -> Marker {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let marker = Marker::new(id);
        let expected = self.operators.read().clone();
        self.progress.write().insert(
            id,
            CheckpointProgress {
                checkpoint_id: id,
                committed: HashSet::new(),
                expected,
                started_at_ms: marker.emitted_at_ms,
                completed_at_ms: None,
            },
        );
        debug!(checkpoint_id = id, "checkpoint controller: opened");
        marker
    }

    /// Acknowledge a snapshot from an operator.
    pub async fn commit_snapshot(&self, snapshot: OperatorSnapshot) -> CheckpointResult<bool> {
        let cp = snapshot.checkpoint_id;
        let op = snapshot.operator_id.clone();
        self.store.put(snapshot).await?;
        let mut progress = self.progress.write();
        if let Some(p) = progress.get_mut(&cp) {
            p.committed.insert(op.clone());
            if p.is_complete() && p.completed_at_ms.is_none() {
                p.completed_at_ms = Some(now_ms());
                info!(checkpoint_id = cp, "checkpoint complete");
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Fetch the current progress snapshot for a checkpoint.
    pub fn progress(&self, cp: CheckpointId) -> Option<CheckpointProgress> {
        self.progress.read().get(&cp).cloned()
    }

    /// Drop progress for an old checkpoint (used when retiring rounds).
    pub fn forget(&self, cp: CheckpointId) {
        self.progress.write().remove(&cp);
    }

    /// Latest checkpoint id known to the underlying store.
    pub async fn latest_committed(&self) -> CheckpointResult<Option<CheckpointId>> {
        self.store.latest().await
    }

    /// Number of rounds opened by the controller so far.
    pub fn opened_rounds(&self) -> u64 {
        self.next_id.load(Ordering::Relaxed).saturating_sub(1)
    }

    /// Snapshot store reference for callers that need to load on recovery.
    pub fn store(&self) -> &Arc<dyn CheckpointStore> {
        &self.store
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn op(name: &str) -> OperatorId {
        name.to_string()
    }

    fn edge(name: &str) -> InputEdgeId {
        name.to_string()
    }

    #[tokio::test]
    async fn marker_propagator_completes_on_all_edges() {
        let prop = MarkerPropagator::new();
        prop.register_operator(op("op1"), ["e1", "e2"]);
        let marker = Marker::new(1);

        let ev1 = prop
            .on_marker(&op("op1"), &edge("e1"), &marker)
            .expect("ok");
        assert_eq!(ev1, MarkerPropagatorEvent::StartSnapshot);
        let ev2 = prop
            .on_marker(&op("op1"), &edge("e2"), &marker)
            .expect("ok");
        assert_eq!(ev2, MarkerPropagatorEvent::Completed);
    }

    #[test]
    fn marker_propagator_records_inflight_only_after_first_marker() {
        let prop = MarkerPropagator::new();
        prop.register_operator(op("op1"), ["e1", "e2"]);
        let marker = Marker::new(2);
        // No marker yet → no recording.
        let ok = prop
            .record_inflight(&op("op1"), 2, &edge("e2"), b"early".to_vec())
            .expect("ok");
        assert!(!ok);
        // First marker on e1.
        let _ = prop
            .on_marker(&op("op1"), &edge("e1"), &marker)
            .expect("ok");
        // Now record on e2.
        let ok = prop
            .record_inflight(&op("op1"), 2, &edge("e2"), b"after-marker".to_vec())
            .expect("ok");
        assert!(ok);
        // Marker on e1 already seen → recording on e1 is suppressed.
        let ok = prop
            .record_inflight(&op("op1"), 2, &edge("e1"), b"x".to_vec())
            .expect("ok");
        assert!(!ok);
    }

    #[test]
    fn marker_propagator_reports_unknown_operator() {
        let prop = MarkerPropagator::new();
        let err = prop
            .on_marker(&op("ghost"), &edge("e1"), &Marker::new(1))
            .expect_err("should fail");
        assert!(matches!(err, CheckpointError::UnknownOperator(_)));
    }

    #[test]
    fn marker_propagator_reports_unknown_edge() {
        let prop = MarkerPropagator::new();
        prop.register_operator(op("op1"), ["e1"]);
        let err = prop
            .on_marker(&op("op1"), &edge("e2"), &Marker::new(1))
            .expect_err("should fail");
        assert!(matches!(err, CheckpointError::UnknownEdge { .. }));
    }

    #[tokio::test]
    async fn controller_drives_full_round() {
        let propagator = Arc::new(MarkerPropagator::new());
        let store = Arc::new(InMemoryCheckpointStore::new());
        let controller = CheckpointController::new(
            CheckpointControllerConfig::default(),
            propagator.clone(),
            store.clone(),
        );
        controller.register_operator(op("op-a"));
        controller.register_operator(op("op-b"));
        propagator.register_operator(op("op-a"), ["src"]);
        propagator.register_operator(op("op-b"), ["a"]);
        let marker = controller.open();

        propagator
            .on_marker(&op("op-a"), &edge("src"), &marker)
            .expect("ok");
        let snap_a = OperatorSnapshot {
            operator_id: op("op-a"),
            checkpoint_id: marker.checkpoint_id,
            state_blob: vec![1, 2, 3],
            channel_logs: HashMap::new(),
            completed_at_ms: now_ms(),
        };
        let done = controller.commit_snapshot(snap_a).await.expect("ok");
        assert!(!done);

        propagator
            .on_marker(&op("op-b"), &edge("a"), &marker)
            .expect("ok");
        let snap_b = OperatorSnapshot {
            operator_id: op("op-b"),
            checkpoint_id: marker.checkpoint_id,
            state_blob: vec![9, 9],
            channel_logs: HashMap::new(),
            completed_at_ms: now_ms(),
        };
        let done = controller.commit_snapshot(snap_b).await.expect("ok");
        assert!(done);

        let prog = controller.progress(marker.checkpoint_id).expect("progress");
        assert!(prog.is_complete());
        let latest = controller.latest_committed().await.expect("ok");
        assert_eq!(latest, Some(marker.checkpoint_id));
    }

    #[tokio::test]
    async fn store_round_trip() {
        let store = InMemoryCheckpointStore::new();
        let snap = OperatorSnapshot {
            operator_id: op("op1"),
            checkpoint_id: 7,
            state_blob: vec![1],
            channel_logs: HashMap::new(),
            completed_at_ms: now_ms(),
        };
        store.put(snap.clone()).await.expect("put");
        let back = store.get(&snap.operator_id, 7).await.expect("get");
        assert_eq!(back.expect("hit").operator_id, op("op1"));
        let latest = store.latest().await.expect("latest");
        assert_eq!(latest, Some(7));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn controller_opens_unique_ids() {
        let propagator = Arc::new(MarkerPropagator::new());
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let controller =
            CheckpointController::new(CheckpointControllerConfig::default(), propagator, store);
        let m1 = controller.open();
        let m2 = controller.open();
        assert_ne!(m1.checkpoint_id, m2.checkpoint_id);
        assert_eq!(controller.opened_rounds(), 2);
    }
}
