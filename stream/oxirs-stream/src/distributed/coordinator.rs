//! # Distributed stream coordinator
//!
//! Top-level entrypoint for distributed stream processing. Glues together:
//!
//! * [`super::shard_manager::ShardManager`] — owns the shard → node mapping.
//! * [`oxirs_cluster::consensus::ConsensusManager`] — Raft-tracks the
//!   committed assignment so every cluster node can replay it locally.
//! * [`super::event_shipper::EventShipper`] — ships incoming events to the
//!   node that owns the relevant shard.
//!
//! ## Lifecycle
//!
//! 1. The coordinator is constructed on every node with the same
//!    [`CoordinatorConfig`] (in particular, the same `n_shards` value).
//! 2. The local node calls [`DistributedStreamCoordinator::join`] to register
//!    itself.
//! 3. The coordinator computes a balanced rebalance plan and persists the new
//!    assignment as a Raft proposal so all nodes converge on the same view.
//! 4. As stream events arrive, [`DistributedStreamCoordinator::route`]
//!    deterministically selects the destination shard / node and either
//!    delivers locally or hands the event to the [`EventShipper`].
//!
//! ## Raft propagation
//!
//! The committed assignment is encoded into the same `RdfCommand::Insert`
//! mechanism used by the W3-S9 cluster sink. We use synthetic triples in the
//! `oxirs://stream-coord/{coord_id}/...` namespace so the proposal is durable
//! and replayable but does not collide with regular RDF data.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info};

use oxirs_cluster::stream_integration::{StreamMessage, StreamTriple};
use oxirs_cluster::streaming::cluster_sink::{SinkError, StreamSink};

use super::event_shipper::{EventShipper, ShippedEvent, ShipperError};
use super::shard_manager::{
    NodeId, RebalancePlan, ShardAssignment, ShardId, ShardManager, ShardManagerConfig,
    ShardManagerError,
};

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by [`DistributedStreamCoordinator`].
#[derive(Debug, Error)]
pub enum CoordinatorError {
    /// Shard manager rejected the operation.
    #[error(transparent)]
    Shard(#[from] ShardManagerError),
    /// Sink rejected the assignment proposal.
    #[error("sink error: {0}")]
    Sink(String),
    /// Shipper failed to deliver an event.
    #[error(transparent)]
    Shipper(#[from] ShipperError),
    /// Encoding the assignment payload failed.
    #[error("encoding error: {0}")]
    Encoding(String),
    /// Routing was attempted but no nodes are registered.
    #[error("no nodes registered")]
    NoNodes,
}

impl From<SinkError> for CoordinatorError {
    fn from(err: SinkError) -> Self {
        CoordinatorError::Sink(err.to_string())
    }
}

/// Convenience alias.
pub type CoordinatorResult<T> = std::result::Result<T, CoordinatorError>;

// ─── RoutedEvent ────────────────────────────────────────────────────────────

/// Output of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutedEvent {
    /// Destination shard.
    pub shard: ShardId,
    /// Destination node id.
    pub node: NodeId,
    /// `true` when the destination is the local node.
    pub local: bool,
}

// ─── Stats ─────────────────────────────────────────────────────────────────

/// Runtime statistics for [`DistributedStreamCoordinator`].
#[derive(Debug, Default)]
pub struct CoordinatorStats {
    pub join_proposals: AtomicU64,
    pub leave_proposals: AtomicU64,
    pub routed: AtomicU64,
    pub locally_delivered: AtomicU64,
    pub remote_shipped: AtomicU64,
    pub assignment_installs: AtomicU64,
    pub failed_proposals: AtomicU64,
}

impl CoordinatorStats {
    /// Plain serialisable snapshot of the counters.
    pub fn snapshot(&self) -> CoordinatorStatsSnapshot {
        CoordinatorStatsSnapshot {
            join_proposals: self.join_proposals.load(Ordering::Relaxed),
            leave_proposals: self.leave_proposals.load(Ordering::Relaxed),
            routed: self.routed.load(Ordering::Relaxed),
            locally_delivered: self.locally_delivered.load(Ordering::Relaxed),
            remote_shipped: self.remote_shipped.load(Ordering::Relaxed),
            assignment_installs: self.assignment_installs.load(Ordering::Relaxed),
            failed_proposals: self.failed_proposals.load(Ordering::Relaxed),
        }
    }
}

/// Plain snapshot of [`CoordinatorStats`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoordinatorStatsSnapshot {
    pub join_proposals: u64,
    pub leave_proposals: u64,
    pub routed: u64,
    pub locally_delivered: u64,
    pub remote_shipped: u64,
    pub assignment_installs: u64,
    pub failed_proposals: u64,
}

// ─── Coordinator ───────────────────────────────────────────────────────────

/// Configuration for [`DistributedStreamCoordinator`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Shared coordinator identifier; the same value must be used on every
    /// node of a single distributed pipeline.
    pub coord_id: String,
    /// Identity of the local node.
    pub local_node: NodeId,
    /// Number of shards across the topology.
    pub n_shards: u32,
    /// Optional stream id used for the Raft proposals (defaults to
    /// `"stream-coord-{coord_id}"`).
    pub stream_id: Option<String>,
    /// If `true`, the coordinator skips shard-manager rebalances on
    /// duplicate joins/leaves instead of returning an error.
    pub idempotent_membership: bool,
}

impl CoordinatorConfig {
    fn stream_id(&self) -> String {
        self.stream_id
            .clone()
            .unwrap_or_else(|| format!("stream-coord-{}", self.coord_id))
    }

    fn assignment_subject(&self) -> String {
        format!("oxirs://stream-coord/{}/assignment", self.coord_id)
    }
}

/// Distributed stream coordinator.
pub struct DistributedStreamCoordinator {
    config: CoordinatorConfig,
    shard_mgr: ShardManager,
    sink: Arc<dyn StreamSink>,
    shipper: Arc<EventShipper>,
    /// Last applied assignment (mirrors `shard_mgr.current_assignment`).
    last_assignment: RwLock<ShardAssignment>,
    proposal_offset: AtomicU64,
    stats: Arc<CoordinatorStats>,
}

impl DistributedStreamCoordinator {
    /// Build a coordinator.
    pub fn new(
        config: CoordinatorConfig,
        sink: Arc<dyn StreamSink>,
        shipper: Arc<EventShipper>,
    ) -> CoordinatorResult<Self> {
        let shard_mgr = ShardManager::new(ShardManagerConfig {
            n_shards: config.n_shards,
        })?;
        Ok(Self {
            config,
            shard_mgr,
            sink,
            shipper,
            last_assignment: RwLock::new(ShardAssignment::default()),
            proposal_offset: AtomicU64::new(0),
            stats: Arc::new(CoordinatorStats::default()),
        })
    }

    /// Stats accessor.
    pub fn stats(&self) -> &Arc<CoordinatorStats> {
        &self.stats
    }

    /// Shard manager handle.
    pub fn shard_manager(&self) -> &ShardManager {
        &self.shard_mgr
    }

    /// Latest installed assignment.
    pub fn current_assignment(&self) -> ShardAssignment {
        self.last_assignment.read().clone()
    }

    /// Local node identifier.
    pub fn local_node(&self) -> &NodeId {
        &self.config.local_node
    }

    /// Number of shards configured for the coordinator.
    pub fn n_shards(&self) -> u32 {
        self.config.n_shards
    }

    /// Register a node with the coordinator (typically the local node when
    /// the cluster boots, then any peer nodes as they appear).
    pub async fn join(&self, node: NodeId) -> CoordinatorResult<RebalancePlan> {
        self.stats.join_proposals.fetch_add(1, Ordering::Relaxed);
        let plan = match self.shard_mgr.add_node(node.clone()) {
            Ok(plan) => plan,
            Err(ShardManagerError::NodeAlreadyExists(_)) if self.config.idempotent_membership => {
                debug!(node = %node, "join: idempotent skip");
                return Ok(RebalancePlan::default());
            }
            Err(err) => {
                self.stats.failed_proposals.fetch_add(1, Ordering::Relaxed);
                return Err(err.into());
            }
        };
        self.persist_assignment(&plan.new_assignment).await?;
        *self.last_assignment.write() = plan.new_assignment.clone();
        self.stats
            .assignment_installs
            .fetch_add(1, Ordering::Relaxed);
        info!(
            node = %node,
            moves = plan.moves.len(),
            "coordinator: node joined"
        );
        Ok(plan)
    }

    /// Deregister a node.
    pub async fn leave(&self, node: &str) -> CoordinatorResult<RebalancePlan> {
        self.stats.leave_proposals.fetch_add(1, Ordering::Relaxed);
        let plan = match self.shard_mgr.remove_node(node) {
            Ok(plan) => plan,
            Err(ShardManagerError::UnknownNode(_)) if self.config.idempotent_membership => {
                debug!(node = %node, "leave: idempotent skip");
                return Ok(RebalancePlan::default());
            }
            Err(err) => {
                self.stats.failed_proposals.fetch_add(1, Ordering::Relaxed);
                return Err(err.into());
            }
        };
        self.persist_assignment(&plan.new_assignment).await?;
        *self.last_assignment.write() = plan.new_assignment.clone();
        self.stats
            .assignment_installs
            .fetch_add(1, Ordering::Relaxed);
        info!(
            node = %node,
            moves = plan.moves.len(),
            "coordinator: node left"
        );
        Ok(plan)
    }

    /// Apply an assignment that was committed elsewhere (e.g. seen on a Raft
    /// follower). The shard manager and the coordinator's cached assignment
    /// are updated to match. No new Raft proposal is issued.
    pub fn install_assignment(&self, assignment: ShardAssignment) -> RebalancePlan {
        let plan = self.shard_mgr.install_assignment(assignment.clone());
        *self.last_assignment.write() = assignment;
        self.stats
            .assignment_installs
            .fetch_add(1, Ordering::Relaxed);
        plan
    }

    /// Decide which node owns the shard for `partition_key`, ship the event
    /// there, and return a [`RoutedEvent`] describing the decision.
    pub async fn route(
        &self,
        partition_key: &str,
        payload: &serde_json::Value,
    ) -> CoordinatorResult<RoutedEvent> {
        self.stats.routed.fetch_add(1, Ordering::Relaxed);
        let assignment = self.last_assignment.read().clone();
        if assignment.map.is_empty() {
            return Err(CoordinatorError::NoNodes);
        }
        let shard = self.shard_for_key(partition_key, &assignment);
        let node = assignment
            .owner_of(shard)
            .cloned()
            .ok_or(CoordinatorError::NoNodes)?;
        let event = ShippedEvent::json(shard, partition_key, payload, &self.config.local_node)
            .map_err(|e| CoordinatorError::Encoding(e.to_string()))?;
        let local = node == self.config.local_node;
        self.shipper.ship(&node, event).await?;
        if local {
            self.stats.locally_delivered.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.remote_shipped.fetch_add(1, Ordering::Relaxed);
        }
        Ok(RoutedEvent { shard, node, local })
    }

    /// Compute the shard id for a partition key without performing any I/O.
    pub fn shard_for_key_value(&self, partition_key: &str) -> Option<ShardId> {
        let assignment = self.last_assignment.read();
        if assignment.map.is_empty() {
            None
        } else {
            Some(self.shard_for_key(partition_key, &assignment))
        }
    }

    fn shard_for_key(&self, partition_key: &str, _assignment: &ShardAssignment) -> ShardId {
        let h = fnv1a_hash(partition_key.as_bytes());
        (h % self.config.n_shards as u64) as ShardId
    }

    async fn persist_assignment(&self, assignment: &ShardAssignment) -> CoordinatorResult<()> {
        let payload = serde_json::to_string(assignment)
            .map_err(|e| CoordinatorError::Encoding(e.to_string()))?;
        let object = format!("\"{}\"", escape_quotes(&payload));
        let triple = StreamTriple::new(
            self.config.assignment_subject(),
            "http://oxirs.dev/stream-coord#assignment",
            object,
        );
        let off = self.proposal_offset.fetch_add(1, Ordering::Relaxed) + 1;
        let msg = StreamMessage::insert(self.config.stream_id(), off, vec![triple]);
        self.sink.write_batch(vec![msg]).await?;
        Ok(())
    }
}

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut h = FNV_OFFSET;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn escape_quotes(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::event_shipper::{
        InProcessShipperTransport, ShipperConfig, ShipperTransport,
    };
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use std::sync::Arc;

    #[derive(Default)]
    struct MockSink {
        commits: Mutex<Vec<Vec<StreamMessage>>>,
    }

    #[async_trait]
    impl StreamSink for MockSink {
        async fn write_batch(&self, events: Vec<StreamMessage>) -> Result<(), SinkError> {
            self.commits.lock().push(events);
            Ok(())
        }
    }

    /// Build a coordinator together with the receiver half of the local sink
    /// so the test can assert that locally-routed events arrive.
    fn make_local_coord() -> (
        Arc<DistributedStreamCoordinator>,
        tokio::sync::mpsc::Receiver<ShippedEvent>,
    ) {
        let sink = Arc::new(MockSink::default());
        let transport = Arc::new(InProcessShipperTransport::new(64));
        let shipper = Arc::new(EventShipper::new(
            ShipperConfig {
                local_node: "local".into(),
            },
            transport,
        ));
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        shipper.install_local_sink(tx);
        let cfg = CoordinatorConfig {
            coord_id: "c1".into(),
            local_node: "local".into(),
            n_shards: 8,
            stream_id: None,
            idempotent_membership: true,
        };
        let coord = Arc::new(DistributedStreamCoordinator::new(cfg, sink, shipper).expect("ok"));
        (coord, rx)
    }

    #[tokio::test]
    async fn join_persists_assignment() {
        let (coord, _rx) = make_local_coord();
        let plan = coord.join("local".into()).await.expect("join");
        assert_eq!(plan.new_assignment.n_shards(), 8);
        for owner in plan.new_assignment.map.values() {
            assert_eq!(owner, "local");
        }
        let stats = coord.stats().snapshot();
        assert_eq!(stats.assignment_installs, 1);
    }

    #[tokio::test]
    async fn route_locally_when_owner() {
        let (coord, mut rx) = make_local_coord();
        coord.join("local".into()).await.expect("join");
        let routed = coord
            .route("k1", &serde_json::json!({"x": 1}))
            .await
            .expect("ok");
        assert!(routed.local);
        assert_eq!(routed.node, "local");
        let received = rx.recv().await.expect("local delivery");
        assert_eq!(received.key, "k1");
    }

    #[tokio::test]
    async fn route_to_remote_node() {
        let sink = Arc::new(MockSink::default());
        let transport = Arc::new(InProcessShipperTransport::new(64));
        let mut rx_remote = transport.spawn_receiver("remote".into());
        let shipper = Arc::new(EventShipper::new(
            ShipperConfig {
                local_node: "local".into(),
            },
            transport.clone() as Arc<dyn ShipperTransport>,
        ));
        let cfg = CoordinatorConfig {
            coord_id: "c2".into(),
            local_node: "local".into(),
            n_shards: 4,
            stream_id: None,
            idempotent_membership: true,
        };
        let coord = DistributedStreamCoordinator::new(cfg, sink, shipper).expect("ok");
        coord.join("local".into()).await.expect("join");
        coord.join("remote".into()).await.expect("join");

        // Find a key that hashes onto "remote".
        let mut chosen = None;
        for i in 0..32 {
            let key = format!("k{i}");
            let assignment = coord.current_assignment();
            let shard = (fnv1a_hash(key.as_bytes()) % 4) as ShardId;
            if assignment.owner_of(shard).map(|s| s.as_str()) == Some("remote") {
                chosen = Some(key);
                break;
            }
        }
        let key = chosen.expect("find a remote-owned key");
        let routed = coord
            .route(&key, &serde_json::json!({"v": 7}))
            .await
            .expect("ok");
        assert_eq!(routed.node, "remote");
        let received = rx_remote.recv().await.expect("received");
        assert_eq!(received.key, key);
    }

    #[tokio::test]
    async fn idempotent_join_skips_duplicate() {
        let (coord, _rx) = make_local_coord();
        coord.join("local".into()).await.expect("first");
        let plan = coord.join("local".into()).await.expect("idempotent");
        assert_eq!(plan.moves.len(), 0);
    }

    #[tokio::test]
    async fn leave_rebalances() {
        let (coord, _rx) = make_local_coord();
        coord.join("local".into()).await.expect("ok");
        coord.join("peer".into()).await.expect("ok");
        let plan = coord.leave("peer").await.expect("ok");
        assert!(!plan.new_assignment.map.is_empty());
        assert!(!plan.new_assignment.map.values().any(|n| n == "peer"));
    }

    #[tokio::test]
    async fn route_without_join_errors() {
        let (coord, _rx) = make_local_coord();
        let err = coord
            .route("key", &serde_json::json!({}))
            .await
            .expect_err("no nodes");
        assert!(matches!(err, CoordinatorError::NoNodes));
    }

    #[test]
    fn fnv1a_is_deterministic() {
        let h1 = fnv1a_hash(b"foo");
        let h2 = fnv1a_hash(b"foo");
        assert_eq!(h1, h2);
        let h3 = fnv1a_hash(b"bar");
        assert_ne!(h1, h3);
    }

    #[tokio::test]
    async fn install_assignment_replays_remote_decision() {
        let (coord, _rx) = make_local_coord();
        coord.join("local".into()).await.expect("ok");
        // Pretend the cluster committed a new assignment that puts every shard
        // on "remote". The local coordinator should accept it without issuing
        // its own proposal.
        let new = ShardAssignment::from_vec(vec!["remote".into(); 8]);
        let plan = coord.install_assignment(new.clone());
        assert_eq!(plan.new_assignment, new);
        assert_eq!(coord.current_assignment(), new);
    }
}
