//! # Shard manager
//!
//! Tracks the assignment of stream shards to cluster nodes and produces
//! deterministic rebalance plans on node join/leave events.
//!
//! Each shard is identified by an integer in `[0, n_shards)`. A
//! [`ShardAssignment`] is a snapshot mapping `shard_id -> node_id`. The
//! manager keeps the latest assignment in memory and exposes operations to:
//!
//! * [`ShardManager::add_node`] / [`ShardManager::remove_node`] —
//!   register / deregister a node and produce a [`RebalancePlan`] describing
//!   what shard moves are needed to keep the assignment balanced.
//! * [`ShardManager::owner_of`] — look up the owning node for a shard.
//! * [`ShardManager::shards_owned_by`] — enumerate the shards a given node
//!   currently owns.
//! * [`ShardManager::current_assignment`] — clone the latest snapshot for
//!   downstream propagation (e.g. through Raft via
//!   [`super::coordinator::DistributedStreamCoordinator`]).
//!
//! The placement policy is **balanced round-robin** — when there are `K`
//! nodes and `S` shards, every node owns either `floor(S/K)` or `ceil(S/K)`
//! shards. This is deterministic for a given node ordering, so the plan a
//! coordinator commits through Raft can be replayed by every node.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::debug;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by [`ShardManager`].
#[derive(Debug, Error)]
pub enum ShardManagerError {
    /// Tried to remove a node that is not registered.
    #[error("unknown node: {0}")]
    UnknownNode(String),
    /// Tried to add a node that is already registered.
    #[error("node already registered: {0}")]
    NodeAlreadyExists(String),
    /// `n_shards` was zero, which is rejected to avoid degenerate plans.
    #[error("n_shards must be >= 1")]
    NoShards,
    /// All nodes were removed, leaving the cluster with no owners.
    #[error("no nodes available to assign shards")]
    NoNodes,
}

/// Convenience alias.
pub type ShardManagerResult<T> = std::result::Result<T, ShardManagerError>;

// ─── Types ─────────────────────────────────────────────────────────────────

/// Stable shard identifier. Always `< n_shards`.
pub type ShardId = u32;

/// Stable node identifier (logical). The mapping to physical addresses is
/// outside this module's responsibility.
pub type NodeId = String;

/// Snapshot of the shard → node mapping.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardAssignment {
    /// Map from shard id to node id.
    pub map: BTreeMap<ShardId, NodeId>,
}

impl ShardAssignment {
    /// Build an assignment from a flat vector indexed by shard id.
    pub fn from_vec(nodes_per_shard: Vec<NodeId>) -> Self {
        let map = nodes_per_shard
            .into_iter()
            .enumerate()
            .map(|(i, n)| (i as ShardId, n))
            .collect();
        Self { map }
    }

    /// Total number of shards.
    pub fn n_shards(&self) -> usize {
        self.map.len()
    }

    /// Owner of a given shard.
    pub fn owner_of(&self, shard: ShardId) -> Option<&NodeId> {
        self.map.get(&shard)
    }

    /// Counts shards per owner.
    pub fn counts(&self) -> HashMap<NodeId, usize> {
        let mut counts = HashMap::new();
        for owner in self.map.values() {
            *counts.entry(owner.clone()).or_insert(0) += 1;
        }
        counts
    }
}

/// A single shard reassignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardMove {
    pub shard: ShardId,
    pub from: Option<NodeId>,
    pub to: NodeId,
}

/// Result of a rebalance: the new full assignment plus the per-shard moves
/// that need to take effect to transition from the old assignment to the new
/// one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebalancePlan {
    /// The new assignment (after applying `moves`).
    pub new_assignment: ShardAssignment,
    /// Ordered list of shard moves.
    pub moves: Vec<ShardMove>,
}

// ─── ShardManager ───────────────────────────────────────────────────────────

/// Configuration for [`ShardManager`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManagerConfig {
    /// Number of shards in the topology.
    pub n_shards: u32,
}

impl Default for ShardManagerConfig {
    fn default() -> Self {
        Self { n_shards: 8 }
    }
}

/// Tracks shard ownership and produces rebalance plans.
pub struct ShardManager {
    config: ShardManagerConfig,
    /// Sorted list of registered nodes. `BTreeMap` preserves order so the
    /// rebalance is deterministic.
    nodes: RwLock<BTreeMap<NodeId, NodeMeta>>,
    /// Latest assignment snapshot.
    assignment: RwLock<ShardAssignment>,
    /// Number of plans produced so far.
    plans_emitted: Arc<RwLock<u64>>,
}

impl std::fmt::Debug for ShardManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardManager")
            .field("config", &self.config)
            .field("nodes", &self.nodes.read().keys().collect::<Vec<_>>())
            .field("plans_emitted", &*self.plans_emitted.read())
            .finish()
    }
}

#[derive(Debug, Clone)]
struct NodeMeta {
    /// Insertion order (used to break ties in the round-robin allocation).
    seq: u64,
}

impl ShardManager {
    /// Build an empty manager. The first [`ShardManager::add_node`] call will
    /// produce an initial plan.
    pub fn new(config: ShardManagerConfig) -> ShardManagerResult<Self> {
        if config.n_shards == 0 {
            return Err(ShardManagerError::NoShards);
        }
        Ok(Self {
            config,
            nodes: RwLock::new(BTreeMap::new()),
            assignment: RwLock::new(ShardAssignment::default()),
            plans_emitted: Arc::new(RwLock::new(0)),
        })
    }

    /// Build a manager pre-populated with the provided node ids.
    pub fn with_nodes(
        config: ShardManagerConfig,
        nodes: impl IntoIterator<Item = impl Into<NodeId>>,
    ) -> ShardManagerResult<Self> {
        let mgr = Self::new(config)?;
        for n in nodes {
            let _ = mgr.add_node(n.into())?;
        }
        Ok(mgr)
    }

    /// Number of plans emitted so far.
    pub fn plans_emitted(&self) -> u64 {
        *self.plans_emitted.read()
    }

    /// Owner of a shard in the latest assignment.
    pub fn owner_of(&self, shard: ShardId) -> Option<NodeId> {
        self.assignment.read().owner_of(shard).cloned()
    }

    /// Shards currently owned by a node.
    pub fn shards_owned_by(&self, node_id: &str) -> Vec<ShardId> {
        self.assignment
            .read()
            .map
            .iter()
            .filter(|(_, owner)| owner.as_str() == node_id)
            .map(|(s, _)| *s)
            .collect()
    }

    /// Snapshot of the current assignment.
    pub fn current_assignment(&self) -> ShardAssignment {
        self.assignment.read().clone()
    }

    /// Total number of currently registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Adds a node and returns the rebalance plan that brings the assignment
    /// back in balance.
    pub fn add_node(&self, node_id: NodeId) -> ShardManagerResult<RebalancePlan> {
        {
            let mut nodes = self.nodes.write();
            if nodes.contains_key(&node_id) {
                return Err(ShardManagerError::NodeAlreadyExists(node_id));
            }
            let seq = nodes.len() as u64;
            nodes.insert(node_id.clone(), NodeMeta { seq });
        }
        let plan = self.recompute_plan()?;
        debug!(node = %node_id, moves = plan.moves.len(), "shard manager: add_node");
        Ok(plan)
    }

    /// Removes a node and returns the rebalance plan.
    pub fn remove_node(&self, node_id: &str) -> ShardManagerResult<RebalancePlan> {
        {
            let mut nodes = self.nodes.write();
            if nodes.remove(node_id).is_none() {
                return Err(ShardManagerError::UnknownNode(node_id.to_string()));
            }
        }
        let plan = self.recompute_plan()?;
        debug!(node = %node_id, moves = plan.moves.len(), "shard manager: remove_node");
        Ok(plan)
    }

    /// Apply an externally-supplied assignment (e.g. one that was committed
    /// through Raft). Returns the diff against the current snapshot.
    pub fn install_assignment(&self, new_assignment: ShardAssignment) -> RebalancePlan {
        let old = self.assignment.read().clone();
        let moves = compute_moves(&old, &new_assignment);
        *self.assignment.write() = new_assignment.clone();
        *self.plans_emitted.write() += 1;
        RebalancePlan {
            new_assignment,
            moves,
        }
    }

    fn recompute_plan(&self) -> ShardManagerResult<RebalancePlan> {
        let nodes_snap = self.nodes.read().clone();
        if nodes_snap.is_empty() {
            // Special case: empty assignment.
            let empty = ShardAssignment::default();
            let old = self.assignment.read().clone();
            let moves: Vec<ShardMove> = old
                .map
                .iter()
                .map(|(shard, owner)| ShardMove {
                    shard: *shard,
                    from: Some(owner.clone()),
                    to: String::new(),
                })
                .collect();
            *self.assignment.write() = empty.clone();
            *self.plans_emitted.write() += 1;
            return Ok(RebalancePlan {
                new_assignment: empty,
                moves,
            });
        }

        let nodes: Vec<NodeId> = {
            let mut by_seq: Vec<(u64, NodeId)> = nodes_snap
                .iter()
                .map(|(id, m)| (m.seq, id.clone()))
                .collect();
            by_seq.sort();
            by_seq.into_iter().map(|(_, id)| id).collect()
        };

        let new_assignment = balanced_assignment(self.config.n_shards, &nodes);
        let old = self.assignment.read().clone();
        let moves = compute_moves(&old, &new_assignment);
        *self.assignment.write() = new_assignment.clone();
        *self.plans_emitted.write() += 1;
        Ok(RebalancePlan {
            new_assignment,
            moves,
        })
    }
}

/// Build a deterministic balanced assignment for `n_shards` over the provided
/// node ordering. Each node receives either `floor(n_shards / N)` or
/// `ceil(n_shards / N)` shards.
fn balanced_assignment(n_shards: u32, nodes: &[NodeId]) -> ShardAssignment {
    if nodes.is_empty() {
        return ShardAssignment::default();
    }
    let n = nodes.len() as u32;
    let mut map = BTreeMap::new();
    for shard in 0..n_shards {
        let owner = &nodes[(shard % n) as usize];
        map.insert(shard, owner.clone());
    }
    ShardAssignment { map }
}

fn compute_moves(old: &ShardAssignment, new_assignment: &ShardAssignment) -> Vec<ShardMove> {
    let mut moves = Vec::new();
    let all_shards: HashSet<ShardId> = old
        .map
        .keys()
        .chain(new_assignment.map.keys())
        .cloned()
        .collect();
    let mut shards: Vec<ShardId> = all_shards.into_iter().collect();
    shards.sort();
    for shard in shards {
        let from = old.map.get(&shard).cloned();
        let to = new_assignment.map.get(&shard).cloned();
        match (from, to) {
            (Some(f), Some(t)) if f == t => {}
            (Some(f), Some(t)) => moves.push(ShardMove {
                shard,
                from: Some(f),
                to: t,
            }),
            (None, Some(t)) => moves.push(ShardMove {
                shard,
                from: None,
                to: t,
            }),
            (Some(f), None) => moves.push(ShardMove {
                shard,
                from: Some(f),
                to: String::new(),
            }),
            (None, None) => {}
        }
    }
    moves
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_assignment_round_robins() {
        let assignment =
            balanced_assignment(6, &["n1".to_string(), "n2".to_string(), "n3".to_string()]);
        let counts = assignment.counts();
        for c in counts.values() {
            assert_eq!(*c, 2);
        }
    }

    #[test]
    fn add_node_initial_plan() {
        let mgr = ShardManager::new(ShardManagerConfig { n_shards: 4 }).expect("ok");
        let plan = mgr.add_node("n1".into()).expect("add");
        assert_eq!(plan.new_assignment.n_shards(), 4);
        for owner in plan.new_assignment.map.values() {
            assert_eq!(owner, "n1");
        }
    }

    #[test]
    fn add_node_balances_existing() {
        let mgr = ShardManager::new(ShardManagerConfig { n_shards: 6 }).expect("ok");
        mgr.add_node("n1".into()).expect("ok");
        let plan = mgr.add_node("n2".into()).expect("ok");
        let counts = plan.new_assignment.counts();
        assert_eq!(counts.get("n1"), Some(&3));
        assert_eq!(counts.get("n2"), Some(&3));
        assert_eq!(plan.moves.len(), 3);
    }

    #[test]
    fn remove_node_redistributes() {
        let mgr = ShardManager::new(ShardManagerConfig { n_shards: 6 }).expect("ok");
        mgr.add_node("n1".into()).expect("ok");
        mgr.add_node("n2".into()).expect("ok");
        mgr.add_node("n3".into()).expect("ok");
        let plan = mgr.remove_node("n2").expect("ok");
        let counts = plan.new_assignment.counts();
        assert!(!counts.contains_key("n2"));
        let total: usize = counts.values().sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn empty_node_list_returns_empty_assignment() {
        let mgr = ShardManager::new(ShardManagerConfig { n_shards: 3 }).expect("ok");
        mgr.add_node("n1".into()).expect("ok");
        let plan = mgr.remove_node("n1").expect("ok");
        assert!(plan.new_assignment.map.is_empty());
        assert_eq!(plan.moves.len(), 3);
    }

    #[test]
    fn install_assignment_overrides_state() {
        let mgr = ShardManager::new(ShardManagerConfig { n_shards: 2 }).expect("ok");
        let new_assignment = ShardAssignment::from_vec(vec!["nA".into(), "nB".into()]);
        let plan = mgr.install_assignment(new_assignment.clone());
        assert_eq!(plan.new_assignment, new_assignment);
        assert_eq!(mgr.owner_of(0), Some("nA".to_string()));
        assert_eq!(mgr.owner_of(1), Some("nB".to_string()));
        assert_eq!(plan.moves.len(), 2);
    }

    #[test]
    fn duplicate_add_rejected() {
        let mgr = ShardManager::new(ShardManagerConfig { n_shards: 2 }).expect("ok");
        mgr.add_node("n1".into()).expect("ok");
        let err = mgr.add_node("n1".into()).expect_err("should fail");
        assert!(matches!(err, ShardManagerError::NodeAlreadyExists(_)));
    }

    #[test]
    fn unknown_remove_rejected() {
        let mgr = ShardManager::new(ShardManagerConfig { n_shards: 2 }).expect("ok");
        let err = mgr.remove_node("ghost").expect_err("should fail");
        assert!(matches!(err, ShardManagerError::UnknownNode(_)));
    }

    #[test]
    fn n_shards_zero_rejected() {
        let err = ShardManager::new(ShardManagerConfig { n_shards: 0 }).expect_err("should fail");
        assert!(matches!(err, ShardManagerError::NoShards));
    }

    #[test]
    fn shards_owned_by_returns_correct_subset() {
        let mgr =
            ShardManager::with_nodes(ShardManagerConfig { n_shards: 4 }, ["n1", "n2"]).expect("ok");
        let s1 = mgr.shards_owned_by("n1");
        let s2 = mgr.shards_owned_by("n2");
        // 0,2 → n1; 1,3 → n2 in deterministic order.
        assert_eq!(s1, vec![0, 2]);
        assert_eq!(s2, vec![1, 3]);
    }
}
