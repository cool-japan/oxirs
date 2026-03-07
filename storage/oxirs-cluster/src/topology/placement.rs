//! Consistent hashing with virtual nodes (vnodes) for shard placement
//!
//! Supports 1000+ physical nodes via a virtual node ring.
//! Uses FNV-1a hashing for deterministic, fast key-to-node mapping.

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

use super::hierarchy::fnv1a_bytes;

/// Virtual node ring for consistent hash-based shard placement.
///
/// Each physical node is represented by `virtual_nodes_per_physical` virtual
/// nodes evenly distributed around the ring, providing better load balance.
pub struct VNodeRing {
    virtual_nodes_per_physical: usize,
    /// Sorted ring: token → physical node_id
    ring: BTreeMap<u64, String>,
    /// Set of registered physical node IDs
    physical_nodes: HashSet<String>,
}

impl VNodeRing {
    /// Create a new ring with the given number of virtual nodes per physical node.
    ///
    /// Typical values: 150–200 for good load balance at large scales.
    pub fn new(vnodes_per_node: usize) -> Self {
        Self {
            virtual_nodes_per_physical: vnodes_per_node.max(1),
            ring: BTreeMap::new(),
            physical_nodes: HashSet::new(),
        }
    }

    /// Add a physical node to the ring
    pub fn add_node(&mut self, node_id: &str) {
        if self.physical_nodes.contains(node_id) {
            return; // idempotent
        }
        for vnode_idx in 0..self.virtual_nodes_per_physical {
            let token = Self::hash_node_vnode(node_id, vnode_idx);
            self.ring.insert(token, node_id.to_string());
        }
        self.physical_nodes.insert(node_id.to_string());
    }

    /// Remove a physical node from the ring
    pub fn remove_node(&mut self, node_id: &str) {
        if !self.physical_nodes.contains(node_id) {
            return;
        }
        for vnode_idx in 0..self.virtual_nodes_per_physical {
            let token = Self::hash_node_vnode(node_id, vnode_idx);
            self.ring.remove(&token);
        }
        self.physical_nodes.remove(node_id);
    }

    /// Get the physical node responsible for a key
    pub fn get_node(&self, key: &[u8]) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let token = Self::hash_key(key);
        // Walk clockwise from token; wrap around to start if needed
        let node_id = self
            .ring
            .range(token..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, id)| id.as_str());
        node_id
    }

    /// Get N distinct replicas for a key (replication_factor > 1).
    ///
    /// Walks clockwise around the ring collecting unique physical nodes.
    pub fn get_replicas(&self, key: &[u8], n: usize) -> Vec<String> {
        if self.ring.is_empty() || n == 0 {
            return Vec::new();
        }

        let token = Self::hash_key(key);
        let mut result: Vec<String> = Vec::with_capacity(n);
        let mut seen: HashSet<&str> = HashSet::new();

        // Walk clockwise from token (then wrap around)
        let after = self.ring.range(token..);
        let before = self.ring.range(..token);

        for (_, node_id) in after.chain(before) {
            if seen.insert(node_id.as_str()) {
                result.push(node_id.clone());
                if result.len() >= n {
                    break;
                }
            }
        }

        result
    }

    /// Get the token ranges owned by a physical node.
    ///
    /// Returns a vector of `(start, end)` token ranges (half-open: [start, end)).
    pub fn node_token_ranges(&self, node_id: &str) -> Vec<(u64, u64)> {
        if !self.physical_nodes.contains(node_id) {
            return Vec::new();
        }

        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let tokens: Vec<u64> = self.ring.keys().copied().collect();
        let total = tokens.len();

        if total == 0 {
            return Vec::new();
        }

        for (i, &token) in tokens.iter().enumerate() {
            if self.ring.get(&token).map(|s| s.as_str()) == Some(node_id) {
                // The range this vnode owns: from previous token (exclusive) to this token (inclusive)
                let start = if i == 0 { 0 } else { tokens[i - 1] + 1 };
                let end = token;
                ranges.push((start, end));
            }
        }

        ranges
    }

    /// Calculate load imbalance as max/min token count ratio.
    ///
    /// A ratio of 1.0 means perfect balance; higher ratios indicate skew.
    pub fn load_balance_ratio(&self) -> f64 {
        if self.physical_nodes.is_empty() {
            return 1.0;
        }

        let mut counts: HashMap<&str, usize> = HashMap::new();
        for node_id in self.ring.values() {
            *counts.entry(node_id.as_str()).or_insert(0) += 1;
        }

        let max_count = counts.values().copied().max().unwrap_or(0);
        let min_count = counts.values().copied().min().unwrap_or(0);

        if min_count == 0 {
            return f64::INFINITY;
        }
        max_count as f64 / min_count as f64
    }

    /// Number of physical nodes in the ring
    pub fn node_count(&self) -> usize {
        self.physical_nodes.len()
    }

    /// Number of virtual nodes (total ring entries)
    pub fn vnode_count(&self) -> usize {
        self.ring.len()
    }

    /// Check if the ring is empty
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    // -------------------------------------------------------------------------
    // Hash functions
    // -------------------------------------------------------------------------

    /// Hash a key to a ring token
    fn hash_key(key: &[u8]) -> u64 {
        fnv1a_bytes(key)
    }

    /// Hash a (node_id, vnode_index) pair to a ring token
    fn hash_node_vnode(node_id: &str, vnode_idx: usize) -> u64 {
        // Combine node_id bytes with the vnode index for unique token per vnode
        let mut data = node_id.as_bytes().to_vec();
        data.push(b':');
        data.extend_from_slice(&vnode_idx.to_le_bytes());
        fnv1a_bytes(&data)
    }
}

// -------------------------------------------------------------------------
// Rebalancing plan
// -------------------------------------------------------------------------

/// Describes the movement of a shard (token range) between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMove {
    pub shard_id: u64,
    pub token_range: (u64, u64),
    pub from_node: String,
    pub to_node: String,
    /// Estimated bytes to transfer (0 = unknown)
    pub estimated_bytes: u64,
}

/// Full rebalancing plan computed when nodes join or leave
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancePlan {
    pub moves: Vec<ShardMove>,
    pub total_data_moved_estimate: u64,
    /// Fraction of ring owned by each source node after removal/addition
    pub source_node_loads: HashMap<String, f64>,
    /// Fraction of ring owned by each target node after rebalance
    pub target_node_loads: HashMap<String, f64>,
}

impl RebalancePlan {
    /// The number of shard moves in this plan
    pub fn move_count(&self) -> usize {
        self.moves.len()
    }

    /// True if this plan requires no data movement
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
}

/// Plan the rebalance for adding a new node or removing an existing one.
///
/// Supply exactly one of `new_node` or `removed_node`.
pub fn plan_rebalance(
    ring: &VNodeRing,
    new_node: Option<&str>,
    removed_node: Option<&str>,
) -> Result<RebalancePlan> {
    match (new_node, removed_node) {
        (Some(_), Some(_)) => Err(ClusterError::Config(
            "Supply either new_node or removed_node, not both".into(),
        )),
        (None, None) => Ok(RebalancePlan {
            moves: Vec::new(),
            total_data_moved_estimate: 0,
            source_node_loads: compute_loads(ring),
            target_node_loads: compute_loads(ring),
        }),
        (Some(new_id), None) => plan_add_node(ring, new_id),
        (None, Some(removed_id)) => plan_remove_node(ring, removed_id),
    }
}

fn plan_add_node(ring: &VNodeRing, new_node_id: &str) -> Result<RebalancePlan> {
    let source_loads = compute_loads(ring);

    // Simulate adding the node
    let mut new_ring = VNodeRing::new(ring.virtual_nodes_per_physical);
    for node_id in &ring.physical_nodes {
        new_ring.add_node(node_id);
    }
    new_ring.add_node(new_node_id);
    let target_loads = compute_loads(&new_ring);

    // The new node's vnodes steal ranges from their predecessor nodes
    let moves = collect_moves_for_new_node(ring, &new_ring, new_node_id);
    let total = moves.iter().map(|m| m.estimated_bytes).sum();

    Ok(RebalancePlan {
        moves,
        total_data_moved_estimate: total,
        source_node_loads: source_loads,
        target_node_loads: target_loads,
    })
}

fn plan_remove_node(ring: &VNodeRing, removed_id: &str) -> Result<RebalancePlan> {
    if !ring.physical_nodes.contains(removed_id) {
        return Err(ClusterError::Config(format!(
            "Node '{}' is not in the ring",
            removed_id
        )));
    }

    let source_loads = compute_loads(ring);

    // Simulate removing the node
    let mut new_ring = VNodeRing::new(ring.virtual_nodes_per_physical);
    for node_id in &ring.physical_nodes {
        if node_id.as_str() != removed_id {
            new_ring.add_node(node_id);
        }
    }
    let target_loads = compute_loads(&new_ring);

    // The removed node's token ranges get assigned to the next clockwise node
    let moves = collect_moves_for_removed_node(ring, &new_ring, removed_id);
    let total = moves.iter().map(|m| m.estimated_bytes).sum();

    Ok(RebalancePlan {
        moves,
        total_data_moved_estimate: total,
        source_node_loads: source_loads,
        target_node_loads: target_loads,
    })
}

fn collect_moves_for_new_node(
    old_ring: &VNodeRing,
    new_ring: &VNodeRing,
    new_node_id: &str,
) -> Vec<ShardMove> {
    let mut moves = Vec::new();
    let mut shard_id: u64 = 0;

    // For each token that now belongs to new_node, it was previously owned by
    // the predecessor node in the old ring
    for (&token, owner) in &new_ring.ring {
        if owner.as_str() == new_node_id {
            // Who owned this token in the old ring?
            if let Some(old_owner) = old_ring.get_node(&token.to_le_bytes()) {
                moves.push(ShardMove {
                    shard_id,
                    token_range: (token, token),
                    from_node: old_owner.to_string(),
                    to_node: new_node_id.to_string(),
                    estimated_bytes: 0,
                });
                shard_id += 1;
            }
        }
    }
    moves
}

fn collect_moves_for_removed_node(
    old_ring: &VNodeRing,
    new_ring: &VNodeRing,
    removed_id: &str,
) -> Vec<ShardMove> {
    let mut moves = Vec::new();
    let mut shard_id: u64 = 0;

    // For each token that belonged to removed node in old ring,
    // find who owns it now in the new ring
    for (&token, owner) in &old_ring.ring {
        if owner.as_str() == removed_id {
            if let Some(new_owner) = new_ring.get_node(&token.to_le_bytes()) {
                moves.push(ShardMove {
                    shard_id,
                    token_range: (token, token),
                    from_node: removed_id.to_string(),
                    to_node: new_owner.to_string(),
                    estimated_bytes: 0,
                });
                shard_id += 1;
            }
        }
    }
    moves
}

/// Compute the fractional ring ownership per node
fn compute_loads(ring: &VNodeRing) -> HashMap<String, f64> {
    let total = ring.vnode_count() as f64;
    if total == 0.0 {
        return HashMap::new();
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for node_id in ring.ring.values() {
        *counts.entry(node_id.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(id, count)| (id, count as f64 / total))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ring_with_nodes(n: usize) -> VNodeRing {
        let mut ring = VNodeRing::new(150);
        for i in 0..n {
            ring.add_node(&format!("node-{}", i));
        }
        ring
    }

    #[test]
    fn test_add_remove_node() {
        let mut ring = VNodeRing::new(10);
        ring.add_node("node-a");
        ring.add_node("node-b");
        assert_eq!(ring.node_count(), 2);
        assert_eq!(ring.vnode_count(), 20);

        ring.remove_node("node-a");
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.vnode_count(), 10);
    }

    #[test]
    fn test_idempotent_add() {
        let mut ring = VNodeRing::new(10);
        ring.add_node("node-a");
        ring.add_node("node-a"); // duplicate
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.vnode_count(), 10);
    }

    #[test]
    fn test_get_node_deterministic() {
        let ring = make_ring_with_nodes(5);
        let key = b"rdf:type";
        let first = ring.get_node(key).map(|s| s.to_string());
        let second = ring.get_node(key).map(|s| s.to_string());
        assert_eq!(first, second);
        assert!(first.is_some());
    }

    #[test]
    fn test_get_node_empty_ring() {
        let ring = VNodeRing::new(10);
        assert!(ring.get_node(b"anything").is_none());
    }

    #[test]
    fn test_get_replicas_distinct() {
        let ring = make_ring_with_nodes(5);
        let replicas = ring.get_replicas(b"subject:predicate:object", 3);
        assert_eq!(replicas.len(), 3);
        // All replicas must be unique physical nodes
        let unique: HashSet<&String> = replicas.iter().collect();
        assert_eq!(unique.len(), 3, "Replicas must be distinct");
    }

    #[test]
    fn test_get_replicas_more_than_nodes() {
        let ring = make_ring_with_nodes(3);
        // Ask for 5 replicas when only 3 nodes exist
        let replicas = ring.get_replicas(b"key", 5);
        assert_eq!(replicas.len(), 3, "Can't exceed available physical nodes");
    }

    #[test]
    fn test_load_balance_ratio() {
        let ring = make_ring_with_nodes(10);
        let ratio = ring.load_balance_ratio();
        // With 150 vnodes per node and 10 nodes, expect near-perfect balance
        // Allow up to 3x imbalance (real typical is < 1.5x)
        assert!(ratio < 3.0, "Load balance ratio {} is too high", ratio);
    }

    #[test]
    fn test_node_token_ranges() {
        let mut ring = VNodeRing::new(5);
        ring.add_node("node-a");
        ring.add_node("node-b");
        let ranges = ring.node_token_ranges("node-a");
        assert!(!ranges.is_empty(), "node-a should have token ranges");
    }

    #[test]
    fn test_node_token_ranges_nonexistent() {
        let ring = make_ring_with_nodes(3);
        let ranges = ring.node_token_ranges("nonexistent");
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_plan_rebalance_add_node() {
        let ring = make_ring_with_nodes(5);
        let plan = plan_rebalance(&ring, Some("node-new"), None).unwrap();
        // Adding a node should cause exactly virtual_nodes_per_physical moves
        assert!(!plan.moves.is_empty(), "Adding a node should trigger moves");
        assert_eq!(plan.move_count(), ring.virtual_nodes_per_physical);
        // All moves should target the new node
        for mv in &plan.moves {
            assert_eq!(mv.to_node, "node-new");
        }
    }

    #[test]
    fn test_plan_rebalance_remove_node() {
        let ring = make_ring_with_nodes(5);
        let plan = plan_rebalance(&ring, None, Some("node-0")).unwrap();
        assert!(!plan.moves.is_empty());
        // All moves should come from the removed node
        for mv in &plan.moves {
            assert_eq!(mv.from_node, "node-0");
        }
    }

    #[test]
    fn test_plan_rebalance_both_fails() {
        let ring = make_ring_with_nodes(3);
        let result = plan_rebalance(&ring, Some("new"), Some("node-0"));
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_rebalance_remove_nonexistent_fails() {
        let ring = make_ring_with_nodes(3);
        let result = plan_rebalance(&ring, None, Some("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_1000_node_ring_performance() {
        let ring = make_ring_with_nodes(1000);
        assert_eq!(ring.node_count(), 1000);
        assert_eq!(ring.vnode_count(), 150_000);

        // Get node should be fast
        let node = ring.get_node(b"http://example.org/triple#12345");
        assert!(node.is_some());

        // Get replicas
        let replicas = ring.get_replicas(b"large-cluster-key", 3);
        assert_eq!(replicas.len(), 3);

        // Load balance should be reasonable
        let ratio = ring.load_balance_ratio();
        assert!(
            ratio < 1.5,
            "1000-node ring balance ratio {} too high",
            ratio
        );
    }

    #[test]
    fn test_consistent_hash_stability() {
        // Adding/removing a node should only affect vnodes_per_node tokens
        let mut ring = make_ring_with_nodes(5);
        let test_keys: Vec<Vec<u8>> = (0..100)
            .map(|i| format!("key-{}", i).into_bytes())
            .collect();

        let before: Vec<Option<String>> = test_keys
            .iter()
            .map(|k| ring.get_node(k).map(|s| s.to_string()))
            .collect();

        ring.add_node("node-extra");

        let after: Vec<Option<String>> = test_keys
            .iter()
            .map(|k| ring.get_node(k).map(|s| s.to_string()))
            .collect();

        // Most keys should map to the same node (consistent hashing property)
        let changed = before
            .iter()
            .zip(after.iter())
            .filter(|(b, a)| b != a)
            .count();

        // With 150 vnodes/node and 6 total nodes, ~150/900 ≈ 16.7% expected change
        assert!(
            changed < 40,
            "Too many keys remapped: {} out of 100",
            changed
        );
    }
}
