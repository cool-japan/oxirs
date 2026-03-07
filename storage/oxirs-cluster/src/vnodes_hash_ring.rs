//! Consistent Hashing with Virtual Nodes
//!
//! This module implements a consistent hash ring with virtual nodes (vnodes) for
//! distributing data across cluster nodes with minimal redistribution on membership
//! changes. It also supports bounded loads to prevent hotspots.
//!
//! # Architecture
//!
//! ```text
//!        ┌──────────────────────────────────────────┐
//!        │            Hash Ring (0..2^64)            │
//!        │                                          │
//!        │    vnode(A,0)     vnode(B,0)              │
//!        │        ●             ●                   │
//!        │       / \           / \                   │
//!        │      /   \         /   \                  │
//!        │   vnode(A,1)   vnode(B,1)                │
//!        │      ●           ●                       │
//!        │       \         /                        │
//!        │        \       /                         │
//!        │       vnode(C,0)                         │
//!        │          ●                               │
//!        └──────────────────────────────────────────┘
//! ```
//!
//! # Bounded Loads
//!
//! When `load_factor` is set, the ring ensures no node receives more than
//! `ceil(average_load * load_factor)` keys. Overloaded nodes are skipped
//! in favor of the next node on the ring.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// A node identifier in the cluster.
pub type NodeId = String;

/// Represents a physical node in the cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Network address (host:port).
    pub address: String,
    /// Weight for proportional vnode allocation (default 1.0).
    pub weight: u32,
    /// Current load (number of assigned keys).
    pub load: usize,
    /// Whether the node is active and accepting traffic.
    pub active: bool,
    /// Optional datacenter/zone label for rack-aware placement.
    pub zone: Option<String>,
}

impl ClusterNode {
    /// Creates a new cluster node with default weight.
    pub fn new(id: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            address: address.into(),
            weight: 1,
            load: 0,
            active: true,
            zone: None,
        }
    }

    /// Sets the weight for proportional vnode allocation.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Sets the datacenter/zone label.
    pub fn with_zone(mut self, zone: impl Into<String>) -> Self {
        self.zone = Some(zone.into());
        self
    }
}

/// Configuration for the consistent hash ring.
#[derive(Debug, Clone)]
pub struct HashRingConfig {
    /// Number of virtual nodes per physical node (per weight unit).
    pub vnodes_per_node: usize,
    /// Load factor for bounded loads (e.g., 1.25 means max 25% above average).
    /// Set to None to disable bounded loads.
    pub load_factor: Option<f64>,
    /// Number of replicas for each key (for replication).
    pub replica_count: usize,
}

impl Default for HashRingConfig {
    fn default() -> Self {
        Self {
            vnodes_per_node: 150,
            load_factor: Some(1.25),
            replica_count: 3,
        }
    }
}

/// Statistics about the hash ring.
#[derive(Debug, Clone, Default)]
pub struct RingStats {
    /// Number of physical nodes.
    pub node_count: usize,
    /// Number of active nodes.
    pub active_node_count: usize,
    /// Total number of virtual nodes on the ring.
    pub vnode_count: usize,
    /// Load distribution (node_id -> load).
    pub load_distribution: HashMap<NodeId, usize>,
    /// Standard deviation of load distribution.
    pub load_stddev: f64,
    /// Maximum load across all nodes.
    pub max_load: usize,
    /// Minimum load across all nodes.
    pub min_load: usize,
}

/// A consistent hash ring with virtual nodes and bounded loads.
pub struct ConsistentHashRing {
    config: HashRingConfig,
    /// The ring: hash position -> (node_id, vnode_index).
    ring: BTreeMap<u64, (NodeId, usize)>,
    /// Physical nodes.
    nodes: HashMap<NodeId, ClusterNode>,
    /// Load tracking per node.
    loads: HashMap<NodeId, usize>,
}

impl ConsistentHashRing {
    /// Creates a new empty hash ring with the given configuration.
    pub fn new(config: HashRingConfig) -> Self {
        Self {
            config,
            ring: BTreeMap::new(),
            nodes: HashMap::new(),
            loads: HashMap::new(),
        }
    }

    /// Creates a new hash ring with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(HashRingConfig::default())
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &HashRingConfig {
        &self.config
    }

    /// Adds a node to the ring, creating virtual nodes.
    ///
    /// Returns the number of vnodes created.
    pub fn add_node(&mut self, node: ClusterNode) -> usize {
        let node_id = node.id.clone();
        let weight = node.weight.max(1) as usize;
        let vnode_count = self.config.vnodes_per_node * weight;

        self.nodes.insert(node_id.clone(), node);
        self.loads.entry(node_id.clone()).or_insert(0);

        let mut created = 0;
        for i in 0..vnode_count {
            let hash = hash_vnode(&node_id, i);
            self.ring.insert(hash, (node_id.clone(), i));
            created += 1;
        }

        created
    }

    /// Removes a node from the ring.
    ///
    /// Returns true if the node existed.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        if self.nodes.remove(node_id).is_some() {
            // Remove all vnodes for this node
            self.ring.retain(|_, (id, _)| id != node_id);
            self.loads.remove(node_id);
            true
        } else {
            false
        }
    }

    /// Deactivates a node (it remains on the ring but doesn't accept new keys).
    pub fn deactivate_node(&mut self, node_id: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.active = false;
            true
        } else {
            false
        }
    }

    /// Reactivates a previously deactivated node.
    pub fn activate_node(&mut self, node_id: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.active = true;
            true
        } else {
            false
        }
    }

    /// Returns the number of physical nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of active nodes.
    pub fn active_node_count(&self) -> usize {
        self.nodes.values().filter(|n| n.active).count()
    }

    /// Returns the total number of vnodes on the ring.
    pub fn vnode_count(&self) -> usize {
        self.ring.len()
    }

    /// Returns a node by ID.
    pub fn get_node(&self, node_id: &str) -> Option<&ClusterNode> {
        self.nodes.get(node_id)
    }

    /// Returns all node IDs.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().cloned().collect()
    }

    /// Looks up the responsible node for a key.
    ///
    /// If bounded loads are enabled, overloaded nodes are skipped.
    pub fn get_node_for_key(&self, key: &[u8]) -> Option<NodeId> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = hash_key(key);
        let max_load = self.max_allowed_load();

        // Walk clockwise from the hash position
        let candidates = self
            .ring
            .range(hash..)
            .chain(self.ring.iter())
            .take(self.ring.len());

        for (_, (node_id, _)) in candidates {
            let node = match self.nodes.get(node_id) {
                Some(n) => n,
                None => continue,
            };

            if !node.active {
                continue;
            }

            // Check bounded load
            if let Some(max) = max_load {
                let current_load = self.loads.get(node_id).copied().unwrap_or(0);
                if current_load >= max {
                    continue;
                }
            }

            return Some(node_id.clone());
        }

        // If all nodes are overloaded, return the first active node anyway
        self.ring
            .range(hash..)
            .chain(self.ring.iter())
            .find(|(_, (id, _))| self.nodes.get(id).is_some_and(|n| n.active))
            .map(|(_, (id, _))| id.clone())
    }

    /// Gets the replica nodes for a key (returns up to `replica_count` distinct nodes).
    pub fn get_replicas(&self, key: &[u8]) -> Vec<NodeId> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let hash = hash_key(key);
        let mut replicas = Vec::new();
        let mut seen = HashSet::new();

        let candidates = self
            .ring
            .range(hash..)
            .chain(self.ring.iter())
            .take(self.ring.len());

        for (_, (node_id, _)) in candidates {
            if seen.contains(node_id) {
                continue;
            }

            if let Some(node) = self.nodes.get(node_id) {
                if node.active {
                    seen.insert(node_id.clone());
                    replicas.push(node_id.clone());

                    if replicas.len() >= self.config.replica_count {
                        break;
                    }
                }
            }
        }

        replicas
    }

    /// Gets zone-aware replicas (no two replicas in the same zone if possible).
    pub fn get_zone_aware_replicas(&self, key: &[u8]) -> Vec<NodeId> {
        if self.ring.is_empty() {
            return Vec::new();
        }

        let hash = hash_key(key);
        let mut replicas = Vec::new();
        let mut seen_nodes = HashSet::new();
        let mut seen_zones = HashSet::new();

        let candidates = self
            .ring
            .range(hash..)
            .chain(self.ring.iter())
            .take(self.ring.len());

        // First pass: prefer different zones
        let all_candidates: Vec<_> = candidates.collect();

        for (_, (node_id, _)) in &all_candidates {
            if seen_nodes.contains(node_id) {
                continue;
            }

            if let Some(node) = self.nodes.get(node_id) {
                if !node.active {
                    continue;
                }

                let zone = node.zone.as_deref().unwrap_or("default");
                if !seen_zones.contains(zone) {
                    seen_zones.insert(zone.to_string());
                    seen_nodes.insert(node_id.clone());
                    replicas.push(node_id.clone());

                    if replicas.len() >= self.config.replica_count {
                        return replicas;
                    }
                }
            }
        }

        // Second pass: fill remaining with any available nodes
        for (_, (node_id, _)) in &all_candidates {
            if seen_nodes.contains(node_id) {
                continue;
            }

            if let Some(node) = self.nodes.get(node_id) {
                if node.active {
                    seen_nodes.insert(node_id.clone());
                    replicas.push(node_id.clone());

                    if replicas.len() >= self.config.replica_count {
                        break;
                    }
                }
            }
        }

        replicas
    }

    /// Increments the load for a node.
    pub fn increment_load(&mut self, node_id: &str) {
        if let Some(load) = self.loads.get_mut(node_id) {
            *load += 1;
        }
    }

    /// Decrements the load for a node.
    pub fn decrement_load(&mut self, node_id: &str) {
        if let Some(load) = self.loads.get_mut(node_id) {
            *load = load.saturating_sub(1);
        }
    }

    /// Resets all loads to zero.
    pub fn reset_loads(&mut self) {
        for load in self.loads.values_mut() {
            *load = 0;
        }
    }

    /// Returns the current load for a node.
    pub fn node_load(&self, node_id: &str) -> usize {
        self.loads.get(node_id).copied().unwrap_or(0)
    }

    /// Computes the maximum allowed load per node (for bounded loads).
    fn max_allowed_load(&self) -> Option<usize> {
        let factor = self.config.load_factor?;
        let active_count = self.active_node_count();
        if active_count == 0 {
            return None;
        }

        let total_load: usize = self.loads.values().sum();
        let avg_load = total_load as f64 / active_count as f64;
        Some((avg_load * factor).ceil() as usize)
    }

    /// Computes what keys would need to be moved if a node is added.
    ///
    /// Returns a map of (source_node -> list of affected vnode positions).
    pub fn affected_ranges_on_add(
        &self,
        new_node_id: &str,
        weight: u32,
    ) -> HashMap<NodeId, Vec<u64>> {
        let mut affected: HashMap<NodeId, Vec<u64>> = HashMap::new();
        let vnode_count = self.config.vnodes_per_node * weight.max(1) as usize;

        for i in 0..vnode_count {
            let hash = hash_vnode(new_node_id, i);

            // Find the node currently responsible for this position
            if let Some((_, (current_owner, _))) = self
                .ring
                .range(hash..)
                .next()
                .or_else(|| self.ring.iter().next())
            {
                affected
                    .entry(current_owner.clone())
                    .or_default()
                    .push(hash);
            }
        }

        affected
    }

    /// Computes what keys would need to be moved if a node is removed.
    ///
    /// Returns a map of (destination_node -> count of affected vnodes).
    pub fn affected_ranges_on_remove(&self, node_id: &str) -> HashMap<NodeId, usize> {
        let mut affected: HashMap<NodeId, usize> = HashMap::new();

        // For each vnode of the removed node, find who takes over
        let vnodes: Vec<u64> = self
            .ring
            .iter()
            .filter(|(_, (id, _))| id == node_id)
            .map(|(&pos, _)| pos)
            .collect();

        for pos in vnodes {
            // The successor on the ring (skipping the removed node) takes over
            let successor = self
                .ring
                .range((pos + 1)..)
                .chain(self.ring.iter())
                .find(|(_, (id, _))| id != node_id)
                .map(|(_, (id, _))| id.clone());

            if let Some(succ_id) = successor {
                *affected.entry(succ_id).or_insert(0) += 1;
            }
        }

        affected
    }

    /// Returns statistics about the ring.
    pub fn stats(&self) -> RingStats {
        let load_distribution: HashMap<NodeId, usize> = self.loads.clone();
        let loads: Vec<usize> = load_distribution.values().copied().collect();

        let (max_load, min_load) = if loads.is_empty() {
            (0, 0)
        } else {
            (
                loads.iter().copied().max().unwrap_or(0),
                loads.iter().copied().min().unwrap_or(0),
            )
        };

        let mean = if loads.is_empty() {
            0.0
        } else {
            loads.iter().sum::<usize>() as f64 / loads.len() as f64
        };

        let variance = if loads.is_empty() {
            0.0
        } else {
            loads
                .iter()
                .map(|&l| (l as f64 - mean).powi(2))
                .sum::<f64>()
                / loads.len() as f64
        };

        RingStats {
            node_count: self.nodes.len(),
            active_node_count: self.active_node_count(),
            vnode_count: self.ring.len(),
            load_distribution,
            load_stddev: variance.sqrt(),
            max_load,
            min_load,
        }
    }
}

/// Hashes a key to a position on the ring.
fn hash_key(key: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in key {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Hashes a virtual node to a position on the ring.
fn hash_vnode(node_id: &str, vnode_index: usize) -> u64 {
    let combined = format!("{node_id}#vnode#{vnode_index}");
    hash_key(combined.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ring() -> ConsistentHashRing {
        ConsistentHashRing::with_defaults()
    }

    fn three_node_ring() -> ConsistentHashRing {
        let mut ring = default_ring();
        ring.add_node(ClusterNode::new("node-a", "10.0.0.1:8080"));
        ring.add_node(ClusterNode::new("node-b", "10.0.0.2:8080"));
        ring.add_node(ClusterNode::new("node-c", "10.0.0.3:8080"));
        ring
    }

    // ── Node management ─────────────────────────────────────────────────────

    #[test]
    fn test_add_node() {
        let mut ring = default_ring();
        let created = ring.add_node(ClusterNode::new("node-1", "10.0.0.1:8080"));
        assert_eq!(created, 150); // default vnodes_per_node
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.vnode_count(), 150);
    }

    #[test]
    fn test_add_weighted_node() {
        let mut ring = default_ring();
        let node = ClusterNode::new("node-1", "10.0.0.1:8080").with_weight(2);
        let created = ring.add_node(node);
        assert_eq!(created, 300); // 150 * weight 2
    }

    #[test]
    fn test_remove_node() {
        let mut ring = three_node_ring();
        assert!(ring.remove_node("node-b"));
        assert_eq!(ring.node_count(), 2);
        assert!(!ring.remove_node("nonexistent"));
    }

    #[test]
    fn test_deactivate_activate_node() {
        let mut ring = three_node_ring();
        assert!(ring.deactivate_node("node-a"));
        assert_eq!(ring.active_node_count(), 2);
        assert!(ring.activate_node("node-a"));
        assert_eq!(ring.active_node_count(), 3);
    }

    #[test]
    fn test_deactivate_nonexistent() {
        let mut ring = default_ring();
        assert!(!ring.deactivate_node("nonexistent"));
    }

    #[test]
    fn test_get_node() {
        let ring = three_node_ring();
        let node = ring.get_node("node-a");
        assert!(node.is_some());
        assert_eq!(node.map(|n| n.address.as_str()), Some("10.0.0.1:8080"));
    }

    #[test]
    fn test_node_ids() {
        let ring = three_node_ring();
        let mut ids = ring.node_ids();
        ids.sort();
        assert_eq!(ids, vec!["node-a", "node-b", "node-c"]);
    }

    // ── Key lookup ──────────────────────────────────────────────────────────

    #[test]
    fn test_get_node_for_key() {
        let ring = three_node_ring();
        let node = ring.get_node_for_key(b"some-key");
        assert!(node.is_some());
    }

    #[test]
    fn test_empty_ring_lookup() {
        let ring = default_ring();
        assert!(ring.get_node_for_key(b"key").is_none());
    }

    #[test]
    fn test_key_consistency() {
        let ring = three_node_ring();
        let node1 = ring.get_node_for_key(b"consistent-key");
        let node2 = ring.get_node_for_key(b"consistent-key");
        assert_eq!(node1, node2);
    }

    #[test]
    fn test_different_keys_may_map_differently() {
        let ring = three_node_ring();
        let mut mappings = HashSet::new();
        for i in 0..100 {
            let key = format!("key-{i}");
            if let Some(node) = ring.get_node_for_key(key.as_bytes()) {
                mappings.insert(node);
            }
        }
        // With 3 nodes, we should see all 3 getting some keys
        assert!(mappings.len() >= 2, "Expected distribution across nodes");
    }

    #[test]
    fn test_inactive_node_skipped() {
        let mut ring = three_node_ring();
        let key = b"test-key";
        let original = ring.get_node_for_key(key).expect("should have node");
        ring.deactivate_node(&original);
        let new_node = ring.get_node_for_key(key).expect("should have node");
        assert_ne!(original, new_node);
    }

    // ── Replicas ────────────────────────────────────────────────────────────

    #[test]
    fn test_get_replicas() {
        let ring = three_node_ring();
        let replicas = ring.get_replicas(b"some-key");
        assert_eq!(replicas.len(), 3); // replica_count = 3
                                       // All replicas should be distinct
        let unique: HashSet<_> = replicas.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_get_replicas_fewer_nodes() {
        let mut ring = ConsistentHashRing::new(HashRingConfig {
            replica_count: 5,
            ..HashRingConfig::default()
        });
        ring.add_node(ClusterNode::new("a", "10.0.0.1:8080"));
        ring.add_node(ClusterNode::new("b", "10.0.0.2:8080"));
        let replicas = ring.get_replicas(b"key");
        assert_eq!(replicas.len(), 2); // only 2 nodes available
    }

    #[test]
    fn test_get_replicas_empty_ring() {
        let ring = default_ring();
        assert!(ring.get_replicas(b"key").is_empty());
    }

    // ── Zone-aware replicas ─────────────────────────────────────────────────

    #[test]
    fn test_zone_aware_replicas() {
        let mut ring = ConsistentHashRing::new(HashRingConfig {
            replica_count: 3,
            ..HashRingConfig::default()
        });
        ring.add_node(ClusterNode::new("a", "10.0.0.1:8080").with_zone("us-east"));
        ring.add_node(ClusterNode::new("b", "10.0.0.2:8080").with_zone("us-west"));
        ring.add_node(ClusterNode::new("c", "10.0.0.3:8080").with_zone("eu-west"));

        let replicas = ring.get_zone_aware_replicas(b"key");
        assert_eq!(replicas.len(), 3);
        let unique: HashSet<_> = replicas.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_zone_aware_same_zone() {
        let mut ring = ConsistentHashRing::new(HashRingConfig {
            replica_count: 3,
            ..HashRingConfig::default()
        });
        ring.add_node(ClusterNode::new("a", "10.0.0.1:8080").with_zone("zone1"));
        ring.add_node(ClusterNode::new("b", "10.0.0.2:8080").with_zone("zone1"));
        ring.add_node(ClusterNode::new("c", "10.0.0.3:8080").with_zone("zone2"));

        let replicas = ring.get_zone_aware_replicas(b"key");
        assert_eq!(replicas.len(), 3);
    }

    // ── Load tracking ───────────────────────────────────────────────────────

    #[test]
    fn test_increment_decrement_load() {
        let mut ring = three_node_ring();
        ring.increment_load("node-a");
        ring.increment_load("node-a");
        assert_eq!(ring.node_load("node-a"), 2);

        ring.decrement_load("node-a");
        assert_eq!(ring.node_load("node-a"), 1);
    }

    #[test]
    fn test_decrement_below_zero() {
        let mut ring = three_node_ring();
        ring.decrement_load("node-a"); // already 0
        assert_eq!(ring.node_load("node-a"), 0);
    }

    #[test]
    fn test_reset_loads() {
        let mut ring = three_node_ring();
        ring.increment_load("node-a");
        ring.increment_load("node-b");
        ring.reset_loads();
        assert_eq!(ring.node_load("node-a"), 0);
        assert_eq!(ring.node_load("node-b"), 0);
    }

    #[test]
    fn test_bounded_load() {
        let mut ring = ConsistentHashRing::new(HashRingConfig {
            load_factor: Some(1.5),
            replica_count: 1,
            vnodes_per_node: 10,
        });
        ring.add_node(ClusterNode::new("a", "10.0.0.1:8080"));
        ring.add_node(ClusterNode::new("b", "10.0.0.2:8080"));

        // Overload node "a"
        for _ in 0..100 {
            ring.increment_load("a");
        }

        // With bounded loads, keys should be redirected away from "a"
        let mut b_count = 0;
        for i in 0..50 {
            let key = format!("key-{i}");
            if let Some(node) = ring.get_node_for_key(key.as_bytes()) {
                if node == "b" {
                    b_count += 1;
                }
            }
        }
        // Most keys should go to "b" since "a" is overloaded
        assert!(
            b_count > 20,
            "Expected most keys to go to node b, got {b_count}"
        );
    }

    // ── Membership changes ──────────────────────────────────────────────────

    #[test]
    fn test_minimal_redistribution_on_add() {
        let ring = three_node_ring();

        // Record assignments before
        let mut before: HashMap<String, String> = HashMap::new();
        for i in 0..100 {
            let key = format!("key-{i}");
            if let Some(node) = ring.get_node_for_key(key.as_bytes()) {
                before.insert(key, node);
            }
        }

        // Add a 4th node
        let mut ring_after = three_node_ring();
        ring_after.add_node(ClusterNode::new("node-d", "10.0.0.4:8080"));

        let mut moved = 0;
        for i in 0..100 {
            let key = format!("key-{i}");
            if let Some(new_node) = ring_after.get_node_for_key(key.as_bytes()) {
                if let Some(old_node) = before.get(&key) {
                    if old_node != &new_node {
                        moved += 1;
                    }
                }
            }
        }

        // With consistent hashing, only ~1/N keys should move (25% for 4 nodes)
        // Allow generous margin
        assert!(
            moved < 60,
            "Too many keys moved: {moved}/100 (expected ~25%)"
        );
    }

    #[test]
    fn test_affected_ranges_on_add() {
        let ring = three_node_ring();
        let affected = ring.affected_ranges_on_add("node-d", 1);
        assert!(!affected.is_empty());
    }

    #[test]
    fn test_affected_ranges_on_remove() {
        let ring = three_node_ring();
        let affected = ring.affected_ranges_on_remove("node-b");
        assert!(!affected.is_empty());
        // All vnodes should be reassigned to remaining nodes
        let total_affected: usize = affected.values().sum();
        assert!(total_affected > 0);
    }

    // ── Statistics ──────────────────────────────────────────────────────────

    #[test]
    fn test_stats() {
        let ring = three_node_ring();
        let stats = ring.stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.active_node_count, 3);
        assert_eq!(stats.vnode_count, 450); // 3 * 150
    }

    #[test]
    fn test_stats_with_loads() {
        let mut ring = three_node_ring();
        ring.increment_load("node-a");
        ring.increment_load("node-a");
        ring.increment_load("node-b");
        let stats = ring.stats();
        assert_eq!(stats.max_load, 2);
        assert_eq!(stats.min_load, 0);
    }

    #[test]
    fn test_stats_stddev() {
        let mut ring = three_node_ring();
        // Equal load -> stddev should be 0
        ring.increment_load("node-a");
        ring.increment_load("node-b");
        ring.increment_load("node-c");
        let stats = ring.stats();
        assert!(stats.load_stddev < 0.001);
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_single_node_ring() {
        let mut ring = default_ring();
        ring.add_node(ClusterNode::new("solo", "10.0.0.1:8080"));
        let node = ring.get_node_for_key(b"any-key");
        assert_eq!(node, Some("solo".to_string()));
    }

    #[test]
    fn test_all_nodes_inactive() {
        let mut ring = three_node_ring();
        ring.deactivate_node("node-a");
        ring.deactivate_node("node-b");
        ring.deactivate_node("node-c");
        assert!(ring.get_node_for_key(b"key").is_none());
    }

    #[test]
    fn test_replicas_with_one_active() {
        let mut ring = three_node_ring();
        ring.deactivate_node("node-a");
        ring.deactivate_node("node-b");
        let replicas = ring.get_replicas(b"key");
        assert_eq!(replicas.len(), 1);
        assert_eq!(replicas[0], "node-c");
    }

    // ── Hash functions ──────────────────────────────────────────────────────

    #[test]
    fn test_hash_key_consistency() {
        let h1 = hash_key(b"test");
        let h2 = hash_key(b"test");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_key_different_inputs() {
        let h1 = hash_key(b"hello");
        let h2 = hash_key(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_vnode_consistency() {
        let h1 = hash_vnode("node-a", 0);
        let h2 = hash_vnode("node-a", 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_vnode_different_indices() {
        let h1 = hash_vnode("node-a", 0);
        let h2 = hash_vnode("node-a", 1);
        assert_ne!(h1, h2);
    }

    // ── ClusterNode builder ─────────────────────────────────────────────────

    #[test]
    fn test_cluster_node_builder() {
        let node = ClusterNode::new("n1", "10.0.0.1:8080")
            .with_weight(3)
            .with_zone("us-east");
        assert_eq!(node.id, "n1");
        assert_eq!(node.weight, 3);
        assert_eq!(node.zone, Some("us-east".to_string()));
        assert!(node.active);
    }

    #[test]
    fn test_cluster_node_default_weight() {
        let node = ClusterNode::new("n1", "10.0.0.1:8080");
        assert_eq!(node.weight, 1);
    }

    // ── Config ──────────────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = HashRingConfig::default();
        assert_eq!(cfg.vnodes_per_node, 150);
        assert_eq!(cfg.replica_count, 3);
        assert!(cfg.load_factor.is_some());
    }

    #[test]
    fn test_config_access() {
        let ring = default_ring();
        assert_eq!(ring.config().vnodes_per_node, 150);
    }
}
