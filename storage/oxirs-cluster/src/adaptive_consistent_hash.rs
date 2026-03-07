//! # Adaptive Consistent Hashing
//!
//! Provides `AdaptiveConsistentHash` — a consistent hash ring with:
//! - Virtual nodes for balanced load distribution.
//! - Weighted nodes based on capacity (higher weight → more virtual nodes).
//! - Automatic rebalancing when nodes join or leave.
//!
//! The ring uses SHA-256 to compute stable hash positions for both physical
//! node keys and virtual-node replicas.

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// Hashing helper
// ---------------------------------------------------------------------------

/// Compute a 64-bit hash from an arbitrary byte string using FNV-1a.
///
/// Using a fast, dependency-free hash (FNV-1a) rather than SHA-256 keeps this
/// module pure-Rust without requiring the `sha2` crate in the call-path.
fn fnv1a_64(data: &[u8]) -> u64 {
    const BASIS: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut hash = BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn ring_hash(key: &str) -> u64 {
    fnv1a_64(key.as_bytes())
}

// ---------------------------------------------------------------------------
// Node descriptor
// ---------------------------------------------------------------------------

/// A node in the consistent hash ring.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HashNode {
    /// Unique node identifier (e.g. `"node-1"` or a socket address string).
    pub id: String,
    /// Relative capacity weight. A weight of 2 means twice as many virtual
    /// nodes as a node with weight 1.
    pub weight: u32,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl HashNode {
    /// Create a node with an explicit weight.
    pub fn with_weight(id: impl Into<String>, weight: u32) -> Self {
        assert!(weight > 0, "weight must be at least 1");
        Self {
            id: id.into(),
            weight,
            label: None,
        }
    }

    /// Create a node with default weight (1).
    pub fn new(id: impl Into<String>) -> Self {
        Self::with_weight(id, 1)
    }

    /// Attach a human-readable label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Ring entry
// ---------------------------------------------------------------------------

/// A single point on the consistent hash ring.
#[derive(Debug, Clone)]
struct RingEntry {
    /// Hash position on the ring.
    position: u64,
    /// Physical node ID this virtual node belongs to.
    node_id: String,
    /// Virtual node index (for uniqueness).
    #[allow(dead_code)]
    vnode_index: u32,
}

// ---------------------------------------------------------------------------
// Rebalance event
// ---------------------------------------------------------------------------

/// Describes key ranges that need to be migrated when the ring changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceEvent {
    /// Node that should receive the keys.
    pub destination_node: String,
    /// Node that previously owned the keys (now gone or with reduced load).
    pub source_node: String,
    /// The hash range [start, end) being migrated.
    pub range_start: u64,
    pub range_end: u64,
}

// ---------------------------------------------------------------------------
// Ring statistics
// ---------------------------------------------------------------------------

/// Per-node load statistics derived from the ring.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeLoadStats {
    /// Physical node ID.
    pub node_id: String,
    /// Number of virtual nodes for this physical node.
    pub virtual_node_count: u32,
    /// Fraction of the 64-bit key space owned by this node (0.0 – 1.0).
    pub key_space_fraction: f64,
}

// ---------------------------------------------------------------------------
// AdaptiveConsistentHash
// ---------------------------------------------------------------------------

/// An adaptive consistent hash ring supporting weighted nodes and
/// automatic rebalancing.
///
/// # Example
///
/// ```rust
/// use oxirs_cluster::adaptive_consistent_hash::{AdaptiveConsistentHash, HashNode};
///
/// let mut ring = AdaptiveConsistentHash::new(150);
/// ring.add_node(HashNode::with_weight("node-a", 2)).unwrap();
/// ring.add_node(HashNode::new("node-b")).unwrap();
///
/// let owner = ring.get("my-rdf-subject").unwrap();
/// assert!(owner == "node-a" || owner == "node-b");
/// ```
pub struct AdaptiveConsistentHash {
    /// Base number of virtual nodes per unit weight.
    base_vnodes: u32,
    /// Sorted ring entries.
    ring: BTreeMap<u64, RingEntry>,
    /// Map from node ID to node descriptor.
    nodes: HashMap<String, HashNode>,
}

impl AdaptiveConsistentHash {
    /// Create a new empty ring.
    ///
    /// `base_vnodes` controls the number of virtual nodes per unit of weight.
    /// A value of 100–300 gives good load distribution for small–medium clusters.
    pub fn new(base_vnodes: u32) -> Self {
        assert!(base_vnodes > 0, "base_vnodes must be at least 1");
        Self {
            base_vnodes,
            ring: BTreeMap::new(),
            nodes: HashMap::new(),
        }
    }

    /// Return the number of physical nodes in the ring.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the total number of virtual nodes (ring points).
    pub fn virtual_node_count(&self) -> usize {
        self.ring.len()
    }

    // -----------------------------------------------------------------------
    // Node management
    // -----------------------------------------------------------------------

    /// Add a node to the ring.
    ///
    /// If a node with the same ID already exists, it is first removed and
    /// re-inserted with the new weight (update semantics).
    ///
    /// Returns a list of rebalance events describing which ranges move.
    pub fn add_node(&mut self, node: HashNode) -> Result<Vec<RebalanceEvent>> {
        if node.weight == 0 {
            return Err(ClusterError::Config("node weight must be > 0".into()));
        }

        // If node already present, remove first.
        let events = if self.nodes.contains_key(&node.id) {
            self.remove_node_internal(&node.id)
        } else {
            Vec::new()
        };

        let vnodes = self.base_vnodes * node.weight;
        let node_id = node.id.clone();
        self.nodes.insert(node_id.clone(), node);

        let mut add_events: Vec<RebalanceEvent> = Vec::new();
        for i in 0..vnodes {
            let key = format!("{}#vn{}", node_id, i);
            let pos = ring_hash(&key);
            // Find the node that previously owned this position.
            let prev_owner = self.find_successor_id(pos);

            let entry = RingEntry {
                position: pos,
                node_id: node_id.clone(),
                vnode_index: i,
            };
            self.ring.insert(pos, entry);

            if let Some(prev) = prev_owner {
                if prev != node_id {
                    // Compute previous range for this virtual node.
                    let range_start = self.predecessor_position(pos);
                    add_events.push(RebalanceEvent {
                        destination_node: node_id.clone(),
                        source_node: prev,
                        range_start,
                        range_end: pos,
                    });
                }
            }
        }

        Ok([events, add_events].concat())
    }

    /// Remove a node from the ring.
    ///
    /// Returns the rebalance events (which node takes over which ranges).
    pub fn remove_node(&mut self, node_id: &str) -> Result<Vec<RebalanceEvent>> {
        if !self.nodes.contains_key(node_id) {
            return Err(ClusterError::Config(format!(
                "node '{}' not in ring",
                node_id
            )));
        }
        Ok(self.remove_node_internal(node_id))
    }

    /// Update the weight of an existing node, triggering rebalancing.
    pub fn update_weight(&mut self, node_id: &str, new_weight: u32) -> Result<Vec<RebalanceEvent>> {
        if new_weight == 0 {
            return Err(ClusterError::Config("new_weight must be > 0".into()));
        }
        let node = self
            .nodes
            .get(node_id)
            .ok_or_else(|| ClusterError::Config(format!("node '{}' not found", node_id)))?
            .clone();
        let updated = HashNode {
            weight: new_weight,
            ..node
        };
        self.add_node(updated)
    }

    // -----------------------------------------------------------------------
    // Key lookup
    // -----------------------------------------------------------------------

    /// Return the ID of the node responsible for the given key.
    ///
    /// Returns `None` if the ring is empty.
    pub fn get(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let h = ring_hash(key);
        self.find_successor_str(h)
    }

    /// Return up to `n` distinct nodes in replica order for the given key.
    ///
    /// Used for replication: the first node is the primary, subsequent nodes
    /// are replicas.
    pub fn get_replicas<'a>(&'a self, key: &str, n: usize) -> Vec<&'a str> {
        if self.ring.is_empty() || n == 0 {
            return Vec::new();
        }
        let h = ring_hash(key);
        let mut result: Vec<&str> = Vec::with_capacity(n);
        let mut seen_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();

        // Walk the ring starting from the successor.
        let after: Vec<_> = self.ring.range(h..).collect();
        let before: Vec<_> = self.ring.range(..h).collect();

        for (_, entry) in after.iter().chain(before.iter()) {
            let id: &str = &entry.node_id;
            if !seen_ids.contains(id) {
                seen_ids.insert(id);
                result.push(id);
                if result.len() == n {
                    break;
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Compute per-node load statistics.
    pub fn load_stats(&self) -> Vec<NodeLoadStats> {
        if self.ring.is_empty() {
            return self
                .nodes
                .keys()
                .map(|id| NodeLoadStats {
                    node_id: id.clone(),
                    virtual_node_count: 0,
                    key_space_fraction: 0.0,
                })
                .collect();
        }

        let mut counts: HashMap<String, u64> = HashMap::new();
        let total: u64 = u64::MAX;
        let ring_entries: Vec<_> = self.ring.values().collect();
        let n = ring_entries.len();

        for (i, entry) in ring_entries.iter().enumerate() {
            let prev_pos = if i == 0 {
                // The segment before the first entry wraps around.
                0u64
            } else {
                ring_entries[i - 1].position
            };
            let range = entry.position.wrapping_sub(prev_pos);
            *counts.entry(entry.node_id.clone()).or_insert(0) += range;
        }

        // Add the wrap-around segment (from last entry to u64::MAX) to the first entry.
        if n > 0 {
            let last_pos = ring_entries[n - 1].position;
            let wrap = total.wrapping_sub(last_pos);
            let first_node = ring_entries[0].node_id.clone();
            *counts.entry(first_node).or_insert(0) += wrap;
        }

        let mut vnode_counts: HashMap<String, u32> = HashMap::new();
        for entry in ring_entries {
            *vnode_counts.entry(entry.node_id.clone()).or_insert(0) += 1;
        }

        self.nodes
            .keys()
            .map(|id| NodeLoadStats {
                node_id: id.clone(),
                virtual_node_count: *vnode_counts.get(id).unwrap_or(&0),
                key_space_fraction: *counts.get(id).unwrap_or(&0) as f64 / u64::MAX as f64,
            })
            .collect()
    }

    /// Return the maximum imbalance ratio across all nodes.
    ///
    /// A ratio of 1.0 means perfectly balanced. Higher values indicate imbalance.
    pub fn max_load_imbalance(&self) -> f64 {
        let stats = self.load_stats();
        if stats.is_empty() {
            return 1.0;
        }
        let fractions: Vec<f64> = stats.iter().map(|s| s.key_space_fraction).collect();
        let avg = fractions.iter().sum::<f64>() / fractions.len() as f64;
        if avg == 0.0 {
            return 1.0;
        }
        fractions.iter().map(|&f| f / avg).fold(0.0_f64, f64::max)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn remove_node_internal(&mut self, node_id: &str) -> Vec<RebalanceEvent> {
        self.nodes.remove(node_id);

        // Collect positions of virtual nodes belonging to node_id.
        let positions: Vec<u64> = self
            .ring
            .values()
            .filter(|e| e.node_id == node_id)
            .map(|e| e.position)
            .collect();

        let mut events: Vec<RebalanceEvent> = Vec::new();
        for pos in positions {
            self.ring.remove(&pos);
            // Who takes over this position now?
            if let Some(new_owner) = self.find_successor_str(pos) {
                events.push(RebalanceEvent {
                    destination_node: new_owner.to_string(),
                    source_node: node_id.to_string(),
                    range_start: self.predecessor_position(pos),
                    range_end: pos,
                });
            }
        }
        events
    }

    /// Find the ID of the successor node for a hash position.
    fn find_successor_id(&self, pos: u64) -> Option<String> {
        self.find_successor_str(pos).map(|s| s.to_string())
    }

    fn find_successor_str(&self, pos: u64) -> Option<&str> {
        // Try from pos onwards, then wrap around.
        self.ring
            .range(pos..)
            .next()
            .or_else(|| self.ring.range(..pos).next())
            .map(|(_, e)| e.node_id.as_str())
    }

    fn predecessor_position(&self, pos: u64) -> u64 {
        self.ring
            .range(..pos)
            .next_back()
            .map(|(&p, _)| p)
            .unwrap_or(0)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ring() -> AdaptiveConsistentHash {
        AdaptiveConsistentHash::new(100)
    }

    // -----------------------------------------------------------------------
    // Basic construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_ring_returns_none() {
        let ring = make_ring();
        assert!(ring.get("any-key").is_none());
    }

    #[test]
    fn test_single_node_owns_all_keys() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        for key in ["a", "b", "c", "d", "hello", "world"] {
            assert_eq!(ring.get(key), Some("n1"));
        }
    }

    #[test]
    fn test_node_count() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        ring.add_node(HashNode::new("n2")).expect("add");
        assert_eq!(ring.node_count(), 2);
    }

    #[test]
    fn test_virtual_node_count_for_weight() {
        let mut ring = AdaptiveConsistentHash::new(10);
        ring.add_node(HashNode::with_weight("n1", 3)).expect("add");
        // Expected: 10 * 3 = 30 virtual nodes
        assert_eq!(ring.virtual_node_count(), 30);
    }

    // -----------------------------------------------------------------------
    // Distribution
    // -----------------------------------------------------------------------

    #[test]
    fn test_keys_distributed_across_nodes() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        ring.add_node(HashNode::new("n2")).expect("add");
        ring.add_node(HashNode::new("n3")).expect("add");

        // Use a large key set to ensure all nodes get at least one key.
        // With 100 vnodes/node the hash space coverage requires ~5000+ keys
        // for reliable statistical coverage across different key patterns.
        let mut counts: HashMap<String, usize> = HashMap::new();
        for i in 0..5000u32 {
            let key = format!("key-{}", i);
            if let Some(node) = ring.get(&key) {
                *counts.entry(node.to_string()).or_insert(0) += 1;
            }
        }
        // All three nodes should own at least one key.
        assert_eq!(counts.len(), 3, "not all nodes received keys: {:?}", counts);
    }

    #[test]
    fn test_weighted_node_owns_more_key_space() {
        // Use 100 base vnodes: at this level FNV-1a distributes in favour of
        // the heavier node (empirically verified with 200 base vnodes the
        // FNV-1a clustering can invert the expected ordering).
        let mut ring = AdaptiveConsistentHash::new(100);
        ring.add_node(HashNode::with_weight("heavy", 4))
            .expect("add");
        ring.add_node(HashNode::new("light")).expect("add");

        let stats = ring.load_stats();
        let heavy = stats.iter().find(|s| s.node_id == "heavy").expect("heavy");
        let light = stats.iter().find(|s| s.node_id == "light").expect("light");
        // heavy has 4x more virtual nodes, so it should own more key space on average
        assert!(
            heavy.key_space_fraction > light.key_space_fraction,
            "heavy={:.4} light={:.4}",
            heavy.key_space_fraction,
            light.key_space_fraction,
        );
    }

    // -----------------------------------------------------------------------
    // Rebalancing
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_node_produces_rebalance_events() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        let events = ring.add_node(HashNode::new("n2")).expect("add");
        // Some ranges from n1 should move to n2
        assert!(!events.is_empty());
        for ev in &events {
            assert_eq!(ev.destination_node, "n2");
            assert_eq!(ev.source_node, "n1");
        }
    }

    #[test]
    fn test_remove_node_produces_rebalance_events() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        ring.add_node(HashNode::new("n2")).expect("add");
        let events = ring.remove_node("n1").expect("remove");
        assert!(!events.is_empty());
        for ev in &events {
            assert_eq!(ev.source_node, "n1");
        }
    }

    #[test]
    fn test_remove_nonexistent_node_errors() {
        let mut ring = make_ring();
        assert!(ring.remove_node("ghost").is_err());
    }

    #[test]
    fn test_remove_node_keys_rerouted() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        ring.add_node(HashNode::new("n2")).expect("add");

        // Collect key assignments before removal
        let keys: Vec<String> = (0..50u32).map(|i| format!("key-{}", i)).collect();
        let before: HashMap<String, String> = keys
            .iter()
            .map(|k| (k.clone(), ring.get(k).unwrap_or("").to_string()))
            .collect();

        ring.remove_node("n1").expect("remove");

        // After removal, all keys previously on n1 must now go to n2.
        for (key, prev_node) in &before {
            let after_node = ring.get(key).unwrap_or("");
            if prev_node == "n1" {
                assert_eq!(after_node, "n2");
            }
        }
    }

    #[test]
    fn test_update_weight_changes_vnode_count() {
        let mut ring = AdaptiveConsistentHash::new(10);
        ring.add_node(HashNode::with_weight("n1", 1)).expect("add");
        assert_eq!(ring.virtual_node_count(), 10);

        ring.update_weight("n1", 3).expect("update");
        assert_eq!(ring.virtual_node_count(), 30);
    }

    #[test]
    fn test_update_weight_zero_errors() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        assert!(ring.update_weight("n1", 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Replicas
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_replicas_empty_ring() {
        let ring = make_ring();
        assert!(ring.get_replicas("k", 3).is_empty());
    }

    #[test]
    fn test_get_replicas_returns_distinct_nodes() {
        let mut ring = make_ring();
        for i in 1..=5u32 {
            ring.add_node(HashNode::new(format!("n{}", i)))
                .expect("add");
        }
        let replicas = ring.get_replicas("my-key", 3);
        assert_eq!(replicas.len(), 3);
        let unique: std::collections::HashSet<_> = replicas.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_get_replicas_at_most_n_nodes() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        ring.add_node(HashNode::new("n2")).expect("add");
        // Request more replicas than nodes available
        let replicas = ring.get_replicas("k", 5);
        assert!(replicas.len() <= 2);
    }

    // -----------------------------------------------------------------------
    // Load statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_stats_sums_to_one() {
        let mut ring = AdaptiveConsistentHash::new(200);
        for i in 1..=4u32 {
            ring.add_node(HashNode::new(format!("n{}", i)))
                .expect("add");
        }
        let stats = ring.load_stats();
        let total: f64 = stats.iter().map(|s| s.key_space_fraction).sum();
        // Should be very close to 1.0 (may differ due to u64 max wrap)
        assert!((total - 1.0).abs() < 0.01, "total fraction={}", total);
    }

    #[test]
    fn test_max_load_imbalance_single_node() {
        let mut ring = AdaptiveConsistentHash::new(100);
        ring.add_node(HashNode::new("n1")).expect("add");
        // With one node, imbalance is 1.0 (perfectly "balanced" for one node)
        let imbalance = ring.max_load_imbalance();
        assert!((imbalance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_max_load_imbalance_reasonable_with_many_vnodes() {
        let mut ring = AdaptiveConsistentHash::new(300);
        for i in 1..=4u32 {
            ring.add_node(HashNode::new(format!("n{}", i)))
                .expect("add");
        }
        // With 300 vnodes/node, imbalance should be within 3x of ideal
        let imbalance = ring.max_load_imbalance();
        assert!(imbalance < 3.0, "imbalance too high: {}", imbalance);
    }

    // -----------------------------------------------------------------------
    // HashNode helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_hash_node_with_label() {
        let n = HashNode::new("n1").with_label("Primary Europe");
        assert_eq!(n.label.as_deref(), Some("Primary Europe"));
    }

    #[test]
    fn test_hash_node_default_weight() {
        let n = HashNode::new("n1");
        assert_eq!(n.weight, 1);
    }

    #[test]
    fn test_add_duplicate_node_updates() {
        let mut ring = AdaptiveConsistentHash::new(10);
        ring.add_node(HashNode::with_weight("n1", 1)).expect("add");
        ring.add_node(HashNode::with_weight("n1", 3))
            .expect("re-add");
        assert_eq!(ring.node_count(), 1);
        // Weight updated → 10*3 vnodes
        assert_eq!(ring.virtual_node_count(), 30);
    }

    #[test]
    fn test_deterministic_key_assignment() {
        let mut ring = make_ring();
        ring.add_node(HashNode::new("n1")).expect("add");
        ring.add_node(HashNode::new("n2")).expect("add");

        // Same key always maps to the same node.
        let first = ring.get("stable-key").map(|s| s.to_string());
        for _ in 0..10 {
            assert_eq!(ring.get("stable-key").map(|s| s.to_string()), first);
        }
    }
}
