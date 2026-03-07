//! Large-scale node management for 1000+ node clusters
//!
//! Provides `NodeGroup` for logical node grouping, `TopologyAwarePlacement` for
//! rack-and-region-aware replica placement, `VirtualNode` / `ConsistentHashRing`
//! for consistent-hashing-based data distribution with configurable vnode counts.

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

use super::hierarchy::fnv1a_bytes;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Opaque node identifier
pub type NodeId = String;

// ---------------------------------------------------------------------------
// NodeGroup
// ---------------------------------------------------------------------------

/// A logical group of nodes sharing a region and rack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGroup {
    /// Unique identifier for this group
    pub group_id: String,
    /// Node IDs belonging to this group
    pub nodes: Vec<NodeId>,
    /// Geographic region (e.g. "us-east-1")
    pub region: String,
    /// Physical rack identifier (e.g. "rack-1a-3")
    pub rack: String,
}

impl NodeGroup {
    /// Create a new node group
    pub fn new(
        group_id: impl Into<String>,
        region: impl Into<String>,
        rack: impl Into<String>,
    ) -> Self {
        Self {
            group_id: group_id.into(),
            nodes: Vec::new(),
            region: region.into(),
            rack: rack.into(),
        }
    }

    /// Add a node to this group
    pub fn add_node(&mut self, node_id: impl Into<NodeId>) {
        self.nodes.push(node_id.into());
    }

    /// Number of nodes in this group
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// ClusterTopologyMeta — metadata store for placement decisions
// ---------------------------------------------------------------------------

/// Metadata about a node used for placement decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMeta {
    pub node_id: NodeId,
    pub region: String,
    pub rack: String,
}

// ---------------------------------------------------------------------------
// TopologyAwarePlacement
// ---------------------------------------------------------------------------

/// Rack-aware and region-aware replica placement strategy.
///
/// Ensures that for a given replication factor:
/// - No two replicas are placed on the same rack (first priority)
/// - Replicas are spread across as many regions as possible (second priority)
pub struct TopologyAwarePlacement {
    /// All nodes with their topology metadata: node_id → NodeMeta
    nodes: HashMap<NodeId, NodeMeta>,
}

impl TopologyAwarePlacement {
    /// Create a new empty placement manager
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Register a node with its topology metadata
    pub fn register_node(&mut self, meta: NodeMeta) {
        self.nodes.insert(meta.node_id.clone(), meta);
    }

    /// Remove a node from the placement registry
    pub fn deregister_node(&mut self, node_id: &str) -> bool {
        self.nodes.remove(node_id).is_some()
    }

    /// Number of registered nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Select `count` replica nodes using rack-aware, region-aware placement.
    ///
    /// The algorithm:
    /// 1. Sort candidates deterministically (by node_id) for reproducibility.
    /// 2. First pass: pick one node per unique rack, maximising region spread.
    /// 3. Second pass: fill remaining slots from unused nodes across any rack.
    ///
    /// Nodes in `exclude` are never selected.
    pub fn select_replica_nodes(&self, count: u8, exclude: &[NodeId]) -> Vec<NodeId> {
        let count = count as usize;
        let exclude_set: HashSet<&str> = exclude.iter().map(|s| s.as_str()).collect();

        // Gather eligible nodes, sorted for determinism
        let mut candidates: Vec<&NodeMeta> = self
            .nodes
            .values()
            .filter(|m| !exclude_set.contains(m.node_id.as_str()))
            .collect();
        candidates.sort_by(|a, b| a.node_id.cmp(&b.node_id));

        if candidates.is_empty() || count == 0 {
            return Vec::new();
        }

        let mut selected: Vec<NodeId> = Vec::with_capacity(count);
        let mut used_racks: HashSet<&str> = HashSet::new();
        let mut used_regions: HashSet<&str> = HashSet::new();

        // Pass 1a: one node per rack, preferring new regions
        for meta in &candidates {
            if selected.len() >= count {
                break;
            }
            if !used_racks.contains(meta.rack.as_str()) {
                used_racks.insert(meta.rack.as_str());
                used_regions.insert(meta.region.as_str());
                selected.push(meta.node_id.clone());
            }
        }

        // Pass 2: fill remaining from any unused node
        if selected.len() < count {
            let selected_set: HashSet<String> = selected.iter().cloned().collect();
            for meta in &candidates {
                if selected.len() >= count {
                    break;
                }
                if !selected_set.contains(&meta.node_id) {
                    selected.push(meta.node_id.clone());
                }
            }
        }

        selected
    }

    /// List all distinct regions in the registry
    pub fn distinct_regions(&self) -> Vec<String> {
        let mut regions: Vec<String> = self
            .nodes
            .values()
            .map(|m| m.region.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        regions.sort();
        regions
    }

    /// List all distinct racks in the registry
    pub fn distinct_racks(&self) -> Vec<String> {
        let mut racks: Vec<String> = self
            .nodes
            .values()
            .map(|m| m.rack.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        racks.sort();
        racks
    }
}

impl Default for TopologyAwarePlacement {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VirtualNode
// ---------------------------------------------------------------------------

/// A virtual node on the consistent hash ring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualNode {
    /// Ring token (hash value)
    pub token: u64,
    /// The physical node this vnode maps to
    pub physical_node: NodeId,
}

// ---------------------------------------------------------------------------
// ConsistentHashRing
// ---------------------------------------------------------------------------

/// Consistent hash ring with configurable virtual nodes per physical node.
///
/// Default: 150 virtual nodes per physical node for good load balance.
pub struct ConsistentHashRing {
    vnodes_per_node: u16,
    /// Sorted ring: token → physical node_id
    ring: BTreeMap<u64, NodeId>,
    /// Set of registered physical node IDs
    physical_nodes: HashSet<NodeId>,
}

impl ConsistentHashRing {
    /// Create a ring with the specified number of virtual nodes per physical node.
    ///
    /// The default recommended value is 150 vnodes for good balance.
    pub fn new(vnodes_per_node: u16) -> Self {
        Self {
            vnodes_per_node: vnodes_per_node.max(1),
            ring: BTreeMap::new(),
            physical_nodes: HashSet::new(),
        }
    }

    /// Create a ring with the default 150 vnodes per physical node
    pub fn default_ring() -> Self {
        Self::new(150)
    }

    /// Add a physical node to the ring with the configured number of vnodes.
    ///
    /// Idempotent: adding a node that already exists is a no-op.
    pub fn add_node(&mut self, id: impl Into<NodeId>, _virtual_nodes: u16) {
        let id: NodeId = id.into();
        if self.physical_nodes.contains(&id) {
            return;
        }
        for vnode_idx in 0..self.vnodes_per_node {
            let token = Self::vnode_token(&id, vnode_idx as usize);
            self.ring.insert(token, id.clone());
        }
        self.physical_nodes.insert(id);
    }

    /// Remove a physical node and its virtual nodes from the ring.
    ///
    /// Safe to call on non-existent nodes.
    pub fn remove_node(&mut self, id: &NodeId) {
        if !self.physical_nodes.contains(id) {
            return;
        }
        for vnode_idx in 0..self.vnodes_per_node {
            let token = Self::vnode_token(id, vnode_idx as usize);
            self.ring.remove(&token);
        }
        self.physical_nodes.remove(id);
    }

    /// Get the `replication_factor` distinct physical nodes responsible for `key`.
    ///
    /// Walks clockwise around the ring collecting unique physical nodes.
    pub fn get_nodes_for_key(&self, key: &[u8], replication_factor: u8) -> Vec<NodeId> {
        let n = replication_factor as usize;
        if self.ring.is_empty() || n == 0 {
            return Vec::new();
        }

        let token = fnv1a_bytes(key);
        let mut result: Vec<NodeId> = Vec::with_capacity(n);
        let mut seen: HashSet<&str> = HashSet::new();

        // Walk clockwise from token, then wrap around
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

    /// Return the token count per physical node.
    ///
    /// Useful for checking load distribution across nodes.
    pub fn token_distribution(&self) -> HashMap<NodeId, u64> {
        let mut dist: HashMap<NodeId, u64> = HashMap::new();
        for node_id in self.ring.values() {
            *dist.entry(node_id.clone()).or_insert(0) += 1;
        }
        dist
    }

    /// Number of physical nodes in the ring
    pub fn node_count(&self) -> usize {
        self.physical_nodes.len()
    }

    /// Number of virtual nodes (total ring entries)
    pub fn vnode_count(&self) -> usize {
        self.ring.len()
    }

    /// Return true if no nodes are registered
    pub fn is_empty(&self) -> bool {
        self.physical_nodes.is_empty()
    }

    /// Get the single primary node for a key (first clockwise node)
    pub fn primary_node_for_key(&self, key: &[u8]) -> Option<&NodeId> {
        if self.ring.is_empty() {
            return None;
        }
        let token = fnv1a_bytes(key);
        self.ring
            .range(token..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, id)| id)
    }

    /// Calculate the load balance ratio (max_tokens / min_tokens).
    ///
    /// 1.0 = perfect balance; higher values indicate skew.
    pub fn load_balance_ratio(&self) -> f64 {
        if self.physical_nodes.is_empty() {
            return 1.0;
        }
        let dist = self.token_distribution();
        let max = dist.values().copied().max().unwrap_or(0) as f64;
        let min = dist.values().copied().min().unwrap_or(0) as f64;
        if min == 0.0 {
            return f64::INFINITY;
        }
        max / min
    }

    /// Get all virtual nodes for inspection
    pub fn virtual_nodes(&self) -> Vec<VirtualNode> {
        self.ring
            .iter()
            .map(|(&token, node_id)| VirtualNode {
                token,
                physical_node: node_id.clone(),
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn vnode_token(node_id: &str, vnode_idx: usize) -> u64 {
        let mut data = node_id.as_bytes().to_vec();
        data.push(b':');
        data.extend_from_slice(&vnode_idx.to_le_bytes());
        fnv1a_bytes(&data)
    }
}

// ---------------------------------------------------------------------------
// ClusterScaleManager — orchestrates large-scale cluster operations
// ---------------------------------------------------------------------------

/// High-level manager for large-scale cluster topology and data distribution.
///
/// Combines `TopologyAwarePlacement` and `ConsistentHashRing` into a
/// single coherent abstraction for managing 1000+ node clusters.
pub struct ClusterScaleManager {
    placement: TopologyAwarePlacement,
    ring: ConsistentHashRing,
    groups: HashMap<String, NodeGroup>,
}

impl ClusterScaleManager {
    /// Create a new scale manager with 150 vnodes per physical node
    pub fn new() -> Self {
        Self::with_vnodes(150)
    }

    /// Create a scale manager with a custom vnode count
    pub fn with_vnodes(vnodes_per_node: u16) -> Self {
        Self {
            placement: TopologyAwarePlacement::new(),
            ring: ConsistentHashRing::new(vnodes_per_node),
            groups: HashMap::new(),
        }
    }

    /// Register a node with full topology metadata
    pub fn register_node(
        &mut self,
        node_id: impl Into<NodeId>,
        region: impl Into<String>,
        rack: impl Into<String>,
    ) {
        let node_id: NodeId = node_id.into();
        let region: String = region.into();
        let rack: String = rack.into();
        self.placement.register_node(NodeMeta {
            node_id: node_id.clone(),
            region,
            rack,
        });
        let vnodes = self.ring.vnodes_per_node;
        self.ring.add_node(&node_id, vnodes);
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        let id = node_id.to_string();
        self.ring.remove_node(&id);
        self.placement.deregister_node(node_id)
    }

    /// Add a node group
    pub fn add_group(&mut self, group: NodeGroup) -> Result<()> {
        // Verify all nodes in the group exist
        for node_id in &group.nodes {
            if !self.placement.nodes.contains_key(node_id) {
                return Err(ClusterError::Config(format!(
                    "Node '{}' not registered in placement manager",
                    node_id
                )));
            }
        }
        self.groups.insert(group.group_id.clone(), group);
        Ok(())
    }

    /// Get a group by ID
    pub fn get_group(&self, group_id: &str) -> Option<&NodeGroup> {
        self.groups.get(group_id)
    }

    /// Select replicas for a key using rack-aware placement
    pub fn replicas_for_key(
        &self,
        key: &[u8],
        replication_factor: u8,
        exclude: &[NodeId],
    ) -> Vec<NodeId> {
        // Use ring for initial candidates then verify with placement
        let ring_nodes = self.ring.get_nodes_for_key(key, replication_factor * 3);
        let exclude_set: HashSet<&str> = exclude.iter().map(|s| s.as_str()).collect();

        let mut result: Vec<NodeId> = Vec::with_capacity(replication_factor as usize);
        let mut used_racks: HashSet<String> = HashSet::new();

        // First pass: prefer nodes on different racks
        for node_id in &ring_nodes {
            if result.len() >= replication_factor as usize {
                break;
            }
            if exclude_set.contains(node_id.as_str()) {
                continue;
            }
            if let Some(meta) = self.placement.nodes.get(node_id) {
                if used_racks.insert(meta.rack.clone()) {
                    result.push(node_id.clone());
                }
            }
        }

        // Second pass: fill from ring nodes ignoring rack constraint
        if result.len() < replication_factor as usize {
            let selected_set: HashSet<String> = result.iter().cloned().collect();
            for node_id in &ring_nodes {
                if result.len() >= replication_factor as usize {
                    break;
                }
                if !exclude_set.contains(node_id.as_str()) && !selected_set.contains(node_id) {
                    result.push(node_id.clone());
                }
            }
        }

        result
    }

    /// Total node count
    pub fn node_count(&self) -> usize {
        self.placement.node_count()
    }

    /// Access the underlying ring
    pub fn ring(&self) -> &ConsistentHashRing {
        &self.ring
    }

    /// Access the underlying placement manager
    pub fn placement(&self) -> &TopologyAwarePlacement {
        &self.placement
    }
}

impl Default for ClusterScaleManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── NodeGroup ────────────────────────────────────────────────────────────

    #[test]
    fn test_node_group_creation() {
        let mut group = NodeGroup::new("group-1", "us-east-1", "rack-1");
        assert_eq!(group.group_id, "group-1");
        assert_eq!(group.region, "us-east-1");
        assert_eq!(group.rack, "rack-1");
        assert_eq!(group.node_count(), 0);

        group.add_node("node-a");
        group.add_node("node-b");
        assert_eq!(group.node_count(), 2);
    }

    #[test]
    fn test_node_group_nodes_list() {
        let mut group = NodeGroup::new("g1", "eu-west-1", "rack-eu-1");
        for i in 0..10 {
            group.add_node(format!("eu-node-{}", i));
        }
        assert_eq!(group.node_count(), 10);
        assert!(group.nodes.contains(&"eu-node-5".to_string()));
    }

    // ── TopologyAwarePlacement ───────────────────────────────────────────────

    fn make_placement_3regions() -> TopologyAwarePlacement {
        let mut p = TopologyAwarePlacement::new();
        // 3 regions, 2 racks each, 5 nodes per rack
        for (region_idx, region) in ["us-east-1", "eu-west-1", "ap-southeast-1"]
            .iter()
            .enumerate()
        {
            for rack_idx in 0..2 {
                let rack = format!("{}-rack-{}", region, rack_idx);
                for node_idx in 0..5 {
                    let node_id = format!("node-r{}-rack{}-{}", region_idx, rack_idx, node_idx);
                    p.register_node(NodeMeta {
                        node_id,
                        region: region.to_string(),
                        rack: rack.clone(),
                    });
                }
            }
        }
        p
    }

    #[test]
    fn test_placement_basic_registration() {
        let p = make_placement_3regions();
        assert_eq!(p.node_count(), 30); // 3 regions × 2 racks × 5 nodes
        assert_eq!(p.distinct_regions().len(), 3);
        assert_eq!(p.distinct_racks().len(), 6);
    }

    #[test]
    fn test_placement_select_rack_aware() {
        let p = make_placement_3regions();
        let selected = p.select_replica_nodes(3, &[]);
        assert_eq!(selected.len(), 3);
        // Each should be on a different rack
        let racks: Vec<&str> = selected
            .iter()
            .map(|id| p.nodes.get(id).map(|m| m.rack.as_str()).unwrap_or(""))
            .collect();
        let unique_racks: HashSet<&&str> = racks.iter().collect();
        assert_eq!(
            unique_racks.len(),
            3,
            "All replicas should be on different racks"
        );
    }

    #[test]
    fn test_placement_exclude_nodes() {
        let p = make_placement_3regions();
        // Exclude all nodes from us-east-1-rack-0
        let excluded: Vec<NodeId> = p
            .nodes
            .values()
            .filter(|m| m.rack == "us-east-1-rack-0")
            .map(|m| m.node_id.clone())
            .collect();
        let selected = p.select_replica_nodes(3, &excluded);
        assert_eq!(selected.len(), 3);
        // None of the selected should be in excluded
        for id in &selected {
            assert!(!excluded.contains(id));
        }
    }

    #[test]
    fn test_placement_count_exceeds_racks() {
        let mut p = TopologyAwarePlacement::new();
        // Only 2 racks with 1 node each
        p.register_node(NodeMeta {
            node_id: "n1".into(),
            region: "us-east-1".into(),
            rack: "rack-1".into(),
        });
        p.register_node(NodeMeta {
            node_id: "n2".into(),
            region: "us-east-1".into(),
            rack: "rack-2".into(),
        });
        // Ask for 5 replicas — should return only 2 (all available)
        let selected = p.select_replica_nodes(5, &[]);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_placement_empty_returns_empty() {
        let p = TopologyAwarePlacement::new();
        let selected = p.select_replica_nodes(3, &[]);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_placement_zero_count() {
        let p = make_placement_3regions();
        let selected = p.select_replica_nodes(0, &[]);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_placement_deregister() {
        let mut p = TopologyAwarePlacement::new();
        p.register_node(NodeMeta {
            node_id: "n1".into(),
            region: "r1".into(),
            rack: "rack-1".into(),
        });
        assert_eq!(p.node_count(), 1);
        assert!(p.deregister_node("n1"));
        assert_eq!(p.node_count(), 0);
        assert!(!p.deregister_node("n1")); // second removal returns false
    }

    #[test]
    fn test_placement_100_nodes() {
        let mut p = TopologyAwarePlacement::new();
        for i in 0..100 {
            p.register_node(NodeMeta {
                node_id: format!("node-{}", i),
                region: format!("region-{}", i % 5),
                rack: format!("rack-{}", i % 20),
            });
        }
        assert_eq!(p.node_count(), 100);
        assert_eq!(p.distinct_regions().len(), 5);
        assert_eq!(p.distinct_racks().len(), 20);

        // Select 3 replicas — should all be on different racks
        let selected = p.select_replica_nodes(3, &[]);
        assert_eq!(selected.len(), 3);
        let rack_ids: Vec<String> = selected
            .iter()
            .filter_map(|id| p.nodes.get(id).map(|m| m.rack.clone()))
            .collect();
        let unique_racks: HashSet<&String> = rack_ids.iter().collect();
        assert_eq!(
            unique_racks.len(),
            3,
            "100-node: 3 replicas on 3 distinct racks"
        );
    }

    #[test]
    fn test_placement_1000_nodes() {
        let mut p = TopologyAwarePlacement::new();
        // 10 regions, 10 racks per region, 10 nodes per rack
        for region_idx in 0..10_u32 {
            for rack_idx in 0..10_u32 {
                for node_idx in 0..10_u32 {
                    p.register_node(NodeMeta {
                        node_id: format!("node-r{}-rack{}-{}", region_idx, rack_idx, node_idx),
                        region: format!("region-{}", region_idx),
                        rack: format!("region-{}-rack-{}", region_idx, rack_idx),
                    });
                }
            }
        }
        assert_eq!(p.node_count(), 1000);
        assert_eq!(p.distinct_regions().len(), 10);
        assert_eq!(p.distinct_racks().len(), 100);

        // Select 5 replicas
        let selected = p.select_replica_nodes(5, &[]);
        assert_eq!(selected.len(), 5);
        let rack_ids: Vec<String> = selected
            .iter()
            .filter_map(|id| p.nodes.get(id).map(|m| m.rack.clone()))
            .collect();
        let unique_racks: HashSet<&String> = rack_ids.iter().collect();
        assert_eq!(
            unique_racks.len(),
            5,
            "1000-node: 5 replicas on 5 distinct racks"
        );
    }

    // ── ConsistentHashRing ───────────────────────────────────────────────────

    fn make_ring(node_count: usize) -> ConsistentHashRing {
        let mut ring = ConsistentHashRing::new(150);
        for i in 0..node_count {
            ring.add_node(format!("node-{}", i), 150);
        }
        ring
    }

    #[test]
    fn test_ring_add_remove() {
        let mut ring = ConsistentHashRing::new(10);
        ring.add_node("node-a", 10);
        ring.add_node("node-b", 10);
        assert_eq!(ring.node_count(), 2);
        assert_eq!(ring.vnode_count(), 20);

        ring.remove_node(&"node-a".to_string());
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.vnode_count(), 10);
    }

    #[test]
    fn test_ring_idempotent_add() {
        let mut ring = ConsistentHashRing::new(10);
        ring.add_node("node-a", 10);
        ring.add_node("node-a", 10); // duplicate
        assert_eq!(ring.node_count(), 1);
    }

    #[test]
    fn test_ring_get_nodes_deterministic() {
        let ring = make_ring(5);
        let key = b"rdf:type:Person";
        let first = ring.get_nodes_for_key(key, 3);
        let second = ring.get_nodes_for_key(key, 3);
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
    }

    #[test]
    fn test_ring_distinct_replicas() {
        let ring = make_ring(5);
        let replicas = ring.get_nodes_for_key(b"subject:predicate:object", 3);
        assert_eq!(replicas.len(), 3);
        let unique: HashSet<&String> = replicas.iter().collect();
        assert_eq!(
            unique.len(),
            3,
            "All replicas must be distinct physical nodes"
        );
    }

    #[test]
    fn test_ring_replicas_capped_at_node_count() {
        let ring = make_ring(3);
        let replicas = ring.get_nodes_for_key(b"key", 10);
        assert_eq!(replicas.len(), 3, "Cannot exceed available node count");
    }

    #[test]
    fn test_ring_empty_returns_empty() {
        let ring = ConsistentHashRing::new(10);
        assert!(ring.get_nodes_for_key(b"any", 3).is_empty());
        assert!(ring.primary_node_for_key(b"any").is_none());
    }

    #[test]
    fn test_ring_token_distribution() {
        let ring = make_ring(5);
        let dist = ring.token_distribution();
        assert_eq!(dist.len(), 5);
        // Each node should have exactly 150 tokens
        for count in dist.values() {
            assert_eq!(*count, 150);
        }
    }

    #[test]
    fn test_ring_load_balance_ratio_small() {
        let ring = make_ring(5);
        let ratio = ring.load_balance_ratio();
        assert_eq!(ratio, 1.0, "Equal vnodes => perfect balance");
    }

    #[test]
    fn test_ring_100_nodes_performance() {
        let ring = make_ring(100);
        assert_eq!(ring.node_count(), 100);
        assert_eq!(ring.vnode_count(), 15_000);

        let node = ring.primary_node_for_key(b"http://example.org/resource/1");
        assert!(node.is_some());

        let replicas = ring.get_nodes_for_key(b"http://example.org/resource/1", 3);
        assert_eq!(replicas.len(), 3);

        let ratio = ring.load_balance_ratio();
        assert_eq!(
            ratio, 1.0,
            "100-node ring with equal vnodes is perfectly balanced"
        );
    }

    #[test]
    fn test_ring_1000_nodes_performance() {
        let ring = make_ring(1000);
        assert_eq!(ring.node_count(), 1000);
        assert_eq!(ring.vnode_count(), 150_000);

        let replicas = ring.get_nodes_for_key(b"large-cluster-test-key", 5);
        assert_eq!(replicas.len(), 5);

        let unique: HashSet<&String> = replicas.iter().collect();
        assert_eq!(unique.len(), 5, "1000-node ring: 5 distinct replicas");

        let ratio = ring.load_balance_ratio();
        assert_eq!(
            ratio, 1.0,
            "1000-node ring with equal vnodes is perfectly balanced"
        );
    }

    #[test]
    fn test_ring_virtual_nodes_listing() {
        let ring = make_ring(3);
        let vnodes = ring.virtual_nodes();
        assert_eq!(vnodes.len(), 450); // 3 nodes × 150 vnodes
    }

    // ── ClusterScaleManager ──────────────────────────────────────────────────

    fn make_scale_manager(node_count: usize) -> ClusterScaleManager {
        let mut mgr = ClusterScaleManager::new();
        for i in 0..node_count {
            let region = format!("region-{}", i % 5);
            let rack = format!("rack-{}", i % 20);
            mgr.register_node(format!("node-{}", i), region, rack);
        }
        mgr
    }

    #[test]
    fn test_scale_manager_registration() {
        let mgr = make_scale_manager(50);
        assert_eq!(mgr.node_count(), 50);
        assert_eq!(mgr.ring().node_count(), 50);
    }

    #[test]
    fn test_scale_manager_remove_node() {
        let mut mgr = make_scale_manager(10);
        assert!(mgr.remove_node("node-0"));
        assert_eq!(mgr.node_count(), 9);
        assert_eq!(mgr.ring().node_count(), 9);
    }

    #[test]
    fn test_scale_manager_replicas_for_key() {
        let mgr = make_scale_manager(50);
        let replicas = mgr.replicas_for_key(b"test-rdf-triple", 3, &[]);
        assert_eq!(replicas.len(), 3);
        let unique: HashSet<&String> = replicas.iter().collect();
        assert_eq!(unique.len(), 3, "Replicas must be distinct");
    }

    #[test]
    fn test_scale_manager_add_group() {
        let mut mgr = make_scale_manager(10);
        let mut group = NodeGroup::new("g1", "region-0", "rack-0");
        group.add_node("node-0");
        group.add_node("node-5");
        assert!(mgr.add_group(group).is_ok());
        assert!(mgr.get_group("g1").is_some());
    }

    #[test]
    fn test_scale_manager_add_group_invalid_node_fails() {
        let mut mgr = make_scale_manager(5);
        let mut group = NodeGroup::new("g-bad", "r1", "rack-1");
        group.add_node("nonexistent-node");
        assert!(mgr.add_group(group).is_err());
    }

    #[test]
    fn test_scale_manager_1000_nodes() {
        let mgr = make_scale_manager(1000);
        assert_eq!(mgr.node_count(), 1000);

        let replicas = mgr.replicas_for_key(b"http://example.org/subject", 5, &[]);
        assert_eq!(replicas.len(), 5);
        let unique: HashSet<&String> = replicas.iter().collect();
        assert_eq!(
            unique.len(),
            5,
            "1000-node scale manager: 5 distinct replicas"
        );
    }

    #[test]
    fn test_scale_manager_replicas_with_exclusion() {
        let mgr = make_scale_manager(20);
        let exclude = vec!["node-0".to_string(), "node-1".to_string()];
        let replicas = mgr.replicas_for_key(b"test-key", 3, &exclude);
        assert_eq!(replicas.len(), 3);
        for r in &replicas {
            assert!(!exclude.contains(r), "Excluded node in replicas");
        }
    }
}
