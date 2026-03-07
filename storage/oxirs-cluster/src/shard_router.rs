//! Consistent-hash shard routing for distributed cluster nodes.
//!
//! This module implements a consistent hash ring that maps arbitrary string keys
//! to cluster nodes.  Virtual nodes (v-nodes) provide uniform key distribution
//! even with a small number of physical nodes.  Replication is achieved by
//! walking clockwise on the ring to select `replication_factor` distinct nodes.

use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by shard routing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterError {
    /// The ring has no nodes and cannot route any keys.
    NoNodes,
    /// There are not enough active nodes to satisfy the replication factor.
    InsufficientNodes {
        /// Number of replicas required.
        needed: usize,
        /// Number of active nodes currently available.
        available: usize,
    },
    /// The named node is not present in the router.
    NodeNotFound(String),
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterError::NoNodes => write!(f, "No nodes available"),
            RouterError::InsufficientNodes { needed, available } => {
                write!(f, "Insufficient nodes: need {}, have {}", needed, available)
            }
            RouterError::NodeNotFound(id) => write!(f, "Node not found: {}", id),
        }
    }
}

impl std::error::Error for RouterError {}

// ---------------------------------------------------------------------------
// ShardNode
// ---------------------------------------------------------------------------

/// A physical node in the cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardNode {
    /// Unique node identifier.
    pub id: String,
    /// Host name or IP address.
    pub host: String,
    /// Port number.
    pub port: u16,
    /// Relative weight (higher = more v-nodes, more load).
    pub weight: u32,
    /// Whether this node is currently accepting new assignments.
    pub is_active: bool,
}

impl ShardNode {
    /// Create a new active node with weight 1.
    pub fn new(id: &str, host: &str, port: u16) -> Self {
        Self {
            id: id.to_string(),
            host: host.to_string(),
            port,
            weight: 1,
            is_active: true,
        }
    }

    /// Create a node with a custom weight.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }
}

// ---------------------------------------------------------------------------
// ShardAssignment
// ---------------------------------------------------------------------------

/// The routing result for a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardAssignment {
    /// The key that was routed.
    pub key: String,
    /// The primary node responsible for this key.
    pub primary_node: String,
    /// Additional replica nodes (excluding primary), in ring order.
    pub replica_nodes: Vec<String>,
}

// ---------------------------------------------------------------------------
// RouterConfig
// ---------------------------------------------------------------------------

/// Configuration for the shard router.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Number of replicas per key (including primary).
    pub replication_factor: usize,
    /// Number of virtual ring positions per physical node (per unit weight).
    pub virtual_nodes_per_shard: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            virtual_nodes_per_shard: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// FNV-1a 32-bit hash
// ---------------------------------------------------------------------------

const FNV_OFFSET_32: u32 = 2_166_136_261;
const FNV_PRIME_32: u32 = 16_777_619;

fn fnv1a_32(data: &[u8]) -> u32 {
    let mut h = FNV_OFFSET_32;
    for &byte in data {
        h ^= byte as u32;
        h = h.wrapping_mul(FNV_PRIME_32);
    }
    h
}

fn hash_key(key: &str) -> u32 {
    fnv1a_32(key.as_bytes())
}

fn hash_vnode(node_id: &str, vnode_index: usize) -> u32 {
    let combined = format!("{}#{}", node_id, vnode_index);
    fnv1a_32(combined.as_bytes())
}

// ---------------------------------------------------------------------------
// ShardRouter
// ---------------------------------------------------------------------------

/// A consistent hash ring router.
///
/// # Ring Layout
/// The ring has 2^32 positions (u32 wrap-around).  Each physical node occupies
/// `weight × virtual_nodes_per_shard` positions.  A key maps to the *next*
/// occupied position clockwise from `hash(key)`.  Replicas are the next
/// `replication_factor - 1` distinct physical nodes in clockwise order.
pub struct ShardRouter {
    config: RouterConfig,
    /// Physical nodes indexed by their ID.
    nodes: HashMap<String, ShardNode>,
    /// Sorted virtual-node ring: ring_position → node_id.
    ring: BTreeMap<u32, String>,
}

impl ShardRouter {
    /// Create a new, empty router with the given configuration.
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            ring: BTreeMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Node management
    // -----------------------------------------------------------------------

    /// Add a node to the ring.  If a node with the same ID already exists it
    /// is replaced (its old virtual nodes are removed first).
    pub fn add_node(&mut self, node: ShardNode) {
        // Remove any existing virtual nodes for this ID.
        if self.nodes.contains_key(&node.id) {
            self.remove_vnodes(&node.id.clone());
        }
        // Insert virtual nodes proportional to weight.
        let vnodes = (node.weight as usize) * self.config.virtual_nodes_per_shard;
        for i in 0..vnodes {
            let pos = hash_vnode(&node.id, i);
            self.ring.insert(pos, node.id.clone());
        }
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove a node from the ring by ID.
    ///
    /// Returns `true` if the node was present and removed.
    pub fn remove_node(&mut self, id: &str) -> bool {
        if self.nodes.remove(id).is_some() {
            self.remove_vnodes(id);
            true
        } else {
            false
        }
    }

    fn remove_vnodes(&mut self, id: &str) {
        self.ring.retain(|_, v| v != id);
    }

    // -----------------------------------------------------------------------
    // Routing
    // -----------------------------------------------------------------------

    /// Route `key` to a primary node and `replication_factor - 1` replicas.
    ///
    /// # Errors
    /// Returns `RouterError::NoNodes` if no nodes have been added.
    /// Returns `RouterError::InsufficientNodes` if fewer active nodes are
    /// available than the configured replication factor.
    pub fn route(&self, key: &str) -> Result<ShardAssignment, RouterError> {
        let active_nodes: Vec<&ShardNode> = self.nodes.values().filter(|n| n.is_active).collect();

        if active_nodes.is_empty() {
            return Err(RouterError::NoNodes);
        }
        if active_nodes.len() < self.config.replication_factor {
            return Err(RouterError::InsufficientNodes {
                needed: self.config.replication_factor,
                available: active_nodes.len(),
            });
        }

        let hash = hash_key(key);
        let selected = self.walk_ring(hash, self.config.replication_factor);

        let primary_node = selected.first().ok_or(RouterError::NoNodes)?.clone();
        let replica_nodes = selected[1..].to_vec();

        Ok(ShardAssignment {
            key: key.to_string(),
            primary_node,
            replica_nodes,
        })
    }

    /// Walk the ring clockwise from `start_hash` and collect up to `count`
    /// distinct *active* node IDs.
    fn walk_ring(&self, start_hash: u32, count: usize) -> Vec<String> {
        let mut selected: Vec<String> = Vec::with_capacity(count);

        // Two passes: from start_hash to u32::MAX, then from 0 to start_hash.
        let iter = self
            .ring
            .range(start_hash..)
            .chain(self.ring.range(..start_hash));

        for (_, node_id) in iter {
            if selected.contains(node_id) {
                continue;
            }
            // Only include active nodes.
            if let Some(node) = self.nodes.get(node_id) {
                if !node.is_active {
                    continue;
                }
            }
            selected.push(node_id.clone());
            if selected.len() == count {
                break;
            }
        }

        selected
    }

    // -----------------------------------------------------------------------
    // Rebalancing
    // -----------------------------------------------------------------------

    /// Compute which key ranges need to move when transitioning from
    /// `old_nodes` to `new_nodes`.
    ///
    /// Returns a list of `(key_range_start, key_range_end, new_node_id)` tuples
    /// describing the decimal hex representation of ring positions that should
    /// migrate.
    pub fn rebalance_keys(
        &self,
        old_nodes: &[ShardNode],
        new_nodes: &[ShardNode],
    ) -> Vec<(String, String, String)> {
        // Build temporary rings for old and new configurations.
        let old_ring = build_ring(old_nodes, self.config.virtual_nodes_per_shard);
        let new_ring = build_ring(new_nodes, self.config.virtual_nodes_per_shard);

        let mut moves: Vec<(String, String, String)> = Vec::new();

        // For each virtual node in the new ring, check if its owner changed.
        let new_keys: Vec<u32> = new_ring.keys().copied().collect();
        for i in 0..new_keys.len() {
            let pos = new_keys[i];
            let range_start = if i == 0 {
                0
            } else {
                new_keys[i - 1].wrapping_add(1)
            };

            let new_owner = new_ring[&pos].clone();

            // Find who owned this position in the old ring.
            let old_owner = ring_lookup(&old_ring, pos);

            if old_owner.as_deref() != Some(&new_owner) {
                moves.push((
                    format!("{:08x}", range_start),
                    format!("{:08x}", pos),
                    new_owner,
                ));
            }
        }

        moves
    }

    // -----------------------------------------------------------------------
    // Counts
    // -----------------------------------------------------------------------

    /// Return the total number of nodes (active + inactive).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of active nodes.
    pub fn active_node_count(&self) -> usize {
        self.nodes.values().filter(|n| n.is_active).count()
    }
}

// ---------------------------------------------------------------------------
// Helpers for rebalancing
// ---------------------------------------------------------------------------

fn build_ring(nodes: &[ShardNode], vnodes_per_shard: usize) -> BTreeMap<u32, String> {
    let mut ring = BTreeMap::new();
    for node in nodes {
        let vnodes = (node.weight as usize) * vnodes_per_shard;
        for i in 0..vnodes {
            let pos = hash_vnode(&node.id, i);
            ring.insert(pos, node.id.clone());
        }
    }
    ring
}

fn ring_lookup(ring: &BTreeMap<u32, String>, hash: u32) -> Option<String> {
    ring.range(hash..)
        .next()
        .or_else(|| ring.iter().next())
        .map(|(_, id)| id.clone())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_router(replication: usize) -> ShardRouter {
        let cfg = RouterConfig {
            replication_factor: replication,
            virtual_nodes_per_shard: 50,
        };
        ShardRouter::new(cfg)
    }

    fn add_nodes(router: &mut ShardRouter, n: usize) {
        for i in 0..n {
            let node = ShardNode::new(&format!("node{}", i), "127.0.0.1", 8000 + i as u16);
            router.add_node(node);
        }
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_route_no_nodes_error() {
        let router = make_router(1);
        assert_eq!(router.route("key"), Err(RouterError::NoNodes));
    }

    #[test]
    fn test_route_insufficient_nodes_error() {
        let mut router = make_router(3);
        router.add_node(ShardNode::new("n1", "127.0.0.1", 8001));
        let err = router.route("key");
        assert!(matches!(err, Err(RouterError::InsufficientNodes { .. })));
    }

    #[test]
    fn test_router_error_display() {
        let e = RouterError::NoNodes;
        assert!(!e.to_string().is_empty());
        let e2 = RouterError::NodeNotFound("x".to_string());
        assert!(e2.to_string().contains("x"));
        let e3 = RouterError::InsufficientNodes {
            needed: 3,
            available: 1,
        };
        assert!(e3.to_string().contains("3"));
    }

    // -----------------------------------------------------------------------
    // Basic routing
    // -----------------------------------------------------------------------

    #[test]
    fn test_route_returns_primary() {
        let mut router = make_router(1);
        add_nodes(&mut router, 3);
        let assignment = router.route("some_key").unwrap();
        assert!(!assignment.primary_node.is_empty());
    }

    #[test]
    fn test_route_rf1_no_replicas() {
        let mut router = make_router(1);
        add_nodes(&mut router, 3);
        let assignment = router.route("key").unwrap();
        assert!(assignment.replica_nodes.is_empty());
    }

    #[test]
    fn test_route_rf3_two_replicas() {
        let mut router = make_router(3);
        add_nodes(&mut router, 5);
        let assignment = router.route("key").unwrap();
        assert_eq!(assignment.replica_nodes.len(), 2);
    }

    #[test]
    fn test_route_primary_not_in_replicas() {
        let mut router = make_router(3);
        add_nodes(&mut router, 5);
        let assignment = router.route("key").unwrap();
        assert!(!assignment.replica_nodes.contains(&assignment.primary_node));
    }

    // -----------------------------------------------------------------------
    // Consistent hashing
    // -----------------------------------------------------------------------

    #[test]
    fn test_same_key_routes_to_same_node() {
        let mut router = make_router(1);
        add_nodes(&mut router, 5);
        let a1 = router.route("hello").unwrap();
        let a2 = router.route("hello").unwrap();
        assert_eq!(a1.primary_node, a2.primary_node);
    }

    #[test]
    fn test_different_keys_may_route_differently() {
        let mut router = make_router(1);
        add_nodes(&mut router, 5);
        // Just verify no panic — different keys may or may not map to the same node.
        let _a1 = router.route("apple").unwrap();
        let _a2 = router.route("banana").unwrap();
    }

    #[test]
    fn test_key_routing_stable_after_read_only() {
        let mut router = make_router(1);
        add_nodes(&mut router, 4);
        let before = router.route("my_key").unwrap().primary_node;
        let after = router.route("my_key").unwrap().primary_node;
        assert_eq!(before, after);
    }

    // -----------------------------------------------------------------------
    // Node add/remove
    // -----------------------------------------------------------------------

    #[test]
    fn test_node_count_after_add() {
        let mut router = make_router(1);
        assert_eq!(router.node_count(), 0);
        router.add_node(ShardNode::new("a", "h", 1));
        assert_eq!(router.node_count(), 1);
        router.add_node(ShardNode::new("b", "h", 2));
        assert_eq!(router.node_count(), 2);
    }

    #[test]
    fn test_node_count_after_remove() {
        let mut router = make_router(1);
        add_nodes(&mut router, 3);
        assert!(router.remove_node("node0"));
        assert_eq!(router.node_count(), 2);
    }

    #[test]
    fn test_remove_nonexistent_node_returns_false() {
        let mut router = make_router(1);
        assert!(!router.remove_node("ghost"));
    }

    #[test]
    fn test_active_node_count() {
        let mut router = make_router(1);
        let mut n1 = ShardNode::new("n1", "h", 1);
        let mut n2 = ShardNode::new("n2", "h", 2);
        n1.is_active = false;
        n2.is_active = true;
        router.add_node(n1);
        router.add_node(n2);
        assert_eq!(router.active_node_count(), 1);
    }

    #[test]
    fn test_route_skips_inactive_nodes() {
        let mut router = make_router(1);
        let mut inactive = ShardNode::new("inactive", "h", 9000);
        inactive.is_active = false;
        router.add_node(inactive);
        let active = ShardNode::new("active", "h", 9001);
        router.add_node(active);
        // Should route to "active" only
        let assignment = router.route("k").unwrap();
        assert_eq!(assignment.primary_node, "active");
    }

    #[test]
    fn test_all_inactive_gives_insufficient_nodes() {
        let mut router = make_router(1);
        let mut node = ShardNode::new("n1", "h", 1);
        node.is_active = false;
        router.add_node(node);
        assert_eq!(router.route("k"), Err(RouterError::NoNodes));
    }

    // -----------------------------------------------------------------------
    // Weighted nodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_weighted_node_has_more_vnodes() {
        let mut router = ShardRouter::new(RouterConfig {
            replication_factor: 1,
            virtual_nodes_per_shard: 10,
        });
        router.add_node(ShardNode::new("heavy", "h", 1).with_weight(3));
        router.add_node(ShardNode::new("light", "h", 2).with_weight(1));
        // With weight 3 vs 1 and 10 vnodes per shard, heavy has 30 ring positions,
        // light has 10 → heavy should win more keys.
        let heavy_count = (0..100)
            .filter(|i| router.route(&format!("key{}", i)).unwrap().primary_node == "heavy")
            .count();
        // Roughly 75% should go to heavy — use a loose bound
        assert!(
            heavy_count > 30,
            "Heavy node only got {} / 100 keys",
            heavy_count
        );
    }

    // -----------------------------------------------------------------------
    // Rebalance
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebalance_returns_vec() {
        let router = make_router(1);
        let old = vec![ShardNode::new("n1", "h", 1)];
        let new_nodes = vec![ShardNode::new("n1", "h", 1), ShardNode::new("n2", "h", 2)];
        let moves = router.rebalance_keys(&old, &new_nodes);
        // Some keys should move when adding a node
        assert!(!moves.is_empty());
    }

    #[test]
    fn test_rebalance_identical_configs_empty() {
        let router = make_router(1);
        let nodes = vec![ShardNode::new("n1", "h", 1)];
        let moves = router.rebalance_keys(&nodes, &nodes);
        assert!(moves.is_empty());
    }

    #[test]
    fn test_rebalance_format() {
        let router = make_router(1);
        let old = vec![ShardNode::new("a", "h", 1)];
        let new_nodes = vec![ShardNode::new("a", "h", 1), ShardNode::new("b", "h", 2)];
        let moves = router.rebalance_keys(&old, &new_nodes);
        for (start, end, node) in &moves {
            // start and end should be 8-char hex strings
            assert_eq!(start.len(), 8, "Bad start: {}", start);
            assert_eq!(end.len(), 8, "Bad end: {}", end);
            assert!(!node.is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------

    #[test]
    fn test_shard_node_new() {
        let n = ShardNode::new("id", "localhost", 9999);
        assert_eq!(n.id, "id");
        assert_eq!(n.port, 9999);
        assert!(n.is_active);
        assert_eq!(n.weight, 1);
    }

    #[test]
    fn test_shard_node_with_weight() {
        let n = ShardNode::new("n", "h", 1).with_weight(5);
        assert_eq!(n.weight, 5);
    }

    #[test]
    fn test_route_key_preserved() {
        let mut router = make_router(1);
        add_nodes(&mut router, 2);
        let assignment = router.route("mykey").unwrap();
        assert_eq!(assignment.key, "mykey");
    }
}
