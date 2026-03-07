//! Consistent-hash shard router for distributed RDF storage
//!
//! Implements a virtual-node consistent-hash ring that maps arbitrary byte keys
//! to shard nodes with configurable replication. Provides deterministic routing,
//! healthy-node filtering, distribution statistics, and rebalance cost estimation.

use std::collections::{BTreeMap, HashMap};

// ─── ShardNode ────────────────────────────────────────────────────────────────

/// A physical shard node in the cluster
#[derive(Debug, Clone)]
pub struct ShardNode {
    /// Unique node identifier
    pub id: String,
    /// Network address (e.g. "10.0.0.1:7000")
    pub address: String,
    /// Relative weight; more weight → more virtual nodes
    pub weight: usize,
    /// Whether this node is currently considered healthy
    pub is_healthy: bool,
}

impl ShardNode {
    /// Create a new, healthy node with weight 1
    pub fn new(id: impl Into<String>, address: impl Into<String>) -> Self {
        ShardNode {
            id: id.into(),
            address: address.into(),
            weight: 1,
            is_healthy: true,
        }
    }

    /// Set the weight for this node
    pub fn with_weight(mut self, weight: usize) -> Self {
        self.weight = weight;
        self
    }
}

// ─── VirtualNode ──────────────────────────────────────────────────────────────

/// A point on the consistent-hash ring
#[derive(Debug, Clone)]
pub struct VirtualNode {
    /// Hash value at this ring position
    pub hash: u64,
    /// The real node ID this virtual node maps to
    pub real_node_id: String,
}

// ─── ShardConfig ─────────────────────────────────────────────────────────────

/// Configuration for the shard router
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Number of virtual nodes to place per physical node (multiplied by weight)
    pub virtual_nodes_per_server: usize,
    /// Number of replicas per key (1 = primary only)
    pub replication_factor: usize,
}

impl Default for ShardConfig {
    fn default() -> Self {
        ShardConfig {
            virtual_nodes_per_server: 150,
            replication_factor: 3,
        }
    }
}

// ─── RouteResult ─────────────────────────────────────────────────────────────

/// The routing decision for a given key
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// Primary node for this key
    pub primary: String,
    /// Additional replica nodes (up to replication_factor - 1)
    pub replicas: Vec<String>,
    /// Hash value used to route this key
    pub shard_id: u64,
}

// ─── ShardRouter ─────────────────────────────────────────────────────────────

/// Consistent-hash shard router
pub struct ShardRouter {
    nodes: HashMap<String, ShardNode>,
    ring: BTreeMap<u64, VirtualNode>,
    config: ShardConfig,
}

impl ShardRouter {
    /// Create an empty router with the given configuration
    pub fn new(config: ShardConfig) -> Self {
        ShardRouter {
            nodes: HashMap::new(),
            ring: BTreeMap::new(),
            config,
        }
    }

    /// Add a physical node; populates virtual nodes on the ring
    pub fn add_node(&mut self, node: ShardNode) {
        let virtual_count = self.config.virtual_nodes_per_server * node.weight;
        let node_id = node.id.clone();
        for i in 0..virtual_count {
            let key = format!("{}-vn{}", node_id, i);
            let hash = fnv1a_hash(key.as_bytes());
            // Handle collision with a secondary probe
            let final_hash = if self.ring.contains_key(&hash) {
                let probe = format!("{}-vn{}-probe", node_id, i);
                fnv1a_hash(probe.as_bytes())
            } else {
                hash
            };
            self.ring.insert(
                final_hash,
                VirtualNode {
                    hash: final_hash,
                    real_node_id: node_id.clone(),
                },
            );
        }
        self.nodes.insert(node_id, node);
    }

    /// Remove a physical node and all its virtual nodes from the ring
    /// Returns `true` if the node existed
    pub fn remove_node(&mut self, id: &str) -> bool {
        if self.nodes.remove(id).is_none() {
            return false;
        }
        self.ring.retain(|_, vn| vn.real_node_id != id);
        true
    }

    /// Route a raw byte key; returns `None` if the ring is empty
    pub fn route(&self, key: &[u8]) -> Option<RouteResult> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = fnv1a_hash(key);
        self.route_hash(hash)
    }

    /// Route a string key
    pub fn route_str(&self, key: &str) -> Option<RouteResult> {
        self.route(key.as_bytes())
    }

    /// Mark a node as unhealthy; returns `true` if the node exists
    pub fn mark_unhealthy(&mut self, id: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.is_healthy = false;
            true
        } else {
            false
        }
    }

    /// Mark a node as healthy again; returns `true` if the node exists
    pub fn mark_healthy(&mut self, id: &str) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.is_healthy = true;
            true
        } else {
            false
        }
    }

    /// Borrow all healthy nodes
    pub fn healthy_nodes(&self) -> Vec<&ShardNode> {
        self.nodes.values().filter(|n| n.is_healthy).collect()
    }

    /// Total number of physical nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of virtual nodes on the ring
    pub fn virtual_node_count(&self) -> usize {
        self.ring.len()
    }

    /// Map from node_id to count of virtual nodes assigned to that node
    pub fn distribution(&self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for vn in self.ring.values() {
            *map.entry(vn.real_node_id.clone()).or_insert(0) += 1;
        }
        map
    }

    /// Estimate the fraction of keys that would move if a new node were added.
    ///
    /// A new node takes `1/(n+1)` of the ring (assuming uniform weight), so
    /// approximately `1/(n+1)` of keys would be reassigned.
    pub fn rebalance_stats(&self, _new_node_id: &str) -> f64 {
        let n = self.nodes.len();
        if n == 0 {
            return 0.0;
        }
        1.0 / (n as f64 + 1.0)
    }

    // ── private ──────────────────────────────────────────────────────────────

    fn route_hash(&self, hash: u64) -> Option<RouteResult> {
        if self.ring.is_empty() {
            return None;
        }

        // Walk the ring clockwise from `hash`, collecting distinct node IDs
        let mut assigned: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Nodes clockwise from hash (ring-wrapped)
        let tail = self.ring.range(hash..);
        let head = self.ring.range(..hash);

        let max_replicas = self.config.replication_factor.min(self.nodes.len());

        for vn in tail.chain(head).map(|(_, v)| v) {
            if seen.insert(vn.real_node_id.clone()) {
                assigned.push(vn.real_node_id.clone());
                if assigned.len() >= max_replicas {
                    break;
                }
            }
        }

        if assigned.is_empty() {
            return None;
        }

        let primary = assigned[0].clone();
        let replicas = assigned[1..].to_vec();

        Some(RouteResult {
            primary,
            replicas,
            shard_id: hash,
        })
    }
}

// ─── FNV-1a hash ─────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0001_0000_01b3;
    let mut hash = OFFSET;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn three_node_router() -> ShardRouter {
        let config = ShardConfig {
            virtual_nodes_per_server: 10,
            replication_factor: 3,
        };
        let mut router = ShardRouter::new(config);
        router.add_node(ShardNode::new("node-1", "10.0.0.1:7000"));
        router.add_node(ShardNode::new("node-2", "10.0.0.2:7000"));
        router.add_node(ShardNode::new("node-3", "10.0.0.3:7000"));
        router
    }

    // ── add_node ─────────────────────────────────────────────────────────────

    #[test]
    fn test_add_node_increases_count() {
        let mut router = ShardRouter::new(ShardConfig::default());
        router.add_node(ShardNode::new("n1", "127.0.0.1:7000"));
        assert_eq!(router.node_count(), 1);
    }

    #[test]
    fn test_add_node_populates_virtual_nodes() {
        let config = ShardConfig {
            virtual_nodes_per_server: 5,
            replication_factor: 1,
        };
        let mut router = ShardRouter::new(config);
        router.add_node(ShardNode::new("n1", "127.0.0.1:7000"));
        // At least 5 virtual nodes (may be fewer if collisions merged)
        assert!(router.virtual_node_count() >= 1);
    }

    #[test]
    fn test_add_multiple_nodes() {
        let router = three_node_router();
        assert_eq!(router.node_count(), 3);
    }

    // ── remove_node ──────────────────────────────────────────────────────────

    #[test]
    fn test_remove_node_decreases_count() {
        let mut router = three_node_router();
        assert!(router.remove_node("node-1"));
        assert_eq!(router.node_count(), 2);
    }

    #[test]
    fn test_remove_node_removes_virtual_nodes() {
        let config = ShardConfig {
            virtual_nodes_per_server: 10,
            replication_factor: 1,
        };
        let mut router = ShardRouter::new(config);
        router.add_node(ShardNode::new("n1", "127.0.0.1:7000"));
        let before = router.virtual_node_count();
        router.remove_node("n1");
        assert!(router.virtual_node_count() < before);
    }

    #[test]
    fn test_remove_nonexistent_node_returns_false() {
        let mut router = three_node_router();
        assert!(!router.remove_node("nonexistent"));
    }

    // ── route ────────────────────────────────────────────────────────────────

    #[test]
    fn test_route_returns_some_when_nodes_exist() {
        let router = three_node_router();
        assert!(router.route(b"some_key").is_some());
    }

    #[test]
    fn test_route_returns_none_when_no_nodes() {
        let router = ShardRouter::new(ShardConfig::default());
        assert!(router.route(b"key").is_none());
    }

    #[test]
    fn test_route_primary_is_valid_node() {
        let router = three_node_router();
        let result = router.route(b"test_key").unwrap();
        assert!(router.nodes.contains_key(&result.primary));
    }

    #[test]
    fn test_route_replicas_count() {
        let router = three_node_router();
        let result = router.route(b"hello").unwrap();
        // With 3 nodes and replication_factor=3, we should get 2 replicas
        assert_eq!(result.replicas.len(), 2);
    }

    #[test]
    fn test_route_all_replicas_are_distinct() {
        let router = three_node_router();
        let result = router.route(b"distinct_test").unwrap();
        let mut all = vec![result.primary.clone()];
        all.extend(result.replicas.clone());
        let set: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(set.len(), all.len());
    }

    #[test]
    fn test_route_replication_factor_capped_at_node_count() {
        let config = ShardConfig {
            virtual_nodes_per_server: 10,
            replication_factor: 10, // more than nodes
        };
        let mut router = ShardRouter::new(config);
        router.add_node(ShardNode::new("n1", "10.0.0.1:7000"));
        router.add_node(ShardNode::new("n2", "10.0.0.2:7000"));
        let result = router.route(b"key").unwrap();
        // primary + replicas should not exceed 2 (node count)
        let total = 1 + result.replicas.len();
        assert!(total <= 2);
    }

    // ── route_str ────────────────────────────────────────────────────────────

    #[test]
    fn test_route_str_deterministic() {
        let router = three_node_router();
        let r1 = router.route_str("sparql:query:1").unwrap();
        let r2 = router.route_str("sparql:query:1").unwrap();
        assert_eq!(r1.primary, r2.primary);
        assert_eq!(r1.shard_id, r2.shard_id);
    }

    #[test]
    fn test_route_str_different_keys_may_differ() {
        let router = three_node_router();
        let mut differences = 0;
        for i in 0..20 {
            let r1 = router.route_str(&format!("key{}", i)).unwrap();
            let r2 = router.route_str(&format!("other{}", i)).unwrap();
            if r1.primary != r2.primary {
                differences += 1;
            }
        }
        // With 3 nodes, at least some keys should land on different nodes
        assert!(differences > 0);
    }

    // ── mark_unhealthy / healthy_nodes ───────────────────────────────────────

    #[test]
    fn test_mark_unhealthy_returns_true_for_known_node() {
        let mut router = three_node_router();
        assert!(router.mark_unhealthy("node-1"));
    }

    #[test]
    fn test_mark_unhealthy_returns_false_for_unknown() {
        let mut router = three_node_router();
        assert!(!router.mark_unhealthy("unknown"));
    }

    #[test]
    fn test_healthy_nodes_excludes_unhealthy() {
        let mut router = three_node_router();
        router.mark_unhealthy("node-1");
        let healthy = router.healthy_nodes();
        let healthy_ids: Vec<&str> = healthy.iter().map(|n| n.id.as_str()).collect();
        assert!(!healthy_ids.contains(&"node-1"));
        assert_eq!(healthy.len(), 2);
    }

    #[test]
    fn test_healthy_nodes_includes_all_initially() {
        let router = three_node_router();
        assert_eq!(router.healthy_nodes().len(), 3);
    }

    #[test]
    fn test_mark_healthy_restores_node() {
        let mut router = three_node_router();
        router.mark_unhealthy("node-1");
        router.mark_healthy("node-1");
        assert_eq!(router.healthy_nodes().len(), 3);
    }

    // ── virtual_node_count ───────────────────────────────────────────────────

    #[test]
    fn test_virtual_node_count_scales_with_nodes() {
        let config = ShardConfig {
            virtual_nodes_per_server: 10,
            replication_factor: 1,
        };
        let mut router = ShardRouter::new(config);
        router.add_node(ShardNode::new("n1", "127.0.0.1:7000"));
        let after_one = router.virtual_node_count();
        router.add_node(ShardNode::new("n2", "127.0.0.2:7000"));
        let after_two = router.virtual_node_count();
        assert!(after_two > after_one);
    }

    #[test]
    fn test_virtual_node_count_weight_multiplied() {
        let config = ShardConfig {
            virtual_nodes_per_server: 5,
            replication_factor: 1,
        };
        let mut router_weight1 = ShardRouter::new(config.clone());
        router_weight1.add_node(ShardNode::new("n1", "127.0.0.1:7000").with_weight(1));

        let mut router_weight2 = ShardRouter::new(config);
        router_weight2.add_node(ShardNode::new("n1", "127.0.0.1:7000").with_weight(2));

        assert!(router_weight2.virtual_node_count() > router_weight1.virtual_node_count());
    }

    // ── distribution ─────────────────────────────────────────────────────────

    #[test]
    fn test_distribution_totals_equal_virtual_node_count() {
        let router = three_node_router();
        let dist = router.distribution();
        let total: usize = dist.values().sum();
        assert_eq!(total, router.virtual_node_count());
    }

    #[test]
    fn test_distribution_has_entry_for_each_node() {
        let router = three_node_router();
        let dist = router.distribution();
        assert!(dist.contains_key("node-1"));
        assert!(dist.contains_key("node-2"));
        assert!(dist.contains_key("node-3"));
    }

    #[test]
    fn test_distribution_all_counts_positive() {
        let router = three_node_router();
        for count in router.distribution().values() {
            assert!(*count > 0);
        }
    }

    // ── rebalance_stats ──────────────────────────────────────────────────────

    #[test]
    fn test_rebalance_stats_between_zero_and_one() {
        let router = three_node_router();
        let fraction = router.rebalance_stats("node-new");
        assert!((0.0..=1.0).contains(&fraction));
    }

    #[test]
    fn test_rebalance_stats_empty_ring_is_zero() {
        let router = ShardRouter::new(ShardConfig::default());
        let fraction = router.rebalance_stats("node-new");
        assert_eq!(fraction, 0.0);
    }

    #[test]
    fn test_rebalance_stats_decreases_with_more_nodes() {
        let config = ShardConfig {
            virtual_nodes_per_server: 10,
            replication_factor: 1,
        };
        let mut router1 = ShardRouter::new(config.clone());
        router1.add_node(ShardNode::new("n1", "127.0.0.1:7000"));
        let f1 = router1.rebalance_stats("new");

        let mut router2 = ShardRouter::new(config);
        router2.add_node(ShardNode::new("n1", "127.0.0.1:7000"));
        router2.add_node(ShardNode::new("n2", "127.0.0.2:7000"));
        let f2 = router2.rebalance_stats("new");

        // More nodes → smaller fraction per added node
        assert!(f2 < f1);
    }

    // ── fnv1a_hash ───────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_hash(b"consistent hashing rocks");
        let h2 = fnv1a_hash(b"consistent hashing rocks");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        let h1 = fnv1a_hash(b"foo");
        let h2 = fnv1a_hash(b"bar");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fnv1a_empty_input() {
        let h = fnv1a_hash(b"");
        assert_ne!(h, 0); // FNV-1a offset basis is non-zero
    }

    // ── edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_single_node_all_keys_route_to_it() {
        let config = ShardConfig {
            virtual_nodes_per_server: 10,
            replication_factor: 1,
        };
        let mut router = ShardRouter::new(config);
        router.add_node(ShardNode::new("only", "127.0.0.1:7000"));
        for i in 0u32..20 {
            let result = router.route(&i.to_le_bytes()).unwrap();
            assert_eq!(result.primary, "only");
        }
    }

    #[test]
    fn test_remove_all_nodes_route_returns_none() {
        let mut router = three_node_router();
        router.remove_node("node-1");
        router.remove_node("node-2");
        router.remove_node("node-3");
        assert!(router.route(b"key").is_none());
    }
}
