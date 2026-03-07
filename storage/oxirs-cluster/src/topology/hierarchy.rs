//! Hierarchical cluster topology for 1000+ node deployments
//!
//! Structure: Global → Regions → Availability Zones → Racks → Nodes
//! Supports rack-aware placement, network distance calculations, and topology digests.

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Geographic region (e.g., us-east-1, eu-west-1)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub id: String,
    pub display_name: String,
    pub latitude: f64,
    pub longitude: f64,
}

impl Region {
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        latitude: f64,
        longitude: f64,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            latitude,
            longitude,
        }
    }
}

/// Availability zone within a region
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvailabilityZone {
    pub id: String,
    pub region_id: String,
    pub display_name: String,
}

impl AvailabilityZone {
    pub fn new(
        id: impl Into<String>,
        region_id: impl Into<String>,
        display_name: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            region_id: region_id.into(),
            display_name: display_name.into(),
        }
    }
}

/// Physical rack within an AZ
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rack {
    pub id: String,
    pub az_id: String,
    /// Maximum nodes per rack
    pub capacity: u32,
}

impl Rack {
    pub fn new(id: impl Into<String>, az_id: impl Into<String>, capacity: u32) -> Self {
        Self {
            id: id.into(),
            az_id: az_id.into(),
            capacity,
        }
    }
}

/// Individual cluster node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub node_id: String,
    pub rack_id: String,
    pub az_id: String,
    pub region_id: String,
    pub address: String,
    pub port: u16,
    pub role: NodeRole,
    pub state: NodeState,
    pub capacity: NodeCapacity,
    pub tags: HashMap<String, String>,
}

impl ClusterNode {
    pub fn new(
        node_id: impl Into<String>,
        rack_id: impl Into<String>,
        az_id: impl Into<String>,
        region_id: impl Into<String>,
        address: impl Into<String>,
        port: u16,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            rack_id: rack_id.into(),
            az_id: az_id.into(),
            region_id: region_id.into(),
            address: address.into(),
            port,
            role: NodeRole::Replica,
            state: NodeState::Joining,
            capacity: NodeCapacity::default(),
            tags: HashMap::new(),
        }
    }
}

/// Role of a node in the cluster
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Full voting member, can be elected leader
    Primary,
    /// Full voting member, follows leader
    Replica,
    /// Read-only, no votes
    Observer,
    /// Query routing only, no data storage
    Coordinator,
}

/// Operational state of a node
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    Active,
    Joining,
    Leaving,
    Down,
    /// Suspected fault, pending investigation
    Quarantined,
}

/// Node hardware/resource capacity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeCapacity {
    pub cpu_cores: u32,
    pub memory_gb: u64,
    pub disk_gb: u64,
    pub network_gbps: f64,
}

impl NodeCapacity {
    pub fn new(cpu_cores: u32, memory_gb: u64, disk_gb: u64, network_gbps: f64) -> Self {
        Self {
            cpu_cores,
            memory_gb,
            disk_gb,
            network_gbps,
        }
    }
}

/// Summary digest for gossip protocol synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyDigest {
    pub version: u64,
    pub node_count: usize,
    pub region_count: usize,
    pub az_count: usize,
    /// FNV-1a hash of all node IDs for quick change detection
    pub hash: u64,
}

/// The cluster topology registry
///
/// Manages the complete hierarchical view: regions → AZs → racks → nodes.
/// Thread-safe via `Arc<RwLock<_>>` for node map mutations.
pub struct ClusterTopology {
    regions: HashMap<String, Region>,
    azs: HashMap<String, AvailabilityZone>,
    racks: HashMap<String, Rack>,
    nodes: Arc<RwLock<HashMap<String, ClusterNode>>>,
    version: u64,
}

impl Default for ClusterTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterTopology {
    /// Create an empty topology
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
            azs: HashMap::new(),
            racks: HashMap::new(),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            version: 0,
        }
    }

    // -------------------------------------------------------------------------
    // Mutating registration methods
    // -------------------------------------------------------------------------

    pub fn add_region(&mut self, region: Region) {
        self.regions.insert(region.id.clone(), region);
        self.version += 1;
    }

    pub fn add_az(&mut self, az: AvailabilityZone) -> Result<()> {
        if !self.regions.contains_key(&az.region_id) {
            return Err(ClusterError::Config(format!(
                "Region '{}' not found for AZ '{}'",
                az.region_id, az.id
            )));
        }
        self.azs.insert(az.id.clone(), az);
        self.version += 1;
        Ok(())
    }

    pub fn add_rack(&mut self, rack: Rack) -> Result<()> {
        if !self.azs.contains_key(&rack.az_id) {
            return Err(ClusterError::Config(format!(
                "AZ '{}' not found for rack '{}'",
                rack.az_id, rack.id
            )));
        }
        self.racks.insert(rack.id.clone(), rack);
        self.version += 1;
        Ok(())
    }

    pub fn register_node(&self, node: ClusterNode) -> Result<()> {
        // Validate rack exists
        if !self.racks.contains_key(&node.rack_id) {
            return Err(ClusterError::Config(format!(
                "Rack '{}' not found for node '{}'",
                node.rack_id, node.node_id
            )));
        }
        let mut nodes = self
            .nodes
            .write()
            .map_err(|e| ClusterError::Lock(format!("Failed to acquire node write lock: {}", e)))?;
        nodes.insert(node.node_id.clone(), node);
        Ok(())
    }

    pub fn deregister_node(&self, node_id: &str) -> bool {
        let Ok(mut nodes) = self.nodes.write() else {
            return false;
        };
        nodes.remove(node_id).is_some()
    }

    pub fn update_node_state(&self, node_id: &str, state: NodeState) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|e| ClusterError::Lock(format!("Failed to acquire node write lock: {}", e)))?;
        match nodes.get_mut(node_id) {
            Some(node) => {
                node.state = state;
                Ok(())
            }
            None => Err(ClusterError::Config(format!(
                "Node '{}' not found",
                node_id
            ))),
        }
    }

    // -------------------------------------------------------------------------
    // Query methods
    // -------------------------------------------------------------------------

    pub fn get_node(&self, node_id: &str) -> Option<ClusterNode> {
        let nodes = self.nodes.read().ok()?;
        nodes.get(node_id).cloned()
    }

    pub fn nodes_in_region(&self, region_id: &str) -> Vec<ClusterNode> {
        let Ok(nodes) = self.nodes.read() else {
            return Vec::new();
        };
        nodes
            .values()
            .filter(|n| n.region_id == region_id)
            .cloned()
            .collect()
    }

    pub fn nodes_in_az(&self, az_id: &str) -> Vec<ClusterNode> {
        let Ok(nodes) = self.nodes.read() else {
            return Vec::new();
        };
        nodes
            .values()
            .filter(|n| n.az_id == az_id)
            .cloned()
            .collect()
    }

    pub fn active_nodes(&self) -> Vec<ClusterNode> {
        let Ok(nodes) = self.nodes.read() else {
            return Vec::new();
        };
        nodes
            .values()
            .filter(|n| n.state == NodeState::Active)
            .cloned()
            .collect()
    }

    pub fn total_node_count(&self) -> usize {
        let Ok(nodes) = self.nodes.read() else {
            return 0;
        };
        nodes.len()
    }

    // -------------------------------------------------------------------------
    // Placement and routing
    // -------------------------------------------------------------------------

    /// Calculate data placement: which nodes hold which shards.
    ///
    /// Returns up to `replication_factor` distinct node IDs chosen to maximize
    /// rack diversity (prefer nodes on different racks).
    pub fn placement_for_shard(&self, shard_id: u64, replication_factor: usize) -> Vec<String> {
        let Ok(nodes) = self.nodes.read() else {
            return Vec::new();
        };
        let active: Vec<&ClusterNode> = nodes
            .values()
            .filter(|n| n.state == NodeState::Active)
            .collect();

        if active.is_empty() {
            return Vec::new();
        }

        // Deterministically order nodes by hashing (node_id XOR shard_id)
        let mut candidates: Vec<(&ClusterNode, u64)> = active
            .iter()
            .map(|n| {
                let score = fnv1a_str(&n.node_id) ^ shard_id.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                (*n, score)
            })
            .collect();
        candidates.sort_by_key(|(_, score)| *score);

        // Rack-aware selection: greedily pick nodes that add a new rack
        let mut selected: Vec<String> = Vec::with_capacity(replication_factor);
        let mut used_racks: HashSet<&str> = HashSet::new();

        // First pass: one per unique rack
        for (node, _) in &candidates {
            if selected.len() >= replication_factor {
                break;
            }
            if used_racks.insert(node.rack_id.as_str()) {
                selected.push(node.node_id.clone());
            }
        }

        // Second pass: fill remainder from any rack if needed
        for (node, _) in &candidates {
            if selected.len() >= replication_factor {
                break;
            }
            if !selected.contains(&node.node_id) {
                selected.push(node.node_id.clone());
            }
        }

        selected
    }

    /// Get rack-aware node selection: maximize rack diversity.
    pub fn rack_aware_selection(&self, count: usize) -> Vec<String> {
        let Ok(nodes) = self.nodes.read() else {
            return Vec::new();
        };
        let active: Vec<&ClusterNode> = nodes
            .values()
            .filter(|n| n.state == NodeState::Active)
            .collect();

        // Group by rack
        let mut by_rack: HashMap<&str, Vec<&ClusterNode>> = HashMap::new();
        for node in &active {
            by_rack.entry(node.rack_id.as_str()).or_default().push(node);
        }

        // Round-robin across racks
        let mut rack_ids: Vec<&str> = by_rack.keys().copied().collect();
        rack_ids.sort();
        let mut indices: HashMap<&str, usize> = HashMap::new();
        let mut result = Vec::with_capacity(count);

        while result.len() < count {
            let before_len = result.len();
            for rack_id in &rack_ids {
                if result.len() >= count {
                    break;
                }
                let idx = indices.entry(rack_id).or_insert(0);
                if let Some(rack_nodes) = by_rack.get(rack_id) {
                    if *idx < rack_nodes.len() {
                        result.push(rack_nodes[*idx].node_id.clone());
                        *idx += 1;
                    }
                }
            }
            // No progress - all racks exhausted
            if result.len() == before_len {
                break;
            }
        }

        result
    }

    /// Network distance between two nodes (in hops):
    /// - same rack   → 0
    /// - same AZ     → 1
    /// - same region → 2
    /// - different regions → 3
    pub fn network_distance(&self, node_a: &str, node_b: &str) -> u32 {
        let Ok(nodes) = self.nodes.read() else {
            return 3;
        };
        let a = match nodes.get(node_a) {
            Some(n) => n,
            None => return 3,
        };
        let b = match nodes.get(node_b) {
            Some(n) => n,
            None => return 3,
        };

        if a.rack_id == b.rack_id {
            0
        } else if a.az_id == b.az_id {
            1
        } else if a.region_id == b.region_id {
            2
        } else {
            3
        }
    }

    /// Generate a topology snapshot for gossip protocol
    pub fn topology_digest(&self) -> TopologyDigest {
        let Ok(nodes) = self.nodes.read() else {
            return TopologyDigest {
                version: self.version,
                node_count: 0,
                region_count: self.regions.len(),
                az_count: self.azs.len(),
                hash: 0,
            };
        };

        // Compute FNV-1a hash over sorted node IDs for determinism
        let mut sorted_ids: Vec<&str> = nodes.keys().map(|s| s.as_str()).collect();
        sorted_ids.sort();
        let hash = sorted_ids.iter().fold(0xcbf2_9ce4_8422_2325u64, |acc, id| {
            fnv1a_str_with_basis(acc, id)
        });

        TopologyDigest {
            version: self.version,
            node_count: nodes.len(),
            region_count: self.regions.len(),
            az_count: self.azs.len(),
            hash,
        }
    }

    /// Current topology version (increments on structural changes)
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Get region by ID
    pub fn get_region(&self, region_id: &str) -> Option<&Region> {
        self.regions.get(region_id)
    }

    /// Get all regions
    pub fn all_regions(&self) -> Vec<&Region> {
        self.regions.values().collect()
    }

    /// Get all AZs in a region
    pub fn azs_in_region(&self, region_id: &str) -> Vec<&AvailabilityZone> {
        self.azs
            .values()
            .filter(|az| az.region_id == region_id)
            .collect()
    }

    /// Get all racks in an AZ
    pub fn racks_in_az(&self, az_id: &str) -> Vec<&Rack> {
        self.racks.values().filter(|r| r.az_id == az_id).collect()
    }
}

// -------------------------------------------------------------------------
// FNV-1a hash helpers
// -------------------------------------------------------------------------

/// FNV-1a 64-bit hash of a string
pub(crate) fn fnv1a_str(s: &str) -> u64 {
    fnv1a_str_with_basis(0xcbf2_9ce4_8422_2325, s)
}

/// FNV-1a 64-bit hash with a given starting basis (for chaining)
pub(crate) fn fnv1a_str_with_basis(mut hash: u64, s: &str) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// FNV-1a 64-bit hash of a byte slice
pub(crate) fn fnv1a_bytes(data: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_topology() -> ClusterTopology {
        let mut t = ClusterTopology::new();
        t.add_region(Region::new(
            "us-east-1",
            "US East (N. Virginia)",
            37.926_868,
            -78.024_902,
        ));
        t.add_region(Region::new(
            "eu-west-1",
            "Europe (Ireland)",
            53.3498,
            -6.2603,
        ));

        t.add_az(AvailabilityZone::new(
            "us-east-1a",
            "us-east-1",
            "US East 1a",
        ))
        .unwrap();
        t.add_az(AvailabilityZone::new(
            "us-east-1b",
            "us-east-1",
            "US East 1b",
        ))
        .unwrap();
        t.add_az(AvailabilityZone::new(
            "eu-west-1a",
            "eu-west-1",
            "EU West 1a",
        ))
        .unwrap();

        t.add_rack(Rack::new("rack-east-1a-1", "us-east-1a", 20))
            .unwrap();
        t.add_rack(Rack::new("rack-east-1a-2", "us-east-1a", 20))
            .unwrap();
        t.add_rack(Rack::new("rack-east-1b-1", "us-east-1b", 20))
            .unwrap();
        t.add_rack(Rack::new("rack-eu-1a-1", "eu-west-1a", 20))
            .unwrap();
        t
    }

    fn register_nodes(t: &ClusterTopology) {
        let nodes = vec![
            ClusterNode {
                node_id: "n1".into(),
                rack_id: "rack-east-1a-1".into(),
                az_id: "us-east-1a".into(),
                region_id: "us-east-1".into(),
                address: "10.0.0.1".into(),
                port: 9000,
                role: NodeRole::Primary,
                state: NodeState::Active,
                capacity: NodeCapacity::default(),
                tags: HashMap::new(),
            },
            ClusterNode {
                node_id: "n2".into(),
                rack_id: "rack-east-1a-2".into(),
                az_id: "us-east-1a".into(),
                region_id: "us-east-1".into(),
                address: "10.0.0.2".into(),
                port: 9000,
                role: NodeRole::Replica,
                state: NodeState::Active,
                capacity: NodeCapacity::default(),
                tags: HashMap::new(),
            },
            ClusterNode {
                node_id: "n3".into(),
                rack_id: "rack-east-1b-1".into(),
                az_id: "us-east-1b".into(),
                region_id: "us-east-1".into(),
                address: "10.0.0.3".into(),
                port: 9000,
                role: NodeRole::Replica,
                state: NodeState::Active,
                capacity: NodeCapacity::default(),
                tags: HashMap::new(),
            },
            ClusterNode {
                node_id: "n4".into(),
                rack_id: "rack-eu-1a-1".into(),
                az_id: "eu-west-1a".into(),
                region_id: "eu-west-1".into(),
                address: "10.1.0.1".into(),
                port: 9000,
                role: NodeRole::Replica,
                state: NodeState::Active,
                capacity: NodeCapacity::default(),
                tags: HashMap::new(),
            },
        ];
        for node in nodes {
            t.register_node(node).unwrap();
        }
    }

    #[test]
    fn test_topology_registration() {
        let t = make_topology();
        assert_eq!(t.regions.len(), 2);
        assert_eq!(t.azs.len(), 3);
        assert_eq!(t.racks.len(), 4);
    }

    #[test]
    fn test_node_registration_and_lookup() {
        let t = make_topology();
        register_nodes(&t);
        assert_eq!(t.total_node_count(), 4);
        assert!(t.get_node("n1").is_some());
        assert!(t.get_node("nonexistent").is_none());
    }

    #[test]
    fn test_nodes_in_region() {
        let t = make_topology();
        register_nodes(&t);
        let us_nodes = t.nodes_in_region("us-east-1");
        assert_eq!(us_nodes.len(), 3);
        let eu_nodes = t.nodes_in_region("eu-west-1");
        assert_eq!(eu_nodes.len(), 1);
    }

    #[test]
    fn test_active_nodes() {
        let t = make_topology();
        register_nodes(&t);
        t.update_node_state("n4", NodeState::Down).unwrap();
        let active = t.active_nodes();
        assert_eq!(active.len(), 3);
    }

    #[test]
    fn test_network_distance() {
        let t = make_topology();
        register_nodes(&t);
        // same rack
        assert_eq!(t.network_distance("n1", "n1"), 0);
        // different rack, same AZ
        assert_eq!(t.network_distance("n1", "n2"), 1);
        // different AZ, same region
        assert_eq!(t.network_distance("n1", "n3"), 2);
        // cross-region
        assert_eq!(t.network_distance("n1", "n4"), 3);
    }

    #[test]
    fn test_placement_for_shard_rack_diversity() {
        let t = make_topology();
        register_nodes(&t);
        // Request 3 replicas: should span 3 different racks
        let placement = t.placement_for_shard(42, 3);
        assert_eq!(placement.len(), 3);
        // All IDs must be unique
        let unique: HashSet<&String> = placement.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_rack_aware_selection() {
        let t = make_topology();
        register_nodes(&t);
        let selected = t.rack_aware_selection(3);
        assert_eq!(selected.len(), 3);
        // Should span distinct racks
        let nodes: Vec<ClusterNode> = selected.iter().filter_map(|id| t.get_node(id)).collect();
        let racks: HashSet<String> = nodes.iter().map(|n| n.rack_id.clone()).collect();
        assert!(
            racks.len() >= 2,
            "rack-aware selection should use multiple racks"
        );
    }

    #[test]
    fn test_topology_digest() {
        let t = make_topology();
        register_nodes(&t);
        let digest = t.topology_digest();
        assert_eq!(digest.node_count, 4);
        assert_eq!(digest.region_count, 2);
        assert_eq!(digest.az_count, 3);
        assert_ne!(digest.hash, 0);
    }

    #[test]
    fn test_add_az_missing_region_fails() {
        let mut t = ClusterTopology::new();
        let result = t.add_az(AvailabilityZone::new("az-1", "nonexistent", "Test AZ"));
        assert!(result.is_err());
    }

    #[test]
    fn test_add_rack_missing_az_fails() {
        let mut t = ClusterTopology::new();
        t.add_region(Region::new("r1", "R1", 0.0, 0.0));
        let result = t.add_rack(Rack::new("rack-1", "nonexistent-az", 10));
        assert!(result.is_err());
    }

    #[test]
    fn test_fnv1a_determinism() {
        let h1 = fnv1a_str("hello-world");
        let h2 = fnv1a_str("hello-world");
        assert_eq!(h1, h2);
        let h3 = fnv1a_str("different");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_1000_node_registration() {
        let mut t = ClusterTopology::new();
        t.add_region(Region::new("us-east-1", "US East", 37.9, -78.0));
        // Create 5 AZs
        for az_idx in 0..5 {
            let az_id = format!("us-east-1{}", (b'a' + az_idx) as char);
            t.add_az(AvailabilityZone::new(
                az_id.clone(),
                "us-east-1",
                format!("AZ {}", az_idx),
            ))
            .unwrap();
            // Create 10 racks per AZ
            for rack_idx in 0..10 {
                let rack_id = format!("{}-rack-{}", az_id, rack_idx);
                t.add_rack(Rack::new(rack_id.clone(), az_id.clone(), 30))
                    .unwrap();
                // Register 20 nodes per rack
                for node_idx in 0..20 {
                    let node = ClusterNode {
                        node_id: format!("{}-node-{}", rack_id, node_idx),
                        rack_id: rack_id.clone(),
                        az_id: az_id.clone(),
                        region_id: "us-east-1".into(),
                        address: format!("10.{}.{}.{}", az_idx, rack_idx, node_idx),
                        port: 9000,
                        role: NodeRole::Replica,
                        state: NodeState::Active,
                        capacity: NodeCapacity::default(),
                        tags: HashMap::new(),
                    };
                    t.register_node(node).unwrap();
                }
            }
        }

        assert_eq!(t.total_node_count(), 1000);
        let active = t.active_nodes();
        assert_eq!(active.len(), 1000);

        // Placement should be fast and correct for 1000+ nodes
        let placement = t.placement_for_shard(99999, 5);
        assert_eq!(placement.len(), 5);
        let unique: HashSet<&String> = placement.iter().collect();
        assert_eq!(unique.len(), 5, "All placement nodes must be distinct");

        let digest = t.topology_digest();
        assert_eq!(digest.node_count, 1000);
    }
}
