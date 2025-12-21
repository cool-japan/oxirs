//! NUMA-aware memory management
//!
//! This module provides Non-Uniform Memory Access (NUMA) aware memory allocation
//! for optimal performance on multi-socket servers. It intelligently allocates
//! memory on NUMA nodes close to the CPU cores that will access the data.

use crate::error::{Result, TdbError};
use crate::storage::PAGE_SIZE;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// NUMA node identifier
pub type NumaNode = usize;

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes in the system
    pub num_nodes: usize,
    /// CPUs per NUMA node
    pub cpus_per_node: Vec<Vec<usize>>,
    /// Memory size per node (bytes)
    pub memory_per_node: Vec<u64>,
    /// Distance matrix between nodes (lower is closer)
    pub distance_matrix: Vec<Vec<u32>>,
}

impl NumaTopology {
    /// Detect NUMA topology automatically
    pub fn detect() -> Self {
        // In a real implementation, this would query the system
        // For now, provide a reasonable default
        Self::default()
    }

    /// Get the closest NUMA node to a given CPU
    pub fn closest_node_for_cpu(&self, cpu: usize) -> NumaNode {
        for (node, cpus) in self.cpus_per_node.iter().enumerate() {
            if cpus.contains(&cpu) {
                return node;
            }
        }
        0 // Default to node 0
    }

    /// Get the closest NUMA node to another node
    pub fn closest_node_to(&self, from_node: NumaNode) -> NumaNode {
        if from_node >= self.num_nodes {
            return 0;
        }

        let distances = &self.distance_matrix[from_node];
        let mut min_distance = u32::MAX;
        let mut closest = 0;

        for (node, &distance) in distances.iter().enumerate() {
            if node != from_node && distance < min_distance {
                min_distance = distance;
                closest = node;
            }
        }

        closest
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        // Default: 2-node NUMA system with 8 CPUs per node
        Self {
            num_nodes: 2,
            cpus_per_node: vec![
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                vec![8, 9, 10, 11, 12, 13, 14, 15],
            ],
            memory_per_node: vec![16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024], // 16GB per node
            distance_matrix: vec![vec![10, 20], vec![20, 10]], // Local=10, Remote=20
        }
    }
}

/// NUMA allocation policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaPolicy {
    /// Allocate on local node (preferred)
    Local,
    /// Interleave allocations across all nodes
    Interleave,
    /// Bind to specific node
    Bind(NumaNode),
    /// Prefer specific node but allow fallback
    Preferred(NumaNode),
}

/// NUMA memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct NumaStats {
    /// Bytes allocated per NUMA node
    pub bytes_per_node: Vec<u64>,
    /// Number of allocations per node
    pub allocations_per_node: Vec<u64>,
    /// Number of remote memory accesses
    pub remote_accesses: u64,
    /// Number of local memory accesses
    pub local_accesses: u64,
    /// Number of page migrations
    pub page_migrations: u64,
}

impl NumaStats {
    /// Calculate locality ratio (0.0 to 1.0, higher is better)
    pub fn locality_ratio(&self) -> f64 {
        let total = self.local_accesses + self.remote_accesses;
        if total == 0 {
            return 1.0;
        }
        self.local_accesses as f64 / total as f64
    }

    /// Get total bytes allocated
    pub fn total_bytes(&self) -> u64 {
        self.bytes_per_node.iter().sum()
    }

    /// Get total allocations
    pub fn total_allocations(&self) -> u64 {
        self.allocations_per_node.iter().sum()
    }
}

/// NUMA-aware memory allocator
pub struct NumaAllocator {
    /// NUMA topology
    topology: Arc<RwLock<NumaTopology>>,
    /// Allocation policy
    policy: NumaPolicy,
    /// Statistics
    stats: Arc<RwLock<NumaStats>>,
    /// Current node for interleave policy
    next_interleave_node: Arc<RwLock<NumaNode>>,
    /// Page allocations (page_id -> numa_node)
    page_allocations: Arc<RwLock<HashMap<u64, NumaNode>>>,
}

impl NumaAllocator {
    /// Create a new NUMA allocator with auto-detected topology
    pub fn new(policy: NumaPolicy) -> Self {
        let topology = NumaTopology::detect();
        let num_nodes = topology.num_nodes;

        Self {
            topology: Arc::new(RwLock::new(topology)),
            policy,
            stats: Arc::new(RwLock::new(NumaStats {
                bytes_per_node: vec![0; num_nodes],
                allocations_per_node: vec![0; num_nodes],
                ..Default::default()
            })),
            next_interleave_node: Arc::new(RwLock::new(0)),
            page_allocations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with specific topology
    pub fn with_topology(topology: NumaTopology, policy: NumaPolicy) -> Self {
        let num_nodes = topology.num_nodes;

        Self {
            topology: Arc::new(RwLock::new(topology)),
            policy,
            stats: Arc::new(RwLock::new(NumaStats {
                bytes_per_node: vec![0; num_nodes],
                allocations_per_node: vec![0; num_nodes],
                ..Default::default()
            })),
            next_interleave_node: Arc::new(RwLock::new(0)),
            page_allocations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Allocate a page on the appropriate NUMA node
    pub fn allocate_page(&self, page_id: u64) -> Result<NumaNode> {
        let node = self.select_node()?;

        // Record allocation
        {
            let mut stats = self.stats.write();
            if node < stats.bytes_per_node.len() {
                stats.bytes_per_node[node] += PAGE_SIZE as u64;
                stats.allocations_per_node[node] += 1;
            }
        }

        // Track page allocation
        {
            let mut allocations = self.page_allocations.write();
            allocations.insert(page_id, node);
        }

        Ok(node)
    }

    /// Get the NUMA node for a specific page
    pub fn get_page_node(&self, page_id: u64) -> Option<NumaNode> {
        let allocations = self.page_allocations.read();
        allocations.get(&page_id).copied()
    }

    /// Record a memory access
    pub fn record_access(&self, page_id: u64, accessing_cpu: usize) {
        let topology = self.topology.read();
        let allocated_node = self.get_page_node(page_id);
        let cpu_node = topology.closest_node_for_cpu(accessing_cpu);

        let mut stats = self.stats.write();
        if let Some(node) = allocated_node {
            if node == cpu_node {
                stats.local_accesses += 1;
            } else {
                stats.remote_accesses += 1;
            }
        }
    }

    /// Migrate a page to a different NUMA node
    pub fn migrate_page(&self, page_id: u64, target_node: NumaNode) -> Result<()> {
        let topology = self.topology.read();
        if target_node >= topology.num_nodes {
            return Err(TdbError::Other(format!(
                "Invalid NUMA node: {}",
                target_node
            )));
        }

        // Update allocation tracking
        {
            let mut allocations = self.page_allocations.write();
            if let Some(old_node) = allocations.get(&page_id) {
                // Update statistics
                let mut stats = self.stats.write();
                if *old_node < stats.bytes_per_node.len() {
                    stats.bytes_per_node[*old_node] -= PAGE_SIZE as u64;
                }
                if target_node < stats.bytes_per_node.len() {
                    stats.bytes_per_node[target_node] += PAGE_SIZE as u64;
                }
                stats.page_migrations += 1;
            }
            allocations.insert(page_id, target_node);
        }

        Ok(())
    }

    /// Suggest optimal NUMA node for a workload
    pub fn suggest_node_for_workload(&self, cpu_mask: &[usize]) -> NumaNode {
        if cpu_mask.is_empty() {
            return 0;
        }

        let topology = self.topology.read();

        // Count CPUs per node in the mask
        let mut cpu_counts = vec![0usize; topology.num_nodes];
        for &cpu in cpu_mask {
            let node = topology.closest_node_for_cpu(cpu);
            if node < cpu_counts.len() {
                cpu_counts[node] += 1;
            }
        }

        // Return node with most CPUs
        cpu_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(node, _)| node)
            .unwrap_or(0)
    }

    /// Get current statistics
    pub fn stats(&self) -> NumaStats {
        self.stats.read().clone()
    }

    /// Get topology information
    pub fn topology(&self) -> NumaTopology {
        self.topology.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let num_nodes = self.topology.read().num_nodes;
        *self.stats.write() = NumaStats {
            bytes_per_node: vec![0; num_nodes],
            allocations_per_node: vec![0; num_nodes],
            ..Default::default()
        };
    }

    /// Balance memory across NUMA nodes
    pub fn rebalance(&self) -> Result<Vec<(u64, NumaNode, NumaNode)>> {
        let stats = self.stats.read();
        let topology = self.topology.read();

        // Calculate average bytes per node
        let total_bytes: u64 = stats.bytes_per_node.iter().sum();
        let avg_bytes = total_bytes / topology.num_nodes as u64;
        let threshold = avg_bytes / 10; // 10% threshold

        let mut migrations = Vec::new();

        // Find imbalanced nodes
        for (node, &bytes) in stats.bytes_per_node.iter().enumerate() {
            if bytes > avg_bytes + threshold {
                // Overloaded node - suggest migrations
                let target_node = topology.closest_node_to(node);

                // Find some pages to migrate
                let allocations = self.page_allocations.read();
                for (&page_id, &alloc_node) in allocations.iter() {
                    if alloc_node == node {
                        migrations.push((page_id, node, target_node));
                        if migrations.len() >= 10 {
                            // Limit to 10 migrations per rebalance
                            break;
                        }
                    }
                }
            }
        }

        Ok(migrations)
    }

    // Private helper methods

    fn select_node(&self) -> Result<NumaNode> {
        let topology = self.topology.read();

        match self.policy {
            NumaPolicy::Local => {
                // Try to get current CPU's node
                Ok(Self::get_current_cpu_node(&topology))
            }
            NumaPolicy::Interleave => {
                let mut next_node = self.next_interleave_node.write();
                let node = *next_node;
                *next_node = (*next_node + 1) % topology.num_nodes;
                Ok(node)
            }
            NumaPolicy::Bind(node) => {
                if node >= topology.num_nodes {
                    Err(TdbError::Other(format!("Invalid NUMA node: {}", node)))
                } else {
                    Ok(node)
                }
            }
            NumaPolicy::Preferred(node) => {
                if node < topology.num_nodes {
                    Ok(node)
                } else {
                    Ok(0) // Fallback to node 0
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn get_current_cpu_node(topology: &NumaTopology) -> NumaNode {
        // In a real implementation, would use sched_getcpu()
        // For now, return node 0
        0
    }

    #[cfg(not(target_os = "linux"))]
    fn get_current_cpu_node(_topology: &NumaTopology) -> NumaNode {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_topology_default() {
        let topology = NumaTopology::default();
        assert_eq!(topology.num_nodes, 2);
        assert_eq!(topology.cpus_per_node.len(), 2);
        assert_eq!(topology.memory_per_node.len(), 2);
    }

    #[test]
    fn test_closest_node_for_cpu() {
        let topology = NumaTopology::default();
        assert_eq!(topology.closest_node_for_cpu(0), 0);
        assert_eq!(topology.closest_node_for_cpu(7), 0);
        assert_eq!(topology.closest_node_for_cpu(8), 1);
        assert_eq!(topology.closest_node_for_cpu(15), 1);
    }

    #[test]
    fn test_closest_node_to() {
        let topology = NumaTopology::default();
        assert_eq!(topology.closest_node_to(0), 1);
        assert_eq!(topology.closest_node_to(1), 0);
    }

    #[test]
    fn test_numa_allocator_local() {
        let allocator = NumaAllocator::new(NumaPolicy::Local);
        let node = allocator.allocate_page(0).unwrap();
        assert!(node < 2);

        let stats = allocator.stats();
        assert_eq!(stats.total_allocations(), 1);
        assert_eq!(stats.total_bytes(), PAGE_SIZE as u64);
    }

    #[test]
    fn test_numa_allocator_interleave() {
        let allocator = NumaAllocator::new(NumaPolicy::Interleave);

        let node1 = allocator.allocate_page(0).unwrap();
        let node2 = allocator.allocate_page(1).unwrap();

        // Should alternate between nodes
        assert_ne!(node1, node2);

        let stats = allocator.stats();
        assert_eq!(stats.total_allocations(), 2);
    }

    #[test]
    fn test_numa_allocator_bind() {
        let allocator = NumaAllocator::new(NumaPolicy::Bind(1));

        let node = allocator.allocate_page(0).unwrap();
        assert_eq!(node, 1);

        let node = allocator.allocate_page(1).unwrap();
        assert_eq!(node, 1);
    }

    #[test]
    fn test_numa_allocator_preferred() {
        let allocator = NumaAllocator::new(NumaPolicy::Preferred(0));

        let node = allocator.allocate_page(0).unwrap();
        assert_eq!(node, 0);
    }

    #[test]
    fn test_get_page_node() {
        let allocator = NumaAllocator::new(NumaPolicy::Bind(1));

        allocator.allocate_page(42).unwrap();
        let node = allocator.get_page_node(42);
        assert_eq!(node, Some(1));

        let node = allocator.get_page_node(999);
        assert_eq!(node, None);
    }

    #[test]
    fn test_record_access() {
        let allocator = NumaAllocator::new(NumaPolicy::Bind(0));

        allocator.allocate_page(0).unwrap();

        // Local access (CPU 0 is on node 0)
        allocator.record_access(0, 0);

        // Remote access (CPU 8 is on node 1)
        allocator.record_access(0, 8);

        let stats = allocator.stats();
        assert_eq!(stats.local_accesses, 1);
        assert_eq!(stats.remote_accesses, 1);
        assert_eq!(stats.locality_ratio(), 0.5);
    }

    #[test]
    fn test_migrate_page() {
        let allocator = NumaAllocator::new(NumaPolicy::Bind(0));

        allocator.allocate_page(0).unwrap();
        assert_eq!(allocator.get_page_node(0), Some(0));

        allocator.migrate_page(0, 1).unwrap();
        assert_eq!(allocator.get_page_node(0), Some(1));

        let stats = allocator.stats();
        assert_eq!(stats.page_migrations, 1);
    }

    #[test]
    fn test_suggest_node_for_workload() {
        let allocator = NumaAllocator::new(NumaPolicy::Local);

        // CPUs 0-3 are on node 0
        let node = allocator.suggest_node_for_workload(&[0, 1, 2, 3]);
        assert_eq!(node, 0);

        // CPUs 8-11 are on node 1
        let node = allocator.suggest_node_for_workload(&[8, 9, 10, 11]);
        assert_eq!(node, 1);

        // Mixed CPUs - should prefer node with more CPUs
        let node = allocator.suggest_node_for_workload(&[0, 1, 8]);
        assert_eq!(node, 0); // Node 0 has 2 CPUs, node 1 has 1
    }

    #[test]
    fn test_stats_calculations() {
        let allocator = NumaAllocator::new(NumaPolicy::Interleave);

        allocator.allocate_page(0).unwrap();
        allocator.allocate_page(1).unwrap();
        allocator.allocate_page(2).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats.total_allocations(), 3);
        assert_eq!(stats.total_bytes(), 3 * PAGE_SIZE as u64);
    }

    #[test]
    fn test_reset_stats() {
        let allocator = NumaAllocator::new(NumaPolicy::Local);

        allocator.allocate_page(0).unwrap();
        allocator.record_access(0, 0);

        allocator.reset_stats();

        let stats = allocator.stats();
        assert_eq!(stats.total_allocations(), 0);
        assert_eq!(stats.local_accesses, 0);
    }

    #[test]
    fn test_rebalance() {
        let allocator = NumaAllocator::new(NumaPolicy::Bind(0));

        // Allocate many pages on node 0
        for i in 0..20 {
            allocator.allocate_page(i).unwrap();
        }

        let migrations = allocator.rebalance().unwrap();
        assert!(!migrations.is_empty());

        // All migrations should be from node 0 to node 1
        for (_, from, to) in migrations {
            assert_eq!(from, 0);
            assert_eq!(to, 1);
        }
    }

    #[test]
    fn test_invalid_node_bind() {
        let allocator = NumaAllocator::new(NumaPolicy::Bind(999));
        let result = allocator.allocate_page(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_node_migration() {
        let allocator = NumaAllocator::new(NumaPolicy::Local);
        allocator.allocate_page(0).unwrap();

        let result = allocator.migrate_page(0, 999);
        assert!(result.is_err());
    }
}
