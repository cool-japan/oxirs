//! Database partitioning for horizontal scaling
//!
//! This module provides partitioning strategies to split large RDF datasets
//! across multiple physical storage units for better performance and manageability.

use crate::dictionary::NodeId;
use crate::error::{Result, TdbError};
use crate::index::Triple;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Partition identifier
pub type PartitionId = usize;

/// Partitioning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Hash-based partitioning (distributes data evenly)
    Hash,
    /// Range-based partitioning (preserves data locality)
    Range,
    /// Subject-based partitioning (all triples for a subject in same partition)
    Subject,
    /// Predicate-based partitioning (groups by predicate)
    Predicate,
    /// Graph-based partitioning (for quads, partitions by graph)
    Graph,
}

/// Partition metadata
#[derive(Debug, Clone)]
pub struct PartitionMetadata {
    /// Partition ID
    pub id: PartitionId,
    /// Number of triples in partition
    pub triple_count: usize,
    /// Estimated size in bytes
    pub size_bytes: u64,
    /// Partition strategy used
    pub strategy: PartitionStrategy,
    /// Optional range bounds (for range partitioning)
    pub range: Option<(NodeId, NodeId)>,
}

/// Partition statistics
#[derive(Debug, Clone, Default)]
pub struct PartitionStats {
    /// Total triples across all partitions
    pub total_triples: usize,
    /// Triples per partition
    pub triples_per_partition: Vec<usize>,
    /// Bytes per partition
    pub bytes_per_partition: Vec<u64>,
    /// Number of cross-partition queries
    pub cross_partition_queries: u64,
    /// Number of single-partition queries
    pub single_partition_queries: u64,
}

impl PartitionStats {
    /// Calculate partition balance ratio (0.0 to 1.0, higher is better)
    pub fn balance_ratio(&self) -> f64 {
        if self.triples_per_partition.is_empty() {
            return 1.0;
        }

        let total: usize = self.triples_per_partition.iter().sum();
        let avg = total as f64 / self.triples_per_partition.len() as f64;

        if avg == 0.0 {
            return 1.0;
        }

        // Calculate coefficient of variation
        let variance: f64 = self
            .triples_per_partition
            .iter()
            .map(|&count| {
                let diff = count as f64 - avg;
                diff * diff
            })
            .sum::<f64>()
            / self.triples_per_partition.len() as f64;

        let std_dev = variance.sqrt();
        let cv = std_dev / avg;

        // Convert to balance ratio (lower CV = higher balance)
        (1.0 - cv.min(1.0)).max(0.0)
    }

    /// Calculate query locality (percentage of single-partition queries)
    pub fn query_locality(&self) -> f64 {
        let total = self.single_partition_queries + self.cross_partition_queries;
        if total == 0 {
            return 1.0;
        }
        self.single_partition_queries as f64 / total as f64
    }
}

/// Partition manager
pub struct PartitionManager {
    /// Number of partitions
    num_partitions: usize,
    /// Partitioning strategy
    strategy: PartitionStrategy,
    /// Partition metadata
    partitions: Arc<RwLock<Vec<PartitionMetadata>>>,
    /// Statistics
    stats: Arc<RwLock<PartitionStats>>,
    /// Range partition bounds (for range-based partitioning)
    range_bounds: Arc<RwLock<Vec<NodeId>>>,
}

impl PartitionManager {
    /// Create a new partition manager
    pub fn new(num_partitions: usize, strategy: PartitionStrategy) -> Self {
        let partitions = (0..num_partitions)
            .map(|id| PartitionMetadata {
                id,
                triple_count: 0,
                size_bytes: 0,
                strategy,
                range: None,
            })
            .collect();

        Self {
            num_partitions,
            strategy,
            partitions: Arc::new(RwLock::new(partitions)),
            stats: Arc::new(RwLock::new(PartitionStats {
                triples_per_partition: vec![0; num_partitions],
                bytes_per_partition: vec![0; num_partitions],
                ..Default::default()
            })),
            range_bounds: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set range bounds for range-based partitioning
    pub fn set_range_bounds(&self, bounds: Vec<NodeId>) -> Result<()> {
        if bounds.len() != self.num_partitions - 1 {
            return Err(TdbError::Other(format!(
                "Expected {} bounds for {} partitions, got {}",
                self.num_partitions - 1,
                self.num_partitions,
                bounds.len()
            )));
        }

        // Verify bounds are sorted
        for i in 1..bounds.len() {
            if bounds[i] <= bounds[i - 1] {
                return Err(TdbError::Other("Range bounds must be sorted".to_string()));
            }
        }

        *self.range_bounds.write() = bounds;
        Ok(())
    }

    /// Get partition for a triple
    pub fn get_partition(&self, triple: &Triple) -> PartitionId {
        match self.strategy {
            PartitionStrategy::Hash => self.hash_partition(triple),
            PartitionStrategy::Range => self.range_partition(triple.subject),
            PartitionStrategy::Subject => self.subject_partition(triple.subject),
            PartitionStrategy::Predicate => self.predicate_partition(triple.predicate),
            PartitionStrategy::Graph => {
                // For quads, would partition by graph
                // For now, use subject as fallback
                self.subject_partition(triple.subject)
            }
        }
    }

    /// Get partitions that need to be queried for a triple pattern
    pub fn get_partitions_for_pattern(&self, pattern: &TriplePattern) -> Vec<PartitionId> {
        match self.strategy {
            PartitionStrategy::Hash => {
                // Hash partitioning requires querying all partitions for patterns
                (0..self.num_partitions).collect()
            }
            PartitionStrategy::Range => {
                if let Some(subject) = pattern.subject {
                    vec![self.range_partition(subject)]
                } else {
                    (0..self.num_partitions).collect()
                }
            }
            PartitionStrategy::Subject => {
                if let Some(subject) = pattern.subject {
                    vec![self.subject_partition(subject)]
                } else {
                    (0..self.num_partitions).collect()
                }
            }
            PartitionStrategy::Predicate => {
                if let Some(predicate) = pattern.predicate {
                    vec![self.predicate_partition(predicate)]
                } else {
                    (0..self.num_partitions).collect()
                }
            }
            PartitionStrategy::Graph => (0..self.num_partitions).collect(),
        }
    }

    /// Record a triple insertion
    pub fn record_insert(&self, partition_id: PartitionId, triple_size: u64) {
        let mut stats = self.stats.write();
        if partition_id < stats.triples_per_partition.len() {
            stats.triples_per_partition[partition_id] += 1;
            stats.bytes_per_partition[partition_id] += triple_size;
            stats.total_triples += 1;
        }

        let mut partitions = self.partitions.write();
        if let Some(partition) = partitions.get_mut(partition_id) {
            partition.triple_count += 1;
            partition.size_bytes += triple_size;
        }
    }

    /// Record a triple deletion
    pub fn record_delete(&self, partition_id: PartitionId, triple_size: u64) {
        let mut stats = self.stats.write();
        if partition_id < stats.triples_per_partition.len() {
            if stats.triples_per_partition[partition_id] > 0 {
                stats.triples_per_partition[partition_id] -= 1;
            }
            if stats.bytes_per_partition[partition_id] >= triple_size {
                stats.bytes_per_partition[partition_id] -= triple_size;
            }
            if stats.total_triples > 0 {
                stats.total_triples -= 1;
            }
        }

        let mut partitions = self.partitions.write();
        if let Some(partition) = partitions.get_mut(partition_id) {
            if partition.triple_count > 0 {
                partition.triple_count -= 1;
            }
            if partition.size_bytes >= triple_size {
                partition.size_bytes -= triple_size;
            }
        }
    }

    /// Record a query
    pub fn record_query(&self, partitions: &[PartitionId]) {
        let mut stats = self.stats.write();
        if partitions.len() == 1 {
            stats.single_partition_queries += 1;
        } else {
            stats.cross_partition_queries += 1;
        }
    }

    /// Get statistics
    pub fn stats(&self) -> PartitionStats {
        self.stats.read().clone()
    }

    /// Get partition metadata
    pub fn partitions(&self) -> Vec<PartitionMetadata> {
        self.partitions.read().clone()
    }

    /// Rebalance partitions (suggest migrations)
    pub fn suggest_rebalancing(&self) -> Vec<(PartitionId, PartitionId, usize)> {
        let stats = self.stats.read();
        let mut suggestions = Vec::new();

        if stats.triples_per_partition.is_empty() {
            return suggestions;
        }

        let avg = stats.total_triples / stats.triples_per_partition.len();
        let threshold = avg / 5; // 20% threshold

        // Find overloaded and underloaded partitions
        let mut overloaded = Vec::new();
        let mut underloaded = Vec::new();

        for (id, &count) in stats.triples_per_partition.iter().enumerate() {
            if count > avg + threshold {
                overloaded.push((id, count - avg));
            } else if count < avg - threshold {
                underloaded.push((id, avg - count));
            }
        }

        // Suggest migrations from overloaded to underloaded
        for (over_id, over_count) in overloaded {
            for (under_id, under_capacity) in &underloaded {
                let migrate_count = over_count.min(*under_capacity);
                if migrate_count > 0 {
                    suggestions.push((over_id, *under_id, migrate_count));
                }
            }
        }

        suggestions
    }

    // Private helper methods

    fn hash_partition(&self, triple: &Triple) -> PartitionId {
        // Use FNV-1a hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        triple.subject.hash(&mut hasher);
        triple.predicate.hash(&mut hasher);
        triple.object.hash(&mut hasher);
        let hash = hasher.finish();
        (hash as usize) % self.num_partitions
    }

    fn range_partition(&self, node_id: NodeId) -> PartitionId {
        let bounds = self.range_bounds.read();

        if bounds.is_empty() {
            // No bounds set, use hash-based fallback
            return (node_id.as_u64() as usize) % self.num_partitions;
        }

        // Binary search for the appropriate partition
        for (i, &bound) in bounds.iter().enumerate() {
            if node_id < bound {
                return i;
            }
        }

        // Last partition
        self.num_partitions - 1
    }

    fn subject_partition(&self, subject: NodeId) -> PartitionId {
        (subject.as_u64() as usize) % self.num_partitions
    }

    fn predicate_partition(&self, predicate: NodeId) -> PartitionId {
        (predicate.as_u64() as usize) % self.num_partitions
    }
}

/// Triple pattern for partition selection
#[derive(Debug, Clone, Copy)]
pub struct TriplePattern {
    /// Subject (None = wildcard)
    pub subject: Option<NodeId>,
    /// Predicate (None = wildcard)
    pub predicate: Option<NodeId>,
    /// Object (None = wildcard)
    pub object: Option<NodeId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: u64, p: u64, o: u64) -> Triple {
        Triple {
            subject: NodeId::new(s),
            predicate: NodeId::new(p),
            object: NodeId::new(o),
        }
    }

    #[test]
    fn test_partition_manager_creation() {
        let manager = PartitionManager::new(4, PartitionStrategy::Hash);
        assert_eq!(manager.num_partitions, 4);

        let partitions = manager.partitions();
        assert_eq!(partitions.len(), 4);
    }

    #[test]
    fn test_hash_partitioning() {
        let manager = PartitionManager::new(4, PartitionStrategy::Hash);

        let triple1 = make_triple(1, 2, 3);
        let triple2 = make_triple(4, 5, 6);

        let part1 = manager.get_partition(&triple1);
        let part2 = manager.get_partition(&triple2);

        // Same triple should always go to same partition
        assert_eq!(part1, manager.get_partition(&triple1));

        // Different triples may go to different partitions
        assert!(part1 < 4);
        assert!(part2 < 4);
    }

    #[test]
    fn test_subject_partitioning() {
        let manager = PartitionManager::new(4, PartitionStrategy::Subject);

        let triple1 = make_triple(1, 2, 3);
        let triple2 = make_triple(1, 5, 6); // Same subject
        let triple3 = make_triple(2, 2, 3); // Different subject

        let part1 = manager.get_partition(&triple1);
        let part2 = manager.get_partition(&triple2);
        let part3 = manager.get_partition(&triple3);

        // Same subject should go to same partition
        assert_eq!(part1, part2);

        // Different subjects may go to different partitions
        assert!(part1 < 4);
        assert!(part3 < 4);
    }

    #[test]
    fn test_predicate_partitioning() {
        let manager = PartitionManager::new(4, PartitionStrategy::Predicate);

        let triple1 = make_triple(1, 2, 3);
        let triple2 = make_triple(4, 2, 6); // Same predicate
        let triple3 = make_triple(1, 3, 3); // Different predicate

        let part1 = manager.get_partition(&triple1);
        let part2 = manager.get_partition(&triple2);
        let part3 = manager.get_partition(&triple3);

        // Same predicate should go to same partition
        assert_eq!(part1, part2);

        assert!(part1 < 4);
        assert!(part3 < 4);
    }

    #[test]
    fn test_range_partitioning() {
        let manager = PartitionManager::new(4, PartitionStrategy::Range);

        // Set up range bounds: [0, 100), [100, 200), [200, 300), [300, ∞)
        manager
            .set_range_bounds(vec![NodeId::new(100), NodeId::new(200), NodeId::new(300)])
            .unwrap();

        assert_eq!(manager.range_partition(NodeId::new(50)), 0);
        assert_eq!(manager.range_partition(NodeId::new(150)), 1);
        assert_eq!(manager.range_partition(NodeId::new(250)), 2);
        assert_eq!(manager.range_partition(NodeId::new(350)), 3);
    }

    #[test]
    fn test_invalid_range_bounds() {
        let manager = PartitionManager::new(4, PartitionStrategy::Range);

        // Wrong number of bounds
        let result = manager.set_range_bounds(vec![NodeId::new(100), NodeId::new(200)]);
        assert!(result.is_err());

        // Unsorted bounds
        let result =
            manager.set_range_bounds(vec![NodeId::new(200), NodeId::new(100), NodeId::new(300)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_record_insert_delete() {
        let manager = PartitionManager::new(4, PartitionStrategy::Hash);

        manager.record_insert(0, 100);
        manager.record_insert(0, 200);
        manager.record_insert(1, 150);

        let stats = manager.stats();
        assert_eq!(stats.total_triples, 3);
        assert_eq!(stats.triples_per_partition[0], 2);
        assert_eq!(stats.triples_per_partition[1], 1);
        assert_eq!(stats.bytes_per_partition[0], 300);

        manager.record_delete(0, 100);

        let stats = manager.stats();
        assert_eq!(stats.total_triples, 2);
        assert_eq!(stats.triples_per_partition[0], 1);
        assert_eq!(stats.bytes_per_partition[0], 200);
    }

    #[test]
    fn test_record_query() {
        let manager = PartitionManager::new(4, PartitionStrategy::Hash);

        manager.record_query(&[0]);
        manager.record_query(&[1]);
        manager.record_query(&[0, 1, 2]);

        let stats = manager.stats();
        assert_eq!(stats.single_partition_queries, 2);
        assert_eq!(stats.cross_partition_queries, 1);
        assert_eq!(stats.query_locality(), 2.0 / 3.0);
    }

    #[test]
    fn test_get_partitions_for_pattern() {
        let manager = PartitionManager::new(4, PartitionStrategy::Subject);

        // Pattern with subject specified
        let pattern = TriplePattern {
            subject: Some(NodeId::new(1)),
            predicate: None,
            object: None,
        };
        let partitions = manager.get_partitions_for_pattern(&pattern);
        assert_eq!(partitions.len(), 1);

        // Pattern without subject (wildcard)
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(NodeId::new(2)),
            object: None,
        };
        let partitions = manager.get_partitions_for_pattern(&pattern);
        assert_eq!(partitions.len(), 4); // All partitions needed
    }

    #[test]
    fn test_partition_balance_ratio() {
        let manager = PartitionManager::new(4, PartitionStrategy::Hash);

        // Balanced partitions
        manager.record_insert(0, 100);
        manager.record_insert(1, 100);
        manager.record_insert(2, 100);
        manager.record_insert(3, 100);

        let stats = manager.stats();
        assert!(stats.balance_ratio() > 0.99); // Nearly perfect balance

        // Unbalanced partitions
        manager.record_insert(0, 100);
        manager.record_insert(0, 100);
        manager.record_insert(0, 100);

        let stats = manager.stats();
        assert!(stats.balance_ratio() < 0.9); // Poor balance
    }

    #[test]
    fn test_suggest_rebalancing() {
        let manager = PartitionManager::new(4, PartitionStrategy::Hash);

        // Create imbalance
        for _ in 0..100 {
            manager.record_insert(0, 100);
        }
        for _ in 0..10 {
            manager.record_insert(1, 100);
        }

        let suggestions = manager.suggest_rebalancing();
        assert!(!suggestions.is_empty());

        // Should suggest moving from partition 0 to others
        for (from, _to, _count) in suggestions {
            assert_eq!(from, 0);
        }
    }

    #[test]
    fn test_partition_metadata() {
        let manager = PartitionManager::new(2, PartitionStrategy::Hash);

        manager.record_insert(0, 100);
        manager.record_insert(0, 200);

        let partitions = manager.partitions();
        assert_eq!(partitions[0].triple_count, 2);
        assert_eq!(partitions[0].size_bytes, 300);
        assert_eq!(partitions[1].triple_count, 0);
    }
}
