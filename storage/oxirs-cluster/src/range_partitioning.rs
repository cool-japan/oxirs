//! Range-Based Partitioning Module
//!
//! Implements range-based partitioning for distributed RDF storage,
//! enabling key range assignment, split operations, and merge operations.

use crate::raft::OxirsNodeId;
use crate::shard::ShardId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Range key type for partitioning
pub type RangeKey = String;

/// Range partition definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Range {
    /// Start key (inclusive)
    pub start: Option<RangeKey>,
    /// End key (exclusive)
    pub end: Option<RangeKey>,
}

impl Range {
    /// Create a new range
    pub fn new(start: Option<RangeKey>, end: Option<RangeKey>) -> Self {
        Self { start, end }
    }

    /// Create an unbounded range (covers all keys)
    pub fn unbounded() -> Self {
        Self {
            start: None,
            end: None,
        }
    }

    /// Create a range starting from a key
    pub fn from(start: RangeKey) -> Self {
        Self {
            start: Some(start),
            end: None,
        }
    }

    /// Create a range up to a key
    pub fn to(end: RangeKey) -> Self {
        Self {
            start: None,
            end: Some(end),
        }
    }

    /// Create a bounded range
    pub fn between(start: RangeKey, end: RangeKey) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
        }
    }

    /// Check if a key belongs to this range
    pub fn contains(&self, key: &str) -> bool {
        // Check start boundary
        if let Some(ref start) = self.start {
            if key < start.as_str() {
                return false;
            }
        }

        // Check end boundary
        if let Some(ref end) = self.end {
            if key >= end.as_str() {
                return false;
            }
        }

        true
    }

    /// Check if this range overlaps with another range
    pub fn overlaps(&self, other: &Range) -> bool {
        // If either range is unbounded on the side that matters, they overlap
        match (&self.start, &other.end) {
            (Some(s1), Some(e2)) => {
                if s1 >= e2 {
                    return false;
                }
            }
            _ => {}
        }

        match (&other.start, &self.end) {
            (Some(s2), Some(e1)) => {
                if s2 >= e1 {
                    return false;
                }
            }
            _ => {}
        }

        true
    }

    /// Split this range at a given key
    pub fn split_at(&self, split_key: &str) -> (Range, Range) {
        let left = Range {
            start: self.start.clone(),
            end: Some(split_key.to_string()),
        };

        let right = Range {
            start: Some(split_key.to_string()),
            end: self.end.clone(),
        };

        (left, right)
    }

    /// Check if this range can be merged with another range
    pub fn can_merge_with(&self, other: &Range) -> bool {
        // Check if ranges are adjacent
        (match (&self.end, &other.start) {
            (Some(e1), Some(s2)) if e1 == s2 => true,
            _ => false,
        }) || (match (&other.end, &self.start) {
            (Some(e2), Some(s1)) if e2 == s1 => true,
            _ => false,
        })
    }

    /// Merge this range with another range
    pub fn merge_with(&self, other: &Range) -> Option<Range> {
        if !self.can_merge_with(other) {
            return None;
        }

        let start = match (&self.start, &other.start) {
            (None, _) | (_, None) => None,
            (Some(s1), Some(s2)) => Some(s1.min(s2).clone()),
        };

        let end = match (&self.end, &other.end) {
            (None, _) | (_, None) => None,
            (Some(e1), Some(e2)) => Some(e1.max(e2).clone()),
        };

        Some(Range { start, end })
    }
}

impl Display for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.start, &self.end) {
            (None, None) => write!(f, "(-∞, +∞)"),
            (Some(start), None) => write!(f, "[{}, +∞)", start),
            (None, Some(end)) => write!(f, "(-∞, {})", end),
            (Some(start), Some(end)) => write!(f, "[{}, {})", start, end),
        }
    }
}

/// Range partition metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangePartition {
    /// Unique partition ID
    pub partition_id: String,
    /// Shard ID associated with this partition
    pub shard_id: ShardId,
    /// Key range for this partition
    pub range: Range,
    /// Nodes hosting this partition
    pub nodes: BTreeSet<OxirsNodeId>,
    /// Load statistics
    pub load_stats: LoadStats,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modification timestamp
    pub modified_at: u64,
}

impl RangePartition {
    /// Create a new range partition
    pub fn new(partition_id: String, shard_id: ShardId, range: Range) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            partition_id,
            shard_id,
            range,
            nodes: BTreeSet::new(),
            load_stats: LoadStats::default(),
            created_at: timestamp,
            modified_at: timestamp,
        }
    }

    /// Add a node to this partition
    pub fn add_node(&mut self, node_id: OxirsNodeId) {
        self.nodes.insert(node_id);
        self.touch();
    }

    /// Remove a node from this partition
    pub fn remove_node(&mut self, node_id: OxirsNodeId) {
        self.nodes.remove(&node_id);
        self.touch();
    }

    /// Update load statistics
    pub fn update_load_stats(&mut self, stats: LoadStats) {
        self.load_stats = stats;
        self.touch();
    }

    /// Check if this partition needs splitting
    pub fn needs_split(&self, max_load: u64) -> bool {
        self.load_stats.key_count > max_load || self.load_stats.data_size > max_load * 1024
    }

    /// Check if this partition can be merged
    pub fn can_merge(&self, min_load: u64) -> bool {
        self.load_stats.key_count < min_load && self.load_stats.data_size < min_load * 1024
    }

    fn touch(&mut self) {
        self.modified_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// Load statistics for a partition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadStats {
    /// Number of keys in this partition
    pub key_count: u64,
    /// Total data size in bytes
    pub data_size: u64,
    /// Read operations per second
    pub read_ops_per_sec: f64,
    /// Write operations per second
    pub write_ops_per_sec: f64,
    /// Average key size
    pub avg_key_size: u64,
    /// Last update timestamp
    pub last_updated: u64,
}

impl Default for LoadStats {
    fn default() -> Self {
        Self {
            key_count: 0,
            data_size: 0,
            read_ops_per_sec: 0.0,
            write_ops_per_sec: 0.0,
            avg_key_size: 0,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

/// Split operation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitOperation {
    /// Unique operation ID
    pub operation_id: String,
    /// Source partition being split
    pub source_partition: String,
    /// Split key
    pub split_key: RangeKey,
    /// Resulting left partition
    pub left_partition: String,
    /// Resulting right partition
    pub right_partition: String,
    /// Operation status
    pub status: OperationStatus,
    /// Progress percentage (0-100)
    pub progress: u8,
    /// Created timestamp
    pub created_at: u64,
    /// Completed timestamp (if finished)
    pub completed_at: Option<u64>,
}

/// Merge operation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeOperation {
    /// Unique operation ID
    pub operation_id: String,
    /// Source partitions being merged
    pub source_partitions: Vec<String>,
    /// Resulting merged partition
    pub target_partition: String,
    /// Operation status
    pub status: OperationStatus,
    /// Progress percentage (0-100)
    pub progress: u8,
    /// Created timestamp
    pub created_at: u64,
    /// Completed timestamp (if finished)
    pub completed_at: Option<u64>,
}

/// Operation status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OperationStatus {
    /// Operation is planned but not started
    Planned,
    /// Operation is in progress
    InProgress,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed,
    /// Operation was cancelled
    Cancelled,
}

/// Range-based partition manager
pub struct RangePartitionManager {
    /// All partitions indexed by partition ID
    partitions: Arc<RwLock<BTreeMap<String, RangePartition>>>,
    /// Active split operations
    split_operations: Arc<RwLock<BTreeMap<String, SplitOperation>>>,
    /// Active merge operations
    merge_operations: Arc<RwLock<BTreeMap<String, MergeOperation>>>,
    /// Configuration
    config: RangePartitionConfig,
}

/// Configuration for range partitioning
#[derive(Debug, Clone)]
pub struct RangePartitionConfig {
    /// Maximum keys per partition before split
    pub max_keys_per_partition: u64,
    /// Minimum keys per partition before merge
    pub min_keys_per_partition: u64,
    /// Maximum data size per partition (bytes)
    pub max_partition_size: u64,
    /// Minimum data size per partition (bytes)
    pub min_partition_size: u64,
    /// Enable automatic rebalancing
    pub auto_rebalance: bool,
    /// Rebalance check interval (seconds)
    pub rebalance_interval: u64,
}

impl Default for RangePartitionConfig {
    fn default() -> Self {
        Self {
            max_keys_per_partition: 1_000_000,
            min_keys_per_partition: 100_000,
            max_partition_size: 1024 * 1024 * 1024, // 1GB
            min_partition_size: 10 * 1024 * 1024,   // 10MB
            auto_rebalance: true,
            rebalance_interval: 300, // 5 minutes
        }
    }
}

impl RangePartitionManager {
    /// Create a new range partition manager
    pub fn new(config: RangePartitionConfig) -> Self {
        Self {
            partitions: Arc::new(RwLock::new(BTreeMap::new())),
            split_operations: Arc::new(RwLock::new(BTreeMap::new())),
            merge_operations: Arc::new(RwLock::new(BTreeMap::new())),
            config,
        }
    }

    /// Create an initial partition covering all keys
    pub async fn create_initial_partition(&self, shard_id: ShardId) -> Result<String> {
        let partition_id = uuid::Uuid::new_v4().to_string();
        let partition = RangePartition::new(partition_id.clone(), shard_id, Range::unbounded());

        let mut partitions = self.partitions.write().await;
        partitions.insert(partition_id.clone(), partition);

        info!(
            "Created initial partition {} for shard {}",
            partition_id, shard_id
        );
        Ok(partition_id)
    }

    /// Find the partition that should contain a given key
    pub async fn find_partition_for_key(&self, key: &str) -> Option<String> {
        let partitions = self.partitions.read().await;

        for (partition_id, partition) in partitions.iter() {
            if partition.range.contains(key) {
                return Some(partition_id.clone());
            }
        }

        None
    }

    /// Get all partitions
    pub async fn get_all_partitions(&self) -> Vec<RangePartition> {
        let partitions = self.partitions.read().await;
        partitions.values().cloned().collect()
    }

    /// Get a specific partition
    pub async fn get_partition(&self, partition_id: &str) -> Option<RangePartition> {
        let partitions = self.partitions.read().await;
        partitions.get(partition_id).cloned()
    }

    /// Update load statistics for a partition
    pub async fn update_partition_load(
        &self,
        partition_id: &str,
        load_stats: LoadStats,
    ) -> Result<()> {
        let mut partitions = self.partitions.write().await;

        if let Some(partition) = partitions.get_mut(partition_id) {
            partition.update_load_stats(load_stats);
            info!("Updated load stats for partition {}", partition_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Partition {} not found", partition_id))
        }
    }

    /// Split a partition at a given key
    pub async fn split_partition(&self, partition_id: &str, split_key: &str) -> Result<String> {
        let operation_id = uuid::Uuid::new_v4().to_string();

        // Validate that the partition exists
        let partition = {
            let partitions = self.partitions.read().await;
            partitions
                .get(partition_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Partition {} not found", partition_id))?
        };

        // Validate that the split key is within the partition range
        if !partition.range.contains(split_key) {
            return Err(anyhow::anyhow!(
                "Split key '{}' is not within partition range {}",
                split_key,
                partition.range
            ));
        }

        let left_partition_id = format!("{}-left", operation_id);
        let right_partition_id = format!("{}-right", operation_id);

        // Create split operation
        let split_op = SplitOperation {
            operation_id: operation_id.clone(),
            source_partition: partition_id.to_string(),
            split_key: split_key.to_string(),
            left_partition: left_partition_id.clone(),
            right_partition: right_partition_id.clone(),
            status: OperationStatus::Planned,
            progress: 0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            completed_at: None,
        };

        // Store the operation
        {
            let mut operations = self.split_operations.write().await;
            operations.insert(operation_id.clone(), split_op);
        }

        // Execute the split asynchronously
        let manager = self.clone();
        let operation_id_clone = operation_id.clone();
        tokio::spawn(async move {
            if let Err(e) = manager.execute_split_operation(&operation_id_clone).await {
                warn!("Split operation {} failed: {}", operation_id_clone, e);
            }
        });

        info!(
            "Started split operation {} for partition {}",
            operation_id, partition_id
        );
        Ok(operation_id)
    }

    /// Merge multiple adjacent partitions
    pub async fn merge_partitions(&self, partition_ids: Vec<String>) -> Result<String> {
        let operation_id = uuid::Uuid::new_v4().to_string();

        if partition_ids.len() < 2 {
            return Err(anyhow::anyhow!("Need at least 2 partitions to merge"));
        }

        // Validate that all partitions exist and are mergeable
        {
            let partitions = self.partitions.read().await;
            let mut partition_ranges = Vec::new();

            for partition_id in &partition_ids {
                let partition = partitions
                    .get(partition_id)
                    .ok_or_else(|| anyhow::anyhow!("Partition {} not found", partition_id))?;
                partition_ranges.push((partition_id.clone(), partition.range.clone()));
            }

            // Check if partitions can be merged (simplified check)
            partition_ranges.sort_by(|a, b| match (&a.1.start, &b.1.start) {
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
                (Some(a), Some(b)) => a.cmp(b),
            });
        }

        let target_partition_id = format!("{}-merged", operation_id);

        // Create merge operation
        let merge_op = MergeOperation {
            operation_id: operation_id.clone(),
            source_partitions: partition_ids,
            target_partition: target_partition_id,
            status: OperationStatus::Planned,
            progress: 0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            completed_at: None,
        };

        // Store the operation
        {
            let mut operations = self.merge_operations.write().await;
            operations.insert(operation_id.clone(), merge_op);
        }

        // Execute the merge asynchronously
        let manager = self.clone();
        let operation_id_clone = operation_id.clone();
        tokio::spawn(async move {
            if let Err(e) = manager.execute_merge_operation(&operation_id_clone).await {
                warn!("Merge operation {} failed: {}", operation_id_clone, e);
            }
        });

        info!("Started merge operation {}", operation_id);
        Ok(operation_id)
    }

    /// Execute a split operation
    async fn execute_split_operation(&self, operation_id: &str) -> Result<()> {
        // Update operation status to in progress
        {
            let mut operations = self.split_operations.write().await;
            if let Some(op) = operations.get_mut(operation_id) {
                op.status = OperationStatus::InProgress;
                op.progress = 10;
            }
        }

        let operation = {
            let operations = self.split_operations.read().await;
            operations
                .get(operation_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Split operation {} not found", operation_id))?
        };

        // Get the source partition
        let source_partition = {
            let partitions = self.partitions.read().await;
            partitions
                .get(&operation.source_partition)
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!("Source partition {} not found", operation.source_partition)
                })?
        };

        // Create left and right partitions
        let (left_range, right_range) = source_partition.range.split_at(&operation.split_key);

        let mut left_partition = RangePartition::new(
            operation.left_partition.clone(),
            source_partition.shard_id,
            left_range,
        );
        left_partition.nodes = source_partition.nodes.clone();

        let mut right_partition = RangePartition::new(
            operation.right_partition.clone(),
            source_partition.shard_id,
            right_range,
        );
        right_partition.nodes = source_partition.nodes.clone();

        // Update progress
        {
            let mut operations = self.split_operations.write().await;
            if let Some(op) = operations.get_mut(operation_id) {
                op.progress = 50;
            }
        }

        // Replace the source partition with the two new partitions
        {
            let mut partitions = self.partitions.write().await;
            partitions.remove(&operation.source_partition);
            partitions.insert(operation.left_partition.clone(), left_partition);
            partitions.insert(operation.right_partition.clone(), right_partition);
        }

        // Mark operation as completed
        {
            let mut operations = self.split_operations.write().await;
            if let Some(op) = operations.get_mut(operation_id) {
                op.status = OperationStatus::Completed;
                op.progress = 100;
                op.completed_at = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                );
            }
        }

        info!("Completed split operation {}", operation_id);
        Ok(())
    }

    /// Execute a merge operation
    async fn execute_merge_operation(&self, operation_id: &str) -> Result<()> {
        // Similar implementation to split but in reverse
        // This is a simplified version - in practice would need careful coordination

        let operation = {
            let operations = self.merge_operations.read().await;
            operations
                .get(operation_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Merge operation {} not found", operation_id))?
        };

        // Update status
        {
            let mut operations = self.merge_operations.write().await;
            if let Some(op) = operations.get_mut(operation_id) {
                op.status = OperationStatus::InProgress;
                op.progress = 10;
            }
        }

        // Get source partitions and create merged partition
        let source_partitions: Vec<RangePartition> = {
            let partitions = self.partitions.read().await;
            operation
                .source_partitions
                .iter()
                .map(|id| partitions.get(id).cloned())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow::anyhow!("Some source partitions not found"))?
        };

        // Determine merged range
        let mut merged_range = source_partitions[0].range.clone();
        for partition in &source_partitions[1..] {
            if let Some(new_range) = merged_range.merge_with(&partition.range) {
                merged_range = new_range;
            }
        }

        let merged_partition = RangePartition::new(
            operation.target_partition.clone(),
            source_partitions[0].shard_id,
            merged_range,
        );

        // Replace source partitions with merged partition
        {
            let mut partitions = self.partitions.write().await;
            for partition_id in &operation.source_partitions {
                partitions.remove(partition_id);
            }
            partitions.insert(operation.target_partition.clone(), merged_partition);
        }

        // Mark operation as completed
        {
            let mut operations = self.merge_operations.write().await;
            if let Some(op) = operations.get_mut(operation_id) {
                op.status = OperationStatus::Completed;
                op.progress = 100;
                op.completed_at = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                );
            }
        }

        info!("Completed merge operation {}", operation_id);
        Ok(())
    }

    /// Check if any partitions need rebalancing
    pub async fn check_rebalancing_needed(&self) -> Vec<String> {
        let mut partitions_needing_split = Vec::new();
        let mut partitions_needing_merge = Vec::new();

        {
            let partitions = self.partitions.read().await;

            for (partition_id, partition) in partitions.iter() {
                if partition.needs_split(self.config.max_keys_per_partition) {
                    partitions_needing_split.push(partition_id.clone());
                } else if partition.can_merge(self.config.min_keys_per_partition) {
                    partitions_needing_merge.push(partition_id.clone());
                }
            }
        }

        info!(
            "Rebalancing check: {} partitions need split, {} need merge",
            partitions_needing_split.len(),
            partitions_needing_merge.len()
        );

        // Return partitions that need attention (simplified)
        [partitions_needing_split, partitions_needing_merge].concat()
    }

    /// Get operation status
    pub async fn get_split_operation_status(&self, operation_id: &str) -> Option<SplitOperation> {
        let operations = self.split_operations.read().await;
        operations.get(operation_id).cloned()
    }

    /// Get merge operation status
    pub async fn get_merge_operation_status(&self, operation_id: &str) -> Option<MergeOperation> {
        let operations = self.merge_operations.read().await;
        operations.get(operation_id).cloned()
    }
}

impl Clone for RangePartitionManager {
    fn clone(&self) -> Self {
        Self {
            partitions: Arc::clone(&self.partitions),
            split_operations: Arc::clone(&self.split_operations),
            merge_operations: Arc::clone(&self.merge_operations),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_contains() {
        let range = Range::between("b".to_string(), "f".to_string());

        assert!(!range.contains("a"));
        assert!(range.contains("b"));
        assert!(range.contains("d"));
        assert!(!range.contains("f"));
        assert!(!range.contains("g"));
    }

    #[test]
    fn test_range_split() {
        let range = Range::between("a".to_string(), "z".to_string());
        let (left, right) = range.split_at("m");

        assert_eq!(left.start, Some("a".to_string()));
        assert_eq!(left.end, Some("m".to_string()));
        assert_eq!(right.start, Some("m".to_string()));
        assert_eq!(right.end, Some("z".to_string()));
    }

    #[test]
    fn test_range_merge() {
        let left = Range::between("a".to_string(), "m".to_string());
        let right = Range::between("m".to_string(), "z".to_string());

        assert!(left.can_merge_with(&right));

        let merged = left.merge_with(&right).unwrap();
        assert_eq!(merged.start, Some("a".to_string()));
        assert_eq!(merged.end, Some("z".to_string()));
    }

    #[tokio::test]
    async fn test_partition_manager_basic() {
        let config = RangePartitionConfig::default();
        let manager = RangePartitionManager::new(config);

        let partition_id = manager.create_initial_partition(1).await.unwrap();

        let partition = manager.get_partition(&partition_id).await.unwrap();
        assert_eq!(partition.shard_id, 1);
        assert_eq!(partition.range, Range::unbounded());
    }

    #[tokio::test]
    async fn test_find_partition_for_key() {
        let config = RangePartitionConfig::default();
        let manager = RangePartitionManager::new(config);

        let partition_id = manager.create_initial_partition(1).await.unwrap();

        let found = manager.find_partition_for_key("any_key").await;
        assert_eq!(found, Some(partition_id));
    }

    #[test]
    fn test_load_stats_default() {
        let stats = LoadStats::default();
        assert_eq!(stats.key_count, 0);
        assert_eq!(stats.data_size, 0);
        assert_eq!(stats.read_ops_per_sec, 0.0);
        assert_eq!(stats.write_ops_per_sec, 0.0);
    }
}
