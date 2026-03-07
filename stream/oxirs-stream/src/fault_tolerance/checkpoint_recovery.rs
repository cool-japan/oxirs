//! # Checkpoint and Recovery for Fault Tolerance
//!
//! Provides:
//! - [`CheckpointManager`]: Periodic state snapshots to durable storage
//! - [`RecoveryManager`]: Restore from latest checkpoint on failure
//! - [`PartitionRebalancer`]: Redistribute work when nodes fail

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::FaultResult;

// ─── Checkpoint Manager ──────────────────────────────────────────────────────

/// Configuration for the checkpoint manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManagerConfig {
    /// Interval between checkpoints
    pub checkpoint_interval: Duration,
    /// Maximum number of checkpoints to retain
    pub max_retained: usize,
    /// Whether to compress checkpoint data
    pub compress: bool,
    /// Minimum size change (bytes) to trigger a new checkpoint
    pub min_change_bytes: usize,
}

impl Default for CheckpointManagerConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: Duration::from_secs(30),
            max_retained: 5,
            compress: false,
            min_change_bytes: 0,
        }
    }
}

/// A stored checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredCheckpoint {
    /// Unique checkpoint ID
    pub checkpoint_id: String,
    /// Checkpoint version (monotonically increasing)
    pub version: u64,
    /// Operator states (operator_id -> serialized state)
    pub operator_states: HashMap<String, Vec<u8>>,
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,
    /// When the checkpoint was created
    pub created_at: SystemTime,
}

/// Metadata for a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Total size in bytes
    pub total_size_bytes: usize,
    /// Number of operators
    pub operator_count: usize,
    /// Duration to create the checkpoint (microseconds)
    pub creation_duration_us: u64,
    /// Integrity hash (FNV-1a)
    pub integrity_hash: String,
}

/// Statistics for the checkpoint manager
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointManagerStats {
    /// Total checkpoints taken
    pub checkpoints_taken: u64,
    /// Total checkpoints failed
    pub checkpoints_failed: u64,
    /// Average checkpoint size (bytes)
    pub avg_checkpoint_size: f64,
    /// Average checkpoint creation time (ms)
    pub avg_creation_time_ms: f64,
    /// Currently retained checkpoints
    pub retained_checkpoints: usize,
    /// Total bytes stored
    pub total_bytes_stored: usize,
}

/// Manages periodic state checkpoints.
///
/// Takes snapshots of all registered operator states at configurable intervals,
/// stores them with integrity verification, and manages retention.
pub struct CheckpointManager {
    config: CheckpointManagerConfig,
    /// Stored checkpoints (most recent first)
    checkpoints: Arc<RwLock<VecDeque<StoredCheckpoint>>>,
    /// Version counter
    version: Arc<RwLock<u64>>,
    /// Last checkpoint time
    last_checkpoint_time: Arc<RwLock<Option<Instant>>>,
    /// Accumulated state changes since last checkpoint
    accumulated_change_bytes: Arc<RwLock<usize>>,
    /// Statistics
    stats: Arc<RwLock<CheckpointManagerStats>>,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager.
    pub fn new(config: CheckpointManagerConfig) -> Self {
        Self {
            config,
            checkpoints: Arc::new(RwLock::new(VecDeque::new())),
            version: Arc::new(RwLock::new(0)),
            last_checkpoint_time: Arc::new(RwLock::new(None)),
            accumulated_change_bytes: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(CheckpointManagerStats::default())),
        }
    }

    /// Records a state change of the given size.
    pub fn record_state_change(&self, size_bytes: usize) {
        *self.accumulated_change_bytes.write() += size_bytes;
    }

    /// Returns whether a checkpoint should be taken now.
    pub fn should_checkpoint(&self) -> bool {
        let time_due = {
            let last = self.last_checkpoint_time.read();
            match *last {
                Some(t) => t.elapsed() >= self.config.checkpoint_interval,
                None => true,
            }
        };

        let change_due = {
            let change = *self.accumulated_change_bytes.read();
            change >= self.config.min_change_bytes || self.config.min_change_bytes == 0
        };

        time_due && change_due
    }

    /// Takes a checkpoint of the given operator states.
    pub fn take_checkpoint(
        &self,
        operator_states: HashMap<String, Vec<u8>>,
    ) -> FaultResult<StoredCheckpoint> {
        let start = Instant::now();

        let mut version = self.version.write();
        *version += 1;
        let current_version = *version;
        drop(version);

        let total_size: usize = operator_states.values().map(|v| v.len()).sum();

        // Compute integrity hash
        let operator_count = operator_states.len();
        let mut all_bytes = Vec::with_capacity(total_size);
        let mut sorted_keys: Vec<String> = operator_states.keys().cloned().collect();
        sorted_keys.sort();
        for key in &sorted_keys {
            if let Some(state) = operator_states.get(key) {
                all_bytes.extend_from_slice(state);
            }
        }
        let integrity_hash = compute_fnv1a_hash(&all_bytes);

        let creation_duration = start.elapsed();

        let checkpoint = StoredCheckpoint {
            checkpoint_id: format!("ckpt-{}", current_version),
            version: current_version,
            operator_states,
            metadata: CheckpointMetadata {
                total_size_bytes: total_size,
                operator_count,
                creation_duration_us: creation_duration.as_micros() as u64,
                integrity_hash,
            },
            created_at: SystemTime::now(),
        };

        // Store and manage retention
        let mut checkpoints = self.checkpoints.write();
        checkpoints.push_front(checkpoint.clone());
        while checkpoints.len() > self.config.max_retained {
            checkpoints.pop_back();
        }
        let retained = checkpoints.len();
        let total_stored: usize = checkpoints
            .iter()
            .map(|c| c.metadata.total_size_bytes)
            .sum();
        drop(checkpoints);

        // Reset change accumulator
        *self.accumulated_change_bytes.write() = 0;
        *self.last_checkpoint_time.write() = Some(Instant::now());

        // Update stats
        let mut stats = self.stats.write();
        stats.checkpoints_taken += 1;
        let n = stats.checkpoints_taken as f64;
        stats.avg_checkpoint_size = (stats.avg_checkpoint_size * (n - 1.0) + total_size as f64) / n;
        stats.avg_creation_time_ms = (stats.avg_creation_time_ms * (n - 1.0)
            + creation_duration.as_micros() as f64 / 1000.0)
            / n;
        stats.retained_checkpoints = retained;
        stats.total_bytes_stored = total_stored;

        info!(
            "Checkpoint {} taken (v{}, {} operators, {} bytes, {:.2}ms)",
            checkpoint.checkpoint_id,
            current_version,
            checkpoint.metadata.operator_count,
            total_size,
            creation_duration.as_micros() as f64 / 1000.0
        );

        Ok(checkpoint)
    }

    /// Returns the latest checkpoint, if any.
    pub fn latest_checkpoint(&self) -> Option<StoredCheckpoint> {
        self.checkpoints.read().front().cloned()
    }

    /// Returns a checkpoint by version.
    pub fn checkpoint_by_version(&self, version: u64) -> Option<StoredCheckpoint> {
        self.checkpoints
            .read()
            .iter()
            .find(|c| c.version == version)
            .cloned()
    }

    /// Returns all checkpoints (most recent first).
    pub fn all_checkpoints(&self) -> Vec<StoredCheckpoint> {
        self.checkpoints.read().iter().cloned().collect()
    }

    /// Returns manager statistics.
    pub fn stats(&self) -> CheckpointManagerStats {
        self.stats.read().clone()
    }

    /// Verifies the integrity of a checkpoint.
    pub fn verify_integrity(&self, checkpoint: &StoredCheckpoint) -> bool {
        let mut all_bytes = Vec::new();
        let mut sorted_keys: Vec<&String> = checkpoint.operator_states.keys().collect();
        sorted_keys.sort();
        for key in &sorted_keys {
            if let Some(state) = checkpoint.operator_states.get(*key) {
                all_bytes.extend_from_slice(state);
            }
        }
        let computed = compute_fnv1a_hash(&all_bytes);
        computed == checkpoint.metadata.integrity_hash
    }
}

// ─── Recovery Manager ────────────────────────────────────────────────────────

/// Recovery strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Restore from the latest checkpoint
    LatestCheckpoint,
    /// Restore from a specific checkpoint version
    SpecificVersion(u64),
    /// Start fresh (discard all state)
    Fresh,
}

/// Result of a recovery operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryResult {
    /// Whether recovery was successful
    pub success: bool,
    /// Strategy used
    pub strategy: RecoveryStrategy,
    /// Checkpoint version restored (if applicable)
    pub restored_version: Option<u64>,
    /// Operators restored
    pub operators_restored: usize,
    /// Total bytes restored
    pub bytes_restored: usize,
    /// Recovery duration (microseconds)
    pub recovery_duration_us: u64,
    /// Message
    pub message: String,
}

/// Statistics for the recovery manager
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecoveryManagerStats {
    /// Total recovery attempts
    pub recovery_attempts: u64,
    /// Successful recoveries
    pub successful_recoveries: u64,
    /// Failed recoveries
    pub failed_recoveries: u64,
    /// Average recovery time (ms)
    pub avg_recovery_time_ms: f64,
}

/// Manages recovery from checkpoints on failure.
///
/// When a failure is detected, the recovery manager restores operator
/// state from the latest (or a specific) checkpoint stored by the
/// [`CheckpointManager`].
pub struct RecoveryManager {
    /// Reference to the checkpoint manager
    checkpoint_manager: Arc<CheckpointManager>,
    /// Default recovery strategy
    default_strategy: RecoveryStrategy,
    /// Recovery history
    history: Arc<RwLock<Vec<RecoveryResult>>>,
    /// Statistics
    stats: Arc<RwLock<RecoveryManagerStats>>,
}

impl RecoveryManager {
    /// Creates a new recovery manager.
    pub fn new(
        checkpoint_manager: Arc<CheckpointManager>,
        default_strategy: RecoveryStrategy,
    ) -> Self {
        Self {
            checkpoint_manager,
            default_strategy,
            history: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(RecoveryManagerStats::default())),
        }
    }

    /// Initiates recovery using the default strategy.
    pub fn recover(&self) -> RecoveryResult {
        self.recover_with_strategy(self.default_strategy.clone())
    }

    /// Initiates recovery with a specific strategy.
    pub fn recover_with_strategy(&self, strategy: RecoveryStrategy) -> RecoveryResult {
        let start = Instant::now();
        self.stats.write().recovery_attempts += 1;

        let result = match &strategy {
            RecoveryStrategy::LatestCheckpoint => {
                match self.checkpoint_manager.latest_checkpoint() {
                    Some(checkpoint) => {
                        if self.checkpoint_manager.verify_integrity(&checkpoint) {
                            let ops = checkpoint.operator_states.len();
                            let bytes: usize =
                                checkpoint.operator_states.values().map(|v| v.len()).sum();
                            RecoveryResult {
                                success: true,
                                strategy: strategy.clone(),
                                restored_version: Some(checkpoint.version),
                                operators_restored: ops,
                                bytes_restored: bytes,
                                recovery_duration_us: start.elapsed().as_micros() as u64,
                                message: format!(
                                    "Restored from checkpoint v{} ({} operators)",
                                    checkpoint.version, ops
                                ),
                            }
                        } else {
                            RecoveryResult {
                                success: false,
                                strategy: strategy.clone(),
                                restored_version: None,
                                operators_restored: 0,
                                bytes_restored: 0,
                                recovery_duration_us: start.elapsed().as_micros() as u64,
                                message: "Checkpoint integrity verification failed".to_string(),
                            }
                        }
                    }
                    None => RecoveryResult {
                        success: false,
                        strategy: strategy.clone(),
                        restored_version: None,
                        operators_restored: 0,
                        bytes_restored: 0,
                        recovery_duration_us: start.elapsed().as_micros() as u64,
                        message: "No checkpoint available for recovery".to_string(),
                    },
                }
            }
            RecoveryStrategy::SpecificVersion(version) => {
                match self.checkpoint_manager.checkpoint_by_version(*version) {
                    Some(checkpoint) => {
                        if self.checkpoint_manager.verify_integrity(&checkpoint) {
                            let ops = checkpoint.operator_states.len();
                            let bytes: usize =
                                checkpoint.operator_states.values().map(|v| v.len()).sum();
                            RecoveryResult {
                                success: true,
                                strategy: strategy.clone(),
                                restored_version: Some(*version),
                                operators_restored: ops,
                                bytes_restored: bytes,
                                recovery_duration_us: start.elapsed().as_micros() as u64,
                                message: format!("Restored from checkpoint v{}", version),
                            }
                        } else {
                            RecoveryResult {
                                success: false,
                                strategy: strategy.clone(),
                                restored_version: None,
                                operators_restored: 0,
                                bytes_restored: 0,
                                recovery_duration_us: start.elapsed().as_micros() as u64,
                                message: format!(
                                    "Checkpoint v{} integrity verification failed",
                                    version
                                ),
                            }
                        }
                    }
                    None => RecoveryResult {
                        success: false,
                        strategy: strategy.clone(),
                        restored_version: None,
                        operators_restored: 0,
                        bytes_restored: 0,
                        recovery_duration_us: start.elapsed().as_micros() as u64,
                        message: format!("Checkpoint v{} not found", version),
                    },
                }
            }
            RecoveryStrategy::Fresh => RecoveryResult {
                success: true,
                strategy: strategy.clone(),
                restored_version: None,
                operators_restored: 0,
                bytes_restored: 0,
                recovery_duration_us: start.elapsed().as_micros() as u64,
                message: "Fresh start - all state discarded".to_string(),
            },
        };

        if result.success {
            self.stats.write().successful_recoveries += 1;
        } else {
            self.stats.write().failed_recoveries += 1;
        }
        {
            let mut stats = self.stats.write();
            let n = stats.recovery_attempts as f64;
            stats.avg_recovery_time_ms = (stats.avg_recovery_time_ms * (n - 1.0)
                + result.recovery_duration_us as f64 / 1000.0)
                / n;
        }

        self.history.write().push(result.clone());
        info!("Recovery: {}", result.message);
        result
    }

    /// Returns the operator states from recovery, if the last recovery was successful.
    pub fn restored_states(&self) -> Option<HashMap<String, Vec<u8>>> {
        let history = self.history.read();
        let last = history.last()?;
        if !last.success {
            return None;
        }
        match &last.strategy {
            RecoveryStrategy::LatestCheckpoint => self
                .checkpoint_manager
                .latest_checkpoint()
                .map(|c| c.operator_states),
            RecoveryStrategy::SpecificVersion(v) => self
                .checkpoint_manager
                .checkpoint_by_version(*v)
                .map(|c| c.operator_states),
            RecoveryStrategy::Fresh => Some(HashMap::new()),
        }
    }

    /// Returns recovery history.
    pub fn history(&self) -> Vec<RecoveryResult> {
        self.history.read().clone()
    }

    /// Returns statistics.
    pub fn stats(&self) -> RecoveryManagerStats {
        self.stats.read().clone()
    }
}

// ─── Partition Rebalancer ────────────────────────────────────────────────────

/// A partition with its current assignment and load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkPartition {
    /// Partition identifier
    pub partition_id: String,
    /// Currently assigned node
    pub assigned_node: String,
    /// Current load (0.0 to 1.0)
    pub load: f64,
    /// Partition weight (higher = more costly to move)
    pub weight: u64,
}

/// A rebalance action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceAction {
    /// Partition to move
    pub partition_id: String,
    /// Move from this node
    pub from_node: String,
    /// Move to this node
    pub to_node: String,
    /// Weight of this partition
    pub weight: u64,
}

/// Configuration for the partition rebalancer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancerConfig {
    /// Maximum load imbalance ratio before rebalancing (e.g., 0.2 = 20%)
    pub imbalance_threshold: f64,
    /// Maximum partitions to move in a single rebalance
    pub max_moves_per_rebalance: usize,
}

impl Default for RebalancerConfig {
    fn default() -> Self {
        Self {
            imbalance_threshold: 0.2,
            max_moves_per_rebalance: 10,
        }
    }
}

/// Statistics for the rebalancer
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RebalancerStats {
    /// Total rebalances performed
    pub rebalances_performed: u64,
    /// Total partitions moved
    pub partitions_moved: u64,
    /// Total weight moved
    pub weight_moved: u64,
}

/// Redistributes work when nodes fail or join.
///
/// Uses a weight-based balancing algorithm: each partition has a weight,
/// and the rebalancer tries to equalize total weight across nodes.
pub struct PartitionRebalancer {
    config: RebalancerConfig,
    /// Active nodes
    nodes: Arc<RwLock<HashSet<String>>>,
    /// Current partition assignments
    partitions: Arc<RwLock<HashMap<String, WorkPartition>>>,
    /// Statistics
    stats: Arc<RwLock<RebalancerStats>>,
}

impl PartitionRebalancer {
    /// Creates a new partition rebalancer.
    pub fn new(config: RebalancerConfig) -> Self {
        Self {
            config,
            nodes: Arc::new(RwLock::new(HashSet::new())),
            partitions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RebalancerStats::default())),
        }
    }

    /// Registers a node.
    pub fn add_node(&self, node_id: impl Into<String>) {
        self.nodes.write().insert(node_id.into());
    }

    /// Removes a node (triggering the need for rebalance).
    pub fn remove_node(&self, node_id: &str) {
        self.nodes.write().remove(node_id);
    }

    /// Registers a partition.
    pub fn add_partition(&self, partition: WorkPartition) {
        self.partitions
            .write()
            .insert(partition.partition_id.clone(), partition);
    }

    /// Returns whether rebalancing is needed based on weight imbalance.
    pub fn needs_rebalance(&self) -> bool {
        let nodes = self.nodes.read();
        let partitions = self.partitions.read();

        if nodes.len() < 2 || partitions.is_empty() {
            return false;
        }

        let weight_per_node = self.compute_weight_per_node(&nodes, &partitions);
        let values: Vec<u64> = weight_per_node.values().copied().collect();
        if values.is_empty() {
            return false;
        }

        let max = values.iter().copied().max().unwrap_or(0);
        let min = values.iter().copied().min().unwrap_or(0);
        let avg = values.iter().sum::<u64>() as f64 / values.len() as f64;

        if avg < 1.0 {
            return false;
        }

        let imbalance = (max - min) as f64 / avg;
        imbalance > self.config.imbalance_threshold
    }

    /// Plans a rebalance to equalize weight across nodes.
    pub fn plan_rebalance(&self) -> Vec<RebalanceAction> {
        let nodes = self.nodes.read();
        let partitions = self.partitions.read();

        if nodes.len() < 2 || partitions.is_empty() {
            return Vec::new();
        }

        let weight_per_node = self.compute_weight_per_node(&nodes, &partitions);
        let total_weight: u64 = partitions.values().map(|p| p.weight).sum();
        let target_weight = total_weight / nodes.len() as u64;

        // Find overloaded and underloaded nodes
        let mut overloaded: Vec<(String, u64)> = Vec::new();
        let mut underloaded: Vec<(String, u64)> = Vec::new();

        for (node, weight) in &weight_per_node {
            if *weight > target_weight + target_weight / 5 {
                overloaded.push((node.clone(), *weight));
            } else if *weight < target_weight.saturating_sub(target_weight / 5) {
                underloaded.push((node.clone(), *weight));
            }
        }

        overloaded.sort_by(|a, b| b.1.cmp(&a.1)); // most overloaded first
        underloaded.sort_by(|a, b| a.1.cmp(&b.1)); // most underloaded first

        let mut actions = Vec::new();
        let mut moved_count = 0;

        for (over_node, _) in &overloaded {
            if moved_count >= self.config.max_moves_per_rebalance {
                break;
            }
            // Find partitions on this overloaded node
            let mut node_partitions: Vec<&WorkPartition> = partitions
                .values()
                .filter(|p| p.assigned_node == *over_node)
                .collect();
            node_partitions.sort_by_key(|p| p.weight); // move lightest first

            for partition in node_partitions {
                if moved_count >= self.config.max_moves_per_rebalance {
                    break;
                }
                if let Some((under_node, _)) = underloaded.first() {
                    actions.push(RebalanceAction {
                        partition_id: partition.partition_id.clone(),
                        from_node: over_node.clone(),
                        to_node: under_node.clone(),
                        weight: partition.weight,
                    });
                    moved_count += 1;
                }
            }
        }

        actions
    }

    /// Executes a set of rebalance actions.
    pub fn execute_rebalance(&self, actions: &[RebalanceAction]) -> usize {
        let mut partitions = self.partitions.write();
        let mut moved = 0;
        let mut weight_moved: u64 = 0;

        for action in actions {
            if let Some(partition) = partitions.get_mut(&action.partition_id) {
                partition.assigned_node = action.to_node.clone();
                moved += 1;
                weight_moved += action.weight;
                debug!(
                    "Rebalanced partition {} from {} to {}",
                    action.partition_id, action.from_node, action.to_node
                );
            }
        }

        let mut stats = self.stats.write();
        stats.rebalances_performed += 1;
        stats.partitions_moved += moved as u64;
        stats.weight_moved += weight_moved;

        info!(
            "Rebalance complete: {} partitions moved (weight: {})",
            moved, weight_moved
        );
        moved
    }

    /// Handles a node failure: removes the node and reassigns its partitions.
    pub fn handle_node_failure(&self, failed_node: &str) -> Vec<RebalanceAction> {
        self.remove_node(failed_node);

        let partitions = self.partitions.read();
        let nodes = self.nodes.read();

        let orphaned: Vec<&WorkPartition> = partitions
            .values()
            .filter(|p| p.assigned_node == failed_node)
            .collect();

        if orphaned.is_empty() || nodes.is_empty() {
            return Vec::new();
        }

        // Round-robin assign orphaned partitions to remaining nodes
        let node_list: Vec<String> = nodes.iter().cloned().collect();
        let mut actions = Vec::new();

        for (i, partition) in orphaned.iter().enumerate() {
            let target = &node_list[i % node_list.len()];
            actions.push(RebalanceAction {
                partition_id: partition.partition_id.clone(),
                from_node: failed_node.to_string(),
                to_node: target.clone(),
                weight: partition.weight,
            });
        }

        actions
    }

    /// Returns current statistics.
    pub fn stats(&self) -> RebalancerStats {
        self.stats.read().clone()
    }

    /// Returns current node count.
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Returns current partition count.
    pub fn partition_count(&self) -> usize {
        self.partitions.read().len()
    }

    fn compute_weight_per_node(
        &self,
        nodes: &HashSet<String>,
        partitions: &HashMap<String, WorkPartition>,
    ) -> HashMap<String, u64> {
        let mut weight_map: HashMap<String, u64> = HashMap::new();
        for node in nodes {
            weight_map.insert(node.clone(), 0);
        }
        for partition in partitions.values() {
            *weight_map
                .entry(partition.assigned_node.clone())
                .or_insert(0) += partition.weight;
        }
        weight_map
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Computes FNV-1a 64-bit hash of data, returned as hex string.
fn compute_fnv1a_hash(data: &[u8]) -> String {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut hash = FNV_OFFSET;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", hash)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CheckpointManager Tests ──────────────────────────────────────────────

    #[test]
    fn test_checkpoint_manager_basic() {
        let mgr = CheckpointManager::new(CheckpointManagerConfig::default());
        let mut states = HashMap::new();
        states.insert("op-1".to_string(), b"state-1".to_vec());

        let ckpt = mgr.take_checkpoint(states).expect("should succeed");
        assert_eq!(ckpt.version, 1);
        assert_eq!(ckpt.operator_states.len(), 1);
        assert!(!ckpt.metadata.integrity_hash.is_empty());
    }

    #[test]
    fn test_checkpoint_manager_retention() {
        let config = CheckpointManagerConfig {
            max_retained: 3,
            ..Default::default()
        };
        let mgr = CheckpointManager::new(config);

        for i in 0..5 {
            let mut states = HashMap::new();
            states.insert("op".to_string(), format!("v{}", i).into_bytes());
            mgr.take_checkpoint(states).expect("should succeed");
        }

        let all = mgr.all_checkpoints();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].version, 5); // most recent
    }

    #[test]
    fn test_checkpoint_manager_verify_integrity() {
        let mgr = CheckpointManager::new(CheckpointManagerConfig::default());
        let mut states = HashMap::new();
        states.insert("op".to_string(), b"data".to_vec());

        let ckpt = mgr.take_checkpoint(states).expect("should succeed");
        assert!(mgr.verify_integrity(&ckpt));

        // Corrupt the checkpoint
        let mut corrupted = ckpt.clone();
        corrupted
            .operator_states
            .insert("op".to_string(), b"corrupted".to_vec());
        assert!(!mgr.verify_integrity(&corrupted));
    }

    #[test]
    fn test_checkpoint_manager_should_checkpoint() {
        let config = CheckpointManagerConfig {
            checkpoint_interval: Duration::from_millis(10),
            min_change_bytes: 0,
            ..Default::default()
        };
        let mgr = CheckpointManager::new(config);
        assert!(mgr.should_checkpoint()); // never checkpointed

        let mut states = HashMap::new();
        states.insert("op".to_string(), b"data".to_vec());
        mgr.take_checkpoint(states).expect("should succeed");
        assert!(!mgr.should_checkpoint()); // just checkpointed

        std::thread::sleep(Duration::from_millis(15));
        assert!(mgr.should_checkpoint());
    }

    #[test]
    fn test_checkpoint_manager_min_change_bytes() {
        let config = CheckpointManagerConfig {
            checkpoint_interval: Duration::from_millis(1),
            min_change_bytes: 100,
            ..Default::default()
        };
        let mgr = CheckpointManager::new(config);
        // Initially true (never checkpointed, min_change=0 accumulated)
        // Need to take first checkpoint, then test
        let mut states = HashMap::new();
        states.insert("op".to_string(), b"data".to_vec());
        mgr.take_checkpoint(states).expect("should succeed");
        std::thread::sleep(Duration::from_millis(5));

        // Not enough changes
        mgr.record_state_change(50);
        assert!(!mgr.should_checkpoint());

        // Enough changes
        mgr.record_state_change(60);
        assert!(mgr.should_checkpoint());
    }

    #[test]
    fn test_checkpoint_manager_stats() {
        let mgr = CheckpointManager::new(CheckpointManagerConfig::default());
        let mut states = HashMap::new();
        states.insert("op".to_string(), b"data123".to_vec());
        mgr.take_checkpoint(states).expect("should succeed");

        let stats = mgr.stats();
        assert_eq!(stats.checkpoints_taken, 1);
        assert!(stats.avg_checkpoint_size > 0.0);
        assert_eq!(stats.retained_checkpoints, 1);
    }

    #[test]
    fn test_checkpoint_by_version() {
        let mgr = CheckpointManager::new(CheckpointManagerConfig::default());
        for i in 0..3 {
            let mut states = HashMap::new();
            states.insert("op".to_string(), format!("v{}", i).into_bytes());
            mgr.take_checkpoint(states).expect("should succeed");
        }

        let ckpt = mgr.checkpoint_by_version(2);
        assert!(ckpt.is_some());
        assert_eq!(ckpt.as_ref().map(|c| c.version), Some(2));

        let missing = mgr.checkpoint_by_version(999);
        assert!(missing.is_none());
    }

    // ── RecoveryManager Tests ────────────────────────────────────────────────

    #[test]
    fn test_recovery_from_latest() {
        let ckpt_mgr = Arc::new(CheckpointManager::new(CheckpointManagerConfig::default()));
        let mut states = HashMap::new();
        states.insert("op".to_string(), b"my-state".to_vec());
        ckpt_mgr.take_checkpoint(states).expect("should succeed");

        let recovery = RecoveryManager::new(ckpt_mgr, RecoveryStrategy::LatestCheckpoint);
        let result = recovery.recover();
        assert!(result.success);
        assert_eq!(result.restored_version, Some(1));
        assert_eq!(result.operators_restored, 1);
    }

    #[test]
    fn test_recovery_no_checkpoint_available() {
        let ckpt_mgr = Arc::new(CheckpointManager::new(CheckpointManagerConfig::default()));
        let recovery = RecoveryManager::new(ckpt_mgr, RecoveryStrategy::LatestCheckpoint);
        let result = recovery.recover();
        assert!(!result.success);
        assert!(result.message.contains("No checkpoint"));
    }

    #[test]
    fn test_recovery_specific_version() {
        let ckpt_mgr = Arc::new(CheckpointManager::new(CheckpointManagerConfig::default()));
        for i in 0..3 {
            let mut states = HashMap::new();
            states.insert("op".to_string(), format!("v{}", i).into_bytes());
            ckpt_mgr.take_checkpoint(states).expect("should succeed");
        }

        let recovery =
            RecoveryManager::new(Arc::clone(&ckpt_mgr), RecoveryStrategy::SpecificVersion(2));
        let result = recovery.recover();
        assert!(result.success);
        assert_eq!(result.restored_version, Some(2));
    }

    #[test]
    fn test_recovery_fresh_start() {
        let ckpt_mgr = Arc::new(CheckpointManager::new(CheckpointManagerConfig::default()));
        let recovery = RecoveryManager::new(ckpt_mgr, RecoveryStrategy::Fresh);
        let result = recovery.recover();
        assert!(result.success);
        assert!(result.restored_version.is_none());
    }

    #[test]
    fn test_recovery_restored_states() {
        let ckpt_mgr = Arc::new(CheckpointManager::new(CheckpointManagerConfig::default()));
        let mut states = HashMap::new();
        states.insert("op-a".to_string(), b"state-a".to_vec());
        states.insert("op-b".to_string(), b"state-b".to_vec());
        ckpt_mgr.take_checkpoint(states).expect("should succeed");

        let recovery =
            RecoveryManager::new(Arc::clone(&ckpt_mgr), RecoveryStrategy::LatestCheckpoint);
        recovery.recover();

        let restored = recovery.restored_states();
        assert!(restored.is_some());
        let restored = restored.expect("should have states");
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get("op-a"), Some(&b"state-a".to_vec()));
    }

    #[test]
    fn test_recovery_stats() {
        let ckpt_mgr = Arc::new(CheckpointManager::new(CheckpointManagerConfig::default()));
        let recovery = RecoveryManager::new(Arc::clone(&ckpt_mgr), RecoveryStrategy::Fresh);
        recovery.recover();
        recovery.recover();

        let stats = recovery.stats();
        assert_eq!(stats.recovery_attempts, 2);
        assert_eq!(stats.successful_recoveries, 2);
    }

    // ── PartitionRebalancer Tests ────────────────────────────────────────────

    #[test]
    fn test_rebalancer_no_rebalance_needed() {
        let rebalancer = PartitionRebalancer::new(RebalancerConfig::default());
        rebalancer.add_node("n1");
        rebalancer.add_node("n2");

        rebalancer.add_partition(WorkPartition {
            partition_id: "p0".to_string(),
            assigned_node: "n1".to_string(),
            load: 0.5,
            weight: 100,
        });
        rebalancer.add_partition(WorkPartition {
            partition_id: "p1".to_string(),
            assigned_node: "n2".to_string(),
            load: 0.5,
            weight: 100,
        });

        assert!(!rebalancer.needs_rebalance());
    }

    #[test]
    fn test_rebalancer_detects_imbalance() {
        let rebalancer = PartitionRebalancer::new(RebalancerConfig {
            imbalance_threshold: 0.2,
            max_moves_per_rebalance: 5,
        });
        rebalancer.add_node("n1");
        rebalancer.add_node("n2");

        // All weight on n1
        for i in 0..4 {
            rebalancer.add_partition(WorkPartition {
                partition_id: format!("p{}", i),
                assigned_node: "n1".to_string(),
                load: 0.5,
                weight: 100,
            });
        }

        assert!(rebalancer.needs_rebalance());
    }

    #[test]
    fn test_rebalancer_plan_and_execute() {
        let rebalancer = PartitionRebalancer::new(RebalancerConfig::default());
        rebalancer.add_node("n1");
        rebalancer.add_node("n2");

        for i in 0..6 {
            rebalancer.add_partition(WorkPartition {
                partition_id: format!("p{}", i),
                assigned_node: "n1".to_string(),
                load: 0.5,
                weight: 50,
            });
        }

        let actions = rebalancer.plan_rebalance();
        assert!(!actions.is_empty());

        let moved = rebalancer.execute_rebalance(&actions);
        assert!(moved > 0);
    }

    #[test]
    fn test_rebalancer_handle_node_failure() {
        let rebalancer = PartitionRebalancer::new(RebalancerConfig::default());
        rebalancer.add_node("n1");
        rebalancer.add_node("n2");
        rebalancer.add_node("n3");

        rebalancer.add_partition(WorkPartition {
            partition_id: "p0".to_string(),
            assigned_node: "n2".to_string(),
            load: 0.5,
            weight: 100,
        });
        rebalancer.add_partition(WorkPartition {
            partition_id: "p1".to_string(),
            assigned_node: "n2".to_string(),
            load: 0.5,
            weight: 100,
        });

        let actions = rebalancer.handle_node_failure("n2");
        assert_eq!(actions.len(), 2); // 2 orphaned partitions

        let moved = rebalancer.execute_rebalance(&actions);
        assert_eq!(moved, 2);
        assert_eq!(rebalancer.node_count(), 2); // n2 removed
    }

    #[test]
    fn test_rebalancer_empty_cluster() {
        let rebalancer = PartitionRebalancer::new(RebalancerConfig::default());
        assert!(!rebalancer.needs_rebalance());
        let actions = rebalancer.plan_rebalance();
        assert!(actions.is_empty());
    }

    #[test]
    fn test_rebalancer_stats() {
        let rebalancer = PartitionRebalancer::new(RebalancerConfig::default());
        rebalancer.add_node("n1");
        rebalancer.add_node("n2");
        for i in 0..4 {
            rebalancer.add_partition(WorkPartition {
                partition_id: format!("p{}", i),
                assigned_node: "n1".to_string(),
                load: 0.5,
                weight: 100,
            });
        }

        let actions = rebalancer.plan_rebalance();
        rebalancer.execute_rebalance(&actions);

        let stats = rebalancer.stats();
        assert_eq!(stats.rebalances_performed, 1);
        assert!(stats.partitions_moved > 0);
    }
}
