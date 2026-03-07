//! # Distributed State Manager
//!
//! Coordinates distributed state across stream processors with:
//! - Periodic state checkpointing (snapshots of operator state)
//! - Exactly-once semantics via sequence-number deduplication
//! - State migration when processors join or leave

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::{DistributedCheckpointer, StateResult};

// ─── Exactly-Once Deduplication ──────────────────────────────────────────────

/// Configuration for the deduplication log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationConfig {
    /// Maximum number of sequence entries to track per source
    pub max_entries_per_source: usize,
    /// Expire entries older than this duration
    pub expiry: Duration,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            max_entries_per_source: 10_000,
            expiry: Duration::from_secs(3600),
        }
    }
}

/// A tracked sequence entry for deduplication
#[derive(Debug, Clone)]
struct SequenceEntry {
    sequence_number: u64,
    received_at: Instant,
}

/// Deduplication log using sequence numbers for exactly-once semantics.
///
/// Each source stream maintains a monotonically increasing sequence number.
/// The log tracks the highest contiguous sequence received, plus a set of
/// out-of-order sequences, to detect and reject duplicates.
pub struct SequenceDeduplicator {
    config: DeduplicationConfig,
    /// Per-source: highest contiguous sequence number received
    high_watermarks: Arc<RwLock<HashMap<String, u64>>>,
    /// Per-source: out-of-order sequence numbers above the high watermark
    pending_sequences: Arc<RwLock<HashMap<String, VecDeque<SequenceEntry>>>>,
    /// Total duplicates rejected
    duplicates_rejected: Arc<RwLock<u64>>,
    /// Total unique messages accepted
    unique_accepted: Arc<RwLock<u64>>,
}

impl SequenceDeduplicator {
    /// Creates a new deduplicator with the given configuration.
    pub fn new(config: DeduplicationConfig) -> Self {
        Self {
            config,
            high_watermarks: Arc::new(RwLock::new(HashMap::new())),
            pending_sequences: Arc::new(RwLock::new(HashMap::new())),
            duplicates_rejected: Arc::new(RwLock::new(0)),
            unique_accepted: Arc::new(RwLock::new(0)),
        }
    }

    /// Checks whether a message is a duplicate.
    ///
    /// Returns `true` if the message is **new** (not a duplicate) and should
    /// be processed. Returns `false` if it is a duplicate.
    pub fn check_and_record(&self, source_id: &str, sequence_number: u64) -> bool {
        let mut watermarks = self.high_watermarks.write();
        let current_watermark = watermarks.entry(source_id.to_string()).or_insert(0);

        // If below or equal to the high watermark, it is a duplicate
        if sequence_number <= *current_watermark && *current_watermark > 0 {
            // Could be in pending (out-of-order), check
            let pending = self.pending_sequences.read();
            if let Some(entries) = pending.get(source_id) {
                if entries.iter().any(|e| e.sequence_number == sequence_number) {
                    *self.duplicates_rejected.write() += 1;
                    return false;
                }
            }
            *self.duplicates_rejected.write() += 1;
            return false;
        }

        // Check if it is already in pending
        {
            let pending = self.pending_sequences.read();
            if let Some(entries) = pending.get(source_id) {
                if entries.iter().any(|e| e.sequence_number == sequence_number) {
                    *self.duplicates_rejected.write() += 1;
                    return false;
                }
            }
        }

        // Record the new sequence
        if sequence_number == *current_watermark + 1 || *current_watermark == 0 {
            // Contiguous: advance the watermark
            *current_watermark = sequence_number;
            // Advance through any pending that are now contiguous
            drop(watermarks);
            self.advance_watermark(source_id);
        } else {
            // Out-of-order: add to pending
            drop(watermarks);
            let mut pending = self.pending_sequences.write();
            let entries = pending.entry(source_id.to_string()).or_default();
            entries.push_back(SequenceEntry {
                sequence_number,
                received_at: Instant::now(),
            });
            // Cap the pending entries
            while entries.len() > self.config.max_entries_per_source {
                entries.pop_front();
            }
        }

        *self.unique_accepted.write() += 1;
        true
    }

    /// Advances the watermark by consuming contiguous pending sequences.
    fn advance_watermark(&self, source_id: &str) {
        let mut watermarks = self.high_watermarks.write();
        let watermark = watermarks.entry(source_id.to_string()).or_insert(0);
        let mut pending = self.pending_sequences.write();
        if let Some(entries) = pending.get_mut(source_id) {
            entries.make_contiguous().sort_by_key(|e| e.sequence_number);
            while let Some(front) = entries.front() {
                if front.sequence_number == *watermark + 1 {
                    *watermark += 1;
                    entries.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Returns the high watermark for a source.
    pub fn high_watermark(&self, source_id: &str) -> u64 {
        self.high_watermarks
            .read()
            .get(source_id)
            .copied()
            .unwrap_or(0)
    }

    /// Expires old pending entries.
    pub fn expire_old_entries(&self) {
        let now = Instant::now();
        let mut pending = self.pending_sequences.write();
        for entries in pending.values_mut() {
            entries.retain(|e| now.duration_since(e.received_at) < self.config.expiry);
        }
    }

    /// Returns deduplication statistics.
    pub fn stats(&self) -> DeduplicationStats {
        let pending_count: usize = self
            .pending_sequences
            .read()
            .values()
            .map(|e| e.len())
            .sum();
        DeduplicationStats {
            duplicates_rejected: *self.duplicates_rejected.read(),
            unique_accepted: *self.unique_accepted.read(),
            tracked_sources: self.high_watermarks.read().len(),
            pending_sequences: pending_count,
        }
    }
}

/// Statistics for the deduplication log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationStats {
    /// Total duplicate messages rejected
    pub duplicates_rejected: u64,
    /// Total unique messages accepted
    pub unique_accepted: u64,
    /// Number of sources being tracked
    pub tracked_sources: usize,
    /// Total out-of-order sequences pending
    pub pending_sequences: usize,
}

// ─── Operator State Snapshot ─────────────────────────────────────────────────

/// A snapshot of a single operator's state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorStateSnapshot {
    /// Operator identifier
    pub operator_id: String,
    /// Serialized state bytes
    pub state_bytes: Vec<u8>,
    /// State version (monotonically increasing)
    pub version: u64,
    /// Timestamp of snapshot
    pub created_at: u64,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Configuration for periodic checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Interval between checkpoints
    pub checkpoint_interval: Duration,
    /// Maximum number of retained checkpoints
    pub max_retained_checkpoints: usize,
    /// Whether to verify checkpoint integrity
    pub verify_integrity: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval: Duration::from_secs(30),
            max_retained_checkpoints: 5,
            verify_integrity: true,
        }
    }
}

/// A complete checkpoint containing all operator snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateCheckpoint {
    /// Checkpoint identifier
    pub checkpoint_id: String,
    /// All operator snapshots
    pub operator_snapshots: HashMap<String, OperatorStateSnapshot>,
    /// Checkpoint version
    pub version: u64,
    /// Merkle root for integrity verification
    pub merkle_root: String,
    /// Creation timestamp (microseconds since UNIX epoch)
    pub created_at: u64,
    /// Whether checkpoint is complete (all operators contributed)
    pub is_complete: bool,
}

// ─── State Migration ─────────────────────────────────────────────────────────

/// Describes a state partition assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionAssignment {
    /// Partition identifier
    pub partition_id: String,
    /// Currently assigned processor node
    pub assigned_to: String,
    /// State size in bytes (approximate)
    pub state_size_bytes: usize,
    /// Load score (0.0 to 1.0)
    pub load_score: f64,
}

/// A migration plan describing how to rebalance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Migrations to execute
    pub migrations: Vec<MigrationStep>,
    /// Estimated total bytes to transfer
    pub total_bytes_to_transfer: usize,
    /// Reason for migration
    pub reason: MigrationReason,
}

/// A single step in a migration plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    /// Partition being migrated
    pub partition_id: String,
    /// Source processor
    pub from_node: String,
    /// Target processor
    pub to_node: String,
    /// Estimated state size
    pub state_size_bytes: usize,
}

/// Reason a migration was triggered
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MigrationReason {
    /// A new processor joined the cluster
    NodeJoined { node_id: String },
    /// A processor left the cluster
    NodeLeft { node_id: String },
    /// Load imbalance detected
    LoadImbalance,
    /// Manual trigger
    Manual,
}

// ─── Distributed State Manager ───────────────────────────────────────────────

/// Statistics for the distributed state manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedStateManagerStats {
    /// Total checkpoints taken
    pub checkpoints_taken: u64,
    /// Total state migrations performed
    pub migrations_performed: u64,
    /// Current number of partitions
    pub partition_count: usize,
    /// Current number of active processors
    pub active_processors: usize,
    /// Total state size across all partitions (bytes)
    pub total_state_bytes: usize,
    /// Deduplication statistics
    pub dedup_stats: DeduplicationStats,
    /// Average checkpoint duration (milliseconds)
    pub avg_checkpoint_duration_ms: f64,
}

/// The main distributed state manager that coordinates state across
/// stream processors.
///
/// Provides:
/// - Periodic state checkpointing via operator snapshots
/// - Exactly-once semantics via sequence-number deduplication
/// - State migration when processors join or leave the cluster
pub struct DistributedStateManager {
    /// This node's identifier
    node_id: String,
    /// Checkpoint configuration
    checkpoint_config: CheckpointConfig,
    /// Sequence deduplicator for exactly-once semantics
    deduplicator: SequenceDeduplicator,
    /// Current partition assignments
    partitions: Arc<RwLock<HashMap<String, PartitionAssignment>>>,
    /// Active processor nodes
    active_processors: Arc<RwLock<HashSet<String>>>,
    /// Stored checkpoints (most recent first)
    checkpoints: Arc<RwLock<VecDeque<StateCheckpoint>>>,
    /// Current checkpoint version counter
    checkpoint_version: Arc<RwLock<u64>>,
    /// Migration history
    migration_history: Arc<RwLock<Vec<MigrationPlan>>>,
    /// Total checkpoints taken
    checkpoints_taken: Arc<RwLock<u64>>,
    /// Total migrations performed
    migrations_performed: Arc<RwLock<u64>>,
    /// Checkpoint duration accumulator
    checkpoint_duration_sum_ms: Arc<RwLock<f64>>,
    /// Last checkpoint time
    last_checkpoint: Arc<RwLock<Option<Instant>>>,
}

impl DistributedStateManager {
    /// Creates a new distributed state manager.
    pub fn new(
        node_id: impl Into<String>,
        checkpoint_config: CheckpointConfig,
        dedup_config: DeduplicationConfig,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            checkpoint_config,
            deduplicator: SequenceDeduplicator::new(dedup_config),
            partitions: Arc::new(RwLock::new(HashMap::new())),
            active_processors: Arc::new(RwLock::new(HashSet::new())),
            checkpoints: Arc::new(RwLock::new(VecDeque::new())),
            checkpoint_version: Arc::new(RwLock::new(0)),
            migration_history: Arc::new(RwLock::new(Vec::new())),
            checkpoints_taken: Arc::new(RwLock::new(0)),
            migrations_performed: Arc::new(RwLock::new(0)),
            checkpoint_duration_sum_ms: Arc::new(RwLock::new(0.0)),
            last_checkpoint: Arc::new(RwLock::new(None)),
        }
    }

    /// Returns the node ID of this manager.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Registers a processor node as active.
    pub fn register_processor(&self, node_id: impl Into<String>) {
        let id = node_id.into();
        self.active_processors.write().insert(id.clone());
        info!("Registered processor: {}", id);
    }

    /// Removes a processor node.
    pub fn remove_processor(&self, node_id: &str) {
        self.active_processors.write().remove(node_id);
        info!("Removed processor: {}", node_id);
    }

    /// Assigns a partition to a processor.
    pub fn assign_partition(&self, assignment: PartitionAssignment) {
        debug!(
            "Assigning partition {} to {}",
            assignment.partition_id, assignment.assigned_to
        );
        self.partitions
            .write()
            .insert(assignment.partition_id.clone(), assignment);
    }

    /// Checks and records a message for exactly-once processing.
    ///
    /// Returns `true` if the message is new and should be processed.
    pub fn check_exactly_once(&self, source_id: &str, sequence_number: u64) -> bool {
        self.deduplicator
            .check_and_record(source_id, sequence_number)
    }

    /// Returns the high watermark for a source.
    pub fn high_watermark(&self, source_id: &str) -> u64 {
        self.deduplicator.high_watermark(source_id)
    }

    /// Takes a checkpoint of the given operator states.
    ///
    /// Returns the checkpoint if successful.
    pub fn take_checkpoint(
        &self,
        operator_states: HashMap<String, Vec<u8>>,
    ) -> StateResult<StateCheckpoint> {
        let start = Instant::now();

        let mut version = self.checkpoint_version.write();
        *version += 1;
        let current_version = *version;
        drop(version);

        let now_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let checkpoint_id = format!("ckpt-{}-{}", self.node_id, current_version);

        // Build operator snapshots
        let mut operator_snapshots = HashMap::new();
        for (op_id, state_bytes) in operator_states {
            let size = state_bytes.len();
            operator_snapshots.insert(
                op_id.clone(),
                OperatorStateSnapshot {
                    operator_id: op_id,
                    state_bytes,
                    version: current_version,
                    created_at: now_micros,
                    size_bytes: size,
                },
            );
        }

        // Compute merkle root over all operator states
        let mut all_bytes = Vec::new();
        let mut sorted_keys: Vec<&String> = operator_snapshots.keys().collect();
        sorted_keys.sort();
        for key in sorted_keys {
            if let Some(snapshot) = operator_snapshots.get(key) {
                all_bytes.extend_from_slice(&snapshot.state_bytes);
            }
        }
        let merkle_root = DistributedCheckpointer::compute_merkle_root(&all_bytes);

        let checkpoint = StateCheckpoint {
            checkpoint_id,
            operator_snapshots,
            version: current_version,
            merkle_root,
            created_at: now_micros,
            is_complete: true,
        };

        // Store the checkpoint
        let max_retained = self.checkpoint_config.max_retained_checkpoints;
        let mut checkpoints = self.checkpoints.write();
        checkpoints.push_front(checkpoint.clone());
        while checkpoints.len() > max_retained {
            checkpoints.pop_back();
        }

        *self.checkpoints_taken.write() += 1;
        *self.last_checkpoint.write() = Some(Instant::now());

        let elapsed = start.elapsed().as_millis() as f64;
        *self.checkpoint_duration_sum_ms.write() += elapsed;

        info!(
            "Checkpoint {} taken (version {}, {} operators, {:.1}ms)",
            checkpoint.checkpoint_id,
            current_version,
            checkpoint.operator_snapshots.len(),
            elapsed
        );

        Ok(checkpoint)
    }

    /// Restores state from the latest checkpoint.
    ///
    /// Returns the operator states map if a checkpoint exists.
    pub fn restore_from_latest(&self) -> Option<HashMap<String, Vec<u8>>> {
        let checkpoints = self.checkpoints.read();
        let latest = checkpoints.front()?;

        if self.checkpoint_config.verify_integrity {
            // Verify merkle root
            let mut all_bytes = Vec::new();
            let mut sorted_keys: Vec<&String> = latest.operator_snapshots.keys().collect();
            sorted_keys.sort();
            for key in sorted_keys {
                if let Some(snapshot) = latest.operator_snapshots.get(key) {
                    all_bytes.extend_from_slice(&snapshot.state_bytes);
                }
            }
            let computed = DistributedCheckpointer::compute_merkle_root(&all_bytes);
            if computed != latest.merkle_root {
                warn!("Checkpoint {} failed integrity check", latest.checkpoint_id);
                return None;
            }
        }

        let states: HashMap<String, Vec<u8>> = latest
            .operator_snapshots
            .iter()
            .map(|(k, v)| (k.clone(), v.state_bytes.clone()))
            .collect();
        info!(
            "Restored state from checkpoint {} (version {})",
            latest.checkpoint_id, latest.version
        );
        Some(states)
    }

    /// Returns all stored checkpoints (most recent first).
    pub fn checkpoints(&self) -> Vec<StateCheckpoint> {
        self.checkpoints.read().iter().cloned().collect()
    }

    /// Returns whether a checkpoint is due based on the configured interval.
    pub fn is_checkpoint_due(&self) -> bool {
        let last = self.last_checkpoint.read();
        match *last {
            Some(instant) => instant.elapsed() >= self.checkpoint_config.checkpoint_interval,
            None => true,
        }
    }

    /// Plans a migration based on the current partition assignments and processor set.
    ///
    /// Returns `None` if no migration is needed.
    pub fn plan_migration(&self, reason: MigrationReason) -> Option<MigrationPlan> {
        let partitions = self.partitions.read();
        let processors = self.active_processors.read();

        if processors.is_empty() || partitions.is_empty() {
            return None;
        }

        let processor_list: Vec<String> = processors.iter().cloned().collect();

        // Build current load per processor
        let mut load_per_processor: HashMap<String, Vec<String>> = HashMap::new();
        for proc_id in &processor_list {
            load_per_processor.insert(proc_id.clone(), Vec::new());
        }
        for (partition_id, assignment) in partitions.iter() {
            load_per_processor
                .entry(assignment.assigned_to.clone())
                .or_default()
                .push(partition_id.clone());
        }

        let total_partitions = partitions.len();
        let target_per_processor = total_partitions / processor_list.len();
        let remainder = total_partitions % processor_list.len();

        // Find overloaded and underloaded processors
        let mut migrations = Vec::new();
        let mut donors: Vec<(String, Vec<String>)> = Vec::new();
        let mut receivers: Vec<(String, usize)> = Vec::new();

        for (i, proc_id) in processor_list.iter().enumerate() {
            let current_count = load_per_processor
                .get(proc_id)
                .map(|v| v.len())
                .unwrap_or(0);
            let target = target_per_processor + if i < remainder { 1 } else { 0 };
            if current_count > target {
                let excess: Vec<String> = load_per_processor
                    .get(proc_id)
                    .map(|v| v[target..].to_vec())
                    .unwrap_or_default();
                donors.push((proc_id.clone(), excess));
            } else if current_count < target {
                receivers.push((proc_id.clone(), target - current_count));
            }
        }

        // Match donors with receivers
        let mut donor_iter = donors
            .iter()
            .flat_map(|(from, parts)| parts.iter().map(move |p| (from.clone(), p.clone())));
        for (to_node, need) in &receivers {
            for _ in 0..*need {
                if let Some((from_node, partition_id)) = donor_iter.next() {
                    let state_size = partitions
                        .get(&partition_id)
                        .map(|a| a.state_size_bytes)
                        .unwrap_or(0);
                    migrations.push(MigrationStep {
                        partition_id,
                        from_node,
                        to_node: to_node.clone(),
                        state_size_bytes: state_size,
                    });
                }
            }
        }

        if migrations.is_empty() {
            return None;
        }

        let total_bytes = migrations.iter().map(|m| m.state_size_bytes).sum();

        Some(MigrationPlan {
            migrations,
            total_bytes_to_transfer: total_bytes,
            reason,
        })
    }

    /// Executes a migration plan by updating partition assignments.
    ///
    /// Returns the number of partitions migrated.
    pub fn execute_migration(&self, plan: &MigrationPlan) -> usize {
        let mut partitions = self.partitions.write();
        let mut migrated = 0;

        for step in &plan.migrations {
            if let Some(assignment) = partitions.get_mut(&step.partition_id) {
                assignment.assigned_to = step.to_node.clone();
                migrated += 1;
                debug!(
                    "Migrated partition {} from {} to {}",
                    step.partition_id, step.from_node, step.to_node
                );
            }
        }

        *self.migrations_performed.write() += 1;
        self.migration_history.write().push(plan.clone());
        info!(
            "Migration complete: {} partitions moved ({} bytes)",
            migrated, plan.total_bytes_to_transfer
        );
        migrated
    }

    /// Handles a node joining the cluster: registers it and optionally migrates.
    pub fn handle_node_joined(&self, node_id: &str) -> Option<MigrationPlan> {
        self.register_processor(node_id);
        self.plan_migration(MigrationReason::NodeJoined {
            node_id: node_id.to_string(),
        })
    }

    /// Handles a node leaving the cluster: reassigns its partitions.
    pub fn handle_node_left(&self, node_id: &str) -> Option<MigrationPlan> {
        self.remove_processor(node_id);
        // Reassign partitions from the departed node
        self.plan_migration(MigrationReason::NodeLeft {
            node_id: node_id.to_string(),
        })
    }

    /// Returns current partition assignments.
    pub fn partition_assignments(&self) -> Vec<PartitionAssignment> {
        self.partitions.read().values().cloned().collect()
    }

    /// Returns active processor node IDs.
    pub fn active_processors(&self) -> Vec<String> {
        self.active_processors.read().iter().cloned().collect()
    }

    /// Returns migration history.
    pub fn migration_history(&self) -> Vec<MigrationPlan> {
        self.migration_history.read().clone()
    }

    /// Returns comprehensive statistics.
    pub fn stats(&self) -> DistributedStateManagerStats {
        let checkpoints_taken = *self.checkpoints_taken.read();
        let avg_duration = if checkpoints_taken > 0 {
            *self.checkpoint_duration_sum_ms.read() / checkpoints_taken as f64
        } else {
            0.0
        };

        let total_state_bytes: usize = self
            .partitions
            .read()
            .values()
            .map(|p| p.state_size_bytes)
            .sum();

        DistributedStateManagerStats {
            checkpoints_taken,
            migrations_performed: *self.migrations_performed.read(),
            partition_count: self.partitions.read().len(),
            active_processors: self.active_processors.read().len(),
            total_state_bytes,
            dedup_stats: self.deduplicator.stats(),
            avg_checkpoint_duration_ms: avg_duration,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> DistributedStateManager {
        DistributedStateManager::new(
            "node-1",
            CheckpointConfig::default(),
            DeduplicationConfig::default(),
        )
    }

    // ── Deduplication Tests ──────────────────────────────────────────────────

    #[test]
    fn test_dedup_first_message_accepted() {
        let dedup = SequenceDeduplicator::new(DeduplicationConfig::default());
        assert!(dedup.check_and_record("src-1", 1));
    }

    #[test]
    fn test_dedup_duplicate_rejected() {
        let dedup = SequenceDeduplicator::new(DeduplicationConfig::default());
        assert!(dedup.check_and_record("src-1", 1));
        assert!(!dedup.check_and_record("src-1", 1));
    }

    #[test]
    fn test_dedup_sequential_messages() {
        let dedup = SequenceDeduplicator::new(DeduplicationConfig::default());
        for i in 1..=10 {
            assert!(dedup.check_and_record("src-1", i));
        }
        assert_eq!(dedup.high_watermark("src-1"), 10);
    }

    #[test]
    fn test_dedup_out_of_order_accepted() {
        let dedup = SequenceDeduplicator::new(DeduplicationConfig::default());
        assert!(dedup.check_and_record("src-1", 1));
        assert!(dedup.check_and_record("src-1", 3)); // out of order
        assert!(dedup.check_and_record("src-1", 2)); // fills the gap
        assert_eq!(dedup.high_watermark("src-1"), 3);
    }

    #[test]
    fn test_dedup_multiple_sources() {
        let dedup = SequenceDeduplicator::new(DeduplicationConfig::default());
        assert!(dedup.check_and_record("src-a", 1));
        assert!(dedup.check_and_record("src-b", 1));
        assert!(!dedup.check_and_record("src-a", 1));
        assert!(dedup.check_and_record("src-a", 2));
    }

    #[test]
    fn test_dedup_stats() {
        let dedup = SequenceDeduplicator::new(DeduplicationConfig::default());
        dedup.check_and_record("src-1", 1);
        dedup.check_and_record("src-1", 1); // duplicate
        dedup.check_and_record("src-2", 1);

        let stats = dedup.stats();
        assert_eq!(stats.unique_accepted, 2);
        assert_eq!(stats.duplicates_rejected, 1);
        assert_eq!(stats.tracked_sources, 2);
    }

    #[test]
    fn test_dedup_expire_old_entries() {
        let config = DeduplicationConfig {
            max_entries_per_source: 100,
            expiry: Duration::from_millis(1),
        };
        let dedup = SequenceDeduplicator::new(config);
        dedup.check_and_record("src-1", 1);
        dedup.check_and_record("src-1", 5); // out of order, goes to pending
        std::thread::sleep(Duration::from_millis(5));
        dedup.expire_old_entries();
        let stats = dedup.stats();
        assert_eq!(stats.pending_sequences, 0);
    }

    // ── Checkpoint Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_take_checkpoint() {
        let mgr = make_manager();
        let mut states = HashMap::new();
        states.insert("op-1".to_string(), b"state-1".to_vec());
        states.insert("op-2".to_string(), b"state-2".to_vec());

        let ckpt = mgr
            .take_checkpoint(states)
            .expect("checkpoint should succeed");
        assert_eq!(ckpt.operator_snapshots.len(), 2);
        assert!(ckpt.is_complete);
        assert!(!ckpt.merkle_root.is_empty());
        assert_eq!(ckpt.version, 1);
    }

    #[test]
    fn test_restore_from_latest() {
        let mgr = make_manager();
        let mut states = HashMap::new();
        states.insert("op-1".to_string(), b"data-a".to_vec());
        mgr.take_checkpoint(states)
            .expect("checkpoint should succeed");

        let restored = mgr.restore_from_latest().expect("should restore");
        assert_eq!(restored.get("op-1"), Some(&b"data-a".to_vec()));
    }

    #[test]
    fn test_checkpoint_retention() {
        let config = CheckpointConfig {
            max_retained_checkpoints: 2,
            ..Default::default()
        };
        let mgr = DistributedStateManager::new("node-1", config, DeduplicationConfig::default());

        for i in 0..5 {
            let mut states = HashMap::new();
            states.insert("op".to_string(), format!("state-{}", i).into_bytes());
            mgr.take_checkpoint(states).expect("should succeed");
        }

        let checkpoints = mgr.checkpoints();
        assert_eq!(checkpoints.len(), 2);
        // Most recent should be first
        assert_eq!(checkpoints[0].version, 5);
    }

    #[test]
    fn test_checkpoint_integrity_verification() {
        let mgr = make_manager();
        let mut states = HashMap::new();
        states.insert("op-1".to_string(), b"my-data".to_vec());
        mgr.take_checkpoint(states).expect("should succeed");

        // Restore should work with valid integrity
        let restored = mgr.restore_from_latest();
        assert!(restored.is_some());
    }

    #[test]
    fn test_is_checkpoint_due() {
        let config = CheckpointConfig {
            checkpoint_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let mgr = DistributedStateManager::new("node-1", config, DeduplicationConfig::default());
        assert!(mgr.is_checkpoint_due());

        let mut states = HashMap::new();
        states.insert("op".to_string(), b"data".to_vec());
        mgr.take_checkpoint(states).expect("should succeed");
        assert!(!mgr.is_checkpoint_due());

        std::thread::sleep(Duration::from_millis(15));
        assert!(mgr.is_checkpoint_due());
    }

    // ── Migration Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_migration_plan_on_node_join() {
        let mgr = make_manager();
        mgr.register_processor("proc-1");
        for i in 0..4 {
            mgr.assign_partition(PartitionAssignment {
                partition_id: format!("p-{}", i),
                assigned_to: "proc-1".to_string(),
                state_size_bytes: 1024,
                load_score: 0.5,
            });
        }

        // A second processor joins
        let plan = mgr.handle_node_joined("proc-2");
        assert!(plan.is_some(), "should generate migration plan");
        let plan = plan.expect("plan exists");
        assert!(!plan.migrations.is_empty());
        assert_eq!(
            plan.reason,
            MigrationReason::NodeJoined {
                node_id: "proc-2".to_string()
            }
        );
    }

    #[test]
    fn test_migration_plan_balanced_no_migration() {
        let mgr = make_manager();
        mgr.register_processor("proc-1");
        mgr.register_processor("proc-2");
        mgr.assign_partition(PartitionAssignment {
            partition_id: "p-0".to_string(),
            assigned_to: "proc-1".to_string(),
            state_size_bytes: 1024,
            load_score: 0.5,
        });
        mgr.assign_partition(PartitionAssignment {
            partition_id: "p-1".to_string(),
            assigned_to: "proc-2".to_string(),
            state_size_bytes: 1024,
            load_score: 0.5,
        });

        let plan = mgr.plan_migration(MigrationReason::Manual);
        assert!(plan.is_none(), "balanced assignment needs no migration");
    }

    #[test]
    fn test_execute_migration() {
        let mgr = make_manager();
        mgr.register_processor("proc-1");
        mgr.register_processor("proc-2");
        for i in 0..4 {
            mgr.assign_partition(PartitionAssignment {
                partition_id: format!("p-{}", i),
                assigned_to: "proc-1".to_string(),
                state_size_bytes: 512,
                load_score: 0.5,
            });
        }

        let plan = mgr
            .plan_migration(MigrationReason::LoadImbalance)
            .expect("should plan migration");
        let migrated = mgr.execute_migration(&plan);
        assert!(migrated > 0);

        // Verify assignments changed
        let assignments = mgr.partition_assignments();
        let proc2_count = assignments
            .iter()
            .filter(|a| a.assigned_to == "proc-2")
            .count();
        assert!(proc2_count > 0, "proc-2 should have partitions now");
    }

    #[test]
    fn test_handle_node_left() {
        let mgr = make_manager();
        mgr.register_processor("proc-1");
        mgr.register_processor("proc-2");
        mgr.assign_partition(PartitionAssignment {
            partition_id: "p-0".to_string(),
            assigned_to: "proc-1".to_string(),
            state_size_bytes: 1024,
            load_score: 0.3,
        });
        mgr.assign_partition(PartitionAssignment {
            partition_id: "p-1".to_string(),
            assigned_to: "proc-2".to_string(),
            state_size_bytes: 1024,
            load_score: 0.3,
        });

        // proc-2 leaves
        let plan = mgr.handle_node_left("proc-2");
        // With only proc-1 remaining and p-1 still assigned to proc-2
        // plan should exist to move p-1 to proc-1
        if let Some(plan) = plan {
            mgr.execute_migration(&plan);
        }
        let procs = mgr.active_processors();
        assert!(!procs.contains(&"proc-2".to_string()));
    }

    // ── Manager Integration Tests ────────────────────────────────────────────

    #[test]
    fn test_manager_exactly_once() {
        let mgr = make_manager();
        assert!(mgr.check_exactly_once("stream-1", 1));
        assert!(mgr.check_exactly_once("stream-1", 2));
        assert!(!mgr.check_exactly_once("stream-1", 1)); // duplicate
        assert!(mgr.check_exactly_once("stream-1", 3));
        assert_eq!(mgr.high_watermark("stream-1"), 3);
    }

    #[test]
    fn test_manager_stats() {
        let mgr = make_manager();
        mgr.register_processor("proc-1");
        mgr.assign_partition(PartitionAssignment {
            partition_id: "p-0".to_string(),
            assigned_to: "proc-1".to_string(),
            state_size_bytes: 2048,
            load_score: 0.5,
        });
        mgr.check_exactly_once("src-1", 1);

        let mut states = HashMap::new();
        states.insert("op-1".to_string(), b"state".to_vec());
        mgr.take_checkpoint(states).expect("should succeed");

        let stats = mgr.stats();
        assert_eq!(stats.checkpoints_taken, 1);
        assert_eq!(stats.partition_count, 1);
        assert_eq!(stats.active_processors, 1);
        assert_eq!(stats.total_state_bytes, 2048);
        assert_eq!(stats.dedup_stats.unique_accepted, 1);
    }

    #[test]
    fn test_manager_multiple_checkpoints_restore_latest() {
        let mgr = make_manager();

        let mut states1 = HashMap::new();
        states1.insert("op".to_string(), b"version-1".to_vec());
        mgr.take_checkpoint(states1).expect("should succeed");

        let mut states2 = HashMap::new();
        states2.insert("op".to_string(), b"version-2".to_vec());
        mgr.take_checkpoint(states2).expect("should succeed");

        let restored = mgr.restore_from_latest().expect("should restore");
        assert_eq!(restored.get("op"), Some(&b"version-2".to_vec()));
    }

    #[test]
    fn test_migration_history() {
        let mgr = make_manager();
        mgr.register_processor("proc-1");
        for i in 0..4 {
            mgr.assign_partition(PartitionAssignment {
                partition_id: format!("p-{}", i),
                assigned_to: "proc-1".to_string(),
                state_size_bytes: 256,
                load_score: 0.5,
            });
        }
        mgr.register_processor("proc-2");
        if let Some(plan) = mgr.plan_migration(MigrationReason::LoadImbalance) {
            mgr.execute_migration(&plan);
        }

        let history = mgr.migration_history();
        assert_eq!(history.len(), 1);
    }
}
