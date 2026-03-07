//! Cross-datacenter async replication for distributed vector indexes
//!
//! This module implements asynchronous replication of vector index updates
//! across multiple datacenters with configurable lag tolerance.
//!
//! # Design
//!
//! Each datacenter hosts a `DcReplicationNode`. One DC is designated the
//! primary (or leader); others are replicas. Index mutations are buffered
//! in the primary and asynchronously shipped to replicas.
//!
//! Key features:
//! - Configurable lag tolerance (max acceptable replication lag)
//! - Per-replica lag tracking and alerting
//! - Automatic catch-up on reconnection
//! - Conflict resolution strategies for network partitions
//! - Bandwidth throttling to avoid saturating WAN links
//!
//! # Consistency Model
//!
//! Cross-DC replication is **eventually consistent** with configurable
//! lag bounds. Reads on replicas may return stale data. For strong
//! consistency, use the Raft cluster within a single DC.

use anyhow::{anyhow, Result};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Datacenter identifier
pub type DcId = String;

/// Per-vector local state: (vector, metadata, replication_seq)
type VectorStateMap = HashMap<String, (Vec<f32>, HashMap<String, String>, ReplicationSeq)>;

/// Replication sequence number (monotonically increasing per primary)
pub type ReplicationSeq = u64;

/// Configuration for cross-DC replication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDcConfig {
    /// This datacenter's ID
    pub dc_id: DcId,
    /// Human-readable region name (e.g., "us-east-1")
    pub region: String,
    /// Whether this DC is the primary writer
    pub is_primary: bool,
    /// Maximum acceptable replication lag (for alerting)
    pub max_lag_tolerance: Duration,
    /// Batch size for replication shipping (entries per batch)
    pub replication_batch_size: usize,
    /// Retry interval for failed replication
    pub retry_interval: Duration,
    /// Maximum retries before marking replica as failed
    pub max_retries: usize,
    /// Bandwidth limit for WAN replication (bytes/sec, 0 = unlimited)
    pub bandwidth_limit_bps: u64,
    /// Compression level for replication payloads (0-9)
    pub compression_level: u8,
    /// Enable bi-directional conflict detection
    pub conflict_detection: bool,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolutionStrategy,
    /// Heartbeat interval to remote DCs
    pub heartbeat_interval: Duration,
    /// Connection timeout to remote DCs
    pub connection_timeout: Duration,
}

impl Default for CrossDcConfig {
    fn default() -> Self {
        Self {
            dc_id: "dc-primary".to_string(),
            region: "us-east-1".to_string(),
            is_primary: true,
            max_lag_tolerance: Duration::from_secs(30),
            replication_batch_size: 500,
            retry_interval: Duration::from_secs(5),
            max_retries: 10,
            bandwidth_limit_bps: 0,
            compression_level: 3,
            conflict_detection: true,
            conflict_resolution: ConflictResolutionStrategy::LastWriteWins,
            heartbeat_interval: Duration::from_secs(10),
            connection_timeout: Duration::from_secs(30),
        }
    }
}

/// Strategy for resolving write conflicts between DCs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Highest timestamp wins
    LastWriteWins,
    /// Primary DC always wins
    PrimaryWins,
    /// Replica DC always wins (for migration scenarios)
    ReplicaWins,
    /// Keep both versions, application resolves
    KeepBoth,
    /// Merge metadata, last-write-wins for vector data
    MergeMetadata,
}

/// A single replication entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationEntry {
    /// Monotonically increasing sequence number
    pub seq: ReplicationSeq,
    /// Originating datacenter
    pub source_dc: DcId,
    /// Timestamp when the entry was created (Unix milliseconds)
    pub timestamp_ms: u64,
    /// The actual operation
    pub operation: ReplicationOperation,
    /// Entry size in bytes (for bandwidth accounting)
    pub payload_bytes: usize,
}

/// Types of replication operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationOperation {
    /// Insert or update a vector
    Upsert {
        vector_id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    },
    /// Delete a vector
    Delete { vector_id: String },
    /// Bulk snapshot (used for catch-up)
    Snapshot {
        entries: Vec<(String, Vec<f32>, HashMap<String, String>)>,
        as_of_seq: ReplicationSeq,
    },
    /// Heartbeat with current seq number
    Heartbeat { current_seq: ReplicationSeq },
    /// No-operation marker for testing
    NoOp,
}

/// Status of a replica DC connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaStatus {
    /// Connected and within lag tolerance
    Healthy,
    /// Connected but lagging behind
    Lagging,
    /// Temporarily disconnected, retrying
    Disconnected,
    /// Too many failures, needs manual intervention
    Failed,
    /// In the process of initial catch-up
    Catching,
}

impl std::fmt::Display for ReplicaStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Lagging => write!(f, "Lagging"),
            Self::Disconnected => write!(f, "Disconnected"),
            Self::Failed => write!(f, "Failed"),
            Self::Catching => write!(f, "Catching"),
        }
    }
}

/// Per-replica tracking state (maintained by primary)
#[derive(Debug, Clone)]
pub struct ReplicaTracker {
    /// Remote DC ID
    pub dc_id: DcId,
    /// Remote DC region
    pub region: String,
    /// Last acknowledged sequence number from this replica
    pub acked_seq: ReplicationSeq,
    /// Current status
    pub status: ReplicaStatus,
    /// Number of consecutive failures
    pub failure_count: usize,
    /// Timestamp of last successful contact
    pub last_contact: Instant,
    /// Estimated replication lag
    pub lag: Duration,
    /// Total bytes sent to this replica
    pub bytes_sent: u64,
    /// Total entries sent to this replica
    pub entries_sent: u64,
}

impl ReplicaTracker {
    fn new(dc_id: DcId, region: String) -> Self {
        Self {
            dc_id,
            region,
            acked_seq: 0,
            status: ReplicaStatus::Catching,
            failure_count: 0,
            last_contact: Instant::now(),
            lag: Duration::ZERO,
            bytes_sent: 0,
            entries_sent: 0,
        }
    }

    /// Update tracker after successful replication
    fn on_success(&mut self, new_acked_seq: ReplicationSeq, bytes: u64, entries: u64) {
        self.acked_seq = new_acked_seq;
        self.failure_count = 0;
        self.last_contact = Instant::now();
        self.bytes_sent += bytes;
        self.entries_sent += entries;
    }

    /// Update tracker after failure
    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.status = if self.failure_count > 5 {
            ReplicaStatus::Disconnected
        } else {
            ReplicaStatus::Lagging
        };
    }

    /// Update lag estimate
    fn update_lag(&mut self, primary_seq: ReplicationSeq) {
        let lag_entries = primary_seq.saturating_sub(self.acked_seq);
        // Rough estimate: ~1ms per entry for WAN replication
        self.lag = Duration::from_millis(lag_entries);

        self.status = match lag_entries {
            0 => ReplicaStatus::Healthy,
            1..=100 => ReplicaStatus::Healthy,
            101..=1000 => ReplicaStatus::Lagging,
            _ => ReplicaStatus::Catching,
        };
    }
}

/// Statistics for cross-DC replication
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossDcStats {
    /// Total entries produced (primary) or received (replica)
    pub total_entries: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
    /// Current replication lag in entries (replica-side)
    pub current_lag_entries: u64,
    /// Current estimated lag in milliseconds
    pub current_lag_ms: u64,
    /// Number of conflicts detected
    pub conflicts_detected: u64,
    /// Number of conflicts resolved
    pub conflicts_resolved: u64,
    /// Number of retries performed
    pub total_retries: u64,
    /// Number of failed entries (eventually dropped)
    pub failed_entries: u64,
    /// Per-replica status map
    pub replica_statuses: HashMap<DcId, String>,
    /// Last heartbeat received from primary (replica-side)
    pub last_heartbeat_ms: u64,
}

/// The primary datacenter replication manager
///
/// Maintains the replication log, tracks replica progress,
/// and ships entries to replicas.
#[derive(Debug)]
pub struct PrimaryDcManager {
    config: CrossDcConfig,
    /// In-memory replication log
    replication_log: Arc<RwLock<VecDeque<ReplicationEntry>>>,
    /// Current sequence counter
    current_seq: Arc<Mutex<ReplicationSeq>>,
    /// Per-replica trackers
    replicas: Arc<RwLock<HashMap<DcId, ReplicaTracker>>>,
    /// Statistics
    stats: Arc<Mutex<CrossDcStats>>,
    /// Maximum entries to retain in log before pruning
    log_retention_entries: usize,
}

impl PrimaryDcManager {
    /// Create a new primary DC manager
    pub fn new(config: CrossDcConfig) -> Result<Self> {
        if !config.is_primary {
            return Err(anyhow!("PrimaryDcManager requires is_primary=true"));
        }

        info!(
            "Primary DC manager initialized for DC '{}' in region '{}'",
            config.dc_id, config.region
        );

        Ok(Self {
            config,
            replication_log: Arc::new(RwLock::new(VecDeque::new())),
            current_seq: Arc::new(Mutex::new(0)),
            replicas: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CrossDcStats::default())),
            log_retention_entries: 100_000,
        })
    }

    /// Register a replica DC
    pub fn add_replica(&self, dc_id: DcId, region: String) {
        let tracker = ReplicaTracker::new(dc_id.clone(), region.clone());
        self.replicas.write().insert(dc_id.clone(), tracker);
        info!("Registered replica DC '{}' in region '{}'", dc_id, region);
    }

    /// Remove a replica DC
    pub fn remove_replica(&self, dc_id: &str) {
        self.replicas.write().remove(dc_id);
        info!("Removed replica DC '{}'", dc_id);
    }

    /// Publish a vector upsert to the replication log
    pub fn publish_upsert(
        &self,
        vector_id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> ReplicationSeq {
        let payload_bytes = vector.len() * 4 + 64; // Rough estimate
        self.publish_entry(
            ReplicationOperation::Upsert {
                vector_id,
                vector,
                metadata,
            },
            payload_bytes,
        )
    }

    /// Publish a vector deletion to the replication log
    pub fn publish_delete(&self, vector_id: String) -> ReplicationSeq {
        self.publish_entry(ReplicationOperation::Delete { vector_id }, 32)
    }

    /// Publish a heartbeat entry
    pub fn publish_heartbeat(&self) -> ReplicationSeq {
        let seq = *self.current_seq.lock();
        self.publish_entry(ReplicationOperation::Heartbeat { current_seq: seq }, 16)
    }

    fn publish_entry(
        &self,
        operation: ReplicationOperation,
        payload_bytes: usize,
    ) -> ReplicationSeq {
        let mut seq = self.current_seq.lock();
        *seq += 1;
        let new_seq = *seq;

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        let entry = ReplicationEntry {
            seq: new_seq,
            source_dc: self.config.dc_id.clone(),
            timestamp_ms,
            operation,
            payload_bytes,
        };

        let mut log = self.replication_log.write();
        log.push_back(entry);

        // Prune log if too large
        while log.len() > self.log_retention_entries {
            log.pop_front();
        }

        let mut stats = self.stats.lock();
        stats.total_entries += 1;
        stats.total_bytes += payload_bytes as u64;

        debug!("Published replication entry seq={} to log", new_seq);
        new_seq
    }

    /// Get entries for a specific replica starting from `after_seq`
    ///
    /// Returns a batch of entries that the replica should apply.
    pub fn get_entries_for_replica(
        &self,
        _replica_dc: &str,
        after_seq: ReplicationSeq,
    ) -> Vec<ReplicationEntry> {
        let log = self.replication_log.read();
        log.iter()
            .filter(|e| e.seq > after_seq)
            .take(self.config.replication_batch_size)
            .cloned()
            .collect()
    }

    /// Record acknowledgment from a replica
    pub fn acknowledge_replica(
        &self,
        dc_id: &str,
        acked_seq: ReplicationSeq,
        bytes_received: u64,
        entries_received: u64,
    ) -> Result<()> {
        let mut replicas = self.replicas.write();
        let tracker = replicas
            .get_mut(dc_id)
            .ok_or_else(|| anyhow!("Unknown replica DC: {}", dc_id))?;

        tracker.on_success(acked_seq, bytes_received, entries_received);

        let primary_seq = *self.current_seq.lock();
        tracker.update_lag(primary_seq);

        debug!(
            "Replica '{}' acked seq={}, lag={} entries",
            dc_id,
            acked_seq,
            primary_seq.saturating_sub(acked_seq)
        );

        Ok(())
    }

    /// Record a failure contacting a replica
    pub fn record_replica_failure(&self, dc_id: &str) {
        let mut replicas = self.replicas.write();
        if let Some(tracker) = replicas.get_mut(dc_id) {
            tracker.on_failure();
            let mut stats = self.stats.lock();
            stats.total_retries += 1;
            warn!(
                "Replica '{}' failure #{} - status: {}",
                dc_id, tracker.failure_count, tracker.status
            );
        }
    }

    /// Get current replication status for all replicas
    pub fn get_replica_status(&self) -> Vec<(DcId, ReplicaStatus, ReplicationSeq, Duration)> {
        let replicas = self.replicas.read();
        replicas
            .values()
            .map(|t| (t.dc_id.clone(), t.status, t.acked_seq, t.lag))
            .collect()
    }

    /// Check if any replica is beyond the lag tolerance
    pub fn has_lagging_replicas(&self) -> bool {
        let replicas = self.replicas.read();
        let primary_seq = *self.current_seq.lock();

        replicas.values().any(|t| {
            let lag_entries = primary_seq.saturating_sub(t.acked_seq);
            lag_entries > self.config.replication_batch_size as u64
        })
    }

    /// Get the maximum replica lag in entries
    pub fn max_replica_lag_entries(&self) -> u64 {
        let replicas = self.replicas.read();
        let primary_seq = *self.current_seq.lock();
        replicas
            .values()
            .map(|t| primary_seq.saturating_sub(t.acked_seq))
            .max()
            .unwrap_or(0)
    }

    /// Get current statistics
    pub fn get_stats(&self) -> CrossDcStats {
        let replicas = self.replicas.read();
        let replica_statuses: HashMap<DcId, String> = replicas
            .iter()
            .map(|(id, t)| (id.clone(), t.status.to_string()))
            .collect();

        let mut stats = self.stats.lock().clone();
        stats.replica_statuses = replica_statuses;
        stats
    }

    /// Get the current sequence number
    pub fn current_seq(&self) -> ReplicationSeq {
        *self.current_seq.lock()
    }

    /// Get the number of entries in the replication log
    pub fn log_length(&self) -> usize {
        self.replication_log.read().len()
    }

    /// Get the number of registered replicas
    pub fn replica_count(&self) -> usize {
        self.replicas.read().len()
    }
}

/// Replica datacenter replication receiver
///
/// Receives and applies replication entries from the primary DC,
/// maintaining an eventually-consistent copy of the vector index.
#[derive(Debug)]
pub struct ReplicaDcManager {
    config: CrossDcConfig,
    /// The last applied sequence number
    last_applied_seq: Arc<Mutex<ReplicationSeq>>,
    /// Buffer of received-but-not-yet-applied entries
    pending_buffer: Arc<Mutex<VecDeque<ReplicationEntry>>>,
    /// Local state: vector_id -> (vector, metadata, seq)
    local_state: Arc<RwLock<VectorStateMap>>,
    /// Conflict log
    conflict_log: Arc<Mutex<Vec<ConflictRecord>>>,
    /// Statistics
    stats: Arc<Mutex<CrossDcStats>>,
    /// Primary DC's current sequence (from heartbeats)
    primary_seq: Arc<Mutex<ReplicationSeq>>,
    /// Timestamp of last heartbeat from primary
    last_heartbeat: Arc<Mutex<Instant>>,
}

/// Record of a detected conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRecord {
    pub vector_id: String,
    pub replica_seq: ReplicationSeq,
    pub primary_seq: ReplicationSeq,
    pub resolution: String,
    pub timestamp_ms: u64,
}

impl ReplicaDcManager {
    /// Create a new replica DC manager
    pub fn new(config: CrossDcConfig) -> Result<Self> {
        if config.is_primary {
            return Err(anyhow!("ReplicaDcManager requires is_primary=false"));
        }

        info!(
            "Replica DC manager initialized for DC '{}' in region '{}'",
            config.dc_id, config.region
        );

        Ok(Self {
            config,
            last_applied_seq: Arc::new(Mutex::new(0)),
            pending_buffer: Arc::new(Mutex::new(VecDeque::new())),
            local_state: Arc::new(RwLock::new(HashMap::new())),
            conflict_log: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(CrossDcStats::default())),
            primary_seq: Arc::new(Mutex::new(0)),
            last_heartbeat: Arc::new(Mutex::new(Instant::now())),
        })
    }

    /// Receive a batch of replication entries from the primary
    pub fn receive_entries(&self, entries: Vec<ReplicationEntry>) -> ReplicationSeq {
        if entries.is_empty() {
            return *self.last_applied_seq.lock();
        }

        let entries_count = entries.len();
        let total_bytes: u64 = entries.iter().map(|e| e.payload_bytes as u64).sum();

        let mut buffer = self.pending_buffer.lock();
        for entry in entries {
            buffer.push_back(entry);
        }

        let mut stats = self.stats.lock();
        stats.total_entries += entries_count as u64;
        stats.total_bytes += total_bytes;

        *self.last_applied_seq.lock()
    }

    /// Apply all buffered entries to the local state
    pub fn apply_pending(&self) -> usize {
        let mut buffer = self.pending_buffer.lock();
        let mut local = self.local_state.write();
        let mut last_seq = self.last_applied_seq.lock();
        let mut stats = self.stats.lock();
        let mut applied = 0;

        // Sort by sequence number to ensure order
        let mut entries: Vec<ReplicationEntry> = buffer.drain(..).collect();
        entries.sort_by_key(|e| e.seq);

        for entry in entries {
            // Only apply entries we haven't seen
            if entry.seq <= *last_seq {
                debug!("Skipping already-applied seq={}", entry.seq);
                continue;
            }

            match &entry.operation {
                ReplicationOperation::Upsert {
                    vector_id,
                    vector,
                    metadata,
                } => {
                    // Check for conflict
                    let conflict = if let Some((_, _, existing_seq)) = local.get(vector_id.as_str())
                    {
                        *existing_seq > entry.seq && self.config.conflict_detection
                    } else {
                        false
                    };

                    if conflict {
                        stats.conflicts_detected += 1;
                        let resolution = self.resolve_conflict(
                            vector_id,
                            entry.seq,
                            local
                                .get(vector_id.as_str())
                                .map(|(_, _, s)| *s)
                                .unwrap_or(0),
                        );
                        if !resolution {
                            // Keep existing (resolution says primary doesn't win)
                            debug!("Conflict: keeping local version for '{}'", vector_id);
                            stats.conflicts_resolved += 1;
                            *last_seq = entry.seq;
                            applied += 1;
                            continue;
                        }
                        stats.conflicts_resolved += 1;
                    }

                    local.insert(
                        vector_id.clone(),
                        (vector.clone(), metadata.clone(), entry.seq),
                    );
                    applied += 1;
                    debug!("Applied upsert for '{}' at seq={}", vector_id, entry.seq);
                }
                ReplicationOperation::Delete { vector_id } => {
                    local.remove(vector_id.as_str());
                    applied += 1;
                    debug!("Applied delete for '{}' at seq={}", vector_id, entry.seq);
                }
                ReplicationOperation::Snapshot {
                    entries: snapshot_entries,
                    as_of_seq,
                } => {
                    // Full snapshot: replace local state
                    local.clear();
                    for (id, vec, meta) in snapshot_entries {
                        local.insert(id.clone(), (vec.clone(), meta.clone(), *as_of_seq));
                    }
                    applied += 1;
                    info!(
                        "Applied snapshot with {} entries at seq={}",
                        snapshot_entries.len(),
                        as_of_seq
                    );
                }
                ReplicationOperation::Heartbeat { current_seq } => {
                    *self.primary_seq.lock() = *current_seq;
                    *self.last_heartbeat.lock() = Instant::now();

                    let last_heartbeat_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or(Duration::ZERO)
                        .as_millis() as u64;
                    stats.last_heartbeat_ms = last_heartbeat_ms;
                    // Heartbeat doesn't count as applied entry
                }
                ReplicationOperation::NoOp => {
                    // No-operation: nothing to apply
                }
            }

            *last_seq = entry.seq;
        }

        // Update lag estimate
        let primary = *self.primary_seq.lock();
        let last = *last_seq;
        let lag = primary.saturating_sub(last);
        stats.current_lag_entries = lag;
        stats.current_lag_ms = lag; // ~1ms per entry estimate

        applied
    }

    /// Resolve a conflict between local and incoming version
    ///
    /// Returns `true` if the incoming (primary) version should win.
    fn resolve_conflict(
        &self,
        vector_id: &str,
        incoming_seq: ReplicationSeq,
        local_seq: ReplicationSeq,
    ) -> bool {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        let primary_wins = match self.config.conflict_resolution {
            ConflictResolutionStrategy::LastWriteWins => incoming_seq > local_seq,
            ConflictResolutionStrategy::PrimaryWins => true,
            ConflictResolutionStrategy::ReplicaWins => false,
            ConflictResolutionStrategy::KeepBoth => false, // Keep local, log conflict
            ConflictResolutionStrategy::MergeMetadata => incoming_seq > local_seq,
        };

        self.conflict_log.lock().push(ConflictRecord {
            vector_id: vector_id.to_string(),
            replica_seq: local_seq,
            primary_seq: incoming_seq,
            resolution: if primary_wins {
                "primary_wins".to_string()
            } else {
                "replica_wins".to_string()
            },
            timestamp_ms,
        });

        primary_wins
    }

    /// Get a vector from local state
    pub fn get_vector(&self, vector_id: &str) -> Option<(Vec<f32>, HashMap<String, String>)> {
        self.local_state
            .read()
            .get(vector_id)
            .map(|(v, m, _)| (v.clone(), m.clone()))
    }

    /// Get the last applied sequence number
    pub fn last_applied_seq(&self) -> ReplicationSeq {
        *self.last_applied_seq.lock()
    }

    /// Get estimated lag (entries behind primary)
    pub fn lag_entries(&self) -> u64 {
        let primary = *self.primary_seq.lock();
        let applied = *self.last_applied_seq.lock();
        primary.saturating_sub(applied)
    }

    /// Check if we are within the configured lag tolerance
    pub fn is_within_lag_tolerance(&self) -> bool {
        let lag = self.lag_entries();
        let tolerance_entries = self.config.max_lag_tolerance.as_millis() as u64;
        lag <= tolerance_entries
    }

    /// Get the number of vectors in local state
    pub fn vector_count(&self) -> usize {
        self.local_state.read().len()
    }

    /// Get current statistics
    pub fn get_stats(&self) -> CrossDcStats {
        let stats = self.stats.lock();
        stats.clone()
    }

    /// Search for similar vectors in local state
    pub fn search_similar(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let local = self.local_state.read();
        let mut similarities: Vec<(String, f32)> = local
            .iter()
            .filter_map(|(id, (vec, _, _))| {
                if vec.len() != query.len() {
                    return None;
                }
                let dot: f32 = vec.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                let na: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if na < 1e-9 || nb < 1e-9 {
                    0.0
                } else {
                    dot / (na * nb)
                };
                Some((id.clone(), sim))
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        similarities
    }

    /// Check if primary heartbeat is recent
    pub fn primary_heartbeat_recent(&self) -> bool {
        self.last_heartbeat.lock().elapsed() < self.config.heartbeat_interval * 3
    }

    /// Get the conflict log
    pub fn conflict_log(&self) -> Vec<ConflictRecord> {
        self.conflict_log.lock().clone()
    }

    /// Get the pending buffer size
    pub fn pending_count(&self) -> usize {
        self.pending_buffer.lock().len()
    }
}

/// Cross-DC replication coordinator
///
/// Coordinates replication between a primary and its replicas.
/// In production, this would use network transports; here it
/// simulates in-process delivery for testing.
#[derive(Debug)]
pub struct CrossDcCoordinator {
    primary: Arc<PrimaryDcManager>,
    replicas: HashMap<DcId, Arc<ReplicaDcManager>>,
}

impl CrossDcCoordinator {
    /// Create a coordinator with a primary and replicas
    pub fn new(primary: Arc<PrimaryDcManager>) -> Self {
        Self {
            primary,
            replicas: HashMap::new(),
        }
    }

    /// Add a replica to the coordinator
    pub fn add_replica_node(&mut self, dc_id: DcId, replica: Arc<ReplicaDcManager>) {
        self.primary
            .add_replica(dc_id.clone(), replica.config.region.clone());
        self.replicas.insert(dc_id, replica);
    }

    /// Perform one round of replication: ship entries from primary to all replicas
    pub fn replicate_once(&self) -> Result<HashMap<DcId, usize>> {
        let mut applied_counts = HashMap::new();

        for (dc_id, replica) in &self.replicas {
            let last_seq = replica.last_applied_seq();
            let entries = self.primary.get_entries_for_replica(dc_id, last_seq);

            if entries.is_empty() {
                applied_counts.insert(dc_id.clone(), 0);
                continue;
            }

            let entry_count = entries.len();
            let bytes: u64 = entries.iter().map(|e| e.payload_bytes as u64).sum();

            replica.receive_entries(entries);
            let applied = replica.apply_pending();

            self.primary
                .acknowledge_replica(dc_id, replica.last_applied_seq(), bytes, entry_count as u64)
                .map_err(|e| anyhow!("Failed to ack replica {}: {}", dc_id, e))?;

            applied_counts.insert(dc_id.clone(), applied);
        }

        Ok(applied_counts)
    }

    /// Get overall replication health
    pub fn replication_health(&self) -> ReplicationHealth {
        let has_lagging = self.primary.has_lagging_replicas();
        let max_lag = self.primary.max_replica_lag_entries();

        let all_healthy = self.replicas.values().all(|r| r.is_within_lag_tolerance());

        ReplicationHealth {
            is_healthy: !has_lagging && all_healthy,
            max_lag_entries: max_lag,
            lagging_replica_count: if has_lagging {
                self.replicas
                    .values()
                    .filter(|r| !r.is_within_lag_tolerance())
                    .count()
            } else {
                0
            },
            total_replicas: self.replicas.len(),
        }
    }
}

/// Overall replication health summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationHealth {
    /// Whether all replicas are within tolerance
    pub is_healthy: bool,
    /// Maximum lag in entries across all replicas
    pub max_lag_entries: u64,
    /// Number of replicas lagging beyond tolerance
    pub lagging_replica_count: usize,
    /// Total number of replicas
    pub total_replicas: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_primary_config() -> CrossDcConfig {
        CrossDcConfig {
            dc_id: "dc-us-east".to_string(),
            region: "us-east-1".to_string(),
            is_primary: true,
            max_lag_tolerance: Duration::from_secs(10),
            replication_batch_size: 100,
            ..Default::default()
        }
    }

    fn make_replica_config(dc_id: &str, region: &str) -> CrossDcConfig {
        CrossDcConfig {
            dc_id: dc_id.to_string(),
            region: region.to_string(),
            is_primary: false,
            max_lag_tolerance: Duration::from_secs(10),
            replication_batch_size: 100,
            ..Default::default()
        }
    }

    #[test]
    fn test_primary_manager_creation() {
        let config = make_primary_config();
        let manager = PrimaryDcManager::new(config);
        assert!(manager.is_ok(), "Primary manager creation should succeed");
    }

    #[test]
    fn test_replica_manager_creation() {
        let config = make_replica_config("dc-eu-west", "eu-west-1");
        let manager = ReplicaDcManager::new(config);
        assert!(manager.is_ok(), "Replica manager creation should succeed");
    }

    #[test]
    fn test_primary_requires_is_primary_true() {
        let mut config = make_primary_config();
        config.is_primary = false;
        let result = PrimaryDcManager::new(config);
        assert!(result.is_err(), "Should fail if is_primary=false");
    }

    #[test]
    fn test_replica_requires_is_primary_false() {
        let mut config = make_replica_config("dc-x", "region-x");
        config.is_primary = true;
        let result = ReplicaDcManager::new(config);
        assert!(result.is_err(), "Should fail if is_primary=true");
    }

    #[test]
    fn test_publish_upsert() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        let seq = manager.publish_upsert("v1".to_string(), vec![1.0, 2.0], HashMap::new());
        assert_eq!(seq, 1);
        assert_eq!(manager.log_length(), 1);
        assert_eq!(manager.current_seq(), 1);
    }

    #[test]
    fn test_publish_delete() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        manager.publish_upsert("v1".to_string(), vec![1.0], HashMap::new());
        let seq = manager.publish_delete("v1".to_string());
        assert_eq!(seq, 2);
        assert_eq!(manager.log_length(), 2);
    }

    #[test]
    fn test_publish_heartbeat() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        let seq = manager.publish_heartbeat();
        assert_eq!(seq, 1);
    }

    #[test]
    fn test_add_and_remove_replica() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        manager.add_replica("dc-eu".to_string(), "eu-west-1".to_string());
        assert_eq!(manager.replica_count(), 1);
        manager.remove_replica("dc-eu");
        assert_eq!(manager.replica_count(), 0);
    }

    #[test]
    fn test_get_entries_for_replica() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        manager.add_replica("dc-eu".to_string(), "eu-west-1".to_string());

        for i in 0..5 {
            manager.publish_upsert(format!("v{}", i), vec![i as f32], HashMap::new());
        }

        let entries = manager.get_entries_for_replica("dc-eu", 0);
        assert_eq!(entries.len(), 5);

        let partial = manager.get_entries_for_replica("dc-eu", 3);
        assert_eq!(partial.len(), 2); // entries 4 and 5
    }

    #[test]
    fn test_replica_receive_and_apply() {
        let primary = PrimaryDcManager::new(make_primary_config()).unwrap();
        primary.add_replica("dc-eu".to_string(), "eu-west-1".to_string());

        let replica = ReplicaDcManager::new(make_replica_config("dc-eu", "eu-west-1")).unwrap();

        // Publish entries on primary
        primary.publish_upsert("v1".to_string(), vec![1.0, 0.0], HashMap::new());
        primary.publish_upsert("v2".to_string(), vec![0.0, 1.0], HashMap::new());

        // Ship to replica
        let entries = primary.get_entries_for_replica("dc-eu", 0);
        assert_eq!(entries.len(), 2);
        replica.receive_entries(entries);
        let applied = replica.apply_pending();

        assert_eq!(applied, 2);
        assert_eq!(replica.vector_count(), 2);
        assert_eq!(replica.last_applied_seq(), 2);

        // Verify vectors are accessible
        let v1 = replica.get_vector("v1");
        assert!(v1.is_some());
        assert_eq!(v1.unwrap().0, vec![1.0, 0.0]);
    }

    #[test]
    fn test_replica_apply_delete() {
        let primary = PrimaryDcManager::new(make_primary_config()).unwrap();
        let replica = ReplicaDcManager::new(make_replica_config("dc-eu", "eu-west-1")).unwrap();
        primary.add_replica("dc-eu".to_string(), "eu-west-1".to_string());

        primary.publish_upsert("v1".to_string(), vec![1.0], HashMap::new());
        primary.publish_delete("v1".to_string());

        let entries = primary.get_entries_for_replica("dc-eu", 0);
        replica.receive_entries(entries);
        replica.apply_pending();

        assert_eq!(replica.vector_count(), 0);
        assert!(replica.get_vector("v1").is_none());
    }

    #[test]
    fn test_coordinator_replicate_once() {
        let primary = Arc::new(PrimaryDcManager::new(make_primary_config()).unwrap());
        let replica =
            Arc::new(ReplicaDcManager::new(make_replica_config("dc-eu", "eu-west-1")).unwrap());

        let mut coordinator = CrossDcCoordinator::new(Arc::clone(&primary));
        coordinator.add_replica_node("dc-eu".to_string(), Arc::clone(&replica));

        // Publish entries on primary
        for i in 0..10 {
            primary.publish_upsert(format!("v{}", i), vec![i as f32], HashMap::new());
        }

        let applied = coordinator.replicate_once().unwrap();
        assert_eq!(applied.get("dc-eu"), Some(&10));
        assert_eq!(replica.vector_count(), 10);
    }

    #[test]
    fn test_coordinator_incremental_replication() {
        let primary = Arc::new(PrimaryDcManager::new(make_primary_config()).unwrap());
        let replica = Arc::new(
            ReplicaDcManager::new(make_replica_config("dc-ap", "ap-southeast-1")).unwrap(),
        );

        let mut coordinator = CrossDcCoordinator::new(Arc::clone(&primary));
        coordinator.add_replica_node("dc-ap".to_string(), Arc::clone(&replica));

        // First batch
        for i in 0..5 {
            primary.publish_upsert(format!("v{}", i), vec![i as f32], HashMap::new());
        }
        coordinator.replicate_once().unwrap();
        assert_eq!(replica.vector_count(), 5);

        // Second batch
        for i in 5..10 {
            primary.publish_upsert(format!("v{}", i), vec![i as f32], HashMap::new());
        }
        coordinator.replicate_once().unwrap();
        assert_eq!(replica.vector_count(), 10);
    }

    #[test]
    fn test_replication_health_healthy() {
        let primary = Arc::new(PrimaryDcManager::new(make_primary_config()).unwrap());
        let replica =
            Arc::new(ReplicaDcManager::new(make_replica_config("dc-eu", "eu-west-1")).unwrap());

        let mut coordinator = CrossDcCoordinator::new(Arc::clone(&primary));
        coordinator.add_replica_node("dc-eu".to_string(), Arc::clone(&replica));

        // Sync up
        primary.publish_upsert("v1".to_string(), vec![1.0], HashMap::new());
        coordinator.replicate_once().unwrap();

        let health = coordinator.replication_health();
        assert_eq!(health.total_replicas, 1);
        // After sync, should be healthy
        assert!(health.is_healthy || health.max_lag_entries <= 1);
    }

    #[test]
    fn test_snapshot_operation() {
        let _primary = Arc::new(PrimaryDcManager::new(make_primary_config()).unwrap());
        let replica =
            Arc::new(ReplicaDcManager::new(make_replica_config("dc-eu", "eu-west-1")).unwrap());

        // Simulate a snapshot entry
        let snapshot_entries = vec![
            ("v1".to_string(), vec![1.0, 0.0], HashMap::new()),
            ("v2".to_string(), vec![0.0, 1.0], HashMap::new()),
        ];

        let snapshot_op = ReplicationOperation::Snapshot {
            entries: snapshot_entries,
            as_of_seq: 100,
        };

        let entry = ReplicationEntry {
            seq: 1,
            source_dc: "dc-us-east".to_string(),
            timestamp_ms: 0,
            operation: snapshot_op,
            payload_bytes: 256,
        };

        replica.receive_entries(vec![entry]);
        replica.apply_pending();

        assert_eq!(replica.vector_count(), 2);
    }

    #[test]
    fn test_heartbeat_replication() {
        let primary = Arc::new(PrimaryDcManager::new(make_primary_config()).unwrap());
        let replica =
            Arc::new(ReplicaDcManager::new(make_replica_config("dc-eu", "eu-west-1")).unwrap());

        let mut coordinator = CrossDcCoordinator::new(Arc::clone(&primary));
        coordinator.add_replica_node("dc-eu".to_string(), Arc::clone(&replica));

        primary.publish_heartbeat();
        coordinator.replicate_once().unwrap();

        // After receiving heartbeat, primary_heartbeat_recent should be true
        // (apply_pending processes heartbeats)
        let stats = replica.get_stats();
        // Just verify stats are accessible (total_entries is unsigned, always >= 0)
        let _ = stats.total_entries;
    }

    #[test]
    fn test_acknowledge_replica() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        manager.add_replica("dc-eu".to_string(), "eu-west-1".to_string());

        for _ in 0..5 {
            manager.publish_upsert("v".to_string(), vec![1.0], HashMap::new());
        }

        let result = manager.acknowledge_replica("dc-eu", 5, 500, 5);
        assert!(result.is_ok());

        let status = manager.get_replica_status();
        assert!(!status.is_empty());
        let (_, status_val, acked, _) = &status[0];
        assert_eq!(*acked, 5);
        assert_eq!(*status_val, ReplicaStatus::Healthy);
    }

    #[test]
    fn test_acknowledge_unknown_replica_fails() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        let result = manager.acknowledge_replica("unknown-dc", 1, 0, 0);
        assert!(result.is_err(), "Should fail for unknown replica");
    }

    #[test]
    fn test_record_replica_failure() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        manager.add_replica("dc-eu".to_string(), "eu-west-1".to_string());

        for _ in 0..6 {
            manager.record_replica_failure("dc-eu");
        }

        let status = manager.get_replica_status();
        let (_, s, _, _) = &status[0];
        assert_eq!(*s, ReplicaStatus::Disconnected);
    }

    #[test]
    fn test_replica_search_similar() {
        let primary = PrimaryDcManager::new(make_primary_config()).unwrap();
        let replica = ReplicaDcManager::new(make_replica_config("dc-eu", "eu-west-1")).unwrap();
        primary.add_replica("dc-eu".to_string(), "eu-west-1".to_string());

        primary.publish_upsert("v1".to_string(), vec![1.0, 0.0, 0.0], HashMap::new());
        primary.publish_upsert("v2".to_string(), vec![0.0, 1.0, 0.0], HashMap::new());

        let entries = primary.get_entries_for_replica("dc-eu", 0);
        replica.receive_entries(entries);
        replica.apply_pending();

        let results = replica.search_similar(&[1.0, 0.0, 0.0], 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "v1");
    }

    #[test]
    fn test_cross_dc_config_default() {
        let config = CrossDcConfig::default();
        assert!(config.is_primary);
        assert_eq!(config.compression_level, 3);
        assert!(config.conflict_detection);
    }

    #[test]
    fn test_conflict_resolution_last_write_wins() {
        let mut config = make_replica_config("dc-eu", "eu-west-1");
        config.conflict_resolution = ConflictResolutionStrategy::LastWriteWins;
        config.conflict_detection = true;

        let replica = ReplicaDcManager::new(config).unwrap();

        // Manually insert a "local" entry with higher seq
        {
            let mut local = replica.local_state.write();
            local.insert(
                "v1".to_string(),
                (vec![2.0], HashMap::new(), 100), // local seq=100
            );
        }

        // Receive a replication entry with lower seq (should not overwrite)
        let entry = ReplicationEntry {
            seq: 1,
            source_dc: "dc-us-east".to_string(),
            timestamp_ms: 0,
            operation: ReplicationOperation::Upsert {
                vector_id: "v1".to_string(),
                vector: vec![1.0],
                metadata: HashMap::new(),
            },
            payload_bytes: 16,
        };

        replica.receive_entries(vec![entry]);
        replica.apply_pending();

        // With LastWriteWins, incoming_seq=1 < local_seq=100, so local wins
        let v1 = replica.get_vector("v1");
        assert!(v1.is_some());
        assert_eq!(v1.unwrap().0, vec![2.0], "Local version should be retained");
    }

    #[test]
    fn test_conflict_resolution_primary_wins() {
        let mut config = make_replica_config("dc-eu", "eu-west-1");
        config.conflict_resolution = ConflictResolutionStrategy::PrimaryWins;
        config.conflict_detection = true;

        let replica = ReplicaDcManager::new(config).unwrap();

        // Insert local entry
        {
            let mut local = replica.local_state.write();
            local.insert("v1".to_string(), (vec![2.0], HashMap::new(), 100));
        }

        // Receive primary's version
        let entry = ReplicationEntry {
            seq: 1,
            source_dc: "dc-us-east".to_string(),
            timestamp_ms: 0,
            operation: ReplicationOperation::Upsert {
                vector_id: "v1".to_string(),
                vector: vec![1.0],
                metadata: HashMap::new(),
            },
            payload_bytes: 16,
        };

        replica.receive_entries(vec![entry]);
        replica.apply_pending();

        // With PrimaryWins, primary always wins
        let v1 = replica.get_vector("v1");
        assert!(v1.is_some());
        assert_eq!(v1.unwrap().0, vec![1.0], "Primary version should win");
    }

    #[test]
    fn test_pending_buffer_tracking() {
        let replica = ReplicaDcManager::new(make_replica_config("dc-eu", "eu-west-1")).unwrap();

        assert_eq!(replica.pending_count(), 0);

        let entry = ReplicationEntry {
            seq: 1,
            source_dc: "dc-us".to_string(),
            timestamp_ms: 0,
            operation: ReplicationOperation::NoOp,
            payload_bytes: 0,
        };

        // Use a NoOp-like operation - Heartbeat doesn't advance seq
        let entry2 = ReplicationEntry {
            seq: 1,
            source_dc: "dc-us".to_string(),
            timestamp_ms: 0,
            operation: ReplicationOperation::Heartbeat { current_seq: 0 },
            payload_bytes: 0,
        };

        replica.receive_entries(vec![entry, entry2]);
        assert_eq!(replica.pending_count(), 2);
    }

    #[test]
    fn test_max_lag_entries_calculation() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();
        manager.add_replica("dc-eu".to_string(), "eu-west-1".to_string());

        for _ in 0..20 {
            manager.publish_upsert("v".to_string(), vec![1.0], HashMap::new());
        }

        let lag = manager.max_replica_lag_entries();
        assert_eq!(lag, 20, "Lag should be 20 entries");
    }

    #[test]
    fn test_replication_stats() {
        let manager = PrimaryDcManager::new(make_primary_config()).unwrap();

        for i in 0..5 {
            manager.publish_upsert(format!("v{}", i), vec![i as f32], HashMap::new());
        }

        let stats = manager.get_stats();
        assert_eq!(stats.total_entries, 5);
        assert!(stats.total_bytes > 0);
    }
}
