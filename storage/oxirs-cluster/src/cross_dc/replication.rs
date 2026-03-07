//! Cross-datacenter replication with configurable consistency levels
//!
//! Supports: async, semi-sync, and synchronous replication modes across
//! multiple datacenter regions. Designed to handle 1000+ node deployments.

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// -------------------------------------------------------------------------
// Consistency level
// -------------------------------------------------------------------------

/// Replication consistency level for cross-DC writes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Write returns after local commit only (lowest latency, highest risk)
    LocalAsync,
    /// Write returns after quorum of local replicas acknowledge
    LocalQuorum,
    /// Write returns after at least one remote DC achieves local quorum
    EachQuorum,
    /// Write returns after all DCs achieve quorum (highest durability)
    AllQuorum,
}

impl ConsistencyLevel {
    /// Minimum number of remote DCs that must acknowledge for this level
    pub fn min_remote_acks(&self, total_remote_dcs: usize) -> usize {
        match self {
            ConsistencyLevel::LocalAsync => 0,
            ConsistencyLevel::LocalQuorum => 0,
            ConsistencyLevel::EachQuorum => {
                if total_remote_dcs == 0 {
                    0
                } else {
                    1
                }
            }
            ConsistencyLevel::AllQuorum => total_remote_dcs,
        }
    }
}

// -------------------------------------------------------------------------
// Replication operations and units
// -------------------------------------------------------------------------

/// The type of operation in a replication unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationOp {
    Insert {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    },
    Delete {
        subject: String,
        predicate: String,
        object: String,
        graph: Option<String>,
    },
    BatchInsert {
        count: usize,
    },
    BatchDelete {
        count: usize,
    },
    SchemaChange {
        description: String,
    },
    /// Checkpoint: indicates all prior operations are committed up to `lsn`
    Checkpoint {
        lsn: u64,
    },
}

/// A single unit of data to be replicated across DCs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationUnit {
    /// Unique ID for this unit (UUIDv4 format)
    pub id: String,
    pub shard_id: u64,
    pub sequence_num: u64,
    pub operation: ReplicationOp,
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    pub source_node: String,
    pub source_region: String,
    /// Serialized payload (compressed data)
    pub payload: Vec<u8>,
}

impl ReplicationUnit {
    pub fn new(
        id: impl Into<String>,
        shard_id: u64,
        sequence_num: u64,
        operation: ReplicationOp,
        source_node: impl Into<String>,
        source_region: impl Into<String>,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            id: id.into(),
            shard_id,
            sequence_num,
            operation,
            timestamp_ms: current_timestamp_ms(),
            source_node: source_node.into(),
            source_region: source_region.into(),
            payload,
        }
    }
}

// -------------------------------------------------------------------------
// Replication stream state
// -------------------------------------------------------------------------

/// State of a cross-DC replication stream
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StreamState {
    Active,
    Paused,
    CatchingUp { progress_pct: f64 },
    Error { message: String },
    Disconnected,
}

/// Tracks the replication stream to one remote region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStream {
    pub stream_id: String,
    pub source_region: String,
    pub target_region: String,
    pub consistency: ConsistencyLevel,
    /// Bytes outstanding (not yet acknowledged)
    pub lag_bytes: u64,
    /// Estimated time lag based on throughput
    pub lag_duration: Duration,
    /// Last sequence number confirmed received by the target
    pub last_applied_seq: u64,
    /// Current throughput in bytes per second
    pub throughput_bps: f64,
    pub state: StreamState,
}

impl ReplicationStream {
    fn new(
        source_region: impl Into<String>,
        target_region: impl Into<String>,
        consistency: ConsistencyLevel,
    ) -> Self {
        let target = target_region.into();
        let source = source_region.into();
        let stream_id = format!("{}->{}", source, target);
        Self {
            stream_id,
            source_region: source,
            target_region: target,
            consistency,
            lag_bytes: 0,
            lag_duration: Duration::ZERO,
            last_applied_seq: 0,
            throughput_bps: 0.0,
            state: StreamState::Active,
        }
    }
}

// -------------------------------------------------------------------------
// Per-region queue tracker
// -------------------------------------------------------------------------

struct RegionTracker {
    stream: ReplicationStream,
    /// Highest sequence number acknowledged by the target
    ack_seq: u64,
    /// Bytes in the queue not yet sent for this region (approximate)
    queue_bytes: u64,
    /// Throughput measurement window
    throughput_window: ThroughputWindow,
}

struct ThroughputWindow {
    bytes_sent: u64,
    window_start: Instant,
}

impl ThroughputWindow {
    fn new() -> Self {
        Self {
            bytes_sent: 0,
            window_start: Instant::now(),
        }
    }

    fn record_sent(&mut self, bytes: u64) {
        self.bytes_sent += bytes;
    }

    fn throughput_bps(&self) -> f64 {
        let elapsed = self.window_start.elapsed().as_secs_f64();
        if elapsed < 0.001 {
            return 0.0;
        }
        self.bytes_sent as f64 / elapsed
    }
}

// -------------------------------------------------------------------------
// Cross-DC replication manager
// -------------------------------------------------------------------------

/// Manages cross-datacenter replication streams from the local region.
///
/// Maintains a shared, ordered queue of `ReplicationUnit`s. Per-region
/// cursors track what has been sent and acknowledged.
pub struct CrossDcReplicationManager {
    local_region: String,
    /// Queue of units pending replication (ordered by sequence_num)
    replication_queue: VecDeque<ReplicationUnit>,
    max_queue_size: usize,
    /// Per-region state: region_id → tracker
    trackers: HashMap<String, RegionTracker>,
    /// Per-region cursor: how far into the queue we've delivered
    cursors: HashMap<String, usize>,
    /// The global highest sequence number enqueued so far
    last_enqueued_seq: u64,
}

impl CrossDcReplicationManager {
    /// Create a new manager for the given local region
    pub fn new(local_region: impl Into<String>) -> Self {
        Self {
            local_region: local_region.into(),
            replication_queue: VecDeque::new(),
            max_queue_size: 100_000,
            trackers: HashMap::new(),
            cursors: HashMap::new(),
            last_enqueued_seq: 0,
        }
    }

    /// Configure maximum in-memory queue size
    pub fn with_max_queue_size(mut self, size: usize) -> Self {
        self.max_queue_size = size.max(1);
        self
    }

    // -----------------------------------------------------------------------
    // Target region management
    // -----------------------------------------------------------------------

    /// Register a target region for replication
    pub fn add_target_region(
        &mut self,
        region_id: impl Into<String>,
        consistency: ConsistencyLevel,
    ) -> Result<()> {
        let region_id = region_id.into();
        if region_id == self.local_region {
            return Err(ClusterError::Config(
                "Cannot replicate to the local region".into(),
            ));
        }
        let stream =
            ReplicationStream::new(self.local_region.clone(), region_id.clone(), consistency);
        self.trackers.insert(
            region_id.clone(),
            RegionTracker {
                stream,
                ack_seq: 0,
                queue_bytes: 0,
                throughput_window: ThroughputWindow::new(),
            },
        );
        self.cursors.insert(region_id, 0);
        Ok(())
    }

    /// Remove a target region
    pub fn remove_target_region(&mut self, region_id: &str) -> bool {
        self.cursors.remove(region_id);
        self.trackers.remove(region_id).is_some()
    }

    // -----------------------------------------------------------------------
    // Enqueueing and draining
    // -----------------------------------------------------------------------

    /// Enqueue a replication unit (called on every local write)
    pub fn enqueue(&mut self, unit: ReplicationUnit) -> Result<()> {
        if self.replication_queue.len() >= self.max_queue_size {
            return Err(ClusterError::ResourceLimit(format!(
                "Replication queue full ({} items)",
                self.max_queue_size
            )));
        }
        let seq = unit.sequence_num;
        let bytes = unit.payload.len() as u64;
        self.replication_queue.push_back(unit);
        self.last_enqueued_seq = self.last_enqueued_seq.max(seq);

        // Update lag bytes for all trackers
        for tracker in self.trackers.values_mut() {
            tracker.queue_bytes += bytes;
        }

        self.gc_queue();
        Ok(())
    }

    /// Drain up to `max_items` replication units for a target region.
    ///
    /// Advances the per-region cursor; items already drained are not re-sent.
    pub fn drain_batch(&mut self, target_region: &str, max_items: usize) -> Vec<ReplicationUnit> {
        let cursor = match self.cursors.get_mut(target_region) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let tracker = match self.trackers.get_mut(target_region) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let queue_len = self.replication_queue.len();
        if *cursor >= queue_len {
            return Vec::new();
        }

        let take = max_items.min(queue_len - *cursor);
        let batch: Vec<ReplicationUnit> = self
            .replication_queue
            .iter()
            .skip(*cursor)
            .take(take)
            .cloned()
            .collect();

        let bytes_drained: u64 = batch.iter().map(|u| u.payload.len() as u64).sum();
        *cursor += batch.len();
        tracker.throughput_window.record_sent(bytes_drained);
        tracker.queue_bytes = tracker.queue_bytes.saturating_sub(bytes_drained);
        tracker.stream.throughput_bps = tracker.throughput_window.throughput_bps();

        batch
    }

    /// Acknowledge that the target region has applied all units up to `up_to_seq`
    pub fn acknowledge(&mut self, target_region: &str, up_to_seq: u64) {
        if let Some(tracker) = self.trackers.get_mut(target_region) {
            tracker.ack_seq = tracker.ack_seq.max(up_to_seq);
            tracker.stream.last_applied_seq = tracker.ack_seq;

            // Recalculate lag
            let lag_bytes: u64 = self
                .replication_queue
                .iter()
                .filter(|u| u.sequence_num > tracker.ack_seq)
                .map(|u| u.payload.len() as u64)
                .sum();
            tracker.stream.lag_bytes = lag_bytes;
        }
        self.gc_queue();
    }

    // -----------------------------------------------------------------------
    // Consistency checking
    // -----------------------------------------------------------------------

    /// Check if a write (identified by sequence number) is fully replicated
    /// under the given consistency level.
    pub fn is_replicated(&self, seq_num: u64, consistency: &ConsistencyLevel) -> bool {
        let total_remote = self.trackers.len();
        let required_acks = consistency.min_remote_acks(total_remote);

        if required_acks == 0 {
            return true; // LocalAsync / LocalQuorum with no remote DCs
        }

        let acked_count = self
            .trackers
            .values()
            .filter(|t| t.ack_seq >= seq_num)
            .count();

        acked_count >= required_acks
    }

    // -----------------------------------------------------------------------
    // Status queries
    // -----------------------------------------------------------------------

    pub fn stream_status(&self, target_region: &str) -> Option<&ReplicationStream> {
        self.trackers.get(target_region).map(|t| &t.stream)
    }

    pub fn all_stream_status(&self) -> Vec<&ReplicationStream> {
        self.trackers.values().map(|t| &t.stream).collect()
    }

    /// Total lag bytes across all target regions
    pub fn total_lag_bytes(&self) -> u64 {
        self.trackers.values().map(|t| t.stream.lag_bytes).sum()
    }

    /// Number of items currently in the replication queue
    pub fn queue_len(&self) -> usize {
        self.replication_queue.len()
    }

    /// The highest sequence number enqueued so far
    pub fn last_enqueued_seq(&self) -> u64 {
        self.last_enqueued_seq
    }

    pub fn local_region(&self) -> &str {
        &self.local_region
    }

    pub fn registered_regions(&self) -> Vec<&str> {
        self.trackers.keys().map(|s| s.as_str()).collect()
    }

    // -----------------------------------------------------------------------
    // Pause / resume
    // -----------------------------------------------------------------------

    pub fn pause_stream(&mut self, target_region: &str) -> Result<()> {
        let tracker = self.trackers.get_mut(target_region).ok_or_else(|| {
            ClusterError::Config(format!("Region '{}' not registered", target_region))
        })?;
        tracker.stream.state = StreamState::Paused;
        Ok(())
    }

    pub fn resume_stream(&mut self, target_region: &str) -> Result<()> {
        let tracker = self.trackers.get_mut(target_region).ok_or_else(|| {
            ClusterError::Config(format!("Region '{}' not registered", target_region))
        })?;
        tracker.stream.state = StreamState::Active;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Remove units from the front of the queue that all regions have acknowledged
    fn gc_queue(&mut self) {
        if self.trackers.is_empty() {
            return;
        }

        // Find the minimum ack_seq across all regions (items up to this are safe to drop)
        let min_ack = self.trackers.values().map(|t| t.ack_seq).min().unwrap_or(0);

        // Count units to drop
        let to_drop = self
            .replication_queue
            .iter()
            .take_while(|u| u.sequence_num <= min_ack)
            .count();

        if to_drop > 0 {
            // Adjust cursors before dropping
            for cursor in self.cursors.values_mut() {
                *cursor = cursor.saturating_sub(to_drop);
            }
            for _ in 0..to_drop {
                self.replication_queue.pop_front();
            }
        }
    }
}

// -------------------------------------------------------------------------
// Timestamp helper
// -------------------------------------------------------------------------

fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit(seq: u64, payload_size: usize) -> ReplicationUnit {
        ReplicationUnit::new(
            format!("unit-{}", seq),
            0,
            seq,
            ReplicationOp::Insert {
                subject: "s".into(),
                predicate: "p".into(),
                object: "o".into(),
                graph: None,
            },
            "node-1",
            "us-east-1",
            vec![0u8; payload_size],
        )
    }

    #[test]
    fn test_add_remove_target_region() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalAsync)
            .unwrap();
        assert_eq!(mgr.registered_regions().len(), 1);
        assert!(mgr.remove_target_region("eu-west-1"));
        assert_eq!(mgr.registered_regions().len(), 0);
    }

    #[test]
    fn test_cannot_add_local_region() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        let result = mgr.add_target_region("us-east-1", ConsistencyLevel::LocalAsync);
        assert!(result.is_err());
    }

    #[test]
    fn test_enqueue_and_drain() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalAsync)
            .unwrap();

        for i in 1..=5 {
            mgr.enqueue(make_unit(i, 100)).unwrap();
        }
        assert_eq!(mgr.queue_len(), 5);

        let batch = mgr.drain_batch("eu-west-1", 3);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].sequence_num, 1);
        assert_eq!(batch[2].sequence_num, 3);

        let batch2 = mgr.drain_batch("eu-west-1", 10);
        assert_eq!(batch2.len(), 2);
        assert_eq!(batch2[0].sequence_num, 4);
    }

    #[test]
    fn test_drain_nonexistent_region_returns_empty() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        let batch = mgr.drain_batch("nonexistent", 10);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_acknowledge_updates_state() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalAsync)
            .unwrap();

        for i in 1..=5 {
            mgr.enqueue(make_unit(i, 100)).unwrap();
        }
        let _ = mgr.drain_batch("eu-west-1", 5);

        mgr.acknowledge("eu-west-1", 3);
        let status = mgr.stream_status("eu-west-1").unwrap();
        assert_eq!(status.last_applied_seq, 3);
    }

    #[test]
    fn test_is_replicated_local_async() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalAsync)
            .unwrap();

        mgr.enqueue(make_unit(1, 10)).unwrap();
        // LocalAsync: always replicated (no remote ack needed)
        assert!(mgr.is_replicated(1, &ConsistencyLevel::LocalAsync));
    }

    #[test]
    fn test_is_replicated_each_quorum() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::EachQuorum)
            .unwrap();
        mgr.add_target_region("ap-southeast-1", ConsistencyLevel::EachQuorum)
            .unwrap();

        mgr.enqueue(make_unit(5, 10)).unwrap();

        // Not yet replicated
        assert!(!mgr.is_replicated(5, &ConsistencyLevel::EachQuorum));

        // Ack from one region
        mgr.acknowledge("eu-west-1", 5);
        assert!(mgr.is_replicated(5, &ConsistencyLevel::EachQuorum));
    }

    #[test]
    fn test_is_replicated_all_quorum() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::AllQuorum)
            .unwrap();
        mgr.add_target_region("ap-southeast-1", ConsistencyLevel::AllQuorum)
            .unwrap();

        mgr.enqueue(make_unit(10, 10)).unwrap();

        // Need both to ack
        mgr.acknowledge("eu-west-1", 10);
        assert!(!mgr.is_replicated(10, &ConsistencyLevel::AllQuorum));

        mgr.acknowledge("ap-southeast-1", 10);
        assert!(mgr.is_replicated(10, &ConsistencyLevel::AllQuorum));
    }

    #[test]
    fn test_total_lag_bytes() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalAsync)
            .unwrap();

        for i in 1..=10 {
            mgr.enqueue(make_unit(i, 200)).unwrap();
        }

        // Drain and ack 5 units
        let batch = mgr.drain_batch("eu-west-1", 5);
        assert_eq!(batch.len(), 5);
        mgr.acknowledge("eu-west-1", 5);

        // Remaining 5 units × 200 bytes = 1000 bytes lag
        let lag = mgr.total_lag_bytes();
        assert_eq!(lag, 1000);
    }

    #[test]
    fn test_gc_queue_after_all_ack() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalAsync)
            .unwrap();

        for i in 1..=10 {
            mgr.enqueue(make_unit(i, 50)).unwrap();
        }
        let _ = mgr.drain_batch("eu-west-1", 10);
        mgr.acknowledge("eu-west-1", 10);

        // All acked, queue should be GC'd
        assert_eq!(mgr.queue_len(), 0);
    }

    #[test]
    fn test_queue_max_size_enforced() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1").with_max_queue_size(5);

        for i in 1..=5 {
            mgr.enqueue(make_unit(i, 10)).unwrap();
        }
        // 6th should fail
        let result = mgr.enqueue(make_unit(6, 10));
        assert!(result.is_err());
    }

    #[test]
    fn test_pause_resume_stream() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalAsync)
            .unwrap();

        mgr.pause_stream("eu-west-1").unwrap();
        assert_eq!(
            mgr.stream_status("eu-west-1").unwrap().state,
            StreamState::Paused
        );

        mgr.resume_stream("eu-west-1").unwrap();
        assert_eq!(
            mgr.stream_status("eu-west-1").unwrap().state,
            StreamState::Active
        );
    }

    #[test]
    fn test_multiple_regions_independent_cursors() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalAsync)
            .unwrap();
        mgr.add_target_region("ap-southeast-1", ConsistencyLevel::LocalAsync)
            .unwrap();

        for i in 1..=6 {
            mgr.enqueue(make_unit(i, 100)).unwrap();
        }

        let eu_batch = mgr.drain_batch("eu-west-1", 6);
        let ap_batch = mgr.drain_batch("ap-southeast-1", 3);

        assert_eq!(eu_batch.len(), 6, "EU should drain all 6");
        assert_eq!(ap_batch.len(), 3, "AP should drain only 3");

        // AP can drain more independently
        let ap_batch2 = mgr.drain_batch("ap-southeast-1", 10);
        assert_eq!(ap_batch2.len(), 3, "AP should get remaining 3");
    }

    #[test]
    fn test_high_volume_replication() {
        let mut mgr = CrossDcReplicationManager::new("us-east-1");
        mgr.add_target_region("eu-west-1", ConsistencyLevel::LocalQuorum)
            .unwrap();

        let batch_size = 10_000;
        for i in 1..=batch_size {
            mgr.enqueue(make_unit(i, 64)).unwrap();
        }
        assert_eq!(mgr.queue_len(), batch_size as usize);

        // Drain in chunks
        let mut total_drained = 0;
        loop {
            let batch = mgr.drain_batch("eu-west-1", 1000);
            if batch.is_empty() {
                break;
            }
            total_drained += batch.len();
        }
        assert_eq!(total_drained, batch_size as usize);
    }

    #[test]
    fn test_consistency_level_min_remote_acks() {
        assert_eq!(ConsistencyLevel::LocalAsync.min_remote_acks(3), 0);
        assert_eq!(ConsistencyLevel::LocalQuorum.min_remote_acks(3), 0);
        assert_eq!(ConsistencyLevel::EachQuorum.min_remote_acks(3), 1);
        assert_eq!(ConsistencyLevel::EachQuorum.min_remote_acks(0), 0);
        assert_eq!(ConsistencyLevel::AllQuorum.min_remote_acks(3), 3);
        assert_eq!(ConsistencyLevel::AllQuorum.min_remote_acks(0), 0);
    }
}
