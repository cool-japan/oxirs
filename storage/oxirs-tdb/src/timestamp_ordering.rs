//! # Timestamp Ordering System
//!
//! Advanced timestamp ordering implementation with vector clocks, Lamport timestamps,
//! and clock synchronization for distributed transaction management and consistency.
//!
//! This module provides sophisticated timestamp management capabilities:
//! - Vector clocks for distributed causality tracking
//! - Lamport timestamps for logical ordering
//! - Physical clock synchronization and skew handling
//! - Hybrid logical clocks (HLC) combining physical and logical time
//! - Time zone aware timestamp management

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Node identifier for distributed systems
pub type NodeId = u64;

/// Logical timestamp type
pub type LogicalTime = u64;

/// Physical timestamp type (microseconds since UNIX epoch)
pub type PhysicalTime = u64;

/// Vector clock for distributed causality tracking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorClock {
    /// Clock values for each node
    clocks: BTreeMap<NodeId, LogicalTime>,
    /// Local node ID
    local_node: NodeId,
}

impl VectorClock {
    /// Create a new vector clock for a specific node
    pub fn new(local_node: NodeId) -> Self {
        let mut clocks = BTreeMap::new();
        clocks.insert(local_node, 0);

        Self { clocks, local_node }
    }

    /// Create from existing clock values
    pub fn from_clocks(clocks: BTreeMap<NodeId, LogicalTime>, local_node: NodeId) -> Self {
        Self { clocks, local_node }
    }

    /// Increment the local clock
    pub fn tick(&mut self) -> LogicalTime {
        let current = self.clocks.entry(self.local_node).or_insert(0);
        *current += 1;
        *current
    }

    /// Update clock when receiving a message from another node
    pub fn update(&mut self, other: &VectorClock) -> LogicalTime {
        // Update all clocks to max of local and received
        for (&node, &time) in &other.clocks {
            let current = self.clocks.entry(node).or_insert(0);
            *current = (*current).max(time);
        }

        // Increment local clock
        self.tick()
    }

    /// Compare two vector clocks for causality
    pub fn compare(&self, other: &VectorClock) -> CausalRelation {
        let mut self_greater = false;
        let mut other_greater = false;

        // Get all nodes from both clocks
        let mut all_nodes = std::collections::HashSet::new();
        all_nodes.extend(self.clocks.keys());
        all_nodes.extend(other.clocks.keys());

        for &node in &all_nodes {
            let self_time = self.clocks.get(&node).copied().unwrap_or(0);
            let other_time = other.clocks.get(&node).copied().unwrap_or(0);

            if self_time > other_time {
                self_greater = true;
            } else if other_time > self_time {
                other_greater = true;
            }
        }

        match (self_greater, other_greater) {
            (true, false) => CausalRelation::HappensBefore, // self -> other
            (false, true) => CausalRelation::HappensAfter,  // other -> self
            (false, false) => CausalRelation::Identical,
            (true, true) => CausalRelation::Concurrent,
        }
    }

    /// Check if this clock happens before another
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        matches!(self.compare(other), CausalRelation::HappensBefore)
    }

    /// Check if this clock is concurrent with another
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        matches!(self.compare(other), CausalRelation::Concurrent)
    }

    /// Get the current time for a specific node
    pub fn get_time(&self, node: NodeId) -> LogicalTime {
        self.clocks.get(&node).copied().unwrap_or(0)
    }

    /// Get all clock values
    pub fn clocks(&self) -> &BTreeMap<NodeId, LogicalTime> {
        &self.clocks
    }

    /// Merge with another vector clock (taking max of all values)
    pub fn merge(&mut self, other: &VectorClock) {
        for (&node, &time) in &other.clocks {
            let current = self.clocks.entry(node).or_insert(0);
            *current = (*current).max(time);
        }
    }

    /// Create a compact representation for storage
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Restore from compact representation
    pub fn from_compact_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| anyhow!("Failed to deserialize vector clock: {}", e))
    }

    /// Get the maximum time across all nodes
    pub fn max_time(&self) -> LogicalTime {
        self.clocks.values().copied().max().unwrap_or(0)
    }

    /// Get the number of nodes tracked
    pub fn node_count(&self) -> usize {
        self.clocks.len()
    }
}

impl fmt::Display for VectorClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VC[")?;
        let mut first = true;
        for (&node, &time) in &self.clocks {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}:{}", node, time)?;
            first = false;
        }
        write!(f, "]")
    }
}

/// Causal relationship between two vector clocks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalRelation {
    /// First event happens before second
    HappensBefore,
    /// Second event happens before first
    HappensAfter,
    /// Events are concurrent
    Concurrent,
    /// Clocks are identical
    Identical,
}

/// Lamport timestamp for simple logical ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LamportTimestamp {
    /// Logical time value
    pub time: LogicalTime,
    /// Node ID for tie-breaking
    pub node: NodeId,
}

impl LamportTimestamp {
    /// Create a new Lamport timestamp
    pub fn new(time: LogicalTime, node: NodeId) -> Self {
        Self { time, node }
    }

    /// Create initial timestamp for a node
    pub fn initial(node: NodeId) -> Self {
        Self { time: 0, node }
    }

    /// Increment timestamp
    pub fn tick(&mut self) {
        self.time += 1;
    }

    /// Update timestamp when receiving a message
    pub fn update(&mut self, other: LamportTimestamp) {
        self.time = self.time.max(other.time) + 1;
    }

    /// Compare with another timestamp
    pub fn compare(&self, other: &LamportTimestamp) -> std::cmp::Ordering {
        match self.time.cmp(&other.time) {
            std::cmp::Ordering::Equal => self.node.cmp(&other.node),
            other => other,
        }
    }
}

impl fmt::Display for LamportTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L({}:{})", self.time, self.node)
    }
}

/// Hybrid Logical Clock combining physical and logical time
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridLogicalClock {
    /// Physical time component (microseconds since UNIX epoch)
    pub physical_time: PhysicalTime,
    /// Logical time component  
    pub logical_time: LogicalTime,
    /// Node ID for tie-breaking
    pub node: NodeId,
}

impl HybridLogicalClock {
    /// Create a new HLC with current physical time
    pub fn new(node: NodeId) -> Self {
        Self {
            physical_time: Self::current_physical_time(),
            logical_time: 0,
            node,
        }
    }

    /// Create HLC with specific time
    pub fn with_time(physical_time: PhysicalTime, logical_time: LogicalTime, node: NodeId) -> Self {
        Self {
            physical_time,
            logical_time,
            node,
        }
    }

    /// Get current physical time in microseconds
    pub fn current_physical_time() -> PhysicalTime {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as PhysicalTime
    }

    /// Advance the clock (local event)
    pub fn tick(&mut self) {
        let current_physical = Self::current_physical_time();

        if current_physical > self.physical_time {
            self.physical_time = current_physical;
            self.logical_time = 0;
        } else {
            self.logical_time += 1;
        }
    }

    /// Update clock when receiving a message
    pub fn update(&mut self, other: &HybridLogicalClock) {
        let current_physical = Self::current_physical_time();
        let max_physical = current_physical.max(other.physical_time);

        if max_physical > self.physical_time {
            self.physical_time = max_physical;
            if max_physical == other.physical_time {
                self.logical_time = other.logical_time + 1;
            } else {
                self.logical_time = 0;
            }
        } else if max_physical == self.physical_time {
            self.logical_time = self.logical_time.max(other.logical_time) + 1;
        } else {
            self.logical_time += 1;
        }
    }

    /// Convert to total ordering value
    pub fn to_ordering_value(&self) -> (PhysicalTime, LogicalTime, NodeId) {
        (self.physical_time, self.logical_time, self.node)
    }

    /// Check if this HLC happens before another
    pub fn happens_before(&self, other: &HybridLogicalClock) -> bool {
        self.to_ordering_value() < other.to_ordering_value()
    }

    /// Get age of this timestamp in microseconds
    pub fn age_micros(&self) -> PhysicalTime {
        Self::current_physical_time().saturating_sub(self.physical_time)
    }

    /// Check if timestamp is within clock skew bounds
    pub fn is_within_skew_bounds(&self, max_skew_micros: PhysicalTime) -> bool {
        let current = Self::current_physical_time();
        let diff = if self.physical_time > current {
            self.physical_time - current
        } else {
            current - self.physical_time
        };
        diff <= max_skew_micros
    }
}

impl PartialOrd for HybridLogicalClock {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HybridLogicalClock {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_ordering_value().cmp(&other.to_ordering_value())
    }
}

impl fmt::Display for HybridLogicalClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HLC({}:{}:{})",
            self.physical_time, self.logical_time, self.node
        )
    }
}

/// Clock synchronization manager for distributed systems
pub struct ClockSyncManager {
    /// Local node ID
    local_node: NodeId,
    /// Clock skew estimates for other nodes
    skew_estimates: Arc<RwLock<HashMap<NodeId, ClockSkewInfo>>>,
    /// Maximum allowed clock skew in microseconds
    max_skew_micros: PhysicalTime,
    /// Synchronization interval
    sync_interval: Duration,
    /// Last synchronization time
    last_sync: Arc<Mutex<Instant>>,
}

#[derive(Debug, Clone)]
struct ClockSkewInfo {
    /// Estimated offset from local clock (positive = remote is ahead)
    offset_micros: i64,
    /// Round-trip time to this node
    rtt_micros: PhysicalTime,
    /// Confidence in this estimate (0.0 to 1.0)
    confidence: f64,
    /// Last update time
    updated_at: Instant,
}

impl ClockSyncManager {
    /// Create a new clock synchronization manager
    pub fn new(local_node: NodeId, max_skew_micros: PhysicalTime) -> Self {
        Self {
            local_node,
            skew_estimates: Arc::new(RwLock::new(HashMap::new())),
            max_skew_micros,
            sync_interval: Duration::from_secs(30), // 30 seconds
            last_sync: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Record a clock synchronization measurement
    pub fn record_sync_measurement(
        &self,
        remote_node: NodeId,
        local_send_time: PhysicalTime,
        remote_time: PhysicalTime,
        local_receive_time: PhysicalTime,
    ) {
        let rtt = local_receive_time.saturating_sub(local_send_time);
        let estimated_remote_recv_time = local_send_time + rtt / 2;
        let offset = remote_time as i64 - estimated_remote_recv_time as i64;

        let confidence = if rtt < 1000 { 0.9 } else { 0.5 }; // High confidence for low RTT

        let skew_info = ClockSkewInfo {
            offset_micros: offset,
            rtt_micros: rtt,
            confidence,
            updated_at: Instant::now(),
        };

        let mut skews = self.skew_estimates.write().unwrap();
        skews.insert(remote_node, skew_info);
    }

    /// Get estimated time at a remote node
    pub fn estimate_remote_time(&self, remote_node: NodeId) -> Option<PhysicalTime> {
        let skews = self.skew_estimates.read().unwrap();
        if let Some(skew_info) = skews.get(&remote_node) {
            let local_time = HybridLogicalClock::current_physical_time();
            let estimated_remote = (local_time as i64 + skew_info.offset_micros) as PhysicalTime;
            Some(estimated_remote)
        } else {
            None
        }
    }

    /// Check if a timestamp is within acceptable bounds considering clock skew
    pub fn is_timestamp_valid(&self, timestamp: PhysicalTime, from_node: NodeId) -> bool {
        let local_time = HybridLogicalClock::current_physical_time();

        if from_node == self.local_node {
            // Local timestamp - check against current time with small tolerance
            let diff = if timestamp > local_time {
                timestamp - local_time
            } else {
                local_time - timestamp
            };
            return diff <= self.max_skew_micros;
        }

        // Remote timestamp - consider estimated skew
        if let Some(estimated_remote_time) = self.estimate_remote_time(from_node) {
            let diff = if timestamp > estimated_remote_time {
                timestamp - estimated_remote_time
            } else {
                estimated_remote_time - timestamp
            };
            diff <= self.max_skew_micros * 2 // More tolerance for remote timestamps
        } else {
            // No skew information - use conservative bounds
            let diff = if timestamp > local_time {
                timestamp - local_time
            } else {
                local_time - timestamp
            };
            diff <= self.max_skew_micros * 3
        }
    }

    /// Get clock skew statistics
    pub fn get_skew_stats(&self) -> HashMap<NodeId, (i64, PhysicalTime, f64)> {
        let skews = self.skew_estimates.read().unwrap();
        skews
            .iter()
            .map(|(&node, info)| (node, (info.offset_micros, info.rtt_micros, info.confidence)))
            .collect()
    }

    /// Clean up old skew estimates
    pub fn cleanup_old_estimates(&self, max_age: Duration) {
        let mut skews = self.skew_estimates.write().unwrap();
        let cutoff = Instant::now() - max_age;
        skews.retain(|_, info| info.updated_at > cutoff);
    }
}

/// Timestamp manager combining all timestamp types
pub struct TimestampManager {
    /// Local node ID
    local_node: NodeId,
    /// Vector clock for causality tracking
    vector_clock: Arc<RwLock<VectorClock>>,
    /// Lamport timestamp for simple ordering
    lamport_timestamp: Arc<RwLock<LamportTimestamp>>,
    /// Hybrid logical clock
    hlc: Arc<RwLock<HybridLogicalClock>>,
    /// Clock synchronization manager
    sync_manager: ClockSyncManager,
    /// Statistics
    stats: Arc<RwLock<TimestampStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct TimestampStats {
    pub vector_clock_updates: u64,
    pub lamport_updates: u64,
    pub hlc_updates: u64,
    pub sync_measurements: u64,
    pub invalid_timestamps: u64,
    pub clock_skew_violations: u64,
}

impl TimestampManager {
    /// Create a new timestamp manager
    pub fn new(local_node: NodeId) -> Self {
        Self {
            local_node,
            vector_clock: Arc::new(RwLock::new(VectorClock::new(local_node))),
            lamport_timestamp: Arc::new(RwLock::new(LamportTimestamp::initial(local_node))),
            hlc: Arc::new(RwLock::new(HybridLogicalClock::new(local_node))),
            sync_manager: ClockSyncManager::new(local_node, 1_000_000), // 1 second max skew
            stats: Arc::new(RwLock::new(TimestampStats::default())),
        }
    }

    /// Generate a new timestamp for a local event
    pub fn tick(&self) -> TimestampBundle {
        let vc = {
            let mut vc = self.vector_clock.write().unwrap();
            vc.tick();
            vc.clone()
        };

        let lamport = {
            let mut lamport = self.lamport_timestamp.write().unwrap();
            lamport.tick();
            *lamport
        };

        let hlc = {
            let mut hlc = self.hlc.write().unwrap();
            hlc.tick();
            *hlc
        };

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.vector_clock_updates += 1;
            stats.lamport_updates += 1;
            stats.hlc_updates += 1;
        }

        TimestampBundle {
            vector_clock: vc,
            lamport_timestamp: lamport,
            hybrid_logical_clock: hlc,
        }
    }

    /// Update timestamps when receiving a message
    pub fn update_from_remote(&self, bundle: &TimestampBundle) -> TimestampBundle {
        let vc = {
            let mut vc = self.vector_clock.write().unwrap();
            vc.update(&bundle.vector_clock);
            vc.clone()
        };

        let lamport = {
            let mut lamport = self.lamport_timestamp.write().unwrap();
            lamport.update(bundle.lamport_timestamp);
            *lamport
        };

        let hlc = {
            let mut hlc = self.hlc.write().unwrap();
            hlc.update(&bundle.hybrid_logical_clock);
            *hlc
        };

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.vector_clock_updates += 1;
            stats.lamport_updates += 1;
            stats.hlc_updates += 1;
        }

        TimestampBundle {
            vector_clock: vc,
            lamport_timestamp: lamport,
            hybrid_logical_clock: hlc,
        }
    }

    /// Validate a timestamp bundle
    pub fn validate_timestamp_bundle(&self, bundle: &TimestampBundle) -> Result<()> {
        // Check HLC for clock skew violations
        if !bundle.hybrid_logical_clock.is_within_skew_bounds(1_000_000) {
            let mut stats = self.stats.write().unwrap();
            stats.clock_skew_violations += 1;
            return Err(anyhow!("HLC timestamp violates clock skew bounds"));
        }

        // Validate with sync manager
        if !self.sync_manager.is_timestamp_valid(
            bundle.hybrid_logical_clock.physical_time,
            bundle.hybrid_logical_clock.node,
        ) {
            let mut stats = self.stats.write().unwrap();
            stats.invalid_timestamps += 1;
            return Err(anyhow!("Timestamp validation failed"));
        }

        Ok(())
    }

    /// Get current timestamp bundle
    pub fn current_timestamp_bundle(&self) -> TimestampBundle {
        let vc = self.vector_clock.read().unwrap().clone();
        let lamport = *self.lamport_timestamp.read().unwrap();
        let hlc = *self.hlc.read().unwrap();

        TimestampBundle {
            vector_clock: vc,
            lamport_timestamp: lamport,
            hybrid_logical_clock: hlc,
        }
    }

    /// Record clock synchronization measurement
    pub fn record_sync_measurement(
        &self,
        remote_node: NodeId,
        local_send_time: PhysicalTime,
        remote_time: PhysicalTime,
        local_receive_time: PhysicalTime,
    ) {
        self.sync_manager.record_sync_measurement(
            remote_node,
            local_send_time,
            remote_time,
            local_receive_time,
        );

        let mut stats = self.stats.write().unwrap();
        stats.sync_measurements += 1;
    }

    /// Get statistics
    pub fn get_stats(&self) -> TimestampStats {
        self.stats.read().unwrap().clone()
    }

    /// Get clock synchronization info
    pub fn get_sync_info(&self) -> HashMap<NodeId, (i64, PhysicalTime, f64)> {
        self.sync_manager.get_skew_stats()
    }
}

/// Bundle containing all timestamp types for consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampBundle {
    pub vector_clock: VectorClock,
    pub lamport_timestamp: LamportTimestamp,
    pub hybrid_logical_clock: HybridLogicalClock,
}

impl TimestampBundle {
    /// Create a new timestamp bundle
    pub fn new(
        vector_clock: VectorClock,
        lamport_timestamp: LamportTimestamp,
        hybrid_logical_clock: HybridLogicalClock,
    ) -> Self {
        Self {
            vector_clock,
            lamport_timestamp,
            hybrid_logical_clock,
        }
    }

    /// Get the primary ordering timestamp (HLC)
    pub fn primary_timestamp(&self) -> HybridLogicalClock {
        self.hybrid_logical_clock
    }

    /// Compare bundles for causality using vector clocks
    pub fn causal_relation(&self, other: &TimestampBundle) -> CausalRelation {
        self.vector_clock.compare(&other.vector_clock)
    }

    /// Check if this bundle happens before another
    pub fn happens_before(&self, other: &TimestampBundle) -> bool {
        self.hybrid_logical_clock
            .happens_before(&other.hybrid_logical_clock)
    }

    /// Serialize bundle for storage
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| anyhow!("Failed to serialize timestamp bundle: {}", e))
    }

    /// Deserialize bundle from storage
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| anyhow!("Failed to deserialize timestamp bundle: {}", e))
    }
}

impl fmt::Display for TimestampBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bundle[{}, {}, {}]",
            self.vector_clock, self.lamport_timestamp, self.hybrid_logical_clock
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_vector_clock_causality() {
        let mut vc1 = VectorClock::new(1);
        let mut vc2 = VectorClock::new(2);

        // Initial state - concurrent
        assert_eq!(vc1.compare(&vc2), CausalRelation::Concurrent);

        // vc1 advances
        vc1.tick();
        assert_eq!(vc1.compare(&vc2), CausalRelation::HappensBefore);
        assert_eq!(vc2.compare(&vc1), CausalRelation::HappensAfter);

        // vc2 learns about vc1 and advances
        vc2.update(&vc1);
        assert_eq!(vc1.compare(&vc2), CausalRelation::HappensAfter);
    }

    #[test]
    fn test_lamport_timestamps() {
        let mut l1 = LamportTimestamp::initial(1);
        let mut l2 = LamportTimestamp::initial(2);

        l1.tick();
        assert_eq!(l1.time, 1);

        l2.update(l1);
        assert_eq!(l2.time, 2);

        l1.update(l2);
        assert_eq!(l1.time, 3);
    }

    #[test]
    fn test_hybrid_logical_clock() {
        let mut hlc1 = HybridLogicalClock::new(1);
        let mut hlc2 = HybridLogicalClock::new(2);

        let initial_time = hlc1.physical_time;

        hlc1.tick();
        assert!(hlc1.physical_time >= initial_time);

        hlc2.update(&hlc1);
        assert!(hlc2.physical_time >= hlc1.physical_time);
    }

    #[test]
    fn test_timestamp_manager() {
        let manager = TimestampManager::new(1);

        let bundle1 = manager.tick();
        let bundle2 = manager.tick();

        assert!(bundle2.happens_before(&bundle1) == false);
        assert!(bundle1.happens_before(&bundle2));
    }

    #[test]
    fn test_clock_synchronization() {
        let sync_manager = ClockSyncManager::new(1, 1_000_000);

        let local_send = HybridLogicalClock::current_physical_time();
        thread::sleep(Duration::from_millis(1));
        let remote_time = HybridLogicalClock::current_physical_time();
        thread::sleep(Duration::from_millis(1));
        let local_receive = HybridLogicalClock::current_physical_time();

        sync_manager.record_sync_measurement(2, local_send, remote_time, local_receive);

        let stats = sync_manager.get_skew_stats();
        assert!(stats.contains_key(&2));
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut vc1 = VectorClock::new(1);
        let mut vc2 = VectorClock::new(2);

        vc1.tick(); // vc1 = [1:1, 2:0]
        vc2.tick(); // vc2 = [1:0, 2:1]

        vc1.merge(&vc2); // vc1 = [1:1, 2:1]

        assert_eq!(vc1.get_time(1), 1);
        assert_eq!(vc1.get_time(2), 1);
    }

    #[test]
    fn test_timestamp_bundle_serialization() {
        let manager = TimestampManager::new(1);
        let bundle = manager.tick();

        let bytes = bundle.to_bytes().unwrap();
        let restored = TimestampBundle::from_bytes(&bytes).unwrap();

        assert_eq!(bundle.lamport_timestamp, restored.lamport_timestamp);
        assert_eq!(bundle.hybrid_logical_clock, restored.hybrid_logical_clock);
    }

    #[test]
    fn test_causal_consistency() {
        let manager1 = TimestampManager::new(1);
        let manager2 = TimestampManager::new(2);

        let bundle1 = manager1.tick();
        let bundle2 = manager2.update_from_remote(&bundle1);
        let bundle3 = manager1.update_from_remote(&bundle2);

        // Check causal ordering
        assert!(bundle1.happens_before(&bundle2));
        assert!(bundle2.happens_before(&bundle3));

        // Check transitive causality
        assert_eq!(
            bundle1.causal_relation(&bundle3),
            CausalRelation::HappensBefore
        );
    }
}
