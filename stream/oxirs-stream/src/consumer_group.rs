//! Kafka-style consumer group coordination.
//!
//! Provides group membership management (join/leave/heartbeat), partition
//! assignment strategies (Range, RoundRobin, Sticky), cooperative incremental
//! rebalancing, offset commit tracking, heartbeat monitoring, coordinator
//! election, consumer lag calculation, and assignment history tracking.

use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Partition assignment strategy used during rebalancing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentStrategy {
    /// Contiguous range of partitions per consumer (ordered by consumer ID).
    Range,
    /// Round-robin distribution across consumers.
    RoundRobin,
    /// Sticky: minimise partition movement across rebalances.
    Sticky,
}

/// State of a consumer within the group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsumerState {
    /// Consumer has joined and is receiving heartbeats.
    Active,
    /// Consumer missed heartbeat deadline but is not yet evicted.
    Lagging,
    /// Consumer has been evicted or explicitly left.
    Dead,
}

/// A member of the consumer group.
#[derive(Debug, Clone)]
pub struct ConsumerMember {
    /// Unique consumer identifier.
    pub consumer_id: String,
    /// Millisecond timestamp of the last heartbeat received.
    pub last_heartbeat_ms: u64,
    /// Current lifecycle state.
    pub state: ConsumerState,
    /// Partitions currently assigned to this consumer.
    pub assigned_partitions: Vec<u32>,
    /// Millisecond timestamp when this consumer joined the group.
    pub joined_at_ms: u64,
}

/// Per-partition committed offset.
#[derive(Debug, Clone)]
pub struct PartitionOffset {
    /// Partition identifier.
    pub partition: u32,
    /// Last committed offset for this partition.
    pub committed_offset: u64,
    /// End-of-log offset (latest produced).
    pub log_end_offset: u64,
    /// Consumer ID that owns this partition, if assigned.
    pub owner: Option<String>,
}

impl PartitionOffset {
    /// Consumer lag: messages produced but not yet committed.
    pub fn lag(&self) -> u64 {
        self.log_end_offset.saturating_sub(self.committed_offset)
    }
}

/// Snapshot of a single rebalance event.
#[derive(Debug, Clone)]
pub struct RebalanceEvent {
    /// Monotonic generation number.
    pub generation: u64,
    /// Timestamp when the rebalance occurred (ms).
    pub timestamp_ms: u64,
    /// Assignment strategy that was in effect.
    pub strategy: AssignmentStrategy,
    /// Partition assignments after the rebalance: consumer_id -> partitions.
    pub assignments: HashMap<String, Vec<u32>>,
    /// Number of partitions that moved between consumers.
    pub partitions_moved: usize,
}

/// Result of a rebalance operation.
#[derive(Debug, Clone)]
pub struct RebalanceResult {
    /// New assignment mapping: consumer_id -> partitions.
    pub assignments: HashMap<String, Vec<u32>>,
    /// Number of partitions that changed owner.
    pub partitions_moved: usize,
    /// Rebalance generation number.
    pub generation: u64,
}

/// Aggregate statistics for the consumer group.
#[derive(Debug, Clone, Default)]
pub struct GroupStats {
    /// Number of active consumers.
    pub active_consumers: usize,
    /// Number of lagging consumers.
    pub lagging_consumers: usize,
    /// Total number of partitions.
    pub total_partitions: u32,
    /// Total consumer lag across all partitions.
    pub total_lag: u64,
    /// Number of rebalances that have occurred.
    pub rebalance_count: u64,
    /// Number of unassigned partitions.
    pub unassigned_partitions: u32,
}

/// Errors from consumer group operations.
#[derive(Debug)]
pub enum GroupError {
    /// Consumer already exists in the group.
    DuplicateConsumer(String),
    /// Consumer not found in the group.
    ConsumerNotFound(String),
    /// Partition is out of range.
    InvalidPartition(u32),
    /// No active consumers available for assignment.
    NoActiveConsumers,
    /// Group has no coordinator elected.
    NoCoordinator,
}

impl std::fmt::Display for GroupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GroupError::DuplicateConsumer(id) => write!(f, "consumer already exists: {id}"),
            GroupError::ConsumerNotFound(id) => write!(f, "consumer not found: {id}"),
            GroupError::InvalidPartition(p) => write!(f, "invalid partition: {p}"),
            GroupError::NoActiveConsumers => write!(f, "no active consumers in group"),
            GroupError::NoCoordinator => write!(f, "no group coordinator elected"),
        }
    }
}

impl std::error::Error for GroupError {}

// ─────────────────────────────────────────────────────────────────────────────
// ConsumerGroup
// ─────────────────────────────────────────────────────────────────────────────

/// Kafka-style consumer group coordinator.
///
/// Manages consumer membership, partition assignment, offset tracking, and
/// rebalancing across the group.
pub struct ConsumerGroup {
    /// Group identifier.
    group_id: String,
    /// Registered consumers keyed by consumer_id.
    members: HashMap<String, ConsumerMember>,
    /// Per-partition offset tracking.
    offsets: HashMap<u32, PartitionOffset>,
    /// Total number of partitions in the topic.
    partition_count: u32,
    /// Assignment strategy in effect.
    strategy: AssignmentStrategy,
    /// Current coordinator consumer ID (if elected).
    coordinator: Option<String>,
    /// Heartbeat timeout in milliseconds; consumers exceeding this are marked lagging.
    heartbeat_timeout_ms: u64,
    /// Session timeout in milliseconds; consumers exceeding this are evicted.
    session_timeout_ms: u64,
    /// Current rebalance generation counter.
    generation: u64,
    /// History of rebalance events (most recent first).
    rebalance_history: VecDeque<RebalanceEvent>,
    /// Maximum number of rebalance events to keep in history.
    max_history: usize,
}

impl ConsumerGroup {
    /// Create a new consumer group.
    ///
    /// `partition_count` is the number of topic partitions to manage.
    /// `heartbeat_timeout_ms` controls when a consumer is marked lagging.
    /// `session_timeout_ms` controls when a consumer is evicted.
    pub fn new(
        group_id: impl Into<String>,
        partition_count: u32,
        strategy: AssignmentStrategy,
        heartbeat_timeout_ms: u64,
        session_timeout_ms: u64,
    ) -> Self {
        let mut offsets = HashMap::new();
        for p in 0..partition_count {
            offsets.insert(
                p,
                PartitionOffset {
                    partition: p,
                    committed_offset: 0,
                    log_end_offset: 0,
                    owner: None,
                },
            );
        }

        Self {
            group_id: group_id.into(),
            members: HashMap::new(),
            offsets,
            partition_count,
            strategy,
            coordinator: None,
            heartbeat_timeout_ms,
            session_timeout_ms,
            generation: 0,
            rebalance_history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Return the group identifier.
    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    /// Return the number of partitions.
    pub fn partition_count(&self) -> u32 {
        self.partition_count
    }

    /// Return the current generation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Return the current coordinator, if any.
    pub fn coordinator(&self) -> Option<&str> {
        self.coordinator.as_deref()
    }

    /// Return the current assignment strategy.
    pub fn strategy(&self) -> &AssignmentStrategy {
        &self.strategy
    }

    /// Set the assignment strategy.
    pub fn set_strategy(&mut self, strategy: AssignmentStrategy) {
        self.strategy = strategy;
    }

    /// Number of registered members.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Get a member by ID.
    pub fn get_member(&self, consumer_id: &str) -> Option<&ConsumerMember> {
        self.members.get(consumer_id)
    }

    /// Get partition offset information.
    pub fn get_offset(&self, partition: u32) -> Option<&PartitionOffset> {
        self.offsets.get(&partition)
    }

    /// Return the rebalance history.
    pub fn rebalance_history(&self) -> &VecDeque<RebalanceEvent> {
        &self.rebalance_history
    }

    // ─── Membership ──────────────────────────────────────────────────────────

    /// Join a consumer to the group.
    ///
    /// Returns an error if the consumer already exists. The first consumer
    /// to join becomes the coordinator.
    pub fn join(&mut self, consumer_id: impl Into<String>, now_ms: u64) -> Result<(), GroupError> {
        let id = consumer_id.into();
        if self.members.contains_key(&id) {
            return Err(GroupError::DuplicateConsumer(id));
        }

        let member = ConsumerMember {
            consumer_id: id.clone(),
            last_heartbeat_ms: now_ms,
            state: ConsumerState::Active,
            assigned_partitions: Vec::new(),
            joined_at_ms: now_ms,
        };
        self.members.insert(id.clone(), member);

        // First consumer becomes coordinator
        if self.coordinator.is_none() {
            self.coordinator = Some(id);
        }

        Ok(())
    }

    /// Remove a consumer from the group.
    ///
    /// If the removed consumer was the coordinator, elects a new one.
    pub fn leave(&mut self, consumer_id: &str) -> Result<(), GroupError> {
        if self.members.remove(consumer_id).is_none() {
            return Err(GroupError::ConsumerNotFound(consumer_id.to_string()));
        }

        // Clear partition ownership for this consumer
        for offset in self.offsets.values_mut() {
            if offset.owner.as_deref() == Some(consumer_id) {
                offset.owner = None;
            }
        }

        // Re-elect coordinator if needed
        if self.coordinator.as_deref() == Some(consumer_id) {
            self.elect_coordinator();
        }

        Ok(())
    }

    /// Record a heartbeat from a consumer.
    pub fn heartbeat(&mut self, consumer_id: &str, now_ms: u64) -> Result<(), GroupError> {
        let member = self
            .members
            .get_mut(consumer_id)
            .ok_or_else(|| GroupError::ConsumerNotFound(consumer_id.to_string()))?;

        member.last_heartbeat_ms = now_ms;
        if member.state == ConsumerState::Lagging {
            member.state = ConsumerState::Active;
        }

        Ok(())
    }

    /// Check all consumers for heartbeat timeouts.
    ///
    /// Marks consumers as `Lagging` if they exceed `heartbeat_timeout_ms`,
    /// and evicts (marks `Dead` and removes partitions) those exceeding
    /// `session_timeout_ms`.
    ///
    /// Returns the IDs of consumers that were evicted.
    pub fn check_heartbeats(&mut self, now_ms: u64) -> Vec<String> {
        let mut evicted = Vec::new();

        let member_ids: Vec<String> = self.members.keys().cloned().collect();
        for id in member_ids {
            if let Some(member) = self.members.get_mut(&id) {
                let elapsed = now_ms.saturating_sub(member.last_heartbeat_ms);

                if elapsed >= self.session_timeout_ms {
                    member.state = ConsumerState::Dead;
                    evicted.push(id.clone());
                } else if elapsed >= self.heartbeat_timeout_ms {
                    member.state = ConsumerState::Lagging;
                }
            }
        }

        // Remove evicted consumers and clear their partition ownership
        for id in &evicted {
            self.members.remove(id);
            for offset in self.offsets.values_mut() {
                if offset.owner.as_deref() == Some(id.as_str()) {
                    offset.owner = None;
                }
            }
        }

        // Re-elect coordinator if evicted
        if let Some(ref coord) = self.coordinator {
            if evicted.contains(coord) {
                self.elect_coordinator();
            }
        }

        evicted
    }

    // ─── Coordinator Election ────────────────────────────────────────────────

    /// Elect a coordinator from the active members.
    ///
    /// Selects the lexicographically smallest active consumer ID.
    pub fn elect_coordinator(&mut self) {
        let mut candidates: Vec<&str> = self
            .members
            .values()
            .filter(|m| m.state == ConsumerState::Active)
            .map(|m| m.consumer_id.as_str())
            .collect();

        candidates.sort();
        self.coordinator = candidates.first().map(|s| s.to_string());
    }

    // ─── Offset Management ──────────────────────────────────────────────────

    /// Commit an offset for a partition.
    pub fn commit_offset(&mut self, partition: u32, offset: u64) -> Result<(), GroupError> {
        let po = self
            .offsets
            .get_mut(&partition)
            .ok_or(GroupError::InvalidPartition(partition))?;
        po.committed_offset = offset;
        Ok(())
    }

    /// Update the log-end offset for a partition (producer side).
    pub fn update_log_end_offset(&mut self, partition: u32, offset: u64) -> Result<(), GroupError> {
        let po = self
            .offsets
            .get_mut(&partition)
            .ok_or(GroupError::InvalidPartition(partition))?;
        po.log_end_offset = offset;
        Ok(())
    }

    /// Calculate the total consumer lag across all partitions.
    pub fn total_lag(&self) -> u64 {
        self.offsets.values().map(|po| po.lag()).sum()
    }

    /// Calculate per-consumer lag: sum of lags on partitions assigned to each consumer.
    pub fn consumer_lag(&self) -> HashMap<String, u64> {
        let mut lags: HashMap<String, u64> = HashMap::new();
        for po in self.offsets.values() {
            if let Some(ref owner) = po.owner {
                *lags.entry(owner.clone()).or_insert(0) += po.lag();
            }
        }
        lags
    }

    // ─── Rebalancing ────────────────────────────────────────────────────────

    /// Trigger a rebalance using the current assignment strategy.
    ///
    /// Returns the result with partition assignments and movement count.
    pub fn rebalance(&mut self, now_ms: u64) -> Result<RebalanceResult, GroupError> {
        let active_ids = self.active_consumer_ids();
        if active_ids.is_empty() {
            return Err(GroupError::NoActiveConsumers);
        }

        // Build old assignments for movement calculation
        let old_assignments = self.current_assignments();

        let new_assignments = match &self.strategy {
            AssignmentStrategy::Range => self.assign_range(&active_ids),
            AssignmentStrategy::RoundRobin => self.assign_round_robin(&active_ids),
            AssignmentStrategy::Sticky => self.assign_sticky(&active_ids, &old_assignments),
        };

        let partitions_moved = self.count_moved(&old_assignments, &new_assignments);

        // Apply assignments to members and offsets
        self.apply_assignments(&new_assignments);

        self.generation += 1;

        // Record in history
        let event = RebalanceEvent {
            generation: self.generation,
            timestamp_ms: now_ms,
            strategy: self.strategy.clone(),
            assignments: new_assignments.clone(),
            partitions_moved,
        };
        self.rebalance_history.push_front(event);
        while self.rebalance_history.len() > self.max_history {
            self.rebalance_history.pop_back();
        }

        Ok(RebalanceResult {
            assignments: new_assignments,
            partitions_moved,
            generation: self.generation,
        })
    }

    /// Get current partition assignments as consumer_id -> partitions.
    pub fn current_assignments(&self) -> HashMap<String, Vec<u32>> {
        let mut assignments: HashMap<String, Vec<u32>> = HashMap::new();
        for member in self.members.values() {
            assignments.insert(
                member.consumer_id.clone(),
                member.assigned_partitions.clone(),
            );
        }
        assignments
    }

    /// Aggregate statistics for the group.
    pub fn stats(&self) -> GroupStats {
        let active_consumers = self
            .members
            .values()
            .filter(|m| m.state == ConsumerState::Active)
            .count();
        let lagging_consumers = self
            .members
            .values()
            .filter(|m| m.state == ConsumerState::Lagging)
            .count();
        let unassigned_partitions = self
            .offsets
            .values()
            .filter(|po| po.owner.is_none())
            .count() as u32;

        GroupStats {
            active_consumers,
            lagging_consumers,
            total_partitions: self.partition_count,
            total_lag: self.total_lag(),
            rebalance_count: self.generation,
            unassigned_partitions,
        }
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    fn active_consumer_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .members
            .values()
            .filter(|m| m.state == ConsumerState::Active)
            .map(|m| m.consumer_id.clone())
            .collect();
        ids.sort();
        ids
    }

    /// Range assignment: partitions are divided into contiguous ranges.
    fn assign_range(&self, consumer_ids: &[String]) -> HashMap<String, Vec<u32>> {
        let n = consumer_ids.len();
        let mut result: HashMap<String, Vec<u32>> = HashMap::new();
        if n == 0 {
            return result;
        }

        let per_consumer = self.partition_count as usize / n;
        let remainder = self.partition_count as usize % n;

        let mut partition = 0u32;
        for (i, cid) in consumer_ids.iter().enumerate() {
            let count = per_consumer + if i < remainder { 1 } else { 0 };
            let mut parts = Vec::new();
            for _ in 0..count {
                parts.push(partition);
                partition += 1;
            }
            result.insert(cid.clone(), parts);
        }

        result
    }

    /// Round-robin assignment: partitions distributed one-by-one across consumers.
    fn assign_round_robin(&self, consumer_ids: &[String]) -> HashMap<String, Vec<u32>> {
        let n = consumer_ids.len();
        let mut result: HashMap<String, Vec<u32>> = HashMap::new();
        if n == 0 {
            return result;
        }

        for cid in consumer_ids {
            result.insert(cid.clone(), Vec::new());
        }

        for p in 0..self.partition_count {
            let idx = p as usize % n;
            if let Some(parts) = result.get_mut(&consumer_ids[idx]) {
                parts.push(p);
            }
        }

        result
    }

    /// Sticky assignment: try to keep existing assignments, only move unassigned
    /// or partitions from departed consumers.
    fn assign_sticky(
        &self,
        consumer_ids: &[String],
        old_assignments: &HashMap<String, Vec<u32>>,
    ) -> HashMap<String, Vec<u32>> {
        let n = consumer_ids.len();
        let mut result: HashMap<String, Vec<u32>> = HashMap::new();
        if n == 0 {
            return result;
        }

        for cid in consumer_ids {
            result.insert(cid.clone(), Vec::new());
        }

        // Phase 1: Keep existing assignments for consumers that are still active.
        let mut assigned: Vec<bool> = vec![false; self.partition_count as usize];
        for (cid, parts) in old_assignments {
            if result.contains_key(cid) {
                for &p in parts {
                    if (p as usize) < assigned.len() {
                        assigned[p as usize] = true;
                        if let Some(v) = result.get_mut(cid) {
                            v.push(p);
                        }
                    }
                }
            }
        }

        // Phase 2: Distribute unassigned partitions round-robin.
        let mut rr_idx = 0usize;
        for p in 0..self.partition_count {
            if !assigned[p as usize] {
                let cid = &consumer_ids[rr_idx % n];
                if let Some(v) = result.get_mut(cid) {
                    v.push(p);
                }
                rr_idx += 1;
            }
        }

        // Phase 3: Rebalance if any consumer has too many.
        let target = self.partition_count as usize / n;
        let target_extra = self.partition_count as usize % n;

        // Collect surplus partitions from over-loaded consumers.
        let mut surplus: Vec<u32> = Vec::new();
        let mut sorted_ids = consumer_ids.to_vec();
        sorted_ids.sort();

        for (i, cid) in sorted_ids.iter().enumerate() {
            let max_allowed = target + if i < target_extra { 1 } else { 0 };
            if let Some(v) = result.get_mut(cid) {
                while v.len() > max_allowed {
                    if let Some(p) = v.pop() {
                        surplus.push(p);
                    }
                }
            }
        }

        // Distribute surplus to under-loaded consumers.
        let mut surplus_iter = surplus.into_iter();
        for (i, cid) in sorted_ids.iter().enumerate() {
            let max_allowed = target + if i < target_extra { 1 } else { 0 };
            if let Some(v) = result.get_mut(cid) {
                while v.len() < max_allowed {
                    if let Some(p) = surplus_iter.next() {
                        v.push(p);
                    } else {
                        break;
                    }
                }
            }
        }

        result
    }

    /// Count how many partitions changed owner between old and new assignments.
    fn count_moved(
        &self,
        old: &HashMap<String, Vec<u32>>,
        new: &HashMap<String, Vec<u32>>,
    ) -> usize {
        // Build partition -> owner maps.
        let mut old_owner: HashMap<u32, &str> = HashMap::new();
        for (cid, parts) in old {
            for &p in parts {
                old_owner.insert(p, cid.as_str());
            }
        }

        let mut new_owner: HashMap<u32, &str> = HashMap::new();
        for (cid, parts) in new {
            for &p in parts {
                new_owner.insert(p, cid.as_str());
            }
        }

        let mut moved = 0usize;
        for p in 0..self.partition_count {
            let old_o = old_owner.get(&p).copied();
            let new_o = new_owner.get(&p).copied();
            if old_o != new_o {
                moved += 1;
            }
        }
        moved
    }

    /// Apply computed assignments to members and partition offsets.
    fn apply_assignments(&mut self, assignments: &HashMap<String, Vec<u32>>) {
        // Clear all member assignments and partition owners first.
        for member in self.members.values_mut() {
            member.assigned_partitions.clear();
        }
        for offset in self.offsets.values_mut() {
            offset.owner = None;
        }

        // Apply new assignments.
        for (cid, parts) in assignments {
            if let Some(member) = self.members.get_mut(cid) {
                member.assigned_partitions = parts.clone();
            }
            for &p in parts {
                if let Some(offset) = self.offsets.get_mut(&p) {
                    offset.owner = Some(cid.clone());
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_group(partitions: u32, strategy: AssignmentStrategy) -> ConsumerGroup {
        ConsumerGroup::new("test-group", partitions, strategy, 3000, 10_000)
    }

    // ── Membership Tests ─────────────────────────────────────────────────

    #[test]
    fn test_join_single_consumer() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        assert!(g.join("c1", 1000).is_ok());
        assert_eq!(g.member_count(), 1);
        assert_eq!(g.coordinator(), Some("c1"));
    }

    #[test]
    fn test_join_duplicate_consumer_error() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 1000).ok();
        let err = g.join("c1", 2000);
        assert!(err.is_err());
    }

    #[test]
    fn test_leave_consumer() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 1000).ok();
        g.join("c2", 1000).ok();
        assert!(g.leave("c1").is_ok());
        assert_eq!(g.member_count(), 1);
    }

    #[test]
    fn test_leave_nonexistent_error() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        let err = g.leave("nobody");
        assert!(err.is_err());
    }

    #[test]
    fn test_coordinator_election_on_join() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c2", 1000).ok();
        assert_eq!(g.coordinator(), Some("c2"));
        g.join("c1", 1000).ok();
        // Coordinator stays as the first joiner
        assert_eq!(g.coordinator(), Some("c2"));
    }

    #[test]
    fn test_coordinator_reelection_on_leave() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 1000).ok();
        g.join("c2", 1000).ok();
        assert_eq!(g.coordinator(), Some("c1"));
        g.leave("c1").ok();
        // c2 becomes coordinator
        assert_eq!(g.coordinator(), Some("c2"));
    }

    // ── Heartbeat Tests ──────────────────────────────────────────────────

    #[test]
    fn test_heartbeat_updates_timestamp() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 1000).ok();
        g.heartbeat("c1", 5000).ok();
        let member = g.get_member("c1");
        assert!(member.is_some());
        assert_eq!(member.map(|m| m.last_heartbeat_ms), Some(5000));
    }

    #[test]
    fn test_heartbeat_nonexistent_error() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        let err = g.heartbeat("nobody", 1000);
        assert!(err.is_err());
    }

    #[test]
    fn test_check_heartbeats_marks_lagging() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        // 4000ms elapsed > 3000ms heartbeat timeout
        let evicted = g.check_heartbeats(4000);
        assert!(evicted.is_empty());
        assert_eq!(
            g.get_member("c1").map(|m| m.state.clone()),
            Some(ConsumerState::Lagging)
        );
    }

    #[test]
    fn test_check_heartbeats_evicts_dead() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        // 11000ms > 10000ms session timeout
        let evicted = g.check_heartbeats(11_000);
        assert_eq!(evicted, vec!["c1"]);
        assert_eq!(g.member_count(), 0);
    }

    #[test]
    fn test_heartbeat_restores_lagging_to_active() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.check_heartbeats(4000);
        assert_eq!(
            g.get_member("c1").map(|m| m.state.clone()),
            Some(ConsumerState::Lagging)
        );
        g.heartbeat("c1", 4500).ok();
        assert_eq!(
            g.get_member("c1").map(|m| m.state.clone()),
            Some(ConsumerState::Active)
        );
    }

    // ── Offset Management Tests ──────────────────────────────────────────

    #[test]
    fn test_commit_offset() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        assert!(g.commit_offset(0, 100).is_ok());
        assert_eq!(g.get_offset(0).map(|o| o.committed_offset), Some(100));
    }

    #[test]
    fn test_commit_offset_invalid_partition() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        assert!(g.commit_offset(99, 100).is_err());
    }

    #[test]
    fn test_update_log_end_offset() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        assert!(g.update_log_end_offset(2, 500).is_ok());
        assert_eq!(g.get_offset(2).map(|o| o.log_end_offset), Some(500));
    }

    #[test]
    fn test_partition_lag() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.update_log_end_offset(0, 1000).ok();
        g.commit_offset(0, 300).ok();
        assert_eq!(g.get_offset(0).map(|o| o.lag()), Some(700));
    }

    #[test]
    fn test_total_lag() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        for p in 0..4 {
            g.update_log_end_offset(p, 100).ok();
            g.commit_offset(p, 30).ok();
        }
        assert_eq!(g.total_lag(), 280); // 70 * 4
    }

    #[test]
    fn test_consumer_lag_per_consumer() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        g.rebalance(0).ok();

        for p in 0..4 {
            g.update_log_end_offset(p, 100).ok();
            g.commit_offset(p, 50).ok();
        }

        let lags = g.consumer_lag();
        assert!(lags.contains_key("c1"));
        assert!(lags.contains_key("c2"));
        let total: u64 = lags.values().sum();
        assert_eq!(total, 200);
    }

    // ── Range Assignment Tests ───────────────────────────────────────────

    #[test]
    fn test_range_assignment_even() {
        let mut g = make_group(6, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        g.join("c3", 0).ok();
        let result = g.rebalance(0);
        assert!(result.is_ok());
        let r = result.ok();
        assert!(r.is_some());
        let r = r.expect("rebalance result");
        assert_eq!(r.assignments.get("c1").map(|v| v.len()), Some(2));
        assert_eq!(r.assignments.get("c2").map(|v| v.len()), Some(2));
        assert_eq!(r.assignments.get("c3").map(|v| v.len()), Some(2));
    }

    #[test]
    fn test_range_assignment_uneven() {
        let mut g = make_group(7, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        g.join("c3", 0).ok();
        let r = g.rebalance(0).expect("rebalance result");
        // 7 / 3 = 2 remainder 1, first consumer gets extra
        let total: usize = r.assignments.values().map(|v| v.len()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn test_range_assignment_single_consumer() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        let r = g.rebalance(0).expect("rebalance result");
        assert_eq!(r.assignments.get("c1").map(|v| v.len()), Some(4));
    }

    #[test]
    fn test_range_assignment_contiguous() {
        let mut g = make_group(6, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        let r = g.rebalance(0).expect("rebalance result");
        let c1_parts = r.assignments.get("c1").cloned().unwrap_or_default();
        let c2_parts = r.assignments.get("c2").cloned().unwrap_or_default();
        // c1 should get [0,1,2] and c2 should get [3,4,5]
        assert_eq!(c1_parts, vec![0, 1, 2]);
        assert_eq!(c2_parts, vec![3, 4, 5]);
    }

    // ── RoundRobin Assignment Tests ──────────────────────────────────────

    #[test]
    fn test_roundrobin_assignment() {
        let mut g = make_group(6, AssignmentStrategy::RoundRobin);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        let r = g.rebalance(0).expect("rebalance result");
        assert_eq!(r.assignments.get("c1").map(|v| v.len()), Some(3));
        assert_eq!(r.assignments.get("c2").map(|v| v.len()), Some(3));
    }

    #[test]
    fn test_roundrobin_interleaved() {
        let mut g = make_group(4, AssignmentStrategy::RoundRobin);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        let r = g.rebalance(0).expect("rebalance result");
        let c1_parts = r.assignments.get("c1").cloned().unwrap_or_default();
        let c2_parts = r.assignments.get("c2").cloned().unwrap_or_default();
        assert_eq!(c1_parts, vec![0, 2]);
        assert_eq!(c2_parts, vec![1, 3]);
    }

    // ── Sticky Assignment Tests ──────────────────────────────────────────

    #[test]
    fn test_sticky_preserves_existing() {
        let mut g = make_group(4, AssignmentStrategy::Sticky);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        g.rebalance(0).ok();

        // Add a third consumer and rebalance
        g.join("c3", 100).ok();
        let r = g.rebalance(100).expect("rebalance result");
        let total: usize = r.assignments.values().map(|v| v.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_sticky_minimal_movement() {
        let mut g = make_group(6, AssignmentStrategy::Sticky);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        let r1 = g.rebalance(0).expect("rebalance result");

        // Add a third consumer
        g.join("c3", 100).ok();
        let r2 = g.rebalance(100).expect("rebalance result");

        // Sticky should move fewer partitions than total
        assert!(r2.partitions_moved <= r1.assignments.values().map(|v| v.len()).sum::<usize>());
    }

    // ── Rebalance Protocol Tests ─────────────────────────────────────────

    #[test]
    fn test_rebalance_no_consumers_error() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        let err = g.rebalance(0);
        assert!(err.is_err());
    }

    #[test]
    fn test_rebalance_increments_generation() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        assert_eq!(g.generation(), 0);
        g.rebalance(0).ok();
        assert_eq!(g.generation(), 1);
        g.rebalance(100).ok();
        assert_eq!(g.generation(), 2);
    }

    #[test]
    fn test_rebalance_history_recorded() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.rebalance(0).ok();
        g.rebalance(100).ok();
        assert_eq!(g.rebalance_history().len(), 2);
        // Most recent first
        assert_eq!(g.rebalance_history().front().map(|e| e.generation), Some(2));
    }

    #[test]
    fn test_rebalance_updates_partition_owners() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.rebalance(0).ok();
        for p in 0..4 {
            assert_eq!(g.get_offset(p).and_then(|o| o.owner.as_deref()), Some("c1"));
        }
    }

    #[test]
    fn test_rebalance_after_leave() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        g.rebalance(0).ok();
        g.leave("c2").ok();
        let r = g.rebalance(100).expect("rebalance result");
        // All partitions should go to c1
        assert_eq!(r.assignments.get("c1").map(|v| v.len()), Some(4));
    }

    // ── Stats Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_group_stats() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        let stats = g.stats();
        assert_eq!(stats.active_consumers, 2);
        assert_eq!(stats.total_partitions, 4);
        assert_eq!(stats.unassigned_partitions, 4); // not yet rebalanced
    }

    #[test]
    fn test_stats_after_rebalance() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.rebalance(0).ok();
        let stats = g.stats();
        assert_eq!(stats.unassigned_partitions, 0);
        assert_eq!(stats.rebalance_count, 1);
    }

    #[test]
    fn test_stats_with_lag() {
        let mut g = make_group(2, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.rebalance(0).ok();
        g.update_log_end_offset(0, 100).ok();
        g.update_log_end_offset(1, 200).ok();
        let stats = g.stats();
        assert_eq!(stats.total_lag, 300);
    }

    // ── Edge Cases ───────────────────────────────────────────────────────

    #[test]
    fn test_zero_partitions() {
        let mut g = make_group(0, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        let r = g.rebalance(0).expect("rebalance result");
        assert_eq!(r.assignments.get("c1").map(|v| v.len()), Some(0));
    }

    #[test]
    fn test_more_consumers_than_partitions() {
        let mut g = make_group(2, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        g.join("c3", 0).ok();
        let r = g.rebalance(0).expect("rebalance result");
        let total: usize = r.assignments.values().map(|v| v.len()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_set_strategy() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.set_strategy(AssignmentStrategy::RoundRobin);
        assert_eq!(g.strategy(), &AssignmentStrategy::RoundRobin);
    }

    #[test]
    fn test_current_assignments_empty() {
        let g = make_group(4, AssignmentStrategy::Range);
        let a = g.current_assignments();
        assert!(a.is_empty());
    }

    #[test]
    fn test_partition_offset_lag_zero() {
        let po = PartitionOffset {
            partition: 0,
            committed_offset: 100,
            log_end_offset: 100,
            owner: None,
        };
        assert_eq!(po.lag(), 0);
    }

    #[test]
    fn test_partition_offset_lag_underflow() {
        let po = PartitionOffset {
            partition: 0,
            committed_offset: 200,
            log_end_offset: 100,
            owner: None,
        };
        assert_eq!(po.lag(), 0); // saturating_sub
    }

    #[test]
    fn test_eviction_clears_partition_ownership() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.rebalance(0).ok();
        // Verify c1 owns partitions
        assert!(g.get_offset(0).and_then(|o| o.owner.as_deref()).is_some());
        // Evict via session timeout
        g.check_heartbeats(11_000);
        // Ownership should be cleared
        for p in 0..4 {
            assert_eq!(g.get_offset(p).and_then(|o| o.owner.as_deref()), None);
        }
    }

    #[test]
    fn test_coordinator_none_after_all_leave() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        g.leave("c1").ok();
        assert_eq!(g.coordinator(), None);
    }

    #[test]
    fn test_roundrobin_three_consumers_five_partitions() {
        let mut g = make_group(5, AssignmentStrategy::RoundRobin);
        g.join("c1", 0).ok();
        g.join("c2", 0).ok();
        g.join("c3", 0).ok();
        let r = g.rebalance(0).expect("rebalance result");
        let c1 = r.assignments.get("c1").map(|v| v.len()).unwrap_or(0);
        let c2 = r.assignments.get("c2").map(|v| v.len()).unwrap_or(0);
        let c3 = r.assignments.get("c3").map(|v| v.len()).unwrap_or(0);
        assert_eq!(c1 + c2 + c3, 5);
        // At most 1 difference between min and max
        let max_val = c1.max(c2).max(c3);
        let min_val = c1.min(c2).min(c3);
        assert!(max_val - min_val <= 1);
    }

    #[test]
    fn test_rebalance_history_max_limit() {
        let mut g = ConsumerGroup::new("test", 4, AssignmentStrategy::Range, 3000, 10_000);
        g.join("c1", 0).ok();
        // Set small history limit
        g.max_history = 3;
        for i in 0..10 {
            g.rebalance(i * 100).ok();
        }
        assert!(g.rebalance_history().len() <= 3);
    }

    #[test]
    fn test_multiple_heartbeats() {
        let mut g = make_group(4, AssignmentStrategy::Range);
        g.join("c1", 0).ok();
        for t in (1000..5000).step_by(500) {
            assert!(g.heartbeat("c1", t).is_ok());
        }
        assert_eq!(g.get_member("c1").map(|m| m.last_heartbeat_ms), Some(4500));
    }

    #[test]
    fn test_group_id_accessor() {
        let g = make_group(4, AssignmentStrategy::Range);
        assert_eq!(g.group_id(), "test-group");
    }

    #[test]
    fn test_partition_count_accessor() {
        let g = make_group(8, AssignmentStrategy::Range);
        assert_eq!(g.partition_count(), 8);
    }

    #[test]
    fn test_group_error_display() {
        let e = GroupError::DuplicateConsumer("c1".to_string());
        assert!(format!("{e}").contains("c1"));

        let e = GroupError::ConsumerNotFound("c2".to_string());
        assert!(format!("{e}").contains("c2"));

        let e = GroupError::InvalidPartition(99);
        assert!(format!("{e}").contains("99"));

        let e = GroupError::NoActiveConsumers;
        assert!(!format!("{e}").is_empty());

        let e = GroupError::NoCoordinator;
        assert!(!format!("{e}").is_empty());
    }
}
