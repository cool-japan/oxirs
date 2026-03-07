//! Automatic failover handling for distributed clusters.
//!
//! [`FailoverController`] monitors cluster health, detects leader failure via
//! heartbeat timeouts, triggers re-election, promotes read replicas, and
//! prevents split-brain scenarios using fencing tokens and quorum decisions.
//! All failover events are logged for auditing.

use std::collections::HashMap;

// ── Failover policy ─────────────────────────────────────────────────────────

/// Policy governing when and how failover is triggered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailoverPolicy {
    /// Failover happens automatically when the leader is unreachable.
    Automatic,
    /// Failover must be initiated manually (the controller still detects, but
    /// does not promote).
    Manual,
}

impl Default for FailoverPolicy {
    fn default() -> Self {
        Self::Automatic
    }
}

// ── Failover event ──────────────────────────────────────────────────────────

/// Types of failover events that are recorded in the history log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailoverEventKind {
    /// A failure was detected for the given node.
    FailureDetected,
    /// A new election was initiated.
    ElectionTriggered,
    /// A replica was promoted to leader.
    ReplicaPromoted,
    /// A split-brain was detected and fenced.
    SplitBrainFenced,
    /// A graceful failover (drain + transfer) completed.
    GracefulTransfer,
    /// Health score dropped below the configured threshold.
    HealthThresholdBreached,
    /// Failover was vetoed because quorum was not met.
    QuorumNotMet,
}

/// A single failover event in the history log.
#[derive(Debug, Clone)]
pub struct FailoverEvent {
    /// Which node this event pertains to.
    pub node_id: String,
    /// Kind of event.
    pub kind: FailoverEventKind,
    /// Wall-clock timestamp in milliseconds.
    pub timestamp: u64,
    /// Optional description / context.
    pub detail: Option<String>,
}

impl FailoverEvent {
    fn new(
        node_id: impl Into<String>,
        kind: FailoverEventKind,
        timestamp: u64,
        detail: Option<String>,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            kind,
            timestamp,
            detail,
        }
    }
}

// ── Node health record ──────────────────────────────────────────────────────

/// Runtime health information for a single node.
#[derive(Debug, Clone)]
pub struct NodeHealthRecord {
    /// Unique node identifier.
    pub node_id: String,
    /// Whether this node is currently the leader.
    pub is_leader: bool,
    /// Whether this node is a read replica (eligible for promotion).
    pub is_replica: bool,
    /// Last heartbeat timestamp (ms).
    pub last_heartbeat: u64,
    /// Consecutive heartbeat failures (incremented on timeout, reset on
    /// successful heartbeat).
    pub consecutive_failures: u32,
    /// Health score in [0.0, 1.0].  A score below the configured threshold
    /// triggers failover.
    pub health_score: f64,
    /// The fencing token held by this node (only the leader should have the
    /// latest token).
    pub fencing_token: u64,
    /// Number of active client connections (for graceful drain).
    pub active_connections: u32,
}

impl NodeHealthRecord {
    /// Create a new healthy follower node.
    pub fn new(node_id: impl Into<String>, now: u64) -> Self {
        Self {
            node_id: node_id.into(),
            is_leader: false,
            is_replica: true,
            last_heartbeat: now,
            consecutive_failures: 0,
            health_score: 1.0,
            fencing_token: 0,
            active_connections: 0,
        }
    }

    /// Create a new leader node record.
    pub fn leader(node_id: impl Into<String>, now: u64, fencing_token: u64) -> Self {
        Self {
            node_id: node_id.into(),
            is_leader: true,
            is_replica: false,
            last_heartbeat: now,
            consecutive_failures: 0,
            health_score: 1.0,
            fencing_token,
            active_connections: 0,
        }
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the failover controller.
#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// How long (ms) without a heartbeat before a node is considered failed.
    pub heartbeat_timeout_ms: u64,
    /// Number of consecutive heartbeat misses required to declare failure.
    pub max_consecutive_failures: u32,
    /// Health score threshold; below this triggers failover.
    pub health_threshold: f64,
    /// The failover policy (automatic vs manual).
    pub policy: FailoverPolicy,
    /// Minimum cluster quorum required for failover decisions.  If fewer
    /// nodes are alive, failover is vetoed.
    pub quorum_size: usize,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout_ms: 5000,
            max_consecutive_failures: 3,
            health_threshold: 0.3,
            policy: FailoverPolicy::Automatic,
            quorum_size: 2,
        }
    }
}

// ── Election result ─────────────────────────────────────────────────────────

/// Result of a leader election attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElectionResult {
    /// A new leader was elected.
    Elected(String),
    /// No eligible candidate was available.
    NoCandidateAvailable,
    /// Quorum was not met; election vetoed.
    QuorumNotMet,
}

// ── FailoverController ──────────────────────────────────────────────────────

/// Manages automatic failover for a distributed cluster.
pub struct FailoverController {
    config: FailoverConfig,
    nodes: HashMap<String, NodeHealthRecord>,
    current_leader: Option<String>,
    fencing_token_counter: u64,
    event_log: Vec<FailoverEvent>,
}

impl FailoverController {
    /// Create a new failover controller with the given configuration.
    pub fn new(config: FailoverConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            current_leader: None,
            fencing_token_counter: 0,
            event_log: Vec::new(),
        }
    }

    /// Create a controller with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(FailoverConfig::default())
    }

    /// Return the current configuration.
    pub fn config(&self) -> &FailoverConfig {
        &self.config
    }

    /// Register a node with the controller.
    pub fn register_node(&mut self, record: NodeHealthRecord) {
        if record.is_leader {
            self.current_leader = Some(record.node_id.clone());
        }
        self.nodes.insert(record.node_id.clone(), record);
    }

    /// Remove a node from the controller.
    pub fn unregister_node(&mut self, node_id: &str) -> bool {
        if self.current_leader.as_deref() == Some(node_id) {
            self.current_leader = None;
        }
        self.nodes.remove(node_id).is_some()
    }

    /// Return the number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the current leader node id, if any.
    pub fn current_leader(&self) -> Option<&str> {
        self.current_leader.as_deref()
    }

    /// Return a reference to a specific node record.
    pub fn node(&self, node_id: &str) -> Option<&NodeHealthRecord> {
        self.nodes.get(node_id)
    }

    /// Return all event log entries.
    pub fn event_log(&self) -> &[FailoverEvent] {
        &self.event_log
    }

    /// Return the number of recorded events.
    pub fn event_count(&self) -> usize {
        self.event_log.len()
    }

    /// Return the latest fencing token value.
    pub fn fencing_token(&self) -> u64 {
        self.fencing_token_counter
    }

    // ── heartbeat ───────────────────────────────────────────────────────────

    /// Record a successful heartbeat from a node.
    pub fn heartbeat(&mut self, node_id: &str, now: u64) -> bool {
        if let Some(record) = self.nodes.get_mut(node_id) {
            record.last_heartbeat = now;
            record.consecutive_failures = 0;
            true
        } else {
            false
        }
    }

    /// Update the health score of a node.
    pub fn update_health_score(&mut self, node_id: &str, score: f64) -> bool {
        if let Some(record) = self.nodes.get_mut(node_id) {
            record.health_score = score.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Update the number of active connections for a node.
    pub fn update_connections(&mut self, node_id: &str, count: u32) -> bool {
        if let Some(record) = self.nodes.get_mut(node_id) {
            record.active_connections = count;
            true
        } else {
            false
        }
    }

    // ── failure detection ───────────────────────────────────────────────────

    /// Check all nodes for heartbeat timeouts and increment failure counters.
    ///
    /// Returns a list of node ids that have exceeded `max_consecutive_failures`.
    pub fn detect_failures(&mut self, now: u64) -> Vec<String> {
        let timeout = self.config.heartbeat_timeout_ms;
        let max_failures = self.config.max_consecutive_failures;
        let mut failed = Vec::new();

        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        for id in &node_ids {
            if let Some(record) = self.nodes.get_mut(id) {
                let elapsed = now.saturating_sub(record.last_heartbeat);
                if elapsed > timeout {
                    record.consecutive_failures += 1;
                    if record.consecutive_failures >= max_failures {
                        failed.push(id.clone());
                    }
                }
            }
        }

        // Log detection events.
        let ts = now;
        for id in &failed {
            self.event_log.push(FailoverEvent::new(
                id.as_str(),
                FailoverEventKind::FailureDetected,
                ts,
                Some(format!("consecutive failures >= {max_failures}")),
            ));
        }

        failed
    }

    /// Check for nodes whose health score is below the configured threshold.
    pub fn detect_health_breaches(&mut self, now: u64) -> Vec<String> {
        let threshold = self.config.health_threshold;
        let mut breached = Vec::new();

        for record in self.nodes.values() {
            if record.health_score < threshold {
                breached.push(record.node_id.clone());
            }
        }

        for id in &breached {
            self.event_log.push(FailoverEvent::new(
                id.as_str(),
                FailoverEventKind::HealthThresholdBreached,
                now,
                Some(format!("score below {threshold}")),
            ));
        }

        breached
    }

    // ── leader election ─────────────────────────────────────────────────────

    /// Trigger a new leader election.
    ///
    /// Selects the healthy replica with the highest health score.  Requires
    /// quorum to proceed.
    pub fn trigger_election(&mut self, now: u64) -> ElectionResult {
        self.event_log.push(FailoverEvent::new(
            self.current_leader.as_deref().unwrap_or("none"),
            FailoverEventKind::ElectionTriggered,
            now,
            None,
        ));

        // Count alive nodes (those within heartbeat timeout).
        let timeout = self.config.heartbeat_timeout_ms;
        let alive_count = self
            .nodes
            .values()
            .filter(|n| now.saturating_sub(n.last_heartbeat) <= timeout)
            .count();

        if alive_count < self.config.quorum_size {
            self.event_log.push(FailoverEvent::new(
                "cluster",
                FailoverEventKind::QuorumNotMet,
                now,
                Some(format!(
                    "alive={alive_count}, quorum={}",
                    self.config.quorum_size
                )),
            ));
            return ElectionResult::QuorumNotMet;
        }

        // Find the best candidate: replica, alive, highest health score.
        let best_candidate = self
            .nodes
            .values()
            .filter(|n| {
                n.is_replica
                    && now.saturating_sub(n.last_heartbeat) <= timeout
                    && n.health_score >= self.config.health_threshold
            })
            .max_by(|a, b| {
                a.health_score
                    .partial_cmp(&b.health_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|n| n.node_id.clone());

        match best_candidate {
            Some(candidate_id) => {
                self.promote_to_leader(&candidate_id, now);
                ElectionResult::Elected(candidate_id)
            }
            None => ElectionResult::NoCandidateAvailable,
        }
    }

    // ── replica promotion ───────────────────────────────────────────────────

    /// Promote a specific replica to leader.
    ///
    /// Demotes the current leader (if any) and issues a new fencing token.
    pub fn promote_to_leader(&mut self, node_id: &str, now: u64) {
        // Demote old leader.
        if let Some(old_leader) = &self.current_leader.clone() {
            if let Some(record) = self.nodes.get_mut(old_leader.as_str()) {
                record.is_leader = false;
                record.is_replica = true;
            }
        }

        // Increment fencing token.
        self.fencing_token_counter += 1;
        let token = self.fencing_token_counter;

        // Promote new leader.
        if let Some(record) = self.nodes.get_mut(node_id) {
            record.is_leader = true;
            record.is_replica = false;
            record.fencing_token = token;
        }
        self.current_leader = Some(node_id.to_string());

        self.event_log.push(FailoverEvent::new(
            node_id,
            FailoverEventKind::ReplicaPromoted,
            now,
            Some(format!("fencing_token={token}")),
        ));
    }

    // ── split-brain prevention ──────────────────────────────────────────────

    /// Check whether a node's fencing token is stale.
    ///
    /// A stale token indicates a split-brain scenario where two nodes believe
    /// they are the leader.
    pub fn validate_fencing_token(&mut self, node_id: &str, token: u64, now: u64) -> bool {
        if token < self.fencing_token_counter {
            // Stale token detected — fence the node.
            self.event_log.push(FailoverEvent::new(
                node_id,
                FailoverEventKind::SplitBrainFenced,
                now,
                Some(format!(
                    "stale token {token} < current {}",
                    self.fencing_token_counter
                )),
            ));
            if let Some(record) = self.nodes.get_mut(node_id) {
                record.is_leader = false;
                record.is_replica = true;
            }
            if self.current_leader.as_deref() == Some(node_id) {
                self.current_leader = None;
            }
            return false;
        }
        true
    }

    /// Check whether quorum is currently met among alive nodes.
    pub fn has_quorum(&self, now: u64) -> bool {
        let timeout = self.config.heartbeat_timeout_ms;
        let alive = self
            .nodes
            .values()
            .filter(|n| now.saturating_sub(n.last_heartbeat) <= timeout)
            .count();
        alive >= self.config.quorum_size
    }

    // ── graceful failover ───────────────────────────────────────────────────

    /// Initiate a graceful failover: drain connections from the current leader
    /// and transfer leadership to the specified node.
    ///
    /// Returns `true` if the transfer was completed.
    pub fn graceful_failover(&mut self, target_id: &str, now: u64) -> bool {
        let old_leader = match &self.current_leader {
            Some(id) => id.clone(),
            None => return false,
        };

        // Target must be registered and a replica.
        let target_is_replica = self
            .nodes
            .get(target_id)
            .map(|n| n.is_replica)
            .unwrap_or(false);
        if !target_is_replica {
            return false;
        }

        // "Drain" the old leader: reset connections.
        if let Some(record) = self.nodes.get_mut(&old_leader) {
            record.active_connections = 0;
        }

        self.promote_to_leader(target_id, now);

        self.event_log.push(FailoverEvent::new(
            target_id,
            FailoverEventKind::GracefulTransfer,
            now,
            Some(format!("from {old_leader}")),
        ));

        true
    }

    // ── automatic failover cycle ────────────────────────────────────────────

    /// Run a single failover check cycle.
    ///
    /// 1. Detect heartbeat failures.
    /// 2. If the leader has failed and policy is automatic, trigger election.
    ///
    /// Returns the election result if one was triggered, or `None`.
    pub fn check_cycle(&mut self, now: u64) -> Option<ElectionResult> {
        let failed = self.detect_failures(now);
        let leader_failed = self
            .current_leader
            .as_ref()
            .map(|id| failed.contains(id))
            .unwrap_or(false);

        if leader_failed && self.config.policy == FailoverPolicy::Automatic {
            Some(self.trigger_election(now))
        } else {
            None
        }
    }

    // ── history queries ─────────────────────────────────────────────────────

    /// Return all events of a specific kind.
    pub fn events_by_kind(&self, kind: &FailoverEventKind) -> Vec<&FailoverEvent> {
        self.event_log.iter().filter(|e| &e.kind == kind).collect()
    }

    /// Return all events for a specific node.
    pub fn events_for_node(&self, node_id: &str) -> Vec<&FailoverEvent> {
        self.event_log
            .iter()
            .filter(|e| e.node_id == node_id)
            .collect()
    }

    /// Return the count of events of a specific kind.
    pub fn event_kind_count(&self, kind: &FailoverEventKind) -> usize {
        self.event_log.iter().filter(|e| &e.kind == kind).count()
    }

    /// Return all alive replicas (healthy, within heartbeat timeout).
    pub fn alive_replicas(&self, now: u64) -> Vec<&NodeHealthRecord> {
        let timeout = self.config.heartbeat_timeout_ms;
        self.nodes
            .values()
            .filter(|n| {
                n.is_replica
                    && now.saturating_sub(n.last_heartbeat) <= timeout
                    && n.health_score >= self.config.health_threshold
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> FailoverController {
        let config = FailoverConfig {
            heartbeat_timeout_ms: 5000,
            max_consecutive_failures: 3,
            health_threshold: 0.3,
            policy: FailoverPolicy::Automatic,
            quorum_size: 2,
        };
        FailoverController::new(config)
    }

    fn add_leader(ctrl: &mut FailoverController, id: &str, now: u64) {
        ctrl.register_node(NodeHealthRecord::leader(id, now, 0));
        ctrl.fencing_token_counter = 1;
    }

    fn add_replica(ctrl: &mut FailoverController, id: &str, now: u64) {
        ctrl.register_node(NodeHealthRecord::new(id, now));
    }

    // ── basics ──────────────────────────────────────────────────────────────

    #[test]
    fn test_new_controller_empty() {
        let ctrl = make_controller();
        assert_eq!(ctrl.node_count(), 0);
        assert!(ctrl.current_leader().is_none());
    }

    #[test]
    fn test_register_node() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        assert_eq!(ctrl.node_count(), 1);
        assert!(ctrl.node("node1").is_some());
    }

    #[test]
    fn test_register_leader() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "leader1", 0);
        assert_eq!(ctrl.current_leader(), Some("leader1"));
    }

    #[test]
    fn test_unregister_node() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        assert!(ctrl.unregister_node("node1"));
        assert_eq!(ctrl.node_count(), 0);
    }

    #[test]
    fn test_unregister_unknown_returns_false() {
        let mut ctrl = make_controller();
        assert!(!ctrl.unregister_node("ghost"));
    }

    #[test]
    fn test_unregister_leader_clears_leader() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "leader1", 0);
        ctrl.unregister_node("leader1");
        assert!(ctrl.current_leader().is_none());
    }

    // ── heartbeat ───────────────────────────────────────────────────────────

    #[test]
    fn test_heartbeat_updates_timestamp() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        assert!(ctrl.heartbeat("node1", 1000));
        let record = ctrl.node("node1").expect("should exist");
        assert_eq!(record.last_heartbeat, 1000);
    }

    #[test]
    fn test_heartbeat_resets_failures() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        // Simulate some failures.
        if let Some(r) = ctrl.nodes.get_mut("node1") {
            r.consecutive_failures = 5;
        }
        ctrl.heartbeat("node1", 1000);
        let record = ctrl.node("node1").expect("should exist");
        assert_eq!(record.consecutive_failures, 0);
    }

    #[test]
    fn test_heartbeat_unknown_returns_false() {
        let mut ctrl = make_controller();
        assert!(!ctrl.heartbeat("ghost", 100));
    }

    // ── health score ────────────────────────────────────────────────────────

    #[test]
    fn test_update_health_score() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.update_health_score("node1", 0.5);
        let record = ctrl.node("node1").expect("should exist");
        assert!((record.health_score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_health_score_clamped() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.update_health_score("node1", 1.5);
        assert!((ctrl.node("node1").expect("exist").health_score - 1.0).abs() < f64::EPSILON);
        ctrl.update_health_score("node1", -0.5);
        assert!(ctrl.node("node1").expect("exist").health_score.abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_health_unknown_returns_false() {
        let mut ctrl = make_controller();
        assert!(!ctrl.update_health_score("ghost", 0.5));
    }

    // ── failure detection ───────────────────────────────────────────────────

    #[test]
    fn test_detect_failures_increments_counter() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        // node1 heartbeat at 0; check at 10000 => elapsed > 5000.
        ctrl.detect_failures(10_000);
        let record = ctrl.node("node1").expect("should exist");
        assert_eq!(record.consecutive_failures, 1);
    }

    #[test]
    fn test_detect_failures_returns_nodes_above_threshold() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        // Need 3 consecutive failures.
        ctrl.detect_failures(10_000); // 1
        ctrl.detect_failures(20_000); // 2
        let failed = ctrl.detect_failures(30_000); // 3
        assert!(failed.contains(&"node1".to_string()));
    }

    #[test]
    fn test_detect_failures_skips_recent_heartbeat() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.heartbeat("node1", 9000);
        let failed = ctrl.detect_failures(10_000); // elapsed = 1000 < 5000
        assert!(failed.is_empty());
    }

    #[test]
    fn test_detect_failures_logs_events() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.detect_failures(10_000);
        ctrl.detect_failures(20_000);
        ctrl.detect_failures(30_000);
        let events = ctrl.events_by_kind(&FailoverEventKind::FailureDetected);
        assert_eq!(events.len(), 1);
    }

    // ── health breach detection ─────────────────────────────────────────────

    #[test]
    fn test_detect_health_breaches() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.update_health_score("node1", 0.1);
        let breached = ctrl.detect_health_breaches(100);
        assert!(breached.contains(&"node1".to_string()));
    }

    #[test]
    fn test_detect_health_breaches_skips_healthy() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.update_health_score("node1", 0.8);
        let breached = ctrl.detect_health_breaches(100);
        assert!(breached.is_empty());
    }

    // ── election ────────────────────────────────────────────────────────────

    #[test]
    fn test_election_elects_best_replica() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "leader", 0);
        add_replica(&mut ctrl, "r1", 0);
        add_replica(&mut ctrl, "r2", 0);
        ctrl.update_health_score("r1", 0.7);
        ctrl.update_health_score("r2", 0.9);
        ctrl.heartbeat("r1", 100);
        ctrl.heartbeat("r2", 100);
        let result = ctrl.trigger_election(200);
        assert_eq!(result, ElectionResult::Elected("r2".to_string()));
        assert_eq!(ctrl.current_leader(), Some("r2"));
    }

    #[test]
    fn test_election_no_candidate() {
        let mut ctrl = FailoverController::new(FailoverConfig {
            quorum_size: 1,
            ..FailoverConfig::default()
        });
        add_leader(&mut ctrl, "leader", 0);
        // Leader is alive but not a replica; no other replicas registered.
        ctrl.heartbeat("leader", 100);
        let result = ctrl.trigger_election(200);
        assert_eq!(result, ElectionResult::NoCandidateAvailable);
    }

    #[test]
    fn test_election_quorum_not_met() {
        let mut ctrl = FailoverController::new(FailoverConfig {
            quorum_size: 5,
            ..FailoverConfig::default()
        });
        add_leader(&mut ctrl, "leader", 0);
        add_replica(&mut ctrl, "r1", 0);
        ctrl.heartbeat("leader", 100);
        ctrl.heartbeat("r1", 100);
        // Only 2 alive, quorum = 5.
        let result = ctrl.trigger_election(200);
        assert_eq!(result, ElectionResult::QuorumNotMet);
    }

    #[test]
    fn test_election_logs_events() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "leader", 0);
        add_replica(&mut ctrl, "r1", 0);
        ctrl.heartbeat("leader", 100);
        ctrl.heartbeat("r1", 100);
        ctrl.trigger_election(200);
        let election_events = ctrl.events_by_kind(&FailoverEventKind::ElectionTriggered);
        assert_eq!(election_events.len(), 1);
    }

    // ── promote_to_leader ───────────────────────────────────────────────────

    #[test]
    fn test_promote_demotes_old_leader() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "old_leader", 0);
        add_replica(&mut ctrl, "new_leader", 0);
        ctrl.promote_to_leader("new_leader", 100);
        let old = ctrl.node("old_leader").expect("should exist");
        assert!(!old.is_leader);
        assert!(old.is_replica);
    }

    #[test]
    fn test_promote_sets_fencing_token() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.promote_to_leader("node1", 100);
        let record = ctrl.node("node1").expect("should exist");
        assert!(record.fencing_token > 0);
    }

    #[test]
    fn test_promote_increments_fencing_token() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "n1", 0);
        add_replica(&mut ctrl, "n2", 0);
        ctrl.promote_to_leader("n1", 100);
        let t1 = ctrl.fencing_token();
        ctrl.promote_to_leader("n2", 200);
        let t2 = ctrl.fencing_token();
        assert!(t2 > t1);
    }

    // ── split-brain prevention ──────────────────────────────────────────────

    #[test]
    fn test_validate_fencing_token_valid() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.promote_to_leader("node1", 100);
        let token = ctrl.fencing_token();
        assert!(ctrl.validate_fencing_token("node1", token, 200));
    }

    #[test]
    fn test_validate_fencing_token_stale() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "n1", 0);
        add_replica(&mut ctrl, "n2", 0);
        ctrl.promote_to_leader("n1", 100);
        let stale_token = ctrl.fencing_token();
        ctrl.promote_to_leader("n2", 200);
        // n1 presents stale token.
        assert!(!ctrl.validate_fencing_token("n1", stale_token, 300));
        let fenced_events = ctrl.events_by_kind(&FailoverEventKind::SplitBrainFenced);
        assert_eq!(fenced_events.len(), 1);
    }

    #[test]
    fn test_validate_stale_demotes_node() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "n1", 0);
        add_replica(&mut ctrl, "n2", 0);
        ctrl.promote_to_leader("n1", 100);
        let stale = ctrl.fencing_token();
        ctrl.promote_to_leader("n2", 200);
        // n1 tries to act as leader with stale token.
        ctrl.validate_fencing_token("n1", stale, 300);
        let n1 = ctrl.node("n1").expect("exists");
        assert!(!n1.is_leader);
    }

    // ── quorum check ────────────────────────────────────────────────────────

    #[test]
    fn test_has_quorum_true() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "n1", 0);
        add_replica(&mut ctrl, "n2", 0);
        ctrl.heartbeat("n1", 100);
        ctrl.heartbeat("n2", 100);
        assert!(ctrl.has_quorum(200));
    }

    #[test]
    fn test_has_quorum_false() {
        let mut ctrl = FailoverController::new(FailoverConfig {
            quorum_size: 5,
            ..FailoverConfig::default()
        });
        add_replica(&mut ctrl, "n1", 0);
        ctrl.heartbeat("n1", 100);
        assert!(!ctrl.has_quorum(200));
    }

    // ── graceful failover ───────────────────────────────────────────────────

    #[test]
    fn test_graceful_failover_transfers_leadership() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "old", 0);
        add_replica(&mut ctrl, "new", 0);
        assert!(ctrl.graceful_failover("new", 100));
        assert_eq!(ctrl.current_leader(), Some("new"));
    }

    #[test]
    fn test_graceful_failover_drains_old_leader() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "old", 0);
        add_replica(&mut ctrl, "new", 0);
        ctrl.update_connections("old", 50);
        ctrl.graceful_failover("new", 100);
        let old = ctrl.node("old").expect("exists");
        assert_eq!(old.active_connections, 0);
    }

    #[test]
    fn test_graceful_failover_no_leader_returns_false() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "n1", 0);
        assert!(!ctrl.graceful_failover("n1", 100));
    }

    #[test]
    fn test_graceful_failover_non_replica_returns_false() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "leader", 0);
        // Register a non-replica node.
        let mut record = NodeHealthRecord::new("non_rep", 0);
        record.is_replica = false;
        ctrl.register_node(record);
        assert!(!ctrl.graceful_failover("non_rep", 100));
    }

    #[test]
    fn test_graceful_failover_logs_event() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "old", 0);
        add_replica(&mut ctrl, "new", 0);
        ctrl.graceful_failover("new", 100);
        let events = ctrl.events_by_kind(&FailoverEventKind::GracefulTransfer);
        assert_eq!(events.len(), 1);
    }

    // ── check_cycle ─────────────────────────────────────────────────────────

    #[test]
    fn test_check_cycle_triggers_election_on_leader_failure() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "leader", 0);
        add_replica(&mut ctrl, "r1", 0);
        ctrl.heartbeat("r1", 0);
        // Leader heartbeat stays at 0; r1 is alive.
        // Need 3 consecutive failures for leader: at 10k, 20k, 30k.
        ctrl.check_cycle(10_000);
        ctrl.check_cycle(20_000);
        let result = ctrl.check_cycle(30_000);
        assert!(result.is_some());
        if let Some(ElectionResult::Elected(id)) = result {
            assert_eq!(id, "r1");
        }
    }

    #[test]
    fn test_check_cycle_no_election_when_leader_alive() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "leader", 0);
        add_replica(&mut ctrl, "r1", 0);
        ctrl.heartbeat("leader", 100);
        ctrl.heartbeat("r1", 100);
        let result = ctrl.check_cycle(200);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_cycle_manual_policy_no_auto_election() {
        let mut ctrl = FailoverController::new(FailoverConfig {
            policy: FailoverPolicy::Manual,
            ..FailoverConfig::default()
        });
        add_leader(&mut ctrl, "leader", 0);
        add_replica(&mut ctrl, "r1", 0);
        ctrl.heartbeat("r1", 0);
        ctrl.check_cycle(10_000);
        ctrl.check_cycle(20_000);
        let result = ctrl.check_cycle(30_000);
        assert!(result.is_none());
    }

    // ── history queries ─────────────────────────────────────────────────────

    #[test]
    fn test_events_for_node() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.detect_failures(10_000);
        ctrl.detect_failures(20_000);
        ctrl.detect_failures(30_000);
        let events = ctrl.events_for_node("node1");
        assert!(!events.is_empty());
    }

    #[test]
    fn test_event_kind_count() {
        let mut ctrl = make_controller();
        add_leader(&mut ctrl, "leader", 0);
        add_replica(&mut ctrl, "r1", 0);
        ctrl.heartbeat("r1", 100);
        ctrl.trigger_election(200);
        assert_eq!(
            ctrl.event_kind_count(&FailoverEventKind::ElectionTriggered),
            1
        );
    }

    // ── alive_replicas ──────────────────────────────────────────────────────

    #[test]
    fn test_alive_replicas() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "r1", 0);
        add_replica(&mut ctrl, "r2", 0);
        ctrl.heartbeat("r1", 100);
        ctrl.heartbeat("r2", 100);
        ctrl.update_health_score("r1", 0.8);
        ctrl.update_health_score("r2", 0.1); // below threshold
        let alive = ctrl.alive_replicas(200);
        assert_eq!(alive.len(), 1);
        assert_eq!(alive[0].node_id, "r1");
    }

    // ── update_connections ──────────────────────────────────────────────────

    #[test]
    fn test_update_connections() {
        let mut ctrl = make_controller();
        add_replica(&mut ctrl, "node1", 0);
        ctrl.update_connections("node1", 42);
        assert_eq!(ctrl.node("node1").expect("exists").active_connections, 42);
    }

    #[test]
    fn test_update_connections_unknown_returns_false() {
        let mut ctrl = make_controller();
        assert!(!ctrl.update_connections("ghost", 10));
    }

    // ── config / defaults ───────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let config = FailoverConfig::default();
        assert_eq!(config.heartbeat_timeout_ms, 5000);
        assert_eq!(config.max_consecutive_failures, 3);
        assert_eq!(config.policy, FailoverPolicy::Automatic);
    }

    #[test]
    fn test_with_defaults_constructor() {
        let ctrl = FailoverController::with_defaults();
        assert_eq!(ctrl.config().heartbeat_timeout_ms, 5000);
    }

    #[test]
    fn test_default_failover_policy_is_automatic() {
        let policy = FailoverPolicy::default();
        assert_eq!(policy, FailoverPolicy::Automatic);
    }
}
