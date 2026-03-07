//! # Rolling Upgrade Orchestrator
//!
//! Provides zero-downtime rolling upgrade capabilities for OxiRS clusters.
//! Nodes are upgraded one at a time (or in configurable batch sizes), with
//! health checks between each step to ensure the cluster remains healthy.
//!
//! ## Upgrade Phases
//!
//! 1. **Pre-flight Check** - Validate cluster health and version compatibility
//! 2. **Drain** - Gracefully drain connections from the target node
//! 3. **Upgrade** - Apply the upgrade (binary swap, config reload, etc.)
//! 4. **Health Check** - Wait for the upgraded node to pass health checks
//! 5. **Rejoin** - Re-add the node to the active pool
//! 6. **Advance** - Move to the next node
//!
//! ## Rollback
//!
//! If a health check fails after upgrade, the orchestrator can automatically
//! roll back to the previous version and pause the upgrade.

use crate::error::Result;
use crate::raft::OxirsNodeId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// A semantic version for cluster nodes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ClusterVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ClusterVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with another (same major version).
    pub fn is_compatible(&self, other: &ClusterVersion) -> bool {
        self.major == other.major
    }

    /// Parse from string "M.m.p".
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts[2].parse().ok()?;
        Some(Self {
            major,
            minor,
            patch,
        })
    }
}

impl fmt::Display for ClusterVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ---------------------------------------------------------------------------
// Upgrade plan
// ---------------------------------------------------------------------------

/// Strategy for ordering node upgrades.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpgradeStrategy {
    /// Upgrade one node at a time.
    OneAtATime,
    /// Upgrade in batches of N.
    Batched(usize),
    /// Upgrade followers first, then leader last.
    FollowersFirst,
    /// Canary: upgrade one node, pause, then continue.
    Canary,
}

/// Configuration for a rolling upgrade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeConfig {
    /// Target version to upgrade to.
    pub target_version: ClusterVersion,
    /// Upgrade strategy.
    pub strategy: UpgradeStrategy,
    /// Maximum time to wait for a node to drain (seconds).
    pub drain_timeout_secs: u64,
    /// Maximum time to wait for a node to become healthy (seconds).
    pub health_check_timeout_secs: u64,
    /// Interval between health check probes (milliseconds).
    pub health_check_interval_ms: u64,
    /// Number of consecutive health checks required.
    pub required_healthy_checks: usize,
    /// Whether to automatically rollback on failure.
    pub auto_rollback: bool,
    /// Whether to pause after the canary node.
    pub pause_after_canary: bool,
    /// Maximum concurrent upgrades.
    pub max_concurrent: usize,
}

impl Default for UpgradeConfig {
    fn default() -> Self {
        Self {
            target_version: ClusterVersion::new(1, 1, 0),
            strategy: UpgradeStrategy::OneAtATime,
            drain_timeout_secs: 60,
            health_check_timeout_secs: 120,
            health_check_interval_ms: 5000,
            required_healthy_checks: 3,
            auto_rollback: true,
            pause_after_canary: true,
            max_concurrent: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Node upgrade state
// ---------------------------------------------------------------------------

/// The upgrade state of a single node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeUpgradeState {
    /// Not yet started.
    Pending,
    /// Draining connections.
    Draining,
    /// Upgrade in progress.
    Upgrading,
    /// Waiting for health checks.
    HealthChecking,
    /// Successfully upgraded.
    Completed,
    /// Upgrade failed.
    Failed,
    /// Rolled back to previous version.
    RolledBack,
    /// Skipped (e.g. already at target version).
    Skipped,
}

impl fmt::Display for NodeUpgradeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeUpgradeState::Pending => write!(f, "pending"),
            NodeUpgradeState::Draining => write!(f, "draining"),
            NodeUpgradeState::Upgrading => write!(f, "upgrading"),
            NodeUpgradeState::HealthChecking => write!(f, "health_checking"),
            NodeUpgradeState::Completed => write!(f, "completed"),
            NodeUpgradeState::Failed => write!(f, "failed"),
            NodeUpgradeState::RolledBack => write!(f, "rolled_back"),
            NodeUpgradeState::Skipped => write!(f, "skipped"),
        }
    }
}

/// Status of a single node's upgrade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeUpgradeStatus {
    /// Node ID.
    pub node_id: OxirsNodeId,
    /// Current state.
    pub state: NodeUpgradeState,
    /// Version before upgrade.
    pub from_version: ClusterVersion,
    /// Version after upgrade (or target).
    pub to_version: ClusterVersion,
    /// When the upgrade started for this node.
    pub started_at: Option<SystemTime>,
    /// When the upgrade completed for this node.
    pub completed_at: Option<SystemTime>,
    /// Number of health checks passed.
    pub health_checks_passed: usize,
    /// Error message if failed.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Upgrade progress
// ---------------------------------------------------------------------------

/// Overall upgrade state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpgradePhase {
    /// Not started.
    NotStarted,
    /// Pre-flight checks.
    PreFlight,
    /// Upgrade in progress.
    InProgress,
    /// Paused (e.g. after canary).
    Paused,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed,
    /// Cancelled by operator.
    Cancelled,
}

impl fmt::Display for UpgradePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UpgradePhase::NotStarted => write!(f, "not_started"),
            UpgradePhase::PreFlight => write!(f, "pre_flight"),
            UpgradePhase::InProgress => write!(f, "in_progress"),
            UpgradePhase::Paused => write!(f, "paused"),
            UpgradePhase::Completed => write!(f, "completed"),
            UpgradePhase::Failed => write!(f, "failed"),
            UpgradePhase::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Overall upgrade progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeProgress {
    /// Current phase.
    pub phase: UpgradePhase,
    /// Total nodes to upgrade.
    pub total_nodes: usize,
    /// Nodes completed.
    pub completed_nodes: usize,
    /// Nodes failed.
    pub failed_nodes: usize,
    /// Nodes skipped.
    pub skipped_nodes: usize,
    /// When the upgrade started.
    pub started_at: Option<SystemTime>,
    /// Per-node status.
    pub node_statuses: Vec<NodeUpgradeStatus>,
}

impl UpgradeProgress {
    /// Completion percentage.
    pub fn percentage(&self) -> f64 {
        if self.total_nodes == 0 {
            return 100.0;
        }
        let done = self.completed_nodes + self.skipped_nodes;
        (done as f64 / self.total_nodes as f64) * 100.0
    }

    /// Whether the upgrade is still running.
    pub fn is_active(&self) -> bool {
        matches!(
            self.phase,
            UpgradePhase::PreFlight | UpgradePhase::InProgress
        )
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Rolling upgrade orchestrator.
pub struct RollingUpgradeOrchestrator {
    config: UpgradeConfig,
    /// Upgrade queue: nodes in order.
    upgrade_queue: Arc<RwLock<VecDeque<OxirsNodeId>>>,
    /// Per-node status.
    node_statuses: Arc<RwLock<HashMap<OxirsNodeId, NodeUpgradeStatus>>>,
    /// Current phase.
    phase: Arc<RwLock<UpgradePhase>>,
    /// Upgrade start time.
    started_at: Arc<RwLock<Option<Instant>>>,
}

impl RollingUpgradeOrchestrator {
    /// Create a new orchestrator.
    pub fn new(config: UpgradeConfig) -> Self {
        Self {
            config,
            upgrade_queue: Arc::new(RwLock::new(VecDeque::new())),
            node_statuses: Arc::new(RwLock::new(HashMap::new())),
            phase: Arc::new(RwLock::new(UpgradePhase::NotStarted)),
            started_at: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize the upgrade for a set of nodes with their current versions.
    pub async fn initialize(&self, nodes: Vec<(OxirsNodeId, ClusterVersion)>) -> Result<()> {
        let mut queue = self.upgrade_queue.write().await;
        let mut statuses = self.node_statuses.write().await;

        queue.clear();
        statuses.clear();

        for (node_id, current_version) in nodes {
            let state = if current_version == self.config.target_version {
                NodeUpgradeState::Skipped
            } else {
                NodeUpgradeState::Pending
            };

            statuses.insert(
                node_id,
                NodeUpgradeStatus {
                    node_id,
                    state,
                    from_version: current_version,
                    to_version: self.config.target_version.clone(),
                    started_at: None,
                    completed_at: None,
                    health_checks_passed: 0,
                    error: None,
                },
            );

            if state == NodeUpgradeState::Pending {
                queue.push_back(node_id);
            }
        }

        *self.phase.write().await = UpgradePhase::PreFlight;
        *self.started_at.write().await = Some(Instant::now());

        info!(
            "Rolling upgrade initialized: {} nodes to upgrade to {}",
            queue.len(),
            self.config.target_version
        );

        Ok(())
    }

    /// Run pre-flight checks.
    pub async fn preflight_check(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();
        let statuses = self.node_statuses.read().await;

        // Check version compatibility
        for status in statuses.values() {
            if !status
                .from_version
                .is_compatible(&self.config.target_version)
            {
                issues.push(format!(
                    "Node {}: version {} is not compatible with target {}",
                    status.node_id, status.from_version, self.config.target_version
                ));
            }
        }

        // Check minimum cluster size
        let pending_count = statuses
            .values()
            .filter(|s| s.state == NodeUpgradeState::Pending)
            .count();
        let total_count = statuses.len();

        if total_count < 2 && pending_count > 0 {
            issues.push(
                "Single-node cluster: cannot guarantee availability during upgrade".to_string(),
            );
        }

        if issues.is_empty() {
            *self.phase.write().await = UpgradePhase::InProgress;
        }

        Ok(issues)
    }

    /// Get the next node to upgrade.
    pub async fn next_node(&self) -> Option<OxirsNodeId> {
        let queue = self.upgrade_queue.read().await;
        queue.front().copied()
    }

    /// Start draining a node.
    pub async fn start_drain(&self, node_id: OxirsNodeId) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            status.state = NodeUpgradeState::Draining;
            status.started_at = Some(SystemTime::now());
            info!("Node {}: draining started", node_id);
        }
        Ok(())
    }

    /// Mark a node as upgrading.
    pub async fn start_upgrade(&self, node_id: OxirsNodeId) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            status.state = NodeUpgradeState::Upgrading;
            info!("Node {}: upgrade started", node_id);
        }
        Ok(())
    }

    /// Start health checking a node.
    pub async fn start_health_check(&self, node_id: OxirsNodeId) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            status.state = NodeUpgradeState::HealthChecking;
            status.health_checks_passed = 0;
            info!("Node {}: health checking started", node_id);
        }
        Ok(())
    }

    /// Record a passed health check.
    pub async fn record_health_check(&self, node_id: OxirsNodeId, passed: bool) -> Result<bool> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            if passed {
                status.health_checks_passed += 1;
                debug!(
                    "Node {}: health check {}/{}",
                    node_id, status.health_checks_passed, self.config.required_healthy_checks
                );

                if status.health_checks_passed >= self.config.required_healthy_checks {
                    return Ok(true); // Node is healthy
                }
            } else {
                status.health_checks_passed = 0;
                warn!("Node {}: health check failed, resetting counter", node_id);
            }
        }
        Ok(false)
    }

    /// Mark a node as completed.
    pub async fn complete_node(&self, node_id: OxirsNodeId) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            status.state = NodeUpgradeState::Completed;
            status.completed_at = Some(SystemTime::now());
            info!("Node {}: upgrade completed", node_id);
        }

        // Remove from queue
        let mut queue = self.upgrade_queue.write().await;
        queue.retain(|&id| id != node_id);

        // Check if all done
        drop(statuses);
        self.check_completion().await;

        Ok(())
    }

    /// Mark a node as failed.
    pub async fn fail_node(&self, node_id: OxirsNodeId, error: String) -> Result<()> {
        let mut statuses = self.node_statuses.write().await;
        if let Some(status) = statuses.get_mut(&node_id) {
            if self.config.auto_rollback {
                status.state = NodeUpgradeState::RolledBack;
                warn!("Node {}: rolled back due to: {}", node_id, error);
            } else {
                status.state = NodeUpgradeState::Failed;
                warn!("Node {}: upgrade failed: {}", node_id, error);
            }
            status.error = Some(error);
            status.completed_at = Some(SystemTime::now());
        }

        // Pause the upgrade
        *self.phase.write().await = UpgradePhase::Paused;

        Ok(())
    }

    /// Resume a paused upgrade.
    pub async fn resume(&self) -> Result<()> {
        let phase = *self.phase.read().await;
        if phase == UpgradePhase::Paused {
            *self.phase.write().await = UpgradePhase::InProgress;
            info!("Rolling upgrade resumed");
        }
        Ok(())
    }

    /// Cancel the upgrade.
    pub async fn cancel(&self) -> Result<()> {
        *self.phase.write().await = UpgradePhase::Cancelled;
        info!("Rolling upgrade cancelled");
        Ok(())
    }

    /// Pause the upgrade.
    pub async fn pause(&self) -> Result<()> {
        *self.phase.write().await = UpgradePhase::Paused;
        info!("Rolling upgrade paused");
        Ok(())
    }

    /// Check if the upgrade is complete.
    async fn check_completion(&self) {
        let statuses = self.node_statuses.read().await;
        let all_done = statuses.values().all(|s| {
            matches!(
                s.state,
                NodeUpgradeState::Completed
                    | NodeUpgradeState::Skipped
                    | NodeUpgradeState::Failed
                    | NodeUpgradeState::RolledBack
            )
        });

        if all_done {
            let has_failures = statuses.values().any(|s| {
                matches!(
                    s.state,
                    NodeUpgradeState::Failed | NodeUpgradeState::RolledBack
                )
            });

            drop(statuses);

            if has_failures {
                *self.phase.write().await = UpgradePhase::Failed;
            } else {
                *self.phase.write().await = UpgradePhase::Completed;
            }
        }
    }

    /// Get current progress.
    pub async fn progress(&self) -> UpgradeProgress {
        let phase = *self.phase.read().await;
        let statuses = self.node_statuses.read().await;

        let total_nodes = statuses.len();
        let completed_nodes = statuses
            .values()
            .filter(|s| s.state == NodeUpgradeState::Completed)
            .count();
        let failed_nodes = statuses
            .values()
            .filter(|s| {
                matches!(
                    s.state,
                    NodeUpgradeState::Failed | NodeUpgradeState::RolledBack
                )
            })
            .count();
        let skipped_nodes = statuses
            .values()
            .filter(|s| s.state == NodeUpgradeState::Skipped)
            .count();

        let started_at = self.started_at.read().await.map(|_| SystemTime::now());

        UpgradeProgress {
            phase,
            total_nodes,
            completed_nodes,
            failed_nodes,
            skipped_nodes,
            started_at,
            node_statuses: statuses.values().cloned().collect(),
        }
    }

    /// Get the upgrade configuration.
    pub fn config(&self) -> &UpgradeConfig {
        &self.config
    }

    /// Whether the upgrade should proceed to the canary check.
    pub async fn should_pause_for_canary(&self) -> bool {
        if !matches!(self.config.strategy, UpgradeStrategy::Canary) {
            return false;
        }
        if !self.config.pause_after_canary {
            return false;
        }
        let statuses = self.node_statuses.read().await;
        let completed = statuses
            .values()
            .filter(|s| s.state == NodeUpgradeState::Completed)
            .count();
        completed == 1
    }

    /// Get remaining nodes in the queue.
    pub async fn remaining_nodes(&self) -> usize {
        self.upgrade_queue.read().await.len()
    }

    /// Get current phase.
    pub async fn current_phase(&self) -> UpgradePhase {
        *self.phase.read().await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u32, minor: u32, patch: u32) -> ClusterVersion {
        ClusterVersion::new(major, minor, patch)
    }

    fn default_config() -> UpgradeConfig {
        UpgradeConfig {
            target_version: v(1, 1, 0),
            ..Default::default()
        }
    }

    // --- ClusterVersion ---

    #[test]
    fn test_version_new() {
        let v = ClusterVersion::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_display() {
        assert_eq!(v(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn test_version_parse() {
        let parsed = ClusterVersion::parse("1.2.3");
        assert_eq!(parsed, Some(v(1, 2, 3)));
    }

    #[test]
    fn test_version_parse_invalid() {
        assert!(ClusterVersion::parse("1.2").is_none());
        assert!(ClusterVersion::parse("a.b.c").is_none());
        assert!(ClusterVersion::parse("").is_none());
    }

    #[test]
    fn test_version_compatible() {
        assert!(v(1, 0, 0).is_compatible(&v(1, 1, 0)));
        assert!(v(1, 0, 0).is_compatible(&v(1, 99, 99)));
        assert!(!v(1, 0, 0).is_compatible(&v(2, 0, 0)));
    }

    #[test]
    fn test_version_ordering() {
        assert!(v(1, 0, 0) < v(1, 1, 0));
        assert!(v(1, 1, 0) < v(1, 1, 1));
        assert!(v(1, 1, 1) < v(2, 0, 0));
    }

    // --- UpgradeConfig ---

    #[test]
    fn test_config_default() {
        let config = UpgradeConfig::default();
        assert_eq!(config.strategy, UpgradeStrategy::OneAtATime);
        assert_eq!(config.drain_timeout_secs, 60);
        assert_eq!(config.required_healthy_checks, 3);
        assert!(config.auto_rollback);
    }

    // --- NodeUpgradeState ---

    #[test]
    fn test_node_state_display() {
        assert_eq!(NodeUpgradeState::Pending.to_string(), "pending");
        assert_eq!(NodeUpgradeState::Draining.to_string(), "draining");
        assert_eq!(NodeUpgradeState::Completed.to_string(), "completed");
        assert_eq!(NodeUpgradeState::Failed.to_string(), "failed");
    }

    // --- UpgradePhase ---

    #[test]
    fn test_phase_display() {
        assert_eq!(UpgradePhase::NotStarted.to_string(), "not_started");
        assert_eq!(UpgradePhase::InProgress.to_string(), "in_progress");
        assert_eq!(UpgradePhase::Completed.to_string(), "completed");
    }

    // --- UpgradeProgress ---

    #[test]
    fn test_progress_percentage() {
        let progress = UpgradeProgress {
            phase: UpgradePhase::InProgress,
            total_nodes: 4,
            completed_nodes: 2,
            failed_nodes: 0,
            skipped_nodes: 1,
            started_at: None,
            node_statuses: vec![],
        };
        assert!((progress.percentage() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_percentage_empty() {
        let progress = UpgradeProgress {
            phase: UpgradePhase::Completed,
            total_nodes: 0,
            completed_nodes: 0,
            failed_nodes: 0,
            skipped_nodes: 0,
            started_at: None,
            node_statuses: vec![],
        };
        assert!((progress.percentage() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_is_active() {
        let active = UpgradeProgress {
            phase: UpgradePhase::InProgress,
            total_nodes: 1,
            completed_nodes: 0,
            failed_nodes: 0,
            skipped_nodes: 0,
            started_at: None,
            node_statuses: vec![],
        };
        assert!(active.is_active());

        let done = UpgradeProgress {
            phase: UpgradePhase::Completed,
            total_nodes: 1,
            completed_nodes: 1,
            failed_nodes: 0,
            skipped_nodes: 0,
            started_at: None,
            node_statuses: vec![],
        };
        assert!(!done.is_active());
    }

    // --- RollingUpgradeOrchestrator ---

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let phase = orch.current_phase().await;
        assert_eq!(phase, UpgradePhase::NotStarted);
    }

    #[tokio::test]
    async fn test_initialize_upgrade() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(1, 0, 0)), (2, v(1, 0, 0)), (3, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");

        let phase = orch.current_phase().await;
        assert_eq!(phase, UpgradePhase::PreFlight);
        assert_eq!(orch.remaining_nodes().await, 3);
    }

    #[tokio::test]
    async fn test_initialize_skips_up_to_date_nodes() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![
            (1, v(1, 0, 0)), // Needs upgrade
            (2, v(1, 1, 0)), // Already at target
            (3, v(1, 0, 0)), // Needs upgrade
        ];
        orch.initialize(nodes).await.expect("should succeed");
        assert_eq!(orch.remaining_nodes().await, 2);
    }

    #[tokio::test]
    async fn test_preflight_check_passes() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(1, 0, 0)), (2, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");

        let issues = orch.preflight_check().await.expect("should succeed");
        assert!(issues.is_empty());
        assert_eq!(orch.current_phase().await, UpgradePhase::InProgress);
    }

    #[tokio::test]
    async fn test_preflight_check_incompatible_version() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(2, 0, 0))]; // Major version mismatch
        orch.initialize(nodes).await.expect("should succeed");

        let issues = orch.preflight_check().await.expect("should succeed");
        assert!(!issues.is_empty());
    }

    #[tokio::test]
    async fn test_next_node() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(1, 0, 0)), (2, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");

        let next = orch.next_node().await;
        assert!(next.is_some());
        assert_eq!(next.expect("should have next"), 1);
    }

    #[tokio::test]
    async fn test_node_lifecycle_drain_upgrade_health_complete() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");
        orch.preflight_check().await.expect("should succeed");

        // Drain
        orch.start_drain(1).await.expect("should succeed");
        let progress = orch.progress().await;
        let node_status = progress
            .node_statuses
            .iter()
            .find(|s| s.node_id == 1)
            .expect("should find node");
        assert_eq!(node_status.state, NodeUpgradeState::Draining);

        // Upgrade
        orch.start_upgrade(1).await.expect("should succeed");

        // Health check
        orch.start_health_check(1).await.expect("should succeed");

        // Pass health checks
        for _ in 0..3 {
            orch.record_health_check(1, true)
                .await
                .expect("should succeed");
        }

        // Complete
        orch.complete_node(1).await.expect("should succeed");

        let progress = orch.progress().await;
        assert_eq!(progress.phase, UpgradePhase::Completed);
        assert_eq!(progress.completed_nodes, 1);
    }

    #[tokio::test]
    async fn test_failed_health_check_resets_counter() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");
        orch.preflight_check().await.expect("should succeed");

        orch.start_drain(1).await.expect("should succeed");
        orch.start_upgrade(1).await.expect("should succeed");
        orch.start_health_check(1).await.expect("should succeed");

        // Pass 2 checks
        orch.record_health_check(1, true)
            .await
            .expect("should succeed");
        orch.record_health_check(1, true)
            .await
            .expect("should succeed");

        // Fail one
        orch.record_health_check(1, false)
            .await
            .expect("should succeed");

        // Need 3 more consecutive passes now
        let healthy = orch
            .record_health_check(1, true)
            .await
            .expect("should succeed");
        assert!(!healthy);
    }

    #[tokio::test]
    async fn test_fail_node_with_rollback() {
        let config = UpgradeConfig {
            auto_rollback: true,
            ..default_config()
        };
        let orch = RollingUpgradeOrchestrator::new(config);
        let nodes = vec![(1, v(1, 0, 0)), (2, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");
        orch.preflight_check().await.expect("should succeed");

        orch.fail_node(1, "test error".to_string())
            .await
            .expect("should succeed");

        let progress = orch.progress().await;
        assert_eq!(progress.phase, UpgradePhase::Paused);

        let node = progress
            .node_statuses
            .iter()
            .find(|s| s.node_id == 1)
            .expect("should find node");
        assert_eq!(node.state, NodeUpgradeState::RolledBack);
        assert!(node.error.is_some());
    }

    #[tokio::test]
    async fn test_fail_node_without_rollback() {
        let config = UpgradeConfig {
            auto_rollback: false,
            ..default_config()
        };
        let orch = RollingUpgradeOrchestrator::new(config);
        let nodes = vec![(1, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");

        orch.fail_node(1, "test error".to_string())
            .await
            .expect("should succeed");

        let progress = orch.progress().await;
        let node = &progress.node_statuses[0];
        assert_eq!(node.state, NodeUpgradeState::Failed);
    }

    #[tokio::test]
    async fn test_resume_paused_upgrade() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(1, 0, 0)), (2, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");
        orch.preflight_check().await.expect("should succeed");

        orch.pause().await.expect("should succeed");
        assert_eq!(orch.current_phase().await, UpgradePhase::Paused);

        orch.resume().await.expect("should succeed");
        assert_eq!(orch.current_phase().await, UpgradePhase::InProgress);
    }

    #[tokio::test]
    async fn test_cancel_upgrade() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");

        orch.cancel().await.expect("should succeed");
        assert_eq!(orch.current_phase().await, UpgradePhase::Cancelled);
    }

    #[tokio::test]
    async fn test_canary_strategy_pause() {
        let config = UpgradeConfig {
            strategy: UpgradeStrategy::Canary,
            pause_after_canary: true,
            ..default_config()
        };
        let orch = RollingUpgradeOrchestrator::new(config);
        let nodes = vec![(1, v(1, 0, 0)), (2, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");
        orch.preflight_check().await.expect("should succeed");

        // Before completing any node
        assert!(!orch.should_pause_for_canary().await);

        // Complete canary node
        orch.start_drain(1).await.expect("should succeed");
        orch.start_upgrade(1).await.expect("should succeed");
        orch.start_health_check(1).await.expect("should succeed");
        for _ in 0..3 {
            orch.record_health_check(1, true)
                .await
                .expect("should succeed");
        }
        orch.complete_node(1).await.expect("should succeed");

        // Now should pause
        assert!(orch.should_pause_for_canary().await);
    }

    #[tokio::test]
    async fn test_config_access() {
        let config = UpgradeConfig {
            drain_timeout_secs: 120,
            ..default_config()
        };
        let orch = RollingUpgradeOrchestrator::new(config);
        assert_eq!(orch.config().drain_timeout_secs, 120);
    }

    #[tokio::test]
    async fn test_upgrade_full_cluster() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![(1, v(1, 0, 0)), (2, v(1, 0, 0)), (3, v(1, 0, 0))];
        orch.initialize(nodes).await.expect("should succeed");
        orch.preflight_check().await.expect("should succeed");

        for node_id in [1, 2, 3] {
            orch.start_drain(node_id).await.expect("should succeed");
            orch.start_upgrade(node_id).await.expect("should succeed");
            orch.start_health_check(node_id)
                .await
                .expect("should succeed");
            for _ in 0..3 {
                orch.record_health_check(node_id, true)
                    .await
                    .expect("should succeed");
            }
            orch.complete_node(node_id).await.expect("should succeed");
        }

        let progress = orch.progress().await;
        assert_eq!(progress.phase, UpgradePhase::Completed);
        assert_eq!(progress.completed_nodes, 3);
        assert!((progress.percentage() - 100.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_mixed_upgrade_complete_and_skip() {
        let orch = RollingUpgradeOrchestrator::new(default_config());
        let nodes = vec![
            (1, v(1, 0, 0)),
            (2, v(1, 1, 0)), // Already at target
            (3, v(1, 0, 0)),
        ];
        orch.initialize(nodes).await.expect("should succeed");
        orch.preflight_check().await.expect("should succeed");

        // Upgrade only nodes 1 and 3
        for node_id in [1, 3] {
            orch.start_drain(node_id).await.expect("should succeed");
            orch.start_upgrade(node_id).await.expect("should succeed");
            orch.start_health_check(node_id)
                .await
                .expect("should succeed");
            for _ in 0..3 {
                orch.record_health_check(node_id, true)
                    .await
                    .expect("should succeed");
            }
            orch.complete_node(node_id).await.expect("should succeed");
        }

        let progress = orch.progress().await;
        assert_eq!(progress.phase, UpgradePhase::Completed);
        assert_eq!(progress.completed_nodes, 2);
        assert_eq!(progress.skipped_nodes, 1);
    }
}
