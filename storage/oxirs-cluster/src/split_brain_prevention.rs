//! # Split-Brain Prevention
//!
//! Prevents split-brain scenarios in distributed clusters using quorum-based
//! decision making, fencing mechanisms, and generation numbers.

use crate::raft::OxirsNodeId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Split-brain prevention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitBrainConfig {
    /// Enable split-brain prevention
    pub enabled: bool,
    /// Quorum size (percentage of total nodes)
    pub quorum_percent: u8,
    /// Enable fencing for split nodes
    pub enable_fencing: bool,
    /// Enable witness nodes
    pub enable_witness_nodes: bool,
    /// Maximum allowed cluster splits
    pub max_allowed_splits: u32,
    /// Generation number increment on each leader election
    pub use_generation_numbers: bool,
    /// Timeout for fence acknowledgment
    pub fence_timeout_ms: u64,
}

impl Default for SplitBrainConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            quorum_percent: 51,
            enable_fencing: true,
            enable_witness_nodes: false,
            max_allowed_splits: 1,
            use_generation_numbers: true,
            fence_timeout_ms: 5000,
        }
    }
}

/// Generation number for detecting stale leaders
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GenerationNumber(pub u64);

impl GenerationNumber {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn increment(&mut self) {
        self.0 += 1;
    }

    pub fn is_newer_than(&self, other: &Self) -> bool {
        self.0 > other.0
    }
}

impl Default for GenerationNumber {
    fn default() -> Self {
        Self::new()
    }
}

/// Quorum decision
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuorumDecision {
    /// Quorum achieved
    Achieved {
        /// Number of votes
        votes: usize,
        /// Required quorum size
        required: usize,
    },
    /// Quorum not achieved
    NotAchieved {
        /// Number of votes
        votes: usize,
        /// Required quorum size
        required: usize,
    },
    /// Insufficient nodes for quorum
    InsufficientNodes {
        /// Total nodes
        total_nodes: usize,
        /// Minimum required
        minimum_required: usize,
    },
}

/// Fence status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FenceStatus {
    /// Node is not fenced
    NotFenced,
    /// Node is fenced
    Fenced,
    /// Fence pending
    FencePending,
    /// Fence failed
    FenceFailed,
}

/// Node fence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFence {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Fence status
    pub status: FenceStatus,
    /// Generation number when fenced
    pub fenced_generation: GenerationNumber,
    /// Timestamp when fenced
    pub fenced_at: SystemTime,
    /// Reason for fencing
    pub reason: String,
}

/// Split-brain detector and preventer
#[derive(Debug, Clone)]
pub struct SplitBrainPrevention {
    node_id: OxirsNodeId,
    config: SplitBrainConfig,
    generation: Arc<RwLock<GenerationNumber>>,
    fenced_nodes: Arc<RwLock<BTreeMap<OxirsNodeId, NodeFence>>>,
    cluster_nodes: Arc<RwLock<BTreeSet<OxirsNodeId>>>,
    witness_nodes: Arc<RwLock<BTreeSet<OxirsNodeId>>>,
    metrics: Arc<RwLock<SplitBrainMetrics>>,
}

/// Split-brain prevention metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitBrainMetrics {
    /// Total split-brain scenarios detected
    pub total_splits_detected: u64,
    /// Total split-brain scenarios prevented
    pub total_splits_prevented: u64,
    /// Total nodes fenced
    pub total_nodes_fenced: u64,
    /// Total fence operations
    pub total_fence_operations: u64,
    /// Successful fence operations
    pub successful_fences: u64,
    /// Failed fence operations
    pub failed_fences: u64,
    /// Quorum decisions made
    pub quorum_decisions: u64,
    /// Last split-brain detection
    pub last_split_detected: Option<SystemTime>,
    /// Last fence operation
    pub last_fence_operation: Option<SystemTime>,
}

impl SplitBrainPrevention {
    /// Create a new split-brain prevention instance
    pub fn new(node_id: OxirsNodeId, config: SplitBrainConfig) -> Self {
        Self {
            node_id,
            config,
            generation: Arc::new(RwLock::new(GenerationNumber::new())),
            fenced_nodes: Arc::new(RwLock::new(BTreeMap::new())),
            cluster_nodes: Arc::new(RwLock::new(BTreeSet::new())),
            witness_nodes: Arc::new(RwLock::new(BTreeSet::new())),
            metrics: Arc::new(RwLock::new(SplitBrainMetrics::default())),
        }
    }

    /// Register a cluster node
    pub async fn register_node(&self, node_id: OxirsNodeId) {
        let mut nodes = self.cluster_nodes.write().await;
        nodes.insert(node_id);

        tracing::debug!(
            "Node {}: Registered node {} for split-brain prevention",
            self.node_id,
            node_id
        );
    }

    /// Unregister a cluster node
    pub async fn unregister_node(&self, node_id: OxirsNodeId) {
        let mut nodes = self.cluster_nodes.write().await;
        nodes.remove(&node_id);

        tracing::debug!(
            "Node {}: Unregistered node {} from split-brain prevention",
            self.node_id,
            node_id
        );
    }

    /// Register a witness node
    pub async fn register_witness(&self, node_id: OxirsNodeId) {
        if !self.config.enable_witness_nodes {
            return;
        }

        let mut witnesses = self.witness_nodes.write().await;
        witnesses.insert(node_id);

        tracing::info!("Node {}: Registered witness node {}", self.node_id, node_id);
    }

    /// Get current generation number
    pub async fn get_generation(&self) -> GenerationNumber {
        *self.generation.read().await
    }

    /// Increment generation number (called on leader election)
    pub async fn increment_generation(&self) -> GenerationNumber {
        if !self.config.use_generation_numbers {
            return GenerationNumber::new();
        }

        let mut generation = self.generation.write().await;
        generation.increment();

        tracing::info!(
            "Node {}: Incremented generation number to {}",
            self.node_id,
            generation.0
        );

        *generation
    }

    /// Check if quorum is achieved
    pub async fn check_quorum(&self, available_nodes: &BTreeSet<OxirsNodeId>) -> QuorumDecision {
        let cluster_nodes = self.cluster_nodes.read().await;
        let witness_nodes = self.witness_nodes.read().await;

        // Total nodes includes cluster nodes + witness nodes
        let total_nodes = cluster_nodes.len() + witness_nodes.len() + 1; // +1 for self

        // Calculate required quorum
        let required_quorum =
            ((total_nodes as f64 * self.config.quorum_percent as f64) / 100.0).ceil() as usize;

        // Minimum nodes required for any quorum
        let min_nodes = 3; // Standard minimum for distributed systems

        if total_nodes < min_nodes {
            let mut metrics = self.metrics.write().await;
            metrics.quorum_decisions += 1;

            return QuorumDecision::InsufficientNodes {
                total_nodes,
                minimum_required: min_nodes,
            };
        }

        // Count available nodes (including self)
        let mut available_count = 1; // Self is always available

        for node_id in available_nodes {
            if cluster_nodes.contains(node_id) || witness_nodes.contains(node_id) {
                available_count += 1;
            }
        }

        let mut metrics = self.metrics.write().await;
        metrics.quorum_decisions += 1;

        if available_count >= required_quorum {
            tracing::info!(
                "Node {}: Quorum achieved ({}/{} nodes, required: {})",
                self.node_id,
                available_count,
                total_nodes,
                required_quorum
            );

            QuorumDecision::Achieved {
                votes: available_count,
                required: required_quorum,
            }
        } else {
            tracing::warn!(
                "Node {}: Quorum NOT achieved ({}/{} nodes, required: {})",
                self.node_id,
                available_count,
                total_nodes,
                required_quorum
            );

            QuorumDecision::NotAchieved {
                votes: available_count,
                required: required_quorum,
            }
        }
    }

    /// Detect split-brain scenario
    pub async fn detect_split_brain(
        &self,
        available_nodes: &BTreeSet<OxirsNodeId>,
        reported_leaders: &BTreeMap<OxirsNodeId, GenerationNumber>,
    ) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Check if multiple nodes claim to be leader
        if reported_leaders.len() > self.config.max_allowed_splits as usize {
            let mut metrics = self.metrics.write().await;
            metrics.total_splits_detected += 1;
            metrics.last_split_detected = Some(SystemTime::now());

            tracing::error!(
                "Node {}: Split-brain detected! {} leaders reported: {:?}",
                self.node_id,
                reported_leaders.len(),
                reported_leaders
            );

            return true;
        }

        // Check for stale leaders using generation numbers
        if self.config.use_generation_numbers {
            let current_gen = *self.generation.read().await;

            for (node_id, gen) in reported_leaders {
                if *gen > current_gen {
                    tracing::warn!(
                        "Node {}: Detected newer generation {} from node {} (current: {})",
                        self.node_id,
                        gen.0,
                        node_id,
                        current_gen.0
                    );
                } else if *gen < current_gen {
                    tracing::warn!(
                        "Node {}: Detected stale leader at node {} with generation {} (current: {})",
                        self.node_id,
                        node_id,
                        gen.0,
                        current_gen.0
                    );
                }
            }
        }

        // Check quorum
        let quorum_result = self.check_quorum(available_nodes).await;

        matches!(quorum_result, QuorumDecision::NotAchieved { .. })
    }

    /// Fence a node to prevent split-brain
    pub async fn fence_node(&self, node_id: OxirsNodeId, reason: String) -> Result<()> {
        if !self.config.enable_fencing {
            return Ok(());
        }

        let current_gen = *self.generation.read().await;

        let fence = NodeFence {
            node_id,
            status: FenceStatus::FencePending,
            fenced_generation: current_gen,
            fenced_at: SystemTime::now(),
            reason: reason.clone(),
        };

        {
            let mut fenced = self.fenced_nodes.write().await;
            fenced.insert(node_id, fence.clone());
        }

        let mut metrics = self.metrics.write().await;
        metrics.total_fence_operations += 1;
        metrics.last_fence_operation = Some(SystemTime::now());

        tracing::warn!(
            "Node {}: Fencing node {} (reason: {}, generation: {})",
            self.node_id,
            node_id,
            reason,
            current_gen.0
        );

        // Simulate fence operation (in production, this would involve network isolation)
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Update fence status
        let mut fenced = self.fenced_nodes.write().await;
        if let Some(fence) = fenced.get_mut(&node_id) {
            fence.status = FenceStatus::Fenced;

            metrics.successful_fences += 1;
            metrics.total_nodes_fenced += 1;
            metrics.total_splits_prevented += 1;

            tracing::info!(
                "Node {}: Successfully fenced node {}",
                self.node_id,
                node_id
            );
        }

        Ok(())
    }

    /// Unfence a node
    pub async fn unfence_node(&self, node_id: OxirsNodeId) -> Result<()> {
        let mut fenced = self.fenced_nodes.write().await;
        fenced.remove(&node_id);

        tracing::info!("Node {}: Unfenced node {}", self.node_id, node_id);

        Ok(())
    }

    /// Check if a node is fenced
    pub async fn is_fenced(&self, node_id: OxirsNodeId) -> bool {
        let fenced = self.fenced_nodes.read().await;
        fenced
            .get(&node_id)
            .map(|f| f.status == FenceStatus::Fenced)
            .unwrap_or(false)
    }

    /// Get all fenced nodes
    pub async fn get_fenced_nodes(&self) -> BTreeMap<OxirsNodeId, NodeFence> {
        self.fenced_nodes.read().await.clone()
    }

    /// Resolve split-brain scenario
    pub async fn resolve_split_brain(
        &self,
        _available_nodes: &BTreeSet<OxirsNodeId>,
        reported_leaders: &BTreeMap<OxirsNodeId, GenerationNumber>,
    ) -> Result<()> {
        if reported_leaders.is_empty() {
            return Ok(());
        }

        // Find the leader with the highest generation number
        let (newest_leader, newest_gen) = reported_leaders
            .iter()
            .max_by_key(|(_, gen)| *gen)
            .expect("reported_leaders validated to be non-empty");

        tracing::info!(
            "Node {}: Resolving split-brain, newest leader is {} with generation {}",
            self.node_id,
            newest_leader,
            newest_gen.0
        );

        // Fence all other leaders
        for (node_id, gen) in reported_leaders {
            if node_id != newest_leader && gen < newest_gen {
                self.fence_node(*node_id, format!("Stale leader with generation {}", gen.0))
                    .await?;
            }
        }

        // Update our generation to match the newest
        let mut current_gen = self.generation.write().await;
        if newest_gen.is_newer_than(&current_gen) {
            *current_gen = *newest_gen;
        }

        Ok(())
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> SplitBrainMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = SplitBrainMetrics::default();
    }

    /// Get cluster size
    pub async fn get_cluster_size(&self) -> usize {
        let cluster_nodes = self.cluster_nodes.read().await;
        let witness_nodes = self.witness_nodes.read().await;
        cluster_nodes.len() + witness_nodes.len() + 1 // +1 for self
    }

    /// Check if this node should step down based on quorum
    pub async fn should_step_down(&self, available_nodes: &BTreeSet<OxirsNodeId>) -> bool {
        let quorum = self.check_quorum(available_nodes).await;

        matches!(
            quorum,
            QuorumDecision::NotAchieved { .. } | QuorumDecision::InsufficientNodes { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_brain_config_default() {
        let config = SplitBrainConfig::default();
        assert!(config.enabled);
        assert_eq!(config.quorum_percent, 51);
        assert!(config.enable_fencing);
        assert!(!config.enable_witness_nodes);
        assert_eq!(config.max_allowed_splits, 1);
        assert!(config.use_generation_numbers);
    }

    #[test]
    fn test_generation_number() {
        let mut gen = GenerationNumber::new();
        assert_eq!(gen.0, 0);

        gen.increment();
        assert_eq!(gen.0, 1);

        let gen2 = GenerationNumber(2);
        assert!(gen2.is_newer_than(&gen));
        assert!(!gen.is_newer_than(&gen2));
    }

    #[tokio::test]
    async fn test_split_brain_prevention_creation() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        assert_eq!(sbp.node_id, 1);
        assert_eq!(sbp.get_generation().await, GenerationNumber(0));
    }

    #[tokio::test]
    async fn test_register_and_unregister_node() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        sbp.register_node(2).await;
        sbp.register_node(3).await;

        let size = sbp.get_cluster_size().await;
        assert_eq!(size, 3); // 1 (self) + 2 registered

        sbp.unregister_node(2).await;
        let size = sbp.get_cluster_size().await;
        assert_eq!(size, 2);
    }

    #[tokio::test]
    async fn test_increment_generation() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        let gen1 = sbp.increment_generation().await;
        assert_eq!(gen1.0, 1);

        let gen2 = sbp.increment_generation().await;
        assert_eq!(gen2.0, 2);
    }

    #[tokio::test]
    async fn test_quorum_achieved() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        sbp.register_node(2).await;
        sbp.register_node(3).await;
        sbp.register_node(4).await;
        sbp.register_node(5).await;

        // 5 total nodes, need 51% = 3 nodes
        let mut available = BTreeSet::new();
        available.insert(2);
        available.insert(3);

        let decision = sbp.check_quorum(&available).await;
        assert!(matches!(decision, QuorumDecision::Achieved { .. }));
    }

    #[tokio::test]
    async fn test_quorum_not_achieved() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        sbp.register_node(2).await;
        sbp.register_node(3).await;
        sbp.register_node(4).await;
        sbp.register_node(5).await;

        // 5 total nodes, need 51% = 3 nodes
        let mut available = BTreeSet::new();
        available.insert(2); // Only 2 nodes available (including self)

        let decision = sbp.check_quorum(&available).await;
        assert!(matches!(decision, QuorumDecision::NotAchieved { .. }));
    }

    #[tokio::test]
    async fn test_fence_and_unfence_node() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        assert!(!sbp.is_fenced(2).await);

        sbp.fence_node(2, "Test fencing".to_string()).await.unwrap();
        assert!(sbp.is_fenced(2).await);

        sbp.unfence_node(2).await.unwrap();
        assert!(!sbp.is_fenced(2).await);
    }

    #[tokio::test]
    async fn test_detect_split_brain() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        sbp.register_node(2).await;
        sbp.register_node(3).await;

        let available = BTreeSet::from([2, 3]);
        let mut leaders = BTreeMap::new();
        leaders.insert(1, GenerationNumber(1));
        leaders.insert(2, GenerationNumber(1)); // Two leaders!

        let is_split = sbp.detect_split_brain(&available, &leaders).await;
        assert!(is_split);

        let metrics = sbp.get_metrics().await;
        assert_eq!(metrics.total_splits_detected, 1);
    }

    #[tokio::test]
    async fn test_resolve_split_brain() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        sbp.register_node(2).await;
        sbp.register_node(3).await;

        let available = BTreeSet::from([2, 3]);
        let mut leaders = BTreeMap::new();
        leaders.insert(2, GenerationNumber(1));
        leaders.insert(3, GenerationNumber(2)); // Node 3 has newer generation

        sbp.resolve_split_brain(&available, &leaders).await.unwrap();

        // Node 2 should be fenced
        assert!(sbp.is_fenced(2).await);
        // Node 3 should not be fenced
        assert!(!sbp.is_fenced(3).await);

        // Our generation should be updated to match the newest
        assert_eq!(sbp.get_generation().await, GenerationNumber(2));
    }

    #[tokio::test]
    async fn test_witness_nodes() {
        let mut config = SplitBrainConfig::default();
        config.enable_witness_nodes = true;

        let sbp = SplitBrainPrevention::new(1, config);

        sbp.register_node(2).await;
        sbp.register_witness(10).await; // Witness node

        let size = sbp.get_cluster_size().await;
        assert_eq!(size, 3); // 1 (self) + 1 regular + 1 witness
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        sbp.register_node(2).await;
        sbp.register_node(3).await;

        // Trigger split-brain detection
        let available = BTreeSet::from([2, 3]);
        let mut leaders = BTreeMap::new();
        leaders.insert(1, GenerationNumber(1));
        leaders.insert(2, GenerationNumber(1));

        sbp.detect_split_brain(&available, &leaders).await;
        sbp.fence_node(2, "Test".to_string()).await.unwrap();

        let metrics = sbp.get_metrics().await;
        assert_eq!(metrics.total_splits_detected, 1);
        assert_eq!(metrics.total_fence_operations, 1);
        assert_eq!(metrics.successful_fences, 1);
        assert_eq!(metrics.total_nodes_fenced, 1);
        assert!(metrics.last_split_detected.is_some());
        assert!(metrics.last_fence_operation.is_some());
    }

    #[tokio::test]
    async fn test_should_step_down() {
        let config = SplitBrainConfig::default();
        let sbp = SplitBrainPrevention::new(1, config);

        sbp.register_node(2).await;
        sbp.register_node(3).await;
        sbp.register_node(4).await;

        // With only 1 node available, should step down
        let available = BTreeSet::from([2]);
        assert!(sbp.should_step_down(&available).await);

        // With quorum, should not step down
        let available = BTreeSet::from([2, 3]);
        assert!(!sbp.should_step_down(&available).await);
    }
}
