//! # Consensus Protocol
//!
//! High-level consensus protocol implementation for distributed agreement.
//! Provides a simplified interface over the Raft implementation.

use crate::raft::{OxirsNodeId, RaftNode, RdfCommand, RdfResponse};
use anyhow::Result;
use std::collections::BTreeSet;

/// Consensus manager for distributed RDF operations
#[derive(Debug)]
pub struct ConsensusManager {
    raft_node: RaftNode,
    peers: BTreeSet<OxirsNodeId>,
}

impl ConsensusManager {
    /// Create a new consensus manager
    pub fn new(node_id: OxirsNodeId, peers: Vec<OxirsNodeId>) -> Self {
        Self {
            raft_node: RaftNode::new(node_id),
            peers: peers.into_iter().collect(),
        }
    }

    /// Initialize the consensus system
    #[cfg(feature = "raft")]
    pub async fn init(&mut self) -> Result<()> {
        self.raft_node.init_raft(self.peers.clone()).await?;
        tracing::info!(
            "Consensus manager initialized for node with {} peers",
            self.peers.len()
        );
        Ok(())
    }

    /// Initialize the consensus system (no-op for non-raft builds)
    #[cfg(not(feature = "raft"))]
    pub async fn init(&mut self) -> Result<()> {
        tracing::info!("Consensus manager initialized in single-node mode");
        Ok(())
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        self.raft_node.is_leader().await
    }

    /// Get current term
    pub async fn current_term(&self) -> u64 {
        self.raft_node.current_term().await
    }

    /// Propose an RDF command for consensus
    pub async fn propose_command(&self, command: RdfCommand) -> Result<RdfResponse> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!("Not the leader - cannot propose commands"));
        }

        let response = self.raft_node.submit_command(command).await?;
        Ok(response)
    }

    /// Insert a triple through consensus
    pub async fn insert_triple(
        &self,
        subject: String,
        predicate: String,
        object: String,
    ) -> Result<RdfResponse> {
        let command = RdfCommand::Insert {
            subject,
            predicate,
            object,
        };
        self.propose_command(command).await
    }

    /// Delete a triple through consensus
    pub async fn delete_triple(
        &self,
        subject: String,
        predicate: String,
        object: String,
    ) -> Result<RdfResponse> {
        let command = RdfCommand::Delete {
            subject,
            predicate,
            object,
        };
        self.propose_command(command).await
    }

    /// Clear all triples through consensus
    pub async fn clear_store(&self) -> Result<RdfResponse> {
        let command = RdfCommand::Clear;
        self.propose_command(command).await
    }

    /// Begin a distributed transaction
    pub async fn begin_transaction(&self, tx_id: String) -> Result<RdfResponse> {
        let command = RdfCommand::BeginTransaction { tx_id };
        self.propose_command(command).await
    }

    /// Commit a distributed transaction
    pub async fn commit_transaction(&self, tx_id: String) -> Result<RdfResponse> {
        let command = RdfCommand::CommitTransaction { tx_id };
        self.propose_command(command).await
    }

    /// Rollback a distributed transaction
    pub async fn rollback_transaction(&self, tx_id: String) -> Result<RdfResponse> {
        let command = RdfCommand::RollbackTransaction { tx_id };
        self.propose_command(command).await
    }

    /// Query the local replica (read operations don't need consensus)
    pub async fn query(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<(String, String, String)> {
        self.raft_node.query(subject, predicate, object).await
    }

    /// Get the number of triples in the store
    pub async fn len(&self) -> usize {
        self.raft_node.len().await
    }

    /// Check if the store is empty
    pub async fn is_empty(&self) -> bool {
        self.raft_node.is_empty().await
    }

    /// Get current peer set
    pub fn get_peers(&self) -> &BTreeSet<OxirsNodeId> {
        &self.peers
    }

    /// Add a peer to the cluster
    pub fn add_peer(&mut self, peer_id: OxirsNodeId) -> bool {
        if self.peers.insert(peer_id) {
            tracing::info!("Added peer {} to consensus manager", peer_id);
            true
        } else {
            false
        }
    }

    /// Remove a peer from the cluster
    pub fn remove_peer(&mut self, peer_id: OxirsNodeId) -> bool {
        if self.peers.remove(&peer_id) {
            tracing::info!("Removed peer {} from consensus manager", peer_id);
            true
        } else {
            false
        }
    }

    /// Get metrics from the underlying Raft node
    #[cfg(feature = "raft")]
    pub async fn get_metrics(
        &self,
    ) -> Option<openraft::RaftMetrics<OxirsNodeId, openraft::BasicNode>> {
        self.raft_node.get_metrics().await
    }

    /// Get cluster status summary
    pub async fn get_status(&self) -> ConsensusStatus {
        ConsensusStatus {
            is_leader: self.is_leader().await,
            current_term: self.current_term().await,
            peer_count: self.peers.len(),
            triple_count: self.len().await,
        }
    }

    /// Add a node to the cluster with consensus (joint consensus protocol)
    pub async fn add_node_with_consensus(
        &mut self,
        node_id: OxirsNodeId,
        address: String,
    ) -> Result<()> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot modify cluster configuration"
            ));
        }

        // Validate the node isn't already in the cluster
        if self.peers.contains(&node_id) {
            return Err(anyhow::anyhow!(
                "Node {} already exists in cluster",
                node_id
            ));
        }

        // Create configuration change command
        let command = RdfCommand::AddNode { node_id, address };

        // Submit through consensus
        let response = self.propose_command(command).await?;

        // Update local peer set on success
        if matches!(response, RdfResponse::Success) {
            self.add_peer(node_id);
            tracing::info!(
                "Successfully added node {} to cluster through consensus",
                node_id
            );
        }

        Ok(())
    }

    /// Remove a node from the cluster with consensus
    pub async fn remove_node_with_consensus(&mut self, node_id: OxirsNodeId) -> Result<()> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot modify cluster configuration"
            ));
        }

        // Validate the node exists in the cluster
        if !self.peers.contains(&node_id) {
            return Err(anyhow::anyhow!("Node {} not found in cluster", node_id));
        }

        // Create configuration change command
        let command = RdfCommand::RemoveNode { node_id };

        // Submit through consensus
        let response = self.propose_command(command).await?;

        // Update local peer set on success
        if matches!(response, RdfResponse::Success) {
            self.remove_peer(node_id);
            tracing::info!(
                "Successfully removed node {} from cluster through consensus",
                node_id
            );
        }

        Ok(())
    }

    /// Gracefully shutdown this node
    pub async fn graceful_shutdown(&mut self) -> Result<()> {
        tracing::info!("Initiating graceful shutdown of consensus manager");

        // If we're the leader, try to transfer leadership
        if self.is_leader().await && !self.peers.is_empty() {
            tracing::info!("Attempting leadership transfer before shutdown");

            // Find the best candidate (node with highest ID for simplicity)
            if let Some(&target_node) = self.peers.iter().max() {
                if let Err(e) = self.transfer_leadership(target_node).await {
                    tracing::warn!("Failed to transfer leadership: {}", e);
                }
            }
        }

        // Wait for any pending operations to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Signal shutdown to raft node
        self.raft_node.shutdown().await?;

        tracing::info!("Consensus manager shutdown completed");
        Ok(())
    }

    /// Transfer leadership to another node
    pub async fn transfer_leadership(&self, target_node: OxirsNodeId) -> Result<()> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot transfer leadership"
            ));
        }

        if !self.peers.contains(&target_node) {
            return Err(anyhow::anyhow!(
                "Target node {} not in cluster",
                target_node
            ));
        }

        // Submit leadership transfer command
        let command = RdfCommand::TransferLeadership { target_node };
        self.propose_command(command).await?;

        tracing::info!("Leadership transfer initiated to node {}", target_node);
        Ok(())
    }

    /// Force evict a non-responsive node
    pub async fn force_evict_node(&mut self, node_id: OxirsNodeId) -> Result<()> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!("Not the leader - cannot evict nodes"));
        }

        tracing::warn!("Force evicting non-responsive node {}", node_id);

        // Create force eviction command
        let command = RdfCommand::ForceEvictNode { node_id };

        // Submit through consensus
        let response = self.propose_command(command).await?;

        // Update local peer set on success
        if matches!(response, RdfResponse::Success) {
            self.remove_peer(node_id);
            tracing::info!("Successfully force evicted node {}", node_id);
        }

        Ok(())
    }

    /// Check health of peer nodes
    pub async fn check_peer_health(&self) -> Result<Vec<NodeHealthStatus>> {
        let mut health_statuses = Vec::new();

        for &peer_id in &self.peers {
            let health = self.check_single_node_health(peer_id).await;
            health_statuses.push(health);
        }

        Ok(health_statuses)
    }

    /// Check health of a single node
    async fn check_single_node_health(&self, node_id: OxirsNodeId) -> NodeHealthStatus {
        // Try to get metrics from the node
        let start_time = std::time::Instant::now();

        // In a real implementation, this would ping the actual node
        // For now, we'll simulate based on raft metrics
        #[cfg(feature = "raft")]
        let is_responsive = if let Some(_metrics) = self.raft_node.get_metrics().await {
            start_time.elapsed() < tokio::time::Duration::from_millis(1000)
        } else {
            false
        };

        #[cfg(not(feature = "raft"))]
        let is_responsive = start_time.elapsed() < tokio::time::Duration::from_millis(100);

        NodeHealthStatus {
            node_id,
            is_responsive,
            last_seen: if is_responsive {
                Some(std::time::SystemTime::now())
            } else {
                None
            },
            latency_ms: start_time.elapsed().as_millis() as u64,
        }
    }

    /// Attempt to recover from a partition or failure
    pub async fn attempt_recovery(&mut self) -> Result<()> {
        tracing::info!("Attempting cluster recovery");

        // Check if we have enough healthy nodes for quorum
        let health_statuses = self.check_peer_health().await?;
        let healthy_nodes: Vec<_> = health_statuses
            .iter()
            .filter(|status| status.is_responsive)
            .collect();

        let quorum_size = (self.peers.len() + 1) / 2 + 1; // +1 for self

        if healthy_nodes.len() + 1 >= quorum_size {
            tracing::info!("Sufficient nodes for quorum, attempting to re-establish consensus");

            // Re-initialize consensus with healthy nodes only
            let healthy_node_ids: std::collections::BTreeSet<_> =
                healthy_nodes.iter().map(|status| status.node_id).collect();

            self.peers = healthy_node_ids;
            self.init().await?;

            tracing::info!(
                "Recovery completed with {} healthy nodes",
                healthy_nodes.len()
            );
        } else {
            tracing::error!(
                "Insufficient nodes for quorum: {} healthy out of {} required",
                healthy_nodes.len() + 1,
                quorum_size
            );
            return Err(anyhow::anyhow!(
                "Cannot recover: insufficient nodes for quorum"
            ));
        }

        Ok(())
    }
}

/// Status information for the consensus system
#[derive(Debug, Clone)]
pub struct ConsensusStatus {
    pub is_leader: bool,
    pub current_term: u64,
    pub peer_count: usize,
    pub triple_count: usize,
}

/// Health status of a cluster node
#[derive(Debug, Clone)]
pub struct NodeHealthStatus {
    pub node_id: OxirsNodeId,
    pub is_responsive: bool,
    pub last_seen: Option<std::time::SystemTime>,
    pub latency_ms: u64,
}

/// Consensus error types
#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("Not the leader")]
    NotLeader,
    #[error("Command failed: {0}")]
    CommandFailed(String),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Timeout: {0}")]
    Timeout(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_manager_creation() {
        let peers = vec![2, 3, 4];
        let manager = ConsensusManager::new(1, peers.clone());

        assert_eq!(manager.get_peers().len(), 3);
        assert!(manager.get_peers().contains(&2));
        assert!(manager.get_peers().contains(&3));
        assert!(manager.get_peers().contains(&4));
    }

    #[test]
    fn test_consensus_manager_add_peer() {
        let mut manager = ConsensusManager::new(1, vec![2, 3]);

        assert!(manager.add_peer(4));
        assert_eq!(manager.get_peers().len(), 3);
        assert!(manager.get_peers().contains(&4));

        // Adding same peer again should return false
        assert!(!manager.add_peer(4));
        assert_eq!(manager.get_peers().len(), 3);
    }

    #[test]
    fn test_consensus_manager_remove_peer() {
        let mut manager = ConsensusManager::new(1, vec![2, 3, 4]);

        assert!(manager.remove_peer(3));
        assert_eq!(manager.get_peers().len(), 2);
        assert!(!manager.get_peers().contains(&3));

        // Removing non-existent peer should return false
        assert!(!manager.remove_peer(5));
        assert_eq!(manager.get_peers().len(), 2);
    }

    #[tokio::test]
    async fn test_consensus_manager_basic_operations() {
        let manager = ConsensusManager::new(1, vec![]);

        // In single-node mode, should be leader
        assert!(manager.is_leader().await);
        assert_eq!(manager.current_term().await, 0);
        assert_eq!(manager.len().await, 0);
        assert!(manager.is_empty().await);
    }

    #[tokio::test]
    async fn test_consensus_status() {
        let manager = ConsensusManager::new(1, vec![2, 3]);
        let status = manager.get_status().await;

        assert!(status.is_leader);
        assert_eq!(status.current_term, 0);
        assert_eq!(status.peer_count, 2);
        assert_eq!(status.triple_count, 0);
    }

    #[test]
    fn test_consensus_error_display() {
        assert_eq!(ConsensusError::NotLeader.to_string(), "Not the leader");

        assert_eq!(
            ConsensusError::CommandFailed("test".to_string()).to_string(),
            "Command failed: test"
        );

        assert_eq!(
            ConsensusError::Network("conn error".to_string()).to_string(),
            "Network error: conn error"
        );

        assert_eq!(
            ConsensusError::Storage("disk error".to_string()).to_string(),
            "Storage error: disk error"
        );

        assert_eq!(
            ConsensusError::Timeout("5s".to_string()).to_string(),
            "Timeout: 5s"
        );
    }
}
