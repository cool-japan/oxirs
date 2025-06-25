//! # Consensus Protocol
//!
//! High-level consensus protocol implementation for distributed agreement.
//! Provides a simplified interface over the Raft implementation.

use crate::raft::{OxirsNodeId, RaftNode, RdfCommand, RdfResponse};
use anyhow::Result;
use std::collections::BTreeSet;

/// Consensus manager for distributed RDF operations
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
}

/// Status information for the consensus system
#[derive(Debug, Clone)]
pub struct ConsensusStatus {
    pub is_leader: bool,
    pub current_term: u64,
    pub peer_count: usize,
    pub triple_count: usize,
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
