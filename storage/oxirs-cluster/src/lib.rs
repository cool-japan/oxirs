//! # OxiRS Cluster
//!
//! Raft-backed distributed dataset for high availability and horizontal scaling.
//!
//! This crate provides distributed storage capabilities using Raft consensus,
//! enabling horizontal scaling and high availability for RDF datasets.
//!
//! ## Features
//!
//! - **Raft Consensus**: Production-ready Raft implementation using openraft
//! - **Distributed RDF Storage**: Scalable, consistent RDF triple storage
//! - **Automatic Failover**: Leader election and automatic recovery
//! - **Node Discovery**: Multiple discovery mechanisms (static, DNS, multicast)
//! - **Replication Management**: Configurable replication strategies
//! - **SPARQL Support**: Distributed SPARQL query execution
//! - **Transaction Support**: Distributed ACID transactions
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_cluster::{ClusterNode, NodeConfig};
//! use std::net::SocketAddr;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = NodeConfig {
//!     node_id: 1,
//!     address: "127.0.0.1:8080".parse()?,
//!     data_dir: "./data".to_string(),
//!     peers: vec![2, 3],
//!     discovery: None,
//!     replication_strategy: None,
//! };
//!
//! let mut node = ClusterNode::new(config).await?;
//! node.start().await?;
//!
//! // Insert data through consensus
//! node.insert_triple(
//!     "<http://example.org/subject>",
//!     "<http://example.org/predicate>",
//!     "\"object\"")
//! .await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod consensus;
pub mod discovery;
pub mod network;
pub mod raft;
pub mod replication;
pub mod storage;

#[cfg(feature = "bft")]
pub mod bft;

use consensus::{ConsensusManager, ConsensusStatus};
use discovery::{DiscoveryConfig, DiscoveryService, NodeInfo};
use raft::{OxirsNodeId, RdfCommand, RdfResponse};
use replication::{ReplicationManager, ReplicationStats, ReplicationStrategy};

/// Cluster node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Unique node identifier
    pub node_id: OxirsNodeId,
    /// Network address for communication
    pub address: SocketAddr,
    /// Data directory for persistent storage
    pub data_dir: String,
    /// List of peer node IDs
    pub peers: Vec<OxirsNodeId>,
    /// Discovery configuration
    pub discovery: Option<DiscoveryConfig>,
    /// Replication strategy
    pub replication_strategy: Option<ReplicationStrategy>,
}

impl NodeConfig {
    /// Create a new node configuration
    pub fn new(node_id: OxirsNodeId, address: SocketAddr) -> Self {
        Self {
            node_id,
            address,
            data_dir: format!("./data/node-{}", node_id),
            peers: Vec::new(),
            discovery: Some(DiscoveryConfig::default()),
            replication_strategy: Some(ReplicationStrategy::default()),
        }
    }

    /// Add a peer to the configuration
    pub fn add_peer(&mut self, peer_id: OxirsNodeId) -> &mut Self {
        if !self.peers.contains(&peer_id) && peer_id != self.node_id {
            self.peers.push(peer_id);
        }
        self
    }

    /// Set the discovery configuration
    pub fn with_discovery(mut self, discovery: DiscoveryConfig) -> Self {
        self.discovery = Some(discovery);
        self
    }

    /// Set the replication strategy
    pub fn with_replication_strategy(mut self, strategy: ReplicationStrategy) -> Self {
        self.replication_strategy = Some(strategy);
        self
    }
}

/// Cluster node implementation
#[derive(Debug)]
pub struct ClusterNode {
    config: NodeConfig,
    consensus: ConsensusManager,
    discovery: DiscoveryService,
    replication: ReplicationManager,
    running: Arc<RwLock<bool>>,
}

impl ClusterNode {
    /// Create a new cluster node
    pub async fn new(config: NodeConfig) -> Result<Self> {
        // Validate configuration
        if config.data_dir.is_empty() {
            return Err(anyhow::anyhow!("Data directory cannot be empty"));
        }

        // Create data directory if it doesn't exist
        tokio::fs::create_dir_all(&config.data_dir)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create data directory: {}", e))?;

        // Initialize consensus manager
        let consensus = ConsensusManager::new(config.node_id, config.peers.clone());

        // Initialize discovery service
        let discovery_config = config.discovery.clone().unwrap_or_default();
        let discovery = DiscoveryService::new(config.node_id, config.address, discovery_config);

        // Initialize replication manager
        let replication_strategy = config.replication_strategy.clone().unwrap_or_default();
        let replication = ReplicationManager::new(replication_strategy, config.node_id);

        Ok(Self {
            config,
            consensus,
            discovery,
            replication,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the cluster node
    pub async fn start(&mut self) -> Result<()> {
        {
            let mut running = self.running.write().await;
            if *running {
                return Ok(());
            }
            *running = true;
        }

        tracing::info!(
            "Starting cluster node {} at {} with {} peers",
            self.config.node_id,
            self.config.address,
            self.config.peers.len()
        );

        // Start discovery service
        self.discovery
            .start()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start discovery service: {}", e))?;

        // Discover initial nodes
        let discovered_nodes = self
            .discovery
            .discover_nodes()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to discover nodes: {}", e))?;

        // Add discovered nodes to replication manager
        for node in discovered_nodes {
            if node.node_id != self.config.node_id {
                self.replication
                    .add_replica(node.node_id, node.address.to_string());
            }
        }

        // Initialize consensus system
        self.consensus
            .init()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize consensus: {}", e))?;

        tracing::info!("Cluster node {} started successfully", self.config.node_id);

        // Start background tasks
        self.start_background_tasks().await;

        Ok(())
    }

    /// Stop the cluster node
    pub async fn stop(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }

        tracing::info!("Stopping cluster node {}", self.config.node_id);

        // Stop discovery service
        self.discovery
            .stop()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to stop discovery service: {}", e))?;

        *running = false;

        tracing::info!("Cluster node {} stopped", self.config.node_id);

        Ok(())
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        self.consensus.is_leader().await
    }

    /// Get current consensus term
    pub async fn current_term(&self) -> u64 {
        self.consensus.current_term().await
    }

    /// Insert a triple through distributed consensus
    pub async fn insert_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<RdfResponse> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot accept write operations"
            ));
        }

        let response = self
            .consensus
            .insert_triple(
                subject.to_string(),
                predicate.to_string(),
                object.to_string(),
            )
            .await?;

        Ok(response)
    }

    /// Delete a triple through distributed consensus
    pub async fn delete_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<RdfResponse> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot accept write operations"
            ));
        }

        let response = self
            .consensus
            .delete_triple(
                subject.to_string(),
                predicate.to_string(),
                object.to_string(),
            )
            .await?;

        Ok(response)
    }

    /// Clear all triples through distributed consensus
    pub async fn clear_store(&self) -> Result<RdfResponse> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot accept write operations"
            ));
        }

        let response = self.consensus.clear_store().await?;
        Ok(response)
    }

    /// Begin a distributed transaction
    pub async fn begin_transaction(&self) -> Result<String> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot begin transactions"
            ));
        }

        let tx_id = uuid::Uuid::new_v4().to_string();
        let _response = self.consensus.begin_transaction(tx_id.clone()).await?;

        Ok(tx_id)
    }

    /// Commit a distributed transaction
    pub async fn commit_transaction(&self, tx_id: &str) -> Result<RdfResponse> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot commit transactions"
            ));
        }

        let response = self.consensus.commit_transaction(tx_id.to_string()).await?;
        Ok(response)
    }

    /// Rollback a distributed transaction
    pub async fn rollback_transaction(&self, tx_id: &str) -> Result<RdfResponse> {
        if !self.is_leader().await {
            return Err(anyhow::anyhow!(
                "Not the leader - cannot rollback transactions"
            ));
        }

        let response = self
            .consensus
            .rollback_transaction(tx_id.to_string())
            .await?;
        Ok(response)
    }

    /// Query triples (can be done on any node)
    pub async fn query_triples(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<(String, String, String)> {
        self.consensus.query(subject, predicate, object).await
    }

    /// Execute SPARQL query (simplified interface)
    pub async fn query_sparql(&self, _sparql: &str) -> Result<Vec<String>> {
        // TODO: Implement full SPARQL query execution
        // For now, return all triples as a simple implementation
        let triples = self.query_triples(None, None, None).await;
        let results = triples
            .into_iter()
            .map(|(s, p, o)| format!("{} {} {} .", s, p, o))
            .collect();
        Ok(results)
    }

    /// Get the number of triples in the store
    pub async fn len(&self) -> usize {
        self.consensus.len().await
    }

    /// Check if the store is empty
    pub async fn is_empty(&self) -> bool {
        self.consensus.is_empty().await
    }

    /// Add a new node to the cluster
    pub async fn add_cluster_node(
        &mut self,
        node_id: OxirsNodeId,
        address: SocketAddr,
    ) -> Result<()> {
        if node_id == self.config.node_id {
            return Err(anyhow::anyhow!("Cannot add self to cluster"));
        }

        // Add to configuration
        self.config.add_peer(node_id);

        // Add to discovery
        let node_info = NodeInfo::new(node_id, address);
        self.discovery.add_node(node_info);

        // Add to replication
        self.replication.add_replica(node_id, address.to_string());

        // Add to consensus (this would trigger Raft membership change)
        self.consensus.add_peer(node_id);

        tracing::info!("Added node {} at {} to cluster", node_id, address);

        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_cluster_node(&mut self, node_id: OxirsNodeId) -> Result<()> {
        if node_id == self.config.node_id {
            return Err(anyhow::anyhow!("Cannot remove self from cluster"));
        }

        // Remove from configuration
        self.config.peers.retain(|&id| id != node_id);

        // Remove from discovery
        self.discovery.remove_node(node_id);

        // Remove from replication
        self.replication.remove_replica(node_id);

        // Remove from consensus (this would trigger Raft membership change)
        self.consensus.remove_peer(node_id);

        tracing::info!("Removed node {} from cluster", node_id);

        Ok(())
    }

    /// Get comprehensive cluster status
    pub async fn get_status(&self) -> ClusterStatus {
        let consensus_status = self.consensus.get_status().await;
        let discovery_stats = self.discovery.get_stats().clone();
        let replication_stats = self.replication.get_stats().clone();

        ClusterStatus {
            node_id: self.config.node_id,
            address: self.config.address,
            is_leader: consensus_status.is_leader,
            current_term: consensus_status.current_term,
            peer_count: consensus_status.peer_count,
            triple_count: consensus_status.triple_count,
            discovery_stats,
            replication_stats,
            is_running: *self.running.read().await,
        }
    }

    /// Start background maintenance tasks
    async fn start_background_tasks(&mut self) {
        let running = Arc::clone(&self.running);

        // Discovery and health check task
        let discovery_config = self.config.discovery.clone().unwrap_or_default();
        let mut discovery_clone =
            DiscoveryService::new(self.config.node_id, self.config.address, discovery_config);

        tokio::spawn(async move {
            while *running.read().await {
                discovery_clone.run_periodic_tasks().await;
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        });

        // Replication maintenance task
        let mut replication_clone = ReplicationManager::with_raft_consensus(self.config.node_id);
        let running_clone = Arc::clone(&self.running);

        tokio::spawn(async move {
            while *running_clone.read().await {
                replication_clone.run_maintenance().await;
                break; // run_maintenance() is infinite loop
            }
        });
    }
}

/// Comprehensive cluster status information
#[derive(Debug, Clone)]
pub struct ClusterStatus {
    /// Local node ID
    pub node_id: OxirsNodeId,
    /// Local node address
    pub address: SocketAddr,
    /// Whether this node is the current leader
    pub is_leader: bool,
    /// Current Raft term
    pub current_term: u64,
    /// Number of peer nodes
    pub peer_count: usize,
    /// Number of triples in the store
    pub triple_count: usize,
    /// Discovery service statistics
    pub discovery_stats: discovery::DiscoveryStats,
    /// Replication statistics
    pub replication_stats: ReplicationStats,
    /// Whether the node is currently running
    pub is_running: bool,
}

/// Distributed RDF store (simplified interface)
pub struct DistributedStore {
    node: ClusterNode,
}

impl DistributedStore {
    /// Create a new distributed store
    pub async fn new(config: NodeConfig) -> Result<Self> {
        let node = ClusterNode::new(config).await?;
        Ok(Self { node })
    }

    /// Start the distributed store
    pub async fn start(&mut self) -> Result<()> {
        self.node.start().await
    }

    /// Stop the distributed store
    pub async fn stop(&mut self) -> Result<()> {
        self.node.stop().await
    }

    /// Insert a triple (only on leader)
    pub async fn insert_triple(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<()> {
        let _response = self.node.insert_triple(subject, predicate, object).await?;
        Ok(())
    }

    /// Query triples using SPARQL
    pub async fn query_sparql(&self, sparql: &str) -> Result<Vec<String>> {
        self.node.query_sparql(sparql).await
    }

    /// Query triples by pattern
    pub async fn query_pattern(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<(String, String, String)> {
        self.node.query_triples(subject, predicate, object).await
    }

    /// Get cluster status
    pub async fn get_status(&self) -> ClusterStatus {
        self.node.get_status().await
    }
}

/// Re-export commonly used types
pub use consensus::ConsensusError;
pub use discovery::DiscoveryError;
pub use replication::ReplicationError;

/// Cluster-specific error types
#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Consensus error: {0}")]
    Consensus(#[from] ConsensusError),

    #[error("Discovery error: {0}")]
    Discovery(#[from] DiscoveryError),

    #[error("Replication error: {0}")]
    Replication(#[from] ReplicationError),

    #[error("Not the leader")]
    NotLeader,

    #[error("Node not found: {node_id}")]
    NodeNotFound { node_id: OxirsNodeId },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network error: {0}")]
    Network(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_node_config_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = NodeConfig::new(1, addr);

        assert_eq!(config.node_id, 1);
        assert_eq!(config.address, addr);
        assert_eq!(config.data_dir, "./data/node-1");
        assert!(config.peers.is_empty());
        assert!(config.discovery.is_some());
        assert!(config.replication_strategy.is_some());
    }

    #[tokio::test]
    async fn test_node_config_add_peer() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut config = NodeConfig::new(1, addr);

        config.add_peer(2);
        config.add_peer(3);
        config.add_peer(2); // Duplicate should be ignored

        assert_eq!(config.peers, vec![2, 3]);
    }

    #[tokio::test]
    async fn test_node_config_no_self_peer() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut config = NodeConfig::new(1, addr);

        config.add_peer(1); // Should not add self

        assert!(config.peers.is_empty());
    }

    #[tokio::test]
    async fn test_cluster_node_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = NodeConfig::new(1, addr);

        let node = ClusterNode::new(config).await;
        assert!(node.is_ok());

        let node = node.unwrap();
        assert_eq!(node.config.node_id, 1);
        assert_eq!(node.config.address, addr);
    }

    #[tokio::test]
    async fn test_cluster_node_empty_data_dir_error() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut config = NodeConfig::new(1, addr);
        config.data_dir = String::new();

        let result = ClusterNode::new(config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Data directory cannot be empty"));
    }

    #[tokio::test]
    async fn test_distributed_store_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = NodeConfig::new(1, addr);

        let store = DistributedStore::new(config).await;
        assert!(store.is_ok());
    }

    #[test]
    fn test_cluster_error_types() {
        let err = ClusterError::Configuration("test error".to_string());
        assert!(err.to_string().contains("Configuration error: test error"));

        let err = ClusterError::NotLeader;
        assert_eq!(err.to_string(), "Not the leader");

        let err = ClusterError::NodeNotFound { node_id: 42 };
        assert!(err.to_string().contains("Node not found: 42"));

        let err = ClusterError::Network("connection failed".to_string());
        assert!(err.to_string().contains("Network error: connection failed"));
    }
}