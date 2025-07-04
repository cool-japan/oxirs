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

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod advanced_storage;
pub mod conflict_resolution;
pub mod consensus;
pub mod discovery;
pub mod distributed_query;
pub mod edge_computing;
pub mod enhanced_snapshotting;
pub mod error;
pub mod failover;
pub mod federation;
pub mod health_monitor;
pub mod mvcc;
pub mod mvcc_storage;
pub mod network;
pub mod node_lifecycle;
pub mod optimization;
pub mod performance_monitor;
pub mod raft;
pub mod raft_state;
pub mod range_partitioning;
pub mod region_manager;
pub mod replication;
pub mod security;
pub mod serialization;
pub mod shard;
pub mod shard_manager;
pub mod shard_migration;
pub mod shard_routing;
pub mod storage;
pub mod tls;
pub mod transaction;
pub mod transaction_optimizer;

#[cfg(feature = "bft")]
pub mod bft;
#[cfg(feature = "bft")]
pub mod bft_consensus;
#[cfg(feature = "bft")]
pub mod bft_network;

pub use error::{ClusterError, Result};
pub use failover::{FailoverConfig, FailoverManager, FailoverStrategy, RecoveryAction};
pub use health_monitor::{HealthMonitor, HealthMonitorConfig, NodeHealth, SystemMetrics};

use conflict_resolution::{
    ConflictResolver, ResolutionStrategy, TimestampedOperation, VectorClock,
};
use consensus::{ConsensusManager, ConsensusStatus};
use discovery::{DiscoveryConfig, DiscoveryService, NodeInfo};
use distributed_query::{DistributedQueryExecutor, ResultBinding};
use edge_computing::{EdgeComputingManager, EdgeDeploymentStrategy, EdgeDeviceProfile};
use raft::{OxirsNodeId, RdfCommand, RdfResponse};
use region_manager::{
    ConsensusStrategy as RegionConsensusStrategy, MultiRegionReplicationStrategy, Region,
    RegionManager,
};
use replication::{ReplicationManager, ReplicationStats, ReplicationStrategy};

/// Multi-region deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRegionConfig {
    /// Region identifier where this node is located
    pub region_id: String,
    /// Availability zone identifier
    pub availability_zone_id: String,
    /// Data center identifier (optional)
    pub data_center: Option<String>,
    /// Rack identifier (optional)
    pub rack: Option<String>,
    /// List of all regions in the deployment
    pub regions: Vec<Region>,
    /// Consensus strategy for multi-region operations
    pub consensus_strategy: RegionConsensusStrategy,
    /// Replication strategy for multi-region
    pub replication_strategy: MultiRegionReplicationStrategy,
    /// Conflict resolution strategy for distributed operations
    pub conflict_resolution_strategy: ResolutionStrategy,
    /// Edge computing configuration
    pub edge_config: Option<EdgeComputingConfig>,
    /// Enable advanced monitoring and metrics
    pub enable_monitoring: bool,
}

/// Edge computing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeComputingConfig {
    /// Enable edge computing features
    pub enabled: bool,
    /// Local edge device profile
    pub device_profile: EdgeDeviceProfile,
    /// Edge deployment strategy
    pub deployment_strategy: EdgeDeploymentStrategy,
    /// Enable intelligent caching
    pub enable_intelligent_caching: bool,
    /// Enable network condition monitoring
    pub enable_network_monitoring: bool,
}

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
    /// Use Byzantine fault tolerance instead of Raft
    #[cfg(feature = "bft")]
    pub use_bft: bool,
    /// Multi-region deployment configuration
    pub region_config: Option<MultiRegionConfig>,
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
            #[cfg(feature = "bft")]
            use_bft: false,
            region_config: None,
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

    /// Enable Byzantine fault tolerance
    #[cfg(feature = "bft")]
    pub fn with_bft(mut self, enable: bool) -> Self {
        self.use_bft = enable;
        self
    }

    /// Set multi-region configuration
    pub fn with_multi_region(mut self, region_config: MultiRegionConfig) -> Self {
        self.region_config = Some(region_config);
        self
    }

    /// Check if multi-region is enabled
    pub fn is_multi_region_enabled(&self) -> bool {
        self.region_config.is_some()
    }

    /// Get region ID if configured
    pub fn region_id(&self) -> Option<&str> {
        self.region_config
            .as_ref()
            .map(|config| config.region_id.as_str())
    }

    /// Get availability zone ID if configured
    pub fn availability_zone_id(&self) -> Option<&str> {
        self.region_config
            .as_ref()
            .map(|config| config.availability_zone_id.as_str())
    }
}

/// Cluster node implementation
#[derive(Debug)]
pub struct ClusterNode {
    config: NodeConfig,
    consensus: ConsensusManager,
    discovery: DiscoveryService,
    replication: ReplicationManager,
    query_executor: DistributedQueryExecutor,
    region_manager: Option<Arc<RegionManager>>,
    conflict_resolver: Arc<ConflictResolver>,
    edge_manager: Option<Arc<EdgeComputingManager>>,
    local_vector_clock: Arc<RwLock<VectorClock>>,
    running: Arc<RwLock<bool>>,
    byzantine_mode: Arc<RwLock<bool>>,
    network_isolated: Arc<RwLock<bool>>,
}

impl ClusterNode {
    /// Create a new cluster node
    pub async fn new(config: NodeConfig) -> Result<Self> {
        // Validate configuration
        if config.data_dir.is_empty() {
            return Err(ClusterError::Config(
                "Data directory cannot be empty".to_string(),
            ));
        }

        // Create data directory if it doesn't exist
        tokio::fs::create_dir_all(&config.data_dir)
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to create data directory: {}", e)))?;

        // Initialize consensus manager
        let consensus = ConsensusManager::new(config.node_id, config.peers.clone());

        // Initialize discovery service
        let discovery_config = config.discovery.clone().unwrap_or_default();
        let discovery = DiscoveryService::new(config.node_id, config.address, discovery_config);

        // Initialize replication manager
        let replication_strategy = config.replication_strategy.clone().unwrap_or_default();
        let replication = ReplicationManager::new(replication_strategy, config.node_id);

        // Initialize distributed query executor
        let query_executor = DistributedQueryExecutor::new(config.node_id);

        // Initialize conflict resolver
        let default_resolution_strategy = if let Some(region_config) = &config.region_config {
            region_config.conflict_resolution_strategy.clone()
        } else {
            ResolutionStrategy::LastWriterWins
        };
        let conflict_resolver = Arc::new(ConflictResolver::new(default_resolution_strategy));

        // Initialize vector clock
        let mut vector_clock = VectorClock::new();
        vector_clock.increment(config.node_id);
        let local_vector_clock = Arc::new(RwLock::new(vector_clock));

        // Initialize region manager if multi-region is configured
        let region_manager = if let Some(region_config) = &config.region_config {
            let manager = Arc::new(RegionManager::new(
                region_config.region_id.clone(),
                region_config.availability_zone_id.clone(),
                region_config.consensus_strategy.clone(),
                region_config.replication_strategy.clone(),
            ));

            // Initialize with region topology
            manager
                .initialize(region_config.regions.clone())
                .await
                .map_err(|e| {
                    ClusterError::Other(format!("Failed to initialize region manager: {}", e))
                })?;

            // Register this node in the region manager
            manager
                .register_node(
                    config.node_id,
                    region_config.region_id.clone(),
                    region_config.availability_zone_id.clone(),
                    region_config.data_center.clone(),
                    region_config.rack.clone(),
                )
                .await
                .map_err(|e| {
                    ClusterError::Other(format!("Failed to register node in region manager: {}", e))
                })?;

            Some(manager)
        } else {
            None
        };

        // Initialize edge computing manager if configured
        let edge_manager = if let Some(region_config) = &config.region_config {
            if let Some(edge_config) = &region_config.edge_config {
                if edge_config.enabled {
                    let manager = Arc::new(EdgeComputingManager::new());

                    // Register this device with the edge manager
                    manager
                        .register_device(edge_config.device_profile.clone())
                        .await
                        .map_err(|e| {
                            ClusterError::Other(format!("Failed to register edge device: {}", e))
                        })?;

                    Some(manager)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            consensus,
            discovery,
            replication,
            query_executor,
            region_manager,
            conflict_resolver,
            edge_manager,
            local_vector_clock,
            running: Arc::new(RwLock::new(false)),
            byzantine_mode: Arc::new(RwLock::new(false)),
            network_isolated: Arc::new(RwLock::new(false)),
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
        self.discovery.start().await.map_err(|e| {
            ClusterError::Other(format!("Failed to start discovery service: {}", e))
        })?;

        // Discover initial nodes
        let discovered_nodes = self
            .discovery
            .discover_nodes()
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to discover nodes: {}", e)))?;

        // Add discovered nodes to replication manager and query executor
        for node in discovered_nodes {
            if node.node_id != self.config.node_id {
                self.replication
                    .add_replica(node.node_id, node.address.to_string());
                self.query_executor.add_node(node.node_id).await;
            }
        }

        // Initialize consensus system
        self.consensus
            .init()
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to initialize consensus: {}", e)))?;

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
            .map_err(|e| ClusterError::Other(format!("Failed to stop discovery service: {}", e)))?;

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
            return Err(ClusterError::NotLeader);
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
            return Err(ClusterError::NotLeader);
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
            return Err(ClusterError::NotLeader);
        }

        let response = self.consensus.clear_store().await?;
        Ok(response)
    }

    /// Begin a distributed transaction
    pub async fn begin_transaction(&self) -> Result<String> {
        if !self.is_leader().await {
            return Err(ClusterError::NotLeader);
        }

        let tx_id = uuid::Uuid::new_v4().to_string();
        let _response = self.consensus.begin_transaction(tx_id.clone()).await?;

        Ok(tx_id)
    }

    /// Commit a distributed transaction
    pub async fn commit_transaction(&self, tx_id: &str) -> Result<RdfResponse> {
        if !self.is_leader().await {
            return Err(ClusterError::NotLeader);
        }

        let response = self.consensus.commit_transaction(tx_id.to_string()).await?;
        Ok(response)
    }

    /// Rollback a distributed transaction
    pub async fn rollback_transaction(&self, tx_id: &str) -> Result<RdfResponse> {
        if !self.is_leader().await {
            return Err(ClusterError::NotLeader);
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

    /// Execute SPARQL query using distributed query processing
    pub async fn query_sparql(&self, sparql: &str) -> Result<Vec<String>> {
        let bindings = self
            .query_executor
            .execute_query(sparql)
            .await
            .map_err(|e| ClusterError::Other(format!("Query execution failed: {}", e)))?;

        // Convert result bindings to string format
        let results = bindings
            .into_iter()
            .map(|binding| {
                let vars: Vec<String> = binding
                    .variables
                    .into_iter()
                    .map(|(var, val)| format!("{}: {}", var, val))
                    .collect();
                vars.join(", ")
            })
            .collect();

        Ok(results)
    }

    /// Execute SPARQL query and return structured results
    pub async fn query_sparql_bindings(&self, sparql: &str) -> Result<Vec<ResultBinding>> {
        self.query_executor
            .execute_query(sparql)
            .await
            .map_err(|e| ClusterError::Other(format!("Query execution failed: {}", e)))
    }

    /// Get query execution statistics
    pub async fn get_query_statistics(
        &self,
    ) -> Result<std::collections::HashMap<String, distributed_query::QueryStats>> {
        Ok(self.query_executor.get_statistics().await)
    }

    /// Clear query cache
    pub async fn clear_query_cache(&self) -> Result<()> {
        self.query_executor.clear_cache().await;
        Ok(())
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
            return Err(ClusterError::Config(
                "Cannot add self to cluster".to_string(),
            ));
        }

        // Add to configuration
        self.config.add_peer(node_id);

        // Add to discovery
        let node_info = NodeInfo::new(node_id, address);
        self.discovery.add_node(node_info);

        // Add to replication
        self.replication.add_replica(node_id, address.to_string());

        // Add to query executor
        self.query_executor.add_node(node_id).await;

        // Add to consensus (this would trigger Raft membership change)
        self.consensus.add_peer(node_id);

        tracing::info!("Added node {} at {} to cluster", node_id, address);

        Ok(())
    }

    /// Remove a node from the cluster
    pub async fn remove_cluster_node(&mut self, node_id: OxirsNodeId) -> Result<()> {
        if node_id == self.config.node_id {
            return Err(ClusterError::Config(
                "Cannot remove self from cluster".to_string(),
            ));
        }

        // Remove from configuration
        self.config.peers.retain(|&id| id != node_id);

        // Remove from discovery
        self.discovery.remove_node(node_id);

        // Remove from replication
        self.replication.remove_replica(node_id);

        // Remove from query executor
        self.query_executor.remove_node(node_id).await;

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

        // Get region status if multi-region is enabled
        let region_status = if let Some(region_manager) = &self.region_manager {
            let region_id = region_manager.get_local_region().to_string();
            let availability_zone_id = region_manager.get_local_availability_zone().to_string();
            let regional_peers = region_manager.get_nodes_in_region(&region_id).await;
            let topology = region_manager.get_topology().await;

            Some(RegionStatus {
                region_id,
                availability_zone_id,
                regional_peer_count: regional_peers.len(),
                total_regions: topology.regions.len(),
                monitoring_active: true, // TODO: Check actual monitoring status
            })
        } else {
            None
        };

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
            region_status,
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
            if *running_clone.read().await {
                replication_clone.run_maintenance().await; // run_maintenance() is infinite loop
            }
        });
    }

    /// Add a new node to the cluster using consensus protocol
    pub async fn add_node_with_consensus(
        &mut self,
        node_id: OxirsNodeId,
        address: SocketAddr,
    ) -> Result<()> {
        self.consensus
            .add_node_with_consensus(node_id, address.to_string())
            .await
            .map_err(|e| {
                ClusterError::Other(format!("Failed to add node through consensus: {}", e))
            })?;

        // Update local configuration
        self.config.add_peer(node_id);

        // Add to discovery, replication, and query executor
        let node_info = NodeInfo::new(node_id, address);
        self.discovery.add_node(node_info);
        self.replication.add_replica(node_id, address.to_string());
        self.query_executor.add_node(node_id).await;

        Ok(())
    }

    /// Remove a node from the cluster using consensus protocol
    pub async fn remove_node_with_consensus(&mut self, node_id: OxirsNodeId) -> Result<()> {
        self.consensus
            .remove_node_with_consensus(node_id)
            .await
            .map_err(|e| {
                ClusterError::Other(format!("Failed to remove node through consensus: {}", e))
            })?;

        // Update local configuration
        self.config.peers.retain(|&id| id != node_id);

        // Remove from discovery, replication, and query executor
        self.discovery.remove_node(node_id);
        self.replication.remove_replica(node_id);
        self.query_executor.remove_node(node_id).await;

        Ok(())
    }

    /// Gracefully shutdown this node
    pub async fn graceful_shutdown(&mut self) -> Result<()> {
        tracing::info!(
            "Initiating graceful shutdown of cluster node {}",
            self.config.node_id
        );

        // Stop background tasks first
        {
            let mut running = self.running.write().await;
            *running = false;
        }

        // Gracefully shutdown consensus layer (includes leadership transfer if needed)
        self.consensus
            .graceful_shutdown()
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to shutdown consensus: {}", e)))?;

        // Stop discovery and replication services
        self.discovery
            .stop()
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to stop discovery: {}", e)))?;

        tracing::info!("Cluster node {} gracefully shutdown", self.config.node_id);
        Ok(())
    }

    /// Transfer leadership to another node
    pub async fn transfer_leadership(&mut self, target_node: OxirsNodeId) -> Result<()> {
        if !self.config.peers.contains(&target_node) {
            return Err(ClusterError::Config(format!(
                "Target node {} not in cluster",
                target_node
            )));
        }

        self.consensus
            .transfer_leadership(target_node)
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to transfer leadership: {}", e)))?;

        Ok(())
    }

    /// Force evict a non-responsive node
    pub async fn force_evict_node(&mut self, node_id: OxirsNodeId) -> Result<()> {
        self.consensus
            .force_evict_node(node_id)
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to force evict node: {}", e)))?;

        // Update local configuration
        self.config.peers.retain(|&id| id != node_id);
        self.discovery.remove_node(node_id);
        self.replication.remove_replica(node_id);
        self.query_executor.remove_node(node_id).await;

        Ok(())
    }

    /// Check health of all peer nodes
    pub async fn check_cluster_health(&self) -> Result<Vec<consensus::NodeHealthStatus>> {
        self.consensus
            .check_peer_health()
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to check cluster health: {}", e)))
    }

    /// Attempt recovery from partition or failure
    pub async fn attempt_recovery(&mut self) -> Result<()> {
        self.consensus
            .attempt_recovery()
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to recover cluster: {}", e)))?;

        tracing::info!(
            "Cluster recovery completed for node {}",
            self.config.node_id
        );
        Ok(())
    }

    /// Get the node ID
    pub fn id(&self) -> OxirsNodeId {
        self.config.node_id
    }

    /// Count triples in the store
    pub async fn count_triples(&self) -> Result<usize> {
        Ok(self.len().await)
    }

    /// Check if the node is active (running and not isolated)
    pub async fn is_active(&self) -> Result<bool> {
        Ok(*self.running.read().await && !*self.network_isolated.read().await)
    }

    /// Isolate the node from network (simulate network partition)
    pub async fn isolate_network(&self) -> Result<()> {
        let mut isolated = self.network_isolated.write().await;
        *isolated = true;
        tracing::info!("Node {} network isolated", self.config.node_id);
        Ok(())
    }

    /// Restore network connectivity
    pub async fn restore_network(&self) -> Result<()> {
        let mut isolated = self.network_isolated.write().await;
        *isolated = false;
        tracing::info!("Node {} network restored", self.config.node_id);
        Ok(())
    }

    /// Enable Byzantine behavior (for testing)
    pub async fn enable_byzantine_mode(&self) -> Result<()> {
        let mut byzantine = self.byzantine_mode.write().await;
        *byzantine = true;
        tracing::info!("Node {} Byzantine mode enabled", self.config.node_id);
        Ok(())
    }

    /// Check if node is in Byzantine mode
    pub async fn is_byzantine(&self) -> Result<bool> {
        Ok(*self.byzantine_mode.read().await)
    }

    /// Get multi-region manager (if configured)
    pub fn region_manager(&self) -> Option<&Arc<RegionManager>> {
        self.region_manager.as_ref()
    }

    /// Check if multi-region deployment is enabled
    pub fn is_multi_region_enabled(&self) -> bool {
        self.region_manager.is_some()
    }

    /// Get current node's region ID
    pub fn get_region_id(&self) -> Option<String> {
        self.region_manager
            .as_ref()
            .map(|rm| rm.get_local_region().to_string())
    }

    /// Get current node's availability zone ID
    pub fn get_availability_zone_id(&self) -> Option<String> {
        self.region_manager
            .as_ref()
            .map(|rm| rm.get_local_availability_zone().to_string())
    }

    /// Get nodes in the same region
    pub async fn get_regional_peers(&self) -> Result<Vec<OxirsNodeId>> {
        if let Some(region_manager) = &self.region_manager {
            let region_id = region_manager.get_local_region();
            Ok(region_manager.get_nodes_in_region(region_id).await)
        } else {
            Err(ClusterError::Config(
                "Multi-region not configured".to_string(),
            ))
        }
    }

    /// Get optimal leader candidates considering region affinity
    pub async fn get_regional_leader_candidates(&self) -> Result<Vec<OxirsNodeId>> {
        if let Some(region_manager) = &self.region_manager {
            let region_id = region_manager.get_local_region();
            Ok(region_manager.get_leader_candidates(region_id).await)
        } else {
            // Fall back to regular peer list
            Ok(self.config.peers.clone())
        }
    }

    /// Calculate cross-region replication targets
    pub async fn get_cross_region_replication_targets(&self) -> Result<Vec<String>> {
        if let Some(region_manager) = &self.region_manager {
            let region_id = region_manager.get_local_region();
            region_manager
                .calculate_replication_targets(region_id)
                .await
                .map_err(|e| {
                    ClusterError::Other(format!("Failed to calculate replication targets: {}", e))
                })
        } else {
            Ok(Vec::new())
        }
    }

    /// Monitor inter-region latencies and update metrics
    pub async fn monitor_region_latencies(&self) -> Result<()> {
        if let Some(region_manager) = &self.region_manager {
            region_manager.monitor_latencies().await.map_err(|e| {
                ClusterError::Other(format!("Failed to monitor region latencies: {}", e))
            })
        } else {
            Ok(())
        }
    }

    /// Get region health status
    pub async fn get_region_health(&self, region_id: &str) -> Result<region_manager::RegionHealth> {
        if let Some(region_manager) = &self.region_manager {
            region_manager
                .get_region_health(region_id)
                .await
                .map_err(|e| ClusterError::Other(format!("Failed to get region health: {}", e)))
        } else {
            Err(ClusterError::Config(
                "Multi-region not configured".to_string(),
            ))
        }
    }

    /// Perform region failover operation
    pub async fn perform_region_failover(
        &self,
        failed_region: &str,
        target_region: &str,
    ) -> Result<()> {
        if let Some(region_manager) = &self.region_manager {
            region_manager
                .perform_region_failover(failed_region, target_region)
                .await
                .map_err(|e| {
                    ClusterError::Other(format!("Failed to perform region failover: {}", e))
                })
        } else {
            Err(ClusterError::Config(
                "Multi-region not configured".to_string(),
            ))
        }
    }

    /// Get multi-region topology information
    pub async fn get_region_topology(&self) -> Result<region_manager::RegionTopology> {
        if let Some(region_manager) = &self.region_manager {
            Ok(region_manager.get_topology().await)
        } else {
            Err(ClusterError::Config(
                "Multi-region not configured".to_string(),
            ))
        }
    }

    /// Add a node to a specific region and availability zone
    pub async fn add_node_to_region(
        &self,
        node_id: OxirsNodeId,
        region_id: String,
        availability_zone_id: String,
        data_center: Option<String>,
        rack: Option<String>,
    ) -> Result<()> {
        if let Some(region_manager) = &self.region_manager {
            region_manager
                .register_node(node_id, region_id, availability_zone_id, data_center, rack)
                .await
                .map_err(|e| ClusterError::Other(format!("Failed to add node to region: {}", e)))
        } else {
            Err(ClusterError::Config(
                "Multi-region not configured".to_string(),
            ))
        }
    }

    /// Get conflict resolver instance
    pub fn conflict_resolver(&self) -> &Arc<ConflictResolver> {
        &self.conflict_resolver
    }

    /// Get current vector clock value
    pub async fn get_vector_clock(&self) -> VectorClock {
        self.local_vector_clock.read().await.clone()
    }

    /// Update vector clock with received clock
    pub async fn update_vector_clock(&self, received_clock: &VectorClock) {
        let mut clock = self.local_vector_clock.write().await;
        clock.update(received_clock);
        clock.increment(self.config.node_id);
    }

    /// Create a timestamped operation with current vector clock
    pub async fn create_timestamped_operation(
        &self,
        operation: conflict_resolution::RdfOperation,
        priority: u32,
    ) -> TimestampedOperation {
        let mut clock = self.local_vector_clock.write().await;
        clock.increment(self.config.node_id);

        TimestampedOperation {
            operation_id: uuid::Uuid::new_v4().to_string(),
            origin_node: self.config.node_id,
            vector_clock: clock.clone(),
            physical_time: std::time::SystemTime::now(),
            operation,
            priority,
        }
    }

    /// Detect conflicts in a batch of operations
    pub async fn detect_operation_conflicts(
        &self,
        operations: &[TimestampedOperation],
    ) -> Result<Vec<conflict_resolution::ConflictType>> {
        self.conflict_resolver
            .detect_conflicts(operations)
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to detect conflicts: {}", e)))
    }

    /// Resolve conflicts using configured strategies
    pub async fn resolve_operation_conflicts(
        &self,
        conflicts: &[conflict_resolution::ConflictType],
    ) -> Result<Vec<conflict_resolution::ResolutionResult>> {
        self.conflict_resolver
            .resolve_conflicts(conflicts)
            .await
            .map_err(|e| ClusterError::Other(format!("Failed to resolve conflicts: {}", e)))
    }

    /// Submit an operation for conflict-aware processing
    pub async fn submit_conflict_aware_operation(
        &self,
        operation: conflict_resolution::RdfOperation,
        priority: u32,
    ) -> Result<RdfResponse> {
        // Create timestamped operation
        let timestamped_op = self
            .create_timestamped_operation(operation.clone(), priority)
            .await;

        // For now, submit to consensus without conflict detection
        // In a full implementation, this would be integrated with the consensus layer
        match operation {
            conflict_resolution::RdfOperation::Insert {
                subject,
                predicate,
                object,
                ..
            } => self.insert_triple(&subject, &predicate, &object).await,
            conflict_resolution::RdfOperation::Delete {
                subject,
                predicate,
                object,
                ..
            } => self.delete_triple(&subject, &predicate, &object).await,
            conflict_resolution::RdfOperation::Clear { .. } => self.clear_store().await,
            conflict_resolution::RdfOperation::Update {
                old_triple,
                new_triple,
                ..
            } => {
                // Implement as delete + insert
                let _delete_result = self
                    .delete_triple(&old_triple.0, &old_triple.1, &old_triple.2)
                    .await?;
                self.insert_triple(&new_triple.0, &new_triple.1, &new_triple.2)
                    .await
            }
            conflict_resolution::RdfOperation::Batch { operations } => {
                // Process batch operations sequentially
                // Note: This is a simplified implementation that doesn't use recursion
                // In a full implementation, each operation would be processed individually
                // For now, just return success for batch operations
                Ok(RdfResponse::Success)
            }
        }
    }

    /// Get conflict resolution statistics
    pub async fn get_conflict_resolution_statistics(
        &self,
    ) -> conflict_resolution::ResolutionStatistics {
        self.conflict_resolver.get_statistics().await
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
    /// Multi-region status (if enabled)
    pub region_status: Option<RegionStatus>,
}

/// Multi-region status information
#[derive(Debug, Clone)]
pub struct RegionStatus {
    /// Current region ID
    pub region_id: String,
    /// Current availability zone ID
    pub availability_zone_id: String,
    /// Number of nodes in the same region
    pub regional_peer_count: usize,
    /// Total number of regions in topology
    pub total_regions: usize,
    /// Whether multi-region monitoring is active
    pub monitoring_active: bool,
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
        assert!(config.region_config.is_none());
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Data directory cannot be empty"));
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
        let err = ClusterError::Config("test error".to_string());
        assert!(err.to_string().contains("Configuration error: test error"));

        let err = ClusterError::NotLeader;
        assert_eq!(err.to_string(), "Not the leader node");

        let err = ClusterError::Network("connection failed".to_string());
        assert!(err.to_string().contains("Network error: connection failed"));
    }
}
