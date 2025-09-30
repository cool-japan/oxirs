//! # Node Lifecycle Management
//!
//! Advanced node lifecycle management with graceful addition/removal,
//! health monitoring, and automated recovery for distributed clusters.

use crate::consensus::ConsensusManager;
use crate::discovery::DiscoveryService;
use crate::error::{ClusterError, Result};
use crate::health_monitor::{HealthMonitor, NodeHealth};
use crate::raft::OxirsNodeId;
use crate::replication::ReplicationManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};

/// Node lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Node is starting up
    Starting,
    /// Node is healthy and operational
    Active,
    /// Node is experiencing issues but still operational
    Degraded,
    /// Node is temporarily unavailable
    Suspended,
    /// Node is being drained for maintenance
    Draining,
    /// Node is gracefully leaving the cluster
    Leaving,
    /// Node has left the cluster
    Left,
    /// Node is unresponsive
    Failed,
    /// Node is suspected to be Byzantine
    Suspected,
}

impl NodeState {
    /// Check if node is operational (can handle requests)
    pub fn is_operational(self) -> bool {
        matches!(self, NodeState::Active | NodeState::Degraded)
    }

    /// Check if node should be included in consensus
    pub fn is_consensus_eligible(self) -> bool {
        matches!(
            self,
            NodeState::Active | NodeState::Degraded | NodeState::Draining
        )
    }

    /// Check if node is healthy
    pub fn is_healthy(self) -> bool {
        matches!(self, NodeState::Active)
    }
}

/// Node lifecycle events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleEvent {
    /// Node is joining the cluster
    NodeJoining {
        node_id: OxirsNodeId,
        address: SocketAddr,
        metadata: NodeMetadata,
    },
    /// Node has successfully joined
    NodeJoined {
        node_id: OxirsNodeId,
        timestamp: u64,
    },
    /// Node state has changed
    NodeStateChanged {
        node_id: OxirsNodeId,
        old_state: NodeState,
        new_state: NodeState,
        reason: String,
        timestamp: u64,
    },
    /// Node is leaving the cluster
    NodeLeaving {
        node_id: OxirsNodeId,
        reason: String,
        graceful: bool,
    },
    /// Node has left the cluster
    NodeLeft {
        node_id: OxirsNodeId,
        timestamp: u64,
    },
    /// Node was forcibly evicted
    NodeEvicted {
        node_id: OxirsNodeId,
        reason: String,
        timestamp: u64,
    },
    /// Cluster membership has changed
    MembershipChanged {
        added_nodes: Vec<OxirsNodeId>,
        removed_nodes: Vec<OxirsNodeId>,
        timestamp: u64,
    },
}

/// Node metadata for join operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Node's advertised capabilities
    pub capabilities: Vec<String>,
    /// Node's data center or region
    pub datacenter: Option<String>,
    /// Node's rack identifier
    pub rack: Option<String>,
    /// Node's version information
    pub version: String,
    /// Custom tags for node identification
    pub tags: HashMap<String, String>,
    /// Node's resource capacity
    pub capacity: ResourceCapacity,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            capabilities: vec!["raft".to_string(), "rdf".to_string()],
            datacenter: None,
            rack: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
            tags: HashMap::new(),
            capacity: ResourceCapacity::default(),
        }
    }
}

/// Resource capacity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapacity {
    /// Maximum CPU cores available
    pub cpu_cores: u32,
    /// Maximum memory in bytes
    pub memory_bytes: u64,
    /// Maximum disk space in bytes
    pub disk_bytes: u64,
    /// Maximum network bandwidth in bytes/sec
    pub network_bandwidth: u64,
    /// Maximum concurrent connections
    pub max_connections: u32,
}

impl Default for ResourceCapacity {
    fn default() -> Self {
        Self {
            cpu_cores: num_cpus::get() as u32,
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB default
            disk_bytes: 100 * 1024 * 1024 * 1024, // 100GB default
            network_bandwidth: 1_000_000_000,     // 1Gbps default
            max_connections: 10000,
        }
    }
}

/// Node lifecycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    /// Health check interval
    pub health_check_interval: Duration,
    /// Node failure detection timeout
    pub failure_timeout: Duration,
    /// Maximum time for graceful shutdown
    pub graceful_shutdown_timeout: Duration,
    /// Time to wait before removing failed nodes
    pub removal_grace_period: Duration,
    /// Enable automatic failure detection
    pub enable_auto_failure_detection: bool,
    /// Enable automatic recovery
    pub enable_auto_recovery: bool,
    /// Minimum cluster size before blocking operations
    pub min_cluster_size: usize,
    /// Maximum time for node join operations
    pub join_timeout: Duration,
    /// Enable Byzantine behavior detection
    pub enable_byzantine_detection: bool,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(30),
            failure_timeout: Duration::from_secs(120),
            graceful_shutdown_timeout: Duration::from_secs(300),
            removal_grace_period: Duration::from_secs(600),
            enable_auto_failure_detection: true,
            enable_auto_recovery: true,
            min_cluster_size: 3,
            join_timeout: Duration::from_secs(60),
            enable_byzantine_detection: true,
        }
    }
}

/// Node status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    /// Node identifier
    pub node_id: OxirsNodeId,
    /// Current state
    pub state: NodeState,
    /// Node address
    pub address: SocketAddr,
    /// Node metadata
    pub metadata: NodeMetadata,
    /// Last seen timestamp
    pub last_seen: u64,
    /// Health information
    pub health: NodeHealth,
    /// Join timestamp
    pub joined_at: u64,
    /// State transition history (last 10 transitions)
    pub state_history: Vec<(NodeState, u64, String)>,
    /// Performance metrics
    pub performance_score: f64,
    /// Failure count since last successful operation
    pub failure_count: u32,
}

impl NodeStatus {
    /// Create new node status
    pub fn new(node_id: OxirsNodeId, address: SocketAddr, metadata: NodeMetadata) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            node_id,
            state: NodeState::Starting,
            address,
            metadata,
            last_seen: now,
            health: NodeHealth::default(),
            joined_at: now,
            state_history: vec![(NodeState::Starting, now, "Initial state".to_string())],
            performance_score: 1.0,
            failure_count: 0,
        }
    }

    /// Update node state
    pub fn update_state(&mut self, new_state: NodeState, reason: String) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Add to history
        self.state_history.push((new_state, now, reason));

        // Keep only last 10 transitions
        if self.state_history.len() > 10 {
            self.state_history.remove(0);
        }

        self.state = new_state;
        self.last_seen = now;

        // Reset failure count on successful state transitions
        if matches!(new_state, NodeState::Active) {
            self.failure_count = 0;
        }
    }

    /// Update health information
    pub fn update_health(&mut self, health: NodeHealth) {
        self.health = health;
        self.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate performance score based on health metrics
        self.performance_score = self.calculate_performance_score();
    }

    /// Calculate performance score (0.0 to 1.0)
    fn calculate_performance_score(&self) -> f64 {
        let mut score: f64 = 1.0;

        // Factor in CPU usage (high usage reduces score)
        if self.health.system_metrics.cpu_usage > 0.8 {
            score *= 0.7;
        } else if self.health.system_metrics.cpu_usage > 0.6 {
            score *= 0.9;
        }

        // Factor in memory usage
        if self.health.system_metrics.memory_usage > 0.9 {
            score *= 0.6;
        } else if self.health.system_metrics.memory_usage > 0.7 {
            score *= 0.8;
        }

        // Factor in response time
        if self.health.response_time > Duration::from_millis(1000) {
            score *= 0.7;
        } else if self.health.response_time > Duration::from_millis(500) {
            score *= 0.9;
        }

        // Factor in failure count
        if self.failure_count > 5 {
            score *= 0.5;
        } else if self.failure_count > 2 {
            score *= 0.8;
        }

        score.clamp(0.0, 1.0)
    }

    /// Check if node has been inactive too long
    pub fn is_stale(&self, timeout: Duration) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now.saturating_sub(self.last_seen) > timeout.as_secs()
    }

    /// Record a failure
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.performance_score = self.calculate_performance_score();
    }
}

/// Node lifecycle manager
pub struct NodeLifecycleManager {
    /// Local node ID
    local_node_id: OxirsNodeId,
    /// Configuration
    config: LifecycleConfig,
    /// Consensus manager
    consensus: Arc<ConsensusManager>,
    /// Discovery service
    discovery: Arc<DiscoveryService>,
    /// Health monitor
    health_monitor: Arc<HealthMonitor>,
    /// Replication manager
    replication: Arc<ReplicationManager>,
    /// Node status map
    node_status: Arc<RwLock<HashMap<OxirsNodeId, NodeStatus>>>,
    /// Event listeners
    event_listeners: Arc<RwLock<Vec<tokio::sync::mpsc::UnboundedSender<LifecycleEvent>>>>,
    /// Background task handle
    task_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl NodeLifecycleManager {
    /// Create new node lifecycle manager
    pub fn new(
        local_node_id: OxirsNodeId,
        config: LifecycleConfig,
        consensus: Arc<ConsensusManager>,
        discovery: Arc<DiscoveryService>,
        health_monitor: Arc<HealthMonitor>,
        replication: Arc<ReplicationManager>,
    ) -> Self {
        Self {
            local_node_id,
            config,
            consensus,
            discovery,
            health_monitor,
            replication,
            node_status: Arc::new(RwLock::new(HashMap::new())),
            event_listeners: Arc::new(RwLock::new(Vec::new())),
            task_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Start the lifecycle manager
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting node lifecycle manager for node {}",
            self.local_node_id
        );

        // Start background monitoring task
        let task_handle = {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.monitoring_loop().await;
            })
        };

        let mut handle = self.task_handle.lock().await;
        *handle = Some(task_handle);

        Ok(())
    }

    /// Stop the lifecycle manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping node lifecycle manager");

        let mut handle = self.task_handle.lock().await;
        if let Some(task) = handle.take() {
            task.abort();
        }

        Ok(())
    }

    /// Add a node to the cluster
    pub async fn add_node(
        &self,
        node_id: OxirsNodeId,
        address: SocketAddr,
        metadata: NodeMetadata,
    ) -> Result<()> {
        info!("Adding node {} at {} to cluster", node_id, address);

        // Check if node already exists
        {
            let nodes = self.node_status.read().await;
            if nodes.contains_key(&node_id) {
                return Err(ClusterError::Config(format!(
                    "Node {node_id} already exists"
                )));
            }
        }

        // Create node status
        let node_status = NodeStatus::new(node_id, address, metadata.clone());

        // Emit joining event
        self.emit_event(LifecycleEvent::NodeJoining {
            node_id,
            address,
            metadata,
        })
        .await;

        // Perform join process with timeout
        let join_result = timeout(
            self.config.join_timeout,
            self.perform_node_join(node_id, address, node_status),
        )
        .await;

        match join_result {
            Ok(Ok(())) => {
                info!("Successfully added node {} to cluster", node_id);

                // Emit joined event
                self.emit_event(LifecycleEvent::NodeJoined {
                    node_id,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                })
                .await;

                Ok(())
            }
            Ok(Err(e)) => {
                error!("Failed to add node {}: {}", node_id, e);
                self.cleanup_failed_join(node_id).await;
                Err(e)
            }
            Err(_) => {
                error!("Node join timed out for node {}", node_id);
                self.cleanup_failed_join(node_id).await;
                Err(ClusterError::Other("Node join timed out".to_string()))
            }
        }
    }

    /// Perform the actual node join process
    async fn perform_node_join(
        &self,
        node_id: OxirsNodeId,
        address: SocketAddr,
        mut node_status: NodeStatus,
    ) -> Result<()> {
        // TODO: Implement interior mutability for these components
        // self.consensus.add_peer(node_id);

        // self.discovery.add_node(node_info);

        // self.replication.add_replica(node_id, address.to_string());

        // Update node state to active
        node_status.update_state(NodeState::Active, "Successfully joined cluster".to_string());

        // Store node status
        {
            let mut nodes = self.node_status.write().await;
            nodes.insert(node_id, node_status);
        }

        // Start health monitoring for the new node
        self.health_monitor
            .start_monitoring(node_id, address.to_string())
            .await;

        Ok(())
    }

    /// Clean up after a failed join
    async fn cleanup_failed_join(&self, node_id: OxirsNodeId) {
        warn!("Cleaning up failed join for node {}", node_id);

        // TODO: Implement interior mutability for these components
        // self.consensus.remove_peer(node_id);
        // self.discovery.remove_node(node_id);
        // self.replication.remove_replica(node_id);

        // Remove from status map
        let mut nodes = self.node_status.write().await;
        nodes.remove(&node_id);
    }

    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: OxirsNodeId, graceful: bool) -> Result<()> {
        info!(
            "Removing node {} from cluster (graceful: {})",
            node_id, graceful
        );

        // Check if node exists
        let node_exists = {
            let nodes = self.node_status.read().await;
            nodes.contains_key(&node_id)
        };

        if !node_exists {
            return Err(ClusterError::Config(format!("Node {node_id} not found")));
        }

        // Emit leaving event
        self.emit_event(LifecycleEvent::NodeLeaving {
            node_id,
            reason: if graceful {
                "Graceful shutdown"
            } else {
                "Forced removal"
            }
            .to_string(),
            graceful,
        })
        .await;

        if graceful {
            // Perform graceful removal
            self.perform_graceful_removal(node_id).await?;
        } else {
            // Perform immediate removal
            self.perform_immediate_removal(node_id).await?;
        }

        // Emit left event
        self.emit_event(LifecycleEvent::NodeLeft {
            node_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
        .await;

        info!("Successfully removed node {} from cluster", node_id);
        Ok(())
    }

    /// Perform graceful node removal
    async fn perform_graceful_removal(&self, node_id: OxirsNodeId) -> Result<()> {
        // Update node state to draining
        {
            let mut nodes = self.node_status.write().await;
            if let Some(node_status) = nodes.get_mut(&node_id) {
                node_status.update_state(
                    NodeState::Draining,
                    "Graceful removal initiated".to_string(),
                );
            }
        }

        // Wait for ongoing operations to complete
        tokio::time::sleep(Duration::from_secs(5)).await;

        // Perform actual removal
        self.perform_immediate_removal(node_id).await
    }

    /// Perform immediate node removal
    async fn perform_immediate_removal(&self, node_id: OxirsNodeId) -> Result<()> {
        // TODO: Implement interior mutability for these components
        // self.consensus.remove_peer(node_id);
        // self.discovery.remove_node(node_id);
        // self.replication.remove_replica(node_id);

        // Stop health monitoring
        self.health_monitor.stop_monitoring(node_id).await;

        // Update node state to left
        {
            let mut nodes = self.node_status.write().await;
            if let Some(node_status) = nodes.get_mut(&node_id) {
                node_status.update_state(NodeState::Left, "Removed from cluster".to_string());
            }
        }

        // Remove from status map after a delay (for audit purposes)
        let nodes = Arc::clone(&self.node_status);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(300)).await; // 5 minutes
            let mut nodes = nodes.write().await;
            nodes.remove(&node_id);
        });

        Ok(())
    }

    /// Force evict a non-responsive node
    pub async fn force_evict_node(&self, node_id: OxirsNodeId, reason: String) -> Result<()> {
        warn!("Force evicting node {}: {}", node_id, reason);

        // Update node state to failed
        {
            let mut nodes = self.node_status.write().await;
            if let Some(node_status) = nodes.get_mut(&node_id) {
                node_status.update_state(NodeState::Failed, reason.clone());
            }
        }

        // Emit eviction event
        self.emit_event(LifecycleEvent::NodeEvicted {
            node_id,
            reason: reason.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
        .await;

        // Perform removal
        self.perform_immediate_removal(node_id).await?;

        info!("Successfully evicted node {}", node_id);
        Ok(())
    }

    /// Get node status
    pub async fn get_node_status(&self, node_id: OxirsNodeId) -> Option<NodeStatus> {
        let nodes = self.node_status.read().await;
        nodes.get(&node_id).cloned()
    }

    /// Get all node statuses
    pub async fn get_all_node_statuses(&self) -> HashMap<OxirsNodeId, NodeStatus> {
        let nodes = self.node_status.read().await;
        nodes.clone()
    }

    /// Get healthy nodes
    pub async fn get_healthy_nodes(&self) -> Vec<OxirsNodeId> {
        let nodes = self.node_status.read().await;
        nodes
            .values()
            .filter(|status| status.state.is_healthy())
            .map(|status| status.node_id)
            .collect()
    }

    /// Get operational nodes
    pub async fn get_operational_nodes(&self) -> Vec<OxirsNodeId> {
        let nodes = self.node_status.read().await;
        nodes
            .values()
            .filter(|status| status.state.is_operational())
            .map(|status| status.node_id)
            .collect()
    }

    /// Check cluster health
    pub async fn check_cluster_health(&self) -> ClusterHealthStatus {
        let nodes = self.node_status.read().await;
        let total_nodes = nodes.len();
        let healthy_nodes = nodes.values().filter(|s| s.state.is_healthy()).count();
        let operational_nodes = nodes.values().filter(|s| s.state.is_operational()).count();
        let failed_nodes = nodes
            .values()
            .filter(|s| matches!(s.state, NodeState::Failed))
            .count();

        let health_ratio = if total_nodes > 0 {
            healthy_nodes as f64 / total_nodes as f64
        } else {
            1.0
        };

        let status = if health_ratio >= 0.8 {
            ClusterHealth::Healthy
        } else if health_ratio >= 0.6 {
            ClusterHealth::Degraded
        } else if operational_nodes >= self.config.min_cluster_size {
            ClusterHealth::Unstable
        } else {
            ClusterHealth::Critical
        };

        ClusterHealthStatus {
            status,
            total_nodes,
            healthy_nodes,
            operational_nodes,
            failed_nodes,
            health_ratio,
            min_cluster_size: self.config.min_cluster_size,
        }
    }

    /// Background monitoring loop
    async fn monitoring_loop(&self) {
        let mut interval = interval(self.config.health_check_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.perform_health_check().await {
                error!("Health check failed: {}", e);
            }

            if self.config.enable_auto_failure_detection {
                if let Err(e) = self.detect_and_handle_failures().await {
                    error!("Failure detection failed: {}", e);
                }
            }

            if self.config.enable_byzantine_detection {
                if let Err(e) = self.detect_byzantine_behavior().await {
                    error!("Byzantine detection failed: {}", e);
                }
            }
        }
    }

    /// Perform health check on all nodes
    async fn perform_health_check(&self) -> Result<()> {
        debug!("Performing health check on all nodes");

        let node_ids: Vec<OxirsNodeId> = {
            let nodes = self.node_status.read().await;
            nodes.keys().copied().collect()
        };

        for node_id in node_ids {
            if let Some(health_status) = self.health_monitor.get_node_health(node_id).await {
                let mut nodes = self.node_status.write().await;
                if let Some(node_status) = nodes.get_mut(&node_id) {
                    node_status.update_health(health_status.health);

                    // Update state based on health
                    let new_state = if node_status.health.status
                        == crate::health_monitor::NodeHealthLevel::Healthy
                    {
                        NodeState::Active
                    } else if node_status.health.system_metrics.cpu_usage > 0.9
                        || node_status.health.system_metrics.memory_usage > 0.95
                    {
                        NodeState::Degraded
                    } else {
                        node_status.state // Keep current state
                    };

                    if new_state != node_status.state {
                        let old_state = node_status.state;
                        node_status
                            .update_state(new_state, "Health check state change".to_string());

                        // Emit state change event
                        self.emit_event(LifecycleEvent::NodeStateChanged {
                            node_id,
                            old_state,
                            new_state,
                            reason: "Health check state change".to_string(),
                            timestamp: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        })
                        .await;
                    }
                }
            }
        }

        Ok(())
    }

    /// Detect and handle node failures
    async fn detect_and_handle_failures(&self) -> Result<()> {
        debug!("Detecting node failures");

        let failed_nodes: Vec<OxirsNodeId> = {
            let nodes = self.node_status.read().await;
            nodes
                .values()
                .filter(|status| {
                    status.is_stale(self.config.failure_timeout)
                        && !matches!(status.state, NodeState::Failed | NodeState::Left)
                })
                .map(|status| status.node_id)
                .collect()
        };

        for node_id in failed_nodes {
            warn!("Detected failed node: {}", node_id);

            // Update state to failed
            {
                let mut nodes = self.node_status.write().await;
                if let Some(node_status) = nodes.get_mut(&node_id) {
                    let old_state = node_status.state;
                    node_status
                        .update_state(NodeState::Failed, "Node failure detected".to_string());
                    node_status.record_failure();

                    // Emit state change event
                    self.emit_event(LifecycleEvent::NodeStateChanged {
                        node_id,
                        old_state,
                        new_state: NodeState::Failed,
                        reason: "Node failure detected".to_string(),
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    })
                    .await;
                }
            }

            // Auto-recovery if enabled
            if self.config.enable_auto_recovery {
                // Wait for grace period before removal
                let manager = self.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(manager.config.removal_grace_period).await;

                    // Check if node is still failed
                    if let Some(status) = manager.get_node_status(node_id).await {
                        if matches!(status.state, NodeState::Failed) {
                            if let Err(e) = manager
                                .force_evict_node(node_id, "Auto-recovery eviction".to_string())
                                .await
                            {
                                error!("Failed to auto-evict node {}: {}", node_id, e);
                            }
                        }
                    }
                });
            }
        }

        Ok(())
    }

    /// Detect Byzantine behavior
    async fn detect_byzantine_behavior(&self) -> Result<()> {
        // This is a simplified Byzantine detection
        // In a production system, this would analyze message patterns,
        // voting behavior, and other consensus-related metrics

        debug!("Checking for Byzantine behavior");

        let suspected_nodes: Vec<OxirsNodeId> = {
            let nodes = self.node_status.read().await;
            nodes
                .values()
                .filter(|status| {
                    status.failure_count > 10 || // High failure count
                    status.performance_score < 0.2 // Very low performance
                })
                .map(|status| status.node_id)
                .collect()
        };

        for node_id in suspected_nodes {
            warn!("Suspected Byzantine behavior from node: {}", node_id);

            // Update state to suspected
            {
                let mut nodes = self.node_status.write().await;
                if let Some(node_status) = nodes.get_mut(&node_id) {
                    let old_state = node_status.state;
                    node_status.update_state(
                        NodeState::Suspected,
                        "Byzantine behavior detected".to_string(),
                    );

                    // Emit state change event
                    self.emit_event(LifecycleEvent::NodeStateChanged {
                        node_id,
                        old_state,
                        new_state: NodeState::Suspected,
                        reason: "Byzantine behavior detected".to_string(),
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    })
                    .await;
                }
            }

            // Consider eviction based on severity
            // This would need more sophisticated analysis in production
        }

        Ok(())
    }

    /// Emit a lifecycle event
    async fn emit_event(&self, event: LifecycleEvent) {
        let listeners = self.event_listeners.read().await;
        for listener in listeners.iter() {
            if listener.send(event.clone()).is_err() {
                // Listener disconnected, will be cleaned up later
            }
        }
    }

    /// Subscribe to lifecycle events
    pub async fn subscribe_events(&self) -> tokio::sync::mpsc::UnboundedReceiver<LifecycleEvent> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut listeners = self.event_listeners.write().await;
        listeners.push(sender);
        receiver
    }

    /// Transfer leadership before graceful shutdown
    pub async fn transfer_leadership(&self, target_node: Option<OxirsNodeId>) -> Result<()> {
        info!("Transferring leadership before shutdown");

        if let Some(target) = target_node {
            // Transfer to specific node
            self.consensus
                .transfer_leadership(target)
                .await
                .map_err(|e| ClusterError::Other(format!("Leadership transfer failed: {e}")))?;
        } else {
            // Let consensus layer choose the best candidate
            let healthy_nodes = self.get_healthy_nodes().await;
            if let Some(&target) = healthy_nodes.first() {
                if target != self.local_node_id {
                    self.consensus
                        .transfer_leadership(target)
                        .await
                        .map_err(|e| {
                            ClusterError::Other(format!("Leadership transfer failed: {e}"))
                        })?;
                }
            }
        }

        info!("Leadership transfer completed");
        Ok(())
    }

    /// Graceful shutdown of local node
    pub async fn graceful_shutdown(&self) -> Result<()> {
        info!("Initiating graceful shutdown of local node");

        // Transfer leadership if we're the leader
        if self.consensus.is_leader().await {
            self.transfer_leadership(None).await?;
        }

        // Remove ourselves from the cluster
        self.remove_node(self.local_node_id, true).await?;

        // Stop the lifecycle manager
        self.stop().await?;

        info!("Graceful shutdown completed");
        Ok(())
    }
}

impl Clone for NodeLifecycleManager {
    fn clone(&self) -> Self {
        Self {
            local_node_id: self.local_node_id,
            config: self.config.clone(),
            consensus: Arc::clone(&self.consensus),
            discovery: Arc::clone(&self.discovery),
            health_monitor: Arc::clone(&self.health_monitor),
            replication: Arc::clone(&self.replication),
            node_status: Arc::clone(&self.node_status),
            event_listeners: Arc::clone(&self.event_listeners),
            task_handle: Arc::clone(&self.task_handle),
        }
    }
}

/// Cluster health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterHealthStatus {
    pub status: ClusterHealth,
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub operational_nodes: usize,
    pub failed_nodes: usize,
    pub health_ratio: f64,
    pub min_cluster_size: usize,
}

/// Overall cluster health
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterHealth {
    Healthy,
    Degraded,
    Unstable,
    Critical,
}

impl ClusterHealth {
    pub fn is_operational(self) -> bool {
        !matches!(self, ClusterHealth::Critical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_node_state_properties() {
        assert!(NodeState::Active.is_operational());
        assert!(NodeState::Active.is_consensus_eligible());
        assert!(NodeState::Active.is_healthy());

        assert!(NodeState::Degraded.is_operational());
        assert!(NodeState::Degraded.is_consensus_eligible());
        assert!(!NodeState::Degraded.is_healthy());

        assert!(!NodeState::Failed.is_operational());
        assert!(!NodeState::Failed.is_consensus_eligible());
        assert!(!NodeState::Failed.is_healthy());
    }

    #[test]
    fn test_node_status_performance_scoring() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let metadata = NodeMetadata::default();
        let mut status = NodeStatus::new(1, addr, metadata);

        // Initially should have good performance score
        assert!(status.performance_score > 0.8);

        // Record failures should decrease score
        status.record_failure();
        status.record_failure();
        status.record_failure();
        assert!(status.performance_score < 0.9);

        // High failure count should significantly decrease score
        for _ in 0..10 {
            status.record_failure();
        }
        assert!(status.performance_score < 0.6);
    }

    #[test]
    fn test_node_status_state_history() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let metadata = NodeMetadata::default();
        let mut status = NodeStatus::new(1, addr, metadata);

        // Should start with Starting state
        assert_eq!(status.state, NodeState::Starting);
        assert_eq!(status.state_history.len(), 1);

        // Update state multiple times
        status.update_state(NodeState::Active, "Joined cluster".to_string());
        status.update_state(NodeState::Degraded, "High CPU usage".to_string());
        status.update_state(NodeState::Active, "CPU usage normalized".to_string());

        // Should track history
        assert_eq!(status.state, NodeState::Active);
        assert_eq!(status.state_history.len(), 4);
        assert_eq!(status.state_history.last().unwrap().0, NodeState::Active);
    }

    #[test]
    fn test_node_status_staleness() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let metadata = NodeMetadata::default();
        let status = NodeStatus::new(1, addr, metadata);

        // Should not be stale immediately
        assert!(!status.is_stale(Duration::from_secs(60)));

        // Create an old status
        let mut old_status = status;
        old_status.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 120; // 2 minutes ago

        // Should be stale with 1 minute timeout
        assert!(old_status.is_stale(Duration::from_secs(60)));
    }

    #[test]
    fn test_resource_capacity_defaults() {
        let capacity = ResourceCapacity::default();

        assert!(capacity.cpu_cores > 0);
        assert!(capacity.memory_bytes > 0);
        assert!(capacity.disk_bytes > 0);
        assert!(capacity.network_bandwidth > 0);
        assert!(capacity.max_connections > 0);
    }

    #[test]
    fn test_cluster_health_operational() {
        assert!(ClusterHealth::Healthy.is_operational());
        assert!(ClusterHealth::Degraded.is_operational());
        assert!(ClusterHealth::Unstable.is_operational());
        assert!(!ClusterHealth::Critical.is_operational());
    }
}
