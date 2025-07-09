//! Individual cluster node implementation
//!
//! This module provides the core node functionality for clustering including
//! node lifecycle management, health monitoring, and inter-node communication.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, RwLock};
use tracing::{error, info, warn};

use super::{ClusterConfig, NodeInfo, NodeMetadata, NodeState};
use crate::error::{FusekiError, FusekiResult};

/// Node lifecycle events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeEvent {
    /// Node joined the cluster
    Joined(NodeInfo),
    /// Node left the cluster
    Left(String),
    /// Node state changed
    StateChanged(String, NodeState),
    /// Node metadata updated
    MetadataUpdated(String, NodeMetadata),
}

/// Inter-node message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeMessage {
    /// Heartbeat message
    Heartbeat {
        node_id: String,
        timestamp: i64,
        metadata: NodeMetadata,
    },
    /// Join request
    JoinRequest { node_info: NodeInfo },
    /// Join response
    JoinResponse {
        accepted: bool,
        cluster_members: Vec<NodeInfo>,
    },
    /// Leave notification
    LeaveNotification { node_id: String },
    /// Leader election message
    LeaderElection { candidate_id: String, term: u64 },
}

/// Node communication interface
#[async_trait]
pub trait NodeCommunication: Send + Sync {
    /// Send message to a specific node
    async fn send_message(&self, target: &str, message: NodeMessage) -> FusekiResult<()>;

    /// Broadcast message to all nodes
    async fn broadcast_message(&self, message: NodeMessage) -> FusekiResult<()>;

    /// Receive messages from other nodes
    async fn receive_messages(&self) -> FusekiResult<mpsc::Receiver<(String, NodeMessage)>>;
}

/// Cluster node implementation
pub struct ClusterNode {
    /// Node configuration
    config: ClusterConfig,
    /// Node information
    node_info: Arc<RwLock<NodeInfo>>,
    /// Communication interface
    communication: Arc<dyn NodeCommunication>,
    /// Event sender
    event_sender: mpsc::UnboundedSender<NodeEvent>,
    /// Known cluster members
    cluster_members: Arc<RwLock<HashMap<String, NodeInfo>>>,
    /// Last heartbeat times
    last_heartbeats: Arc<RwLock<HashMap<String, Instant>>>,
    /// Node metrics
    metrics: Arc<RwLock<NodeMetrics>>,
}

/// Node performance metrics
#[derive(Debug, Default, Clone)]
pub struct NodeMetrics {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Failed message attempts
    pub message_failures: u64,
    /// Current connections
    pub active_connections: usize,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network bytes in/out
    pub network_io: (u64, u64),
}

impl ClusterNode {
    /// Create a new cluster node
    pub async fn new(
        config: ClusterConfig,
        communication: Arc<dyn NodeCommunication>,
    ) -> FusekiResult<Self> {
        let node_info = Arc::new(RwLock::new(NodeInfo {
            id: config.node_id.clone(),
            addr: config.bind_addr,
            state: NodeState::Joining,
            metadata: NodeMetadata {
                datacenter: None,
                rack: None,
                capacity: 1000,
                load: 0.0,
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            last_heartbeat: chrono::Utc::now().timestamp_millis(),
        }));

        let (event_sender, _) = mpsc::unbounded_channel();

        Ok(Self {
            config,
            node_info,
            communication,
            event_sender,
            cluster_members: Arc::new(RwLock::new(HashMap::new())),
            last_heartbeats: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(NodeMetrics::default())),
        })
    }

    /// Start the node
    pub async fn start(&self) -> FusekiResult<()> {
        info!("Starting cluster node {}", self.config.node_id);

        // Update node state to active
        {
            let mut node_info = self.node_info.write().await;
            node_info.state = NodeState::Active;
        }

        // Start message processing
        self.start_message_processing().await?;

        // Start heartbeat
        self.start_heartbeat().await;

        // Start failure detection
        self.start_failure_detection().await;

        // Start metrics collection
        self.start_metrics_collection().await;

        Ok(())
    }

    /// Stop the node
    pub async fn stop(&self) -> FusekiResult<()> {
        info!("Stopping cluster node {}", self.config.node_id);

        // Send leave notification
        let leave_msg = NodeMessage::LeaveNotification {
            node_id: self.config.node_id.clone(),
        };
        let _ = self.communication.broadcast_message(leave_msg).await;

        // Update node state
        {
            let mut node_info = self.node_info.write().await;
            node_info.state = NodeState::Leaving;
        }

        Ok(())
    }

    /// Join an existing cluster
    pub async fn join_cluster(&self, seed_nodes: &[String]) -> FusekiResult<()> {
        info!("Joining cluster via seeds: {:?}", seed_nodes);

        let node_info = self.node_info.read().await.clone();
        let join_request = NodeMessage::JoinRequest { node_info };

        for seed in seed_nodes {
            match self
                .communication
                .send_message(seed, join_request.clone())
                .await
            {
                Ok(()) => {
                    info!("Successfully contacted seed node: {}", seed);
                    break;
                }
                Err(e) => {
                    warn!("Failed to contact seed node {}: {}", seed, e);
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Start message processing loop
    async fn start_message_processing(&self) -> FusekiResult<()> {
        let mut receiver = self.communication.receive_messages().await?;
        let node_info = self.node_info.clone();
        let cluster_members = self.cluster_members.clone();
        let last_heartbeats = self.last_heartbeats.clone();
        let metrics = self.metrics.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            while let Some((sender_id, message)) = receiver.recv().await {
                // Update metrics
                {
                    let mut m = metrics.write().await;
                    m.messages_received += 1;
                }

                match message {
                    NodeMessage::Heartbeat {
                        node_id,
                        timestamp,
                        metadata,
                    } => {
                        // Update heartbeat time
                        {
                            let mut heartbeats = last_heartbeats.write().await;
                            heartbeats.insert(node_id.clone(), Instant::now());
                        }

                        // Update cluster member info
                        {
                            let mut members = cluster_members.write().await;
                            if let Some(member) = members.get_mut(&node_id) {
                                member.last_heartbeat = timestamp;
                                member.metadata = metadata;
                            }
                        }
                    }

                    NodeMessage::JoinRequest {
                        node_info: joining_node,
                    } => {
                        info!("Received join request from node: {}", joining_node.id);

                        // Add to cluster members
                        {
                            let mut members = cluster_members.write().await;
                            members.insert(joining_node.id.clone(), joining_node.clone());
                        }

                        // Send event
                        let _ = event_sender.send(NodeEvent::Joined(joining_node));
                    }

                    NodeMessage::JoinResponse {
                        accepted,
                        cluster_members: members,
                    } => {
                        if accepted {
                            info!("Join request accepted, updating cluster membership");
                            let mut local_members = cluster_members.write().await;
                            for member in members {
                                local_members.insert(member.id.clone(), member);
                            }
                        } else {
                            warn!("Join request was rejected");
                        }
                    }

                    NodeMessage::LeaveNotification { node_id } => {
                        info!("Node {} is leaving the cluster", node_id);

                        // Remove from cluster members
                        {
                            let mut members = cluster_members.write().await;
                            members.remove(&node_id);
                        }

                        // Remove heartbeat tracking
                        {
                            let mut heartbeats = last_heartbeats.write().await;
                            heartbeats.remove(&node_id);
                        }

                        // Send event
                        let _ = event_sender.send(NodeEvent::Left(node_id));
                    }

                    NodeMessage::LeaderElection { candidate_id, term } => {
                        info!(
                            "Received leader election message from {} for term {}",
                            candidate_id, term
                        );
                        // TODO: Implement leader election logic
                    }
                }
            }
        });

        Ok(())
    }

    /// Start heartbeat broadcasting
    async fn start_heartbeat(&self) {
        let communication = self.communication.clone();
        let node_info = self.node_info.clone();
        let metrics = self.metrics.clone();
        let node_id = self.config.node_id.clone();
        let interval = self.config.raft.heartbeat_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;

                let (metadata, timestamp) = {
                    let info = node_info.read().await;
                    (info.metadata.clone(), chrono::Utc::now().timestamp_millis())
                };

                let heartbeat = NodeMessage::Heartbeat {
                    node_id: node_id.clone(),
                    timestamp,
                    metadata,
                };

                if let Err(e) = communication.broadcast_message(heartbeat).await {
                    error!("Failed to send heartbeat: {}", e);
                    let mut m = metrics.write().await;
                    m.message_failures += 1;
                } else {
                    let mut m = metrics.write().await;
                    m.messages_sent += 1;
                }
            }
        });
    }

    /// Start failure detection
    async fn start_failure_detection(&self) {
        let cluster_members = self.cluster_members.clone();
        let last_heartbeats = self.last_heartbeats.clone();
        let event_sender = self.event_sender.clone();
        let timeout = Duration::from_secs(30); // 30 second timeout

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(10));
            loop {
                ticker.tick().await;

                let now = Instant::now();
                let mut failed_nodes = Vec::new();

                // Check for failed nodes
                {
                    let heartbeats = last_heartbeats.read().await;
                    for (node_id, last_heartbeat) in heartbeats.iter() {
                        if now.duration_since(*last_heartbeat) > timeout {
                            failed_nodes.push(node_id.clone());
                        }
                    }
                }

                // Mark failed nodes as down
                for node_id in failed_nodes {
                    warn!("Node {} marked as down due to missed heartbeats", node_id);

                    {
                        let mut members = cluster_members.write().await;
                        if let Some(member) = members.get_mut(&node_id) {
                            member.state = NodeState::Down;
                            let _ = event_sender
                                .send(NodeEvent::StateChanged(node_id.clone(), NodeState::Down));
                        }
                    }
                }
            }
        });
    }

    /// Start metrics collection
    async fn start_metrics_collection(&self) {
        let metrics = self.metrics.clone();
        let node_info = self.node_info.clone();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(5));
            loop {
                ticker.tick().await;

                // Collect system metrics
                let cpu_usage = Self::get_cpu_usage().await;
                let memory_usage = Self::get_memory_usage().await;
                let network_io = Self::get_network_io().await;

                // Update metrics
                {
                    let mut m = metrics.write().await;
                    m.cpu_usage = cpu_usage;
                    m.memory_usage = memory_usage;
                    m.network_io = network_io;
                }

                // Update node metadata
                {
                    let mut info = node_info.write().await;
                    info.metadata.load = cpu_usage / 100.0;
                }
            }
        });
    }

    /// Get current CPU usage percentage
    async fn get_cpu_usage() -> f64 {
        // TODO: Implement actual CPU usage collection
        0.0
    }

    /// Get current memory usage in bytes
    async fn get_memory_usage() -> u64 {
        // TODO: Implement actual memory usage collection
        0
    }

    /// Get network I/O statistics (bytes in, bytes out)
    async fn get_network_io() -> (u64, u64) {
        // TODO: Implement actual network I/O collection
        (0, 0)
    }

    /// Get node information
    pub async fn get_node_info(&self) -> NodeInfo {
        self.node_info.read().await.clone()
    }

    /// Get cluster members
    pub async fn get_cluster_members(&self) -> HashMap<String, NodeInfo> {
        self.cluster_members.read().await.clone()
    }

    /// Get node metrics
    pub async fn get_metrics(&self) -> NodeMetrics {
        self.metrics.read().await.clone()
    }

    /// Check if node is leader
    pub async fn is_leader(&self) -> bool {
        // TODO: Implement leader detection logic
        false
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> mpsc::UnboundedReceiver<NodeEvent> {
        let (_, receiver) = mpsc::unbounded_channel();
        receiver
    }
}

/// Basic TCP-based node communication implementation
pub struct TcpNodeCommunication {
    bind_addr: SocketAddr,
    known_nodes: Arc<RwLock<HashMap<String, SocketAddr>>>,
}

impl TcpNodeCommunication {
    pub fn new(bind_addr: SocketAddr) -> Self {
        Self {
            bind_addr,
            known_nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_node(&self, node_id: String, addr: SocketAddr) {
        let mut nodes = self.known_nodes.write().await;
        nodes.insert(node_id, addr);
    }
}

#[async_trait]
impl NodeCommunication for TcpNodeCommunication {
    async fn send_message(&self, target: &str, message: NodeMessage) -> FusekiResult<()> {
        let nodes = self.known_nodes.read().await;
        let target_addr = nodes
            .get(target)
            .ok_or_else(|| FusekiError::internal(format!("Unknown target node: {target}")))?;

        // TODO: Implement actual TCP message sending
        info!(
            "Would send message to {} at {}: {:?}",
            target, target_addr, message
        );
        Ok(())
    }

    async fn broadcast_message(&self, message: NodeMessage) -> FusekiResult<()> {
        let nodes = self.known_nodes.read().await;
        for (node_id, addr) in nodes.iter() {
            // TODO: Implement actual TCP message sending
            info!(
                "Would broadcast message to {} at {}: {:?}",
                node_id, addr, message
            );
        }
        Ok(())
    }

    async fn receive_messages(&self) -> FusekiResult<mpsc::Receiver<(String, NodeMessage)>> {
        let (sender, receiver) = mpsc::channel(1000);

        // TODO: Implement actual TCP message receiving
        tokio::spawn(async move {
            // This would normally listen on a TCP port and decode messages
            // For now, just keep the channel open
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });

        Ok(receiver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_cluster_node_creation() {
        let config = ClusterConfig::default();
        let communication = Arc::new(TcpNodeCommunication::new(config.bind_addr));

        let node = ClusterNode::new(config.clone(), communication)
            .await
            .unwrap();
        let info = node.get_node_info().await;

        assert_eq!(info.id, config.node_id);
        assert_eq!(info.addr, config.bind_addr);
        assert_eq!(info.state, NodeState::Joining);
    }

    #[tokio::test]
    async fn test_tcp_communication() {
        let addr = "127.0.0.1:7000".parse().unwrap();
        let comm = TcpNodeCommunication::new(addr);

        comm.add_node("test-node".to_string(), addr).await;

        let message = NodeMessage::Heartbeat {
            node_id: "test".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            metadata: NodeMetadata {
                datacenter: None,
                rack: None,
                capacity: 1000,
                load: 0.5,
                version: "1.0.0".to_string(),
            },
        };

        // Should not fail (just logs for now)
        comm.send_message("test-node", message).await.unwrap();
    }
}
