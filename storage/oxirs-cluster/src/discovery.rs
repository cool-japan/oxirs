//! # Node Discovery
//!
//! Node discovery and membership management for the cluster.
//! Supports multiple discovery mechanisms including static configuration,
//! DNS-based discovery, and gossip protocols.

use crate::raft::OxirsNodeId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use tokio::time::{sleep, timeout};

/// Discovery mechanism configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DiscoveryConfig {
    /// Static list of known nodes
    Static {
        nodes: Vec<(OxirsNodeId, SocketAddr)>,
    },
    /// DNS-based service discovery
    Dns {
        service_name: String,
        domain: String,
        port: u16,
    },
    /// Multicast-based discovery
    Multicast {
        group: String,
        port: u16,
        interface: Option<String>,
    },
    /// File-based discovery (for development)
    File { path: String, watch: bool },
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self::Static { nodes: Vec::new() }
    }
}

/// Information about a cluster node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeInfo {
    /// Unique node identifier
    pub node_id: OxirsNodeId,
    /// Network address for communication
    pub address: SocketAddr,
    /// Last time this node was seen
    pub last_seen: SystemTime,
    /// Whether the node is currently alive
    pub is_alive: bool,
    /// Node metadata (version, capabilities, etc.)
    pub metadata: NodeMetadata,
    /// Response time for health checks
    pub response_time: Duration,
}

/// Node metadata for capability negotiation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct NodeMetadata {
    /// Node version string
    pub version: String,
    /// Supported features
    pub features: BTreeSet<String>,
    /// Node role (leader, follower, etc.)
    pub role: Option<String>,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl NodeInfo {
    /// Create a new node info
    pub fn new(node_id: OxirsNodeId, address: SocketAddr) -> Self {
        Self {
            node_id,
            address,
            last_seen: SystemTime::now(),
            is_alive: true,
            metadata: NodeMetadata::default(),
            response_time: Duration::from_millis(0),
        }
    }

    /// Create node info with metadata
    pub fn with_metadata(
        node_id: OxirsNodeId,
        address: SocketAddr,
        metadata: NodeMetadata,
    ) -> Self {
        Self {
            node_id,
            address,
            last_seen: SystemTime::now(),
            is_alive: true,
            metadata,
            response_time: Duration::from_millis(0),
        }
    }

    /// Check if node is stale based on last seen time
    pub fn is_stale(&self, threshold: Duration) -> bool {
        self.last_seen.elapsed().unwrap_or(Duration::MAX) > threshold
    }

    /// Update node liveness and last seen time
    pub fn update_status(&mut self, is_alive: bool, response_time: Option<Duration>) {
        self.is_alive = is_alive;
        if is_alive {
            self.last_seen = SystemTime::now();
            if let Some(rt) = response_time {
                self.response_time = rt;
            }
        }
    }

    /// Check if node supports a specific feature
    pub fn supports_feature(&self, feature: &str) -> bool {
        self.metadata.features.contains(feature)
    }
}

/// Node discovery statistics
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    pub total_nodes: usize,
    pub alive_nodes: usize,
    pub discovery_rounds: u64,
    pub last_discovery: Option<SystemTime>,
    pub average_response_time: Duration,
}

/// Node discovery service
#[derive(Debug)]
pub struct DiscoveryService {
    local_node_id: OxirsNodeId,
    local_address: SocketAddr,
    local_metadata: NodeMetadata,
    config: DiscoveryConfig,
    known_nodes: HashMap<OxirsNodeId, NodeInfo>,
    stats: DiscoveryStats,
    running: bool,
}

impl DiscoveryService {
    /// Create a new discovery service
    pub fn new(node_id: OxirsNodeId, address: SocketAddr, config: DiscoveryConfig) -> Self {
        let mut metadata = NodeMetadata::default();
        metadata.version = env!("CARGO_PKG_VERSION").to_string();
        metadata.features.insert("raft".to_string());
        metadata.features.insert("rdf".to_string());

        Self {
            local_node_id: node_id,
            local_address: address,
            local_metadata: metadata,
            config,
            known_nodes: HashMap::new(),
            stats: DiscoveryStats::default(),
            running: false,
        }
    }

    /// Create discovery service with metadata
    pub fn with_metadata(
        node_id: OxirsNodeId,
        address: SocketAddr,
        config: DiscoveryConfig,
        metadata: NodeMetadata,
    ) -> Self {
        Self {
            local_node_id: node_id,
            local_address: address,
            local_metadata: metadata,
            config,
            known_nodes: HashMap::new(),
            stats: DiscoveryStats::default(),
            running: false,
        }
    }

    /// Start the discovery service
    pub async fn start(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }

        tracing::info!(
            "Starting node discovery service for node {} at {} with {:?}",
            self.local_node_id,
            self.local_address,
            self.config
        );

        // Initialize with static nodes if configured
        if let DiscoveryConfig::Static { nodes } = &self.config {
            for (node_id, address) in nodes {
                if *node_id != self.local_node_id {
                    let node_info = NodeInfo::new(*node_id, *address);
                    self.known_nodes.insert(*node_id, node_info);
                }
            }
            tracing::info!("Added {} static nodes", nodes.len());
        }

        self.running = true;
        self.update_stats();

        Ok(())
    }

    /// Stop the discovery service
    pub async fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        tracing::info!(
            "Stopping node discovery service for node {}",
            self.local_node_id
        );
        self.running = false;

        Ok(())
    }

    /// Discover nodes using the configured mechanism
    pub async fn discover_nodes(&mut self) -> Result<Vec<NodeInfo>> {
        if !self.running {
            return Ok(Vec::new());
        }

        let discovered = match self.config.clone() {
            DiscoveryConfig::Static { nodes: _ } => {
                // Static nodes are already loaded, just return current state
                self.known_nodes.values().cloned().collect()
            }
            DiscoveryConfig::Dns {
                service_name,
                domain,
                port,
            } => self.discover_via_dns(&service_name, &domain, port).await?,
            DiscoveryConfig::Multicast {
                group,
                port,
                interface: _,
            } => self.discover_via_multicast(&group, port).await?,
            DiscoveryConfig::File { path, watch: _ } => self.discover_via_file(&path).await?,
        };

        self.stats.discovery_rounds += 1;
        self.stats.last_discovery = Some(SystemTime::now());
        self.update_stats();

        Ok(discovered)
    }

    /// Add a node to the known nodes list
    pub fn add_node(&mut self, node_info: NodeInfo) -> bool {
        if node_info.node_id == self.local_node_id {
            return false; // Don't add self
        }

        let is_new = !self.known_nodes.contains_key(&node_info.node_id);
        let node_id = node_info.node_id;
        let address = node_info.address;

        self.known_nodes.insert(node_info.node_id, node_info);

        if is_new {
            tracing::info!("Discovered new node {} at {}", node_id, address);
            self.update_stats();
        }

        is_new
    }

    /// Remove a node from the known nodes list
    pub fn remove_node(&mut self, node_id: OxirsNodeId) -> bool {
        if let Some(node_info) = self.known_nodes.remove(&node_id) {
            tracing::info!("Removed node {} at {}", node_id, node_info.address);
            self.update_stats();
            true
        } else {
            false
        }
    }

    /// Get all known nodes
    pub fn get_nodes(&self) -> &HashMap<OxirsNodeId, NodeInfo> {
        &self.known_nodes
    }

    /// Get alive nodes only
    pub fn get_alive_nodes(&self) -> Vec<&NodeInfo> {
        self.known_nodes
            .values()
            .filter(|node| node.is_alive)
            .collect()
    }

    /// Get nodes that support a specific feature
    pub fn get_nodes_with_feature(&self, feature: &str) -> Vec<&NodeInfo> {
        self.known_nodes
            .values()
            .filter(|node| node.is_alive && node.supports_feature(feature))
            .collect()
    }

    /// Get a specific node by ID
    pub fn get_node(&self, node_id: OxirsNodeId) -> Option<&NodeInfo> {
        self.known_nodes.get(&node_id)
    }

    /// Update node status
    pub fn update_node_status(
        &mut self,
        node_id: OxirsNodeId,
        is_alive: bool,
        response_time: Option<Duration>,
    ) -> bool {
        if let Some(node) = self.known_nodes.get_mut(&node_id) {
            let was_alive = node.is_alive;
            node.update_status(is_alive, response_time);

            if was_alive != is_alive {
                tracing::info!(
                    "Node {} status changed: {} -> {}",
                    node_id,
                    was_alive,
                    is_alive
                );
                self.update_stats();
            }

            true
        } else {
            false
        }
    }

    /// Ping all known nodes to check their health
    pub async fn ping_nodes(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        let ping_timeout = Duration::from_secs(5);
        let mut tasks = Vec::new();

        for node in self.known_nodes.values() {
            let node_id = node.node_id;
            let address = node.address;

            let task = tokio::spawn(async move {
                let start = std::time::Instant::now();

                // Simple TCP connection test as health check
                let result = timeout(ping_timeout, tokio::net::TcpStream::connect(address)).await;
                let response_time = start.elapsed();

                match result {
                    Ok(Ok(_)) => (node_id, true, Some(response_time)),
                    _ => (node_id, false, None),
                }
            });

            tasks.push(task);
        }

        // Wait for all ping tasks to complete
        for task in tasks {
            if let Ok((node_id, is_alive, response_time)) = task.await {
                self.update_node_status(node_id, is_alive, response_time);
            }
        }

        Ok(())
    }

    /// Run periodic health checks and discovery
    pub async fn run_periodic_tasks(&mut self) {
        const DISCOVERY_INTERVAL: Duration = Duration::from_secs(30);
        const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(10);
        const STALE_THRESHOLD: Duration = Duration::from_secs(60);

        let mut discovery_timer = tokio::time::interval(DISCOVERY_INTERVAL);
        let mut health_timer = tokio::time::interval(HEALTH_CHECK_INTERVAL);

        while self.running {
            tokio::select! {
                _ = discovery_timer.tick() => {
                    if let Err(e) = self.discover_nodes().await {
                        tracing::warn!("Discovery failed: {}", e);
                    }
                    self.cleanup_stale_nodes(STALE_THRESHOLD);
                }
                _ = health_timer.tick() => {
                    if let Err(e) = self.ping_nodes().await {
                        tracing::warn!("Health check failed: {}", e);
                    }
                }
            }
        }
    }

    /// Remove stale nodes that haven't been seen recently
    pub fn cleanup_stale_nodes(&mut self, threshold: Duration) {
        let stale_nodes: Vec<_> = self
            .known_nodes
            .iter()
            .filter(|(_, node)| node.is_stale(threshold))
            .map(|(id, _)| *id)
            .collect();

        for node_id in stale_nodes {
            tracing::info!("Removing stale node {}", node_id);
            self.remove_node(node_id);
        }
    }

    /// Get discovery statistics
    pub fn get_stats(&self) -> &DiscoveryStats {
        &self.stats
    }

    /// Get local node information
    pub fn get_local_node_info(&self) -> NodeInfo {
        NodeInfo::with_metadata(
            self.local_node_id,
            self.local_address,
            self.local_metadata.clone(),
        )
    }

    /// Update internal statistics
    fn update_stats(&mut self) {
        let alive_count = self.known_nodes.values().filter(|n| n.is_alive).count();
        let alive_response_times: Vec<Duration> = self
            .known_nodes
            .values()
            .filter(|n| n.is_alive)
            .map(|n| n.response_time)
            .collect();

        self.stats.total_nodes = self.known_nodes.len();
        self.stats.alive_nodes = alive_count;

        if !alive_response_times.is_empty() {
            let total_response_time: Duration = alive_response_times.iter().sum();
            self.stats.average_response_time =
                total_response_time / alive_response_times.len() as u32;
        } else {
            self.stats.average_response_time = Duration::from_millis(0);
        }
    }

    /// DNS-based discovery implementation
    async fn discover_via_dns(
        &mut self,
        _service_name: &str,
        _domain: &str,
        _port: u16,
    ) -> Result<Vec<NodeInfo>> {
        // TODO: Implement DNS SRV record lookup
        tracing::debug!("DNS discovery not yet implemented");
        Ok(self.known_nodes.values().cloned().collect())
    }

    /// Multicast-based discovery implementation
    async fn discover_via_multicast(&mut self, _group: &str, _port: u16) -> Result<Vec<NodeInfo>> {
        // TODO: Implement multicast discovery protocol
        tracing::debug!("Multicast discovery not yet implemented");
        Ok(self.known_nodes.values().cloned().collect())
    }

    /// File-based discovery implementation  
    async fn discover_via_file(&mut self, _path: &str) -> Result<Vec<NodeInfo>> {
        // TODO: Implement file-based discovery
        tracing::debug!("File-based discovery not yet implemented");
        Ok(self.known_nodes.values().cloned().collect())
    }
}

/// Discovery-related errors
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("DNS resolution failed: {message}")]
    DnsResolution { message: String },

    #[error("Network error: {message}")]
    Network { message: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Timeout: {message}")]
    Timeout { message: String },

    #[error("Serialization error: {message}")]
    Serialization { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();
        match config {
            DiscoveryConfig::Static { nodes } => assert!(nodes.is_empty()),
            _ => panic!("Default should be Static with empty nodes"),
        }
    }

    #[test]
    fn test_node_info_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = NodeInfo::new(1, addr);

        assert_eq!(node.node_id, 1);
        assert_eq!(node.address, addr);
        assert!(node.is_alive);
        assert_eq!(node.metadata.version, "");
        assert!(node.metadata.features.is_empty());
    }

    #[test]
    fn test_node_info_with_metadata() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut metadata = NodeMetadata::default();
        metadata.version = "1.0.0".to_string();
        metadata.features.insert("raft".to_string());

        let node = NodeInfo::with_metadata(1, addr, metadata.clone());

        assert_eq!(node.node_id, 1);
        assert_eq!(node.address, addr);
        assert_eq!(node.metadata.version, "1.0.0");
        assert!(node.metadata.features.contains("raft"));
    }

    #[test]
    fn test_node_info_staleness() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = NodeInfo::new(1, addr);

        // Fresh node should not be stale
        assert!(!node.is_stale(Duration::from_secs(10)));

        // Simulate old node by checking against very short threshold
        assert!(node.is_stale(Duration::from_nanos(1)));
    }

    #[test]
    fn test_node_info_update_status() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut node = NodeInfo::new(1, addr);

        // Update to unhealthy
        node.update_status(false, None);
        assert!(!node.is_alive);

        // Update to healthy with response time
        let response_time = Duration::from_millis(50);
        node.update_status(true, Some(response_time));
        assert!(node.is_alive);
        assert_eq!(node.response_time, response_time);
    }

    #[test]
    fn test_node_info_feature_support() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut metadata = NodeMetadata::default();
        metadata.features.insert("raft".to_string());
        metadata.features.insert("rdf".to_string());

        let node = NodeInfo::with_metadata(1, addr, metadata);

        assert!(node.supports_feature("raft"));
        assert!(node.supports_feature("rdf"));
        assert!(!node.supports_feature("unknown"));
    }

    #[tokio::test]
    async fn test_discovery_service_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = DiscoveryConfig::Static { nodes: vec![] };
        let service = DiscoveryService::new(1, addr, config);

        assert_eq!(service.local_node_id, 1);
        assert_eq!(service.local_address, addr);
        assert!(!service.running);
        assert!(service.known_nodes.is_empty());
    }

    #[tokio::test]
    async fn test_discovery_service_start_stop() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = DiscoveryConfig::Static { nodes: vec![] };
        let mut service = DiscoveryService::new(1, addr, config);

        assert!(!service.running);

        service.start().await.unwrap();
        assert!(service.running);

        service.stop().await.unwrap();
        assert!(!service.running);
    }

    #[tokio::test]
    async fn test_discovery_service_static_nodes() {
        let addr1 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let addr2 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081);
        let addr3 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8082);

        let config = DiscoveryConfig::Static {
            nodes: vec![(1, addr1), (2, addr2), (3, addr3)],
        };

        let mut service = DiscoveryService::new(1, addr1, config);
        service.start().await.unwrap();

        // Should not add self to known nodes
        assert!(!service.known_nodes.contains_key(&1));
        assert!(service.known_nodes.contains_key(&2));
        assert!(service.known_nodes.contains_key(&3));
        assert_eq!(service.known_nodes.len(), 2);
    }

    #[tokio::test]
    async fn test_discovery_service_add_remove_nodes() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = DiscoveryConfig::Static { nodes: vec![] };
        let mut service = DiscoveryService::new(1, addr, config);

        let node_info = NodeInfo::new(2, addr);

        // Add node
        assert!(service.add_node(node_info.clone()));
        assert!(service.known_nodes.contains_key(&2));
        assert_eq!(service.known_nodes.len(), 1);

        // Adding same node again should return false
        assert!(!service.add_node(node_info));
        assert_eq!(service.known_nodes.len(), 1);

        // Remove node
        assert!(service.remove_node(2));
        assert!(!service.known_nodes.contains_key(&2));
        assert!(service.known_nodes.is_empty());

        // Removing non-existent node should return false
        assert!(!service.remove_node(3));
    }

    #[tokio::test]
    async fn test_discovery_service_cannot_add_self() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = DiscoveryConfig::Static { nodes: vec![] };
        let mut service = DiscoveryService::new(1, addr, config);

        let self_node_info = NodeInfo::new(1, addr);

        // Should not be able to add self
        assert!(!service.add_node(self_node_info));
        assert!(service.known_nodes.is_empty());
    }

    #[tokio::test]
    async fn test_discovery_service_get_alive_nodes() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = DiscoveryConfig::Static { nodes: vec![] };
        let mut service = DiscoveryService::new(1, addr, config);

        let mut node2 = NodeInfo::new(2, addr);
        let mut node3 = NodeInfo::new(3, addr);

        // Set node3 as unhealthy
        node3.is_alive = false;

        service.add_node(node2);
        service.add_node(node3);

        let alive_nodes = service.get_alive_nodes();
        assert_eq!(alive_nodes.len(), 1);
        assert_eq!(alive_nodes[0].node_id, 2);
    }

    #[tokio::test]
    async fn test_discovery_service_get_nodes_with_feature() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = DiscoveryConfig::Static { nodes: vec![] };
        let mut service = DiscoveryService::new(1, addr, config);

        let mut metadata2 = NodeMetadata::default();
        metadata2.features.insert("raft".to_string());
        let node2 = NodeInfo::with_metadata(2, addr, metadata2);

        let mut metadata3 = NodeMetadata::default();
        metadata3.features.insert("rdf".to_string());
        let node3 = NodeInfo::with_metadata(3, addr, metadata3);

        service.add_node(node2);
        service.add_node(node3);

        let raft_nodes = service.get_nodes_with_feature("raft");
        assert_eq!(raft_nodes.len(), 1);
        assert_eq!(raft_nodes[0].node_id, 2);

        let rdf_nodes = service.get_nodes_with_feature("rdf");
        assert_eq!(rdf_nodes.len(), 1);
        assert_eq!(rdf_nodes[0].node_id, 3);

        let unknown_nodes = service.get_nodes_with_feature("unknown");
        assert!(unknown_nodes.is_empty());
    }

    #[tokio::test]
    async fn test_discovery_service_update_node_status() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = DiscoveryConfig::Static { nodes: vec![] };
        let mut service = DiscoveryService::new(1, addr, config);

        let node_info = NodeInfo::new(2, addr);
        service.add_node(node_info);

        // Update status
        let response_time = Duration::from_millis(100);
        assert!(service.update_node_status(2, false, Some(response_time)));

        let node = service.get_node(2).unwrap();
        assert!(!node.is_alive);
        // Response time only gets updated when the node is healthy
        assert_eq!(node.response_time, Duration::from_millis(0));

        // Update to healthy with response time
        assert!(service.update_node_status(2, true, Some(response_time)));
        let node = service.get_node(2).unwrap();
        assert!(node.is_alive);
        assert_eq!(node.response_time, response_time);

        // Update non-existent node should return false
        assert!(!service.update_node_status(3, true, None));
    }

    #[tokio::test]
    async fn test_discovery_service_cleanup_stale_nodes() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let config = DiscoveryConfig::Static { nodes: vec![] };
        let mut service = DiscoveryService::new(1, addr, config);

        let node_info = NodeInfo::new(2, addr);
        service.add_node(node_info);
        assert_eq!(service.known_nodes.len(), 1);

        // Clean up with very short threshold - should remove the node
        service.cleanup_stale_nodes(Duration::from_nanos(1));
        assert!(service.known_nodes.is_empty());
    }

    #[test]
    fn test_discovery_stats_default() {
        let stats = DiscoveryStats::default();
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.alive_nodes, 0);
        assert_eq!(stats.discovery_rounds, 0);
        assert!(stats.last_discovery.is_none());
        assert_eq!(stats.average_response_time, Duration::from_millis(0));
    }

    #[test]
    fn test_discovery_error_display() {
        let err = DiscoveryError::DnsResolution {
            message: "failed".to_string(),
        };
        assert!(err.to_string().contains("DNS resolution failed: failed"));

        let err = DiscoveryError::Network {
            message: "timeout".to_string(),
        };
        assert!(err.to_string().contains("Network error: timeout"));

        let err = DiscoveryError::Configuration {
            message: "invalid".to_string(),
        };
        assert!(err.to_string().contains("Configuration error: invalid"));

        let err = DiscoveryError::Timeout {
            message: "5s".to_string(),
        };
        assert!(err.to_string().contains("Timeout: 5s"));

        let err = DiscoveryError::Serialization {
            message: "json".to_string(),
        };
        assert!(err.to_string().contains("Serialization error: json"));
    }
}
