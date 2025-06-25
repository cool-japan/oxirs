//! # Node Discovery
//!
//! Node discovery and membership management for the cluster.
//! Supports multiple discovery mechanisms including static configuration,
//! DNS-based discovery, and gossip protocols.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeSet};
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use tokio::time::{sleep, timeout};
use crate::raft::OxirsNodeId;

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
    File {
        path: String,
        watch: bool,
    },
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
    pub fn with_metadata(node_id: OxirsNodeId, address: SocketAddr, metadata: NodeMetadata) -> Self {
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
    pub fn new(
        node_id: OxirsNodeId, 
        address: SocketAddr, 
        config: DiscoveryConfig
    ) -> Self {
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
        
        tracing::info!("Stopping node discovery service for node {}", self.local_node_id);
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
            DiscoveryConfig::Dns { service_name, domain, port } => {
                self.discover_via_dns(&service_name, &domain, port).await?
            }
            DiscoveryConfig::Multicast { group, port, interface: _ } => {
                self.discover_via_multicast(&group, port).await?
            }
            DiscoveryConfig::File { path, watch: _ } => {
                self.discover_via_file(&path).await?
            }
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
        self.known_nodes.values()
            .filter(|node| node.is_alive)
            .collect()
    }
    
    /// Get nodes that support a specific feature
    pub fn get_nodes_with_feature(&self, feature: &str) -> Vec<&NodeInfo> {
        self.known_nodes.values()
            .filter(|node| node.is_alive && node.supports_feature(feature))
            .collect()
    }
    
    /// Get a specific node by ID
    pub fn get_node(&self, node_id: OxirsNodeId) -> Option<&NodeInfo> {
        self.known_nodes.get(&node_id)
    }
    
    /// Update node status
    pub fn update_node_status(&mut self, node_id: OxirsNodeId, is_alive: bool, response_time: Option<Duration>) -> bool {
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
        let stale_nodes: Vec<_> = self.known_nodes
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
        NodeInfo::with_metadata(self.local_node_id, self.local_address, self.local_metadata.clone())
    }
    
    /// Update internal statistics
    fn update_stats(&mut self) {
        let alive_count = self.known_nodes.values().filter(|n| n.is_alive).count();
        let alive_response_times: Vec<Duration> = self.known_nodes.values().filter(|n| n.is_alive).map(|n| n.response_time).collect();
        
        self.stats.total_nodes = self.known_nodes.len();
        self.stats.alive_nodes = alive_count;
        
        if !alive_response_times.is_empty() {
            let total_response_time: Duration = alive_response_times.iter().sum();
            self.stats.average_response_time = total_response_time / alive_response_times.len() as u32;
        } else {
            self.stats.average_response_time = Duration::from_millis(0);
        }
    }
    
    /// DNS-based discovery implementation
    async fn discover_via_dns(&mut self, _service_name: &str, _domain: &str, _port: u16) -> Result<Vec<NodeInfo>> {
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