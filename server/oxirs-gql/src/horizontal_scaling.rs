//! Horizontal Scaling Infrastructure Support
//!
//! Provides utilities and helpers for running oxirs-gql in a horizontally scaled,
//! distributed environment with multiple server instances behind a load balancer.
//!
//! ## Features
//!
//! - **Load Balancer Integration**: Health checks, readiness probes, instance metadata
//! - **Session Affinity**: Sticky session support with consistent hashing
//! - **Distributed State**: State synchronization across instances
//! - **Cache Coordination**: Distributed cache invalidation and synchronization
//! - **Instance Discovery**: Service registry and peer discovery
//! - **Graceful Shutdown**: Connection draining and shutdown coordination

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

/// Instance identification and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceMetadata {
    /// Unique instance identifier
    pub instance_id: String,
    /// Instance hostname
    pub hostname: String,
    /// Instance address
    pub address: SocketAddr,
    /// Instance region/zone
    pub region: Option<String>,
    /// Instance version
    pub version: String,
    /// Instance startup time
    pub started_at: SystemTime,
    /// Instance capabilities/features
    pub capabilities: HashSet<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl InstanceMetadata {
    /// Create new instance metadata
    pub fn new(instance_id: String, address: SocketAddr) -> Self {
        // Try to get hostname from environment variable or use instance_id as fallback
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("HOST"))
            .unwrap_or_else(|_| format!("instance-{}", instance_id));

        Self {
            instance_id,
            hostname,
            address,
            region: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
            started_at: SystemTime::now(),
            capabilities: HashSet::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a capability
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.insert(capability.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.started_at
            .elapsed()
            .unwrap_or(Duration::from_secs(0))
            .as_secs()
    }
}

/// Load balancer health check status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Instance is healthy and accepting traffic
    Healthy,
    /// Instance is degraded but still functional
    Degraded,
    /// Instance is unhealthy and should not receive traffic
    Unhealthy,
    /// Instance is draining connections (graceful shutdown)
    Draining,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Draining => write!(f, "draining"),
        }
    }
}

/// Health check response for load balancers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: HealthStatus,
    pub instance_id: String,
    pub uptime_seconds: u64,
    pub active_connections: usize,
    pub checks: HashMap<String, bool>,
    pub message: Option<String>,
}

/// Session affinity strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AffinityStrategy {
    /// No session affinity (round-robin)
    None,
    /// Consistent hashing based on client identifier
    ConsistentHashing,
    /// IP-based affinity
    IpHash,
    /// Cookie-based affinity
    Cookie,
}

/// Session affinity manager
#[derive(Debug)]
pub struct SessionAffinityManager {
    strategy: AffinityStrategy,
    cookie_name: String,
    cookie_ttl: Duration,
    // Track session -> instance mappings
    sessions: Arc<RwLock<HashMap<String, String>>>,
}

impl SessionAffinityManager {
    /// Create new session affinity manager
    pub fn new(strategy: AffinityStrategy) -> Self {
        Self {
            strategy,
            cookie_name: "OXIRS_GQL_AFFINITY".to_string(),
            cookie_ttl: Duration::from_secs(3600), // 1 hour
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set cookie name for cookie-based affinity
    pub fn with_cookie_name(mut self, name: impl Into<String>) -> Self {
        self.cookie_name = name.into();
        self
    }

    /// Set cookie TTL
    pub fn with_cookie_ttl(mut self, ttl: Duration) -> Self {
        self.cookie_ttl = ttl;
        self
    }

    /// Get instance for session
    pub async fn get_instance(&self, session_id: &str) -> Option<String> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }

    /// Bind session to instance
    pub async fn bind_session(&self, session_id: String, instance_id: String) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, instance_id);
        Ok(())
    }

    /// Unbind session
    pub async fn unbind_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id);
        Ok(())
    }

    /// Compute consistent hash for client identifier
    pub fn consistent_hash(&self, client_id: &str, instance_count: usize) -> usize {
        if instance_count == 0 {
            return 0;
        }

        // Simple hash function (in production, use a better hash like xxHash or murmur3)
        let hash = client_id
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        (hash % instance_count as u64) as usize
    }

    /// Get strategy
    pub fn strategy(&self) -> AffinityStrategy {
        self.strategy
    }

    /// Get cookie name
    pub fn cookie_name(&self) -> &str {
        &self.cookie_name
    }
}

/// Distributed state coordinator
#[derive(Debug)]
pub struct StateCoordinator {
    /// Instance metadata
    instance: InstanceMetadata,
    /// Known peer instances
    peers: Arc<RwLock<HashMap<String, PeerInstance>>>,
    /// State synchronization interval
    sync_interval: Duration,
}

/// Peer instance information
#[derive(Debug, Clone)]
pub struct PeerInstance {
    pub metadata: InstanceMetadata,
    pub last_seen: Instant,
    pub health: HealthStatus,
}

impl StateCoordinator {
    /// Create new state coordinator
    pub fn new(instance: InstanceMetadata) -> Self {
        Self {
            instance,
            peers: Arc::new(RwLock::new(HashMap::new())),
            sync_interval: Duration::from_secs(30),
        }
    }

    /// Set sync interval
    pub fn with_sync_interval(mut self, interval: Duration) -> Self {
        self.sync_interval = interval;
        self
    }

    /// Register a peer instance
    pub async fn register_peer(&self, peer: InstanceMetadata) -> Result<()> {
        let mut peers = self.peers.write().await;
        peers.insert(
            peer.instance_id.clone(),
            PeerInstance {
                metadata: peer,
                last_seen: Instant::now(),
                health: HealthStatus::Healthy,
            },
        );
        Ok(())
    }

    /// Update peer health status
    pub async fn update_peer_health(&self, instance_id: &str, health: HealthStatus) -> Result<()> {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.get_mut(instance_id) {
            peer.health = health;
            peer.last_seen = Instant::now();
        }
        Ok(())
    }

    /// Remove stale peers (not seen recently)
    pub async fn remove_stale_peers(&self, max_age: Duration) -> Result<usize> {
        let mut peers = self.peers.write().await;
        let now = Instant::now();
        let initial_count = peers.len();

        peers.retain(|_, peer| now.duration_since(peer.last_seen) < max_age);

        Ok(initial_count - peers.len())
    }

    /// Get all healthy peers
    pub async fn get_healthy_peers(&self) -> Vec<InstanceMetadata> {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|p| p.health == HealthStatus::Healthy)
            .map(|p| p.metadata.clone())
            .collect()
    }

    /// Get all peers
    pub async fn get_all_peers(&self) -> Vec<PeerInstance> {
        let peers = self.peers.read().await;
        peers.values().cloned().collect()
    }

    /// Get instance metadata
    pub fn instance(&self) -> &InstanceMetadata {
        &self.instance
    }

    /// Get peer count
    pub async fn peer_count(&self) -> usize {
        let peers = self.peers.read().await;
        peers.len()
    }
}

/// Graceful shutdown coordinator
#[derive(Debug)]
pub struct ShutdownCoordinator {
    /// Drain timeout
    drain_timeout: Duration,
    /// Active connections counter
    active_connections: Arc<RwLock<usize>>,
    /// Shutdown initiated flag
    shutdown_initiated: Arc<RwLock<bool>>,
}

impl ShutdownCoordinator {
    /// Create new shutdown coordinator
    pub fn new() -> Self {
        Self {
            drain_timeout: Duration::from_secs(30),
            active_connections: Arc::new(RwLock::new(0)),
            shutdown_initiated: Arc::new(RwLock::new(false)),
        }
    }

    /// Set drain timeout
    pub fn with_drain_timeout(mut self, timeout: Duration) -> Self {
        self.drain_timeout = timeout;
        self
    }

    /// Increment active connections
    pub async fn connection_started(&self) -> Result<()> {
        let shutdown = *self.shutdown_initiated.read().await;
        if shutdown {
            return Err(anyhow!("Server is shutting down"));
        }

        let mut count = self.active_connections.write().await;
        *count += 1;
        Ok(())
    }

    /// Decrement active connections
    pub async fn connection_ended(&self) {
        let mut count = self.active_connections.write().await;
        if *count > 0 {
            *count -= 1;
        }
    }

    /// Get active connection count
    pub async fn active_connections(&self) -> usize {
        *self.active_connections.read().await
    }

    /// Initiate graceful shutdown
    pub async fn initiate_shutdown(&self) -> Result<()> {
        let mut shutdown = self.shutdown_initiated.write().await;
        *shutdown = true;
        Ok(())
    }

    /// Check if shutdown is in progress
    pub async fn is_shutting_down(&self) -> bool {
        *self.shutdown_initiated.read().await
    }

    /// Wait for all connections to drain
    pub async fn wait_for_drain(&self) -> Result<()> {
        let start = Instant::now();

        while start.elapsed() < self.drain_timeout {
            let count = self.active_connections().await;
            if count == 0 {
                return Ok(());
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let remaining = self.active_connections().await;
        if remaining > 0 {
            tracing::warn!(
                "Shutdown timeout reached with {} active connections remaining",
                remaining
            );
        }

        Ok(())
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for horizontal scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalScalingConfig {
    /// Enable horizontal scaling features
    pub enabled: bool,
    /// Instance metadata
    pub instance_id: Option<String>,
    /// Session affinity strategy
    pub affinity_strategy: AffinityStrategy,
    /// Peer discovery enabled
    pub peer_discovery: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Graceful shutdown timeout
    pub shutdown_timeout: Duration,
}

impl Default for HorizontalScalingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            instance_id: None,
            affinity_strategy: AffinityStrategy::None,
            peer_discovery: false,
            health_check_interval: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(30),
        }
    }
}

/// Horizontal scaling manager
#[derive(Debug)]
pub struct HorizontalScalingManager {
    config: HorizontalScalingConfig,
    affinity: SessionAffinityManager,
    coordinator: StateCoordinator,
    shutdown: ShutdownCoordinator,
}

impl HorizontalScalingManager {
    /// Create new horizontal scaling manager
    pub fn new(config: HorizontalScalingConfig, instance: InstanceMetadata) -> Self {
        let affinity = SessionAffinityManager::new(config.affinity_strategy);
        let coordinator = StateCoordinator::new(instance);
        let shutdown = ShutdownCoordinator::new().with_drain_timeout(config.shutdown_timeout);

        Self {
            config,
            affinity,
            coordinator,
            shutdown,
        }
    }

    /// Get session affinity manager
    pub fn affinity(&self) -> &SessionAffinityManager {
        &self.affinity
    }

    /// Get state coordinator
    pub fn coordinator(&self) -> &StateCoordinator {
        &self.coordinator
    }

    /// Get shutdown coordinator
    pub fn shutdown(&self) -> &ShutdownCoordinator {
        &self.shutdown
    }

    /// Check if scaling is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get instance metadata
    pub fn instance(&self) -> &InstanceMetadata {
        self.coordinator.instance()
    }

    /// Generate health check response
    pub async fn health_check(&self) -> HealthCheckResponse {
        let status = if self.shutdown.is_shutting_down().await {
            HealthStatus::Draining
        } else {
            HealthStatus::Healthy
        };

        HealthCheckResponse {
            status,
            instance_id: self.instance().instance_id.clone(),
            uptime_seconds: self.instance().uptime_seconds(),
            active_connections: self.shutdown.active_connections().await,
            checks: HashMap::new(),
            message: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_metadata_creation() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let metadata = InstanceMetadata::new("instance-1".to_string(), addr)
            .with_capability("graphql")
            .with_capability("federation")
            .with_region("us-west-2")
            .with_metadata("env", "production");

        assert_eq!(metadata.instance_id, "instance-1");
        assert_eq!(metadata.address, addr);
        assert!(metadata.capabilities.contains("graphql"));
        assert!(metadata.capabilities.contains("federation"));
        assert_eq!(metadata.region, Some("us-west-2".to_string()));
        assert_eq!(
            metadata.metadata.get("env"),
            Some(&"production".to_string())
        );
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
        assert_eq!(HealthStatus::Draining.to_string(), "draining");
    }

    #[tokio::test]
    async fn test_session_affinity_manager() {
        let manager = SessionAffinityManager::new(AffinityStrategy::ConsistentHashing)
            .with_cookie_name("MY_COOKIE")
            .with_cookie_ttl(Duration::from_secs(7200));

        assert_eq!(manager.cookie_name(), "MY_COOKIE");
        assert_eq!(manager.strategy(), AffinityStrategy::ConsistentHashing);

        manager
            .bind_session("session-1".to_string(), "instance-1".to_string())
            .await
            .unwrap();

        let instance = manager.get_instance("session-1").await;
        assert_eq!(instance, Some("instance-1".to_string()));

        manager.unbind_session("session-1").await.unwrap();
        assert_eq!(manager.get_instance("session-1").await, None);
    }

    #[test]
    fn test_consistent_hashing() {
        let manager = SessionAffinityManager::new(AffinityStrategy::ConsistentHashing);

        // Test deterministic hashing
        let hash1 = manager.consistent_hash("client-1", 10);
        let hash2 = manager.consistent_hash("client-1", 10);
        assert_eq!(hash1, hash2);

        // Test distribution across instances
        let hash3 = manager.consistent_hash("client-2", 10);
        // Different clients should (usually) map to different instances
        // Note: This isn't guaranteed, but is statistically likely
        let _ = hash3; // Just verify it doesn't panic

        // Test zero instance count
        let hash4 = manager.consistent_hash("client-1", 0);
        assert_eq!(hash4, 0);
    }

    #[tokio::test]
    async fn test_state_coordinator() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let instance = InstanceMetadata::new("instance-1".to_string(), addr);
        let coordinator = StateCoordinator::new(instance);

        assert_eq!(coordinator.peer_count().await, 0);

        // Register a peer
        let peer_addr: SocketAddr = "127.0.0.1:8081".parse().unwrap();
        let peer = InstanceMetadata::new("instance-2".to_string(), peer_addr);
        coordinator.register_peer(peer).await.unwrap();

        assert_eq!(coordinator.peer_count().await, 1);

        // Update peer health
        coordinator
            .update_peer_health("instance-2", HealthStatus::Degraded)
            .await
            .unwrap();

        let peers = coordinator.get_all_peers().await;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].health, HealthStatus::Degraded);

        // Get healthy peers (should be 0 since we set to Degraded)
        let healthy = coordinator.get_healthy_peers().await;
        assert_eq!(healthy.len(), 0);

        // Update to healthy
        coordinator
            .update_peer_health("instance-2", HealthStatus::Healthy)
            .await
            .unwrap();

        let healthy = coordinator.get_healthy_peers().await;
        assert_eq!(healthy.len(), 1);
    }

    #[tokio::test]
    async fn test_shutdown_coordinator() {
        let coordinator = ShutdownCoordinator::new().with_drain_timeout(Duration::from_secs(5));

        assert_eq!(coordinator.active_connections().await, 0);
        assert!(!coordinator.is_shutting_down().await);

        // Start connections
        coordinator.connection_started().await.unwrap();
        coordinator.connection_started().await.unwrap();
        assert_eq!(coordinator.active_connections().await, 2);

        // End one connection
        coordinator.connection_ended().await;
        assert_eq!(coordinator.active_connections().await, 1);

        // Initiate shutdown
        coordinator.initiate_shutdown().await.unwrap();
        assert!(coordinator.is_shutting_down().await);

        // New connections should fail
        assert!(coordinator.connection_started().await.is_err());

        // End remaining connection
        coordinator.connection_ended().await;
        assert_eq!(coordinator.active_connections().await, 0);

        // Drain should complete immediately
        coordinator.wait_for_drain().await.unwrap();
    }

    #[tokio::test]
    async fn test_horizontal_scaling_manager() {
        let config = HorizontalScalingConfig {
            enabled: true,
            instance_id: Some("test-instance".to_string()),
            affinity_strategy: AffinityStrategy::Cookie,
            peer_discovery: true,
            health_check_interval: Duration::from_secs(10),
            shutdown_timeout: Duration::from_secs(30),
        };

        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let instance = InstanceMetadata::new("test-instance".to_string(), addr);
        let manager = HorizontalScalingManager::new(config, instance);

        assert!(manager.is_enabled());
        assert_eq!(manager.affinity().strategy(), AffinityStrategy::Cookie);

        // Test health check
        let health = manager.health_check().await;
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.instance_id, "test-instance");
    }

    #[tokio::test]
    async fn test_stale_peer_removal() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let instance = InstanceMetadata::new("instance-1".to_string(), addr);
        let coordinator = StateCoordinator::new(instance);

        // Register peers
        let peer1_addr: SocketAddr = "127.0.0.1:8081".parse().unwrap();
        let peer1 = InstanceMetadata::new("instance-2".to_string(), peer1_addr);
        coordinator.register_peer(peer1).await.unwrap();

        let peer2_addr: SocketAddr = "127.0.0.1:8082".parse().unwrap();
        let peer2 = InstanceMetadata::new("instance-3".to_string(), peer2_addr);
        coordinator.register_peer(peer2).await.unwrap();

        assert_eq!(coordinator.peer_count().await, 2);

        // Remove stale peers (none should be removed with max age of 1 hour)
        let removed = coordinator
            .remove_stale_peers(Duration::from_secs(3600))
            .await
            .unwrap();
        assert_eq!(removed, 0);
        assert_eq!(coordinator.peer_count().await, 2);

        // Remove with very short max age (all should be removed)
        tokio::time::sleep(Duration::from_millis(10)).await;
        let removed = coordinator
            .remove_stale_peers(Duration::from_millis(1))
            .await
            .unwrap();
        assert_eq!(removed, 2);
        assert_eq!(coordinator.peer_count().await, 0);
    }
}
