//! # Read Replica Support
//!
//! Provides read-only replicas for horizontal read scalability without
//! affecting write performance on the primary nodes.
//!
//! ## Overview
//!
//! Read replicas are secondary nodes that:
//! - Receive data asynchronously from primary nodes
//! - Serve read-only queries with eventual consistency
//! - Can be scaled independently of write capacity
//! - Support load balancing across multiple read replicas
//!
//! ## Features
//!
//! - Asynchronous replication from primary to read replicas
//! - Configurable replication lag tolerance
//! - Health monitoring and automatic failover
//! - Load balancing strategies (round-robin, least-connections, latency-based)
//! - Metrics collection using SciRS2
//! - Stale read detection and handling

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::error::{ClusterError, Result};
use crate::raft::OxirsNodeId;

/// Read replica role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaRole {
    /// Primary node (accepts writes and reads)
    Primary,
    /// Read replica (read-only)
    ReadReplica,
}

/// Read replica status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaStatus {
    /// Replica is healthy and serving requests
    Healthy,
    /// Replica is lagging behind
    Lagging,
    /// Replica is unhealthy
    Unhealthy,
    /// Replica is offline
    Offline,
}

/// Read replica configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadReplicaConfig {
    /// Maximum acceptable replication lag (milliseconds)
    pub max_replication_lag_ms: u64,
    /// Replication batch size
    pub replication_batch_size: usize,
    /// Replication interval (milliseconds)
    pub replication_interval_ms: u64,
    /// Health check interval (seconds)
    pub health_check_interval_secs: u64,
    /// Enable stale read warnings
    pub warn_on_stale_reads: bool,
    /// Maximum stale read age (milliseconds)
    pub max_stale_read_age_ms: u64,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for ReadReplicaConfig {
    fn default() -> Self {
        Self {
            max_replication_lag_ms: 1000,
            replication_batch_size: 100,
            replication_interval_ms: 100,
            health_check_interval_secs: 10,
            warn_on_stale_reads: true,
            max_stale_read_age_ms: 5000,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategy for read replicas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Simple round-robin
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Latency-based (prefer lowest latency)
    LatencyBased,
    /// Random selection
    Random,
}

/// Read replica information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadReplicaInfo {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Replica role
    pub role: ReplicaRole,
    /// Current status
    pub status: ReplicaStatus,
    /// Last sync timestamp
    pub last_sync: Option<SystemTime>,
    /// Replication lag (milliseconds)
    pub replication_lag_ms: u64,
    /// Number of active connections
    pub active_connections: u32,
    /// Average query latency (milliseconds)
    pub avg_query_latency_ms: f64,
    /// Total queries served
    pub total_queries: u64,
    /// Last health check
    pub last_health_check: Option<SystemTime>,
}

impl ReadReplicaInfo {
    /// Create a new read replica info
    pub fn new(node_id: OxirsNodeId, role: ReplicaRole) -> Self {
        Self {
            node_id,
            role,
            status: ReplicaStatus::Healthy,
            last_sync: None,
            replication_lag_ms: 0,
            active_connections: 0,
            avg_query_latency_ms: 0.0,
            total_queries: 0,
            last_health_check: Some(SystemTime::now()),
        }
    }
}

/// Replication statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplicationStats {
    /// Total bytes replicated
    pub total_bytes_replicated: u64,
    /// Total operations replicated
    pub total_operations_replicated: u64,
    /// Average replication lag (milliseconds)
    pub avg_replication_lag_ms: f64,
    /// Maximum replication lag observed (milliseconds)
    pub max_replication_lag_ms: u64,
    /// Number of replication failures
    pub replication_failures: u64,
    /// Last replication timestamp
    pub last_replication: Option<SystemTime>,
}

/// Read replica manager
pub struct ReadReplicaManager {
    config: ReadReplicaConfig,
    /// Registered replicas
    replicas: Arc<RwLock<HashMap<OxirsNodeId, ReadReplicaInfo>>>,
    /// Primary node ID
    primary_node: Arc<RwLock<Option<OxirsNodeId>>>,
    /// Round-robin counter
    round_robin_counter: Arc<RwLock<usize>>,
    /// Replication statistics
    stats: Arc<RwLock<ReplicationStats>>,
}

impl ReadReplicaManager {
    /// Create a new read replica manager
    pub fn new(config: ReadReplicaConfig) -> Self {
        Self {
            config,
            replicas: Arc::new(RwLock::new(HashMap::new())),
            primary_node: Arc::new(RwLock::new(None)),
            round_robin_counter: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(ReplicationStats::default())),
        }
    }

    /// Register a primary node
    pub async fn register_primary(&self, node_id: OxirsNodeId) {
        let mut replicas = self.replicas.write().await;
        replicas.insert(node_id, ReadReplicaInfo::new(node_id, ReplicaRole::Primary));

        let mut primary = self.primary_node.write().await;
        *primary = Some(node_id);

        info!("Registered primary node: {}", node_id);
    }

    /// Register a read replica
    pub async fn register_replica(&self, node_id: OxirsNodeId) {
        let mut replicas = self.replicas.write().await;
        replicas.insert(
            node_id,
            ReadReplicaInfo::new(node_id, ReplicaRole::ReadReplica),
        );

        info!("Registered read replica: {}", node_id);
    }

    /// Unregister a replica
    pub async fn unregister_replica(&self, node_id: OxirsNodeId) {
        let mut replicas = self.replicas.write().await;
        replicas.remove(&node_id);

        info!("Unregistered replica: {}", node_id);
    }

    /// Get the primary node ID
    pub async fn get_primary(&self) -> Option<OxirsNodeId> {
        *self.primary_node.read().await
    }

    /// Get all read replicas
    pub async fn get_read_replicas(&self) -> Vec<OxirsNodeId> {
        let replicas = self.replicas.read().await;
        replicas
            .values()
            .filter(|info| info.role == ReplicaRole::ReadReplica)
            .map(|info| info.node_id)
            .collect()
    }

    /// Select a read replica for query execution
    pub async fn select_replica_for_read(&self) -> Result<OxirsNodeId> {
        let replicas = self.replicas.read().await;

        // Filter healthy read replicas
        let healthy_replicas: Vec<_> = replicas
            .values()
            .filter(|info| {
                info.role == ReplicaRole::ReadReplica && info.status == ReplicaStatus::Healthy
            })
            .collect();

        if healthy_replicas.is_empty() {
            // Fall back to primary if no healthy replicas
            if let Some(primary) = *self.primary_node.read().await {
                debug!("No healthy read replicas available, falling back to primary");
                return Ok(primary);
            }
            return Err(ClusterError::Other(
                "No healthy replicas available".to_string(),
            ));
        }

        match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                let mut counter = self.round_robin_counter.write().await;
                let index = *counter % healthy_replicas.len();
                *counter = (*counter + 1) % healthy_replicas.len();
                Ok(healthy_replicas[index].node_id)
            }
            LoadBalancingStrategy::LeastConnections => {
                // Select replica with least active connections
                let min_replica = healthy_replicas
                    .iter()
                    .min_by_key(|info| info.active_connections)
                    .ok_or_else(|| ClusterError::Other("No replicas available".to_string()))?;
                Ok(min_replica.node_id)
            }
            LoadBalancingStrategy::LatencyBased => {
                // Select replica with lowest query latency
                let min_latency_replica = healthy_replicas
                    .iter()
                    .min_by(|a, b| {
                        a.avg_query_latency_ms
                            .partial_cmp(&b.avg_query_latency_ms)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or_else(|| ClusterError::Other("No replicas available".to_string()))?;
                Ok(min_latency_replica.node_id)
            }
            LoadBalancingStrategy::Random => {
                // Use scirs2_core random for better performance
                use scirs2_core::random::{rng, Rng};
                let index = rng().random_range(0..healthy_replicas.len());
                Ok(healthy_replicas[index].node_id)
            }
        }
    }

    /// Update replication lag for a replica
    pub async fn update_replication_lag(&self, node_id: OxirsNodeId, lag_ms: u64) {
        let mut replicas = self.replicas.write().await;
        if let Some(info) = replicas.get_mut(&node_id) {
            info.replication_lag_ms = lag_ms;
            info.last_sync = Some(SystemTime::now());

            // Update status based on lag
            if lag_ms > self.config.max_replication_lag_ms {
                if info.status != ReplicaStatus::Lagging {
                    warn!(
                        "Replica {} is lagging (lag: {}ms > max: {}ms)",
                        node_id, lag_ms, self.config.max_replication_lag_ms
                    );
                    info.status = ReplicaStatus::Lagging;
                }
            } else if info.status == ReplicaStatus::Lagging {
                info!("Replica {} recovered from lag", node_id);
                info.status = ReplicaStatus::Healthy;
            }
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.last_replication = Some(SystemTime::now());

        // Update average replication lag
        let replica_count = replicas.len() as f64;
        if replica_count > 0.0 {
            let total_lag: u64 = replicas.values().map(|info| info.replication_lag_ms).sum();
            stats.avg_replication_lag_ms = total_lag as f64 / replica_count;
        }

        // Update max lag
        if lag_ms > stats.max_replication_lag_ms {
            stats.max_replication_lag_ms = lag_ms;
        }
    }

    /// Update query statistics for a replica
    pub async fn record_query(&self, node_id: OxirsNodeId, latency_ms: f64) {
        let mut replicas = self.replicas.write().await;
        if let Some(info) = replicas.get_mut(&node_id) {
            info.total_queries += 1;

            // Update average query latency
            if info.total_queries > 1 {
                info.avg_query_latency_ms =
                    (info.avg_query_latency_ms * (info.total_queries - 1) as f64 + latency_ms)
                        / info.total_queries as f64;
            } else {
                info.avg_query_latency_ms = latency_ms;
            }
        }
    }

    /// Increment active connections for a replica
    pub async fn increment_connections(&self, node_id: OxirsNodeId) {
        let mut replicas = self.replicas.write().await;
        if let Some(info) = replicas.get_mut(&node_id) {
            info.active_connections += 1;
        }
    }

    /// Decrement active connections for a replica
    pub async fn decrement_connections(&self, node_id: OxirsNodeId) {
        let mut replicas = self.replicas.write().await;
        if let Some(info) = replicas.get_mut(&node_id) {
            info.active_connections = info.active_connections.saturating_sub(1);
        }
    }

    /// Mark a replica as unhealthy
    pub async fn mark_unhealthy(&self, node_id: OxirsNodeId) {
        let mut replicas = self.replicas.write().await;
        if let Some(info) = replicas.get_mut(&node_id) {
            if info.status != ReplicaStatus::Unhealthy {
                warn!("Marking replica {} as unhealthy", node_id);
                info.status = ReplicaStatus::Unhealthy;
            }
        }
    }

    /// Mark a replica as healthy
    pub async fn mark_healthy(&self, node_id: OxirsNodeId) {
        let mut replicas = self.replicas.write().await;
        if let Some(info) = replicas.get_mut(&node_id) {
            if info.status == ReplicaStatus::Unhealthy {
                info!("Marking replica {} as healthy", node_id);
                info.status = ReplicaStatus::Healthy;
            }
        }
    }

    /// Get replica information
    pub async fn get_replica_info(&self, node_id: OxirsNodeId) -> Option<ReadReplicaInfo> {
        let replicas = self.replicas.read().await;
        replicas.get(&node_id).cloned()
    }

    /// Get all replica information
    pub async fn get_all_replica_info(&self) -> Vec<ReadReplicaInfo> {
        let replicas = self.replicas.read().await;
        replicas.values().cloned().collect()
    }

    /// Get replication statistics
    pub async fn get_stats(&self) -> ReplicationStats {
        self.stats.read().await.clone()
    }

    /// Check if a read is stale
    pub async fn is_stale_read(&self, node_id: OxirsNodeId) -> bool {
        let replicas = self.replicas.read().await;
        if let Some(info) = replicas.get(&node_id) {
            if let Some(last_sync) = info.last_sync {
                if let Ok(elapsed) = SystemTime::now().duration_since(last_sync) {
                    return elapsed.as_millis() > self.config.max_stale_read_age_ms as u128;
                }
            }
        }
        true // Consider stale if no sync info available
    }

    /// Record replication operation
    pub async fn record_replication(&self, bytes: u64, operations: u64) {
        let mut stats = self.stats.write().await;
        stats.total_bytes_replicated += bytes;
        stats.total_operations_replicated += operations;
        stats.last_replication = Some(SystemTime::now());
    }

    /// Record replication failure
    pub async fn record_replication_failure(&self) {
        let mut stats = self.stats.write().await;
        stats.replication_failures += 1;
    }

    /// Get replica health status
    pub async fn get_replica_health(&self) -> HashMap<OxirsNodeId, ReplicaStatus> {
        let replicas = self.replicas.read().await;
        replicas
            .iter()
            .map(|(node_id, info)| (*node_id, info.status))
            .collect()
    }

    /// Perform health check on all replicas
    pub async fn health_check(&self) -> HashMap<OxirsNodeId, bool> {
        let mut replicas = self.replicas.write().await;
        let mut health_status = HashMap::new();

        let now = SystemTime::now();
        let timeout = Duration::from_secs(self.config.health_check_interval_secs * 2);

        for (node_id, info) in replicas.iter_mut() {
            let is_healthy = if let Some(last_check) = info.last_health_check {
                if let Ok(elapsed) = now.duration_since(last_check) {
                    elapsed < timeout
                } else {
                    false
                }
            } else {
                false
            };

            if is_healthy && info.status == ReplicaStatus::Unhealthy {
                info!("Replica {} recovered", node_id);
                info.status = ReplicaStatus::Healthy;
            } else if !is_healthy && info.status == ReplicaStatus::Healthy {
                warn!("Replica {} became unhealthy", node_id);
                info.status = ReplicaStatus::Unhealthy;
            }

            health_status.insert(*node_id, is_healthy);
        }

        health_status
    }

    /// Update health check timestamp for a replica
    pub async fn update_health_check(&self, node_id: OxirsNodeId) {
        let mut replicas = self.replicas.write().await;
        if let Some(info) = replicas.get_mut(&node_id) {
            info.last_health_check = Some(SystemTime::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_replica_manager_creation() {
        let config = ReadReplicaConfig::default();
        let manager = ReadReplicaManager::new(config);

        assert!(manager.get_primary().await.is_none());
        assert!(manager.get_read_replicas().await.is_empty());
    }

    #[tokio::test]
    async fn test_register_primary_and_replica() {
        let config = ReadReplicaConfig::default();
        let manager = ReadReplicaManager::new(config);

        manager.register_primary(1).await;
        manager.register_replica(2).await;
        manager.register_replica(3).await;

        assert_eq!(manager.get_primary().await, Some(1));
        assert_eq!(manager.get_read_replicas().await.len(), 2);
    }

    #[tokio::test]
    async fn test_select_replica_round_robin() {
        let mut config = ReadReplicaConfig::default();
        config.load_balancing = LoadBalancingStrategy::RoundRobin;
        let manager = ReadReplicaManager::new(config);

        manager.register_primary(1).await;
        manager.register_replica(2).await;
        manager.register_replica(3).await;

        let replica1 = manager.select_replica_for_read().await.unwrap();
        let replica2 = manager.select_replica_for_read().await.unwrap();

        assert_ne!(replica1, replica2);
    }

    #[tokio::test]
    async fn test_update_replication_lag() {
        let config = ReadReplicaConfig::default();
        let manager = ReadReplicaManager::new(config);

        manager.register_replica(2).await;
        manager.update_replication_lag(2, 500).await;

        let info = manager.get_replica_info(2).await.unwrap();
        assert_eq!(info.replication_lag_ms, 500);
        assert_eq!(info.status, ReplicaStatus::Healthy);
    }

    #[tokio::test]
    async fn test_replica_lagging_detection() {
        let mut config = ReadReplicaConfig::default();
        config.max_replication_lag_ms = 1000;
        let manager = ReadReplicaManager::new(config);

        manager.register_replica(2).await;
        manager.update_replication_lag(2, 1500).await;

        let info = manager.get_replica_info(2).await.unwrap();
        assert_eq!(info.status, ReplicaStatus::Lagging);
    }

    #[tokio::test]
    async fn test_query_statistics() {
        let config = ReadReplicaConfig::default();
        let manager = ReadReplicaManager::new(config);

        manager.register_replica(2).await;
        manager.record_query(2, 10.0).await;
        manager.record_query(2, 20.0).await;

        let info = manager.get_replica_info(2).await.unwrap();
        assert_eq!(info.total_queries, 2);
        assert_eq!(info.avg_query_latency_ms, 15.0);
    }

    #[tokio::test]
    async fn test_connection_tracking() {
        let config = ReadReplicaConfig::default();
        let manager = ReadReplicaManager::new(config);

        manager.register_replica(2).await;
        manager.increment_connections(2).await;
        manager.increment_connections(2).await;

        let info = manager.get_replica_info(2).await.unwrap();
        assert_eq!(info.active_connections, 2);

        manager.decrement_connections(2).await;
        let info = manager.get_replica_info(2).await.unwrap();
        assert_eq!(info.active_connections, 1);
    }

    #[tokio::test]
    async fn test_health_status() {
        let config = ReadReplicaConfig::default();
        let manager = ReadReplicaManager::new(config);

        manager.register_replica(2).await;
        manager.mark_unhealthy(2).await;

        let info = manager.get_replica_info(2).await.unwrap();
        assert_eq!(info.status, ReplicaStatus::Unhealthy);

        manager.mark_healthy(2).await;
        let info = manager.get_replica_info(2).await.unwrap();
        assert_eq!(info.status, ReplicaStatus::Healthy);
    }

    #[tokio::test]
    async fn test_fallback_to_primary() {
        let config = ReadReplicaConfig::default();
        let manager = ReadReplicaManager::new(config);

        manager.register_primary(1).await;

        // No read replicas, should fall back to primary
        let selected = manager.select_replica_for_read().await.unwrap();
        assert_eq!(selected, 1);
    }
}
