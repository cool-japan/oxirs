//! # Data Replication
//!
//! High-level data replication management for distributed RDF storage.
//! Works with Raft consensus to ensure consistent replication.

use crate::raft::OxirsNodeId;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tokio::time::sleep;

/// Replication strategy for the cluster
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReplicationStrategy {
    /// Synchronous replication - wait for all replicas
    Synchronous,
    /// Asynchronous replication - fire and forget
    Asynchronous,
    /// Semi-synchronous - wait for minimum replicas
    SemiSynchronous { min_replicas: usize },
    /// Raft consensus - use Raft for replication
    RaftConsensus,
}

impl Default for ReplicationStrategy {
    fn default() -> Self {
        Self::RaftConsensus
    }
}

/// Information about a replica node
#[derive(Debug, Clone)]
pub struct ReplicaInfo {
    /// Unique node identifier
    pub node_id: OxirsNodeId,
    /// Address of the replica
    pub address: String,
    /// Last successfully applied log index
    pub last_applied_index: u64,
    /// Whether the replica is currently healthy
    pub is_healthy: bool,
    /// Last time we successfully communicated with this replica
    pub last_contact: SystemTime,
    /// Current replication lag in log entries
    pub replication_lag: u64,
    /// Network latency to this replica
    pub latency: Duration,
}

impl ReplicaInfo {
    /// Create a new replica info
    pub fn new(node_id: OxirsNodeId, address: String) -> Self {
        Self {
            node_id,
            address,
            last_applied_index: 0,
            is_healthy: true,
            last_contact: SystemTime::now(),
            replication_lag: 0,
            latency: Duration::from_millis(0),
        }
    }

    /// Check if replica is stale based on contact time
    pub fn is_stale(&self, threshold: Duration) -> bool {
        self.last_contact.elapsed().unwrap_or(Duration::MAX) > threshold
    }

    /// Update health status and contact time
    pub fn update_health(&mut self, is_healthy: bool) {
        self.is_healthy = is_healthy;
        if is_healthy {
            self.last_contact = SystemTime::now();
        }
    }
}

/// Replication statistics
#[derive(Debug, Clone, Default)]
pub struct ReplicationStats {
    pub total_replicas: usize,
    pub healthy_replicas: usize,
    pub average_lag: f64,
    pub max_lag: u64,
    pub min_lag: u64,
    pub average_latency: Duration,
    pub replication_throughput: f64, // operations per second
}

/// Replication manager for distributed RDF data
#[derive(Debug)]
pub struct ReplicationManager {
    strategy: ReplicationStrategy,
    replicas: HashMap<OxirsNodeId, ReplicaInfo>,
    local_node_id: OxirsNodeId,
    stats: ReplicationStats,
}

impl ReplicationManager {
    /// Create a new replication manager
    pub fn new(strategy: ReplicationStrategy, local_node_id: OxirsNodeId) -> Self {
        Self {
            strategy,
            replicas: HashMap::new(),
            local_node_id,
            stats: ReplicationStats::default(),
        }
    }

    /// Create a new replication manager with Raft consensus (default)
    pub fn with_raft_consensus(local_node_id: OxirsNodeId) -> Self {
        Self::new(ReplicationStrategy::RaftConsensus, local_node_id)
    }

    /// Add a replica to the replication set
    pub fn add_replica(&mut self, node_id: OxirsNodeId, address: String) -> bool {
        if node_id == self.local_node_id {
            tracing::warn!("Cannot add local node as replica");
            return false;
        }

        let replica_info = ReplicaInfo::new(node_id, address.clone());
        let is_new = !self.replicas.contains_key(&node_id);

        self.replicas.insert(node_id, replica_info);

        if is_new {
            tracing::info!("Added replica {} at {}", node_id, address);
            self.update_stats();
        }

        is_new
    }

    /// Remove a replica from the replication set
    pub fn remove_replica(&mut self, node_id: OxirsNodeId) -> bool {
        if let Some(replica) = self.replicas.remove(&node_id) {
            tracing::info!("Removed replica {} at {}", node_id, replica.address);
            self.update_stats();
            true
        } else {
            false
        }
    }

    /// Get all replicas
    pub fn get_replicas(&self) -> &HashMap<OxirsNodeId, ReplicaInfo> {
        &self.replicas
    }

    /// Get healthy replicas only
    pub fn get_healthy_replicas(&self) -> Vec<&ReplicaInfo> {
        self.replicas
            .values()
            .filter(|replica| replica.is_healthy)
            .collect()
    }

    /// Get replica by node ID
    pub fn get_replica(&self, node_id: OxirsNodeId) -> Option<&ReplicaInfo> {
        self.replicas.get(&node_id)
    }

    /// Update replica health status
    pub fn update_replica_health(&mut self, node_id: OxirsNodeId, is_healthy: bool) -> bool {
        if let Some(replica) = self.replicas.get_mut(&node_id) {
            let was_healthy = replica.is_healthy;
            replica.update_health(is_healthy);

            if was_healthy != is_healthy {
                tracing::info!(
                    "Replica {} health changed: {} -> {}",
                    node_id,
                    was_healthy,
                    is_healthy
                );
                self.update_stats();
            }

            true
        } else {
            false
        }
    }

    /// Update replica lag information
    pub fn update_replica_lag(
        &mut self,
        node_id: OxirsNodeId,
        applied_index: u64,
        current_index: u64,
    ) {
        if let Some(replica) = self.replicas.get_mut(&node_id) {
            replica.last_applied_index = applied_index;
            replica.replication_lag = current_index.saturating_sub(applied_index);
            self.update_stats();
        }
    }

    /// Check all replicas and mark stale ones as unhealthy
    pub async fn health_check(&mut self, stale_threshold: Duration) {
        let mut changed = false;

        for replica in self.replicas.values_mut() {
            let was_healthy = replica.is_healthy;

            if replica.is_stale(stale_threshold) {
                replica.is_healthy = false;
            }

            if was_healthy != replica.is_healthy {
                changed = true;
                tracing::warn!(
                    "Replica {} marked as unhealthy due to staleness",
                    replica.node_id
                );
            }
        }

        if changed {
            self.update_stats();
        }
    }

    /// Get the current replication strategy
    pub fn get_strategy(&self) -> &ReplicationStrategy {
        &self.strategy
    }

    /// Change the replication strategy
    pub fn set_strategy(&mut self, strategy: ReplicationStrategy) {
        if self.strategy != strategy {
            tracing::info!(
                "Changing replication strategy from {:?} to {:?}",
                self.strategy,
                strategy
            );
            self.strategy = strategy;
        }
    }

    /// Get replication statistics
    pub fn get_stats(&self) -> &ReplicationStats {
        &self.stats
    }

    /// Check if replication requirements are met
    pub fn is_replication_healthy(&self) -> bool {
        let healthy_count = self.get_healthy_replicas().len();

        match &self.strategy {
            ReplicationStrategy::Synchronous => healthy_count == self.replicas.len(),
            ReplicationStrategy::Asynchronous => true, // Always considered healthy
            ReplicationStrategy::SemiSynchronous { min_replicas } => healthy_count >= *min_replicas,
            ReplicationStrategy::RaftConsensus => {
                // For Raft, we need a majority of nodes to be healthy
                let total_nodes = self.replicas.len() + 1; // +1 for local node
                let majority = (total_nodes / 2) + 1;
                healthy_count + 1 >= majority // +1 for local node
            }
        }
    }

    /// Get required replica count for the current strategy
    pub fn required_replica_count(&self) -> usize {
        match &self.strategy {
            ReplicationStrategy::Synchronous => self.replicas.len(),
            ReplicationStrategy::Asynchronous => 0,
            ReplicationStrategy::SemiSynchronous { min_replicas } => *min_replicas,
            ReplicationStrategy::RaftConsensus => {
                let total_nodes = self.replicas.len() + 1;
                (total_nodes / 2) + 1 - 1 // -1 because local node is not a replica
            }
        }
    }

    /// Update internal statistics
    fn update_stats(&mut self) {
        let healthy_replicas_count = self.replicas.values().filter(|r| r.is_healthy).count();
        let healthy_lags: Vec<u64> = self
            .replicas
            .values()
            .filter(|r| r.is_healthy)
            .map(|r| r.replication_lag)
            .collect();
        let healthy_latencies: Vec<Duration> = self
            .replicas
            .values()
            .filter(|r| r.is_healthy)
            .map(|r| r.latency)
            .collect();

        self.stats.total_replicas = self.replicas.len();
        self.stats.healthy_replicas = healthy_replicas_count;

        if !healthy_lags.is_empty() {
            let total_lag: u64 = healthy_lags.iter().sum();
            self.stats.average_lag = total_lag as f64 / healthy_lags.len() as f64;
            self.stats.max_lag = healthy_lags.iter().copied().max().unwrap_or(0);
            self.stats.min_lag = healthy_lags.iter().copied().min().unwrap_or(0);

            let total_latency: Duration = healthy_latencies.iter().sum();
            self.stats.average_latency = total_latency / healthy_latencies.len() as u32;
        } else {
            self.stats.average_lag = 0.0;
            self.stats.max_lag = 0;
            self.stats.min_lag = 0;
            self.stats.average_latency = Duration::from_millis(0);
        }
    }

    /// Run periodic maintenance tasks
    pub async fn run_maintenance(&mut self) {
        const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(30);
        const STALE_THRESHOLD: Duration = Duration::from_secs(60);

        loop {
            sleep(HEALTH_CHECK_INTERVAL).await;

            self.health_check(STALE_THRESHOLD).await;

            // Log stats periodically
            if self.stats.total_replicas > 0 {
                tracing::debug!(
                    "Replication stats: {}/{} healthy, avg lag: {:.1}, max lag: {}",
                    self.stats.healthy_replicas,
                    self.stats.total_replicas,
                    self.stats.average_lag,
                    self.stats.max_lag
                );
            }
        }
    }
}

/// Replication-related errors
#[derive(Debug, thiserror::Error)]
pub enum ReplicationError {
    #[error("Insufficient replicas: need {required}, have {available}")]
    InsufficientReplicas { required: usize, available: usize },

    #[error("Replica {node_id} is unhealthy")]
    UnhealthyReplica { node_id: OxirsNodeId },

    #[error("Replication timeout after {timeout:?}")]
    Timeout { timeout: Duration },

    #[error("Network error: {message}")]
    Network { message: String },

    #[error("Serialization error: {message}")]
    Serialization { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replication_strategy_default() {
        let strategy = ReplicationStrategy::default();
        assert_eq!(strategy, ReplicationStrategy::RaftConsensus);
    }

    #[test]
    fn test_replica_info_creation() {
        let replica = ReplicaInfo::new(1, "127.0.0.1:8080".to_string());

        assert_eq!(replica.node_id, 1);
        assert_eq!(replica.address, "127.0.0.1:8080");
        assert_eq!(replica.last_applied_index, 0);
        assert!(replica.is_healthy);
        assert_eq!(replica.replication_lag, 0);
        assert_eq!(replica.latency, Duration::from_millis(0));
    }

    #[test]
    fn test_replica_info_staleness() {
        let replica = ReplicaInfo::new(1, "127.0.0.1:8080".to_string());

        // Fresh replica should not be stale
        assert!(!replica.is_stale(Duration::from_secs(10)));

        // Wait a tiny bit to ensure elapsed time passes the threshold
        std::thread::sleep(Duration::from_micros(1));

        // Simulate old replica by checking against very short threshold
        assert!(replica.is_stale(Duration::from_nanos(1)));
    }

    #[test]
    fn test_replica_info_update_health() {
        let mut replica = ReplicaInfo::new(1, "127.0.0.1:8080".to_string());

        // Update to unhealthy
        replica.update_health(false);
        assert!(!replica.is_healthy);

        // Update to healthy
        replica.update_health(true);
        assert!(replica.is_healthy);
    }

    #[test]
    fn test_replication_manager_creation() {
        let manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);

        assert_eq!(manager.strategy, ReplicationStrategy::Synchronous);
        assert_eq!(manager.local_node_id, 1);
        assert!(manager.replicas.is_empty());
        assert_eq!(manager.stats.total_replicas, 0);
    }

    #[test]
    fn test_replication_manager_with_raft_consensus() {
        let manager = ReplicationManager::with_raft_consensus(1);

        assert_eq!(manager.strategy, ReplicationStrategy::RaftConsensus);
        assert_eq!(manager.local_node_id, 1);
    }

    #[test]
    fn test_replication_manager_add_replica() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);

        // Add replica
        assert!(manager.add_replica(2, "127.0.0.1:8081".to_string()));
        assert_eq!(manager.replicas.len(), 1);
        assert!(manager.replicas.contains_key(&2));

        // Adding same replica again should return false
        assert!(!manager.add_replica(2, "127.0.0.1:8081".to_string()));
        assert_eq!(manager.replicas.len(), 1);

        // Cannot add local node as replica
        assert!(!manager.add_replica(1, "127.0.0.1:8080".to_string()));
        assert_eq!(manager.replicas.len(), 1);
    }

    #[test]
    fn test_replication_manager_remove_replica() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);

        manager.add_replica(2, "127.0.0.1:8081".to_string());
        manager.add_replica(3, "127.0.0.1:8082".to_string());
        assert_eq!(manager.replicas.len(), 2);

        // Remove replica
        assert!(manager.remove_replica(2));
        assert_eq!(manager.replicas.len(), 1);
        assert!(!manager.replicas.contains_key(&2));

        // Removing non-existent replica should return false
        assert!(!manager.remove_replica(4));
        assert_eq!(manager.replicas.len(), 1);
    }

    #[test]
    fn test_replication_manager_get_healthy_replicas() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);

        manager.add_replica(2, "127.0.0.1:8081".to_string());
        manager.add_replica(3, "127.0.0.1:8082".to_string());

        // Mark one replica as unhealthy
        manager.update_replica_health(3, false);

        let healthy_replicas = manager.get_healthy_replicas();
        assert_eq!(healthy_replicas.len(), 1);
        assert_eq!(healthy_replicas[0].node_id, 2);
    }

    #[test]
    fn test_replication_manager_update_replica_health() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);

        manager.add_replica(2, "127.0.0.1:8081".to_string());

        // Update health status
        assert!(manager.update_replica_health(2, false));
        let replica = manager.get_replica(2).unwrap();
        assert!(!replica.is_healthy);

        // Update non-existent replica should return false
        assert!(!manager.update_replica_health(3, true));
    }

    #[test]
    fn test_replication_manager_update_replica_lag() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);

        manager.add_replica(2, "127.0.0.1:8081".to_string());

        // Update lag information
        manager.update_replica_lag(2, 50, 100);
        let replica = manager.get_replica(2).unwrap();
        assert_eq!(replica.last_applied_index, 50);
        assert_eq!(replica.replication_lag, 50);
    }

    #[tokio::test]
    async fn test_replication_manager_health_check() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);

        manager.add_replica(2, "127.0.0.1:8081".to_string());
        manager.add_replica(3, "127.0.0.1:8082".to_string());

        // Both replicas should be healthy initially
        assert_eq!(manager.get_healthy_replicas().len(), 2);

        // Run health check with very short threshold - should mark all as unhealthy
        manager.health_check(Duration::from_nanos(1)).await;
        assert_eq!(manager.get_healthy_replicas().len(), 0);
    }

    #[test]
    fn test_replication_manager_strategy_change() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);

        assert_eq!(manager.get_strategy(), &ReplicationStrategy::Synchronous);

        manager.set_strategy(ReplicationStrategy::Asynchronous);
        assert_eq!(manager.get_strategy(), &ReplicationStrategy::Asynchronous);
    }

    #[test]
    fn test_replication_manager_health_status() {
        let mut manager =
            ReplicationManager::new(ReplicationStrategy::SemiSynchronous { min_replicas: 2 }, 1);

        // Add replicas
        manager.add_replica(2, "127.0.0.1:8081".to_string());
        manager.add_replica(3, "127.0.0.1:8082".to_string());
        manager.add_replica(4, "127.0.0.1:8083".to_string());

        // All healthy - should meet requirements
        assert!(manager.is_replication_healthy());

        // Mark one as unhealthy - should still meet requirements (2 out of 3)
        manager.update_replica_health(4, false);
        assert!(manager.is_replication_healthy());

        // Mark another as unhealthy - should not meet requirements (1 out of 3)
        manager.update_replica_health(3, false);
        assert!(!manager.is_replication_healthy());
    }

    #[test]
    fn test_replication_manager_required_replica_count() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::Synchronous, 1);
        manager.add_replica(2, "127.0.0.1:8081".to_string());
        manager.add_replica(3, "127.0.0.1:8082".to_string());

        // Synchronous requires all replicas
        assert_eq!(manager.required_replica_count(), 2);

        // Asynchronous requires no replicas
        manager.set_strategy(ReplicationStrategy::Asynchronous);
        assert_eq!(manager.required_replica_count(), 0);

        // Semi-synchronous requires minimum specified
        manager.set_strategy(ReplicationStrategy::SemiSynchronous { min_replicas: 1 });
        assert_eq!(manager.required_replica_count(), 1);

        // Raft consensus requires majority
        manager.set_strategy(ReplicationStrategy::RaftConsensus);
        // With 2 replicas + 1 local = 3 total, majority is 2, so we need 1 replica (since local is always available)
        assert_eq!(manager.required_replica_count(), 1);
    }

    #[test]
    fn test_replication_manager_raft_consensus_health() {
        let mut manager = ReplicationManager::new(ReplicationStrategy::RaftConsensus, 1);

        // Single node cluster (1 local + 0 replicas = 1 total) - should be healthy
        assert!(manager.is_replication_healthy());

        // Add replicas to form 3-node cluster
        manager.add_replica(2, "127.0.0.1:8081".to_string());
        manager.add_replica(3, "127.0.0.1:8082".to_string());

        // All healthy (3 total, majority = 2, local + 2 replicas = 3) - should be healthy
        assert!(manager.is_replication_healthy());

        // Mark one replica as unhealthy (local + 1 replica = 2, still majority) - should be healthy
        manager.update_replica_health(3, false);
        assert!(manager.is_replication_healthy());

        // Mark both replicas as unhealthy (only local = 1, not majority) - should not be healthy
        manager.update_replica_health(2, false);
        assert!(!manager.is_replication_healthy());
    }

    #[test]
    fn test_replication_stats_default() {
        let stats = ReplicationStats::default();
        assert_eq!(stats.total_replicas, 0);
        assert_eq!(stats.healthy_replicas, 0);
        assert_eq!(stats.average_lag, 0.0);
        assert_eq!(stats.max_lag, 0);
        assert_eq!(stats.min_lag, 0);
        assert_eq!(stats.average_latency, Duration::from_millis(0));
        assert_eq!(stats.replication_throughput, 0.0);
    }

    #[test]
    fn test_replication_error_display() {
        let err = ReplicationError::InsufficientReplicas {
            required: 3,
            available: 1,
        };
        assert!(err
            .to_string()
            .contains("Insufficient replicas: need 3, have 1"));

        let err = ReplicationError::UnhealthyReplica { node_id: 42 };
        assert!(err.to_string().contains("Replica 42 is unhealthy"));

        let err = ReplicationError::Timeout {
            timeout: Duration::from_secs(5),
        };
        assert!(err.to_string().contains("Replication timeout after 5s"));

        let err = ReplicationError::Network {
            message: "connection failed".to_string(),
        };
        assert!(err.to_string().contains("Network error: connection failed"));

        let err = ReplicationError::Serialization {
            message: "json error".to_string(),
        };
        assert!(err.to_string().contains("Serialization error: json error"));
    }
}
