//! # Data Replication
//!
//! High-level data replication management for distributed RDF storage.
//! Works with Raft consensus to ensure consistent replication.

use anyhow::Result;
use std::collections::{HashMap, BTreeSet};
use std::time::{Duration, SystemTime};
use tokio::time::sleep;
use crate::raft::{RdfCommand, RdfResponse, OxirsNodeId};

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
        self.replicas.values()
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
    pub fn update_replica_lag(&mut self, node_id: OxirsNodeId, applied_index: u64, current_index: u64) {
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
            tracing::info!("Changing replication strategy from {:?} to {:?}", self.strategy, strategy);
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
        let healthy_lags: Vec<u64> = self.replicas.values().filter(|r| r.is_healthy).map(|r| r.replication_lag).collect();
        let healthy_latencies: Vec<Duration> = self.replicas.values().filter(|r| r.is_healthy).map(|r| r.latency).collect();
        
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