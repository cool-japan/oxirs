//! Shard manager for coordinating distributed shards
//!
//! This module manages the lifecycle of shards including creation,
//! splitting, merging, and migration operations.

use crate::shard::{ShardId, ShardMetadata, ShardRouter, ShardState, ShardingStrategy};
use crate::storage::StorageBackend;
use crate::network::{NetworkService, RpcMessage};
use crate::raft::OxirsNodeId;
use crate::{ClusterError, Result};
use oxirs_core::model::Triple;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, error, info, warn};

/// Shard operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardOperation {
    /// Create a new shard
    Create {
        shard_id: ShardId,
        node_ids: Vec<OxirsNodeId>,
    },
    /// Split a shard into multiple shards
    Split {
        source_shard: ShardId,
        target_shards: Vec<ShardId>,
        split_points: Vec<String>,
    },
    /// Merge multiple shards into one
    Merge {
        source_shards: Vec<ShardId>,
        target_shard: ShardId,
    },
    /// Migrate a shard to different nodes
    Migrate {
        shard_id: ShardId,
        from_nodes: Vec<OxirsNodeId>,
        to_nodes: Vec<OxirsNodeId>,
    },
    /// Rebalance shards across nodes
    Rebalance {
        rebalance_plan: RebalancePlan,
    },
}

/// Rebalance plan for shard distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancePlan {
    /// Shard movements
    pub movements: Vec<ShardMovement>,
    /// Expected improvement in balance
    pub balance_improvement: f64,
    /// Estimated data transfer size
    pub data_transfer_bytes: u64,
}

/// Individual shard movement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMovement {
    pub shard_id: ShardId,
    pub from_node: OxirsNodeId,
    pub to_node: OxirsNodeId,
    pub triple_count: usize,
}

/// Shard manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManagerConfig {
    /// Replication factor for each shard
    pub replication_factor: usize,
    /// Maximum triples per shard before splitting
    pub max_triples_per_shard: usize,
    /// Minimum triples per shard before merging
    pub min_triples_per_shard: usize,
    /// Maximum imbalance ratio before rebalancing
    pub max_imbalance_ratio: f64,
    /// Enable automatic shard management
    pub auto_manage: bool,
    /// Check interval for auto management
    pub check_interval_secs: u64,
}

impl Default for ShardManagerConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            max_triples_per_shard: 10_000_000,
            min_triples_per_shard: 100_000,
            max_imbalance_ratio: 2.0,
            auto_manage: true,
            check_interval_secs: 60,
        }
    }
}

/// Shard manager for coordinating distributed shards
pub struct ShardManager {
    /// Node ID
    node_id: OxirsNodeId,
    /// Shard router
    router: Arc<ShardRouter>,
    /// Configuration
    config: ShardManagerConfig,
    /// Storage backend
    storage: Arc<dyn StorageBackend>,
    /// Network service
    network: Arc<NetworkService>,
    /// Shard ownership mapping (shard_id -> node_ids)
    shard_ownership: Arc<RwLock<HashMap<ShardId, HashSet<OxirsNodeId>>>>,
    /// Active operations
    active_operations: Arc<RwLock<HashMap<String, ShardOperation>>>,
    /// Operation sender
    operation_tx: mpsc::Sender<ShardOperation>,
    /// Operation receiver
    operation_rx: Arc<RwLock<mpsc::Receiver<ShardOperation>>>,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new(
        node_id: OxirsNodeId,
        router: Arc<ShardRouter>,
        config: ShardManagerConfig,
        storage: Arc<dyn StorageBackend>,
        network: Arc<NetworkService>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(100);
        
        Self {
            node_id,
            router,
            config,
            storage,
            network,
            shard_ownership: Arc::new(RwLock::new(HashMap::new())),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            operation_tx: tx,
            operation_rx: Arc::new(RwLock::new(rx)),
        }
    }
    
    /// Start the shard manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting shard manager on node {}", self.node_id);
        
        // Start operation processor
        self.start_operation_processor().await;
        
        // Start auto-management if enabled
        if self.config.auto_manage {
            self.start_auto_management().await;
        }
        
        Ok(())
    }
    
    /// Initialize shards based on strategy
    pub async fn initialize_shards(
        &self,
        strategy: &ShardingStrategy,
        nodes: Vec<OxirsNodeId>,
    ) -> Result<()> {
        let num_shards = match strategy {
            ShardingStrategy::Hash { num_shards } |
            ShardingStrategy::Subject { num_shards } => *num_shards,
            ShardingStrategy::Predicate { predicate_groups } => predicate_groups.len() as u32,
            ShardingStrategy::Namespace { namespace_mapping } => namespace_mapping.len() as u32,
            ShardingStrategy::Graph { graph_mapping } => graph_mapping.len() as u32,
            ShardingStrategy::Semantic { concept_clusters, .. } => concept_clusters.len() as u32,
            ShardingStrategy::Hybrid { .. } => 4, // Default for hybrid
        };
        
        // Initialize router shards
        self.router.init_shards(num_shards, self.config.replication_factor).await?;
        
        // Assign nodes to shards
        let nodes_per_shard = (nodes.len() / num_shards as usize).max(self.config.replication_factor);
        let mut node_iter = nodes.into_iter().cycle();
        
        for shard_id in 0..num_shards {
            let mut shard_nodes = Vec::new();
            for _ in 0..nodes_per_shard.min(self.config.replication_factor) {
                if let Some(node) = node_iter.next() {
                    shard_nodes.push(node);
                }
            }
            
            // Create shard on assigned nodes
            self.create_shard(shard_id, shard_nodes).await?;
        }
        
        info!("Initialized {} shards", num_shards);
        Ok(())
    }
    
    /// Create a new shard
    async fn create_shard(&self, shard_id: ShardId, node_ids: Vec<OxirsNodeId>) -> Result<()> {
        if node_ids.is_empty() {
            return Err(ClusterError::Config("No nodes assigned to shard".to_string()));
        }
        
        // Update ownership
        self.shard_ownership.write().await.insert(shard_id, node_ids.iter().copied().collect());
        
        // Create shard metadata
        let metadata = ShardMetadata {
            shard_id,
            node_ids: node_ids.iter().map(|&id| id as u64).collect(),
            primary_node: node_ids[0] as u64,
            triple_count: 0,
            size_bytes: 0,
            state: ShardState::Active,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        // Update router metadata
        self.router.update_shard_metadata(metadata).await?;
        
        // Notify nodes to create shard storage
        for &node_id in &node_ids {
            if node_id == self.node_id {
                // Create local shard storage
                self.storage.create_shard(shard_id).await?;
            } else {
                // Send create shard message to remote node
                let msg = RpcMessage::ShardOperation(ShardOperation::Create {
                    shard_id,
                    node_ids: node_ids.clone(),
                });
                self.network.send_message(node_id, msg).await?;
            }
        }
        
        info!("Created shard {} on nodes {:?}", shard_id, node_ids);
        Ok(())
    }
    
    /// Route and store a triple
    pub async fn store_triple(&self, triple: Triple) -> Result<()> {
        // Route triple to appropriate shard
        let shard_id = self.router.route_triple(&triple).await?;
        
        // Check if we own this shard
        let ownership = self.shard_ownership.read().await;
        if let Some(owners) = ownership.get(&shard_id) {
            if owners.contains(&self.node_id) {
                // Store locally
                self.storage.insert_triple_to_shard(shard_id, triple.clone()).await?;
                
                // Replicate to other owners
                for &owner in owners {
                    if owner != self.node_id {
                        let msg = RpcMessage::ReplicateTriple {
                            shard_id,
                            triple: triple.clone(),
                        };
                        self.network.send_message(owner, msg).await?;
                    }
                }
            } else {
                // Forward to primary owner
                if let Some(metadata) = self.router.get_shard_metadata(shard_id).await {
                    let primary = metadata.primary_node as OxirsNodeId;
                    let msg = RpcMessage::StoreTriple {
                        shard_id,
                        triple,
                    };
                    self.network.send_message(primary, msg).await?;
                } else {
                    return Err(ClusterError::Other(format!("Shard {} not found", shard_id)));
                }
            }
        } else {
            return Err(ClusterError::Other(format!("No owners found for shard {}", shard_id)));
        }
        
        Ok(())
    }
    
    /// Query triples from shards
    pub async fn query_triples(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<Triple>> {
        // Determine which shards to query
        let shard_ids = self.router.route_query_pattern(subject, predicate, object).await?;
        
        let mut all_results = Vec::new();
        
        for shard_id in shard_ids {
            let ownership = self.shard_ownership.read().await;
            if let Some(owners) = ownership.get(&shard_id) {
                if owners.contains(&self.node_id) {
                    // Query local shard
                    let results = self.storage.query_shard(shard_id, subject, predicate, object).await?;
                    all_results.extend(results);
                } else {
                    // Query remote shard
                    if let Some(metadata) = self.router.get_shard_metadata(shard_id).await {
                        let primary = metadata.primary_node as OxirsNodeId;
                        let msg = RpcMessage::QueryShard {
                            shard_id,
                            subject: subject.map(String::from),
                            predicate: predicate.map(String::from),
                            object: object.map(String::from),
                        };
                        
                        // Send query and wait for response
                        // In a real implementation, this would use a request-response pattern
                        self.network.send_message(primary, msg).await?;
                    }
                }
            }
        }
        
        Ok(all_results)
    }
    
    /// Get the primary node for a shard
    pub async fn get_primary_node(&self, shard_id: ShardId) -> Result<OxirsNodeId> {
        if let Some(metadata) = self.router.get_shard_metadata(shard_id).await {
            Ok(metadata.primary_node as OxirsNodeId)
        } else {
            Err(ClusterError::ShardNotFound(shard_id).into())
        }
    }
    
    /// Check if a shard needs splitting
    async fn check_shard_split(&self, shard_id: ShardId) -> Result<bool> {
        if let Some(metadata) = self.router.get_shard_metadata(shard_id).await {
            Ok(metadata.triple_count > self.config.max_triples_per_shard)
        } else {
            Ok(false)
        }
    }
    
    /// Check if shards need merging
    async fn check_shard_merge(&self, shard_ids: &[ShardId]) -> Result<bool> {
        let mut total_triples = 0;
        
        for &shard_id in shard_ids {
            if let Some(metadata) = self.router.get_shard_metadata(shard_id).await {
                total_triples += metadata.triple_count;
            }
        }
        
        Ok(total_triples < self.config.min_triples_per_shard * shard_ids.len())
    }
    
    /// Calculate load imbalance
    async fn calculate_imbalance(&self) -> Result<f64> {
        let stats = self.router.get_statistics().await;
        
        if stats.distribution.is_empty() {
            return Ok(0.0);
        }
        
        let avg_load = stats.total_triples as f64 / stats.distribution.len() as f64;
        let max_load = stats.distribution.iter()
            .map(|d| d.triple_count as f64)
            .fold(0.0, f64::max);
        let min_load = stats.distribution.iter()
            .map(|d| d.triple_count as f64)
            .fold(f64::INFINITY, f64::min);
        
        if avg_load > 0.0 {
            Ok(max_load / min_load)
        } else {
            Ok(0.0)
        }
    }
    
    /// Start operation processor
    async fn start_operation_processor(&self) {
        let rx = self.operation_rx.clone();
        let active_ops = self.active_operations.clone();
        let storage = self.storage.clone();
        let network = self.network.clone();
        let node_id = self.node_id;
        
        tokio::spawn(async move {
            let mut rx = rx.write().await;
            while let Some(operation) = rx.recv().await {
                let op_id = uuid::Uuid::new_v4().to_string();
                active_ops.write().await.insert(op_id.clone(), operation.clone());
                
                match Self::process_operation(operation, storage.clone(), network.clone(), node_id).await {
                    Ok(()) => {
                        info!("Completed shard operation {}", op_id);
                    }
                    Err(e) => {
                        error!("Failed to process shard operation {}: {}", op_id, e);
                    }
                }
                
                active_ops.write().await.remove(&op_id);
            }
        });
    }
    
    /// Process a shard operation
    async fn process_operation(
        operation: ShardOperation,
        storage: Arc<dyn StorageBackend>,
        network: Arc<NetworkService>,
        node_id: OxirsNodeId,
    ) -> Result<()> {
        match operation {
            ShardOperation::Create { shard_id, node_ids } => {
                if node_ids.contains(&node_id) {
                    storage.create_shard(shard_id).await?;
                }
            }
            
            ShardOperation::Split { source_shard, target_shards, split_points } => {
                // TODO: Implement shard splitting logic
                warn!("Shard split not yet implemented");
            }
            
            ShardOperation::Merge { source_shards, target_shard } => {
                // TODO: Implement shard merging logic
                warn!("Shard merge not yet implemented");
            }
            
            ShardOperation::Migrate { shard_id, from_nodes, to_nodes } => {
                // TODO: Implement shard migration logic
                warn!("Shard migration not yet implemented");
            }
            
            ShardOperation::Rebalance { rebalance_plan } => {
                // TODO: Implement rebalancing logic
                warn!("Shard rebalancing not yet implemented");
            }
        }
        
        Ok(())
    }
    
    /// Start automatic shard management
    async fn start_auto_management(&self) {
        let config = self.config.clone();
        let router = self.router.clone();
        let tx = self.operation_tx.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(config.check_interval_secs)
            );
            
            loop {
                interval.tick().await;
                
                // Check for shards that need splitting
                let stats = router.get_statistics().await;
                for dist in &stats.distribution {
                    if dist.triple_count > config.max_triples_per_shard {
                        warn!("Shard {} needs splitting ({}  triples)", dist.shard_id, dist.triple_count);
                        // TODO: Create split operation
                    }
                }
                
                // Check for load imbalance
                if stats.distribution.len() > 1 {
                    let max_load = stats.distribution.iter().map(|d| d.triple_count).max().unwrap_or(0);
                    let min_load = stats.distribution.iter().map(|d| d.triple_count).min().unwrap_or(0);
                    
                    if min_load > 0 && (max_load as f64 / min_load as f64) > config.max_imbalance_ratio {
                        warn!("Shard imbalance detected: max={}, min={}", max_load, min_load);
                        // TODO: Create rebalance operation
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::MockStorageBackend;
    use crate::network::NetworkConfig;
    
    #[tokio::test]
    async fn test_shard_manager_creation() {
        let strategy = ShardingStrategy::Hash { num_shards: 4 };
        let router = Arc::new(ShardRouter::new(strategy));
        let config = ShardManagerConfig::default();
        let storage = Arc::new(MockStorageBackend::new());
        let network = Arc::new(NetworkService::new(1, NetworkConfig::default()));
        
        let manager = ShardManager::new(1, router, config, storage, network);
        assert_eq!(manager.node_id, 1);
    }
    
    #[tokio::test]
    async fn test_shard_initialization() {
        let strategy = ShardingStrategy::Hash { num_shards: 2 };
        let router = Arc::new(ShardRouter::new(strategy.clone()));
        let config = ShardManagerConfig {
            replication_factor: 2,
            ..Default::default()
        };
        let storage = Arc::new(MockStorageBackend::new());
        let network = Arc::new(NetworkService::new(1, NetworkConfig::default()));
        
        let manager = ShardManager::new(1, router, config, storage, network);
        
        let nodes = vec![1, 2, 3, 4];
        manager.initialize_shards(&strategy, nodes).await.unwrap();
        
        // Check that shards were created
        let ownership = manager.shard_ownership.read().await;
        assert_eq!(ownership.len(), 2);
    }
}