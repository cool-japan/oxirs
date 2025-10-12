//! # Automatic Data Rebalancing
//!
//! Provides intelligent data rebalancing with live migration support:
//! - Zero-downtime data migration
//! - Load-based automatic triggering
//! - Incremental and bulk migration strategies
//! - Bandwidth throttling
//! - Progress tracking and rollback
//! - Comprehensive statistics

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::raft::OxirsNodeId;

/// Rebalancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalancingStrategy {
    /// Incremental migration (small batches)
    Incremental,
    /// Bulk migration (large batches)
    Bulk,
    /// Adaptive (adjusts based on load)
    Adaptive,
}

/// Migration state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationState {
    /// Not migrating
    Idle,
    /// Planning migration
    Planning,
    /// Migrating data
    InProgress,
    /// Verifying migration
    Verifying,
    /// Migration completed
    Completed,
    /// Migration failed
    Failed,
    /// Rolling back
    RollingBack,
}

/// Rebalancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingConfig {
    /// Rebalancing strategy
    pub strategy: RebalancingStrategy,
    /// Enable automatic rebalancing
    pub enable_auto_rebalancing: bool,
    /// Load imbalance threshold (0.0-1.0)
    pub load_imbalance_threshold: f64,
    /// Minimum nodes for rebalancing
    pub min_nodes_for_rebalancing: usize,
    /// Batch size for incremental migration
    pub incremental_batch_size: usize,
    /// Batch size for bulk migration
    pub bulk_batch_size: usize,
    /// Bandwidth limit (bytes per second, 0 = unlimited)
    pub bandwidth_limit_bytes_per_sec: usize,
    /// Enable migration verification
    pub enable_verification: bool,
    /// Enable rollback on failure
    pub enable_rollback: bool,
    /// Migration timeout (seconds)
    pub migration_timeout_secs: u64,
}

impl Default for RebalancingConfig {
    fn default() -> Self {
        Self {
            strategy: RebalancingStrategy::Adaptive,
            enable_auto_rebalancing: true,
            load_imbalance_threshold: 0.2, // 20% imbalance triggers rebalancing
            min_nodes_for_rebalancing: 2,
            incremental_batch_size: 100,
            bulk_batch_size: 10000,
            bandwidth_limit_bytes_per_sec: 10 * 1024 * 1024, // 10 MB/s
            enable_verification: true,
            enable_rollback: true,
            migration_timeout_secs: 3600, // 1 hour
        }
    }
}

/// Migration plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Plan ID
    pub plan_id: String,
    /// Source node
    pub source_node: OxirsNodeId,
    /// Target node
    pub target_node: OxirsNodeId,
    /// Number of keys to migrate
    pub key_count: usize,
    /// Estimated data size (bytes)
    pub estimated_size_bytes: usize,
    /// Partition IDs to migrate
    pub partition_ids: Vec<usize>,
    /// Created timestamp
    pub created_at: SystemTime,
    /// Estimated duration (seconds)
    pub estimated_duration_secs: u64,
}

/// Migration progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationProgress {
    /// Plan ID
    pub plan_id: String,
    /// Current state
    pub state: MigrationState,
    /// Keys migrated
    pub keys_migrated: usize,
    /// Total keys
    pub total_keys: usize,
    /// Bytes migrated
    pub bytes_migrated: usize,
    /// Total bytes
    pub total_bytes: usize,
    /// Started at
    pub started_at: Option<SystemTime>,
    /// Completed at
    pub completed_at: Option<SystemTime>,
    /// Errors encountered
    pub errors: Vec<String>,
}

impl MigrationProgress {
    fn new(plan_id: String, total_keys: usize, total_bytes: usize) -> Self {
        Self {
            plan_id,
            state: MigrationState::Idle,
            keys_migrated: 0,
            total_keys,
            bytes_migrated: 0,
            total_bytes,
            started_at: None,
            completed_at: None,
            errors: Vec::new(),
        }
    }

    pub fn progress_percentage(&self) -> f64 {
        if self.total_keys == 0 {
            return 0.0;
        }
        (self.keys_migrated as f64 / self.total_keys as f64) * 100.0
    }
}

/// Node load statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLoad {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Number of keys
    pub key_count: usize,
    /// Data size (bytes)
    pub data_size_bytes: usize,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f64,
    /// Network bandwidth usage (bytes/sec)
    pub network_bandwidth_usage: usize,
    /// Last updated
    pub last_updated: SystemTime,
}

impl Default for NodeLoad {
    fn default() -> Self {
        Self {
            node_id: 0,
            key_count: 0,
            data_size_bytes: 0,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            network_bandwidth_usage: 0,
            last_updated: SystemTime::now(),
        }
    }
}

/// Rebalancing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingStats {
    /// Total rebalancing operations
    pub total_rebalancing_ops: u64,
    /// Successful migrations
    pub successful_migrations: u64,
    /// Failed migrations
    pub failed_migrations: u64,
    /// Total keys migrated
    pub total_keys_migrated: usize,
    /// Total bytes migrated
    pub total_bytes_migrated: usize,
    /// Average migration speed (bytes/sec)
    pub avg_migration_speed_bytes_per_sec: f64,
    /// Last rebalancing time
    pub last_rebalancing: Option<SystemTime>,
    /// Average rebalancing duration (ms)
    pub avg_rebalancing_duration_ms: f64,
}

impl Default for RebalancingStats {
    fn default() -> Self {
        Self {
            total_rebalancing_ops: 0,
            successful_migrations: 0,
            failed_migrations: 0,
            total_keys_migrated: 0,
            total_bytes_migrated: 0,
            avg_migration_speed_bytes_per_sec: 0.0,
            last_rebalancing: None,
            avg_rebalancing_duration_ms: 0.0,
        }
    }
}

/// Data rebalancing manager
pub struct DataRebalancingManager {
    config: RebalancingConfig,
    /// Node load statistics
    node_loads: Arc<RwLock<BTreeMap<OxirsNodeId, NodeLoad>>>,
    /// Active migrations
    active_migrations: Arc<RwLock<HashMap<String, MigrationProgress>>>,
    /// Migration history
    migration_history: Arc<RwLock<Vec<MigrationProgress>>>,
    /// Statistics
    stats: Arc<RwLock<RebalancingStats>>,
}

impl DataRebalancingManager {
    /// Create a new data rebalancing manager
    pub fn new(config: RebalancingConfig) -> Self {
        Self {
            config,
            node_loads: Arc::new(RwLock::new(BTreeMap::new())),
            active_migrations: Arc::new(RwLock::new(HashMap::new())),
            migration_history: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(RebalancingStats::default())),
        }
    }

    /// Register a node
    pub async fn register_node(&self, node_id: OxirsNodeId) {
        let mut node_loads = self.node_loads.write().await;
        node_loads.insert(
            node_id,
            NodeLoad {
                node_id,
                ..Default::default()
            },
        );
        info!("Registered node {} for rebalancing", node_id);
    }

    /// Update node load statistics
    pub async fn update_node_load(
        &self,
        node_id: OxirsNodeId,
        key_count: usize,
        data_size_bytes: usize,
        cpu_utilization: f64,
        memory_utilization: f64,
        network_bandwidth: usize,
    ) {
        let mut node_loads = self.node_loads.write().await;
        if let Some(load) = node_loads.get_mut(&node_id) {
            load.key_count = key_count;
            load.data_size_bytes = data_size_bytes;
            load.cpu_utilization = cpu_utilization;
            load.memory_utilization = memory_utilization;
            load.network_bandwidth_usage = network_bandwidth;
            load.last_updated = SystemTime::now();
        }
    }

    /// Check if rebalancing is needed
    pub async fn check_rebalancing_needed(&self) -> bool {
        if !self.config.enable_auto_rebalancing {
            return false;
        }

        let node_loads = self.node_loads.read().await;

        if node_loads.len() < self.config.min_nodes_for_rebalancing {
            return false;
        }

        // Calculate load imbalance
        let loads: Vec<usize> = node_loads.values().map(|l| l.data_size_bytes).collect();

        if loads.is_empty() {
            return false;
        }

        let max_load = loads.iter().max().unwrap_or(&0);
        let min_load = loads.iter().min().unwrap_or(&0);
        let avg_load = loads.iter().sum::<usize>() / loads.len();

        if avg_load == 0 {
            return false;
        }

        let imbalance = (*max_load as f64 - *min_load as f64) / avg_load as f64;

        imbalance > self.config.load_imbalance_threshold
    }

    /// Create a migration plan
    pub async fn create_migration_plan(&self) -> Result<MigrationPlan, String> {
        let node_loads = self.node_loads.read().await;

        if node_loads.len() < 2 {
            return Err("Not enough nodes for migration".to_string());
        }

        // Find most loaded and least loaded nodes
        let mut loads: Vec<_> = node_loads.values().cloned().collect();
        loads.sort_by_key(|l| l.data_size_bytes);

        let source_load = loads.last().unwrap();
        let target_load = loads.first().unwrap();

        // Calculate how much data to migrate
        let total_data: usize = loads.iter().map(|l| l.data_size_bytes).sum();
        let avg_data = total_data / loads.len();

        let migrate_size = (source_load.data_size_bytes - avg_data) / 2; // Move half the excess
        let migrate_keys = (migrate_size as f64 / source_load.data_size_bytes as f64
            * source_load.key_count as f64) as usize;

        // Estimate duration based on bandwidth
        let estimated_duration_secs = if self.config.bandwidth_limit_bytes_per_sec > 0 {
            migrate_size as u64 / self.config.bandwidth_limit_bytes_per_sec as u64
        } else {
            60 // Assume 60 seconds if no limit
        };

        let plan_id = format!(
            "migration-{}-to-{}-{}",
            source_load.node_id,
            target_load.node_id,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        Ok(MigrationPlan {
            plan_id,
            source_node: source_load.node_id,
            target_node: target_load.node_id,
            key_count: migrate_keys,
            estimated_size_bytes: migrate_size,
            partition_ids: vec![0], // Simplified: migrate partition 0
            created_at: SystemTime::now(),
            estimated_duration_secs,
        })
    }

    /// Execute a migration plan
    pub async fn execute_migration(&self, plan: MigrationPlan) -> Result<(), String> {
        let start = std::time::Instant::now();

        let mut progress = MigrationProgress::new(
            plan.plan_id.clone(),
            plan.key_count,
            plan.estimated_size_bytes,
        );

        progress.state = MigrationState::Planning;
        progress.started_at = Some(SystemTime::now());

        // Store in active migrations
        self.active_migrations
            .write()
            .await
            .insert(plan.plan_id.clone(), progress.clone());

        info!(
            "Starting migration {} from node {} to node {} ({} keys, {} bytes)",
            plan.plan_id,
            plan.source_node,
            plan.target_node,
            plan.key_count,
            plan.estimated_size_bytes
        );

        // Determine batch size based on strategy
        let batch_size = match self.config.strategy {
            RebalancingStrategy::Incremental => self.config.incremental_batch_size,
            RebalancingStrategy::Bulk => self.config.bulk_batch_size,
            RebalancingStrategy::Adaptive => {
                // Adaptive: use incremental for high load, bulk for low load
                let node_loads = self.node_loads.read().await;
                if let Some(source_load) = node_loads.get(&plan.source_node) {
                    if source_load.cpu_utilization > 0.7 {
                        self.config.incremental_batch_size
                    } else {
                        self.config.bulk_batch_size
                    }
                } else {
                    self.config.incremental_batch_size
                }
            }
        };

        // Migration phase
        progress.state = MigrationState::InProgress;
        self.active_migrations
            .write()
            .await
            .insert(plan.plan_id.clone(), progress.clone());

        let mut migrated_keys = 0;
        let mut migrated_bytes = 0;

        while migrated_keys < plan.key_count {
            let batch_keys = (plan.key_count - migrated_keys).min(batch_size);
            let batch_bytes = (batch_keys as f64 / plan.key_count as f64
                * plan.estimated_size_bytes as f64) as usize;

            // Simulate migration with bandwidth throttling
            if self.config.bandwidth_limit_bytes_per_sec > 0 {
                let sleep_duration = Duration::from_secs_f64(
                    batch_bytes as f64 / self.config.bandwidth_limit_bytes_per_sec as f64,
                );
                tokio::time::sleep(sleep_duration).await;
            }

            // Simulate actual data migration
            self.migrate_batch(
                &plan.plan_id,
                plan.source_node,
                plan.target_node,
                batch_keys,
                batch_bytes,
            )
            .await?;

            migrated_keys += batch_keys;
            migrated_bytes += batch_bytes;

            // Update progress
            progress.keys_migrated = migrated_keys;
            progress.bytes_migrated = migrated_bytes;
            self.active_migrations
                .write()
                .await
                .insert(plan.plan_id.clone(), progress.clone());
        }

        // Verification phase
        if self.config.enable_verification {
            progress.state = MigrationState::Verifying;
            self.active_migrations
                .write()
                .await
                .insert(plan.plan_id.clone(), progress.clone());

            self.verify_migration(&plan).await?;
        }

        // Completion
        progress.state = MigrationState::Completed;
        progress.completed_at = Some(SystemTime::now());
        self.active_migrations
            .write()
            .await
            .insert(plan.plan_id.clone(), progress.clone());

        // Update statistics
        let duration = start.elapsed();
        let mut stats = self.stats.write().await;
        stats.total_rebalancing_ops += 1;
        stats.successful_migrations += 1;
        stats.total_keys_migrated += plan.key_count;
        stats.total_bytes_migrated += plan.estimated_size_bytes;
        stats.last_rebalancing = Some(SystemTime::now());

        let total = stats.successful_migrations as f64;
        stats.avg_rebalancing_duration_ms = (stats.avg_rebalancing_duration_ms * (total - 1.0)
            + duration.as_millis() as f64)
            / total;

        let speed = plan.estimated_size_bytes as f64 / duration.as_secs_f64();
        stats.avg_migration_speed_bytes_per_sec =
            (stats.avg_migration_speed_bytes_per_sec * (total - 1.0) + speed) / total;

        // Move to history
        self.migration_history.write().await.push(progress.clone());
        self.active_migrations.write().await.remove(&plan.plan_id);

        info!(
            "Migration {} completed in {:?} ({:.2} MB/s)",
            plan.plan_id,
            duration,
            speed / (1024.0 * 1024.0)
        );

        Ok(())
    }

    /// Migrate a batch of data (simulated)
    async fn migrate_batch(
        &self,
        _plan_id: &str,
        _source: OxirsNodeId,
        _target: OxirsNodeId,
        _keys: usize,
        _bytes: usize,
    ) -> Result<(), String> {
        // In production: actual data transfer
        Ok(())
    }

    /// Verify migration (simulated)
    async fn verify_migration(&self, _plan: &MigrationPlan) -> Result<(), String> {
        // In production: verify data integrity, checksums, etc.
        Ok(())
    }

    /// Rollback a migration
    pub async fn rollback_migration(&self, plan_id: &str) -> Result<(), String> {
        if !self.config.enable_rollback {
            return Err("Rollback is disabled".to_string());
        }

        let mut active = self.active_migrations.write().await;
        if let Some(progress) = active.get_mut(plan_id) {
            warn!("Rolling back migration {}", plan_id);
            progress.state = MigrationState::RollingBack;

            // Simulate rollback
            tokio::time::sleep(Duration::from_secs(1)).await;

            progress.state = MigrationState::Failed;
            progress.completed_at = Some(SystemTime::now());
            progress.errors.push("Migration rolled back".to_string());

            // Update stats
            let mut stats = self.stats.write().await;
            stats.failed_migrations += 1;

            Ok(())
        } else {
            Err(format!("Migration {} not found", plan_id))
        }
    }

    /// Get migration progress
    pub async fn get_migration_progress(&self, plan_id: &str) -> Option<MigrationProgress> {
        self.active_migrations.read().await.get(plan_id).cloned()
    }

    /// Get all active migrations
    pub async fn get_active_migrations(&self) -> Vec<MigrationProgress> {
        self.active_migrations
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    /// Get migration history
    pub async fn get_migration_history(&self) -> Vec<MigrationProgress> {
        self.migration_history.read().await.clone()
    }

    /// Get node loads
    pub async fn get_node_loads(&self) -> BTreeMap<OxirsNodeId, NodeLoad> {
        self.node_loads.read().await.clone()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> RebalancingStats {
        self.stats.read().await.clone()
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.node_loads.write().await.clear();
        self.active_migrations.write().await.clear();
        self.migration_history.write().await.clear();
        *self.stats.write().await = RebalancingStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rebalancing_creation() {
        let config = RebalancingConfig::default();
        let manager = DataRebalancingManager::new(config);

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_rebalancing_ops, 0);
    }

    #[tokio::test]
    async fn test_register_node() {
        let config = RebalancingConfig::default();
        let manager = DataRebalancingManager::new(config);

        manager.register_node(1).await;
        manager.register_node(2).await;

        let loads = manager.get_node_loads().await;
        assert_eq!(loads.len(), 2);
    }

    #[tokio::test]
    async fn test_update_node_load() {
        let config = RebalancingConfig::default();
        let manager = DataRebalancingManager::new(config);

        manager.register_node(1).await;
        manager
            .update_node_load(1, 1000, 50000, 0.5, 0.6, 1000000)
            .await;

        let loads = manager.get_node_loads().await;
        let load = loads.get(&1).unwrap();
        assert_eq!(load.key_count, 1000);
        assert_eq!(load.data_size_bytes, 50000);
        assert_eq!(load.cpu_utilization, 0.5);
    }

    #[tokio::test]
    async fn test_check_rebalancing_needed() {
        let config = RebalancingConfig {
            enable_auto_rebalancing: true,
            load_imbalance_threshold: 0.2,
            min_nodes_for_rebalancing: 2,
            ..Default::default()
        };
        let manager = DataRebalancingManager::new(config);

        manager.register_node(1).await;
        manager.register_node(2).await;

        // Balanced load
        manager.update_node_load(1, 100, 10000, 0.5, 0.5, 0).await;
        manager.update_node_load(2, 100, 10000, 0.5, 0.5, 0).await;

        let needed = manager.check_rebalancing_needed().await;
        assert!(!needed);

        // Imbalanced load
        manager.update_node_load(1, 100, 10000, 0.5, 0.5, 0).await;
        manager.update_node_load(2, 100, 50000, 0.5, 0.5, 0).await;

        let needed = manager.check_rebalancing_needed().await;
        assert!(needed);
    }

    #[tokio::test]
    async fn test_create_migration_plan() {
        let config = RebalancingConfig::default();
        let manager = DataRebalancingManager::new(config);

        manager.register_node(1).await;
        manager.register_node(2).await;

        manager.update_node_load(1, 100, 10000, 0.5, 0.5, 0).await;
        manager.update_node_load(2, 200, 50000, 0.5, 0.5, 0).await;

        let plan = manager.create_migration_plan().await;
        assert!(plan.is_ok());

        let plan = plan.unwrap();
        assert_eq!(plan.source_node, 2); // Most loaded
        assert_eq!(plan.target_node, 1); // Least loaded
        assert!(plan.key_count > 0);
    }

    #[tokio::test]
    async fn test_execute_migration() {
        let config = RebalancingConfig {
            incremental_batch_size: 10,
            bandwidth_limit_bytes_per_sec: 0, // No limit for test
            enable_verification: false,
            ..Default::default()
        };
        let manager = DataRebalancingManager::new(config);

        manager.register_node(1).await;
        manager.register_node(2).await;

        manager.update_node_load(1, 100, 10000, 0.5, 0.5, 0).await;
        manager.update_node_load(2, 200, 50000, 0.5, 0.5, 0).await;

        let plan = manager.create_migration_plan().await.unwrap();
        let result = manager.execute_migration(plan).await;

        assert!(result.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.successful_migrations, 1);
    }

    #[tokio::test]
    async fn test_migration_progress() {
        let progress = MigrationProgress::new("test-1".to_string(), 100, 10000);

        assert_eq!(progress.progress_percentage(), 0.0);

        let mut progress = progress;
        progress.keys_migrated = 50;

        assert_eq!(progress.progress_percentage(), 50.0);
    }

    #[tokio::test]
    async fn test_migration_states() {
        let config = RebalancingConfig {
            bandwidth_limit_bytes_per_sec: 10000, // Slow enough to observe progress
            enable_verification: true,
            incremental_batch_size: 10,
            ..Default::default()
        };
        let manager = DataRebalancingManager::new(config);

        manager.register_node(1).await;
        manager.register_node(2).await;

        manager.update_node_load(1, 100, 10000, 0.5, 0.5, 0).await;
        manager.update_node_load(2, 200, 50000, 0.5, 0.5, 0).await;

        let plan = manager.create_migration_plan().await.unwrap();
        let plan_id = plan.plan_id.clone();

        // Execute in background
        let manager_clone = Arc::new(manager);
        let manager_ref = manager_clone.clone();
        tokio::spawn(async move {
            let _ = manager_ref.execute_migration(plan).await;
        });

        // Wait a bit for migration to start
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Check progress - should be in active migrations
        let progress = manager_clone.get_migration_progress(&plan_id).await;
        assert!(progress.is_some());

        // Wait for completion
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    #[tokio::test]
    async fn test_rollback() {
        let config = RebalancingConfig {
            enable_rollback: true,
            ..Default::default()
        };
        let manager = DataRebalancingManager::new(config);

        let progress = MigrationProgress::new("test-rollback".to_string(), 100, 10000);
        manager
            .active_migrations
            .write()
            .await
            .insert("test-rollback".to_string(), progress);

        let result = manager.rollback_migration("test-rollback").await;
        assert!(result.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.failed_migrations, 1);
    }

    #[tokio::test]
    async fn test_clear() {
        let config = RebalancingConfig::default();
        let manager = DataRebalancingManager::new(config);

        manager.register_node(1).await;
        manager.register_node(2).await;

        manager.clear().await;

        let loads = manager.get_node_loads().await;
        assert!(loads.is_empty());
    }

    #[test]
    fn test_rebalancing_strategy() {
        assert_eq!(
            RebalancingStrategy::Incremental,
            RebalancingStrategy::Incremental
        );
        assert_ne!(RebalancingStrategy::Incremental, RebalancingStrategy::Bulk);
    }

    #[test]
    fn test_migration_state() {
        assert_eq!(MigrationState::Idle, MigrationState::Idle);
        assert_ne!(MigrationState::Idle, MigrationState::InProgress);
    }
}
