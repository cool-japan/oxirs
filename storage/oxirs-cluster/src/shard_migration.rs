//! Shard Migration Module
//!
//! Provides live migration capabilities for shards with zero-downtime moves,
//! consistency preservation, and progress tracking.

use crate::network::{NetworkService, RpcMessage};
use crate::raft::OxirsNodeId;
use crate::shard::ShardId;
use crate::storage::StorageBackend;
use anyhow::Result;
use oxirs_core::model::Triple;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio::time::Instant;
use tracing::{debug, error, info, warn};

/// Migration strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// Copy all data then switch (safer but longer downtime)
    CopyThenSwitch,
    /// Live migration with incremental sync (zero downtime)
    LiveMigration,
    /// Hot migration with dual writes (minimal downtime)
    HotMigration,
}

/// Migration phase tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MigrationPhase {
    /// Migration planned but not started
    Planned,
    /// Initial data copy phase
    InitialCopy,
    /// Incremental sync phase
    IncrementalSync,
    /// Final cutover phase
    Cutover,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed,
    /// Migration was cancelled
    Cancelled,
}

/// Migration operation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationOperation {
    /// Unique migration ID
    pub migration_id: String,
    /// Shard being migrated
    pub shard_id: ShardId,
    /// Source nodes
    pub source_nodes: HashSet<OxirsNodeId>,
    /// Target nodes
    pub target_nodes: HashSet<OxirsNodeId>,
    /// Migration strategy
    pub strategy: MigrationStrategy,
    /// Current phase
    pub phase: MigrationPhase,
    /// Progress percentage (0-100)
    pub progress: f64,
    /// Estimated data size to migrate
    pub estimated_size: u64,
    /// Data migrated so far
    pub migrated_size: u64,
    /// Migration statistics
    pub stats: MigrationStats,
    /// Created timestamp
    pub created_at: u64,
    /// Started timestamp
    pub started_at: Option<u64>,
    /// Completed timestamp
    pub completed_at: Option<u64>,
    /// Error message if failed
    pub error_message: Option<String>,
}

impl MigrationOperation {
    pub fn new(
        shard_id: ShardId,
        source_nodes: HashSet<OxirsNodeId>,
        target_nodes: HashSet<OxirsNodeId>,
        strategy: MigrationStrategy,
        estimated_size: u64,
    ) -> Self {
        Self {
            migration_id: uuid::Uuid::new_v4().to_string(),
            shard_id,
            source_nodes,
            target_nodes,
            strategy,
            phase: MigrationPhase::Planned,
            progress: 0.0,
            estimated_size,
            migrated_size: 0,
            stats: MigrationStats::default(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            started_at: None,
            completed_at: None,
            error_message: None,
        }
    }

    pub fn update_progress(&mut self, migrated: u64) {
        self.migrated_size = migrated;
        if self.estimated_size > 0 {
            self.progress = (migrated as f64 / self.estimated_size as f64 * 100.0).min(100.0);
        }
    }

    pub fn advance_phase(&mut self, new_phase: MigrationPhase) {
        if matches!(new_phase, MigrationPhase::InitialCopy) && self.started_at.is_none() {
            self.started_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("SystemTime should be after UNIX_EPOCH")
                    .as_secs(),
            );
        }
        self.phase = new_phase.clone();
        if matches!(
            new_phase,
            MigrationPhase::Completed | MigrationPhase::Failed | MigrationPhase::Cancelled
        ) {
            self.completed_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("SystemTime should be after UNIX_EPOCH")
                    .as_secs(),
            );
        }
    }

    pub fn set_error(&mut self, error: String) {
        self.phase = MigrationPhase::Failed;
        self.error_message = Some(error);
        self.advance_phase(MigrationPhase::Failed);
    }
}

/// Migration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStats {
    /// Number of triples migrated
    pub triples_migrated: u64,
    /// Number of triples to migrate
    pub total_triples: u64,
    /// Migration throughput (triples/second)
    pub throughput: f64,
    /// Network bandwidth used (bytes/second)
    pub network_bandwidth: f64,
    /// Average latency per operation (milliseconds)
    pub avg_latency_ms: f64,
    /// Number of retries
    pub retries: u32,
    /// Last update timestamp
    pub last_updated: u64,
}

impl Default for MigrationStats {
    fn default() -> Self {
        Self {
            triples_migrated: 0,
            total_triples: 0,
            throughput: 0.0,
            network_bandwidth: 0.0,
            avg_latency_ms: 0.0,
            retries: 0,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        }
    }
}

/// Migration batch for transferring data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationBatch {
    /// Batch ID
    pub batch_id: String,
    /// Migration ID this batch belongs to
    pub migration_id: String,
    /// Batch sequence number
    pub sequence: u64,
    /// Triples in this batch
    pub triples: Vec<Triple>,
    /// Batch checksum for integrity verification
    pub checksum: u32,
    /// Batch creation timestamp
    pub created_at: u64,
}

impl MigrationBatch {
    pub fn new(migration_id: String, sequence: u64, triples: Vec<Triple>) -> Self {
        let serialized = oxicode::serde::encode_to_vec(&triples, oxicode::config::standard())
            .unwrap_or_default();
        let checksum = crc32fast::hash(&serialized);

        Self {
            batch_id: uuid::Uuid::new_v4().to_string(),
            migration_id,
            sequence,
            triples,
            checksum,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
        }
    }

    pub fn verify_integrity(&self) -> bool {
        let serialized = oxicode::serde::encode_to_vec(&self.triples, oxicode::config::standard())
            .unwrap_or_default();
        let computed_checksum = crc32fast::hash(&serialized);
        computed_checksum == self.checksum
    }
}

/// Configuration for migration operations
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Batch size for data transfer
    pub batch_size: usize,
    /// Maximum concurrent batches
    pub max_concurrent_batches: usize,
    /// Retry attempts for failed operations
    pub max_retries: u32,
    /// Timeout for individual operations
    pub operation_timeout: Duration,
    /// Progress update interval
    pub progress_interval: Duration,
    /// Enable consistency verification
    pub verify_consistency: bool,
    /// Enable rollback on failure
    pub enable_rollback: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 10000,
            max_concurrent_batches: 5,
            max_retries: 3,
            operation_timeout: Duration::from_secs(30),
            progress_interval: Duration::from_secs(5),
            verify_consistency: true,
            enable_rollback: true,
        }
    }
}

/// Shard migration manager
pub struct ShardMigrationManager {
    /// Node ID
    node_id: OxirsNodeId,
    /// Storage backend
    storage: Arc<dyn StorageBackend>,
    /// Network service
    network: Arc<NetworkService>,
    /// Active migrations
    active_migrations: Arc<RwLock<HashMap<String, MigrationOperation>>>,
    /// Migration configuration
    config: MigrationConfig,
    /// Semaphore for controlling concurrent migrations
    migration_semaphore: Arc<Semaphore>,
    /// Shutdown signal
    #[allow(dead_code)]
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl ShardMigrationManager {
    /// Create a new shard migration manager
    pub fn new(
        node_id: OxirsNodeId,
        storage: Arc<dyn StorageBackend>,
        network: Arc<NetworkService>,
        config: MigrationConfig,
    ) -> Self {
        let max_concurrent = config.max_concurrent_batches;

        Self {
            node_id,
            storage,
            network,
            active_migrations: Arc::new(RwLock::new(HashMap::new())),
            config,
            migration_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            shutdown_tx: None,
        }
    }

    /// Start a shard migration
    pub async fn start_migration(
        &mut self,
        shard_id: ShardId,
        source_nodes: HashSet<OxirsNodeId>,
        target_nodes: HashSet<OxirsNodeId>,
        strategy: MigrationStrategy,
    ) -> Result<String> {
        // Estimate migration size
        let estimated_size = self.estimate_migration_size(shard_id).await?;

        let migration = MigrationOperation::new(
            shard_id,
            source_nodes,
            target_nodes,
            strategy,
            estimated_size,
        );

        let migration_id = migration.migration_id.clone();

        // Validate migration
        self.validate_migration(&migration).await?;

        // Store migration
        {
            let mut migrations = self.active_migrations.write().await;
            migrations.insert(migration_id.clone(), migration);
        }

        // Start migration execution
        let manager = self.clone();
        let migration_id_clone = migration_id.clone();

        tokio::spawn(async move {
            if let Err(e) = manager.execute_migration(&migration_id_clone).await {
                error!("Migration {} failed: {}", migration_id_clone, e);

                let mut migrations = manager.active_migrations.write().await;
                if let Some(migration) = migrations.get_mut(&migration_id_clone) {
                    migration.set_error(e.to_string());
                }
            }
        });

        info!("Started migration {} for shard {}", migration_id, shard_id);
        Ok(migration_id)
    }

    /// Execute a migration operation
    async fn execute_migration(&self, migration_id: &str) -> Result<()> {
        info!("Executing migration {}", migration_id);

        // Get migration details
        let (shard_id, strategy) = {
            let migrations = self.active_migrations.read().await;
            let migration = migrations
                .get(migration_id)
                .ok_or_else(|| anyhow::anyhow!("Migration {} not found", migration_id))?;
            (migration.shard_id, migration.strategy.clone())
        };

        match strategy {
            MigrationStrategy::CopyThenSwitch => {
                self.execute_copy_then_switch_migration(migration_id, shard_id)
                    .await
            }
            MigrationStrategy::LiveMigration => {
                self.execute_live_migration(migration_id, shard_id).await
            }
            MigrationStrategy::HotMigration => {
                self.execute_hot_migration(migration_id, shard_id).await
            }
        }
    }

    /// Execute copy-then-switch migration
    async fn execute_copy_then_switch_migration(
        &self,
        migration_id: &str,
        shard_id: ShardId,
    ) -> Result<()> {
        // Phase 1: Initial Copy
        self.update_migration_phase(migration_id, MigrationPhase::InitialCopy)
            .await?;
        self.copy_shard_data(migration_id, shard_id).await?;

        // Phase 2: Cutover
        self.update_migration_phase(migration_id, MigrationPhase::Cutover)
            .await?;
        self.switch_shard_ownership(migration_id, shard_id).await?;

        // Complete migration
        self.update_migration_phase(migration_id, MigrationPhase::Completed)
            .await?;
        Ok(())
    }

    /// Execute live migration
    async fn execute_live_migration(&self, migration_id: &str, shard_id: ShardId) -> Result<()> {
        // Phase 1: Initial Copy
        self.update_migration_phase(migration_id, MigrationPhase::InitialCopy)
            .await?;
        self.copy_shard_data(migration_id, shard_id).await?;

        // Phase 2: Incremental Sync
        self.update_migration_phase(migration_id, MigrationPhase::IncrementalSync)
            .await?;
        self.sync_incremental_changes(migration_id, shard_id)
            .await?;

        // Phase 3: Cutover
        self.update_migration_phase(migration_id, MigrationPhase::Cutover)
            .await?;
        self.switch_shard_ownership(migration_id, shard_id).await?;

        // Complete migration
        self.update_migration_phase(migration_id, MigrationPhase::Completed)
            .await?;
        Ok(())
    }

    /// Execute hot migration
    async fn execute_hot_migration(&self, migration_id: &str, shard_id: ShardId) -> Result<()> {
        // Phase 1: Start dual writes
        self.update_migration_phase(migration_id, MigrationPhase::InitialCopy)
            .await?;
        self.start_dual_writes(migration_id, shard_id).await?;

        // Phase 2: Copy existing data
        self.copy_shard_data(migration_id, shard_id).await?;

        // Phase 3: Switch reads and stop dual writes
        self.update_migration_phase(migration_id, MigrationPhase::Cutover)
            .await?;
        self.switch_shard_ownership(migration_id, shard_id).await?;
        self.stop_dual_writes(migration_id, shard_id).await?;

        // Complete migration
        self.update_migration_phase(migration_id, MigrationPhase::Completed)
            .await?;
        Ok(())
    }

    /// Copy shard data to target nodes
    async fn copy_shard_data(&self, migration_id: &str, shard_id: ShardId) -> Result<()> {
        info!(
            "Copying shard {} data for migration {}",
            shard_id, migration_id
        );

        // Export shard data
        let triples = self.storage.export_shard(shard_id).await?;
        let total_triples = triples.len() as u64;

        // Update total count
        {
            let mut migrations = self.active_migrations.write().await;
            if let Some(migration) = migrations.get_mut(migration_id) {
                migration.stats.total_triples = total_triples;
            }
        }

        // Process in batches
        let batch_size = self.config.batch_size;
        let mut processed = 0u64;

        for (sequence, chunk) in triples.chunks(batch_size).enumerate() {
            let batch =
                MigrationBatch::new(migration_id.to_string(), sequence as u64, chunk.to_vec());

            // Transfer batch to target nodes
            self.transfer_batch(migration_id, &batch).await?;

            processed += chunk.len() as u64;

            // Update progress
            self.update_migration_progress(migration_id, processed)
                .await?;

            // Small delay to avoid overwhelming the system
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        info!(
            "Completed copying {} triples for migration {}",
            processed, migration_id
        );
        Ok(())
    }

    /// Transfer a batch to target nodes
    async fn transfer_batch(&self, migration_id: &str, batch: &MigrationBatch) -> Result<()> {
        let target_nodes = {
            let migrations = self.active_migrations.read().await;
            let migration = migrations
                .get(migration_id)
                .ok_or_else(|| anyhow::anyhow!("Migration {} not found", migration_id))?;
            migration.target_nodes.clone()
        };

        // Acquire semaphore permit to limit concurrent transfers
        let _permit = self.migration_semaphore.acquire().await?;

        // Send batch to all target nodes
        for &target_node in &target_nodes {
            let message = RpcMessage::MigrationBatch {
                migration_id: migration_id.to_string(),
                batch: batch.clone(),
            };

            // Retry logic
            let mut attempts = 0;
            while attempts < self.config.max_retries {
                match self
                    .network
                    .send_message(target_node, message.clone())
                    .await
                {
                    Ok(_) => break,
                    Err(e) => {
                        attempts += 1;
                        warn!(
                            "Failed to send batch {} to node {} (attempt {}): {}",
                            batch.batch_id, target_node, attempts, e
                        );

                        if attempts >= self.config.max_retries {
                            return Err(anyhow::anyhow!(
                                "Failed to send batch after {} attempts: {}",
                                self.config.max_retries,
                                e
                            ));
                        }

                        tokio::time::sleep(Duration::from_secs(1 << (attempts - 1))).await;
                    }
                }
            }
        }

        debug!(
            "Successfully transferred batch {} for migration {}",
            batch.batch_id, migration_id
        );
        Ok(())
    }

    /// Sync incremental changes during live migration
    async fn sync_incremental_changes(&self, migration_id: &str, shard_id: ShardId) -> Result<()> {
        info!("Syncing incremental changes for migration {}", migration_id);

        // In a real implementation, this would:
        // 1. Track changes that occurred during initial copy
        // 2. Apply those changes to target nodes
        // 3. Continue tracking until cutover

        // For now, we'll simulate a brief sync period
        let sync_duration = Duration::from_secs(5);
        let start_time = Instant::now();

        while start_time.elapsed() < sync_duration {
            // Simulate checking for and applying incremental changes
            tokio::time::sleep(Duration::from_millis(100)).await;

            // In practice, this would query the WAL or change log
            debug!("Checking for incremental changes for shard {}", shard_id);
        }

        info!("Completed incremental sync for migration {}", migration_id);
        Ok(())
    }

    /// Switch shard ownership to target nodes
    async fn switch_shard_ownership(&self, migration_id: &str, shard_id: ShardId) -> Result<()> {
        info!(
            "Switching ownership for shard {} in migration {}",
            shard_id, migration_id
        );

        // In a real implementation, this would:
        // 1. Update cluster metadata to point to new nodes
        // 2. Redirect traffic to target nodes
        // 3. Notify all nodes of the ownership change
        // 4. Wait for acknowledgments

        // Simulate cutover delay
        tokio::time::sleep(Duration::from_millis(500)).await;

        info!("Completed ownership switch for migration {}", migration_id);
        Ok(())
    }

    /// Start dual writes for hot migration
    async fn start_dual_writes(&self, migration_id: &str, shard_id: ShardId) -> Result<()> {
        info!(
            "Starting dual writes for shard {} in migration {}",
            shard_id, migration_id
        );
        // Implementation would set up dual write mechanism
        Ok(())
    }

    /// Stop dual writes after hot migration
    async fn stop_dual_writes(&self, migration_id: &str, shard_id: ShardId) -> Result<()> {
        info!(
            "Stopping dual writes for shard {} in migration {}",
            shard_id, migration_id
        );
        // Implementation would tear down dual write mechanism
        Ok(())
    }

    /// Update migration phase
    async fn update_migration_phase(
        &self,
        migration_id: &str,
        phase: MigrationPhase,
    ) -> Result<()> {
        let mut migrations = self.active_migrations.write().await;
        if let Some(migration) = migrations.get_mut(migration_id) {
            migration.advance_phase(phase);
            info!(
                "Migration {} advanced to phase {:?}",
                migration_id, migration.phase
            );
        }
        Ok(())
    }

    /// Update migration progress
    async fn update_migration_progress(&self, migration_id: &str, migrated: u64) -> Result<()> {
        let mut migrations = self.active_migrations.write().await;
        if let Some(migration) = migrations.get_mut(migration_id) {
            let old_migrated = migration.stats.triples_migrated;
            migration.stats.triples_migrated = migrated;
            migration.update_progress(migrated);

            // Calculate throughput
            let elapsed = migration
                .started_at
                .map(|start| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("SystemTime should be after UNIX_EPOCH")
                        .as_secs()
                        .saturating_sub(start)
                })
                .unwrap_or(1);

            if elapsed > 0 {
                migration.stats.throughput = migrated as f64 / elapsed as f64;
            }

            if migrated > old_migrated {
                debug!(
                    "Migration {} progress: {:.1}% ({} / {} triples)",
                    migration_id, migration.progress, migrated, migration.stats.total_triples
                );
            }
        }
        Ok(())
    }

    /// Estimate migration size
    async fn estimate_migration_size(&self, shard_id: ShardId) -> Result<u64> {
        let triple_count = self.storage.get_shard_triple_count(shard_id).await? as u64;
        let size_bytes = self.storage.get_shard_size(shard_id).await?;

        // Return the larger of triple count or byte size as the estimation metric
        Ok(triple_count.max(size_bytes))
    }

    /// Validate migration parameters
    async fn validate_migration(&self, migration: &MigrationOperation) -> Result<()> {
        if migration.source_nodes.is_empty() {
            return Err(anyhow::anyhow!("No source nodes specified"));
        }

        if migration.target_nodes.is_empty() {
            return Err(anyhow::anyhow!("No target nodes specified"));
        }

        if migration
            .source_nodes
            .intersection(&migration.target_nodes)
            .count()
            > 0
        {
            return Err(anyhow::anyhow!("Source and target nodes must be disjoint"));
        }

        Ok(())
    }

    /// Get migration status
    pub async fn get_migration_status(&self, migration_id: &str) -> Option<MigrationOperation> {
        let migrations = self.active_migrations.read().await;
        migrations.get(migration_id).cloned()
    }

    /// List all active migrations
    pub async fn list_active_migrations(&self) -> Vec<MigrationOperation> {
        let migrations = self.active_migrations.read().await;
        migrations.values().cloned().collect()
    }

    /// Cancel a migration
    pub async fn cancel_migration(&self, migration_id: &str) -> Result<()> {
        let mut migrations = self.active_migrations.write().await;
        if let Some(migration) = migrations.get_mut(migration_id) {
            if matches!(
                migration.phase,
                MigrationPhase::Completed | MigrationPhase::Failed
            ) {
                return Err(anyhow::anyhow!(
                    "Cannot cancel completed or failed migration"
                ));
            }

            migration.advance_phase(MigrationPhase::Cancelled);
            info!("Cancelled migration {}", migration_id);
        }
        Ok(())
    }

    /// Clean up completed migrations
    pub async fn cleanup_completed_migrations(&self, retention_hours: u64) -> Result<usize> {
        let cutoff_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs()
            .saturating_sub(retention_hours * 3600);

        let mut migrations = self.active_migrations.write().await;
        let initial_count = migrations.len();

        migrations.retain(|_id, migration| {
            !matches!(
                migration.phase,
                MigrationPhase::Completed | MigrationPhase::Failed | MigrationPhase::Cancelled
            ) || migration.completed_at.unwrap_or(u64::MAX) > cutoff_time
        });

        let cleaned_count = initial_count - migrations.len();
        if cleaned_count > 0 {
            info!("Cleaned up {} completed migrations", cleaned_count);
        }

        Ok(cleaned_count)
    }
}

impl Clone for ShardMigrationManager {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id,
            storage: Arc::clone(&self.storage),
            network: Arc::clone(&self.network),
            active_migrations: Arc::clone(&self.active_migrations),
            config: self.config.clone(),
            migration_semaphore: Arc::clone(&self.migration_semaphore),
            shutdown_tx: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_migration() -> MigrationOperation {
        let source_nodes = [1, 2].iter().cloned().collect();
        let target_nodes = [3, 4].iter().cloned().collect();

        MigrationOperation::new(
            1,
            source_nodes,
            target_nodes,
            MigrationStrategy::LiveMigration,
            1000,
        )
    }

    #[tokio::test]
    async fn test_migration_operation_creation() {
        let migration = create_test_migration();

        assert_eq!(migration.shard_id, 1);
        assert_eq!(migration.phase, MigrationPhase::Planned);
        assert_eq!(migration.progress, 0.0);
        assert!(migration.started_at.is_none());
    }

    #[tokio::test]
    async fn test_migration_progress_update() {
        let mut migration = create_test_migration();

        migration.update_progress(500);
        assert_eq!(migration.migrated_size, 500);
        assert_eq!(migration.progress, 50.0);
    }

    #[tokio::test]
    async fn test_migration_phase_advancement() {
        let mut migration = create_test_migration();

        migration.advance_phase(MigrationPhase::InitialCopy);
        assert_eq!(migration.phase, MigrationPhase::InitialCopy);
        assert!(migration.started_at.is_some());

        migration.advance_phase(MigrationPhase::Completed);
        assert_eq!(migration.phase, MigrationPhase::Completed);
        assert!(migration.completed_at.is_some());
    }

    #[test]
    fn test_migration_batch_integrity() {
        use oxirs_core::model::{NamedNode, Triple as CoreTriple};

        let triples = vec![CoreTriple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            NamedNode::new("http://example.org/o").unwrap(),
        )];

        let batch = MigrationBatch::new("test-migration".to_string(), 1, triples);
        assert!(batch.verify_integrity());
    }

    #[test]
    fn test_migration_config_defaults() {
        let config = MigrationConfig::default();
        assert_eq!(config.batch_size, 10000);
        assert_eq!(config.max_concurrent_batches, 5);
        assert_eq!(config.max_retries, 3);
        assert!(config.verify_consistency);
        assert!(config.enable_rollback);
    }
}
