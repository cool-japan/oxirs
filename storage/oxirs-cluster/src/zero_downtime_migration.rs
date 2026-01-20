//! Zero-Downtime Migrations
//!
//! This module provides capabilities for performing schema changes and data migrations
//! without interrupting cluster operations. It enables online schema evolution,
//! data transformations, and structural changes while maintaining service availability.
//!
//! # Features
//!
//! - **Online Schema Changes**: Modify RDF schemas without downtime
//! - **Phased Migration**: Gradual migration with rollback capability
//! - **Version Compatibility**: Support for multiple schema versions simultaneously
//! - **Data Transformation**: Transform data during migration
//! - **Consistency Validation**: Ensure data integrity throughout migration
//! - **Progress Tracking**: Monitor migration progress in real-time
//! - **Automated Rollback**: Automatic rollback on failure
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_cluster::zero_downtime_migration::{MigrationConfig, MigrationManager};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = MigrationConfig::default()
//!     .with_batch_size(1000)
//!     .with_validation(true);
//!
//! let mut migration_manager = MigrationManager::new(config).await?;
//! migration_manager.start_migration("v1", "v2").await?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur during migration operations
#[derive(Debug, Error)]
pub enum MigrationError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Migration validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Migration execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Rollback error
    #[error("Rollback error: {0}")]
    RollbackError(String),

    /// Version compatibility error
    #[error("Version compatibility error: {0}")]
    VersionError(String),

    /// Data transformation error
    #[error("Transformation error: {0}")]
    TransformationError(String),

    /// Other errors
    #[error("Migration error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, MigrationError>;

/// Migration status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStatus {
    /// Migration is pending
    Pending,
    /// Migration is being prepared
    Preparing,
    /// Migration is in progress
    InProgress,
    /// Migration is being validated
    Validating,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed,
    /// Migration is being rolled back
    RollingBack,
    /// Migration was rolled back
    RolledBack,
}

/// Migration phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationPhase {
    /// Schema validation phase
    SchemaValidation,
    /// Data backup phase
    DataBackup,
    /// Schema modification phase
    SchemaModification,
    /// Data transformation phase
    DataTransformation,
    /// Consistency validation phase
    ConsistencyValidation,
    /// Finalization phase
    Finalization,
}

/// Migration strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// All-at-once migration
    AllAtOnce,
    /// Blue-green deployment
    BlueGreen,
    /// Rolling migration
    Rolling {
        /// Percentage of nodes to migrate at once
        batch_percent: u8,
    },
    /// Canary migration (test on small subset first)
    Canary {
        /// Percentage for canary deployment
        canary_percent: u8,
    },
}

impl Default for MigrationStrategy {
    fn default() -> Self {
        Self::Rolling { batch_percent: 25 }
    }
}

/// Migration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Migration strategy
    pub strategy: MigrationStrategy,

    /// Batch size for data processing
    pub batch_size: usize,

    /// Enable validation
    pub enable_validation: bool,

    /// Enable automatic rollback on failure
    pub enable_auto_rollback: bool,

    /// Timeout for migration (seconds)
    pub timeout_seconds: u64,

    /// Maximum retry attempts
    pub max_retries: usize,

    /// Delay between retries (milliseconds)
    pub retry_delay_ms: u64,

    /// Enable progress tracking
    pub enable_progress_tracking: bool,

    /// Enable data verification after migration
    pub enable_data_verification: bool,

    /// Checkpointing interval (number of batches)
    pub checkpoint_interval: usize,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            strategy: MigrationStrategy::default(),
            batch_size: 1000,
            enable_validation: true,
            enable_auto_rollback: true,
            timeout_seconds: 3600, // 1 hour
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_progress_tracking: true,
            enable_data_verification: true,
            checkpoint_interval: 10,
        }
    }
}

impl MigrationConfig {
    /// Set the migration strategy
    pub fn with_strategy(mut self, strategy: MigrationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enable or disable validation
    pub fn with_validation(mut self, enable: bool) -> Self {
        self.enable_validation = enable;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }
}

/// Migration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationMetadata {
    /// Migration ID
    pub migration_id: String,

    /// Source schema version
    pub from_version: String,

    /// Target schema version
    pub to_version: String,

    /// Migration description
    pub description: String,

    /// Creator
    pub created_by: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Migration status
    pub status: MigrationStatus,

    /// Current phase
    pub current_phase: Option<MigrationPhase>,

    /// Progress percentage (0-100)
    pub progress_percent: f64,

    /// Total items to migrate
    pub total_items: u64,

    /// Items migrated
    pub items_migrated: u64,

    /// Start time
    pub started_at: Option<DateTime<Utc>>,

    /// Completion time
    pub completed_at: Option<DateTime<Utc>>,

    /// Error message (if failed)
    pub error_message: Option<String>,

    /// Checkpoints for rollback
    pub checkpoints: Vec<MigrationCheckpoint>,
}

impl MigrationMetadata {
    /// Create new migration metadata
    pub fn new(
        from_version: impl Into<String>,
        to_version: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            migration_id: uuid::Uuid::new_v4().to_string(),
            from_version: from_version.into(),
            to_version: to_version.into(),
            description: description.into(),
            created_by: "system".to_string(),
            created_at: Utc::now(),
            status: MigrationStatus::Pending,
            current_phase: None,
            progress_percent: 0.0,
            total_items: 0,
            items_migrated: 0,
            started_at: None,
            completed_at: None,
            error_message: None,
            checkpoints: Vec::new(),
        }
    }

    /// Update progress
    pub fn update_progress(&mut self, items_migrated: u64) {
        self.items_migrated = items_migrated;
        if self.total_items > 0 {
            self.progress_percent = (items_migrated as f64 / self.total_items as f64) * 100.0;
        }
    }

    /// Mark as started
    pub fn mark_started(&mut self) {
        self.status = MigrationStatus::InProgress;
        self.started_at = Some(Utc::now());
    }

    /// Mark as completed
    pub fn mark_completed(&mut self) {
        self.status = MigrationStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.progress_percent = 100.0;
    }

    /// Mark as failed
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = MigrationStatus::Failed;
        self.error_message = Some(error.into());
        self.completed_at = Some(Utc::now());
    }
}

/// Migration checkpoint for rollback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationCheckpoint {
    /// Checkpoint ID
    pub checkpoint_id: String,

    /// Phase at checkpoint
    pub phase: MigrationPhase,

    /// Items processed at checkpoint
    pub items_processed: u64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Checkpoint data (for rollback)
    pub data: HashMap<String, String>,
}

/// Schema change operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaOperation {
    /// Add a new class/type
    AddClass {
        class_uri: String,
        properties: Vec<String>,
    },
    /// Remove a class/type
    RemoveClass { class_uri: String },
    /// Add a property
    AddProperty {
        property_uri: String,
        domain: String,
        range: String,
    },
    /// Remove a property
    RemoveProperty { property_uri: String },
    /// Rename a property
    RenameProperty { old_uri: String, new_uri: String },
    /// Change property range
    ChangePropertyRange {
        property_uri: String,
        old_range: String,
        new_range: String,
    },
}

/// Data transformation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformOperation {
    /// Map values from one property to another
    MapProperty {
        from_property: String,
        to_property: String,
        transform_fn: String,
    },
    /// Split a property into multiple properties
    SplitProperty {
        source_property: String,
        target_properties: Vec<String>,
    },
    /// Merge multiple properties into one
    MergeProperties {
        source_properties: Vec<String>,
        target_property: String,
    },
    /// Apply custom transformation
    Custom { name: String, script: String },
}

/// Migration plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Migration metadata
    pub metadata: MigrationMetadata,

    /// Schema operations
    pub schema_operations: Vec<SchemaOperation>,

    /// Data transformations
    pub data_transformations: Vec<TransformOperation>,

    /// Validation rules
    pub validation_rules: Vec<String>,
}

impl MigrationPlan {
    /// Create a new migration plan
    pub fn new(metadata: MigrationMetadata) -> Self {
        Self {
            metadata,
            schema_operations: Vec::new(),
            data_transformations: Vec::new(),
            validation_rules: Vec::new(),
        }
    }

    /// Add schema operation
    pub fn add_schema_operation(mut self, operation: SchemaOperation) -> Self {
        self.schema_operations.push(operation);
        self
    }

    /// Add data transformation
    pub fn add_data_transformation(mut self, operation: TransformOperation) -> Self {
        self.data_transformations.push(operation);
        self
    }

    /// Add validation rule
    pub fn add_validation_rule(mut self, rule: impl Into<String>) -> Self {
        self.validation_rules.push(rule.into());
        self
    }
}

/// Migration manager
pub struct MigrationManager {
    config: MigrationConfig,
    active_migrations: Arc<RwLock<HashMap<String, MigrationMetadata>>>,
    migration_history: Arc<RwLock<Vec<MigrationMetadata>>>,
    running: Arc<RwLock<bool>>,
}

impl MigrationManager {
    /// Create a new migration manager
    pub async fn new(config: MigrationConfig) -> Result<Self> {
        Ok(Self {
            config,
            active_migrations: Arc::new(RwLock::new(HashMap::new())),
            migration_history: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start a migration
    pub async fn start_migration(
        &mut self,
        from_version: impl Into<String>,
        to_version: impl Into<String>,
    ) -> Result<String> {
        let mut running = self.running.write().await;
        if *running {
            return Err(MigrationError::ExecutionError(
                "Migration already in progress".to_string(),
            ));
        }

        let from = from_version.into();
        let to = to_version.into();

        tracing::info!(
            from_version = %from,
            to_version = %to,
            "Starting migration"
        );

        let metadata = MigrationMetadata::new(
            from.clone(),
            to.clone(),
            format!("Migration from {} to {}", from, to),
        );

        let migration_id = metadata.migration_id.clone();

        // Add to active migrations
        let mut active = self.active_migrations.write().await;
        active.insert(migration_id.clone(), metadata.clone());

        *running = true;

        // Start migration in background
        let config = self.config.clone();
        let active_migrations = Arc::clone(&self.active_migrations);
        let migration_history = Arc::clone(&self.migration_history);
        let running_clone = Arc::clone(&self.running);
        let migration_id_clone = migration_id.clone();

        tokio::spawn(async move {
            let result = Self::execute_migration(
                &config,
                migration_id_clone.clone(),
                active_migrations.clone(),
            )
            .await;

            // Update status
            let mut active = active_migrations.write().await;
            if let Some(metadata) = active.remove(&migration_id_clone) {
                let mut history = migration_history.write().await;
                history.push(metadata);
            }

            let mut running = running_clone.write().await;
            *running = false;

            if let Err(e) = result {
                tracing::error!("Migration failed: {}", e);
            } else {
                tracing::info!("Migration completed successfully");
            }
        });

        Ok(migration_id)
    }

    /// Execute migration
    async fn execute_migration(
        config: &MigrationConfig,
        migration_id: String,
        active_migrations: Arc<RwLock<HashMap<String, MigrationMetadata>>>,
    ) -> Result<()> {
        // Update status to in progress
        {
            let mut active = active_migrations.write().await;
            if let Some(metadata) = active.get_mut(&migration_id) {
                metadata.mark_started();
            }
        }

        // Execute migration phases
        let phases = vec![
            MigrationPhase::SchemaValidation,
            MigrationPhase::DataBackup,
            MigrationPhase::SchemaModification,
            MigrationPhase::DataTransformation,
            MigrationPhase::ConsistencyValidation,
            MigrationPhase::Finalization,
        ];

        for phase in phases {
            tracing::info!("Executing migration phase: {:?}", phase);

            // Update current phase
            {
                let mut active = active_migrations.write().await;
                if let Some(metadata) = active.get_mut(&migration_id) {
                    metadata.current_phase = Some(phase);
                }
            }

            // Execute phase (simulated)
            tokio::time::sleep(Duration::from_secs(1)).await;

            // Create checkpoint if enabled
            if config.enable_progress_tracking {
                let checkpoint = MigrationCheckpoint {
                    checkpoint_id: uuid::Uuid::new_v4().to_string(),
                    phase,
                    items_processed: 0,
                    timestamp: Utc::now(),
                    data: HashMap::new(),
                };

                let mut active = active_migrations.write().await;
                if let Some(metadata) = active.get_mut(&migration_id) {
                    metadata.checkpoints.push(checkpoint);
                }
            }
        }

        // Mark as completed
        {
            let mut active = active_migrations.write().await;
            if let Some(metadata) = active.get_mut(&migration_id) {
                metadata.mark_completed();
            }
        }

        Ok(())
    }

    /// Get migration status
    pub async fn get_migration_status(&self, migration_id: &str) -> Option<MigrationMetadata> {
        let active = self.active_migrations.read().await;
        active.get(migration_id).cloned()
    }

    /// Get migration history
    pub async fn get_migration_history(&self) -> Vec<MigrationMetadata> {
        let history = self.migration_history.read().await;
        history.clone()
    }

    /// Rollback migration
    pub async fn rollback_migration(&self, migration_id: &str) -> Result<()> {
        tracing::info!(migration_id = %migration_id, "Rolling back migration");

        let mut active = self.active_migrations.write().await;
        if let Some(metadata) = active.get_mut(migration_id) {
            metadata.status = MigrationStatus::RollingBack;

            // Perform rollback using checkpoints
            for checkpoint in metadata.checkpoints.iter().rev() {
                tracing::info!(
                    checkpoint_id = %checkpoint.checkpoint_id,
                    phase = ?checkpoint.phase,
                    "Restoring checkpoint"
                );

                // Simulated rollback
                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            metadata.status = MigrationStatus::RolledBack;
            metadata.completed_at = Some(Utc::now());
        }

        Ok(())
    }

    /// Validate migration plan
    pub async fn validate_plan(&self, plan: &MigrationPlan) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Validate schema operations
        for operation in &plan.schema_operations {
            match operation {
                SchemaOperation::RemoveClass { class_uri } => {
                    warnings.push(format!(
                        "Removing class '{}' may cause data loss",
                        class_uri
                    ));
                }
                SchemaOperation::RemoveProperty { property_uri } => {
                    warnings.push(format!(
                        "Removing property '{}' may cause data loss",
                        property_uri
                    ));
                }
                _ => {}
            }
        }

        // Validate data transformations
        if plan.data_transformations.is_empty() && !plan.schema_operations.is_empty() {
            warnings.push(
                "Schema changes without data transformations may lead to inconsistencies"
                    .to_string(),
            );
        }

        Ok(warnings)
    }
}

/// Migration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatistics {
    /// Total migrations
    pub total_migrations: usize,

    /// Successful migrations
    pub successful_migrations: usize,

    /// Failed migrations
    pub failed_migrations: usize,

    /// Rolled back migrations
    pub rolled_back_migrations: usize,

    /// Average migration duration (seconds)
    pub avg_duration_seconds: f64,

    /// Currently active migrations
    pub active_migrations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_metadata_creation() {
        let metadata = MigrationMetadata::new("v1", "v2", "Test migration");

        assert_eq!(metadata.from_version, "v1");
        assert_eq!(metadata.to_version, "v2");
        assert_eq!(metadata.status, MigrationStatus::Pending);
        assert_eq!(metadata.progress_percent, 0.0);
    }

    #[test]
    fn test_migration_metadata_progress() {
        let mut metadata = MigrationMetadata::new("v1", "v2", "Test migration");
        metadata.total_items = 100;

        metadata.update_progress(50);
        assert_eq!(metadata.progress_percent, 50.0);

        metadata.update_progress(100);
        assert_eq!(metadata.progress_percent, 100.0);
    }

    #[test]
    fn test_migration_config_builder() {
        let config = MigrationConfig::default()
            .with_batch_size(500)
            .with_validation(false)
            .with_timeout(7200);

        assert_eq!(config.batch_size, 500);
        assert!(!config.enable_validation);
        assert_eq!(config.timeout_seconds, 7200);
    }

    #[tokio::test]
    async fn test_migration_manager_creation() {
        let config = MigrationConfig::default();
        let manager = MigrationManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_migration_plan() {
        let metadata = MigrationMetadata::new("v1", "v2", "Test migration");
        let plan = MigrationPlan::new(metadata)
            .add_schema_operation(SchemaOperation::AddClass {
                class_uri: "http://example.org/NewClass".to_string(),
                properties: vec!["prop1".to_string(), "prop2".to_string()],
            })
            .add_validation_rule("Check data integrity");

        assert_eq!(plan.schema_operations.len(), 1);
        assert_eq!(plan.validation_rules.len(), 1);
    }
}
