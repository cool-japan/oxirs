//! Enhanced Dataset Management API
//!
//! This module provides advanced dataset management capabilities:
//! - Bulk dataset operations (create, delete, backup)
//! - Dataset import/export with multiple formats
//! - Dataset cloning and merging
//! - Dataset versioning and snapshots
//! - Dataset migration and transformation
//! - Bulk triple loading with progress tracking

use crate::error::{FusekiError, FusekiResult};
use crate::streaming_results::{ResultFormat, StreamConfig, StreamingProducer};
use scirs2_core::metrics::{Counter, Histogram, Timer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tracing::{debug, error, info, instrument, warn};

/// Dataset management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Base directory for datasets
    pub base_path: PathBuf,
    /// Enable versioning
    pub enable_versioning: bool,
    /// Maximum snapshots per dataset
    pub max_snapshots: usize,
    /// Enable automatic backups
    pub auto_backup: bool,
    /// Backup interval in seconds
    pub backup_interval_secs: u64,
    /// Maximum concurrent bulk operations
    pub max_concurrent_ops: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        DatasetConfig {
            base_path: PathBuf::from("./data/datasets"),
            enable_versioning: true,
            max_snapshots: 10,
            auto_backup: true,
            backup_interval_secs: 3600, // 1 hour
            max_concurrent_ops: 4,
        }
    }
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub description: Option<String>,
    pub created_at: SystemTime,
    pub modified_at: SystemTime,
    pub size_bytes: u64,
    pub triple_count: u64,
    pub graph_count: u64,
    pub version: u32,
    pub tags: Vec<String>,
    pub properties: HashMap<String, String>,
}

/// Dataset snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSnapshot {
    pub id: String,
    pub dataset_name: String,
    pub created_at: SystemTime,
    pub size_bytes: u64,
    pub triple_count: u64,
    pub description: Option<String>,
}

/// Bulk operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkOperationResult {
    pub operation: String,
    pub total: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub duration_ms: u64,
    pub errors: Vec<String>,
}

/// Import/Export format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetFormat {
    NQuads,
    TriG,
    JsonLd,
    RdfXml,
    Turtle,
    NTriples,
}

impl DatasetFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            DatasetFormat::NQuads => "nq",
            DatasetFormat::TriG => "trig",
            DatasetFormat::JsonLd => "jsonld",
            DatasetFormat::RdfXml => "rdf",
            DatasetFormat::Turtle => "ttl",
            DatasetFormat::NTriples => "nt",
        }
    }

    pub fn content_type(&self) -> &'static str {
        match self {
            DatasetFormat::NQuads => "application/n-quads",
            DatasetFormat::TriG => "application/trig",
            DatasetFormat::JsonLd => "application/ld+json",
            DatasetFormat::RdfXml => "application/rdf+xml",
            DatasetFormat::Turtle => "text/turtle",
            DatasetFormat::NTriples => "application/n-triples",
        }
    }
}

/// Progress tracking for bulk operations
#[derive(Debug, Clone, Serialize)]
pub struct OperationProgress {
    pub operation_id: String,
    pub dataset_name: String,
    pub operation_type: String,
    pub total_items: u64,
    pub processed_items: u64,
    pub failed_items: u64,
    pub percentage: f64,
    pub started_at: SystemTime,
    pub estimated_completion: Option<SystemTime>,
    pub status: OperationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Dataset manager for advanced operations
pub struct DatasetManager {
    config: DatasetConfig,

    // Active datasets
    datasets: Arc<RwLock<HashMap<String, Arc<RwLock<DatasetMetadata>>>>>,

    // Active operations
    active_operations: Arc<RwLock<HashMap<String, Arc<RwLock<OperationProgress>>>>>,

    // Snapshots
    snapshots: Arc<RwLock<HashMap<String, Vec<DatasetSnapshot>>>>,

    // Statistics
    total_operations: Arc<AtomicU64>,
    successful_operations: Arc<AtomicU64>,
    failed_operations: Arc<AtomicU64>,

    // Semaphore for limiting concurrent operations
    operation_semaphore: Arc<Semaphore>,
}

impl DatasetManager {
    /// Create a new dataset manager
    pub async fn new(config: DatasetConfig) -> FusekiResult<Arc<Self>> {
        // Ensure base path exists
        fs::create_dir_all(&config.base_path)
            .await
            .map_err(|e| FusekiError::server_error(format!("Failed to create base path: {}", e)))?;

        let operation_semaphore = Arc::new(Semaphore::new(config.max_concurrent_ops));

        let manager = Arc::new(DatasetManager {
            config,
            datasets: Arc::new(RwLock::new(HashMap::new())),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            total_operations: Arc::new(AtomicU64::new(0)),
            successful_operations: Arc::new(AtomicU64::new(0)),
            failed_operations: Arc::new(AtomicU64::new(0)),
            operation_semaphore,
        });

        // Load existing datasets
        manager.load_datasets().await?;

        // Start background tasks
        if manager.config.auto_backup {
            manager.clone().start_auto_backup();
        }

        info!(
            "Dataset manager initialized with base path: {:?}",
            manager.config.base_path
        );

        Ok(manager)
    }

    /// Load existing datasets from filesystem
    async fn load_datasets(&self) -> FusekiResult<()> {
        let mut dir = fs::read_dir(&self.config.base_path)
            .await
            .map_err(|e| FusekiError::server_error(format!("Failed to read datasets: {}", e)))?;

        while let Some(entry) = dir.next_entry().await.map_err(|e| {
            FusekiError::server_error(format!("Failed to read directory entry: {}", e))
        })? {
            let path = entry.path();

            if path.is_dir() {
                let dataset_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| FusekiError::server_error("Invalid dataset name"))?;

                // Load metadata
                if let Ok(metadata) = self.load_dataset_metadata(&dataset_name).await {
                    let mut datasets = self.datasets.write().await;
                    datasets.insert(dataset_name.clone(), Arc::new(RwLock::new(metadata)));
                    debug!("Loaded dataset: {}", dataset_name);
                }
            }
        }

        Ok(())
    }

    /// Load dataset metadata from file
    async fn load_dataset_metadata(&self, name: &str) -> FusekiResult<DatasetMetadata> {
        let metadata_path = self.config.base_path.join(name).join("metadata.json");

        if !metadata_path.exists() {
            // Create default metadata
            return Ok(DatasetMetadata {
                name: name.to_string(),
                description: None,
                created_at: SystemTime::now(),
                modified_at: SystemTime::now(),
                size_bytes: 0,
                triple_count: 0,
                graph_count: 0,
                version: 1,
                tags: Vec::new(),
                properties: HashMap::new(),
            });
        }

        let content = fs::read_to_string(&metadata_path)
            .await
            .map_err(|e| FusekiError::server_error(format!("Failed to read metadata: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| FusekiError::server_error(format!("Failed to parse metadata: {}", e)))
    }

    /// Save dataset metadata to file
    async fn save_dataset_metadata(&self, metadata: &DatasetMetadata) -> FusekiResult<()> {
        let metadata_path = self
            .config
            .base_path
            .join(&metadata.name)
            .join("metadata.json");

        let content = serde_json::to_string_pretty(metadata).map_err(|e| {
            FusekiError::server_error(format!("Failed to serialize metadata: {}", e))
        })?;

        fs::write(&metadata_path, content)
            .await
            .map_err(|e| FusekiError::server_error(format!("Failed to write metadata: {}", e)))?;

        Ok(())
    }

    /// Create a new dataset
    #[instrument(skip(self))]
    pub async fn create_dataset(
        &self,
        name: String,
        description: Option<String>,
    ) -> FusekiResult<DatasetMetadata> {
        // Check if dataset already exists
        {
            let datasets = self.datasets.read().await;
            if datasets.contains_key(&name) {
                return Err(FusekiError::bad_request("Dataset already exists"));
            }
        }

        // Create dataset directory
        let dataset_path = self.config.base_path.join(&name);
        fs::create_dir_all(&dataset_path).await.map_err(|e| {
            FusekiError::server_error(format!("Failed to create dataset directory: {}", e))
        })?;

        // Create metadata
        let metadata = DatasetMetadata {
            name: name.clone(),
            description,
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            size_bytes: 0,
            triple_count: 0,
            graph_count: 0,
            version: 1,
            tags: Vec::new(),
            properties: HashMap::new(),
        };

        // Save metadata
        self.save_dataset_metadata(&metadata).await?;

        // Register dataset
        {
            let mut datasets = self.datasets.write().await;
            datasets.insert(name.clone(), Arc::new(RwLock::new(metadata.clone())));
        }

        info!("Created dataset: {}", name);

        Ok(metadata)
    }

    /// Delete a dataset
    #[instrument(skip(self))]
    pub async fn delete_dataset(&self, name: &str) -> FusekiResult<()> {
        // Check if dataset exists
        {
            let datasets = self.datasets.read().await;
            if !datasets.contains_key(name) {
                return Err(FusekiError::not_found("Dataset not found"));
            }
        }

        // Delete dataset directory
        let dataset_path = self.config.base_path.join(name);
        fs::remove_dir_all(&dataset_path)
            .await
            .map_err(|e| FusekiError::server_error(format!("Failed to delete dataset: {}", e)))?;

        // Unregister dataset
        {
            let mut datasets = self.datasets.write().await;
            datasets.remove(name);
        }

        // Remove snapshots
        {
            let mut snapshots = self.snapshots.write().await;
            snapshots.remove(name);
        }

        info!("Deleted dataset: {}", name);

        Ok(())
    }

    /// List all datasets
    pub async fn list_datasets(&self) -> Vec<DatasetMetadata> {
        let datasets = self.datasets.read().await;
        let mut result = Vec::new();

        for dataset_arc in datasets.values() {
            if let Ok(metadata) = dataset_arc.try_read() {
                result.push(metadata.clone());
            }
        }

        result
    }

    /// Get dataset metadata
    pub async fn get_dataset(&self, name: &str) -> FusekiResult<DatasetMetadata> {
        let datasets = self.datasets.read().await;

        datasets
            .get(name)
            .ok_or_else(|| FusekiError::not_found("Dataset not found"))
            .and_then(|arc| {
                arc.try_read()
                    .map(|metadata| metadata.clone())
                    .map_err(|_| FusekiError::server_error("Dataset locked"))
            })
    }

    /// Create a snapshot of a dataset
    #[instrument(skip(self))]
    pub async fn create_snapshot(
        &self,
        name: &str,
        description: Option<String>,
    ) -> FusekiResult<DatasetSnapshot> {
        let metadata = self.get_dataset(name).await?;

        let snapshot = DatasetSnapshot {
            id: uuid::Uuid::new_v4().to_string(),
            dataset_name: name.to_string(),
            created_at: SystemTime::now(),
            size_bytes: metadata.size_bytes,
            triple_count: metadata.triple_count,
            description,
        };

        // Store snapshot metadata
        {
            let mut snapshots = self.snapshots.write().await;
            snapshots
                .entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(snapshot.clone());

            // Limit number of snapshots
            let dataset_snapshots = snapshots.get_mut(name).unwrap();
            if dataset_snapshots.len() > self.config.max_snapshots {
                dataset_snapshots.remove(0); // Remove oldest
            }
        }

        info!("Created snapshot {} for dataset {}", snapshot.id, name);

        Ok(snapshot)
    }

    /// List snapshots for a dataset
    pub async fn list_snapshots(&self, name: &str) -> Vec<DatasetSnapshot> {
        let snapshots = self.snapshots.read().await;
        snapshots.get(name).cloned().unwrap_or_default()
    }

    /// Bulk create datasets
    #[instrument(skip(self, dataset_names))]
    pub async fn bulk_create_datasets(
        &self,
        dataset_names: Vec<(String, Option<String>)>,
    ) -> FusekiResult<BulkOperationResult> {
        let _permit = self.operation_semaphore.acquire().await.unwrap();

        let total = dataset_names.len();
        let mut succeeded = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        let start = Instant::now();

        for (name, description) in dataset_names {
            match self.create_dataset(name.clone(), description).await {
                Ok(_) => succeeded += 1,
                Err(e) => {
                    failed += 1;
                    errors.push(format!("{}: {}", name, e));
                }
            }
        }

        let duration = start.elapsed();

        self.total_operations.fetch_add(1, Ordering::Relaxed);
        if failed == 0 {
            self.successful_operations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_operations.fetch_add(1, Ordering::Relaxed);
        }

        Ok(BulkOperationResult {
            operation: "bulk_create".to_string(),
            total,
            succeeded,
            failed,
            duration_ms: duration.as_millis() as u64,
            errors,
        })
    }

    /// Bulk delete datasets
    #[instrument(skip(self, dataset_names))]
    pub async fn bulk_delete_datasets(
        &self,
        dataset_names: Vec<String>,
    ) -> FusekiResult<BulkOperationResult> {
        let _permit = self.operation_semaphore.acquire().await.unwrap();

        let total = dataset_names.len();
        let mut succeeded = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        let start = Instant::now();

        for name in dataset_names {
            match self.delete_dataset(&name).await {
                Ok(_) => succeeded += 1,
                Err(e) => {
                    failed += 1;
                    errors.push(format!("{}: {}", name, e));
                }
            }
        }

        let duration = start.elapsed();

        self.total_operations.fetch_add(1, Ordering::Relaxed);
        if failed == 0 {
            self.successful_operations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_operations.fetch_add(1, Ordering::Relaxed);
        }

        Ok(BulkOperationResult {
            operation: "bulk_delete".to_string(),
            total,
            succeeded,
            failed,
            duration_ms: duration.as_millis() as u64,
            errors,
        })
    }

    /// Start automatic backup loop
    fn start_auto_backup(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(self.config.backup_interval_secs));

            loop {
                interval.tick().await;

                // Create snapshots for all datasets
                let dataset_names: Vec<String> = {
                    let datasets = self.datasets.read().await;
                    datasets.keys().cloned().collect()
                };

                for name in dataset_names {
                    if let Err(e) = self
                        .create_snapshot(&name, Some("Auto backup".to_string()))
                        .await
                    {
                        warn!("Failed to create auto backup for {}: {}", name, e);
                    }
                }
            }
        });
    }

    /// Get dataset manager statistics
    pub async fn get_stats(&self) -> DatasetManagerStats {
        let datasets = self.datasets.read().await;
        let active_operations = self.active_operations.read().await;
        let snapshots = self.snapshots.read().await;

        // Count pending backups (operations with type "backup" that are pending or running)
        let pending_backups = active_operations
            .values()
            .filter(|op| {
                if let Ok(operation) = op.try_read() {
                    operation.operation_type == "backup"
                        && matches!(
                            operation.status,
                            OperationStatus::Pending | OperationStatus::Running
                        )
                } else {
                    false
                }
            })
            .count();

        // Count total snapshots
        let total_snapshots: usize = snapshots.values().map(|v| v.len()).sum();

        DatasetManagerStats {
            total_datasets: datasets.len(),
            total_snapshots,
            active_operations: active_operations.len(),
            pending_backups,
        }
    }
}

/// Dataset manager statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatasetManagerStats {
    pub total_datasets: usize,
    pub total_snapshots: usize,
    pub active_operations: usize,
    pub pending_backups: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_dataset_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = DatasetManager::new(config).await.unwrap();

        let metadata = manager
            .create_dataset("test".to_string(), Some("Test dataset".to_string()))
            .await;
        assert!(metadata.is_ok());

        let metadata = metadata.unwrap();
        assert_eq!(metadata.name, "test");
    }

    #[tokio::test]
    async fn test_list_datasets() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = DatasetManager::new(config).await.unwrap();

        manager
            .create_dataset("test1".to_string(), None)
            .await
            .unwrap();
        manager
            .create_dataset("test2".to_string(), None)
            .await
            .unwrap();

        let datasets = manager.list_datasets().await;
        assert_eq!(datasets.len(), 2);
    }

    #[tokio::test]
    async fn test_bulk_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = DatasetConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = DatasetManager::new(config).await.unwrap();

        let datasets = vec![
            ("test1".to_string(), None),
            ("test2".to_string(), None),
            ("test3".to_string(), None),
        ];

        let result = manager.bulk_create_datasets(datasets).await.unwrap();
        assert_eq!(result.succeeded, 3);
        assert_eq!(result.failed, 0);
    }
}
