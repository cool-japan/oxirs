//! Core traits for the real-time embedding pipeline

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

use crate::Vector;

/// Content to be processed for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentItem {
    /// Unique identifier for the content
    pub id: String,
    /// Content type identifier
    pub content_type: String,
    /// Raw content data
    pub content: String,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
    /// Content priority
    pub priority: ProcessingPriority,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Optional expiration time
    pub expires_at: Option<SystemTime>,
}

/// Processing priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProcessingPriority {
    /// Low priority - batch processing
    Low,
    /// Normal priority - standard processing
    Normal,
    /// High priority - expedited processing
    High,
    /// Critical priority - immediate processing
    Critical,
}

/// Processing status for content items
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProcessingStatus {
    /// Pending processing
    Pending,
    /// Currently being processed
    Processing,
    /// Successfully processed
    Completed,
    /// Processing failed
    Failed { reason: String },
    /// Processing was retried
    Retried { attempt: usize },
}

/// Result of processing a content item
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Item that was processed
    pub item: ContentItem,
    /// Generated vector (if successful)
    pub vector: Option<Vector>,
    /// Processing status
    pub status: ProcessingStatus,
    /// Processing duration
    pub duration: Duration,
    /// Any error that occurred
    pub error: Option<String>,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Trait for generating embeddings from content
pub trait EmbeddingGenerator: Send + Sync {
    /// Generate embedding vector from content
    fn generate_embedding(&self, content: &ContentItem) -> Result<Vector>;

    /// Generate embeddings for a batch of content items
    fn generate_batch_embeddings(&self, content: &[ContentItem]) -> Result<Vec<ProcessingResult>>;

    /// Get the embedding dimensions
    fn embedding_dimensions(&self) -> usize;

    /// Get generator configuration
    fn get_config(&self) -> serde_json::Value;

    /// Check if the generator is ready
    fn is_ready(&self) -> bool;

    /// Get generator statistics
    fn get_statistics(&self) -> GeneratorStatistics;
}

/// Statistics for embedding generators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorStatistics {
    /// Total embeddings generated
    pub total_embeddings: u64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Average processing time per embedding
    pub average_processing_time: Duration,
    /// Error count
    pub error_count: u64,
    /// Last error message
    pub last_error: Option<String>,
}

/// Trait for incremental vector indices
pub trait IncrementalVectorIndex: Send + Sync {
    /// Insert or update a vector
    fn upsert_vector(&mut self, id: String, vector: Vector) -> Result<()>;

    /// Remove a vector
    fn remove_vector(&mut self, id: &str) -> Result<bool>;

    /// Batch upsert vectors
    fn batch_upsert(&mut self, vectors: Vec<(String, Vector)>) -> Result<Vec<Result<()>>>;

    /// Get index statistics
    fn get_statistics(&self) -> IndexStatistics;

    /// Optimize index structure
    fn optimize(&mut self) -> Result<()>;

    /// Check index health
    fn health_check(&self) -> Result<HealthStatus>;
}

/// Statistics for vector indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    /// Total vectors in index
    pub total_vectors: usize,
    /// Index memory usage in bytes
    pub memory_usage: u64,
    /// Last optimization time
    pub last_optimization: Option<SystemTime>,
    /// Total operations performed
    pub total_operations: u64,
    /// Error count
    pub error_count: u64,
}

/// Health status for components
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    /// Component is healthy
    Healthy,
    /// Component has warnings but is functional
    Warning { message: String },
    /// Component is unhealthy
    Unhealthy { message: String },
    /// Component status is unknown
    Unknown,
}

/// Trait for handling alerts
pub trait AlertHandler: Send + Sync {
    /// Handle an alert
    fn handle_alert(&self, alert: &Alert) -> Result<()>;

    /// Get alert configuration
    fn get_config(&self) -> AlertConfig;

    /// Check if handler is enabled
    fn is_enabled(&self) -> bool;
}

/// Alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert identifier
    pub id: Uuid,
    /// Alert severity level
    pub severity: AlertSeverity,
    /// Alert category
    pub category: AlertCategory,
    /// Alert message
    pub message: String,
    /// Alert details
    pub details: HashMap<String, String>,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Source component
    pub source: String,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Information level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

/// Alert categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertCategory {
    /// Performance related
    Performance,
    /// Quality related
    Quality,
    /// System health related
    Health,
    /// Security related
    Security,
    /// Configuration related
    Configuration,
}

/// Alert handler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Minimum severity level to handle
    pub min_severity: AlertSeverity,
    /// Alert throttling settings
    pub throttling: AlertThrottling,
    /// Enable notifications
    pub enable_notifications: bool,
}

/// Alert throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThrottling {
    /// Enable throttling
    pub enabled: bool,
    /// Throttling window duration
    pub window_duration: Duration,
    /// Maximum alerts per window
    pub max_alerts_per_window: usize,
}

/// Trait for storing metrics
pub trait MetricsStorage: Send + Sync {
    /// Store a metric value
    fn store_metric(
        &mut self,
        name: &str,
        value: f64,
        timestamp: SystemTime,
        tags: HashMap<String, String>,
    ) -> Result<()>;

    /// Get metric values within a time range
    fn get_metrics(
        &self,
        name: &str,
        start: SystemTime,
        end: SystemTime,
    ) -> Result<Vec<MetricPoint>>;

    /// Get available metric names
    fn get_metric_names(&self) -> Result<Vec<String>>;

    /// Delete old metrics
    fn cleanup_old_metrics(&mut self, cutoff: SystemTime) -> Result<usize>;
}

/// A single metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric value
    pub value: f64,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Associated tags
    pub tags: HashMap<String, String>,
}

/// Trait for version storage
pub trait VersionStorage: Send + Sync {
    /// Store a new version
    fn store_version(&mut self, id: &str, version: &Version) -> Result<()>;

    /// Get a specific version
    fn get_version(&self, id: &str, version_number: u64) -> Result<Option<Version>>;

    /// Get all versions for an ID
    fn get_all_versions(&self, id: &str) -> Result<Vec<Version>>;

    /// Delete old versions
    fn cleanup_old_versions(&mut self, id: &str, keep_count: usize) -> Result<usize>;
}

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    /// Version number
    pub version: u64,
    /// Vector data
    pub vector: Vector,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Checksum for integrity
    pub checksum: String,
}

/// Trait for conflict resolution functions
pub trait ConflictResolutionFunction: Send + Sync {
    /// Resolve conflicts between versions
    fn resolve_conflict(&self, versions: &[Version]) -> Result<Vector>;

    /// Get resolution strategy name
    fn get_strategy_name(&self) -> &str;
}

/// Trait for transaction logging
pub trait TransactionLog: Send + Sync {
    /// Log a transaction
    fn log_transaction(&mut self, transaction: &Transaction) -> Result<()>;

    /// Get transactions within a time range
    fn get_transactions(&self, start: SystemTime, end: SystemTime) -> Result<Vec<Transaction>>;

    /// Replay transactions from a specific point
    fn replay_from(&self, checkpoint: SystemTime) -> Result<Vec<Transaction>>;
}

/// Transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction ID
    pub id: Uuid,
    /// Transaction type
    pub transaction_type: TransactionType,
    /// Affected resource ID
    pub resource_id: String,
    /// Transaction timestamp
    pub timestamp: SystemTime,
    /// Transaction data
    pub data: serde_json::Value,
    /// Transaction status
    pub status: TransactionStatus,
}

/// Transaction types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionType {
    /// Insert operation
    Insert,
    /// Update operation
    Update,
    /// Delete operation
    Delete,
    /// Batch operation
    Batch,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    /// Transaction is pending
    Pending,
    /// Transaction is committed
    Committed,
    /// Transaction was rolled back
    RolledBack,
    /// Transaction failed
    Failed { reason: String },
}

/// Trait for inconsistency detection
pub trait InconsistencyDetectionAlgorithm: Send + Sync {
    /// Detect inconsistencies in the system
    fn detect_inconsistencies(&self) -> Result<Vec<Inconsistency>>;

    /// Get detection algorithm name
    fn get_algorithm_name(&self) -> &str;
}

/// Inconsistency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inconsistency {
    /// Inconsistency type
    pub inconsistency_type: InconsistencyType,
    /// Affected resources
    pub affected_resources: Vec<String>,
    /// Inconsistency description
    pub description: String,
    /// Severity level
    pub severity: InconsistencySeverity,
    /// Detection timestamp
    pub detected_at: SystemTime,
}

/// Types of inconsistencies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InconsistencyType {
    /// Data mismatch
    DataMismatch,
    /// Missing data
    MissingData,
    /// Duplicate data
    DuplicateData,
    /// Stale data
    StaleData,
    /// Corrupted data
    CorruptedData,
}

/// Inconsistency severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum InconsistencySeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Trait for consistency repair strategies
pub trait ConsistencyRepairStrategy: Send + Sync {
    /// Repair inconsistencies
    fn repair_inconsistencies(
        &self,
        inconsistencies: &[Inconsistency],
    ) -> Result<Vec<RepairResult>>;

    /// Get repair strategy name
    fn get_strategy_name(&self) -> &str;
}

/// Result of a repair operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairResult {
    /// Inconsistency that was repaired
    pub inconsistency: Inconsistency,
    /// Repair status
    pub status: RepairStatus,
    /// Repair actions taken
    pub actions: Vec<String>,
    /// Repair timestamp
    pub repaired_at: SystemTime,
}

/// Status of repair operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RepairStatus {
    /// Repair was successful
    Success,
    /// Repair was partially successful
    PartialSuccess,
    /// Repair failed
    Failed { reason: String },
    /// Repair was skipped
    Skipped { reason: String },
}
