//! Common types and data structures for the real-time embedding pipeline

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Statistics for the entire pipeline
#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineStatistics {
    /// Total items processed
    pub total_processed: AtomicU64,
    /// Total items failed
    pub total_failed: AtomicU64,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Current throughput (items per second)
    pub current_throughput: f64,
    /// Peak throughput achieved
    pub peak_throughput: f64,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Pipeline start time
    pub start_time: SystemTime,
    /// Last update time
    pub last_update: SystemTime,
}

impl Default for PipelineStatistics {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            total_processed: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            avg_processing_time: Duration::from_millis(0),
            current_throughput: 0.0,
            peak_throughput: 0.0,
            total_processing_time: Duration::from_millis(0),
            start_time: now,
            last_update: now,
        }
    }
}

/// Versioning strategy for content updates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VersioningStrategy {
    /// No versioning - always overwrite
    None,
    /// Simple versioning - keep N versions
    Simple { max_versions: usize },
    /// Time-based versioning - keep versions within time window
    TimeBased { retention_period: Duration },
    /// Smart versioning - adaptive based on change frequency
    Smart { change_threshold: f64 },
}

impl Default for VersioningStrategy {
    fn default() -> Self {
        Self::Simple { max_versions: 5 }
    }
}

/// Update operation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UpdateOperation {
    /// Insert new content
    Insert { id: String, content: String },
    /// Update existing content
    Update {
        id: String,
        content: String,
        version: Option<u64>,
    },
    /// Delete content
    Delete { id: String },
    /// Batch operations
    Batch { operations: Vec<UpdateOperation> },
}

/// Update batch for efficient processing
#[derive(Debug, Clone)]
pub struct UpdateBatch {
    /// Batch identifier
    pub id: Uuid,
    /// Operations in this batch
    pub operations: Vec<UpdateOperation>,
    /// Batch creation time
    pub created_at: Instant,
    /// Batch priority
    pub priority: crate::real_time_embedding_pipeline::traits::ProcessingPriority,
    /// Batch metadata
    pub metadata: HashMap<String, String>,
}

/// Priority levels for update processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    /// Background processing
    Background,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Urgent processing
    Urgent,
}

/// Configuration for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Maximum concurrent updates
    pub max_concurrent_updates: usize,
    /// Update queue size
    pub queue_size: usize,
    /// Enable real-time indexing
    pub enable_real_time_indexing: bool,
    /// Enable background optimization
    pub enable_background_optimization: bool,
    /// Optimization interval in seconds
    pub optimization_interval_seconds: u64,
}

impl Default for RealTimeConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            batch_timeout_ms: 100,
            max_concurrent_updates: 10,
            queue_size: 10000,
            enable_real_time_indexing: true,
            enable_background_optimization: true,
            optimization_interval_seconds: 300,
        }
    }
}

/// Statistics for update operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStats {
    /// Total updates processed
    pub total_updates: u64,
    /// Updates per second
    pub updates_per_second: f64,
    /// Average update latency
    pub avg_latency: Duration,
    /// Queue depth
    pub queue_depth: usize,
    /// Failed updates
    pub failed_updates: u64,
    /// Last update time
    pub last_update: SystemTime,
}

impl Default for UpdateStats {
    fn default() -> Self {
        Self {
            total_updates: 0,
            updates_per_second: 0.0,
            avg_latency: Duration::from_millis(0),
            queue_depth: 0,
            failed_updates: 0,
            last_update: SystemTime::now(),
        }
    }
}

/// Stream processing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamState {
    /// Stream identifier
    pub stream_id: String,
    /// Current offset
    pub offset: u64,
    /// Last processed timestamp
    pub last_processed: SystemTime,
    /// Stream status
    pub status: StreamStatus,
    /// Error count
    pub error_count: u64,
    /// Last error message
    pub last_error: Option<String>,
}

/// Stream processing status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamStatus {
    /// Stream is active and processing
    Active,
    /// Stream is paused
    Paused,
    /// Stream has stopped due to error
    Error,
    /// Stream has completed
    Completed,
    /// Stream is initializing
    Initializing,
}

/// Coordination state for distributed processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationState {
    /// Node identifier
    pub node_id: String,
    /// Leader node identifier
    pub leader_id: Option<String>,
    /// Node status
    pub status: NodeStatus,
    /// Last heartbeat time
    pub last_heartbeat: SystemTime,
    /// Node load factor
    pub load_factor: f64,
    /// Active tasks
    pub active_tasks: usize,
}

/// Node status in the cluster
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    /// Node is healthy and active
    Active,
    /// Node is temporarily unavailable
    Unavailable,
    /// Node is joining the cluster
    Joining,
    /// Node is leaving the cluster
    Leaving,
    /// Node has failed
    Failed,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network bandwidth usage
    pub network_usage: f64,
    /// Disk I/O operations per second
    pub disk_iops: f64,
    /// Queue depths for different operations
    pub queue_depths: HashMap<String, usize>,
    /// Latency measurements
    pub latencies: HashMap<String, Duration>,
    /// Throughput measurements
    pub throughputs: HashMap<String, f64>,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            network_usage: 0.0,
            disk_iops: 0.0,
            queue_depths: HashMap::new(),
            latencies: HashMap::new(),
            throughputs: HashMap::new(),
            error_rates: HashMap::new(),
        }
    }
}

/// Quality metrics for content processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Vector quality score (0.0 to 1.0)
    pub vector_quality: f64,
    /// Embedding consistency score
    pub consistency_score: f64,
    /// Anomaly score
    pub anomaly_score: f64,
    /// Confidence score
    pub confidence_score: f64,
    /// Validation errors
    pub validation_errors: Vec<String>,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            vector_quality: 1.0,
            consistency_score: 1.0,
            anomaly_score: 0.0,
            confidence_score: 1.0,
            validation_errors: Vec::new(),
        }
    }
}

/// Backpressure control state
#[derive(Debug, Clone)]
pub struct BackpressureState {
    /// Current pressure level (0.0 to 1.0)
    pub pressure_level: f64,
    /// Active backpressure strategy
    pub active_strategy: crate::real_time_embedding_pipeline::config::BackpressureStrategy,
    /// Pressure history for trend analysis
    pub pressure_history: VecDeque<(SystemTime, f64)>,
    /// Number of dropped items
    pub dropped_items: u64,
    /// Number of delayed items
    pub delayed_items: u64,
}

impl Default for BackpressureState {
    fn default() -> Self {
        Self {
            pressure_level: 0.0,
            active_strategy:
                crate::real_time_embedding_pipeline::config::BackpressureStrategy::Adaptive,
            pressure_history: VecDeque::new(),
            dropped_items: 0,
            delayed_items: 0,
        }
    }
}

/// Circuit breaker state for fault tolerance
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    /// Circuit is closed - normal operation
    Closed,
    /// Circuit is open - failing fast
    Open { opened_at: SystemTime },
    /// Circuit is half-open - testing recovery
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: usize,
    /// Time window for failure counting
    pub failure_window: Duration,
    /// Recovery timeout before trying half-open
    pub recovery_timeout: Duration,
    /// Success threshold to close circuit from half-open
    pub success_threshold: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            failure_window: Duration::from_secs(60),
            recovery_timeout: Duration::from_secs(300),
            success_threshold: 3,
        }
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU cores in use
    pub cpu_cores: f64,
    /// Memory in use (bytes)
    pub memory_bytes: u64,
    /// Storage in use (bytes)
    pub storage_bytes: u64,
    /// Network bandwidth in use (bytes/sec)
    pub network_bandwidth: u64,
    /// Open file descriptors
    pub open_files: usize,
    /// Thread count
    pub thread_count: usize,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_cores: 0.0,
            memory_bytes: 0,
            storage_bytes: 0,
            network_bandwidth: 0,
            open_files: 0,
            thread_count: 0,
        }
    }
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Overall health status
    pub status: crate::real_time_embedding_pipeline::traits::HealthStatus,
    /// Individual component health
    pub components: HashMap<String, crate::real_time_embedding_pipeline::traits::HealthStatus>,
    /// Health check timestamp
    pub timestamp: SystemTime,
    /// Additional details
    pub details: HashMap<String, String>,
}
