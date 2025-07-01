//! Configuration types for the real-time embedding pipeline

use serde::{Deserialize, Serialize};

/// Pipeline configuration for real-time embedding updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Maximum concurrent updates
    pub max_concurrent_updates: usize,
    /// Buffer size for each stream
    pub stream_buffer_size: usize,
    /// Update timeout in seconds
    pub update_timeout_seconds: u64,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Backpressure strategy
    pub backpressure_strategy: BackpressureStrategy,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Performance monitoring settings
    pub monitoring_config: MonitoringConfig,
    /// Version control settings
    pub version_control: VersionControlConfig,
    /// Quality assurance settings
    pub quality_assurance: QualityAssuranceConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            batch_timeout_ms: 100,
            max_concurrent_updates: 10,
            stream_buffer_size: 10000,
            update_timeout_seconds: 30,
            consistency_level: ConsistencyLevel::Session,
            backpressure_strategy: BackpressureStrategy::Adaptive,
            retry_config: RetryConfig::default(),
            monitoring_config: MonitoringConfig::default(),
            version_control: VersionControlConfig::default(),
            quality_assurance: QualityAssuranceConfig::default(),
        }
    }
}

/// Consistency levels for updates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsistencyLevel {
    /// Eventually consistent - fast but may have temporary inconsistencies
    Eventual,
    /// Session consistent - consistent within a session
    Session,
    /// Strong consistency - always consistent but slower
    Strong,
    /// Causal consistency - maintains causal ordering
    Causal,
    /// Monotonic read consistency
    MonotonicRead,
}

/// Backpressure handling strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackpressureStrategy {
    /// Drop oldest updates when buffer is full
    DropOldest,
    /// Drop newest updates when buffer is full
    DropNewest,
    /// Block until buffer has space
    Block,
    /// Adaptive strategy based on system load
    Adaptive,
    /// Exponential backoff
    ExponentialBackoff { initial_delay_ms: u64, max_delay_ms: u64 },
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: usize,
    /// Base delay between retries
    pub base_delay_ms: u64,
    /// Maximum delay between retries
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor for randomization
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Metrics collection interval
    pub metrics_interval_ms: u64,
    /// Performance alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Enable detailed tracing
    pub enable_tracing: bool,
    /// Metrics retention period
    pub metrics_retention_hours: u64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_interval_ms: 1000,
            alert_thresholds: AlertThresholds::default(),
            enable_tracing: false,
            metrics_retention_hours: 24,
        }
    }
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Maximum processing latency in milliseconds
    pub max_processing_latency_ms: u64,
    /// Maximum queue depth
    pub max_queue_depth: usize,
    /// Maximum error rate (0.0 to 1.0)
    pub max_error_rate: f64,
    /// Maximum memory usage in MB
    pub max_memory_usage_mb: f64,
    /// Minimum throughput (updates per second)
    pub min_throughput_ups: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_processing_latency_ms: 1000,
            max_queue_depth: 10000,
            max_error_rate: 0.05,
            max_memory_usage_mb: 1024.0,
            min_throughput_ups: 100.0,
        }
    }
}

/// Version control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionControlConfig {
    /// Enable versioning
    pub enable_versioning: bool,
    /// Maximum versions to keep
    pub max_versions: usize,
    /// Version compression threshold
    pub compression_threshold: usize,
    /// Enable incremental versioning
    pub enable_incremental: bool,
}

impl Default for VersionControlConfig {
    fn default() -> Self {
        Self {
            enable_versioning: true,
            max_versions: 10,
            compression_threshold: 1000,
            enable_incremental: true,
        }
    }
}

/// Quality assurance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssuranceConfig {
    /// Enable quality checking
    pub enable_quality_checks: bool,
    /// Quality threshold (0.0 to 1.0)
    pub quality_threshold: f64,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly sensitivity (0.0 to 1.0)
    pub anomaly_sensitivity: f64,
}

impl Default for QualityAssuranceConfig {
    fn default() -> Self {
        Self {
            enable_quality_checks: true,
            quality_threshold: 0.8,
            enable_anomaly_detection: true,
            anomaly_sensitivity: 0.7,
        }
    }
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enable_autoscaling: bool,
    /// Minimum number of workers
    pub min_workers: usize,
    /// Maximum number of workers
    pub max_workers: usize,
    /// CPU threshold for scaling up
    pub scale_up_cpu_threshold: f64,
    /// CPU threshold for scaling down
    pub scale_down_cpu_threshold: f64,
    /// Memory threshold for scaling up
    pub scale_up_memory_threshold: f64,
    /// Queue depth threshold for scaling up
    pub scale_up_queue_threshold: usize,
    /// Cooldown period between scaling operations (seconds)
    pub scaling_cooldown_seconds: u64,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enable_autoscaling: true,
            min_workers: 2,
            max_workers: 20,
            scale_up_cpu_threshold: 0.8,
            scale_down_cpu_threshold: 0.3,
            scale_up_memory_threshold: 0.8,
            scale_up_queue_threshold: 1000,
            scaling_cooldown_seconds: 300,
        }
    }
}

/// Compression configuration for vector storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable vector compression
    pub enable_compression: bool,
    /// Compression method
    pub compression_method: CompressionMethod,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Compression threshold (minimum vector size to compress)
    pub compression_threshold: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_method: CompressionMethod::Quantization,
            compression_level: 6,
            compression_threshold: 100,
        }
    }
}

/// Compression methods for vectors
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressionMethod {
    /// Product quantization
    ProductQuantization,
    /// Scalar quantization
    ScalarQuantization,
    /// General quantization
    Quantization,
    /// PCA compression
    PCA,
    /// Dictionary compression
    Dictionary,
}