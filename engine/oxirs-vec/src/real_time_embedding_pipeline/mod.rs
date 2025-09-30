//! Real-time embedding pipeline module
//!
//! This module provides a comprehensive real-time embedding update system that can handle
//! high-throughput streaming data with low-latency updates while maintaining consistency
//! and performance guarantees.
//!
//! ## Architecture
//!
//! The real-time embedding pipeline is organized into several components:
//!
//! - **Configuration**: Pipeline configuration and settings
//! - **Traits**: Core traits for extensibility
//! - **Types**: Common data structures and types
//! - **Pipeline**: Main pipeline implementation
//! - **Streaming**: Stream processing components
//! - **Coordination**: Update coordination and synchronization
//! - **Monitoring**: Performance monitoring and alerting
//! - **Versioning**: Version management and storage
//! - **Consistency**: Consistency management and repair
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxirs_vec::real_time_embedding_pipeline::{
//!     RealTimeEmbeddingPipeline, PipelineConfig, ConsistencyLevel
//! };
//!
//! // Create pipeline configuration
//! let config = PipelineConfig {
//!     max_batch_size: 1000,
//!     consistency_level: ConsistencyLevel::Session,
//!     ..Default::default()
//! };
//!
//! // Create and start pipeline
//! let mut pipeline = RealTimeEmbeddingPipeline::new(config)?;
//! pipeline.start().await?;
//! ```

pub mod config;
pub mod pipeline;
pub mod streaming;
pub mod traits;
pub mod types;
// TODO: Implement these modules
// pub mod coordination;
// pub mod monitoring;
// pub mod versioning;
// pub mod consistency;

// Re-export commonly used types
pub use config::{
    AlertThresholds, AutoScalingConfig, BackpressureStrategy, CompressionConfig, CompressionMethod,
    ConsistencyLevel, MonitoringConfig, PipelineConfig, QualityAssuranceConfig, RetryConfig,
    VersionControlConfig,
};

pub use traits::{
    Alert, AlertCategory, AlertConfig, AlertHandler, AlertSeverity, AlertThrottling,
    ConflictResolutionFunction, ConsistencyRepairStrategy, ContentItem, EmbeddingGenerator,
    GeneratorStatistics, HealthStatus, Inconsistency, InconsistencyDetectionAlgorithm,
    InconsistencySeverity, InconsistencyType, IncrementalVectorIndex, IndexStatistics, MetricPoint,
    MetricsStorage, ProcessingPriority, ProcessingResult, ProcessingStatus, RepairResult,
    RepairStatus, Transaction, TransactionLog, TransactionStatus, TransactionType, Version,
    VersionStorage,
};

pub use types::{
    BackpressureState, CircuitBreakerConfig, CircuitBreakerState, CoordinationState,
    HealthCheckResult, NodeStatus, PerformanceMetrics, PipelineStatistics, QualityMetrics,
    RealTimeConfig, ResourceUtilization, StreamState, StreamStatus, UpdateBatch, UpdateOperation,
    UpdatePriority, UpdateStats, VersioningStrategy,
};

pub use pipeline::RealTimeEmbeddingPipeline;
pub use streaming::{StreamProcessor, StreamProcessorConfig};
// TODO: Uncomment when modules are implemented
// pub use coordination::{UpdateCoordinator, CoordinationConfig};
// pub use monitoring::{
//     PipelinePerformanceMonitor, AlertManager, MetricsCollector,
//     MonitoringConfig as PipelineMonitoringConfig,
// };
// pub use versioning::{VersionManager, VersionManagerConfig};
// pub use consistency::{ConsistencyManager, ConsistencyConfig};

// Re-export error types
pub use anyhow::{Context, Error, Result};

// Pipeline-specific error types
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Pipeline not initialized")]
    NotInitialized,

    #[error("Pipeline already running")]
    AlreadyRunning,

    #[error("Pipeline not running")]
    NotRunning,

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Stream processing error: {message}")]
    StreamProcessingError { message: String },

    #[error("Coordination error: {message}")]
    CoordinationError { message: String },

    #[error("Consistency error: {message}")]
    ConsistencyError { message: String },

    #[error("Monitoring error: {message}")]
    MonitoringError { message: String },

    #[error("Version management error: {message}")]
    VersionError { message: String },

    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    #[error("Timeout error: {operation}")]
    Timeout { operation: String },

    #[error("Backpressure limit exceeded")]
    BackpressureExceeded,

    #[error("Quality check failed: {reason}")]
    QualityCheckFailed { reason: String },

    #[error("Circuit breaker open for: {component}")]
    CircuitBreakerOpen { component: String },
}

/// Result type alias for pipeline operations
pub type PipelineResult<T> = std::result::Result<T, PipelineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.max_batch_size, 1000);
        assert_eq!(config.consistency_level, ConsistencyLevel::Session);
        assert_eq!(config.backpressure_strategy, BackpressureStrategy::Adaptive);
    }

    #[test]
    fn test_processing_priority_ordering() {
        use ProcessingPriority::*;
        assert!(Critical > High);
        assert!(High > Normal);
        assert!(Normal > Low);
    }

    #[test]
    fn test_update_priority_ordering() {
        use UpdatePriority::*;
        assert!(Urgent > High);
        assert!(High > Normal);
        assert!(Normal > Background);
    }

    #[test]
    fn test_alert_severity_ordering() {
        use AlertSeverity::*;
        assert!(Critical > Error);
        assert!(Error > Warning);
        assert!(Warning > Info);
    }

    #[test]
    fn test_inconsistency_severity_ordering() {
        use InconsistencySeverity::*;
        assert!(Critical > High);
        assert!(High > Medium);
        assert!(Medium > Low);
    }
}
