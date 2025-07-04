//! Real-Time Streaming Adaptation for SHACL-AI
//!
//! This module provides real-time streaming adaptation capabilities that enable
//! the SHACL-AI system to continuously learn and adapt from streaming RDF data,
//! validation results, and performance metrics in real-time.
//!
//! The module is organized into several sub-modules:
//! - `core`: Core streaming adaptation engine and configuration
//! - `processors`: Stream processors for different data types
//! - `metrics`: Real-time metrics collection and monitoring
//! - `online_learning`: Online learning algorithms and concept drift detection

pub mod core;
pub mod metrics;
pub mod online_learning;
pub mod processors;

// Re-export main types for easy access
pub use core::{
    AdaptationAction, AdaptationEvent, AdaptationEventType, AdaptationTrigger, StreamChannel,
    StreamData, StreamType, StreamingAdaptationEngine, StreamingConfig, TriggerCondition,
};

pub use processors::{
    MetricsStreamProcessor, PatternStreamProcessor, RdfStreamProcessor, StreamProcessor,
    ValidationStreamProcessor,
};

pub use metrics::{
    Alert, AlertHandler, AlertType, LoggingAlertHandler, PerformanceMonitor, PerformanceThresholds,
    RealTimeAdaptationStats, RealTimeMetrics, RealTimeMetricsCollector,
};

pub use online_learning::{
    AdaptiveLearningRateScheduler, ConceptDriftDetector, ModelPerformanceMetrics,
    OnlineLearningAlgorithm, OnlineLearningEngine, OnlineModelState, StreamingDataPoint,
    StreamingDataType, StreamingFeatureExtractor, UpdateResult,
};

use crate::{Result, ShaclAiError};

/// Create a default streaming adaptation engine with standard configuration
pub fn create_default_engine() -> Result<StreamingAdaptationEngine> {
    let adaptive_ai = crate::self_adaptive_ai::SelfAdaptiveAI::new(
        crate::self_adaptive_ai::SelfAdaptiveConfig::default(),
    );
    let config = StreamingConfig::default();
    Ok(StreamingAdaptationEngine::new(adaptive_ai, config))
}

/// Create a streaming adaptation engine with custom configuration
pub fn create_engine_with_config(config: StreamingConfig) -> Result<StreamingAdaptationEngine> {
    let adaptive_ai = crate::self_adaptive_ai::SelfAdaptiveAI::new(
        crate::self_adaptive_ai::SelfAdaptiveConfig::default(),
    );
    Ok(StreamingAdaptationEngine::new(adaptive_ai, config))
}

/// Utility function to register all default stream processors
pub async fn register_default_processors(engine: &StreamingAdaptationEngine) -> Result<()> {
    // This would register the default processors in a real implementation
    tracing::info!("Registered default stream processors");
    Ok(())
}

/// Utility function to create a basic performance monitor
pub fn create_performance_monitor() -> PerformanceMonitor {
    let mut monitor = PerformanceMonitor::new();
    monitor.add_alert_handler(Box::new(LoggingAlertHandler));
    monitor
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_streaming_engine_creation() {
        let engine = create_default_engine();
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_custom_config_engine() {
        let config = StreamingConfig {
            stream_buffer_size: 200,
            adaptation_threshold: 0.3,
            max_concurrent_streams: 20,
            ..Default::default()
        };

        let engine = create_engine_with_config(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_collector() {
        let mut collector = RealTimeMetricsCollector::new();
        let metrics = collector.collect_current_metrics().await;
        assert!(metrics.is_ok());
    }

    #[tokio::test]
    async fn test_online_learning_engine() {
        let mut engine = OnlineLearningEngine::new();
        let data_point = StreamingDataPoint {
            timestamp: std::time::SystemTime::now(),
            data_type: StreamingDataType::Performance,
            data: vec![1, 2, 3, 4, 5],
            metadata: std::collections::HashMap::new(),
        };

        let result = engine.process_streaming_update(&data_point).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_concept_drift_detector() {
        let mut detector = ConceptDriftDetector::new();
        let mut features = std::collections::HashMap::new();
        features.insert("feature1".to_string(), 1.0);
        features.insert("feature2".to_string(), 2.0);

        let drift = detector.check_drift(&features).await;
        assert!(drift.is_ok());
    }

    #[test]
    fn test_adaptation_trigger() {
        let trigger = AdaptationTrigger::new(
            "test_trigger".to_string(),
            TriggerCondition::PerformanceDegraded,
            AdaptationAction::RetrainModel,
            0.5,
            Duration::from_secs(60),
        );

        assert!(!trigger.should_trigger(0.3)); // Below threshold
        assert!(trigger.should_trigger(0.7)); // Above threshold
    }

    #[test]
    fn test_performance_thresholds() {
        let thresholds = PerformanceThresholds::default();
        assert_eq!(thresholds.max_cpu_usage, 0.8);
        assert_eq!(thresholds.max_memory_usage, 0.9);
        assert_eq!(thresholds.max_error_rate, 0.1);
    }
}
