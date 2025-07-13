//! Advanced Analytics Module
//!
//! This module provides comprehensive analytics capabilities for SHACL validation,
//! including performance monitoring, validation pattern analysis, and real-time insights.

pub mod pattern_detection;
pub mod performance_monitor;
pub mod real_time_metrics;
pub mod validation_analytics;

pub use pattern_detection::*;
pub use performance_monitor::{
    PerformanceEvent, PerformanceMonitor, PerformanceSummary,
    TrendDirection as PerformanceTrendDirection,
};
pub use real_time_metrics::*;
pub use validation_analytics::{
    TrendDirection as ValidationTrendDirection, ValidationAnalytics, ValidationSummary,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Comprehensive analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable real-time monitoring
    pub enable_real_time: bool,
    /// Enable pattern detection
    pub enable_pattern_detection: bool,
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
    /// Metrics retention duration
    pub retention_duration: Duration,
    /// Sampling rate for metrics (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Enable advanced ML insights
    pub enable_ml_insights: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_real_time: true,
            enable_pattern_detection: true,
            enable_performance_profiling: true,
            retention_duration: Duration::from_secs(86400), // 24 hours
            sampling_rate: 1.0,
            enable_ml_insights: false,
        }
    }
}

/// Main analytics engine that coordinates all analytics components
#[derive(Debug)]
pub struct AnalyticsEngine {
    config: AnalyticsConfig,
    performance_monitor: PerformanceMonitor,
    validation_analytics: ValidationAnalytics,
    real_time_metrics: RealTimeMetrics,
    pattern_detector: PatternDetector,
}

impl AnalyticsEngine {
    /// Create a new analytics engine with default configuration
    pub fn new() -> Self {
        Self::with_config(AnalyticsConfig::default())
    }

    /// Create a new analytics engine with custom configuration
    pub fn with_config(config: AnalyticsConfig) -> Self {
        Self {
            config: config.clone(),
            performance_monitor: PerformanceMonitor::new(),
            validation_analytics: ValidationAnalytics::new(),
            real_time_metrics: RealTimeMetrics::new(),
            pattern_detector: PatternDetector::new(PatternDetectionConfig::default()),
        }
    }

    /// Record a validation event
    pub fn record_validation_event(&mut self, event: ValidationEvent) {
        if self.config.sampling_rate < 1.0 && rand::random::<f64>() > self.config.sampling_rate {
            return; // Skip this event based on sampling rate
        }

        if self.config.enable_performance_profiling {
            self.performance_monitor.record_event(&event);
        }

        if self.config.enable_real_time {
            self.real_time_metrics.update(&event);
        }

        if self.config.enable_pattern_detection {
            // Analyze events for pattern detection
            match event.event_type {
                ValidationEventType::ValidationStarted => {}
                ValidationEventType::ConstraintEvaluated => {
                    if let Some(dur) = event.duration {
                        let constraint_type = event
                            .constraint_id
                            .as_deref()
                            .unwrap_or("unknown_constraint");
                        self.pattern_detector
                            .record_performance_data(constraint_type, dur);
                    }
                }
                ValidationEventType::ViolationDetected => {
                    let shape_id = event.shape_id.as_deref().unwrap_or("unknown_shape");
                    let constraint_id = event
                        .constraint_id
                        .as_deref()
                        .unwrap_or("unknown_constraint");
                    self.pattern_detector.record_validation_failure(
                        shape_id,
                        constraint_id,
                        "validation_violation",
                    );
                }
                ValidationEventType::MemoryWarning => {
                    if let Some(memory_usage) = event.memory_usage {
                        self.pattern_detector.record_data_quality_issue(
                            "high_memory_usage",
                            &format!("memory_{memory_usage}"),
                        );
                    }
                }
                ValidationEventType::PerformanceWarning => {
                    let shape_id = event.shape_id.as_deref().unwrap_or("unknown_shape");
                    self.pattern_detector
                        .record_data_quality_issue("performance_warning", shape_id);
                }
                _ => {}
            }
        }

        self.validation_analytics.record_event(event);
    }

    /// Get comprehensive analytics report
    pub fn get_analytics_report(&self) -> AnalyticsReport {
        let detected_patterns = self
            .pattern_detector
            .get_detected_patterns()
            .into_iter()
            .cloned()
            .collect();

        AnalyticsReport {
            timestamp: Utc::now(),
            performance_summary: self.performance_monitor.get_summary(),
            validation_summary: self.validation_analytics.get_summary(),
            real_time_metrics: self.real_time_metrics.get_current_metrics(),
            detected_patterns,
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generate intelligent recommendations based on analytics
    fn generate_recommendations(&self) -> Vec<AnalyticsRecommendation> {
        let mut recommendations = Vec::new();

        // Performance-based recommendations
        let performance_summary = self.performance_monitor.get_summary();
        if performance_summary.average_validation_time > Duration::from_millis(1000) {
            recommendations.push(AnalyticsRecommendation {
                category: RecommendationCategory::Performance,
                severity: RecommendationSeverity::Medium,
                title: "Slow Validation Performance".to_string(),
                description: "Average validation time exceeds 1 second. Consider enabling parallel validation or optimizing constraint ordering.".to_string(),
                suggested_actions: vec![
                    "Enable parallel validation strategy".to_string(),
                    "Review constraint complexity".to_string(),
                    "Consider caching frequently validated shapes".to_string(),
                ],
            });
        }

        // Memory usage recommendations
        if let Some(memory_usage) = performance_summary.peak_memory_usage {
            if memory_usage > 100 * 1024 * 1024 {
                // 100MB
                recommendations.push(AnalyticsRecommendation {
                    category: RecommendationCategory::Memory,
                    severity: RecommendationSeverity::High,
                    title: "High Memory Usage".to_string(),
                    description: format!("Peak memory usage of {:.2} MB detected. Consider memory optimization strategies.", memory_usage as f64 / (1024.0 * 1024.0)),
                    suggested_actions: vec![
                        "Enable memory optimization".to_string(),
                        "Use streaming validation for large datasets".to_string(),
                        "Clear validation caches periodically".to_string(),
                    ],
                });
            }
        }

        // Pattern-based recommendations
        let patterns = self.pattern_detector.get_detected_patterns();
        if patterns
            .iter()
            .any(|p| p.pattern_type == PatternType::ValidationFailure)
        {
            recommendations.push(AnalyticsRecommendation {
                category: RecommendationCategory::DataQuality,
                severity: RecommendationSeverity::Medium,
                title: "Frequent Validation Violations".to_string(),
                description: "Detected patterns of frequent violations. Consider reviewing shape definitions or data quality processes.".to_string(),
                suggested_actions: vec![
                    "Review frequently violated constraints".to_string(),
                    "Implement data quality monitoring".to_string(),
                    "Consider shape relaxation for development environments".to_string(),
                ],
            });
        }

        if patterns
            .iter()
            .any(|p| p.pattern_type == PatternType::PerformanceBottleneck)
        {
            recommendations.push(AnalyticsRecommendation {
                category: RecommendationCategory::Performance,
                severity: RecommendationSeverity::High,
                title: "Performance Bottleneck Patterns Detected".to_string(),
                description: "Recurring performance bottlenecks identified. Optimization needed."
                    .to_string(),
                suggested_actions: vec![
                    "Optimize slow constraint evaluations".to_string(),
                    "Enable parallel processing".to_string(),
                    "Consider constraint ordering optimization".to_string(),
                ],
            });
        }

        if patterns
            .iter()
            .any(|p| p.pattern_type == PatternType::DataQuality)
        {
            recommendations.push(AnalyticsRecommendation {
                category: RecommendationCategory::DataQuality,
                severity: RecommendationSeverity::Medium,
                title: "Data Quality Issues Detected".to_string(),
                description: "Patterns of data quality issues found in the dataset.".to_string(),
                suggested_actions: vec![
                    "Implement data validation pipelines".to_string(),
                    "Review data source quality".to_string(),
                    "Add data cleansing processes".to_string(),
                ],
            });
        }

        recommendations
    }

    /// Clear old analytics data based on retention settings
    pub fn cleanup_old_data(&mut self) {
        let cutoff = Utc::now()
            - chrono::Duration::from_std(self.config.retention_duration).unwrap_or_default();

        self.performance_monitor.cleanup_before(cutoff);
        self.validation_analytics.cleanup_before(cutoff);
        self.real_time_metrics.cleanup_before(cutoff);
        self.pattern_detector.cleanup_old_patterns();
    }

    /// Get access to the pattern detector for advanced pattern analysis
    pub fn get_pattern_detector(&self) -> &PatternDetector {
        &self.pattern_detector
    }

    /// Get mutable access to the pattern detector
    pub fn get_pattern_detector_mut(&mut self) -> &mut PatternDetector {
        &mut self.pattern_detector
    }
}

impl Default for AnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a validation event for analytics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: ValidationEventType,
    pub duration: Option<Duration>,
    pub shape_id: Option<String>,
    pub constraint_id: Option<String>,
    pub target_count: Option<usize>,
    pub violation_count: Option<usize>,
    pub memory_usage: Option<usize>,
    pub additional_metadata: HashMap<String, String>,
}

/// Types of validation events
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationEventType {
    ValidationStarted,
    ValidationCompleted,
    ConstraintEvaluated,
    ViolationDetected,
    CacheHit,
    CacheMiss,
    PerformanceWarning,
    MemoryWarning,
}

/// Comprehensive analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReport {
    pub timestamp: DateTime<Utc>,
    pub performance_summary: PerformanceSummary,
    pub validation_summary: ValidationSummary,
    pub real_time_metrics: RealTimeMetricsSnapshot,
    pub detected_patterns: Vec<DetectedPattern>,
    pub recommendations: Vec<AnalyticsRecommendation>,
}

// DetectedPattern is now defined in pattern_detection module

/// Analytics recommendation for improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsRecommendation {
    pub category: RecommendationCategory,
    pub severity: RecommendationSeverity,
    pub title: String,
    pub description: String,
    pub suggested_actions: Vec<String>,
}

/// Categories of recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Memory,
    DataQuality,
    Security,
    Configuration,
}

/// Severity levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_engine_creation() {
        let engine = AnalyticsEngine::new();
        assert!(engine.config.enable_real_time);
        assert!(engine.config.enable_pattern_detection);
        assert!(engine.config.enable_performance_profiling);
    }

    #[test]
    fn test_analytics_config_default() {
        let config = AnalyticsConfig::default();
        assert_eq!(config.sampling_rate, 1.0);
        assert_eq!(config.retention_duration, Duration::from_secs(86400));
        assert!(!config.enable_ml_insights);
    }

    #[test]
    fn test_validation_event_creation() {
        let event = ValidationEvent {
            timestamp: Utc::now(),
            event_type: ValidationEventType::ValidationStarted,
            duration: Some(Duration::from_millis(100)),
            shape_id: Some("test_shape".to_string()),
            constraint_id: None,
            target_count: Some(10),
            violation_count: Some(2),
            memory_usage: Some(1024 * 1024),
            additional_metadata: HashMap::new(),
        };

        assert_eq!(event.event_type, ValidationEventType::ValidationStarted);
        assert_eq!(event.target_count, Some(10));
    }
}
