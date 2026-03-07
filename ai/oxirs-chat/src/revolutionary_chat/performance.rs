//! Performance measurement and correlation for revolutionary chat

use serde::{Deserialize, Serialize};

/// Correlation between chat components and performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCorrelation {
    pub component_name: String,
    pub metric_name: String,
    pub correlation_coefficient: f64,
    pub sample_count: usize,
    pub is_significant: bool,
}

impl PerformanceCorrelation {
    /// Create a new performance correlation record
    pub fn new(
        component_name: String,
        metric_name: String,
        correlation_coefficient: f64,
        sample_count: usize,
    ) -> Self {
        let is_significant = correlation_coefficient.abs() > 0.3 && sample_count >= 30;
        Self {
            component_name,
            metric_name,
            correlation_coefficient,
            sample_count,
            is_significant,
        }
    }
}

/// Performance optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub component: String,
    pub suggestion: String,
    pub expected_improvement: f64,
    pub confidence: f64,
}

/// Performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub response_time_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput_mps: f64,
    pub quality_score: f64,
    pub error_rate: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            response_time_ms: 0.0,
            memory_usage_mb: 0.0,
            throughput_mps: 0.0,
            quality_score: 0.0,
            error_rate: 0.0,
        }
    }
}
