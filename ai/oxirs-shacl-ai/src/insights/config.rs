//! Configuration for insight generation

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for insight generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightGenerationConfig {
    /// Minimum confidence threshold for insights
    pub min_confidence_threshold: f64,

    /// Enable validation insights
    pub enable_validation_insights: bool,

    /// Enable quality insights
    pub enable_quality_insights: bool,

    /// Enable performance insights
    pub enable_performance_insights: bool,

    /// Enable shape insights
    pub enable_shape_insights: bool,

    /// Enable data insights
    pub enable_data_insights: bool,

    /// Maximum insights per category
    pub max_insights_per_category: usize,

    /// Time window for trend analysis (in seconds)
    pub trend_analysis_window: Duration,

    /// Enable advanced analytics
    pub enable_advanced_analytics: bool,

    /// Historical data retention period
    pub historical_retention_days: u32,
}

impl Default for InsightGenerationConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
            enable_validation_insights: true,
            enable_quality_insights: true,
            enable_performance_insights: true,
            enable_shape_insights: true,
            enable_data_insights: true,
            max_insights_per_category: 10,
            trend_analysis_window: Duration::from_secs(3600), // 1 hour
            enable_advanced_analytics: true,
            historical_retention_days: 30,
        }
    }
}
