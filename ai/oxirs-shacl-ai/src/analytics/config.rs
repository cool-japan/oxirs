//! Configuration structures for analytics engine

use serde::{Deserialize, Serialize};

/// Configuration for analytics engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable comprehensive analytics
    pub enable_analytics: bool,

    /// Enable performance analytics
    pub enable_performance_analytics: bool,

    /// Enable quality analytics
    pub enable_quality_analytics: bool,

    /// Enable validation analytics
    pub enable_validation_analytics: bool,

    /// Enable trend analysis
    pub enable_trend_analysis: bool,

    /// Analytics collection settings
    pub collection_settings: AnalyticsCollectionSettings,

    /// Reporting settings
    pub reporting_settings: ReportingSettings,

    /// Enable training
    pub enable_training: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_analytics: true,
            enable_performance_analytics: true,
            enable_quality_analytics: true,
            enable_validation_analytics: true,
            enable_trend_analysis: true,
            collection_settings: AnalyticsCollectionSettings::default(),
            reporting_settings: ReportingSettings::default(),
            enable_training: true,
        }
    }
}

/// Analytics collection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsCollectionSettings {
    /// Collection interval in seconds
    pub collection_interval_seconds: u64,

    /// Maximum metrics retention period (days)
    pub retention_period_days: u32,

    /// Enable real-time metrics
    pub enable_realtime_metrics: bool,

    /// Enable batch processing
    pub enable_batch_processing: bool,

    /// Batch size for processing
    pub batch_size: usize,

    /// Sampling rate for metrics (0.0 - 1.0)
    pub sampling_rate: f64,
}

impl Default for AnalyticsCollectionSettings {
    fn default() -> Self {
        Self {
            collection_interval_seconds: 60,
            retention_period_days: 30,
            enable_realtime_metrics: true,
            enable_batch_processing: true,
            batch_size: 1000,
            sampling_rate: 1.0,
        }
    }
}

/// Reporting settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingSettings {
    /// Enable automated reporting
    pub enable_automated_reports: bool,

    /// Report generation interval (hours)
    pub report_interval_hours: u32,

    /// Report formats to generate
    pub report_formats: Vec<ReportFormat>,

    /// Include detailed metrics
    pub include_detailed_metrics: bool,

    /// Include visualizations
    pub include_visualizations: bool,
}

impl Default for ReportingSettings {
    fn default() -> Self {
        Self {
            enable_automated_reports: true,
            report_interval_hours: 24,
            report_formats: vec![ReportFormat::Json, ReportFormat::Html],
            include_detailed_metrics: true,
            include_visualizations: false,
        }
    }
}

/// Report formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Html,
    Pdf,
    Csv,
    Excel,
}
