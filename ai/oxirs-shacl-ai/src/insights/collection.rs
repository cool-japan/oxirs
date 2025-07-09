//! Insight collection and management

use super::types::{
    DataInsight, PerformanceInsight, QualityInsight, ShapeInsight,
    ValidationInsight,
};
use crate::analytics::InsightSeverity;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Collection of all types of insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightCollection {
    /// Validation insights
    pub validation_insights: Vec<ValidationInsight>,

    /// Quality insights
    pub quality_insights: Vec<QualityInsight>,

    /// Performance insights
    pub performance_insights: Vec<PerformanceInsight>,

    /// Shape insights
    pub shape_insights: Vec<ShapeInsight>,

    /// Data insights
    pub data_insights: Vec<DataInsight>,

    /// Collection metadata
    pub metadata: InsightMetadata,
}

impl InsightCollection {
    /// Create a new empty insight collection
    pub fn new() -> Self {
        Self {
            validation_insights: Vec::new(),
            quality_insights: Vec::new(),
            performance_insights: Vec::new(),
            shape_insights: Vec::new(),
            data_insights: Vec::new(),
            metadata: InsightMetadata::default(),
        }
    }

    /// Get total number of insights
    pub fn total_count(&self) -> usize {
        self.validation_insights.len()
            + self.quality_insights.len()
            + self.performance_insights.len()
            + self.shape_insights.len()
            + self.data_insights.len()
    }

    /// Get high-priority insights count
    pub fn high_priority_count(&self) -> usize {
        let mut count = 0;

        count += self
            .validation_insights
            .iter()
            .filter(|i| i.is_high_priority())
            .count();
        count += self
            .quality_insights
            .iter()
            .filter(|i| {
                matches!(
                    i.severity,
                    InsightSeverity::Critical | InsightSeverity::High
                )
            })
            .count();
        count += self
            .performance_insights
            .iter()
            .filter(|i| {
                matches!(
                    i.severity,
                    InsightSeverity::Critical | InsightSeverity::High
                )
            })
            .count();
        count += self
            .shape_insights
            .iter()
            .filter(|i| {
                matches!(
                    i.severity,
                    InsightSeverity::Critical | InsightSeverity::High
                )
            })
            .count();
        count += self
            .data_insights
            .iter()
            .filter(|i| {
                matches!(
                    i.severity,
                    InsightSeverity::Critical | InsightSeverity::High
                )
            })
            .count();

        count
    }

    /// Add a validation insight
    pub fn add_validation_insight(&mut self, insight: ValidationInsight) {
        self.validation_insights.push(insight);
    }

    /// Add a quality insight
    pub fn add_quality_insight(&mut self, insight: QualityInsight) {
        self.quality_insights.push(insight);
    }

    /// Add a performance insight
    pub fn add_performance_insight(&mut self, insight: PerformanceInsight) {
        self.performance_insights.push(insight);
    }

    /// Add a shape insight
    pub fn add_shape_insight(&mut self, insight: ShapeInsight) {
        self.shape_insights.push(insight);
    }

    /// Add a data insight
    pub fn add_data_insight(&mut self, insight: DataInsight) {
        self.data_insights.push(insight);
    }

    /// Generate insight summary
    pub fn summary(&self) -> InsightSummary {
        InsightSummary {
            total_insights: self.total_count(),
            high_priority_insights: self.high_priority_count(),
            validation_insights_count: self.validation_insights.len(),
            quality_insights_count: self.quality_insights.len(),
            performance_insights_count: self.performance_insights.len(),
            shape_insights_count: self.shape_insights.len(),
            data_insights_count: self.data_insights.len(),
            generated_at: self.metadata.generated_at,
        }
    }
}

impl Default for InsightCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for insight collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightMetadata {
    /// When this collection was generated
    pub generated_at: DateTime<Utc>,

    /// Version of insight generation algorithm
    pub version: String,

    /// Configuration used for generation
    pub generation_config: HashMap<String, String>,

    /// Processing time for generation
    pub processing_time_ms: u64,
}

impl Default for InsightMetadata {
    fn default() -> Self {
        Self {
            generated_at: Utc::now(),
            version: "1.0.0".to_string(),
            generation_config: HashMap::new(),
            processing_time_ms: 0,
        }
    }
}

/// Summary of insights in a collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightSummary {
    /// Total number of insights
    pub total_insights: usize,

    /// Number of high-priority insights
    pub high_priority_insights: usize,

    /// Number of validation insights
    pub validation_insights_count: usize,

    /// Number of quality insights
    pub quality_insights_count: usize,

    /// Number of performance insights
    pub performance_insights_count: usize,

    /// Number of shape insights
    pub shape_insights_count: usize,

    /// Number of data insights
    pub data_insights_count: usize,

    /// When this summary was generated
    pub generated_at: DateTime<Utc>,
}
