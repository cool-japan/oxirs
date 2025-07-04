//! AI-powered insights for SHACL validation and data quality
//!
//! This module defines various types of insights that can be generated
//! from SHACL validation data, quality assessments, and performance metrics.

pub mod collection;
pub mod config;
pub mod generator;
pub mod types;

// Re-export main types for easy access
pub use collection::{InsightCollection, InsightMetadata, InsightSummary};
pub use config::InsightGenerationConfig;
pub use generator::{
    DataAnalysisData, InsightGenerator, PerformanceData, QualityData, ShapeData, ValidationData,
};
pub use types::{
    ComplexityLevel, DataInsight, DataInsightType, InsightCategory, InsightTrait,
    PerformanceInsight, PerformanceInsightType, PerformanceStatus, QualityInsight,
    QualityInsightType, QualityStatus, ShapeComplexityMetrics, ShapeInsight, ShapeInsightType,
    ValidationInsight, ValidationInsightType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::InsightSeverity;

    #[test]
    fn test_insight_collection_creation() {
        let collection = InsightCollection::new();
        assert_eq!(collection.total_count(), 0);
        assert_eq!(collection.high_priority_count(), 0);
    }

    #[test]
    fn test_insight_generator_creation() {
        let generator = InsightGenerator::new();
        assert!(generator.config().enable_validation_insights);
    }

    #[test]
    fn test_quality_status() {
        use crate::analytics::TrendDirection;

        let insight = QualityInsight {
            insight_type: QualityInsightType::Completeness,
            title: "Test Insight".to_string(),
            description: "Test Description".to_string(),
            severity: InsightSeverity::Medium,
            confidence: 0.8,
            quality_dimension: "completeness".to_string(),
            current_score: 0.9,
            trend_direction: TrendDirection::Increasing,
            recommendations: vec!["Test recommendation".to_string()],
            supporting_data: std::collections::HashMap::new(),
        };

        assert_eq!(insight.quality_status(), QualityStatus::Excellent);
        assert!(insight.is_improving());
        assert!(!insight.is_degrading());
    }

    #[test]
    fn test_performance_status() {
        use crate::analytics::TrendDirection;

        let insight = PerformanceInsight {
            insight_type: PerformanceInsightType::ThroughputIssue,
            title: "Test Performance Insight".to_string(),
            description: "Test Description".to_string(),
            severity: InsightSeverity::Medium,
            confidence: 0.7,
            metric_name: "throughput".to_string(),
            current_value: 100.0,
            trend_direction: TrendDirection::Increasing,
            recommendations: vec!["Optimize queries".to_string()],
            supporting_data: std::collections::HashMap::new(),
        };

        assert_eq!(insight.performance_status(), PerformanceStatus::Improving);
        assert!(insight.is_improving());
        assert!(!insight.is_degrading());
    }
}
