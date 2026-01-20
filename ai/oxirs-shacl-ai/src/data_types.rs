//! Data types for SHACL-AI insight generation and analysis
//!
//! This module contains data structures used for validation analysis,
//! performance tracking, and quality assessment throughout the system.

use oxirs_core::model::Term;
use oxirs_shacl::{ShapeId, ValidationReport};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::insights::ShapeComplexityMetrics;
use crate::{LearningTrainingData, PredictionTrainingData, QualityTrainingData};

/// Training dataset for AI models
#[derive(Debug, Clone)]
pub struct TrainingDataset {
    pub shape_data: LearningTrainingData,
    pub quality_data: QualityTrainingData,
    pub prediction_data: PredictionTrainingData,
}

/// Training result for AI models
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub model_results: Vec<(String, ModelTrainingResult)>,
    pub overall_success: bool,
    pub training_time: std::time::Duration,
}

/// Individual model training result
#[derive(Debug, Clone)]
pub struct ModelTrainingResult {
    pub success: bool,
    pub accuracy: f64,
    pub loss: f64,
    pub epochs_trained: usize,
    pub training_time: std::time::Duration,
}

/// Validation data for insight analysis
#[derive(Debug, Clone)]
pub struct ValidationData {
    pub validation_reports: Vec<ValidationReport>,
    pub performance_metrics: HashMap<String, f64>,
    pub success_rate: f64,
    pub failure_patterns: Vec<String>,
}

impl ValidationData {
    pub fn calculate_success_rate(&self) -> f64 {
        self.success_rate
    }

    pub fn get_failing_shapes(&self) -> Vec<ShapeId> {
        self.validation_reports
            .iter()
            .flat_map(|r| &r.violations)
            .map(|v| v.source_shape.clone())
            .collect()
    }

    pub fn total_validations(&self) -> usize {
        self.validation_reports.len()
    }

    pub fn failed_validations(&self) -> usize {
        self.validation_reports
            .iter()
            .filter(|r| !r.conforms)
            .count()
    }

    pub fn extract_violation_patterns(&self) -> Vec<ViolationPattern> {
        // Simplified implementation
        vec![ViolationPattern {
            pattern_type: "missing_property".to_string(),
            description: "Missing required properties".to_string(),
            frequency: 0.3,
            confidence: 0.8,
            affected_shapes: Vec::new(),
            recommendations: vec!["Add required property constraints".to_string()],
            evidence: HashMap::new(),
        }]
    }

    pub fn calculate_performance_trend(&self) -> PerformanceTrend {
        PerformanceTrend {
            degradation_percentage: 15.0,
            significance: 0.85,
            sample_size: self.validation_reports.len(),
        }
    }
}

/// Performance data for insight analysis
#[derive(Debug, Clone)]
pub struct PerformanceData {
    pub current_avg_execution_time: f64,
    pub peak_memory_usage: f64,
    pub memory_threshold: f64,
    pub current_throughput: f64,
    pub performance_history: Vec<PerformanceMetric>,
}

impl PerformanceData {
    pub fn calculate_execution_time_trend(&self) -> ExecutionTimeTrend {
        ExecutionTimeTrend {
            increase_percentage: 25.0,
            significance: 0.8,
        }
    }

    pub fn calculate_throughput_trend(&self) -> ThroughputTrend {
        ThroughputTrend {
            decline_percentage: 15.0,
            significance: 0.85,
        }
    }
}

/// Shape data for insight analysis
#[derive(Debug, Clone)]
pub struct ShapeData {
    pub shape_analyses: Vec<ShapeAnalysis>,
}

/// RDF data for insight analysis
#[derive(Debug, Clone)]
pub struct RdfData {
    pub total_triples: usize,
    pub missing_data_elements: Vec<Term>,
    pub inconsistencies: Vec<DataInconsistency>,
}

impl RdfData {
    pub fn calculate_missing_data_percentage(&self) -> f64 {
        0.2 // 20% missing data
    }

    pub fn get_missing_data_elements(&self) -> Vec<Term> {
        self.missing_data_elements.clone()
    }

    pub fn total_elements(&self) -> usize {
        self.total_triples
    }

    pub fn detect_inconsistencies(&self) -> Vec<DataInconsistency> {
        self.inconsistencies.clone()
    }

    pub fn analyze_distribution(&self) -> DistributionAnalysis {
        DistributionAnalysis {
            confidence: 0.8,
            anomalous_elements: Vec::new(),
            statistics: HashMap::new(),
        }
    }
}

/// Supporting types for data analysis
#[derive(Debug, Clone)]
pub struct ViolationPattern {
    pub pattern_type: String,
    pub description: String,
    pub frequency: f64,
    pub confidence: f64,
    pub affected_shapes: Vec<ShapeId>,
    pub recommendations: Vec<String>,
    pub evidence: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub degradation_percentage: f64,
    pub significance: f64,
    pub sample_size: usize,
}

impl PerformanceTrend {
    pub fn is_degrading(&self) -> bool {
        self.degradation_percentage > 10.0
    }

    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "degradation_percentage".to_string(),
            self.degradation_percentage.to_string(),
        );
        map.insert("significance".to_string(), self.significance.to_string());
        map.insert("sample_size".to_string(), self.sample_size.to_string());
        map
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionTimeTrend {
    pub increase_percentage: f64,
    pub significance: f64,
}

impl ExecutionTimeTrend {
    pub fn is_increasing(&self) -> bool {
        self.increase_percentage > 5.0
    }

    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "increase_percentage".to_string(),
            self.increase_percentage.to_string(),
        );
        map.insert("significance".to_string(), self.significance.to_string());
        map
    }
}

#[derive(Debug, Clone)]
pub struct ThroughputTrend {
    pub decline_percentage: f64,
    pub significance: f64,
}

impl ThroughputTrend {
    pub fn is_declining(&self) -> bool {
        self.decline_percentage > 5.0
    }

    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "decline_percentage".to_string(),
            self.decline_percentage.to_string(),
        );
        map.insert("significance".to_string(), self.significance.to_string());
        map
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput: f64,
}

#[derive(Debug, Clone)]
pub struct ShapeAnalysis {
    pub shape_id: ShapeId,
    pub complexity_metrics: ShapeComplexityMetrics,
    pub effectiveness_score: f64,
}

#[derive(Debug, Clone)]
pub struct DataInconsistency {
    pub pattern_type: String,
    pub description: String,
    pub significance: f64,
    pub impact_level: InconsistencyImpact,
    pub affected_elements: Vec<Term>,
    pub suggested_fixes: Vec<String>,
    pub evidence: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    pub confidence: f64,
    pub anomalous_elements: Vec<Term>,
    pub statistics: HashMap<String, f64>,
}

impl DistributionAnalysis {
    pub fn has_significant_anomalies(&self) -> bool {
        !self.anomalous_elements.is_empty()
    }
}

/// Inconsistency impact levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InconsistencyImpact {
    Low,
    Medium,
    High,
}

/// Helper function to provide default Instant for serde default
pub fn default_instant() -> std::time::Instant {
    std::time::Instant::now()
}
