//! Core types and traits for advanced validation strategies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use oxirs_core::Store;
use oxirs_shacl::{Shape, ValidationReport};

use super::config::*;
use super::types::*;
use crate::Result;

/// Trait for validation strategies
pub trait ValidationStrategy: Send + Sync + std::fmt::Debug {
    /// Strategy name
    fn name(&self) -> &str;

    /// Strategy description
    fn description(&self) -> &str;

    /// Validate using this strategy
    fn validate(
        &self,
        store: &dyn Store,
        shapes: &[Shape],
        context: &ValidationContext,
    ) -> Result<StrategyValidationResult>;

    /// Get strategy capabilities
    fn capabilities(&self) -> StrategyCapabilities;

    /// Get strategy configuration parameters
    fn parameters(&self) -> HashMap<String, f64>;

    /// Update strategy parameters based on performance feedback
    fn update_parameters(&mut self, feedback: &PerformanceFeedback) -> Result<()>;

    /// Get strategy confidence for given context
    fn confidence_for_context(&self, context: &ValidationContext) -> f64;
}

/// Strategy validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyValidationResult {
    pub strategy_name: String,
    pub validation_report: ValidationReport,
    pub execution_time: Duration,
    pub memory_usage_mb: f64,
    pub confidence_score: f64,
    pub uncertainty_score: f64,
    pub quality_metrics: QualityMetrics,
    pub explanation: Option<ValidationExplanation>,
}

/// Strategy capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCapabilities {
    pub supports_temporal_validation: bool,
    pub supports_semantic_enrichment: bool,
    pub supports_parallel_processing: bool,
    pub supports_incremental_validation: bool,
    pub supports_uncertainty_quantification: bool,
    pub optimal_data_size_range: (usize, usize),
    pub optimal_shape_complexity_range: (f64, f64),
    pub computational_complexity: ComputationalComplexity,
}

/// Quality metrics for validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub accuracy: f64,
    pub specificity: f64,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
    pub matthews_correlation_coefficient: f64,
    pub area_under_roc_curve: f64,
}

/// Validation explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationExplanation {
    pub summary: String,
    pub detailed_explanation: String,
    pub constraint_contributions: HashMap<String, ContributionScore>,
    pub key_factors: Vec<KeyFactor>,
    pub confidence_factors: Vec<ConfidenceFactor>,
    pub recommendations: Vec<ValidationRecommendation>,
}

/// Advanced validation result
#[derive(Debug)]
pub struct AdvancedValidationResult {
    pub strategy_result: StrategyValidationResult,
    pub selected_strategy_name: String,
    pub context: ValidationContext,
    pub explanation: Option<ValidationExplanation>,
    pub uncertainty_metrics: Option<UncertaintyMetrics>,
    pub total_execution_time: Duration,
}

/// Uncertainty metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyMetrics {
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    pub total_uncertainty: f64,
    pub confidence_interval: ConfidenceInterval,
    pub uncertainty_sources: Vec<UncertaintySource>,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

/// Uncertainty source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintySource {
    pub source_type: UncertaintySourceType,
    pub contribution: f64,
    pub description: String,
}
